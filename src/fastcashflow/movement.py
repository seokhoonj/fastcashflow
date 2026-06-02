"""Period-close roll-forward -- the IFRS 17 analysis of change.

A reporting period's movement bridges the opening insurance contract
liability to the closing one and decomposes the change into its drivers --
the analysis of change (AoC). This is the step from a measurement
calculator towards a reporting engine.

``roll_forward`` slices a GMM :class:`~fastcashflow.GMMMeasurement` into
reporting periods, reconciling each period's opening and closing BEL, RA
and CSM. It models all three drivers of the movement:

* the expected unwind -- interest accretion at the locked-in rate, and the
  expected release of cash flows and of the CSM;
* an assumption revision -- a change in the estimate of future cash flows;
* in-force experience -- the actual in-force at the period end differing
  from what was projected.

The latter two both relate to future service, so each adjusts the CSM
(floored at zero; any excess falls into the loss component) rather than
profit or loss.

``reconcile`` aggregates the per-model-point movements into portfolio-total
reconciliation tables, in the layout of IFRS 17 paragraph 101.

``roll_forward`` and ``reconcile`` also accept a PAA measurement -- the roll
of the liability for remaining coverage -- or a VFA measurement -- the roll
of its BEL, RA and CSM.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.engine import GMMMeasurement
from fastcashflow.numerics import _csm_kernel
from fastcashflow._paa import PAAMeasurement
from fastcashflow._vfa import VFAMeasurement


@dataclass(frozen=True, slots=True, eq=False)
class PeriodMovement:
    """One reporting period's analysis of change.

    The period covers months ``[month_start, month_end)``. Every array is
    ``(n_mp,)``, and each block reconciles exactly::

        bel_opening + bel_assumption_change + bel_experience
            + bel_interest - bel_release == bel_closing

    and likewise for RA and CSM (with ``csm_accretion`` in place of
    ``*_interest``).

    ``*_interest`` / ``csm_accretion`` is the unwind of discount at the
    locked-in rate; ``*_release`` is the expected run-off over the period.
    ``*_assumption_change`` and ``*_experience`` are the effect of an
    assumption revision and of in-force experience -- non-zero only in the
    period the change is recognised. Both relate to future service and so
    adjust the CSM. ``loss_component_recognised`` is the part of an
    unfavourable change beyond the CSM, which falls into the loss component.
    """

    month_start: int
    month_end: int
    bel_opening: FloatArray
    bel_assumption_change: FloatArray
    bel_experience: FloatArray
    bel_interest: FloatArray
    bel_release: FloatArray
    bel_closing: FloatArray
    ra_opening: FloatArray
    ra_assumption_change: FloatArray
    ra_experience: FloatArray
    ra_interest: FloatArray
    ra_release: FloatArray
    ra_closing: FloatArray
    csm_opening: FloatArray
    csm_assumption_change: FloatArray
    csm_experience: FloatArray
    csm_accretion: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray
    loss_component_recognised: FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class PAAPeriodMovement:
    """One reporting period's movement of the PAA insurance contract liability.

    The period covers months ``[month_start, month_end)``. Every array is
    ``(n_mp,)``; the three components each reconcile exactly::

        lrc_opening + premiums        - revenue     == lrc_closing
        lc_opening                    - lc_release  == lc_closing
        lic_opening + claims_incurred - claims_paid == lic_closing

    The LRC (liability for remaining coverage) is built up by premiums and
    released by insurance revenue; the loss component runs off over the
    coverage; the LIC (liability for incurred claims) is built up as claims
    are incurred and run off as they are paid. All are held undiscounted.
    """

    month_start: int
    month_end: int
    lrc_opening: FloatArray
    premiums: FloatArray
    revenue: FloatArray
    lrc_closing: FloatArray
    lc_opening: FloatArray
    lc_release: FloatArray
    lc_closing: FloatArray
    lic_opening: FloatArray
    claims_incurred: FloatArray
    claims_paid: FloatArray
    lic_closing: FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class VFAPeriodMovement:
    """One reporting period's movement of the VFA insurance contract liability.

    The period covers months ``[month_start, month_end)``. Every array is
    ``(n_mp,)`` and each block reconciles exactly::

        bel_opening + bel_interest  - bel_release  == bel_closing
        ra_opening  + ra_interest   - ra_release   == ra_closing
        csm_opening + csm_accretion - csm_release  == csm_closing

    ``*_interest`` / ``csm_accretion`` is the unwind at the underlying-items
    return; ``*_release`` is the expected run-off over the period. Under the
    VFA the CSM absorbs the variability of the underlying items, so the
    entity's profit emerges as the CSM is released.
    """

    month_start: int
    month_end: int
    bel_opening: FloatArray
    bel_interest: FloatArray
    bel_release: FloatArray
    bel_closing: FloatArray
    ra_opening: FloatArray
    ra_interest: FloatArray
    ra_release: FloatArray
    ra_closing: FloatArray
    csm_opening: FloatArray
    csm_accretion: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray


@singledispatch
def roll_forward(
    measurement,
    period_months: int = 12,
    *,
    revised=None,
    revised_at=None,
    actual_inforce=None,
    experience_at=None,
):
    """Slice a measurement into reporting-period movements.

    Returns one movement per reporting period of ``period_months`` months,
    reconciling the opening and closing figures; consecutive periods chain
    and a partial final period is allowed. Dispatches on the measurement
    type -- a new model registers with ``@roll_forward.register``.

    For a GMM measurement, an assumption revision is recognised by passing
    ``revised`` (a second measurement of the same book under updated basis)
    and ``revised_at`` (the month it takes effect); in-force experience by
    ``actual_inforce`` (the ``(n_mp,)`` in-force remaining at the period end,
    or a 2-D ``(n_periods, n_mp)`` array to roll experience through every
    period) and ``experience_at``. Either change adjusts the CSM by the
    resulting change in fulfilment cash flows (floored at zero, any excess
    falling into the loss component); v1 recognises one or the other, not
    both in a single call. A PAA or VFA measurement is also accepted -- the
    movement is then the roll of the LRC or of the CSM, to which the
    revision and experience options do not apply.
    """
    raise TypeError(
        f"roll_forward does not handle {type(measurement).__name__}"
    )


def _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at):
    if any(opt is not None for opt in
           (revised, revised_at, actual_inforce, experience_at)):
        raise ValueError(
            "the revision and experience options apply to a GMM "
            "measurement only"
        )


@roll_forward.register
def _(measurement: PAAMeasurement, period_months: int = 12, *,
      revised=None, revised_at=None, actual_inforce=None, experience_at=None):
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at)
    return _roll_forward_paa(measurement, period_months)


@roll_forward.register
def _(measurement: VFAMeasurement, period_months: int = 12, *,
      revised=None, revised_at=None, actual_inforce=None, experience_at=None):
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at)
    return _roll_forward_vfa(measurement, period_months)


@roll_forward.register
def _(
    measurement: GMMMeasurement,
    period_months: int = 12,
    *,
    revised: GMMMeasurement | None = None,
    revised_at: int | None = None,
    actual_inforce: FloatArray | None = None,
    experience_at: int | None = None,
) -> list[PeriodMovement]:
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    n_time = measurement.bel_path.shape[1] - 1
    n_mp = measurement.bel_path.shape[0]
    if actual_inforce is not None:
        actual_inforce = np.asarray(actual_inforce, dtype=np.float64)
        if actual_inforce.ndim == 2:
            if experience_at is not None or revised is not None:
                raise ValueError(
                    "a 2-D actual_inforce rolls experience through every "
                    "reporting period; experience_at and revised do not apply"
                )
            if actual_inforce.shape[1] != n_mp:
                raise ValueError(
                    f"actual_inforce must have {n_mp} columns -- one per "
                    "model point"
                )
            return _roll_forward_experience_chain(
                measurement, period_months, actual_inforce
            )
    if (revised is None) != (revised_at is None):
        raise ValueError("pass revised and revised_at together, or neither")
    if (actual_inforce is None) != (experience_at is None):
        raise ValueError("pass actual_inforce and experience_at together, or neither")
    if revised is not None and actual_inforce is not None:
        raise ValueError(
            "v1 recognises an assumption revision or in-force experience, "
            "not both in a single call"
        )

    discount_start = measurement.discount_start
    monthly_rate = discount_start[:-1] / discount_start[1:] - 1.0   # (n_time,)
    zero = np.zeros(n_mp)

    bel, ra, csm = measurement.bel_path, measurement.ra_path, measurement.csm_path
    csm_accretion = measurement.csm_accretion
    csm_release = measurement.csm_release
    change_at: int | None = None
    change_kind = ""
    post_bel = post_ra = csm_after = None
    loss = zero

    if revised is not None:
        if revised.bel_path.shape != measurement.bel_path.shape:
            raise ValueError("revised must measure the same book as measurement")
        change_at, change_kind = revised_at, "assumption"
        post_bel, post_ra = revised.bel_path, revised.ra_path
        post_inforce = revised.cashflows.inforce
    elif actual_inforce is not None:
        actual_inforce = np.asarray(actual_inforce, dtype=np.float64)
        if actual_inforce.shape != (n_mp,):
            raise ValueError(f"actual_inforce must have shape ({n_mp},)")
        change_at, change_kind = experience_at, "experience"
        expected = measurement.cashflows.inforce[:, experience_at]
        safe = np.where(expected > 1e-12, expected, 1.0)
        # In-force experience scales the remaining contract: the future
        # projection uses the same basis, so the closing FCF scales
        # linearly with the in-force actually remaining.
        ratio = np.where(expected > 1e-12, actual_inforce / safe, 1.0)
        post_bel = measurement.bel_path * ratio[:, None]
        post_ra = measurement.ra_path * ratio[:, None]
        post_inforce = measurement.cashflows.inforce

    if change_at is not None:
        k = change_at
        if k % period_months != 0 or not 0 < k < n_time:
            raise ValueError(
                "the change month must be a positive multiple of "
                f"period_months below the horizon ({n_time}), got {k}"
            )
        delta_fcf = ((post_bel[:, k] + post_ra[:, k])
                     - (measurement.bel_path[:, k] + measurement.ra_path[:, k]))
        csm_before = measurement.csm_path[:, k]
        csm_after = np.maximum(0.0, csm_before - delta_fcf)
        loss = np.maximum(0.0, delta_fcf - csm_before)
        re_csm, re_acc, re_rel = _csm_kernel(
            csm_after, np.ascontiguousarray(post_inforce[:, k:]),
            monthly_rate[k:],
        )
        bel = np.concatenate([measurement.bel_path[:, :k + 1], post_bel[:, k + 1:]],
                             axis=1)
        ra = np.concatenate([measurement.ra_path[:, :k + 1], post_ra[:, k + 1:]],
                            axis=1)
        csm = np.concatenate([measurement.csm_path[:, :k + 1], re_csm[:, 1:]], axis=1)
        csm_accretion = np.concatenate(
            [measurement.csm_accretion[:, :k], re_acc], axis=1)
        csm_release = np.concatenate(
            [measurement.csm_release[:, :k], re_rel], axis=1)

    movements: list[PeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_open, ra_open, csm_open = bel[:, a], ra[:, a], csm[:, a]
        bel_ac = bel_ex = ra_ac = ra_ex = csm_ac = csm_ex = loss_line = zero
        bel_traj, ra_traj = bel, ra
        if change_at is not None and a == change_at:
            d_bel = post_bel[:, a] - bel_open
            d_ra = post_ra[:, a] - ra_open
            d_csm = csm_after - csm_open
            if change_kind == "assumption":
                bel_ac, ra_ac, csm_ac = d_bel, d_ra, d_csm
            else:
                bel_ex, ra_ex, csm_ex = d_bel, d_ra, d_csm
            loss_line = loss
            bel_traj, ra_traj = post_bel, post_ra
        bel_interest = (bel_traj[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        ra_interest = (ra_traj[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        movements.append(PeriodMovement(
            month_start=a,
            month_end=b,
            bel_opening=bel_open,
            bel_assumption_change=bel_ac,
            bel_experience=bel_ex,
            bel_interest=bel_interest,
            bel_release=bel_open + bel_ac + bel_ex + bel_interest - bel[:, b],
            bel_closing=bel[:, b],
            ra_opening=ra_open,
            ra_assumption_change=ra_ac,
            ra_experience=ra_ex,
            ra_interest=ra_interest,
            ra_release=ra_open + ra_ac + ra_ex + ra_interest - ra[:, b],
            ra_closing=ra[:, b],
            csm_opening=csm_open,
            csm_assumption_change=csm_ac,
            csm_experience=csm_ex,
            csm_accretion=csm_accretion[:, a:b].sum(axis=1),
            csm_release=csm_release[:, a:b].sum(axis=1),
            csm_closing=csm[:, b],
            loss_component_recognised=loss_line,
        ))
    return movements


def _roll_forward_experience_chain(
    measurement: GMMMeasurement, period_months: int, actual_inforce: FloatArray
) -> list[PeriodMovement]:
    """Roll a GMM measurement through in-force experience at every period.

    Row ``j`` of ``actual_inforce`` is the in-force actually remaining at
    month ``(j+1) * period_months``. The cumulative ratio at each boundary
    is the actual over the originally expected in-force; the CSM is rolled
    segment by segment, each segment releasing over the in-force expected at
    its start, with the experience jump applied at each boundary.
    """
    base_bel = measurement.bel_path
    base_ra = measurement.ra_path
    base_inforce = measurement.cashflows.inforce
    n_mp, n_time = base_inforce.shape
    n_known = actual_inforce.shape[0]
    boundaries = [(j + 1) * period_months for j in range(n_known)]
    if boundaries[-1] >= n_time:
        raise ValueError(
            f"actual_inforce has {n_known} rows; the last boundary "
            f"({boundaries[-1]}) reaches the projection horizon ({n_time})"
        )
    discount_start = measurement.discount_start
    monthly_rate = discount_start[:-1] / discount_start[1:] - 1.0

    # Cumulative in-force ratio at each boundary, laid out as a per-month
    # step factor -- 1 up to the first boundary, then each ratio onward.
    step = np.ones((n_mp, n_time + 1))
    cumratios: list[FloatArray] = []
    for j, b in enumerate(boundaries):
        expected = base_inforce[:, b]
        safe = np.where(expected > 1e-12, expected, 1.0)
        cr = np.where(expected > 1e-12, actual_inforce[j] / safe, 1.0)
        cumratios.append(cr)
        step[:, b + 1:] = cr[:, None]
    bel = base_bel * step
    ra = base_ra * step

    # CSM -- rolled segment by segment, with the experience jump at each
    # boundary. Each segment releases over the in-force expected at its
    # start, so later boundaries do not disturb the earlier releases.
    csm = np.empty((n_mp, n_time + 1))
    csm_accretion = np.empty((n_mp, n_time))
    csm_release = np.empty((n_mp, n_time))
    csm[:, 0] = measurement.csm_path[:, 0]
    cur = measurement.csm_path[:, 0]
    exp_lines: dict[int, tuple] = {}
    s = 0
    for j, e in enumerate(boundaries + [n_time]):
        seg_csm, seg_acc, seg_rel = _csm_kernel(
            np.ascontiguousarray(cur),
            np.ascontiguousarray(base_inforce[:, s:]),
            monthly_rate[s:],
        )
        width = e - s
        csm[:, s + 1:e + 1] = seg_csm[:, 1:width + 1]
        csm_accretion[:, s:e] = seg_acc[:, :width]
        csm_release[:, s:e] = seg_rel[:, :width]
        if e < n_time:
            cr_prev = cumratios[j - 1] if j > 0 else np.ones(n_mp)
            bel_ex = base_bel[:, e] * (cumratios[j] - cr_prev)
            ra_ex = base_ra[:, e] * (cumratios[j] - cr_prev)
            delta_fcf = bel_ex + ra_ex
            csm_before = csm[:, e]
            csm_after = np.maximum(0.0, csm_before - delta_fcf)
            exp_lines[e] = (
                bel_ex, ra_ex, csm_after - csm_before,
                np.maximum(0.0, delta_fcf - csm_before),
            )
            cur = csm_after
        s = e

    zero = np.zeros(n_mp)
    movements: list[PeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_ex, ra_ex, csm_ex, loss = exp_lines.get(a, (zero, zero, zero, zero))
        bel_interest = ((bel[:, a:b] * monthly_rate[a:b]).sum(axis=1)
                        + bel_ex * monthly_rate[a])
        ra_interest = ((ra[:, a:b] * monthly_rate[a:b]).sum(axis=1)
                       + ra_ex * monthly_rate[a])
        movements.append(PeriodMovement(
            month_start=a,
            month_end=b,
            bel_opening=bel[:, a],
            bel_assumption_change=zero,
            bel_experience=bel_ex,
            bel_interest=bel_interest,
            bel_release=bel[:, a] + bel_ex + bel_interest - bel[:, b],
            bel_closing=bel[:, b],
            ra_opening=ra[:, a],
            ra_assumption_change=zero,
            ra_experience=ra_ex,
            ra_interest=ra_interest,
            ra_release=ra[:, a] + ra_ex + ra_interest - ra[:, b],
            ra_closing=ra[:, b],
            csm_opening=csm[:, a],
            csm_assumption_change=zero,
            csm_experience=csm_ex,
            csm_accretion=csm_accretion[:, a:b].sum(axis=1),
            csm_release=csm_release[:, a:b].sum(axis=1),
            csm_closing=csm[:, b],
            loss_component_recognised=loss,
        ))
    return movements


def _roll_forward_paa(
    measurement: PAAMeasurement, period_months: int
) -> list[PAAPeriodMovement]:
    """Slice a PAA measurement into LRC, loss-component and LIC movements."""
    lrc = measurement.lrc_path
    lic = measurement.lic
    premium_cf = measurement.cashflows.premium_cf
    revenue = measurement.revenue
    incurred = measurement.cashflows.claim_cf + measurement.cashflows.morbidity_cf
    loss_component = measurement.loss_component
    n_time = lrc.shape[1] - 1
    total_revenue = revenue.sum(axis=1)
    safe_revenue = np.where(total_revenue > 0.0, total_revenue, 1.0)
    movements: list[PAAPeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        period_incurred = incurred[:, a:b].sum(axis=1)
        # the loss component runs off in proportion to insurance revenue
        lc_open = loss_component * revenue[:, a:].sum(axis=1) / safe_revenue
        lc_close = loss_component * revenue[:, b:].sum(axis=1) / safe_revenue
        movements.append(PAAPeriodMovement(
            month_start=a,
            month_end=b,
            lrc_opening=lrc[:, a],
            premiums=premium_cf[:, a:b].sum(axis=1),
            revenue=revenue[:, a:b].sum(axis=1),
            lrc_closing=lrc[:, b],
            lc_opening=lc_open,
            lc_release=lc_open - lc_close,
            lc_closing=lc_close,
            lic_opening=lic[:, a],
            claims_incurred=period_incurred,
            claims_paid=period_incurred - (lic[:, b] - lic[:, a]),
            lic_closing=lic[:, b],
        ))
    return movements


def _roll_forward_vfa(
    measurement: VFAMeasurement, period_months: int
) -> list[VFAPeriodMovement]:
    """Slice a VFA measurement into BEL, RA and CSM movements."""
    bel, ra, csm = measurement.bel_path, measurement.ra_path, measurement.csm_path
    csm_accretion = measurement.csm_accretion
    csm_release = measurement.csm_release
    n_time = csm.shape[1] - 1
    discount_start = measurement.discount_start
    monthly_rate = discount_start[:-1] / discount_start[1:] - 1.0
    movements: list[VFAPeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_interest = (bel[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        ra_interest = (ra[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        movements.append(VFAPeriodMovement(
            month_start=a,
            month_end=b,
            bel_opening=bel[:, a],
            bel_interest=bel_interest,
            bel_release=bel[:, a] + bel_interest - bel[:, b],
            bel_closing=bel[:, b],
            ra_opening=ra[:, a],
            ra_interest=ra_interest,
            ra_release=ra[:, a] + ra_interest - ra[:, b],
            ra_closing=ra[:, b],
            csm_opening=csm[:, a],
            csm_accretion=csm_accretion[:, a:b].sum(axis=1),
            csm_release=csm_release[:, a:b].sum(axis=1),
            csm_closing=csm[:, b],
        ))
    return movements


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """An IFRS 17 reconciliation of the insurance contract liability.

    Portfolio totals for one reporting period, in the layout of IFRS 17
    paragraph 101: the estimates of the present value of future cash flows
    (``bel``), the risk adjustment (``ra``) and the CSM each reconcile from
    opening to closing. ``*_future_service`` is the assumption and
    experience effect; ``*_finance`` is the interest unwind; ``*_release``
    is the run-off, shown negative -- so opening plus every row equals
    closing.
    """

    month_start: int
    month_end: int
    bel_opening: float
    bel_future_service: float
    bel_finance: float
    bel_release: float
    bel_closing: float
    ra_opening: float
    ra_future_service: float
    ra_finance: float
    ra_release: float
    ra_closing: float
    csm_opening: float
    csm_future_service: float
    csm_finance: float
    csm_release: float
    csm_closing: float
    loss_component_recognised: float

    def __str__(self) -> str:
        rows = (
            ("Opening", self.bel_opening, self.ra_opening, self.csm_opening),
            ("Future service", self.bel_future_service,
             self.ra_future_service, self.csm_future_service),
            ("Finance", self.bel_finance, self.ra_finance, self.csm_finance),
            ("Release", self.bel_release, self.ra_release, self.csm_release),
            ("Closing", self.bel_closing, self.ra_closing, self.csm_closing),
        )
        lines = [
            f"Reconciliation -- months {self.month_start + 1}-{self.month_end}",
            f"{'':16}{'BEL':>18}{'RA':>18}{'CSM':>18}",
        ]
        for name, bel, ra, csm in rows:
            lines.append(f"{name:16}{bel:>18,.0f}{ra:>18,.0f}{csm:>18,.0f}")
        if self.loss_component_recognised:
            lines.append(
                f"{'Loss component':16}"
                f"{self.loss_component_recognised:>18,.0f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PAAReconciliation:
    """An IFRS 17 paragraph-100 reconciliation of the PAA liability.

    Portfolio totals for one reporting period, split into the three
    components -- the liability for remaining coverage (excluding the loss
    component), the loss component, and the liability for incurred claims.
    Run-off rows are shown negative, so opening plus every row equals
    closing.
    """

    month_start: int
    month_end: int
    lrc_opening: float
    premiums: float
    revenue: float
    lrc_closing: float
    lc_opening: float
    lc_release: float
    lc_closing: float
    lic_opening: float
    claims_incurred: float
    claims_paid: float
    lic_closing: float

    def __str__(self) -> str:
        blocks = (
            ("LRC (excluding loss component)", (
                ("Opening", self.lrc_opening),
                ("Premiums received", self.premiums),
                ("Insurance revenue", self.revenue),
                ("Closing", self.lrc_closing),
            )),
            ("Loss component", (
                ("Opening", self.lc_opening),
                ("Released", self.lc_release),
                ("Closing", self.lc_closing),
            )),
            ("Liability for incurred claims", (
                ("Opening", self.lic_opening),
                ("Claims incurred", self.claims_incurred),
                ("Claims paid", self.claims_paid),
                ("Closing", self.lic_closing),
            )),
        )
        lines = [
            f"PAA reconciliation -- months {self.month_start + 1}-{self.month_end}"
        ]
        for title, rows in blocks:
            lines.append(f"  {title}")
            for name, value in rows:
                lines.append(f"    {name:22}{value:>18,.0f}")
        return "\n".join(lines)


def _reconcile_paa(
    movements: list[PAAPeriodMovement],
) -> list[PAAReconciliation]:
    """Aggregate PAA period movements into portfolio-total reconciliations."""
    return [
        PAAReconciliation(
            month_start=m.month_start,
            month_end=m.month_end,
            lrc_opening=float(m.lrc_opening.sum()),
            premiums=float(m.premiums.sum()),
            revenue=float(-m.revenue.sum()),
            lrc_closing=float(m.lrc_closing.sum()),
            lc_opening=float(m.lc_opening.sum()),
            lc_release=float(-m.lc_release.sum()),
            lc_closing=float(m.lc_closing.sum()),
            lic_opening=float(m.lic_opening.sum()),
            claims_incurred=float(m.claims_incurred.sum()),
            claims_paid=float(-m.claims_paid.sum()),
            lic_closing=float(m.lic_closing.sum()),
        )
        for m in movements
    ]


@dataclass(frozen=True, slots=True)
class VFAReconciliation:
    """An IFRS 17 VFA reconciliation of the insurance contract liability.

    Portfolio totals for one reporting period -- the BEL, RA and CSM each
    reconciled from opening to closing. ``*_finance`` is the unwind at the
    underlying-items return; ``*_release`` is the run-off, shown negative --
    so opening plus every row equals closing.
    """

    month_start: int
    month_end: int
    bel_opening: float
    bel_finance: float
    bel_release: float
    bel_closing: float
    ra_opening: float
    ra_finance: float
    ra_release: float
    ra_closing: float
    csm_opening: float
    csm_finance: float
    csm_release: float
    csm_closing: float

    def __str__(self) -> str:
        rows = (
            ("Opening", self.bel_opening, self.ra_opening, self.csm_opening),
            ("Finance", self.bel_finance, self.ra_finance, self.csm_finance),
            ("Release", self.bel_release, self.ra_release, self.csm_release),
            ("Closing", self.bel_closing, self.ra_closing, self.csm_closing),
        )
        lines = [
            f"VFA reconciliation -- months {self.month_start + 1}-{self.month_end}",
            f"{'':16}{'BEL':>18}{'RA':>18}{'CSM':>18}",
        ]
        for name, bel, ra, csm in rows:
            lines.append(f"{name:16}{bel:>18,.0f}{ra:>18,.0f}{csm:>18,.0f}")
        return "\n".join(lines)


def _reconcile_vfa(
    movements: list[VFAPeriodMovement],
) -> list[VFAReconciliation]:
    """Aggregate VFA period movements into portfolio-total reconciliations."""
    return [
        VFAReconciliation(
            month_start=m.month_start,
            month_end=m.month_end,
            bel_opening=float(m.bel_opening.sum()),
            bel_finance=float(m.bel_interest.sum()),
            bel_release=float(-m.bel_release.sum()),
            bel_closing=float(m.bel_closing.sum()),
            ra_opening=float(m.ra_opening.sum()),
            ra_finance=float(m.ra_interest.sum()),
            ra_release=float(-m.ra_release.sum()),
            ra_closing=float(m.ra_closing.sum()),
            csm_opening=float(m.csm_opening.sum()),
            csm_finance=float(m.csm_accretion.sum()),
            csm_release=float(-m.csm_release.sum()),
            csm_closing=float(m.csm_closing.sum()),
        )
        for m in movements
    ]


def reconcile(
    movements: (list[PeriodMovement] | list[PAAPeriodMovement]
                | list[VFAPeriodMovement]),
) -> list[Reconciliation] | list[PAAReconciliation] | list[VFAReconciliation]:
    """Aggregate period movements into IFRS 17 reconciliation tables.

    Each :class:`PeriodMovement` -- per model point -- becomes one
    portfolio-total :class:`Reconciliation` in the layout of IFRS 17
    paragraph 101. Run-off rows are shown negative, so opening plus every
    row equals closing. A list of :class:`PAAPeriodMovement` or
    :class:`VFAPeriodMovement` is reconciled instead into the PAA
    liability-for-remaining-coverage or VFA contractual-service-margin
    tables.
    """
    if movements and isinstance(movements[0], PAAPeriodMovement):
        return _reconcile_paa(movements)
    if movements and isinstance(movements[0], VFAPeriodMovement):
        return _reconcile_vfa(movements)
    out: list[Reconciliation] = []
    for m in movements:
        out.append(Reconciliation(
            month_start=m.month_start,
            month_end=m.month_end,
            bel_opening=float(m.bel_opening.sum()),
            bel_future_service=float(
                (m.bel_assumption_change + m.bel_experience).sum()),
            bel_finance=float(m.bel_interest.sum()),
            bel_release=float(-m.bel_release.sum()),
            bel_closing=float(m.bel_closing.sum()),
            ra_opening=float(m.ra_opening.sum()),
            ra_future_service=float(
                (m.ra_assumption_change + m.ra_experience).sum()),
            ra_finance=float(m.ra_interest.sum()),
            ra_release=float(-m.ra_release.sum()),
            ra_closing=float(m.ra_closing.sum()),
            csm_opening=float(m.csm_opening.sum()),
            csm_future_service=float(
                (m.csm_assumption_change + m.csm_experience).sum()),
            csm_finance=float(m.csm_accretion.sum()),
            csm_release=float(-m.csm_release.sum()),
            csm_closing=float(m.csm_closing.sum()),
            loss_component_recognised=float(m.loss_component_recognised.sum()),
        ))
    return out
