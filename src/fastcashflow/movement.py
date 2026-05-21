"""Period-close roll-forward -- the IFRS 17 analysis of change.

A reporting period's movement bridges the opening insurance contract
liability to the closing one and decomposes the change into its drivers --
the analysis of change (AoC). This is the step from a measurement
calculator towards a reporting engine.

``roll_forward`` slices a GMM :class:`~fastcashflow.Measurement` into
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

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.engine import Measurement
from fastcashflow.gmm import _csm_kernel
from fastcashflow.paa import PAAMeasurement
from fastcashflow.vfa import VFAMeasurement


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class PAAPeriodMovement:
    """One reporting period's movement of the PAA liability for remaining
    coverage (LRC).

    The period covers months ``[month_start, month_end)``. Every array is
    ``(n_mp,)`` and the block reconciles exactly::

        lrc_opening + premiums - revenue == lrc_closing

    Premiums received build the LRC up; insurance revenue earned releases
    it. The LRC is held undiscounted, so there is no interest row.
    """

    month_start: int
    month_end: int
    lrc_opening: FloatArray
    premiums: FloatArray
    revenue: FloatArray
    lrc_closing: FloatArray


@dataclass(frozen=True, slots=True)
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


def roll_forward(
    measurement: Measurement | PAAMeasurement | VFAMeasurement,
    period_months: int = 12,
    *,
    revised: Measurement | None = None,
    revised_at: int | None = None,
    actual_inforce: FloatArray | None = None,
    experience_at: int | None = None,
) -> list[PeriodMovement] | list[PAAPeriodMovement] | list[VFAPeriodMovement]:
    """Slice a GMM measurement into reporting-period movements.

    Returns one :class:`PeriodMovement` per reporting period of
    ``period_months`` months, reconciling the opening and closing BEL, RA
    and CSM. Consecutive periods chain, and a horizon that is not a whole
    number of periods gives a shorter final period.

    An assumption revision is recognised by passing ``revised`` -- a second
    measurement of the same book under updated assumptions -- and
    ``revised_at``, the month it takes effect. In-force experience is
    recognised by passing ``actual_inforce`` -- the ``(n_mp,)`` in-force
    actually remaining at the period end -- and ``experience_at``, that
    month. Passing ``actual_inforce`` as a 2-D ``(n_periods, n_mp)`` array
    instead rolls experience through every reporting period, row ``j`` being
    the in-force at month ``(j+1) * period_months``.
    The revision month and the single experience month must be positive
    multiples of ``period_months``. Either change adjusts the CSM by the
    resulting change in fulfilment cash flows (floored at zero, any excess
    falling into the loss component); v1 recognises one or the other, not
    both in a single call.

    A PAA or VFA measurement is also accepted -- the movement is then the
    roll of the liability for remaining coverage or of the contractual
    service margin, to which the revision and experience options do not
    apply.
    """
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    if isinstance(measurement, (PAAMeasurement, VFAMeasurement)):
        if any(opt is not None for opt in
               (revised, revised_at, actual_inforce, experience_at)):
            raise ValueError(
                "the revision and experience options apply to a GMM "
                "measurement only"
            )
        if isinstance(measurement, PAAMeasurement):
            return _roll_forward_paa(measurement, period_months)
        return _roll_forward_vfa(measurement, period_months)
    n_time = measurement.bel.shape[1] - 1
    n_mp = measurement.bel.shape[0]
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

    bel, ra, csm = measurement.bel, measurement.ra, measurement.csm
    csm_accretion = measurement.csm_accretion
    csm_release = measurement.csm_release
    change_at: int | None = None
    change_kind = ""
    post_bel = post_ra = csm_after = None
    loss = zero

    if revised is not None:
        if revised.bel.shape != measurement.bel.shape:
            raise ValueError("revised must measure the same book as measurement")
        change_at, change_kind = revised_at, "assumption"
        post_bel, post_ra = revised.bel, revised.ra
        post_inforce = revised.cashflows.inforce
    elif actual_inforce is not None:
        actual_inforce = np.asarray(actual_inforce, dtype=np.float64)
        if actual_inforce.shape != (n_mp,):
            raise ValueError(f"actual_inforce must have shape ({n_mp},)")
        change_at, change_kind = experience_at, "experience"
        expected = measurement.cashflows.inforce[:, experience_at]
        safe = np.where(expected > 1e-12, expected, 1.0)
        # In-force experience scales the remaining contract: the future
        # projection uses the same assumptions, so the closing FCF scales
        # linearly with the in-force actually remaining.
        ratio = np.where(expected > 1e-12, actual_inforce / safe, 1.0)
        post_bel = measurement.bel * ratio[:, None]
        post_ra = measurement.ra * ratio[:, None]
        post_inforce = measurement.cashflows.inforce

    if change_at is not None:
        k = change_at
        if k % period_months != 0 or not 0 < k < n_time:
            raise ValueError(
                "the change month must be a positive multiple of "
                f"period_months below the horizon ({n_time}), got {k}"
            )
        rate = 1.0 / discount_start[1] - 1.0
        delta_fcf = ((post_bel[:, k] + post_ra[:, k])
                     - (measurement.bel[:, k] + measurement.ra[:, k]))
        csm_before = measurement.csm[:, k]
        csm_after = np.maximum(0.0, csm_before - delta_fcf)
        loss = np.maximum(0.0, delta_fcf - csm_before)
        re_csm, re_acc, re_rel = _csm_kernel(
            csm_after, np.ascontiguousarray(post_inforce[:, k:]), rate
        )
        bel = np.concatenate([measurement.bel[:, :k + 1], post_bel[:, k + 1:]],
                             axis=1)
        ra = np.concatenate([measurement.ra[:, :k + 1], post_ra[:, k + 1:]],
                            axis=1)
        csm = np.concatenate([measurement.csm[:, :k + 1], re_csm[:, 1:]], axis=1)
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
    measurement: Measurement, period_months: int, actual_inforce: FloatArray
) -> list[PeriodMovement]:
    """Roll a GMM measurement through in-force experience at every period.

    Row ``j`` of ``actual_inforce`` is the in-force actually remaining at
    month ``(j+1) * period_months``. The cumulative ratio at each boundary
    is the actual over the originally expected in-force; the CSM is rolled
    segment by segment, each segment releasing over the in-force expected at
    its start, with the experience jump applied at each boundary.
    """
    base_bel = measurement.bel
    base_ra = measurement.ra
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
    rate = 1.0 / discount_start[1] - 1.0

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
    csm[:, 0] = measurement.csm[:, 0]
    cur = measurement.csm[:, 0]
    exp_lines: dict[int, tuple] = {}
    s = 0
    for j, e in enumerate(boundaries + [n_time]):
        seg_csm, seg_acc, seg_rel = _csm_kernel(
            np.ascontiguousarray(cur),
            np.ascontiguousarray(base_inforce[:, s:]),
            rate,
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
    """Slice a PAA measurement into liability-for-remaining-coverage movements."""
    lrc = measurement.lrc
    premium_cf = measurement.cashflows.premium_cf
    revenue = measurement.revenue
    n_time = lrc.shape[1] - 1
    movements: list[PAAPeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        movements.append(PAAPeriodMovement(
            month_start=a,
            month_end=b,
            lrc_opening=lrc[:, a],
            premiums=premium_cf[:, a:b].sum(axis=1),
            revenue=revenue[:, a:b].sum(axis=1),
            lrc_closing=lrc[:, b],
        ))
    return movements


def _roll_forward_vfa(
    measurement: VFAMeasurement, period_months: int
) -> list[VFAPeriodMovement]:
    """Slice a VFA measurement into BEL, RA and CSM movements."""
    bel, ra, csm = measurement.bel, measurement.ra, measurement.csm
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
            f"Reconciliation -- months {self.month_start}-{self.month_end}",
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
    """An IFRS 17 PAA reconciliation of the liability for remaining coverage.

    Portfolio totals for one reporting period: ``revenue`` is shown negative
    -- it releases the LRC -- so opening plus every row equals closing.
    """

    month_start: int
    month_end: int
    lrc_opening: float
    premiums: float
    revenue: float
    lrc_closing: float

    def __str__(self) -> str:
        rows = (
            ("Opening LRC", self.lrc_opening),
            ("Premiums received", self.premiums),
            ("Insurance revenue", self.revenue),
            ("Closing LRC", self.lrc_closing),
        )
        lines = [
            f"PAA reconciliation -- months {self.month_start}-{self.month_end}"
        ]
        for name, value in rows:
            lines.append(f"{name:20}{value:>18,.0f}")
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
            f"VFA reconciliation -- months {self.month_start}-{self.month_end}",
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
