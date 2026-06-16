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
from fastcashflow.curves import forward_rates
from fastcashflow.engine import GMMMeasurement, _require_full
from fastcashflow._measurement_basis import _require_inception
from fastcashflow.io import write_measurement, _write_measurement_columns
from fastcashflow.numerics import _csm_roll
from fastcashflow.projection import reject_account_book
from fastcashflow._paa import PAAMeasurement, _require_full_paa
from fastcashflow._vfa import (
    CSM_BASIS_PARAGRAPH_45, _CSM_TO_MEASUREMENT_BASIS, VFAMeasurement,
    _require_settlement_csm)
from fastcashflow._reinsurance import ReinsuranceMeasurement


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

        lrc_opening + premiums        - revenue                == lrc_closing
        loss_component_opening        - loss_component_release == loss_component_closing
        lic_opening + claims_incurred - claims_paid            == lic_closing

    The LRC (liability for remaining coverage) is built up by premiums and
    released by insurance revenue; the loss component runs off over the
    coverage; the LIC (liability for incurred claims) is built up as claims
    are incurred and run off as they are paid. All are held undiscounted.

    When a settlement tail runs past the horizon, the final period's
    ``lic_closing`` stays non-zero -- the parked LIC residual of claims still
    outstanding at the horizon. The invariant above still holds.
    """

    month_start: int
    month_end: int
    lrc_opening: FloatArray
    premiums: FloatArray
    revenue: FloatArray
    lrc_closing: FloatArray
    loss_component_opening: FloatArray
    loss_component_release: FloatArray
    loss_component_closing: FloatArray
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


@dataclass(frozen=True, slots=True)
class ReinsurancePeriodMovement:
    """One reporting period's movement of a reinsurance-held asset/liability.

    The period covers months ``[month_start, month_end)``. Every array is
    ``(n_mp,)`` and each block reconciles exactly::

        bel_opening + bel_interest  - bel_release  == bel_closing
        ra_opening  + ra_interest   - ra_release   == ra_closing
        csm_opening + csm_accretion - csm_release  == csm_closing

    ``bel`` is the present value of reinsurance premiums less recoveries (a net
    cost when positive); ``csm`` is the net cost / gain of the cover and may be
    negative. ``*_interest`` / ``csm_accretion`` is the unwind at the discount
    rate; ``*_release`` is the expected run-off over the period. There is no
    loss component (Sec. 65).
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

    A mixed-portfolio container
    (:class:`~fastcashflow.portfolio.PortfolioMeasurement` or
    :class:`~fastcashflow.portfolio.PortfolioGroups`) is also accepted: each
    model slot is rolled forward on its own measurement and a
    :class:`~fastcashflow.portfolio.PortfolioMovements` is returned (the
    revision / experience options, being single-GMM-measurement features, are
    rejected on the container).
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
    _require_inception(measurement, "roll_forward()")
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at)
    return _roll_forward_paa(measurement, period_months)


@roll_forward.register
def _(measurement: VFAMeasurement, period_months: int = 12, *,
      revised=None, revised_at=None, actual_inforce=None, experience_at=None):
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _require_settlement_csm(measurement, "roll_forward")
    _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at)
    return _roll_forward_vfa(measurement, period_months)


@roll_forward.register
def _(measurement: ReinsuranceMeasurement, period_months: int = 12, *,
      revised=None, revised_at=None, actual_inforce=None, experience_at=None):
    _require_inception(measurement, "roll_forward()")
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at)
    return _roll_forward_reinsurance(measurement, period_months)


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
    _require_inception(measurement, "roll_forward()")
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _require_full(measurement, "roll_forward")
    # The movement reads claim_cf / morbidity_cf raw as incurred claims; a
    # universal-life account book's account death benefit is not a priced claim,
    # so reject it until the movement nets the account (a follow-up).
    reject_account_book(measurement.cashflows, "roll_forward")
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

    discount_bom = measurement.discount_bom
    # discount_bom is (n_time+1,) for a single basis, or (n_mp, n_time+1) for
    # a segmented (multi-basis) measurement; the last axis is time either way,
    # so the rate is (n_time,) or (n_mp, n_time) accordingly.
    discount_monthly = forward_rates(discount_bom)
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
        re_csm, re_acc, re_rel = _csm_roll(
            csm_after, np.ascontiguousarray(post_inforce[:, k:]),
            discount_monthly[..., k:],
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
        bel_interest = (bel_traj[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
        ra_interest = (ra_traj[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
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
    discount_bom = measurement.discount_bom
    discount_monthly = forward_rates(discount_bom)

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
        seg_csm, seg_acc, seg_rel = _csm_roll(
            np.ascontiguousarray(cur),
            np.ascontiguousarray(base_inforce[:, s:]),
            discount_monthly[..., s:],
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
        bel_interest = ((bel[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
                        + bel_ex * discount_monthly[..., a])
        ra_interest = ((ra[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
                       + ra_ex * discount_monthly[..., a])
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
    _require_full_paa(measurement, "roll_forward")
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
        loss_open = loss_component * revenue[:, a:].sum(axis=1) / safe_revenue
        loss_close = loss_component * revenue[:, b:].sum(axis=1) / safe_revenue
        movements.append(PAAPeriodMovement(
            month_start=a,
            month_end=b,
            lrc_opening=lrc[:, a],
            premiums=premium_cf[:, a:b].sum(axis=1),
            revenue=revenue[:, a:b].sum(axis=1),
            lrc_closing=lrc[:, b],
            loss_component_opening=loss_open,
            loss_component_release=loss_open - loss_close,
            loss_component_closing=loss_close,
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
    _require_full(measurement, "roll_forward")
    bel, ra, csm = measurement.bel_path, measurement.ra_path, measurement.csm_path
    csm_accretion = measurement.csm_accretion
    csm_release = measurement.csm_release
    n_time = csm.shape[1] - 1
    discount_bom = measurement.discount_bom
    # discount_bom is (n_time+1,) for a single basis, or (n_mp, n_time+1) for a
    # segmented (portfolio-stitched) measurement; the last axis is time either
    # way, so discount_monthly is (n_time,) or (n_mp, n_time). The trailing-axis
    # slice serves both -- a bare [a:b] would slice the model-point axis on the
    # 2-D curve.
    discount_monthly = forward_rates(discount_bom)
    movements: list[VFAPeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_interest = (bel[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
        ra_interest = (ra[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
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


def _roll_forward_reinsurance(
    measurement: ReinsuranceMeasurement, period_months: int
) -> list[ReinsurancePeriodMovement]:
    """Slice a reinsurance-held measurement into BEL, RA and CSM movements.

    The reinsurance counterpart of :func:`_roll_forward_vfa`: BEL / RA unwind at
    the discount rate and the CSM accretes and releases over coverage units,
    with no loss component (Sec. 65)."""
    _require_full(measurement, "roll_forward")
    bel, ra, csm = measurement.bel_path, measurement.ra_path, measurement.csm_path
    csm_accretion = measurement.csm_accretion
    csm_release = measurement.csm_release
    n_time = csm.shape[1] - 1
    discount_monthly = forward_rates(measurement.discount_bom)
    movements: list[ReinsurancePeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_interest = (bel[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
        ra_interest = (ra[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
        movements.append(ReinsurancePeriodMovement(
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
    loss_component_opening: float
    loss_component_release: float
    loss_component_closing: float
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
                ("Opening", self.loss_component_opening),
                ("Released", self.loss_component_release),
                ("Closing", self.loss_component_closing),
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
            loss_component_opening=float(m.loss_component_opening.sum()),
            loss_component_release=float(-m.loss_component_release.sum()),
            loss_component_closing=float(m.loss_component_closing.sum()),
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


@dataclass(frozen=True, slots=True, eq=False)
class VFASettlementMovement:
    """One period's IFRS 17 paragraph-45 settlement movement of a VFA book.

    What :func:`fastcashflow.vfa.settle` returns: the opening -> closing
    movement of the BEL, RA, CSM and loss component over one reporting period
    of ``period_months``, per model point. Unlike :class:`VFAPeriodMovement`
    (the *expected* movement sliced from an inception measurement), this is a
    *subsequent measurement*: the closing figures respond to the observed
    account value and in-force count, and the CSM absorbs the future-service
    change per paragraph 45. Every array is ``(n_mp,)`` and each block
    reconciles exactly::

        bel_closing == bel_opening + bel_interest - bel_release + bel_experience
        ra_closing  == ra_opening  + ra_interest  - ra_release  + ra_experience
        csm_closing == csm_opening + csm_accretion + csm_fv_share
                       + csm_future_service + csm_premium_experience
                       + csm_investment_experience
                       - loss_component_reversed
                       + loss_component_recognised - csm_release
        loss_component_closing == loss_component_opening
                       + loss_component_finance - loss_component_amortised
                       - loss_component_reversed + loss_component_recognised
        lic_closing == lic_opening + claims_incurred + lic_finance - claims_paid

    The liability for incurred claims (``lic_opening`` / ``claims_incurred`` /
    ``lic_finance`` / ``claims_paid`` / ``lic_closing``, paragraphs 40(b) /
    42(c) / 103(b)) is present when the basis carries a ``settlement_pattern``:
    benefit claims build it up as incurred and run it off over the pattern. The
    LIC is measured at fulfilment cash flows -- the discounted PV of the unpaid
    run-off (42(c)). It carries NO risk adjustment: the VFA RA prices expense
    risk only (the benefit risk sits in the variable fee), so the incurred
    benefits carry no RA in the LIC either. ``claims_incurred`` and
    ``claims_paid`` stay nominal (``claims_paid`` the residual on the
    undiscounted trajectory); ``lic_finance`` is the reconciling residual (the
    42(c) discount unwind + discounting measurement effect), zero at both dates
    without a pattern. It mirrors the GMM block (which adds the LIC RA).

    and the blocks tie across: ``csm_fv_share + csm_future_service ==
    -(bel_experience + ra_experience)`` -- the paragraph-45 future-service
    change is exactly minus the observed-vs-expected FCF difference.
    ``csm_premium_experience`` (B96(a)) and ``premium_experience_revenue``
    (B97(c)) are the two legs of the premium experience adjustment (actual
    premium received less the expected premium, split by the entity's
    future-service fraction). The future leg is a NEW future-service change with
    no BEL/RA counterpart, so it enters the CSM block but does NOT appear in the
    cross-tie above; the current/past leg is a P&L memo, in no balance
    recursion. Both are zero unless ``state.actual_premium`` is given.
    ``csm_investment_experience`` (B96(c)) is the same for the investment
    component (the account value returned on exits): expected less actual
    account value payable, the whole difference into the CSM, outside the
    cross-tie; zero unless ``state.actual_investment_component`` is given.
    ``loss_component_finance`` / ``loss_component_amortised`` are the
    paragraph-50(a)/51 incurred-service channel of an onerous book -- the
    guarantee-excess + expense release (the claims+expenses pool, excluding the
    account-value investment component) split on the loss-component ratio,
    running the loss component to zero by the end of coverage (52); zero on a
    profitable book.

    Line semantics:

    * ``bel_interest`` / ``ra_interest`` -- the unwind at the underlying-items
      return over the period (the engine's roll-forward convention); the
      fee / crediting wedge of the fund's own growth sits inside the release.
    * ``bel_release`` / ``ra_release`` -- the *expected* run-off, the one
      residual line per block.
    * ``bel_experience`` / ``ra_experience`` -- observed minus expected close,
      the future effect of the account-value and count deviation.
    * ``csm_accretion`` -- ``prior_csm * ((1 + r_m)**period - 1)``, the
      expected financial growth of the CSM. Under the VFA there is no
      paragraph-B72(b) locked-rate accretion; this is the expected part of
      the paragraph-45(b) change, presented jointly with ``csm_fv_share``
      as the financial / entity's-share block.
    * ``csm_fv_share`` -- paragraph 45(b), the change in the entity's share
      of the underlying items: the observed-vs-expected variable-fee PV at
      the closing date (fund-consistent end-of-month weight).
    * ``csm_future_service`` -- paragraph 45(c), every other future-service
      change: the guarantee cost (GMDB / GMAB), the crediting-floor cost and
      the count deviation's future effect.
    * ``loss_component_reversed`` / ``loss_component_recognised`` --
      paragraphs 48 / 50(b): a favourable change reverses the loss component
      before rebuilding the CSM; an unfavourable change beyond the CSM falls
      into the loss component.
    * ``csm_release`` -- paragraph B119, one period-end release of the
      post-adjustment balance over the coverage units provided in the period
      against those provided plus expected from the *opening* date.
    * ``coverage_units_provided`` / ``coverage_units_future`` -- the B119
      numerator and remainder behind that release (expected scale over the
      period, observed scale from the closing date), kept per model point
      so a group-of-contracts settlement can re-pool the release fraction
      at the group grain.
    * ``account_value_closing`` -- the *observed* fund value at the closing
      date, echoed from the input state; with the closing balances it seeds
      the next period's state (:meth:`closing_inputs`).

    v1 limitations (documented, not silent): within-period experience --
    actual deaths, lapses, benefits, expenses, AND the fees actually skimmed
    on the realized fund path -- is assumed equal to expected; only the
    closing count and observed account value deviate. Part of the
    paragraph 45(b) realized entity share is therefore not captured, and the
    period's total comprehensive income is approximate even though
    opening-to-closing balances reconcile. The loss component moves through both
    channels: the paragraph-48/50(b) future-service adjustments (``reversed`` /
    ``recognised``) and the paragraph-50(a)-52 systematic incurred-service
    allocation (``loss_component_finance`` / ``loss_component_amortised``, the
    guarantee-excess + expense pool excluding the account-value investment
    component), which runs the loss component to zero by the end of coverage. An
    opening CSM that embeds a stochastic guarantee time value is accreted
    and released but its time-value component is never remeasured (the
    movement is deterministic, intrinsic-guarantee only). Floors and the
    loss-component algebra operate per model point; within-group offsetting
    between favourable and unfavourable contracts (the group-level CSM floor
    of paragraphs 47-52) is not performed, consistent with the rest of the
    engine.
    """

    period_months: int
    bel_opening: FloatArray
    bel_interest: FloatArray
    bel_release: FloatArray
    bel_experience: FloatArray
    bel_closing: FloatArray
    ra_opening: FloatArray
    ra_interest: FloatArray
    ra_release: FloatArray
    ra_experience: FloatArray
    ra_closing: FloatArray
    csm_opening: FloatArray
    csm_accretion: FloatArray
    csm_fv_share: FloatArray
    csm_future_service: FloatArray
    csm_premium_experience: FloatArray  # B96(a): future-service premium exp, into CSM
    premium_experience_revenue: FloatArray  # B97(c): current/past premium exp, P&L memo
    csm_investment_experience: FloatArray  # B96(c): investment-component exp, into CSM
    claims_experience: FloatArray        # B97(b)/(c): actual-vs-expected claims, P&L memo
    expense_experience: FloatArray       # B97(b)/(c): actual-vs-expected expenses, P&L memo
    loss_component_reversed: FloatArray
    loss_component_recognised: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray
    loss_component_opening: FloatArray
    loss_component_finance: FloatArray   # 51(c): r x pool interest unwind
    loss_component_amortised: FloatArray  # 50(a)/51(a)+(b): the systematic loss reversal
    loss_component_closing: FloatArray
    variable_fee_closing: FloatArray
    coverage_units_provided: FloatArray  # B119 numerator, expected scale
    coverage_units_future: FloatArray    # B119 remainder, observed scale
    account_value_closing: FloatArray    # observed fund value at the close
    lic_opening: FloatArray              # 40(b)/42(c): discounted PV of incurred claims
    claims_incurred: FloatArray          # 42(a)/103(b)(i): claims incurred this period (nominal)
    lic_finance: FloatArray              # 42(c): discount unwind + discounting measurement
    claims_paid: FloatArray              # the settlement-pattern run-off (nominal residual)
    lic_closing: FloatArray
    lock_in_rate: float = 0.0            # state echo only; no VFA locked rate
    model_points: "object | None" = None
    csm_basis: str = CSM_BASIS_PARAGRAPH_45

    @property
    def measurement_basis(self) -> str:
        """Cross-model time-basis discriminator, derived from ``csm_basis``
        (mirrors :class:`~fastcashflow.vfa.VFAMeasurement`)."""
        return _CSM_TO_MEASUREMENT_BASIS[self.csm_basis]

    def closing_inputs(self):
        """The closing-date ``(ModelPoints, InforceState)`` pair that seeds
        the next period's settle: ``prior_csm`` / ``prior_loss_component``
        are this period's closing balances, ``prior_count`` the closing
        count and ``prior_account_value`` the observed closing fund value.
        The caller advances the pair to the next observation date
        (``elapsed_months`` / ``count`` / ``account_value``) before the
        next call."""
        from fastcashflow.model_points import InforceState
        mp = self.model_points
        if mp is None or mp.mp_id is None:
            raise ValueError(
                "closing_inputs() needs the source model points with mp_id "
                "(the settle entry stamps them; per-MP chaining joins by id)")
        state = InforceState(
            mp_id=mp.mp_id,
            elapsed_months=np.asarray(mp.elapsed_months, dtype=np.int64),
            count=np.asarray(mp.count, dtype=np.float64),
            prior_csm=self.csm_closing,
            lock_in_rate=self.lock_in_rate,
            account_value=self.account_value_closing,
            prior_count=np.asarray(mp.count, dtype=np.float64),
            prior_account_value=self.account_value_closing,
            prior_loss_component=self.loss_component_closing,
        )
        return mp, state

    def closing_measurement(self) -> VFAMeasurement:
        """The closing balance sheet as a headline-only
        :class:`~fastcashflow.vfa.VFAMeasurement`, tagged
        ``csm_basis='paragraph_45_settlement'`` -- a settlement figure,
        unlike the carry-only diagnostic, so the carry-only guard does not
        reject it: ``write_measurement`` serialises it, and its figures seed
        next period's ``prior_*`` state. (``report`` / ``group`` /
        ``roll_forward`` still need the full trajectories a headline-only
        result does not carry.) ``time_value`` is zero (the movement is
        intrinsic-guarantee only)."""
        return VFAMeasurement(
            bel=self.bel_closing,
            ra=self.ra_closing,
            csm=self.csm_closing,
            variable_fee=self.variable_fee_closing,
            time_value=np.zeros_like(self.bel_closing),
            loss_component=self.loss_component_closing,
            model_points=self.model_points,
            csm_basis=CSM_BASIS_PARAGRAPH_45,
        )


# Settlement reconciliation display / disclosure block specs -- the single
# source for each settlement family's line spine, shared by the __str__ methods
# below and disclosure.py's reconciliation_to_frame / line_metadata (which
# imports them). Each line: (display name, reconciliation field, IFRS 17
# paragraph, is P&L memo). loss_component_reversed / recognised legitimately
# appear in BOTH the CSM block (where they enter the CSM) and the Loss component
# block (where they run it off).
_GMM_RECON_BLOCKS = (
    ("BEL", (
        ("Opening", "bel_opening", "100(a)", False),
        ("Interest accreted", "bel_interest", "B72(a)", False),
        ("Release for service", "bel_release", "B123", False),
        ("Experience", "bel_experience", "B96", False),
        ("Closing", "bel_closing", "100(a)", False),
    )),
    ("RA", (
        ("Opening", "ra_opening", "101(b)", False),
        ("Interest accreted", "ra_interest", "B72(a)", False),
        ("Release for service", "ra_release", "B124", False),
        ("Experience", "ra_experience", "B96(d)", False),
        ("Closing", "ra_closing", "101(b)", False),
    )),
    ("CSM", (
        ("Opening", "csm_opening", "101(c)", False),
        ("Accretion", "csm_accretion", "44(b)/B72(b)", False),
        ("Experience unlocking", "csm_experience_unlocking", "44(c)/B96", False),
        ("Premium experience", "csm_premium_experience", "B96(a)", False),
        ("Investment experience", "csm_investment_experience", "B96(c)", False),
        ("Loss component reversed", "loss_component_reversed", "50(b)", False),
        ("Loss component recognised", "loss_component_recognised", "48", False),
        ("Release for service", "csm_release", "44(e)/B119", False),
        ("Closing", "csm_closing", "101(c)", False),
    )),
    ("Loss component", (
        ("Opening", "loss_component_opening", "49", False),
        ("Finance", "loss_component_finance", "51(c)", False),
        ("Amortised", "loss_component_amortised", "50(a)", False),
        ("Reversed", "loss_component_reversed", "50(b)", False),
        ("Recognised", "loss_component_recognised", "48", False),
        ("Closing", "loss_component_closing", "49", False),
    )),
    ("LIC", (
        ("Opening", "lic_opening", "100(c)", False),
        ("Claims incurred", "claims_incurred", "42(a)", False),
        ("Finance", "lic_finance", "42(c)", False),
        ("Claims paid", "claims_paid", "100(c)", False),
        ("Closing", "lic_closing", "100(c)", False),
    )),
    ("Memo (P&L)", (
        ("Finance wedge", "finance_wedge", "B97(a)", True),
        ("Premium experience (revenue)", "premium_experience_revenue", "B97(c)", True),
        ("Claims experience", "claims_experience", "B97(b)", True),
        ("Expense experience", "expense_experience", "B97(b)", True),
    )),
)

# VFA settlement reconciliation -- the paragraph-45 CSM (fair-value share +
# future service, no finance wedge) and an account-value-linked LIC.
_VFA_RECON_BLOCKS = (
    ("BEL", (
        ("Opening", "bel_opening", "100(a)", False),
        ("Interest accreted", "bel_interest", "B72(a)", False),
        ("Release for service", "bel_release", "B123", False),
        ("Experience", "bel_experience", "B96", False),
        ("Closing", "bel_closing", "100(a)", False),
    )),
    ("RA", (
        ("Opening", "ra_opening", "101(b)", False),
        ("Interest accreted", "ra_interest", "B72(a)", False),
        ("Release for service", "ra_release", "B124", False),
        ("Experience", "ra_experience", "B96(d)", False),
        ("Closing", "ra_closing", "101(b)", False),
    )),
    ("CSM", (
        ("Opening", "csm_opening", "101(c)", False),
        ("Accretion", "csm_accretion", "45(b)/B72(b)", False),
        ("Fair value share", "csm_fv_share", "45(b)", False),
        ("Future service", "csm_future_service", "45(c)", False),
        ("Premium experience", "csm_premium_experience", "B96(a)", False),
        ("Investment experience", "csm_investment_experience", "B96(c)", False),
        ("Loss component reversed", "loss_component_reversed", "50(b)", False),
        ("Loss component recognised", "loss_component_recognised", "48", False),
        ("Release for service", "csm_release", "45(e)/B119", False),
        ("Closing", "csm_closing", "101(c)", False),
    )),
    ("Loss component", (
        ("Opening", "loss_component_opening", "49", False),
        ("Finance", "loss_component_finance", "51(c)", False),
        ("Amortised", "loss_component_amortised", "50(a)", False),
        ("Reversed", "loss_component_reversed", "50(b)", False),
        ("Recognised", "loss_component_recognised", "48", False),
        ("Closing", "loss_component_closing", "49", False),
    )),
    ("LIC", (
        ("Opening", "lic_opening", "100(c)", False),
        ("Claims incurred", "claims_incurred", "42(a)", False),
        ("Finance", "lic_finance", "42(c)", False),
        ("Claims paid", "claims_paid", "100(c)", False),
        ("Closing", "lic_closing", "100(c)", False),
    )),
    ("Memo (P&L)", (
        ("Premium experience (revenue)", "premium_experience_revenue", "B97(c)", True),
        ("Claims experience", "claims_experience", "B97(b)", True),
        ("Expense experience", "expense_experience", "B97(b)", True),
    )),
)

# Reinsurance-held settlement reconciliation -- no loss component (paragraph 65,
# a reinsurance contract held cannot be onerous); a loss-RECOVERY component
# (66A-66B) instead, and no LIC block.
_REINSURANCE_RECON_BLOCKS = (
    ("BEL", (
        ("Opening", "bel_opening", "100(a)", False),
        ("Interest accreted", "bel_interest", "B72(a)", False),
        ("Release for service", "bel_release", "B123", False),
        ("Experience", "bel_experience", "B96", False),
        ("Closing", "bel_closing", "100(a)", False),
    )),
    ("RA", (
        ("Opening", "ra_opening", "101(b)", False),
        ("Interest accreted", "ra_interest", "B72(a)", False),
        ("Release for service", "ra_release", "B124", False),
        ("Experience", "ra_experience", "B96(d)", False),
        ("Closing", "ra_closing", "101(b)", False),
    )),
    ("CSM", (
        ("Opening", "csm_opening", "101(c)", False),
        ("Accretion", "csm_accretion", "66(b)/B72(b)", False),
        ("Experience unlocking", "csm_experience_unlocking", "66(c)/B96", False),
        ("Release for service", "csm_release", "66(e)/B119", False),
        ("Closing", "csm_closing", "101(c)", False),
    )),
    ("Loss-recovery component", (
        ("Opening", "loss_recovery_opening", "66B", False),
        ("Recognised", "loss_recovery_recognised", "66A", False),
        ("Reversed", "loss_recovery_reversed", "66B", False),
        ("Closing", "loss_recovery_closing", "66B", False),
    )),
    ("Memo (P&L)", (
        ("Finance wedge", "finance_wedge", "B97(a)", True),
    )),
)

# PAA settlement reconciliation -- an LRC (unearned premium) roll, no BEL/RA/CSM.
_PAA_RECON_BLOCKS = (
    ("LRC", (
        ("Opening", "lrc_opening", "100(a)", False),
        ("Premiums received", "premiums", "55(a)", False),
        ("Revenue recognised", "revenue", "B126", False),
        ("Experience", "lrc_experience", "55(b)", False),
        ("Closing", "lrc_closing", "100(a)", False),
    )),
    ("Loss component", (
        ("Opening", "loss_component_opening", "57", False),
        ("Recognised", "loss_component_recognised", "58", False),
        ("Reversed", "loss_component_reversed", "58", False),
        ("Closing", "loss_component_closing", "57", False),
    )),
    ("LIC", (
        ("Opening", "lic_opening", "100(c)", False),
        ("Claims incurred", "claims_incurred", "42(a)", False),
        ("Finance", "lic_finance", "42(c)", False),
        ("Claims paid", "claims_paid", "100(c)", False),
        ("Closing", "lic_closing", "100(c)", False),
    )),
    ("Memo (P&L)", (
        ("Claims experience", "claims_experience", "B97(b)", True),
        ("Expense experience", "expense_experience", "B97(b)", True),
    )),
)


def _format_settlement_reconciliation(recon, title: str, blocks) -> str:
    """Render a settlement reconciliation as a blocked, right-aligned table.

    Shared by the four settlement reconciliation ``__str__`` methods, driven from
    the same ``_*_RECON_BLOCKS`` spec that disclosure.py serialises -- so the
    printed table and the disclosure frame never drift. The Memo (P&L) block is
    rendered like any other block (its lines sit outside the balance recursion;
    the spec flags them ``is_memo`` for the disclosure layer, not for display)."""
    lines = [f"{title} -- {recon.period_months}-month period"]
    for block_title, rows in blocks:
        lines.append(f"  {block_title}")
        for name, field, _para, _memo in rows:
            lines.append(f"    {name:30}{getattr(recon, field):>18,.0f}")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class VFASettlementReconciliation:
    """Portfolio totals of a paragraph-45 VFA settlement movement.

    One reporting period of ``period_months`` (the per-MP valuation dates
    are elapsed months, which differ across cohorts -- so the table is
    labelled by the period length, not by a policy-month range). The
    release and loss-component-reversed rows are *stored* negative -- the
    convention of every reconciliation type here -- so within each block
    the opening plus every row equals the closing.
    """

    period_months: int
    bel_opening: float
    bel_interest: float
    bel_release: float
    bel_experience: float
    bel_closing: float
    ra_opening: float
    ra_interest: float
    ra_release: float
    ra_experience: float
    ra_closing: float
    csm_opening: float
    csm_accretion: float
    csm_fv_share: float
    csm_future_service: float
    csm_premium_experience: float
    premium_experience_revenue: float
    csm_investment_experience: float
    claims_experience: float
    expense_experience: float
    loss_component_reversed: float
    loss_component_recognised: float
    csm_release: float
    csm_closing: float
    loss_component_opening: float
    loss_component_finance: float
    loss_component_amortised: float
    loss_component_closing: float
    lic_opening: float = 0.0
    claims_incurred: float = 0.0
    lic_finance: float = 0.0
    claims_paid: float = 0.0
    lic_closing: float = 0.0

    def __str__(self) -> str:
        return _format_settlement_reconciliation(
            self, "VFA settlement reconciliation", _VFA_RECON_BLOCKS)


def _reconcile_vfa_settlement(
    movements: list[VFASettlementMovement],
) -> list[VFASettlementReconciliation]:
    """Aggregate paragraph-45 settlement movements into portfolio totals.

    Release and loss-component-reversed rows are stored negative (the
    reconciliation display convention), so opening plus every row equals
    closing in each block. Note the CSM block reads the same
    ``loss_component_reversed`` row: a favourable change reverses the loss
    component *instead of* crediting the CSM, so the row subtracts there
    too.
    """
    return [
        VFASettlementReconciliation(
            period_months=m.period_months,
            bel_opening=float(m.bel_opening.sum()),
            bel_interest=float(m.bel_interest.sum()),
            bel_release=float(-m.bel_release.sum()),
            bel_experience=float(m.bel_experience.sum()),
            bel_closing=float(m.bel_closing.sum()),
            ra_opening=float(m.ra_opening.sum()),
            ra_interest=float(m.ra_interest.sum()),
            ra_release=float(-m.ra_release.sum()),
            ra_experience=float(m.ra_experience.sum()),
            ra_closing=float(m.ra_closing.sum()),
            csm_opening=float(m.csm_opening.sum()),
            csm_accretion=float(m.csm_accretion.sum()),
            csm_fv_share=float(m.csm_fv_share.sum()),
            csm_future_service=float(m.csm_future_service.sum()),
            csm_premium_experience=float(m.csm_premium_experience.sum()),
            premium_experience_revenue=float(m.premium_experience_revenue.sum()),
            csm_investment_experience=float(m.csm_investment_experience.sum()),
            claims_experience=float(m.claims_experience.sum()),
            expense_experience=float(m.expense_experience.sum()),
            loss_component_finance=float(m.loss_component_finance.sum()),
            loss_component_amortised=float(-m.loss_component_amortised.sum()),
            loss_component_reversed=float(-m.loss_component_reversed.sum()),
            loss_component_recognised=float(m.loss_component_recognised.sum()),
            csm_release=float(-m.csm_release.sum()),
            csm_closing=float(m.csm_closing.sum()),
            loss_component_opening=float(m.loss_component_opening.sum()),
            loss_component_closing=float(m.loss_component_closing.sum()),
            lic_opening=float(m.lic_opening.sum()),
            claims_incurred=float(m.claims_incurred.sum()),
            lic_finance=float(m.lic_finance.sum()),
            claims_paid=float(-m.claims_paid.sum()),
            lic_closing=float(m.lic_closing.sum()),
        )
        for m in movements
    ]


@dataclass(frozen=True, slots=True)
class ReinsuranceReconciliation:
    """An IFRS 17 reconciliation of a reinsurance-held asset/liability.

    Portfolio totals for one reporting period -- the BEL, RA and CSM each
    reconciled from opening to closing. ``*_finance`` is the unwind at the
    discount rate; ``*_release`` is the run-off, shown negative -- so opening
    plus every row equals closing. There is no loss component (Sec. 65).
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
            f"Reinsurance reconciliation -- months "
            f"{self.month_start + 1}-{self.month_end}",
            f"{'':16}{'BEL':>18}{'RA':>18}{'CSM':>18}",
        ]
        for name, bel, ra, csm in rows:
            lines.append(f"{name:16}{bel:>18,.0f}{ra:>18,.0f}{csm:>18,.0f}")
        return "\n".join(lines)


def _reconcile_reinsurance(
    movements: list[ReinsurancePeriodMovement],
) -> list[ReinsuranceReconciliation]:
    """Aggregate reinsurance period movements into portfolio-total reconciliations."""
    return [
        ReinsuranceReconciliation(
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


@dataclass(frozen=True, slots=True, eq=False)
class GMMSettlementMovement:
    """One period's IFRS 17 paragraph-44 settlement movement of a GMM book.

    What :func:`fastcashflow.gmm.settle` returns: the opening -> closing
    movement of the BEL, RA, CSM and loss component over one reporting
    period, per model point. Every measurement array is ``(n_mp,)`` and each
    block reconciles exactly::

        bel_closing == bel_opening + bel_interest - bel_release + bel_experience
        ra_closing  == ra_opening  + ra_interest  - ra_release  + ra_experience
        csm_closing == csm_opening + csm_accretion + csm_experience_unlocking
                       + csm_premium_experience + csm_investment_experience
                       - loss_component_reversed + loss_component_recognised
                       - csm_release
        loss_component_closing == loss_component_opening
                       + loss_component_finance - loss_component_amortised
                       - loss_component_reversed + loss_component_recognised
        lic_closing == lic_opening + claims_incurred + lic_finance - claims_paid

    The GMM cross identity is THREE-term (unlike the VFA's two-term tie)::

        csm_experience_unlocking + finance_wedge
            == -(bel_experience + ra_experience)

    because B72(c) measures the paragraph-44(c) CSM adjustment at the rates
    determined on initial recognition while the BEL block is current-rate
    (B72(a)); the gap is insurance finance income/expense (B97(a)), carried
    as the named ``finance_wedge`` line OUTSIDE the CSM block. The RA part
    of the change has no rate prescription (B96(d)) and enters the CSM at
    its current measure -- a documented accounting policy.

    ``csm_premium_experience`` (B96(a)) and ``premium_experience_revenue``
    (B97(c)) are the two legs of the premium experience adjustment (actual
    premium received over the period less the expected premium), split by the
    entity's future-service fraction. The future-service leg enters the CSM
    block (it is a NEW future-service change with no BEL/RA counterpart, so it
    does NOT appear in the three-term tie above); the current/past leg is a
    P&L memo (insurance revenue), in NO balance recursion, exactly like
    ``finance_wedge``. Both are zero unless ``state.actual_premium`` is given.

    ``claims_experience`` / ``expense_experience`` (B97(b)/(c)) are the
    within-period claims / expense experience -- the actual claims / expenses
    incurred over the period less the expected -- recognised in the insurance
    service result (P&L memos, in NO balance recursion, not the CSM). Zero
    unless ``state.actual_claims`` / ``state.actual_expenses`` are given.

    ``csm_investment_experience`` (B96(c)) is the investment-component
    counterpart: the expected less the actual investment component (surrender /
    annuity repayments) that becomes payable over the period. The WHOLE
    difference enters the CSM (no fraction -- B96(c) is entirely future
    service), a new future-service change outside the three-term tie; the
    investment component does not touch insurance revenue. Zero unless
    ``state.actual_investment_component`` is given.

    ``csm_accretion`` is direct compounding of the prior CSM at the
    locked-in rate (44(b)/B72(b)); ``csm_release`` is the single period-end
    B119 release on the post-adjustment balance, with the coverage-unit
    fraction ``coverage_units_provided / (coverage_units_provided +
    coverage_units_future)`` (em_open denominator, k_exp/k_obs mixed scale).

    ``loss_component_finance`` (B97(a)/51(c)) and ``loss_component_amortised``
    (49/50(a)/51(a)+(b)) are the INCURRED-service channel of an onerous group,
    distinct from the FUTURE-service ``reversed`` / ``recognised`` lines
    (48/50(b)). As coverage is provided the period's paragraph-51 changes are
    split on the systematic loss-component ratio ``r = loss_component_opening /
    pool_opening`` (``pool_opening`` = the opening PV of remaining claims and
    expenses plus the RA): the loss component accretes ``r`` x the pool's
    interest unwind and amortises ``r`` x the pool's release. The amortised
    amount is the paragraph-49/B123(b) loss reversal -- presented in P&L and
    EXCLUDED from insurance revenue (B124(a)(i) / (b)(iii)). Both are zero on a
    profitable book (``r`` = 0) and the cumulative amortisation runs the loss
    component to zero by the end of coverage (paragraph 52), exact because
    ``r`` is re-derived every period. The future-service algebra acts on the
    POST-amortisation loss component, so ``loss_component_reversed`` is capped
    by the loss component net of this channel.

    ``lic_opening`` / ``claims_incurred`` / ``lic_finance`` / ``claims_paid`` /
    ``lic_closing`` are the liability for incurred claims (paragraphs 40(b) / 42
    / 103(b) / 37), meaningful when the basis carries a ``settlement_pattern``:
    claims build the LIC up as incurred (42(a)) and run it off over the pattern.
    The LIC is measured at fulfilment cash flows -- the discounted PV of the
    unpaid run-off plus the risk adjustment (40(b)/42(c)/37). ``claims_incurred``
    and ``claims_paid`` stay NOMINAL cash amounts (``claims_paid`` the nominal
    residual on the undiscounted trajectory); the discounting and RA move only
    the balances, and ``lic_finance`` is the reconciling residual -- the insurance
    finance (42(c) discount unwind) plus the discounting / RA measurement effect
    -- so ``lic_closing == lic_opening + claims_incurred + lic_finance -
    claims_paid``. The block is entirely expected-scale, reconstructed from the
    projection each period. The LIC RA is the confidence-level margin on the
    discounted run-off, split by risk class (a cost-of-capital LIC run-off is a
    refinement). Without a settlement pattern claims are paid as incurred, so the
    LIC is zero at both dates and ``lic_finance`` is zero.

    v1 presentation limitation: ``lic_finance`` is a single reconciling line, so
    the RA run-off / remeasurement is bundled with the 42(c) time-value movement
    rather than separated into its own insurance-service line. The balances
    (``lic_opening`` / ``lic_closing``) are the correct 40(b)/37 fulfilment cash
    flow; a fully separated P&L attribution (pure 42(c) finance vs RA release vs
    the nominal-minus-PV measurement of newly incurred claims) needs a monthly
    finance-accrual decomposition and is a future refinement.
    """

    bel_opening: FloatArray
    bel_interest: FloatArray
    bel_release: FloatArray
    bel_experience: FloatArray
    bel_closing: FloatArray
    ra_opening: FloatArray
    ra_interest: FloatArray
    ra_release: FloatArray
    ra_experience: FloatArray
    ra_closing: FloatArray
    csm_opening: FloatArray
    csm_accretion: FloatArray            # 44(b)/B72(b): locked-in, direct compounding
    csm_experience_unlocking: FloatArray  # 44(c)/B96(b)(d): locked-in measure
    csm_premium_experience: FloatArray   # B96(a): future-service premium exp, into CSM
    csm_investment_experience: FloatArray  # B96(c): investment-component exp, into CSM
    finance_wedge: FloatArray            # B97(a): current-vs-locked-in gap, not CSM
    premium_experience_revenue: FloatArray  # B97(c): current/past premium exp, P&L memo
    claims_experience: FloatArray        # B97(b)/(c): actual-vs-expected claims, P&L memo
    expense_experience: FloatArray       # B97(b)/(c): actual-vs-expected expenses, P&L memo
    csm_release: FloatArray              # 44(e)/B119: single period-end release
    csm_closing: FloatArray
    loss_component_opening: FloatArray
    loss_component_finance: FloatArray   # 51(c): r x pool interest unwind
    loss_component_amortised: FloatArray  # 50(a)/51(a)+(b): the systematic loss reversal
    loss_component_reversed: FloatArray
    loss_component_recognised: FloatArray
    loss_component_closing: FloatArray
    coverage_units_provided: FloatArray  # k_exp x (tail[em_open] - tail[em_close])
    coverage_units_future: FloatArray    # k_obs x tail[em_close]
    lic_opening: FloatArray              # 40(b)/42/37: discounted PV + RA of incurred claims
    claims_incurred: FloatArray          # 42(a)/103(b)(i): claims incurred this period (nominal)
    lic_finance: FloatArray              # 42(c): discount unwind + discounting/RA measurement
    claims_paid: FloatArray              # the settlement-pattern run-off (nominal residual)
    lic_closing: FloatArray
    period_months: int = 12
    lock_in_rate: float = 0.0
    model_points: object | None = None
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """The closing-date ``(ModelPoints, InforceState)`` pair that seeds
        the next period's settle: ``prior_csm`` / ``prior_loss_component``
        are this period's closing balances and ``prior_count`` the closing
        count. The caller advances the pair to the next observation date
        (``elapsed_months`` / ``count``) before the next call."""
        from fastcashflow.model_points import InforceState
        mp = self.model_points
        if mp is None or mp.mp_id is None:
            raise ValueError(
                "closing_inputs() needs the source model points with mp_id "
                "(the settle entry stamps them; per-MP chaining joins by id)")
        state = InforceState(
            mp_id=mp.mp_id,
            elapsed_months=np.asarray(mp.elapsed_months, dtype=np.int64),
            count=np.asarray(mp.count, dtype=np.float64),
            prior_csm=self.csm_closing,
            lock_in_rate=self.lock_in_rate,
            prior_count=np.asarray(mp.count, dtype=np.float64),
            prior_loss_component=self.loss_component_closing,
        )
        return mp, state


@dataclass(frozen=True, slots=True)
class GMMSettlementReconciliation:
    """Portfolio totals of a :class:`GMMSettlementMovement` -- the
    paragraph-44 settlement table. Release and loss-component-reversed rows
    are stored negative (display convention), so opening plus every row of a
    block equals its closing; ``finance_wedge`` keeps the movement sign (it
    is a P&L line outside the CSM block, not a CSM row)."""

    period_months: int
    bel_opening: float
    bel_interest: float
    bel_release: float
    bel_experience: float
    bel_closing: float
    ra_opening: float
    ra_interest: float
    ra_release: float
    ra_experience: float
    ra_closing: float
    csm_opening: float
    csm_accretion: float
    csm_experience_unlocking: float
    csm_premium_experience: float
    csm_investment_experience: float
    finance_wedge: float
    premium_experience_revenue: float
    claims_experience: float
    expense_experience: float
    loss_component_finance: float
    loss_component_amortised: float
    loss_component_reversed: float
    loss_component_recognised: float
    csm_release: float
    csm_closing: float
    loss_component_opening: float
    loss_component_closing: float
    lic_opening: float = 0.0
    claims_incurred: float = 0.0
    lic_finance: float = 0.0
    claims_paid: float = 0.0
    lic_closing: float = 0.0

    def __str__(self) -> str:
        return _format_settlement_reconciliation(
            self, "GMM settlement reconciliation", _GMM_RECON_BLOCKS)


def _reconcile_gmm_settlement(
    movements: list[GMMSettlementMovement],
) -> list[GMMSettlementReconciliation]:
    """Aggregate paragraph-44 settlement movements into portfolio totals."""
    return [
        GMMSettlementReconciliation(
            period_months=m.period_months,
            bel_opening=float(m.bel_opening.sum()),
            bel_interest=float(m.bel_interest.sum()),
            bel_release=float(-m.bel_release.sum()),
            bel_experience=float(m.bel_experience.sum()),
            bel_closing=float(m.bel_closing.sum()),
            ra_opening=float(m.ra_opening.sum()),
            ra_interest=float(m.ra_interest.sum()),
            ra_release=float(-m.ra_release.sum()),
            ra_experience=float(m.ra_experience.sum()),
            ra_closing=float(m.ra_closing.sum()),
            csm_opening=float(m.csm_opening.sum()),
            csm_accretion=float(m.csm_accretion.sum()),
            csm_experience_unlocking=float(m.csm_experience_unlocking.sum()),
            csm_premium_experience=float(m.csm_premium_experience.sum()),
            csm_investment_experience=float(m.csm_investment_experience.sum()),
            finance_wedge=float(m.finance_wedge.sum()),
            premium_experience_revenue=float(m.premium_experience_revenue.sum()),
            claims_experience=float(m.claims_experience.sum()),
            expense_experience=float(m.expense_experience.sum()),
            loss_component_finance=float(m.loss_component_finance.sum()),
            loss_component_amortised=float(-m.loss_component_amortised.sum()),
            loss_component_reversed=float(-m.loss_component_reversed.sum()),
            loss_component_recognised=float(m.loss_component_recognised.sum()),
            csm_release=float(-m.csm_release.sum()),
            csm_closing=float(m.csm_closing.sum()),
            loss_component_opening=float(m.loss_component_opening.sum()),
            loss_component_closing=float(m.loss_component_closing.sum()),
            lic_opening=float(m.lic_opening.sum()),
            claims_incurred=float(m.claims_incurred.sum()),
            lic_finance=float(m.lic_finance.sum()),
            claims_paid=float(-m.claims_paid.sum()),
            lic_closing=float(m.lic_closing.sum()),
        )
        for m in movements
    ]


@dataclass(frozen=True, slots=True, eq=False)
class ReinsuranceSettlementMovement:
    """One period's IFRS 17 paragraph-66 settlement movement of a reinsurance
    contract held.

    The reinsurance counterpart of :class:`GMMSettlementMovement`. The BEL / RA
    blocks and the CSM accretion / future-service unlocking / finance wedge /
    B119 release are identical to the GMM settlement, with ONE modification:
    a reinsurance contract held cannot be onerous (paragraph 65), so the CSM is
    NOT floored and there is NO loss component. The closing CSM is simply::

        csm_closing == csm_opening + csm_accretion + csm_experience_unlocking
                       - csm_release

    and may be negative throughout -- a net cost of cover, deferred and
    amortised. The three-term cross identity still holds (the future-service
    change is measured at the B72(c) locked-in rate, the wedge to the
    current-rate BEL block is insurance finance income/expense)::

        csm_experience_unlocking + finance_wedge
            == -(bel_experience + ra_experience)

    ``loss_recovery_opening`` / ``loss_recovery_recognised`` /
    ``loss_recovery_reversed`` / ``loss_recovery_closing`` are the
    loss-recovery component (paragraphs 66A-66B), present when the cover is held
    over an ONEROUS underlying group: a separate tracked balance on the asset
    for remaining coverage, re-derived each period as the underlying group's
    loss component x the claim recovery % (B95B / B119D) and amortised in
    lock-step with the underlying loss component (B119F, paragraphs 50-52) --
    its change is a recovery recognised / reversed in P&L, excluded from the
    premium allocation. It does NOT adjust the CSM here (the 66A CSM effect is a
    one-time inception event in ``measure_reinsurance``: csm_after = csm0 -
    loss_recovery). Identity::

        loss_recovery_closing == loss_recovery_opening
            + loss_recovery_recognised - loss_recovery_reversed

    Zero unless ``underlying_loss_opening`` / ``underlying_loss_closing`` are
    supplied (byte-identical to a book with no onerous underlying).
    """

    bel_opening: FloatArray
    bel_interest: FloatArray
    bel_release: FloatArray
    bel_experience: FloatArray
    bel_closing: FloatArray
    ra_opening: FloatArray
    ra_interest: FloatArray
    ra_release: FloatArray
    ra_experience: FloatArray
    ra_closing: FloatArray
    csm_opening: FloatArray
    csm_accretion: FloatArray            # 66(b)/B72(b): locked-in, direct compounding
    csm_experience_unlocking: FloatArray  # 66(c): future-service change, no floor
    finance_wedge: FloatArray            # B97(a): current-vs-locked-in gap, P&L
    csm_release: FloatArray              # 66(e)/B119: single period-end release
    csm_closing: FloatArray
    loss_recovery_opening: FloatArray      # 66B/B119F: underlying loss x recovery %
    loss_recovery_recognised: FloatArray   # more underlying loss -> more recovery
    loss_recovery_reversed: FloatArray     # underlying loss amortises -> recovery reverses (P&L)
    loss_recovery_closing: FloatArray
    coverage_units_provided: FloatArray
    coverage_units_future: FloatArray
    period_months: int = 12
    lock_in_rate: float = 0.0
    model_points: object | None = None
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """The closing-date ``(ModelPoints, InforceState)`` pair that seeds the
        next period's settle: ``prior_csm`` is this period's closing CSM (which
        may be negative -- there is no loss component) and ``prior_count`` the
        closing count. The caller advances ``elapsed_months`` / ``count`` to the
        next observation date before the next call."""
        from fastcashflow.model_points import InforceState
        mp = self.model_points
        if mp is None or mp.mp_id is None:
            raise ValueError(
                "closing_inputs() needs the source model points with mp_id "
                "(the settle entry stamps them; per-MP chaining joins by id)")
        state = InforceState(
            mp_id=mp.mp_id,
            elapsed_months=np.asarray(mp.elapsed_months, dtype=np.int64),
            count=np.asarray(mp.count, dtype=np.float64),
            prior_csm=self.csm_closing,
            lock_in_rate=self.lock_in_rate,
            prior_count=np.asarray(mp.count, dtype=np.float64),
        )
        return mp, state


@dataclass(frozen=True, slots=True)
class ReinsuranceSettlementReconciliation:
    """Portfolio totals of a :class:`ReinsuranceSettlementMovement` -- the
    paragraph-66 settlement table. Release rows are stored negative (display
    convention); ``finance_wedge`` keeps the movement sign (a P&L line outside
    the CSM block). There is no loss-component row -- a reinsurance contract
    held cannot be onerous."""

    period_months: int
    bel_opening: float
    bel_interest: float
    bel_release: float
    bel_experience: float
    bel_closing: float
    ra_opening: float
    ra_interest: float
    ra_release: float
    ra_experience: float
    ra_closing: float
    csm_opening: float
    csm_accretion: float
    csm_experience_unlocking: float
    finance_wedge: float
    csm_release: float
    csm_closing: float
    loss_recovery_opening: float = 0.0
    loss_recovery_recognised: float = 0.0
    loss_recovery_reversed: float = 0.0
    loss_recovery_closing: float = 0.0

    def __str__(self) -> str:
        return _format_settlement_reconciliation(
            self, "Reinsurance settlement reconciliation",
            _REINSURANCE_RECON_BLOCKS)


def _reconcile_reinsurance_settlement(
    movements: list[ReinsuranceSettlementMovement],
) -> list[ReinsuranceSettlementReconciliation]:
    """Aggregate paragraph-66 reinsurance settlement movements into totals."""
    return [
        ReinsuranceSettlementReconciliation(
            period_months=m.period_months,
            bel_opening=float(m.bel_opening.sum()),
            bel_interest=float(m.bel_interest.sum()),
            bel_release=float(-m.bel_release.sum()),
            bel_experience=float(m.bel_experience.sum()),
            bel_closing=float(m.bel_closing.sum()),
            ra_opening=float(m.ra_opening.sum()),
            ra_interest=float(m.ra_interest.sum()),
            ra_release=float(-m.ra_release.sum()),
            ra_experience=float(m.ra_experience.sum()),
            ra_closing=float(m.ra_closing.sum()),
            csm_opening=float(m.csm_opening.sum()),
            csm_accretion=float(m.csm_accretion.sum()),
            csm_experience_unlocking=float(m.csm_experience_unlocking.sum()),
            finance_wedge=float(m.finance_wedge.sum()),
            csm_release=float(-m.csm_release.sum()),
            csm_closing=float(m.csm_closing.sum()),
            loss_recovery_opening=float(m.loss_recovery_opening.sum()),
            loss_recovery_recognised=float(m.loss_recovery_recognised.sum()),
            loss_recovery_reversed=float(-m.loss_recovery_reversed.sum()),
            loss_recovery_closing=float(m.loss_recovery_closing.sum()),
        )
        for m in movements
    ]


@dataclass(frozen=True, slots=True, eq=False)
class PAASettlementMovement:
    """One period's IFRS 17 paragraph-55(b) settlement movement of a PAA book.

    What :func:`fastcashflow.paa.settle` returns: the opening -> closing
    movement of the LRC, loss component, and LIC over one reporting period,
    per model point. Every measurement array is ``(n_mp,)`` and each block
    reconciles exactly::

        lrc_closing == lrc_opening + premiums - revenue + lrc_experience
        loss_component_closing == loss_component_opening
                       + loss_component_recognised - loss_component_reversed
        lic_closing == lic_opening + claims_incurred + lic_finance - claims_paid

    The LRC follows Sec. 55(b), with insurance revenue allocated under
    Sec. B126. The loss component is recalculated under Sec. 57-58 at each
    date rather than carried, so exactly one of the recognised / reversed
    rows is positive. The LIC block supports settlement-pattern books and
    provides the Sec. 100(c) incurred-claims movement, measured at fulfilment
    cash flows -- the discounted PV of the unpaid run-off plus the risk
    adjustment (40(b)/42(c)/37), exactly like the GMM LIC; ``claims_incurred`` /
    ``claims_paid`` stay nominal and ``lic_finance`` is the reconciling
    residual. (Sec. 59(b) permits omitting the LIC discounting for <=1yr claims;
    discounting is also compliant and kept uniform with the GMM block.) There is
    no CSM block -- the PAA carries no CSM -- and the LRC itself stays
    undiscounted (Sec. 56); the finance line is on the LIC only.
    """

    lrc_opening: FloatArray
    premiums: FloatArray
    revenue: FloatArray
    lrc_experience: FloatArray
    lrc_closing: FloatArray
    loss_component_opening: FloatArray
    loss_component_recognised: FloatArray
    loss_component_reversed: FloatArray
    loss_component_closing: FloatArray
    lic_opening: FloatArray
    claims_incurred: FloatArray
    lic_finance: FloatArray
    claims_paid: FloatArray
    lic_closing: FloatArray
    claims_experience: FloatArray        # B97(b)/(c): actual-vs-expected claims, P&L memo
    expense_experience: FloatArray       # B97(b)/(c): actual-vs-expected expenses, P&L memo
    period_months: int = 12
    revenue_basis: str = "time"
    model_points: object | None = None
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """The closing-date ``(ModelPoints, InforceState)`` pair that seeds
        the next period's settle. The PAA has no CSM and no locked-in rate,
        so those state slots carry neutral values; the closing loss component
        is preserved for state-file continuity, though the next settle
        recalculates it under Sec. 57-58 rather than reading it."""
        from fastcashflow.model_points import InforceState
        mp = self.model_points
        if mp is None or mp.mp_id is None:
            raise ValueError(
                "closing_inputs() needs the source model points with mp_id "
                "(the settle entry stamps them; per-MP chaining joins by id)")
        n_mp = self.lrc_closing.shape[0]
        state = InforceState(
            mp_id=mp.mp_id,
            elapsed_months=np.asarray(mp.elapsed_months, dtype=np.int64),
            count=np.asarray(mp.count, dtype=np.float64),
            prior_csm=np.zeros(n_mp, dtype=np.float64),
            lock_in_rate=0.0,
            prior_count=np.asarray(mp.count, dtype=np.float64),
            prior_loss_component=self.loss_component_closing,
            # carry the closing LIC so the next period -- in particular a
            # pure-LIC-runoff close past the contract boundary -- can run the
            # incurred-claims tail down with no in-force to reconstruct it from.
            prior_lic=self.lic_closing,
        )
        return mp, state


@dataclass(frozen=True, slots=True)
class PAASettlementReconciliation:
    """Portfolio totals of a :class:`PAASettlementMovement` -- the
    paragraph-55(b) settlement table. Revenue, claims-paid and
    loss-component-reversed rows are stored negative (display convention),
    so opening plus every row of a block equals its closing; the movement
    keeps those lines positive."""

    period_months: int
    revenue_basis: str
    lrc_opening: float
    premiums: float
    revenue: float
    lrc_experience: float
    lrc_closing: float
    loss_component_opening: float
    loss_component_recognised: float
    loss_component_reversed: float
    loss_component_closing: float
    lic_opening: float
    claims_incurred: float
    lic_finance: float
    claims_paid: float
    lic_closing: float
    claims_experience: float = 0.0
    expense_experience: float = 0.0

    def __str__(self) -> str:
        return _format_settlement_reconciliation(
            self, "PAA settlement reconciliation", _PAA_RECON_BLOCKS)


def _reconcile_paa_settlement(
    movements: list[PAASettlementMovement],
) -> list[PAASettlementReconciliation]:
    """Aggregate paragraph-55(b) settlement movements into portfolio totals."""
    return [
        PAASettlementReconciliation(
            period_months=m.period_months,
            revenue_basis=m.revenue_basis,
            lrc_opening=float(m.lrc_opening.sum()),
            premiums=float(m.premiums.sum()),
            revenue=float(-m.revenue.sum()),
            lrc_experience=float(m.lrc_experience.sum()),
            lrc_closing=float(m.lrc_closing.sum()),
            loss_component_opening=float(m.loss_component_opening.sum()),
            loss_component_recognised=float(m.loss_component_recognised.sum()),
            loss_component_reversed=float(-m.loss_component_reversed.sum()),
            loss_component_closing=float(m.loss_component_closing.sum()),
            lic_opening=float(m.lic_opening.sum()),
            claims_incurred=float(m.claims_incurred.sum()),
            lic_finance=float(m.lic_finance.sum()),
            claims_paid=float(-m.claims_paid.sum()),
            lic_closing=float(m.lic_closing.sum()),
            claims_experience=float(m.claims_experience.sum()),
            expense_experience=float(m.expense_experience.sum()),
        )
        for m in movements
    ]


@write_measurement.register
def _(movement: PAASettlementMovement, path, *, ids=None):
    n = movement.lrc_closing.shape[0]
    cols = {name: getattr(movement, name) for name in _PAA_SETTLEMENT_LINES}
    cols["revenue_basis"] = [movement.revenue_basis] * n
    cols["measurement_basis"] = [movement.measurement_basis] * n
    # The closing-state chain columns ride only when the source model
    # points are stamped (the settle entry always stamps them); a
    # hand-built movement writes the lines and markers alone.
    if movement.model_points is not None:
        cols["elapsed_months"] = np.asarray(
            movement.model_points.elapsed_months, dtype=np.int64)
        cols["count"] = np.asarray(
            movement.model_points.count, dtype=np.float64)
    _write_measurement_columns(cols, path, ids)


@write_measurement.register
def _(movement: GMMSettlementMovement, path, *, ids=None):
    n = movement.bel_closing.shape[0]
    cols = {name: getattr(movement, name) for name in _GMM_SETTLEMENT_LINES}
    cols["lock_in_rate"] = np.full(n, movement.lock_in_rate)
    cols["measurement_basis"] = [movement.measurement_basis] * n
    # The closing-state chain columns ride only when the source model
    # points are stamped (the settle entries always stamp them); a
    # hand-built movement writes the lines and markers alone.
    if movement.model_points is not None:
        cols["elapsed_months"] = np.asarray(
            movement.model_points.elapsed_months, dtype=np.int64)
        cols["count"] = np.asarray(
            movement.model_points.count, dtype=np.float64)
    _write_measurement_columns(cols, path, ids)


@write_measurement.register
def _(movement: ReinsuranceSettlementMovement, path, *, ids=None):
    n = movement.bel_closing.shape[0]
    cols = {name: getattr(movement, name) for name in _REINSURANCE_SETTLEMENT_LINES}
    cols["lock_in_rate"] = np.full(n, movement.lock_in_rate)
    cols["measurement_basis"] = [movement.measurement_basis] * n
    if movement.model_points is not None:
        cols["elapsed_months"] = np.asarray(
            movement.model_points.elapsed_months, dtype=np.int64)
        cols["count"] = np.asarray(
            movement.model_points.count, dtype=np.float64)
    _write_measurement_columns(cols, path, ids)


@write_measurement.register
def _(movement: VFASettlementMovement, path, *, ids=None):
    n = movement.bel_closing.shape[0]
    cols = {name: getattr(movement, name) for name in _VFA_SETTLEMENT_LINES}
    cols["lock_in_rate"] = np.full(n, movement.lock_in_rate)
    cols["measurement_basis"] = [movement.measurement_basis] * n
    if movement.model_points is not None:
        cols["elapsed_months"] = np.asarray(
            movement.model_points.elapsed_months, dtype=np.int64)
        cols["count"] = np.asarray(
            movement.model_points.count, dtype=np.float64)
    _write_measurement_columns(cols, path, ids)


@singledispatch
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

    The base implementation takes a list of movements (dispatch falls through
    to it for any list); a mixed-portfolio
    :class:`~fastcashflow.portfolio.PortfolioMovements` registers its own arm
    (returning a :class:`~fastcashflow.portfolio.PortfolioReconciliation`).
    """
    if movements and isinstance(movements[0], PAAPeriodMovement):
        return _reconcile_paa(movements)
    if movements and isinstance(movements[0], VFAPeriodMovement):
        return _reconcile_vfa(movements)
    if movements and isinstance(movements[0], VFASettlementMovement):
        return _reconcile_vfa_settlement(movements)
    if movements and isinstance(movements[0], GMMSettlementMovement):
        return _reconcile_gmm_settlement(movements)
    if movements and isinstance(movements[0], PAASettlementMovement):
        return _reconcile_paa_settlement(movements)
    if movements and isinstance(movements[0], ReinsuranceSettlementMovement):
        return _reconcile_reinsurance_settlement(movements)
    if movements and isinstance(movements[0], ReinsurancePeriodMovement):
        return _reconcile_reinsurance(movements)
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


# ---------------------------------------------------------------------------
# Settlement aggregates -- bounded-memory portfolio totals of the movements
# ---------------------------------------------------------------------------

# Every per-MP array line of the settlement movements, in movement sign.
# The aggregate entries sum exactly these; the scalar / reference fields
# (period_months, lock_in_rate, model_points, csm_basis) follow their own
# rules -- identity across chunks for the scalars, dropped for the
# references (a sum has no per-MP source to point back to).
# The order is the write_measurement output column order -- the writer arm drives
# its columns from this tuple, so the spine has one source. (The disclosure /
# __str__ block order is separate, _GMM_RECON_BLOCKS.)
_GMM_SETTLEMENT_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_experience_unlocking",
    "csm_premium_experience", "csm_investment_experience",
    "claims_experience", "expense_experience",
    "finance_wedge", "premium_experience_revenue",
    "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_finance",
    "loss_component_amortised", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
    "coverage_units_provided", "coverage_units_future",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
)

_VFA_SETTLEMENT_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_fv_share", "csm_future_service",
    "csm_premium_experience", "premium_experience_revenue",
    "csm_investment_experience", "claims_experience", "expense_experience",
    "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_finance",
    "loss_component_amortised", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
    "variable_fee_closing", "account_value_closing",
    "coverage_units_provided", "coverage_units_future",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
)

_REINSURANCE_SETTLEMENT_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_experience_unlocking",
    "finance_wedge", "csm_release", "csm_closing",
    "loss_recovery_opening", "loss_recovery_recognised",
    "loss_recovery_reversed", "loss_recovery_closing",
    "coverage_units_provided", "coverage_units_future",
)

_PAA_SETTLEMENT_LINES = (
    "lrc_opening", "premiums", "revenue", "lrc_experience", "lrc_closing",
    "loss_component_opening", "loss_component_recognised",
    "loss_component_reversed", "loss_component_closing",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
    "claims_experience", "expense_experience",
)

_AGGREGATE_NO_CHAIN = (
    "an aggregate cannot seed the next period: chaining needs the per-MP "
    "closing balances, which the sums no longer carry. Chain through the "
    "per-MP movement's closing_inputs() instead (settle the book in row "
    "blocks if it does not fit in memory)."
)


@dataclass(frozen=True, slots=True)
class GMMSettlementAggregate:
    """Portfolio totals of the paragraph-44 settlement movement.

    What :func:`fastcashflow.gmm.settle_aggregate` returns: every line of
    :class:`GMMSettlementMovement` summed over the model-point axis, in
    bounded memory. The lines keep the MOVEMENT sign -- the release and
    loss-component-reversed totals are positive run-offs, exactly like the
    per-MP movement; :func:`reconcile` applies the display negation. Each
    block therefore foots in movement form::

        bel_closing == bel_opening + bel_interest - bel_release + bel_experience

    and ``reconcile(aggregate)`` equals the per-MP movement's
    reconciliation table fieldwise.

    An aggregate is not a chaining citizen: the next period's settle needs
    per-MP prior balances, which the sums no longer carry --
    :meth:`closing_inputs` raises ValueError.
    """

    period_months: int
    lock_in_rate: float
    bel_opening: float
    bel_interest: float
    bel_release: float
    bel_experience: float
    bel_closing: float
    ra_opening: float
    ra_interest: float
    ra_release: float
    ra_experience: float
    ra_closing: float
    csm_opening: float
    csm_accretion: float
    csm_experience_unlocking: float
    csm_premium_experience: float
    csm_investment_experience: float
    finance_wedge: float
    premium_experience_revenue: float
    claims_experience: float
    expense_experience: float
    csm_release: float
    csm_closing: float
    loss_component_opening: float
    loss_component_finance: float
    loss_component_amortised: float
    loss_component_reversed: float
    loss_component_recognised: float
    loss_component_closing: float
    coverage_units_provided: float
    coverage_units_future: float
    lic_opening: float = 0.0
    claims_incurred: float = 0.0
    lic_finance: float = 0.0
    claims_paid: float = 0.0
    lic_closing: float = 0.0
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """Always raises -- see the class docstring."""
        raise ValueError(_AGGREGATE_NO_CHAIN)


@dataclass(frozen=True, slots=True)
class ReinsuranceSettlementAggregate:
    """Portfolio totals of the paragraph-66 reinsurance settlement movement.

    What :func:`fastcashflow.reinsurance.settle_aggregate` returns: every line
    of :class:`ReinsuranceSettlementMovement` summed over the model-point axis,
    movement-positive (``reconcile`` applies the display negation and
    reproduces the per-MP movement's table). There is no loss-component line --
    a reinsurance contract held cannot be onerous. :meth:`closing_inputs`
    raises -- chaining needs the per-MP balances.
    """

    period_months: int
    lock_in_rate: float
    bel_opening: float
    bel_interest: float
    bel_release: float
    bel_experience: float
    bel_closing: float
    ra_opening: float
    ra_interest: float
    ra_release: float
    ra_experience: float
    ra_closing: float
    csm_opening: float
    csm_accretion: float
    csm_experience_unlocking: float
    finance_wedge: float
    csm_release: float
    csm_closing: float
    coverage_units_provided: float
    coverage_units_future: float
    loss_recovery_opening: float = 0.0
    loss_recovery_recognised: float = 0.0
    loss_recovery_reversed: float = 0.0
    loss_recovery_closing: float = 0.0
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """Always raises -- see the class docstring."""
        raise ValueError(_AGGREGATE_NO_CHAIN)


@dataclass(frozen=True, slots=True)
class VFASettlementAggregate:
    """Portfolio totals of the paragraph-45 settlement movement.

    What :func:`fastcashflow.vfa.settle_aggregate` returns: every line of
    :class:`VFASettlementMovement` summed over the model-point axis, in
    bounded memory, movement-positive (the display negation happens in
    :func:`reconcile`). ``reconcile(aggregate)`` equals the per-MP
    movement's reconciliation table fieldwise, and :meth:`closing_inputs`
    raises ValueError -- chaining needs per-MP balances.
    """

    period_months: int
    bel_opening: float
    bel_interest: float
    bel_release: float
    bel_experience: float
    bel_closing: float
    ra_opening: float
    ra_interest: float
    ra_release: float
    ra_experience: float
    ra_closing: float
    csm_opening: float
    csm_accretion: float
    csm_fv_share: float
    csm_future_service: float
    csm_premium_experience: float
    premium_experience_revenue: float
    csm_investment_experience: float
    claims_experience: float
    expense_experience: float
    csm_release: float
    csm_closing: float
    loss_component_opening: float
    loss_component_finance: float
    loss_component_amortised: float
    loss_component_reversed: float
    loss_component_recognised: float
    loss_component_closing: float
    variable_fee_closing: float
    account_value_closing: float
    coverage_units_provided: float
    coverage_units_future: float
    lic_opening: float = 0.0
    claims_incurred: float = 0.0
    lic_finance: float = 0.0
    claims_paid: float = 0.0
    lic_closing: float = 0.0
    lock_in_rate: float = 0.0            # state echo only; no VFA locked rate
    csm_basis: str = CSM_BASIS_PARAGRAPH_45

    @property
    def measurement_basis(self) -> str:
        """Cross-model time-basis discriminator, derived from ``csm_basis``
        (mirrors :class:`VFASettlementMovement`)."""
        return _CSM_TO_MEASUREMENT_BASIS[self.csm_basis]

    def closing_inputs(self):
        """Always raises -- see the class docstring."""
        raise ValueError(_AGGREGATE_NO_CHAIN)


@dataclass(frozen=True, slots=True)
class PAASettlementAggregate:
    """Portfolio totals of the paragraph-55(b) PAA settlement movement.

    What :func:`fastcashflow.paa.settle_aggregate` returns: every line of
    :class:`PAASettlementMovement` summed over the model-point axis,
    movement-positive (``reconcile`` applies the display negation of the
    revenue / claims-paid / loss-component-reversed rows and reproduces the
    per-MP movement's table). There is no CSM block -- the PAA holds the LRC
    undiscounted and carries no CSM -- but the LIC carries a finance line (the
    discount unwind on incurred claims). :meth:`closing_inputs` raises --
    chaining needs the per-MP balances.
    """

    period_months: int
    revenue_basis: str
    lrc_opening: float
    premiums: float
    revenue: float
    lrc_experience: float
    lrc_closing: float
    loss_component_opening: float
    loss_component_recognised: float
    loss_component_reversed: float
    loss_component_closing: float
    lic_opening: float
    claims_incurred: float
    lic_finance: float
    claims_paid: float
    lic_closing: float
    claims_experience: float = 0.0
    expense_experience: float = 0.0
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """Always raises -- see the class docstring."""
        raise ValueError(_AGGREGATE_NO_CHAIN)


@reconcile.register
def _(aggregate: GMMSettlementAggregate) -> GMMSettlementReconciliation:
    """The paragraph-44 settlement table of an aggregate -- identical to
    reconciling the per-MP movement (the oracle identity); the display
    negation of the run-off rows happens here, never in the aggregate."""
    a = aggregate
    return GMMSettlementReconciliation(
        period_months=a.period_months,
        bel_opening=a.bel_opening,
        bel_interest=a.bel_interest,
        bel_release=-a.bel_release,
        bel_experience=a.bel_experience,
        bel_closing=a.bel_closing,
        ra_opening=a.ra_opening,
        ra_interest=a.ra_interest,
        ra_release=-a.ra_release,
        ra_experience=a.ra_experience,
        ra_closing=a.ra_closing,
        csm_opening=a.csm_opening,
        csm_accretion=a.csm_accretion,
        csm_experience_unlocking=a.csm_experience_unlocking,
        csm_premium_experience=a.csm_premium_experience,
        csm_investment_experience=a.csm_investment_experience,
        finance_wedge=a.finance_wedge,
        premium_experience_revenue=a.premium_experience_revenue,
        claims_experience=a.claims_experience,
        expense_experience=a.expense_experience,
        loss_component_finance=a.loss_component_finance,
        loss_component_amortised=-a.loss_component_amortised,
        loss_component_reversed=-a.loss_component_reversed,
        loss_component_recognised=a.loss_component_recognised,
        csm_release=-a.csm_release,
        csm_closing=a.csm_closing,
        loss_component_opening=a.loss_component_opening,
        loss_component_closing=a.loss_component_closing,
        lic_opening=a.lic_opening,
        claims_incurred=a.claims_incurred,
        lic_finance=a.lic_finance,
        claims_paid=-a.claims_paid,
        lic_closing=a.lic_closing,
    )


@reconcile.register
def _(aggregate: ReinsuranceSettlementAggregate
      ) -> ReinsuranceSettlementReconciliation:
    """The paragraph-66 reinsurance settlement table of an aggregate --
    identical to reconciling the per-MP movement; run-off rows display-negated
    here, never in the aggregate."""
    a = aggregate
    return ReinsuranceSettlementReconciliation(
        period_months=a.period_months,
        bel_opening=a.bel_opening,
        bel_interest=a.bel_interest,
        bel_release=-a.bel_release,
        bel_experience=a.bel_experience,
        bel_closing=a.bel_closing,
        ra_opening=a.ra_opening,
        ra_interest=a.ra_interest,
        ra_release=-a.ra_release,
        ra_experience=a.ra_experience,
        ra_closing=a.ra_closing,
        csm_opening=a.csm_opening,
        csm_accretion=a.csm_accretion,
        csm_experience_unlocking=a.csm_experience_unlocking,
        finance_wedge=a.finance_wedge,
        csm_release=-a.csm_release,
        csm_closing=a.csm_closing,
        loss_recovery_opening=a.loss_recovery_opening,
        loss_recovery_recognised=a.loss_recovery_recognised,
        loss_recovery_reversed=-a.loss_recovery_reversed,
        loss_recovery_closing=a.loss_recovery_closing,
    )


@reconcile.register
def _(aggregate: PAASettlementAggregate) -> PAASettlementReconciliation:
    """The paragraph-55(b) PAA settlement table of an aggregate -- identical to
    reconciling the per-MP movement; the revenue / claims-paid /
    loss-component-reversed rows are display-negated here, never in the
    aggregate."""
    a = aggregate
    return PAASettlementReconciliation(
        period_months=a.period_months,
        revenue_basis=a.revenue_basis,
        lrc_opening=a.lrc_opening,
        premiums=a.premiums,
        revenue=-a.revenue,
        lrc_experience=a.lrc_experience,
        lrc_closing=a.lrc_closing,
        loss_component_opening=a.loss_component_opening,
        loss_component_recognised=a.loss_component_recognised,
        loss_component_reversed=-a.loss_component_reversed,
        loss_component_closing=a.loss_component_closing,
        lic_opening=a.lic_opening,
        claims_incurred=a.claims_incurred,
        lic_finance=a.lic_finance,
        claims_paid=-a.claims_paid,
        lic_closing=a.lic_closing,
        claims_experience=a.claims_experience,
        expense_experience=a.expense_experience,
    )


@reconcile.register
def _(aggregate: VFASettlementAggregate) -> VFASettlementReconciliation:
    """The paragraph-45 settlement table of an aggregate -- identical to
    reconciling the per-MP movement (the oracle identity); the display
    negation of the run-off rows happens here, never in the aggregate."""
    a = aggregate
    return VFASettlementReconciliation(
        period_months=a.period_months,
        bel_opening=a.bel_opening,
        bel_interest=a.bel_interest,
        bel_release=-a.bel_release,
        bel_experience=a.bel_experience,
        bel_closing=a.bel_closing,
        ra_opening=a.ra_opening,
        ra_interest=a.ra_interest,
        ra_release=-a.ra_release,
        ra_experience=a.ra_experience,
        ra_closing=a.ra_closing,
        csm_opening=a.csm_opening,
        csm_accretion=a.csm_accretion,
        csm_fv_share=a.csm_fv_share,
        csm_future_service=a.csm_future_service,
        csm_premium_experience=a.csm_premium_experience,
        premium_experience_revenue=a.premium_experience_revenue,
        csm_investment_experience=a.csm_investment_experience,
        claims_experience=a.claims_experience,
        expense_experience=a.expense_experience,
        loss_component_finance=a.loss_component_finance,
        loss_component_amortised=-a.loss_component_amortised,
        loss_component_reversed=-a.loss_component_reversed,
        loss_component_recognised=a.loss_component_recognised,
        csm_release=-a.csm_release,
        csm_closing=a.csm_closing,
        loss_component_opening=a.loss_component_opening,
        loss_component_closing=a.loss_component_closing,
        lic_opening=a.lic_opening,
        claims_incurred=a.claims_incurred,
        lic_finance=a.lic_finance,
        claims_paid=-a.claims_paid,
        lic_closing=a.lic_closing,
    )
