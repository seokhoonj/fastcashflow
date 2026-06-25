"""Period-close roll-forward -- the IFRS 17 analysis of change.

A reporting period's movement bridges the opening insurance contract
liability to the closing one and decomposes the change into its drivers --
the analysis of change (AoC). This is the step from a measurement
calculator towards a reporting engine.

``roll_forward`` slices a GMM :class:`~fastcashflow.gmm.Measurement` into
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

from typing import ClassVar

from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from fastcashflow._measurement.model import GMM, VFA, PAA, REINSURANCE, model_tag
from fastcashflow._typing import FloatArray
from fastcashflow.curves import forward_rates
from fastcashflow._measurement.gmm import _require_full
from fastcashflow._measurement.basis import _require_inception
from fastcashflow.io import write_measurement, _write_measurement_columns
from fastcashflow.numerics import _csm_roll
from fastcashflow._measurement.paa import _require_full as _require_full_paa
from fastcashflow._measurement.vfa import (
    CSM_BASIS_PARAGRAPH_45, _CSM_TO_MEASUREMENT_BASIS)
from fastcashflow._measurement.vfa import _require_settlement_csm
from fastcashflow._measurement import gmm as _gmm
from fastcashflow._measurement import paa as _paa
from fastcashflow._measurement import vfa as _vfa
from fastcashflow._measurement import reinsurance as _reinsurance


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
        f"roll_forward does not handle {model_tag(measurement)}"
    )


def _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at):
    if any(opt is not None for opt in
           (revised, revised_at, actual_inforce, experience_at)):
        raise ValueError(
            "the revision and experience options apply to a GMM "
            "measurement only"
        )


@roll_forward.register
def _(measurement: _paa.Measurement, period_months: int = 12, *,
      revised=None, revised_at=None, actual_inforce=None, experience_at=None):
    _require_inception(measurement, "roll_forward()")
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at)
    return _roll_forward_paa(measurement, period_months)


@roll_forward.register
def _(measurement: _vfa.Measurement, period_months: int = 12, *,
      revised=None, revised_at=None, actual_inforce=None, experience_at=None):
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _require_settlement_csm(measurement, "roll_forward")
    _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at)
    return _roll_forward_vfa(measurement, period_months)


@roll_forward.register
def _(measurement: _reinsurance.Measurement, period_months: int = 12, *,
      revised=None, revised_at=None, actual_inforce=None, experience_at=None):
    _require_inception(measurement, "roll_forward()")
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _reject_gmm_only_opts(revised, revised_at, actual_inforce, experience_at)
    return _roll_forward_reinsurance(measurement, period_months)


@roll_forward.register
def _(
    measurement: _gmm.Measurement,
    period_months: int = 12,
    *,
    revised: _gmm.Measurement | None = None,
    revised_at: int | None = None,
    actual_inforce: FloatArray | None = None,
    experience_at: int | None = None,
) -> list[_gmm.PeriodMovement]:
    _require_inception(measurement, "roll_forward()")
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    _require_full(measurement, "roll_forward")
    # A universal-life account book needs no guard here: this roll-forward reads
    # only the account-netted bel_path / ra_path / csm_path and the in-force
    # count, never the raw benefit cash flows, so the BEL / RA / CSM waterfall
    # telescopes correctly (the account was netted once at measurement).
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

    discount_factor_bom = measurement.discount_factor_bom
    # discount_factor_bom is (n_time+1,) for a single basis, or (n_mp, n_time+1) for
    # a segmented (multi-basis) measurement; the last axis is time either way,
    # so the rate is (n_time,) or (n_mp, n_time) accordingly.
    discount_monthly = forward_rates(discount_factor_bom)
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

    movements: list[_gmm.PeriodMovement] = []
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
        movements.append(_gmm.PeriodMovement(
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
    measurement: _gmm.Measurement, period_months: int, actual_inforce: FloatArray
) -> list[_gmm.PeriodMovement]:
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
    discount_factor_bom = measurement.discount_factor_bom
    discount_monthly = forward_rates(discount_factor_bom)

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
    movements: list[_gmm.PeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_ex, ra_ex, csm_ex, loss = exp_lines.get(a, (zero, zero, zero, zero))
        bel_interest = ((bel[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
                        + bel_ex * discount_monthly[..., a])
        ra_interest = ((ra[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
                       + ra_ex * discount_monthly[..., a])
        movements.append(_gmm.PeriodMovement(
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
    measurement: _paa.Measurement, period_months: int
) -> list[_paa.PeriodMovement]:
    """Slice a PAA measurement into LRC, loss-component and LIC movements."""
    _require_full_paa(measurement, "roll_forward")
    lrc = measurement.lrc_path
    lic_path = measurement.lic_path
    premium_cf = measurement.cashflows.premium_cf
    revenue = measurement.revenue
    incurred = measurement.cashflows.mortality_cf + measurement.cashflows.morbidity_cf
    loss_component = measurement.loss_component
    n_time = lrc.shape[1] - 1
    total_revenue = revenue.sum(axis=1)
    safe_revenue = np.where(total_revenue > 0.0, total_revenue, 1.0)
    movements: list[_paa.PeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        period_incurred = incurred[:, a:b].sum(axis=1)
        # the loss component runs off in proportion to insurance revenue
        loss_open = loss_component * revenue[:, a:].sum(axis=1) / safe_revenue
        loss_close = loss_component * revenue[:, b:].sum(axis=1) / safe_revenue
        movements.append(_paa.PeriodMovement(
            month_start=a,
            month_end=b,
            lrc_opening=lrc[:, a],
            premiums=premium_cf[:, a:b].sum(axis=1),
            revenue=revenue[:, a:b].sum(axis=1),
            lrc_closing=lrc[:, b],
            loss_component_opening=loss_open,
            loss_component_release=loss_open - loss_close,
            loss_component_closing=loss_close,
            lic_opening=lic_path[:, a],
            claims_incurred=period_incurred,
            claims_paid=period_incurred - (lic_path[:, b] - lic_path[:, a]),
            lic_closing=lic_path[:, b],
        ))
    return movements


def _roll_forward_vfa(
    measurement: _vfa.Measurement, period_months: int
) -> list[_vfa.PeriodMovement]:
    """Slice a VFA measurement into BEL, RA and CSM movements."""
    _require_full(measurement, "roll_forward")
    bel, ra, csm = measurement.bel_path, measurement.ra_path, measurement.csm_path
    csm_accretion = measurement.csm_accretion
    csm_release = measurement.csm_release
    n_time = csm.shape[1] - 1
    discount_factor_bom = measurement.discount_factor_bom
    # discount_factor_bom is (n_time+1,) for a single basis, or (n_mp, n_time+1) for a
    # segmented (portfolio-stitched) measurement; the last axis is time either
    # way, so discount_monthly is (n_time,) or (n_mp, n_time). The trailing-axis
    # slice serves both -- a bare [a:b] would slice the model-point axis on the
    # 2-D curve.
    discount_monthly = forward_rates(discount_factor_bom)
    movements: list[_vfa.PeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_interest = (bel[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
        ra_interest = (ra[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
        movements.append(_vfa.PeriodMovement(
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
    measurement: _reinsurance.Measurement, period_months: int
) -> list[_reinsurance.PeriodMovement]:
    """Slice a reinsurance-held measurement into BEL, RA and CSM movements.

    The reinsurance counterpart of :func:`_roll_forward_vfa`: BEL / RA unwind at
    the discount rate and the CSM accretes and releases over coverage units,
    with no loss component (Sec. 65)."""
    _require_full(measurement, "roll_forward")
    bel, ra, csm = measurement.bel_path, measurement.ra_path, measurement.csm_path
    csm_accretion = measurement.csm_accretion
    csm_release = measurement.csm_release
    n_time = csm.shape[1] - 1
    discount_monthly = forward_rates(measurement.discount_factor_bom)
    movements: list[_reinsurance.PeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_interest = (bel[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
        ra_interest = (ra[:, a:b] * discount_monthly[..., a:b]).sum(axis=1)
        movements.append(_reinsurance.PeriodMovement(
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


def _reconcile_paa(
    movements: list[_paa.PeriodMovement],
) -> list[_paa.Reconciliation]:
    """Aggregate PAA period movements into portfolio-total reconciliations."""
    return [
        _paa.Reconciliation(
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


def _reconcile_vfa(
    movements: list[_vfa.PeriodMovement],
) -> list[_vfa.Reconciliation]:
    """Aggregate VFA period movements into portfolio-total reconciliations."""
    return [
        _vfa.Reconciliation(
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


# Settlement reconciliation display / disclosure block specs -- the single
# source for each settlement family's line spine, shared by the __str__ methods
# below and disclosure.py's reconciliation_to_frame / line_metadata (which
# imports them). Each line: (display name, reconciliation field, IFRS 17
# paragraph, is P&L memo). loss_component_reversed / recognised legitimately
# appear in BOTH the CSM block (where they enter the CSM) and the Loss component
# block (where they run it off).

# VFA settlement reconciliation -- the paragraph-45 CSM (fair-value share +
# future service, no finance wedge) and an account-value-linked LIC.

# Reinsurance-held settlement reconciliation -- no loss component (paragraph 65,
# a reinsurance contract held cannot be onerous); a loss-RECOVERY component
# (66A-66B) instead, and no LIC block.

# PAA settlement reconciliation -- an LRC (unearned premium) roll, no BEL/RA/CSM.


def _reconcile_vfa_settlement(
    movements: list[_vfa.SettlementMovement],
) -> list[_vfa.SettlementReconciliation]:
    """Aggregate paragraph-45 settlement movements into portfolio totals.

    Release and loss-component-reversed rows are stored negative (the
    reconciliation display convention), so opening plus every row equals
    closing in each block. Note the CSM block reads the same
    ``loss_component_reversed`` row: a favourable change reverses the loss
    component *instead of* crediting the CSM, so the row subtracts there
    too.
    """
    return [
        _vfa.SettlementReconciliation(
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


def _reconcile_reinsurance(
    movements: list[_reinsurance.PeriodMovement],
) -> list[_reinsurance.Reconciliation]:
    """Aggregate reinsurance period movements into portfolio-total reconciliations."""
    return [
        _reinsurance.Reconciliation(
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


def _reconcile_gmm_settlement(
    movements: list[_gmm.SettlementMovement],
) -> list[_gmm.SettlementReconciliation]:
    """Aggregate paragraph-44 settlement movements into portfolio totals."""
    return [
        _gmm.SettlementReconciliation(
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


def _reconcile_reinsurance_settlement(
    movements: list[_reinsurance.SettlementMovement],
) -> list[_reinsurance.SettlementReconciliation]:
    """Aggregate paragraph-66 reinsurance settlement movements into totals."""
    return [
        _reinsurance.SettlementReconciliation(
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


def _reconcile_paa_settlement(
    movements: list[_paa.SettlementMovement],
) -> list[_paa.SettlementReconciliation]:
    """Aggregate paragraph-55(b) settlement movements into portfolio totals."""
    return [
        _paa.SettlementReconciliation(
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
def _(movement: _paa.SettlementMovement, path, *, ids=None):
    n = movement.lrc_closing.shape[0]
    cols = {name: getattr(movement, name) for name in _paa._PAA_SETTLEMENT_LINES}
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
def _(movement: _gmm.SettlementMovement, path, *, ids=None):
    n = movement.bel_closing.shape[0]
    cols = {name: getattr(movement, name) for name in _gmm._GMM_SETTLEMENT_LINES}
    # Scalar (shared) or per-row (cohort-aware, Sec. B72(b)) locked-in rate:
    # broadcast handles both, so each row's own rate rides onto the part and
    # seeds the next period's settle from disk.
    cols["lock_in_rate"] = np.broadcast_to(
        np.asarray(movement.lock_in_rate, dtype=np.float64), (n,))
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
def _(movement: _reinsurance.SettlementMovement, path, *, ids=None):
    n = movement.bel_closing.shape[0]
    cols = {name: getattr(movement, name) for name in _reinsurance._REINSURANCE_SETTLEMENT_LINES}
    cols["lock_in_rate"] = np.full(n, movement.lock_in_rate)
    cols["measurement_basis"] = [movement.measurement_basis] * n
    if movement.model_points is not None:
        cols["elapsed_months"] = np.asarray(
            movement.model_points.elapsed_months, dtype=np.int64)
        cols["count"] = np.asarray(
            movement.model_points.count, dtype=np.float64)
    _write_measurement_columns(cols, path, ids)


@write_measurement.register
def _(movement: _vfa.SettlementMovement, path, *, ids=None):
    n = movement.bel_closing.shape[0]
    cols = {name: getattr(movement, name) for name in _vfa._VFA_SETTLEMENT_LINES}
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
    movements: (list[_gmm.PeriodMovement] | list[_paa.PeriodMovement]
                | list[_vfa.PeriodMovement]),
) -> list[_gmm.Reconciliation] | list[_paa.Reconciliation] | list[_vfa.Reconciliation]:
    """Aggregate period movements into IFRS 17 reconciliation tables.

    Each :class:`_gmm.PeriodMovement` -- per model point -- becomes one
    portfolio-total :class:`_gmm.Reconciliation` in the layout of IFRS 17
    paragraph 101. Run-off rows are shown negative, so opening plus every
    row equals closing. A list of :class:`_paa.PeriodMovement` or
    :class:`_vfa.PeriodMovement` is reconciled instead into the PAA
    liability-for-remaining-coverage or VFA contractual-service-margin
    tables.

    The base implementation takes a list of movements (dispatch falls through
    to it for any list); a mixed-portfolio
    :class:`~fastcashflow.portfolio.PortfolioMovements` registers its own arm
    (returning a :class:`~fastcashflow.portfolio.PortfolioReconciliation`).
    """
    if movements and isinstance(movements[0], _paa.PeriodMovement):
        return _reconcile_paa(movements)
    if movements and isinstance(movements[0], _vfa.PeriodMovement):
        return _reconcile_vfa(movements)
    if movements and isinstance(movements[0], _vfa.SettlementMovement):
        return _reconcile_vfa_settlement(movements)
    if movements and isinstance(movements[0], _gmm.SettlementMovement):
        return _reconcile_gmm_settlement(movements)
    if movements and isinstance(movements[0], _paa.SettlementMovement):
        return _reconcile_paa_settlement(movements)
    if movements and isinstance(movements[0], _reinsurance.SettlementMovement):
        return _reconcile_reinsurance_settlement(movements)
    if movements and isinstance(movements[0], _reinsurance.PeriodMovement):
        return _reconcile_reinsurance(movements)
    out: list[_gmm.Reconciliation] = []
    for m in movements:
        out.append(_gmm.Reconciliation(
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






@reconcile.register
def _(aggregate: _gmm.SettlementAggregate) -> _gmm.SettlementReconciliation:
    """The paragraph-44 settlement table of an aggregate -- identical to
    reconciling the per-MP movement (the oracle identity); the display
    negation of the run-off rows happens here, never in the aggregate."""
    a = aggregate
    return _gmm.SettlementReconciliation(
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
def _(aggregate: _reinsurance.SettlementAggregate
      ) -> _reinsurance.SettlementReconciliation:
    """The paragraph-66 reinsurance settlement table of an aggregate --
    identical to reconciling the per-MP movement; run-off rows display-negated
    here, never in the aggregate."""
    a = aggregate
    return _reinsurance.SettlementReconciliation(
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
def _(aggregate: _paa.SettlementAggregate) -> _paa.SettlementReconciliation:
    """The paragraph-55(b) PAA settlement table of an aggregate -- identical to
    reconciling the per-MP movement; the revenue / claims-paid /
    loss-component-reversed rows are display-negated here, never in the
    aggregate."""
    a = aggregate
    return _paa.SettlementReconciliation(
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
def _(aggregate: _vfa.SettlementAggregate) -> _vfa.SettlementReconciliation:
    """The paragraph-45 settlement table of an aggregate -- identical to
    reconciling the per-MP movement (the oracle identity); the display
    negation of the run-off rows happens here, never in the aggregate."""
    a = aggregate
    return _vfa.SettlementReconciliation(
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
