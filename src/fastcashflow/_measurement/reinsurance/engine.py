"""IFRS 17 reinsurance contracts held -- a quota-share treaty.

A cedant buys reinsurance to transfer risk. This module measures a
proportional (quota-share) reinsurance contract held over a direct
portfolio: the cedant cedes a fixed fraction of its claims and pays the
same fraction of its premiums to the reinsurer.

IFRS 17 measures reinsurance held with the general model but with two
modifications (paragraphs 60-70):

* The risk adjustment is the amount of risk *transferred* to the reinsurer
  (paragraph 64) -- here, the margin on the ceded claims.
* There is no unearned profit; the CSM is instead the net cost or net gain
  of buying the cover (paragraph 65). So the CSM may be negative -- a net
  cost is deferred and amortised, not expensed -- and there is no loss
  component.

v1 scope: a single quota-share cession rate over the portfolio, with no
ceding commission. The reinsurer's non-performance risk and the
loss-recovery component (for onerous underlying contracts) are left for
later.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow._measurement.basis import (
    MEASUREMENT_BASIS_SETTLEMENT_CARRY, _inforce_marker_columns)
from fastcashflow.basis import Basis, _single_basis
from fastcashflow.curves import (
    discount_factors, discount_monthly_curve, forward_rates)
from fastcashflow.numerics import _csm_kernel, _norm_ppf
from fastcashflow.model_points import InforceState, ModelPoints
from fastcashflow.projection import project_cashflows
from fastcashflow.io import (
    write_measurement, _write_measurement_columns, _stream_policies_coverages)
# In-force helpers shared with the GMM path (engine does not import
# _reinsurance, so this top-level import is cycle-free -- same pattern as _paa).
from fastcashflow._measurement.inforce import _reconcile_state
from fastcashflow._measurement.reinsurance.results import (
    Measurement, Aggregate, InforceAggregate, SettlementMovement,
    SettlementAggregate, Treaty, _REINSURANCE_SETTLEMENT_LINES)


@write_measurement.register
def _(measurement: Measurement, path, *, ids=None):
    cols = {"bel": measurement.bel, "ra": measurement.ra,
            "csm": measurement.csm}
    # In-force output gets marker columns (see _measurement.basis).
    cols.update(_inforce_marker_columns(measurement, measurement.bel.shape[0]))
    _write_measurement_columns(cols, path, ids)


def _resolve_recovery_pct(recovery_percentage, treaty) -> float:
    """The percentage of claims expected to recover (B119D / B95B). An explicit
    ``recovery_percentage`` wins; otherwise the proportional treaty cession
    (the claim cession a QuotaShare applies). A treaty without a single claim
    cession (excess-of-loss, limits) must pass ``recovery_percentage``."""
    if recovery_percentage is not None:
        pct = float(recovery_percentage)
    else:
        cession = getattr(treaty, "cession", None)
        if cession is None:
            raise ValueError(
                "a loss-recovery component needs the claim recovery percentage "
                "(B119D); this treaty has no single `cession`, so pass "
                "recovery_percentage explicitly.")
        pct = float(cession)
    if not np.isfinite(pct) or not 0.0 <= pct <= 1.0:
        raise ValueError(f"recovery_percentage must be in [0, 1], got {pct}")
    return pct


def measure(
    model_points: ModelPoints,
    basis: Basis,
    *,
    treaty: Treaty,
    underlying_loss_component: FloatArray | None = None,
    recovery_percentage: float | None = None,
    full: bool = True,
) -> Measurement:
    """Measure a reinsurance contract held over a direct portfolio.

    ``treaty`` describes how the cover cedes the direct cash flows -- e.g.
    :class:`QuotaShare(cession=0.5)`. The BEL is the present value of
    reinsurance premiums less recoveries; the RA is the margin on the ceded
    claims (the risk transferred). The CSM is ``-(BEL - RA)`` -- the net cost
    or gain of the cover -- and may be negative; it is accreted and released
    by coverage units like a direct contract's CSM, but with no loss
    component (paragraph 65).
    ``basis`` must resolve to a single :class:`Basis`; multi-segment routers are not
    accepted. ``full=False`` returns only the headline BEL / RA / CSM and leaves
    all trajectory and cash-flow fields ``None``.
    """
    basis = _single_basis(basis, entry="measure")
    proj = project_cashflows(model_points, basis)
    if proj.account is not None:
        # Cede the net amount at risk, not the gross account death benefit -- the
        # account-value part is the policyholder's deposit, not reinsured risk.
        proj = replace(proj, mortality_cf=proj.mortality_cf
                       - proj.deaths * proj.account.av_mid)
    discount_factor_bom, discount_factor_mid = discount_factors(basis, proj.n_time)

    ceded_mortality, ceded_morbidity, reinsurance_premium = treaty.cede(proj)
    recovery = ceded_mortality + ceded_morbidity

    pv_recovery = (recovery * discount_factor_mid).sum(axis=1)
    pv_reinsurance_premium = (reinsurance_premium * discount_factor_bom[:-1]).sum(axis=1)
    bel = pv_reinsurance_premium - pv_recovery

    # RA -- the risk transferred, i.e. the margin on the ceded claims.
    z = _norm_ppf(basis.ra_confidence)
    pv_ceded_mortality = (ceded_mortality * discount_factor_mid).sum(axis=1)
    pv_ceded_morbidity = (ceded_morbidity * discount_factor_mid).sum(axis=1)
    ra = z * (basis.mortality_cv * pv_ceded_mortality
              + basis.morbidity_cv * pv_ceded_morbidity)

    # CSM -- the net cost or gain of the cover. No loss component: a net cost
    # is a negative CSM, deferred and amortised over the coverage.
    csm0 = -(bel - ra)

    # 66A/66B loss recovery: when the cover is held over an ONEROUS underlying
    # group, the matching recovery is recognised as immediate income and the
    # reinsurance CSM is reduced by it -- csm_after = csm0 - loss_recovery
    # (IASB AP2C Dec 2019, Examples 1-3). loss_recovery = underlying loss x the
    # claim recovery % (B95B / B119D). No floor (paragraph 65): the CSM may go
    # negative. Absent the input => zero (byte-identical). B119C (timing: the
    # cover entered before/at the onerous underlying) is the caller's
    # responsibility -- supply underlying_loss_component only when it holds.
    n_mp = bel.shape[0]
    if underlying_loss_component is not None:
        rec_pct = _resolve_recovery_pct(recovery_percentage, treaty)
        loss_recovery_component = np.maximum(
            0.0, np.asarray(underlying_loss_component, dtype=np.float64)) * rec_pct
        loss_recovery_component = np.broadcast_to(
            loss_recovery_component, (n_mp,)).astype(np.float64, copy=True)
    else:
        loss_recovery_component = np.zeros(n_mp)
    csm0 = csm0 - loss_recovery_component
    if not full:
        return Measurement(
            bel=bel, ra=ra, csm=csm0,
            loss_recovery_component=loss_recovery_component,
            model_points=model_points)

    bel_path = (_pv_path(reinsurance_premium * discount_factor_bom[:-1], discount_factor_bom)
                - _pv_path(recovery * discount_factor_mid, discount_factor_bom))
    ra_path = z * (
        basis.mortality_cv * _pv_path(ceded_mortality * discount_factor_mid, discount_factor_bom)
        + basis.morbidity_cv * _pv_path(ceded_morbidity * discount_factor_mid, discount_factor_bom)
    )
    csm, csm_accretion, csm_release = _csm_kernel(
        csm0, proj.inforce,
        discount_monthly_curve(basis, proj.n_time),
        basis.coverage_unit_discount,
    )

    return Measurement(
        bel=bel,
        ra=ra,
        csm=csm[:, 0],
        bel_path=bel_path,
        ra_path=ra_path,
        csm_path=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        loss_recovery_component=loss_recovery_component,
        recovery=recovery,
        reinsurance_premium=reinsurance_premium,
        cashflows=proj,
        discount_factor_bom=discount_factor_bom,
        model_points=model_points,
    )


def measure_aggregate(
    model_points: ModelPoints,
    basis: Basis,
    *,
    treaty: Treaty,
    chunk_size: int = 200_000,
) -> Aggregate:
    """Portfolio-aggregate reinsurance-held measurement in bounded memory.

    The reinsurance counterpart of :func:`~fastcashflow.gmm.measure_aggregate`:
    BEL / RA / CSM and the ceded cash flows are additive across contracts, so the
    ceded book's run-off is the per-model-point trajectories summed over the
    model-point axis. Runs :func:`measure` over row-blocks of
    ``chunk_size`` model points and accumulates only the ``(n_time+1,)`` /
    ``(n_time,)`` sums, so peak memory is ``O(chunk_size x n_time)`` regardless of
    ``n_mp``. Returns a :class:`Aggregate` (scalar totals + aggregate
    ``csm_path`` / ``recovery`` / ``reinsurance_premium``) -- a scalable sum of
    the measured results, not a group remeasurement. ``basis`` is a single
    :class:`Basis`, as for :func:`measure`.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    n_mp = model_points.n_mp
    # A chunk projects only to its own boundary.max(); its (shorter) aggregate
    # path adds into the leading slice of the global one -- a contract carries
    # nothing past its coverage period.
    n_time = int(np.asarray(model_points.contract_boundary_months).max())
    bel_path = np.zeros(n_time + 1)
    ra_path = np.zeros(n_time + 1)
    csm_path = np.zeros(n_time + 1)
    recovery = np.zeros(n_time)
    reinsurance_premium = np.zeros(n_time)
    bel = ra = csm = 0.0
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        m = measure(
            model_points.subset(idx), basis, treaty=treaty, full=True)
        nt1 = m.csm_path.shape[1]
        nt = m.recovery.shape[1]
        bel_path[:nt1] += m.bel_path.sum(axis=0)
        ra_path[:nt1] += m.ra_path.sum(axis=0)
        csm_path[:nt1] += m.csm_path.sum(axis=0)
        recovery[:nt] += m.recovery.sum(axis=0)
        reinsurance_premium[:nt] += m.reinsurance_premium.sum(axis=0)
        bel += float(m.bel.sum())
        ra += float(m.ra.sum())
        csm += float(m.csm.sum())
    return Aggregate(
        bel=bel, ra=ra, csm=csm, bel_path=bel_path, ra_path=ra_path,
        csm_path=csm_path, recovery=recovery,
        reinsurance_premium=reinsurance_premium)


def measure_stream(
    input_path,
    output_dir,
    basis: Basis,
    *,
    treaty: Treaty,
    coverages=None,
    calculation_methods=None,
    chunk_size: int = 20_000_000,
    id_column: str | None = None,
    validate_unique_mp_id: bool = True,
) -> int:
    """Stream a reinsurance-held valuation through a parquet file, chunk by chunk.

    The reinsurance counterpart of :func:`~fastcashflow.gmm.measure_stream`:
    reads the direct policies + coverages parquet in ``chunk_size`` blocks,
    cedes and measures each with ``treaty``, and writes per-chunk
    ``part-NNNNN.parquet`` results (bel / ra / csm). Returns the number of model
    points processed. ``basis`` is a single :class:`Basis`.

    Marginal benefit note: streaming exists for portfolios too large to hold in
    memory (a GMM book of 1e8 rows). A ceded reinsurance book is almost always
    far smaller -- one treaty over a direct portfolio -- so in practice
    :func:`~fastcashflow.reinsurance.measure` or
    :func:`~fastcashflow.reinsurance.measure_aggregate` is enough and this adds
    little over them. It exists for API symmetry with the other models.
    """
    basis = _single_basis(basis, entry="reinsurance.measure_stream")
    return _stream_policies_coverages(
        input_path, output_dir, coverages=coverages,
        calculation_methods=calculation_methods, chunk_size=chunk_size,
        id_column=id_column, validate_unique_mp_id=validate_unique_mp_id,
        measure_fn=lambda mp: measure(
            mp, basis, treaty=treaty, full=False),
    )


def _pv_path(month_pv: FloatArray, discount_factor_bom: FloatArray) -> FloatArray:
    """PV-at-t trajectory of a stream whose per-month PV-to-inception is
    ``month_pv`` (shape ``(n_mp, n_time)``).

    Reverse-cumsum gives ``sum_{s>=t}`` of the inception-discounted flows
    (the PV to inception of everything from month ``t`` on); dividing by
    ``discount_factor_bom[t]`` re-anchors it to time ``t``. Column ``n_time`` (no
    remaining flow) is 0. Column 0 reproduces the inception PV.
    """
    rev = np.cumsum(month_pv[:, ::-1], axis=1)[:, ::-1]          # sum_{s>=t}
    rev = np.concatenate([rev, np.zeros((rev.shape[0], 1))], axis=1)  # +t=n_time
    return rev / discount_factor_bom[None, :]


def measure_inforce(
    model_points: ModelPoints,
    state: InforceState,
    basis: Basis,
    *,
    treaty: Treaty,
    period_months: int | None = None,
    full: bool = True,
) -> Measurement:
    """In-force subsequent measurement of a reinsurance contract held (IFRS 17
    paragraph 44, modified by paragraphs 60-70) at the valuation date.

    The reinsurance counterpart of :func:`~fastcashflow.gmm.measure_inforce`.
    Each model point is valued at its ``elapsed_months`` duration: the BEL
    (PV of remaining reinsurance premiums less recoveries) and the RA (risk
    still transferred) are the inception projection sliced at the valuation
    date and re-based by ``count / inforce[elapsed]``, and the prior period's
    closing reinsurance CSM (``state.prior_csm``) is carried forward -- accreted
    at ``state.lock_in_rate`` and released over the coverage units across
    ``period_months`` (default 12). The CSM is the net cost or gain of the
    cover and may be negative; there is no loss component (paragraph 65), so the
    onerous unlocking deferred in ``gmm.measure_inforce`` does not arise here.

    ``state`` (an :class:`~fastcashflow.InforceState`) supplies the period-close
    ``elapsed_months`` / ``count`` (reconciled onto ``model_points`` by
    :func:`~fastcashflow.apply_inforce_state`) plus ``prior_csm`` /
    ``lock_in_rate``. ``treaty`` is the same cession as the new-business
    :func:`measure`.
    ``basis`` must resolve to a single :class:`Basis`; multi-segment routers are not
    accepted. ``full=False`` returns only the as-of headline BEL / RA / CSM and
    leaves all trajectory and cash-flow fields ``None``.
    """
    warnings.warn(
        "reinsurance.measure_inforce is a carry bridge superseded by "
        "reinsurance.settle (the paragraph-66 subsequent measurement): it "
        "rolls the prior CSM forward without the future-service unlocking. "
        "Use reinsurance.settle.", DeprecationWarning, stacklevel=2)
    basis = _single_basis(basis, entry="reinsurance.measure_inforce")
    state = _reconcile_state(model_points, state)
    proj = project_cashflows(model_points, basis)
    if proj.account is not None:
        proj = replace(proj, mortality_cf=proj.mortality_cf
                       - proj.deaths * proj.account.av_mid)
    n_time = proj.n_time
    n_mp = proj.inforce.shape[0]
    discount_factor_bom, discount_factor_mid = discount_factors(basis, n_time)

    ceded_mortality, ceded_morbidity, reinsurance_premium = treaty.cede(proj)
    recovery = ceded_mortality + ceded_morbidity

    # BEL / RA trajectories (PV at each t of the remaining ceded flows), so the
    # valuation-date slice is the PV of the *future* cash flows. Premiums are
    # bom-timed, recoveries / ceded claims mid-timed -- same convention as the
    # inception measure, so column 0 reproduces measure's headline.
    z = _norm_ppf(basis.ra_confidence)
    bel_path = (_pv_path(reinsurance_premium * discount_factor_bom[:-1], discount_factor_bom)
                - _pv_path(recovery * discount_factor_mid, discount_factor_bom))
    ra_path = z * (
        basis.mortality_cv * _pv_path(ceded_mortality * discount_factor_mid, discount_factor_bom)
        + basis.morbidity_cv * _pv_path(ceded_morbidity * discount_factor_mid, discount_factor_bom)
    )

    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    runoff = em >= boundary
    if np.any(runoff):
        bad = int(np.argmax(runoff))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} >= "
            f"contract_boundary_months[{bad}]={int(boundary[bad])} (the paragraph 34 "
            "horizon; equal to term_months when no boundary cut); the contract "
            "has no remaining coverage at the valuation date. "
            "reinsurance.measure_inforce needs an as-of date strictly before the "
            "contract boundary.")
    rows = np.arange(n_mp)
    # Re-base the inception-run slice to the valuation-date count (exact for the
    # BEL / RA, which are linear in the in-force).
    inforce_em = proj.inforce[rows, em]
    count = np.asarray(model_points.count, dtype=np.float64)
    rescale = np.where(inforce_em > 0.0, count / np.where(inforce_em > 0.0, inforce_em, 1.0), 1.0)
    bel = bel_path[rows, em] * rescale
    ra = ra_path[rows, em] * rescale

    # CSM carry-forward (paragraph 44): roll the prior closing CSM one period over the
    # coverage units (the direct in-force, which the recoveries scale with) from
    # t = em - period_months to t = em. No loss-component floor -- a reinsurance
    # CSM may stay negative (a net cost deferred).
    prior_csm = np.asarray(state.prior_csm, dtype=np.float64)
    if prior_csm.shape != (n_mp,):
        raise ValueError(f"prior_csm must have shape ({n_mp},), got {prior_csm.shape}")
    period_months = int(period_months) if period_months is not None else 12
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    prior_t = em - period_months
    if np.any(prior_t < 0):
        bad = int(np.argmin(prior_t))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} < period_months={period_months}; "
            "the prior closing date precedes inception, which has no CSM to carry")
    max_len = proj.inforce.shape[1] - int(prior_t.min())
    src_cols = prior_t[:, None] + np.arange(max_len)[None, :]
    mask = src_cols < proj.inforce.shape[1]
    inforce_seg = np.ascontiguousarray(
        np.where(mask, proj.inforce[rows[:, None], np.where(mask, src_cols, 0)], 0.0))
    lock_in_monthly = (1.0 + float(state.lock_in_rate)) ** (1.0 / 12.0) - 1.0
    csm_traj, csm_accretion, csm_release = _csm_kernel(
        prior_csm, inforce_seg, np.full(max_len, lock_in_monthly),
        basis.coverage_unit_discount)
    csm = csm_traj[:, period_months]

    if not full:
        return Measurement(
            bel=bel, ra=ra, csm=csm, model_points=model_points,
            measurement_basis=MEASUREMENT_BASIS_SETTLEMENT_CARRY)

    return Measurement(
        bel=bel, ra=ra, csm=csm,
        measurement_basis=MEASUREMENT_BASIS_SETTLEMENT_CARRY,
        bel_path=bel_path,
        ra_path=ra_path,
        csm_path=csm_traj,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        recovery=recovery,
        reinsurance_premium=reinsurance_premium,
        cashflows=proj,
        discount_factor_bom=discount_factor_bom,
        model_points=model_points,
    )


def measure_inforce_aggregate(
    model_points: ModelPoints,
    state: InforceState,
    basis: Basis,
    *,
    treaty: Treaty,
    period_months: int | None = None,
    chunk_size: int = 200_000,
) -> InforceAggregate:
    """Portfolio-aggregate reinsurance-held in-force carry in bounded memory.

    The carry bridge for a ceded book's period close: runs
    :func:`measure_inforce` over row-blocks of ``chunk_size`` model
    points and accumulates only the headline BEL / RA / CSM, so peak memory is
    ``O(chunk_size x n_time)`` regardless of ``n_mp``. Returns a
    :class:`InforceAggregate` -- a scalable SUM of the measured
    per-model-point results, equal to them to machine precision, not a group
    remeasurement.

    It is a bridge, not a settlement (``measurement_basis == 'settlement_carry'``):
    the reinsurance leaf has no ``settle``, so the prior CSM is rolled one
    period (paragraph 44) without the paragraph 66 reinsurance-specific unlocking or a
    loss-recovery component, and the function is deprecated once
    ``reinsurance.settle`` lands. ``basis`` is a single :class:`Basis`, as for
    :func:`measure_inforce`.

    A zero-count row is REJECTED: this carry bridge cannot value a contract
    derecognized during the period (paragraph 76) -- that needs a settlement, which
    ``reinsurance.settle`` will provide. (``gmm.settle`` handles count=0 as
    normal derecognition; this bridge does not.)
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    warnings.warn(
        "reinsurance.measure_inforce_aggregate is a carry bridge superseded "
        "by reinsurance.settle_aggregate (the paragraph-66 subsequent "
        "measurement): it rolls the prior CSM forward without the "
        "future-service unlocking. Use reinsurance.settle_aggregate.",
        DeprecationWarning, stacklevel=2)
    # Align the period-close state onto the model points ONCE, before chunking,
    # so a shuffled state file cannot pair one contract's rows with another's
    # prior CSM after a chunk slice (measure_inforce re-reconciles
    # each chunk, which is then a no-op).
    state = _reconcile_state(model_points, state)
    count = np.asarray(model_points.count, dtype=np.float64)
    if np.any(count == 0.0):
        bad = int(np.argmin(count))
        raise ValueError(
            f"count[{bad}]=0 is a row derecognized during the period (paragraph 76); "
            "this carry bridge cannot value it. Use reinsurance.settle for a "
            "period close with derecognition, or drop the row before valuing.")
    period = 12 if period_months is None else int(period_months)
    n_mp = model_points.n_mp
    bel = ra = csm = 0.0
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        with warnings.catch_warnings():       # the inner per-MP bridge already warned above
            warnings.simplefilter("ignore", DeprecationWarning)
            m = measure_inforce(
                model_points.subset(idx), state.subset(idx), basis,
                treaty=treaty, period_months=period, full=False)
        bel += float(m.bel.sum())
        ra += float(m.ra.sum())
        csm += float(m.csm.sum())
    return InforceAggregate(
        bel=bel, ra=ra, csm=csm, period_months=period)


def settle(
    model_points: ModelPoints,
    state: InforceState,
    basis: Basis,
    *,
    treaty: Treaty,
    period_months: int | None = None,
    underlying_loss_opening: FloatArray | None = None,
    underlying_loss_closing: FloatArray | None = None,
    recovery_percentage: float | None = None,
) -> "SettlementMovement":
    """Paragraph-66 subsequent-measurement settlement of a reinsurance contract
    held (the reinsurance counterpart of :func:`~fastcashflow.gmm.settle`).

    The opening -> closing movement over one reporting period: BEL / RA
    re-measured at current rates, the CSM accreted at the locked-in rate
    (66(b)/B72(b)), adjusted for the future-service change measured at the
    locked-in rate (66(c)/B72(c) -- the current-rate gap is the
    ``finance_wedge``), and released once at the period end over coverage units
    (66(e)/B119). The ONE difference from ``gmm.settle``: a reinsurance contract
    held cannot be onerous (paragraph 65), so the CSM is NOT floored and there
    is no loss component -- the closing CSM may be negative (a net cost of
    cover). The BEL is PV(reinsurance premium) - PV(recovery), so the locked-in
    second leg re-prices the ceded cash flows at the locked-in rate.

    On-track experience makes every experience line zero and telescopes the
    closing CSM to the carry bridge (:func:`measure_inforce`)
    exactly. A row whose closing date reaches the contract boundary with
    ``count = 0`` is a final settlement (full B119 derecognition, paragraph 76).

    The loss-recovery component (paragraphs 66A-66B / B119F) is tracked when the
    cover is held over an onerous underlying group: pass
    ``underlying_loss_opening`` / ``underlying_loss_closing`` (the underlying
    group's loss component at the two dates, from the direct ``gmm.settle``) and,
    for a non-proportional treaty, ``recovery_percentage``. The four
    ``loss_recovery_*`` movement lines re-derive the component as the underlying
    loss x the claim recovery % and amortise it in lock-step with the underlying
    loss (the recovery reverses in P&L as the underlying runs off). The CSM is
    NOT re-adjusted here -- the 66A CSM effect (csm_after = csm0 - loss_recovery)
    is a one-time inception event in ``reinsurance.measure``. Absent the inputs
    => zero (byte-identical). B119C timing (the cover entered before/at the
    onerous underlying) is the caller's responsibility.
    """
    basis = _single_basis(basis, entry="reinsurance.settle")
    state = _reconcile_state(model_points, state)
    # The within-period experience inputs (actual_premium / actual_claims /
    # actual_expenses / actual_investment_component) are gmm.settle's B96-B97
    # lines; reinsurance.settle does not model them, so a state file reused from
    # a direct settle would SILENTLY drop them. Reject rather than ignore.
    dropped = [nm for nm in ("actual_premium", "actual_claims",
                             "actual_expenses", "actual_investment_component")
               if getattr(state, nm) is not None]
    if dropped:
        raise NotImplementedError(
            "reinsurance.settle does not model within-period experience; "
            f"state carries {dropped} which would be silently dropped. Clear "
            "them (they belong to a direct gmm.settle state).")
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period}")
    if state.prior_count is None:
        raise ValueError(
            "reinsurance.settle needs state.prior_count -- the in-force count "
            "at the opening date (the expected leg's scale and the B119 "
            "release denominator).")
    n_mp = model_points.n_mp
    prior_csm = np.asarray(state.prior_csm, dtype=np.float64)
    if prior_csm.shape != (n_mp,):
        raise ValueError(f"prior_csm must have shape ({n_mp},), got {prior_csm.shape}")
    # No floor / xor check: a reinsurance CSM may be negative (net cost) and
    # there is no loss component (paragraph 65).

    em_close = np.asarray(model_points.elapsed_months, dtype=np.int64)
    em_open = em_close - period
    if np.any(em_open < 0):
        bad = int(np.argmin(em_open))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em_close[bad])} < period_months={period}; "
            "the opening date precedes inception, which has no CSM to settle from.")
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    if np.any(em_open >= boundary):
        bad = int(np.argmax(em_open >= boundary))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em_close[bad])} - period_months={period} "
            f"is at or past the contract boundary ({int(boundary[bad])}); the "
            "opening date must lie strictly inside the coverage period.")
    count = np.asarray(model_points.count, dtype=np.float64)
    final = em_close >= boundary
    if np.any(final & (count > 0.0)):
        bad = int(np.argmax(final & (count > 0.0)))
        raise ValueError(
            f"row {bad} closes at or past the contract boundary with "
            f"count={count[bad]}; a final settlement needs a zero closing "
            "snapshot (full B119 derecognition).")

    unit = replace(model_points, count=np.ones(n_mp))
    m = measure(unit, basis, treaty=treaty)
    inforce = m.cashflows.inforce
    n_time = inforce.shape[1]
    rows = np.arange(n_mp)
    bel_path = m.bel_path
    ra_path = m.ra_path
    discount_factor_bom = m.discount_factor_bom

    # Locked-in BEL leg: re-price the ceded cash flows at the flat locked-in
    # rate (the reinsurance analogue of gmm.settle's locked-in kernel pass).
    lock = float(state.lock_in_rate)
    lock_m = (1.0 + lock) ** (1.0 / 12.0) - 1.0
    db_lock = (1.0 + lock_m) ** -np.arange(n_time + 1, dtype=np.float64)
    dm_lock = (1.0 + lock_m) ** -(np.arange(n_time, dtype=np.float64) + 0.5)
    bel_lock = (_pv_path(m.reinsurance_premium * db_lock[:-1], db_lock)
                - _pv_path(m.recovery * dm_lock, db_lock))

    surv_open = inforce[rows, em_open]
    k_exp = np.where(surv_open > 0.0,
                     np.asarray(state.prior_count, dtype=np.float64)
                     / np.where(surv_open > 0.0, surv_open, 1.0), 0.0)
    close_idx = np.minimum(em_close, n_time - 1)
    surv_close = np.where(final, 0.0, inforce[rows, close_idx])
    dead_unit = (~final) & (surv_close <= 0.0) & (count > 0.0)
    if np.any(dead_unit):
        bad = int(np.argmax(dead_unit))
        raise ValueError(
            f"row {bad}: the projection has no survivors at the closing date "
            f"but the observed count is {count[bad]}; reconcile the snapshot.")
    k_obs = np.where(surv_close > 0.0,
                     count / np.where(surv_close > 0.0, surv_close, 1.0), 0.0)
    live_close = np.where(final, 0.0, 1.0)
    em_c = np.minimum(em_close, n_time)
    discount_monthly = forward_rates(discount_factor_bom)
    cols = em_open[:, None] + np.arange(period)[None, :]
    col_ok = cols < n_time
    cols_safe = np.where(col_ok, cols, n_time - 1)

    def _block(path):
        opening = k_exp * path[rows, em_open]
        close_unit = path[rows, em_c] * live_close
        close_exp = k_exp * close_unit
        closing = k_obs * close_unit
        interest = k_exp * (path[rows[:, None], cols_safe]
                            * discount_monthly[cols_safe] * col_ok).sum(axis=1)
        release = opening + interest - close_exp
        experience = closing - close_exp
        return opening, interest, release, experience, closing

    bel_o, bel_i, bel_r, bel_e, bel_c = _block(bel_path)
    ra_o, ra_i, ra_r, ra_e, ra_c = _block(ra_path)
    delta_lock = (k_obs - k_exp) * bel_lock[rows, em_c] * live_close
    csm_experience_unlocking = -(delta_lock + ra_e)
    finance_wedge = -(bel_e - delta_lock)
    csm_accretion = prior_csm * ((1.0 + lock) ** (period / 12.0) - 1.0)
    # Paragraph 65: NO floor, NO loss component -- the net cost/gain rolls on.
    csm_after = prior_csm + csm_accretion + csm_experience_unlocking

    tail = np.zeros((n_mp, n_time + 1))
    tail[:, :n_time] = np.cumsum(inforce[:, ::-1], axis=1)[:, ::-1]
    cu_provided = k_exp * (tail[rows, em_open] - tail[rows, em_c])
    cu_future = k_obs * tail[rows, em_c]
    denom = cu_provided + cu_future
    frac = np.where(denom > 0.0,
                    cu_provided / np.where(denom > 0.0, denom, 1.0), 1.0)
    csm_release = csm_after * frac
    csm_closing = csm_after - csm_release

    # 66B/B119F loss-recovery component: a SEPARATE tracked balance on the asset
    # for remaining coverage (NOT a CSM adjustment -- the 66A CSM effect is a
    # one-time inception event in measure). Re-derived each period
    # as the underlying loss component x the claim recovery %, amortised in
    # lock-step with the underlying loss (B119F, paragraphs 50-52): as the
    # underlying loss runs off, the recovery reverses in P&L (excluded from the
    # premium allocation). Absent the underlying-loss inputs => zero
    # (byte-identical to the pre-feature settle).
    if (underlying_loss_opening is not None
            or underlying_loss_closing is not None):
        rec_pct = _resolve_recovery_pct(recovery_percentage, treaty)
        ul_open = (np.zeros(n_mp) if underlying_loss_opening is None
                   else np.maximum(0.0, np.broadcast_to(np.asarray(
                       underlying_loss_opening, dtype=np.float64), (n_mp,))))
        ul_close = (np.zeros(n_mp) if underlying_loss_closing is None
                    else np.maximum(0.0, np.broadcast_to(np.asarray(
                        underlying_loss_closing, dtype=np.float64), (n_mp,))))
        loss_recovery_opening = ul_open * rec_pct
        loss_recovery_closing = ul_close * rec_pct
        loss_recovery_recognised = np.maximum(
            0.0, loss_recovery_closing - loss_recovery_opening)
        loss_recovery_reversed = np.maximum(
            0.0, loss_recovery_opening - loss_recovery_closing)
    else:
        loss_recovery_opening = np.zeros(n_mp)
        loss_recovery_closing = np.zeros(n_mp)
        loss_recovery_recognised = np.zeros(n_mp)
        loss_recovery_reversed = np.zeros(n_mp)

    return SettlementMovement(
        bel_opening=bel_o, bel_interest=bel_i, bel_release=bel_r,
        bel_experience=bel_e, bel_closing=bel_c,
        ra_opening=ra_o, ra_interest=ra_i, ra_release=ra_r,
        ra_experience=ra_e, ra_closing=ra_c,
        csm_opening=prior_csm, csm_accretion=csm_accretion,
        csm_experience_unlocking=csm_experience_unlocking,
        finance_wedge=finance_wedge,
        csm_release=csm_release, csm_closing=csm_closing,
        loss_recovery_opening=loss_recovery_opening,
        loss_recovery_recognised=loss_recovery_recognised,
        loss_recovery_reversed=loss_recovery_reversed,
        loss_recovery_closing=loss_recovery_closing,
        coverage_units_provided=cu_provided, coverage_units_future=cu_future,
        period_months=period, lock_in_rate=lock,
        model_points=model_points)


def settle_aggregate(
    model_points: ModelPoints,
    state: InforceState,
    basis: Basis,
    *,
    treaty: Treaty,
    period_months: int | None = None,
    chunk_size: int = 200_000,
    underlying_loss_opening: FloatArray | None = None,
    underlying_loss_closing: FloatArray | None = None,
    recovery_percentage: float | None = None,
) -> "SettlementAggregate":
    """Portfolio-total paragraph-66 reinsurance settlement in bounded memory.

    Runs :func:`settle` over row blocks of ``chunk_size`` model
    points and accumulates only the scalar line totals (every settlement line
    is additive across contracts), combined with ``math.fsum`` so the total
    does not depend on the chunking. Replaces the carry-bridge aggregate
    :func:`measure_inforce_aggregate` with a true settlement.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period}")
    state = _reconcile_state(model_points, state)
    n_mp = model_points.n_mp

    # The underlying loss components ride per chunk (scalar or per-MP), so the
    # aggregate equals the per-MP settle sum even when the loss varies by row.
    def _slice(v, idx):
        if v is None:
            return None
        a = np.asarray(v, dtype=np.float64)
        return float(a) if a.ndim == 0 else a[idx]

    parts: dict[str, list[float]] = {n: [] for n in _REINSURANCE_SETTLEMENT_LINES}
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        mv = settle(
            model_points.subset(idx), state.subset(idx), basis, treaty=treaty,
            period_months=period,
            underlying_loss_opening=_slice(underlying_loss_opening, idx),
            underlying_loss_closing=_slice(underlying_loss_closing, idx),
            recovery_percentage=recovery_percentage)
        for name in _REINSURANCE_SETTLEMENT_LINES:
            parts[name].append(float(getattr(mv, name).sum()))
    return SettlementAggregate(
        period_months=period, lock_in_rate=float(state.lock_in_rate),
        **{name: math.fsum(vals) for name, vals in parts.items()})


def settle_stream(
    input_path,
    output_dir,
    basis: Basis,
    *,
    treaty: Treaty,
    coverages=None,
    calculation_methods=None,
    state_path=None,
    period_months: int | None = None,
    chunk_size: int = 200_000,
    id_column: str | None = None,
    validate_unique_mp_id: bool = True,
) -> int:
    """Stream a paragraph-66 reinsurance period close through a parquet file.

    The out-of-core variant of :func:`~fastcashflow.reinsurance.settle`: reads
    the direct policies + coverages parquet in ``chunk_size`` blocks, cedes
    with ``treaty``, settles each block, and writes the per-MP settlement
    movements (one ``part-NNNNN.parquet`` per chunk). Same one-combined-file /
    two-file (``state_path``) layouts as
    :func:`~fastcashflow.gmm.settle_stream`; the reinsurance state carries
    ``prior_csm`` (which may be negative -- a net cost) and ``prior_count``.
    Returns the number of model points processed. A ceded book is usually small
    enough for :func:`settle` / :func:`settle_aggregate`; this exists for API
    symmetry with the other models.
    """
    from fastcashflow.io import _settle_stream_driver, _coverages_build_mp
    basis = _single_basis(basis, entry="reinsurance.settle_stream")
    build_mp = _coverages_build_mp(coverages, calculation_methods,
                                   entry="reinsurance.settle_stream")
    return _settle_stream_driver(
        input_path, output_dir, state_path=state_path, chunk_size=chunk_size,
        id_column=id_column, validate_unique_mp_id=validate_unique_mp_id,
        build_mp=build_mp,
        settle_fn=lambda mp, st: settle(
            mp, st, basis, treaty=treaty, period_months=period_months),
        entry="reinsurance.settle_stream")
