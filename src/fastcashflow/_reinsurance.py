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

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.basis import Basis, _single_basis
from fastcashflow.curves import discount_factors, discount_monthly_curve
from fastcashflow.numerics import _csm_kernel, _norm_ppf
from fastcashflow.modelpoints import InforceState, ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows
from fastcashflow.io import (
    write_measurement, _write_measurement_columns, _stream_policies_coverages)
# In-force helpers shared with the GMM path (engine does not import
# _reinsurance, so this top-level import is cycle-free -- same pattern as _paa).
from fastcashflow.engine import _reconcile_state


@dataclass(frozen=True, slots=True, eq=False)
class ReinsuranceMeasurement:
    """Measurement of a reinsurance contract held.

    Headline ``bel``, ``ra`` and ``csm`` are ``(n_mp,)`` inception figures --
    ``bel`` is the present value of reinsurance premiums less recoveries (a
    net cost when positive), ``ra`` is the risk transferred, ``csm`` is the
    inception net cost or gain (may be negative). The trajectory fields are
    populated only on the full path; ``csm_path`` reconciles as
    ``csm_path[:, t+1] = csm_path[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    """

    # headline -- always present, shape (n_mp,)
    bel: FloatArray            # PV(reinsurance premiums) - PV(recoveries)
    ra: FloatArray             # risk transferred to the reinsurer
    csm: FloatArray            # inception net cost/gain
    # trajectory -- full only (None on the headline-only path)
    bel_path: FloatArray | None = None         # (n_mp, n_time+1)
    ra_path: FloatArray | None = None          # (n_mp, n_time+1)
    csm_path: FloatArray | None = None         # (n_mp, n_time+1) -- net cost/gain trajectory
    csm_accretion: FloatArray | None = None    # (n_mp, n_time)
    csm_release: FloatArray | None = None      # (n_mp, n_time)
    recovery: FloatArray | None = None         # (n_mp, n_time) -- recoveries received
    reinsurance_premium: FloatArray | None = None    # (n_mp, n_time) -- reinsurance premiums paid
    cashflows: "Cashflows | None" = None
    discount_bom: FloatArray | None = None     # (n_time+1,) -- for grouped CSM re-derivation
    model_points: "ModelPoints | None" = None  # stamped by measure_reinsurance, for group axes
    group_labels: "np.ndarray | None" = None   # per-group label on a grouped result
    group_sizes: IntArray | None = None     # model points per group, aligned with labels


@write_measurement.register
def _(measurement: ReinsuranceMeasurement, path, *, ids=None):
    _write_measurement_columns(
        {"bel": measurement.bel, "ra": measurement.ra, "csm": measurement.csm},
        path, ids)


@dataclass(frozen=True, slots=True, eq=False)
class ReinsuranceAggregate:
    """Portfolio-aggregate reinsurance-held trajectories -- the scalable view.

    BEL / RA / CSM are additive across contracts, so a large ceded book's
    reinsurance asset/liability run-off is its per-model-point trajectories
    summed over the model-point axis. Holds the scalar inception totals plus the
    ``(n_time+1,)`` aggregate ``csm_path`` and the ``(n_time,)`` aggregate
    ``recovery`` / ``reinsurance_premium``. There is no loss component (Sec. 65).
    What :func:`~fastcashflow.reinsurance.measure_aggregate` returns, computed in
    bounded memory.
    """

    bel: float                      # portfolio inception BEL total
    ra: float                       # portfolio inception RA total
    csm: float                      # portfolio inception CSM total
    csm_path: FloatArray            # (n_time+1,) -- aggregate CSM trajectory
    recovery: FloatArray            # (n_time,)   -- aggregate recoveries
    reinsurance_premium: FloatArray  # (n_time,)  -- aggregate reinsurance premiums


class Treaty(Protocol):
    """How a reinsurance treaty cedes the direct cash flows.

    ``cede`` receives the direct portfolio's projected :class:`Cashflows` and
    returns ``(ceded_mortality_cf, ceded_morbidity_cf, reinsurance_premium_cf)`` --
    each ``(n_mp, n_time)``. The two ceded-claim streams are kept split by
    risk so the risk adjustment can weight them by the right cv; their sum is
    the recovery. A new treaty type (excess-of-loss, surplus, ...) implements
    this one method.
    """

    def cede(self, proj: Cashflows) -> tuple[FloatArray, FloatArray, FloatArray]:
        ...


@dataclass(frozen=True, slots=True)
class QuotaShare:
    """Proportional reinsurance -- cede a fixed fraction of claims and premiums.

    ``cession`` (in ``[0, 1]``) is the ceded fraction: the cedant recovers
    that fraction of its claims and pays the same fraction of its premiums as
    reinsurance premium.
    """

    cession: float

    def __post_init__(self) -> None:
        # Validate at construction, not deep in cede(): a non-numeric, NaN or
        # out-of-range cession otherwise surfaces late or as a cryptic error.
        c = float(self.cession)  # ValueError for a non-numeric cession
        if not np.isfinite(c):
            raise ValueError(f"cession must be finite, got {self.cession!r}")
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"cession must be in [0, 1], got {self.cession!r}")

    def cede(self, proj: Cashflows) -> tuple[FloatArray, FloatArray, FloatArray]:
        if not 0.0 <= self.cession <= 1.0:
            raise ValueError(f"cession must be in [0, 1], got {self.cession}")
        return (self.cession * proj.claim_cf,
                self.cession * proj.morbidity_cf,
                self.cession * proj.premium_cf)


def measure_reinsurance(
    model_points: ModelPoints,
    basis: Basis,
    *,
    treaty: Treaty,
    full: bool = True,
) -> ReinsuranceMeasurement:
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
    basis = _single_basis(basis, entry="measure_reinsurance")
    proj = project_cashflows(model_points, basis)
    discount_bom, discount_mid = discount_factors(basis, proj.n_time)

    ceded_mortality, ceded_morbidity, reinsurance_premium = treaty.cede(proj)
    recovery = ceded_mortality + ceded_morbidity

    pv_recovery = (recovery * discount_mid).sum(axis=1)
    pv_reinsurance_premium = (reinsurance_premium * discount_bom[:-1]).sum(axis=1)
    bel = pv_reinsurance_premium - pv_recovery

    # RA -- the risk transferred, i.e. the margin on the ceded claims.
    z = _norm_ppf(basis.ra_confidence)
    pv_ceded_mortality = (ceded_mortality * discount_mid).sum(axis=1)
    pv_ceded_morbidity = (ceded_morbidity * discount_mid).sum(axis=1)
    ra = z * (basis.mortality_cv * pv_ceded_mortality
              + basis.morbidity_cv * pv_ceded_morbidity)

    # CSM -- the net cost or gain of the cover. No loss component: a net cost
    # is a negative CSM, deferred and amortised over the coverage.
    csm0 = -(bel - ra)
    if not full:
        return ReinsuranceMeasurement(
            bel=bel, ra=ra, csm=csm0, model_points=model_points)

    bel_path = (_pv_path(reinsurance_premium * discount_bom[:-1], discount_bom)
                - _pv_path(recovery * discount_mid, discount_bom))
    ra_path = z * (
        basis.mortality_cv * _pv_path(ceded_mortality * discount_mid, discount_bom)
        + basis.morbidity_cv * _pv_path(ceded_morbidity * discount_mid, discount_bom)
    )
    csm, csm_accretion, csm_release = _csm_kernel(
        csm0, proj.inforce,
        discount_monthly_curve(basis, proj.n_time),
    )

    return ReinsuranceMeasurement(
        bel=bel,
        ra=ra,
        csm=csm[:, 0],
        bel_path=bel_path,
        ra_path=ra_path,
        csm_path=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        recovery=recovery,
        reinsurance_premium=reinsurance_premium,
        cashflows=proj,
        discount_bom=discount_bom,
        model_points=model_points,
    )


def measure_reinsurance_aggregate(
    model_points: ModelPoints,
    basis: Basis,
    treaty: Treaty,
    *,
    chunk_size: int = 200_000,
) -> ReinsuranceAggregate:
    """Portfolio-aggregate reinsurance-held measurement in bounded memory.

    The reinsurance counterpart of :func:`~fastcashflow.gmm.measure_aggregate`:
    BEL / RA / CSM and the ceded cash flows are additive across contracts, so the
    ceded book's run-off is the per-model-point trajectories summed over the
    model-point axis. Runs :func:`measure_reinsurance` over row-blocks of
    ``chunk_size`` model points and accumulates only the ``(n_time+1,)`` /
    ``(n_time,)`` sums, so peak memory is ``O(chunk_size x n_time)`` regardless of
    ``n_mp``. Returns a :class:`ReinsuranceAggregate` (scalar totals + aggregate
    ``csm_path`` / ``recovery`` / ``reinsurance_premium``) -- a scalable sum of
    the measured results, not a group remeasurement. ``basis`` is a single
    :class:`Basis`, as for :func:`measure_reinsurance`.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    n_mp = model_points.n_mp
    # A chunk projects only to its own boundary.max(); its (shorter) aggregate
    # path adds into the leading slice of the global one -- a contract carries
    # nothing past its coverage period.
    n_time = int(np.asarray(model_points.contract_boundary_months).max())
    csm_path = np.zeros(n_time + 1)
    recovery = np.zeros(n_time)
    reinsurance_premium = np.zeros(n_time)
    bel = ra = csm = 0.0
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        m = measure_reinsurance(
            model_points.subset(idx), basis, treaty=treaty, full=True)
        nt1 = m.csm_path.shape[1]
        nt = m.recovery.shape[1]
        csm_path[:nt1] += m.csm_path.sum(axis=0)
        recovery[:nt] += m.recovery.sum(axis=0)
        reinsurance_premium[:nt] += m.reinsurance_premium.sum(axis=0)
        bel += float(m.bel.sum())
        ra += float(m.ra.sum())
        csm += float(m.csm.sum())
    return ReinsuranceAggregate(
        bel=bel, ra=ra, csm=csm, csm_path=csm_path,
        recovery=recovery, reinsurance_premium=reinsurance_premium)


def measure_reinsurance_stream(
    input_path,
    output_dir,
    basis: Basis,
    treaty: Treaty,
    *,
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
        measure_fn=lambda mp: measure_reinsurance(
            mp, basis, treaty=treaty, full=False),
    )


def _pv_path(month_pv: FloatArray, discount_bom: FloatArray) -> FloatArray:
    """PV-at-t trajectory of a stream whose per-month PV-to-inception is
    ``month_pv`` (shape ``(n_mp, n_time)``).

    Reverse-cumsum gives ``sum_{s>=t}`` of the inception-discounted flows
    (the PV to inception of everything from month ``t`` on); dividing by
    ``discount_bom[t]`` re-anchors it to time ``t``. Column ``n_time`` (no
    remaining flow) is 0. Column 0 reproduces the inception PV.
    """
    rev = np.cumsum(month_pv[:, ::-1], axis=1)[:, ::-1]          # sum_{s>=t}
    rev = np.concatenate([rev, np.zeros((rev.shape[0], 1))], axis=1)  # +t=n_time
    return rev / discount_bom[None, :]


def measure_reinsurance_inforce(
    model_points: ModelPoints,
    state: InforceState,
    basis: Basis,
    treaty: Treaty,
    *,
    period_months: int | None = None,
    full: bool = True,
) -> ReinsuranceMeasurement:
    """In-force subsequent measurement of a reinsurance contract held (IFRS 17
    Sec. 44, modified by Sec. 60-70) at the valuation date.

    The reinsurance counterpart of :func:`~fastcashflow.gmm.measure_inforce`.
    Each model point is valued at its ``elapsed_months`` duration: the BEL
    (PV of remaining reinsurance premiums less recoveries) and the RA (risk
    still transferred) are the inception projection sliced at the valuation
    date and re-based by ``count / inforce[elapsed]``, and the prior period's
    closing reinsurance CSM (``state.prior_csm``) is carried forward -- accreted
    at ``state.lock_in_rate`` and released over the coverage units across
    ``period_months`` (default 12). The CSM is the net cost or gain of the
    cover and may be negative; there is no loss component (Sec. 65), so the
    onerous unlocking deferred in ``gmm.measure_inforce`` does not arise here.

    ``state`` (an :class:`~fastcashflow.InforceState`) supplies the period-close
    ``elapsed_months`` / ``count`` (reconciled onto ``model_points`` by
    :func:`~fastcashflow.apply_inforce_state`) plus ``prior_csm`` /
    ``lock_in_rate``. ``treaty`` is the same cession as the new-business
    :func:`measure_reinsurance`.
    ``basis`` must resolve to a single :class:`Basis`; multi-segment routers are not
    accepted. ``full=False`` returns only the as-of headline BEL / RA / CSM and
    leaves all trajectory and cash-flow fields ``None``.
    """
    basis = _single_basis(basis, entry="reinsurance.measure_inforce")
    state = _reconcile_state(model_points, state)
    proj = project_cashflows(model_points, basis)
    n_time = proj.n_time
    n_mp = proj.inforce.shape[0]
    discount_bom, discount_mid = discount_factors(basis, n_time)

    ceded_mortality, ceded_morbidity, reinsurance_premium = treaty.cede(proj)
    recovery = ceded_mortality + ceded_morbidity

    # BEL / RA trajectories (PV at each t of the remaining ceded flows), so the
    # valuation-date slice is the PV of the *future* cash flows. Premiums are
    # bom-timed, recoveries / ceded claims mid-timed -- same convention as the
    # inception measure, so column 0 reproduces measure_reinsurance's headline.
    z = _norm_ppf(basis.ra_confidence)
    bel_path = (_pv_path(reinsurance_premium * discount_bom[:-1], discount_bom)
                - _pv_path(recovery * discount_mid, discount_bom))
    ra_path = z * (
        basis.mortality_cv * _pv_path(ceded_mortality * discount_mid, discount_bom)
        + basis.morbidity_cv * _pv_path(ceded_morbidity * discount_mid, discount_bom)
    )

    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    runoff = em >= boundary
    if np.any(runoff):
        bad = int(np.argmax(runoff))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} >= "
            f"contract_boundary_months[{bad}]={int(boundary[bad])} (the Sec. 34 "
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

    # CSM carry-forward (Sec. 44): roll the prior closing CSM one period over the
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
        prior_csm, inforce_seg, np.full(max_len, lock_in_monthly))
    csm = csm_traj[:, period_months]

    if not full:
        return ReinsuranceMeasurement(
            bel=bel, ra=ra, csm=csm, model_points=model_points)

    return ReinsuranceMeasurement(
        bel=bel, ra=ra, csm=csm,
        bel_path=bel_path,
        ra_path=ra_path,
        csm_path=csm_traj,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        recovery=recovery,
        reinsurance_premium=reinsurance_premium,
        cashflows=proj,
        discount_bom=discount_bom,
        model_points=model_points,
    )
