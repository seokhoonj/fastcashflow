"""Engine entry points.

The GMM measurement is :func:`measure`, with two paths selected by ``full``:

* ``measure(..., full=True)`` -- detailed: full monthly cash flow and CSM
  trajectories. Use it for inspection, validation and movement analysis.
* ``measure(..., full=False)`` -- fast: a single fused, parallel kernel producing
  only the headline valuation (BEL, RA, CSM, loss component) per model point. It
  materialises no per-month arrays, so it is memory-minimal and the fastest path
  for large-scale valuation.

Both paths share the same arithmetic, so the fast path reproduces the full
path's headline numbers exactly (cross-checked in the tests).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis, annual_to_monthly
from fastcashflow.curves import (
    discount_factors,
    discount_factors_from_curve,
    discount_monthly_curve,
)
from fastcashflow.numerics import (
    _cost_of_capital_ra,
    _csm_kernel,
    _norm_ppf,
    _rollforward_kernel,
    _settlement_factor,
    _settlement_lic,
)
from fastcashflow.coverage import (
    align_coverages, build_coverage_rates, coverage_arrays, validate_csr_codes,
)
from fastcashflow.io import write_measurement, _write_measurement_columns
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows
from fastcashflow.statemodel import (
    compile_state_model,
    compile_state_model_with_duration,
    is_semi_markov,
    model_references_rate,
    resolve_state_model,
)


# ---------------------------------------------------------------------------
# Detailed path
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, eq=False)
class GMMMeasurement:
    """IFRS 17 GMM measurement: BEL, RA and CSM.

    The headline fields (``bel``, ``ra``, ``csm``, ``loss_component``) are
    ``(n_mp,)`` inception values and are **always** present.

    The trajectory fields are the roll-forward over time and are populated
    only by ``measure(..., full=True)``; on the headline-only fast path
    (``full=False``) they are ``None``. ``bel_path`` / ``ra_path`` /
    ``csm_path`` are the ``(n_mp, n_time+1)`` trajectories whose column 0 is
    the inception value (so ``bel == bel_path[:, 0]`` when full). The CSM
    roll-forward decomposes as
    ``csm_path[:, t+1] = csm_path[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    ``lic`` is the liability for incurred claims -- zero unless a claims
    settlement pattern is set, which also discounts claims to their payment
    dates in the BEL.
    """

    # Headline -- always present, shape (n_mp,)
    bel: FloatArray              # inception Best Estimate of Liability
    ra: FloatArray               # inception Risk Adjustment
    csm: FloatArray              # inception Contractual Service Margin
    loss_component: FloatArray   # loss component at inception (onerous contracts)

    # Trajectory -- full=True only (None on the headline-only fast path)
    bel_path: FloatArray | None = None        # (n_mp, n_time+1) -- BEL trajectory
    ra_path: FloatArray | None = None         # (n_mp, n_time+1) -- RA trajectory
    csm_path: FloatArray | None = None        # (n_mp, n_time+1) -- CSM trajectory
    csm_accretion: FloatArray | None = None   # (n_mp, n_time)   -- CSM interest accreted
    csm_release: FloatArray | None = None     # (n_mp, n_time)   -- CSM released each month
    lic: FloatArray | None = None             # (n_mp, n_time+1) -- liability for incurred claims
    cashflows: "Cashflows | None" = None
    # bom = beginning of month, mom = mid of month: discount factors for a flow
    # at the start vs the middle of each month. Shape (n_time+1,) / (n_time,)
    # for a single basis; (n_mp, n_time+1) / (n_mp, n_time) when measured under
    # a per-segment basis dict, where each row discounts on its own curve.
    discount_bom: FloatArray | None = None  # beginning-of-month discount factors
    discount_mid: FloatArray | None = None  # mid-of-month discount factors
    # Source model points, stamped by ``measure`` so ``group(m, by=[...])`` can
    # resolve axis names without re-passing them. A reference, not a copy; None
    # on a grouped result (its rows are groups, not model points).
    model_points: "ModelPoints | None" = None

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm),
                ("loss", self.loss_component)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr("GMMMeasurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str("GMMMeasurement", self._columns())


@write_measurement.register
def _(measurement: GMMMeasurement, path, *, ids=None):
    _write_measurement_columns(
        {"bel": measurement.bel, "ra": measurement.ra, "csm": measurement.csm,
         "loss_component": measurement.loss_component}, path, ids)


def _compute_csm(bel0, ra0, inforce, monthly_rate):
    """CSM at initial recognition (Sec. 38) and deterministic roll-forward (Sec. 44).

    Pure-array orchestration: fulfilment cash flows ``FCF = BEL + RA``,
    initial CSM = ``max(0, -FCF)``, loss component = ``max(0, FCF)``, then
    the CSM is rolled forward in :func:`_csm_kernel` (interest accretion at
    the locked-in monthly rate, release proportional to coverage units --
    in-force here).

    ``inforce`` is ``(n_mp, n_time)`` (the coverage-unit series), ``bel0`` /
    ``ra0`` are ``(n_mp,)``. Returns
    ``(csm, accretion, release, loss_component)``.
    """
    fcf = bel0 + ra0
    csm0 = np.maximum(0.0, -fcf)
    loss_component = np.maximum(0.0, fcf)
    csm, accretion, release = _csm_kernel(csm0, inforce, monthly_rate)
    return csm, accretion, release, loss_component


def _measure_full(model_points: ModelPoints, basis: Basis) -> GMMMeasurement:
    """Full GMM measurement: BEL, RA and CSM rolled forward over time.

    Returns a :class:`GMMMeasurement` carrying both the ``(n_mp,)`` inception
    headline (column 0 of each trajectory) and the ``(n_mp, n_time+1)``
    ``*_path`` trajectories. Reached by ``measure(..., full=True)``.
    """
    proj = project_cashflows(model_points, basis)
    claim_cf, morbidity_cf = proj.claim_cf, proj.morbidity_cf
    monthly_rate = discount_monthly_curve(basis, proj.n_time)
    if basis.settlement_pattern is None:
        lic = np.zeros((claim_cf.shape[0], proj.n_time + 1))
    else:
        lic = _settlement_lic(claim_cf + morbidity_cf, basis.settlement_pattern)
        # Claims are paid over the pattern, not at incurrence -- discount
        # them to their payment dates in the fulfilment cash flows. With a
        # discount curve we use the in-year scalar (Sec. 40 / B71 -- the
        # rate at the month of incurrence is the right reference); the
        # full-curve treatment would require a time-varying settlement
        # factor inside the kernel, deferred.
        factor = _settlement_factor(basis.settlement_pattern, basis.discount_monthly)
        claim_cf = claim_cf * factor
        morbidity_cf = morbidity_cf * factor
    discount_bom, discount_mid = discount_factors_from_curve(monthly_rate)

    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _rollforward_kernel(
        claim_cf, morbidity_cf, proj.disability_cf, proj.expense_cf,
        proj.premium_cf, proj.annuity_cf, proj.maturity_cf, proj.surrender_cf,
        model_points.term_months, monthly_rate,
    )
    z = _norm_ppf(basis.ra_confidence)
    cl_margin = z * (basis.mortality_cv * pv_claims
                     + basis.morbidity_cv * pv_morbidity
                     + basis.disability_cv * pv_disability
                     + basis.longevity_cv * pv_survival)
    if basis.ra_method == "confidence_level":
        ra = cl_margin
    elif basis.ra_method == "cost_of_capital":
        ra = _cost_of_capital_ra(
            cl_margin, monthly_rate, basis.cost_of_capital_rate
        )
    else:
        raise ValueError(
            "ra_method must be 'confidence_level' or 'cost_of_capital', "
            f"got {basis.ra_method!r}"
        )
    csm, csm_accretion, csm_release, loss_component = _compute_csm(
        bel[:, 0], ra[:, 0], proj.inforce, monthly_rate,
    )

    return GMMMeasurement(
        bel=bel[:, 0],
        ra=ra[:, 0],
        csm=csm[:, 0],
        loss_component=loss_component,
        bel_path=bel,
        ra_path=ra,
        csm_path=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        lic=lic,
        cashflows=proj,
        discount_bom=discount_bom,
        discount_mid=discount_mid,
    )


def measure(
    model_points: ModelPoints,
    basis: "Basis | dict[tuple[str, str], Basis]",
    *,
    full: bool = True,
    backend: str = "cpu",
    discount_curve: FloatArray | None = None,
    segment_by=None,
) -> GMMMeasurement:
    """GMM measurement -- the single entry point.

    ``full=True`` (default) returns the complete roll-forward: the
    ``(n_mp,)`` inception headline *and* the ``(n_mp, n_time+1)`` ``*_path``
    trajectories. ``full=False`` is the fused, memory-minimal fast path --
    it fills only the headline (``*_path`` are ``None``) and is the right
    choice for large-scale valuation.

    ``basis`` may be a single :class:`Basis` (uniform portfolio) or a
    ``{(product_code, channel_code): Basis}`` dict; with a dict each segment is routed
    to its own basis. ``segment_by`` names the routing axes (resolved via
    :meth:`ModelPoints.axis`, so any ``attributes`` column works) and the dict
    keys are tuples of those axes in order. Left as ``None`` (the default) it is
    taken from the basis: a :class:`~fastcashflow.io.SegmentedBasis` from
    :func:`read_basis` carries the axes its workbook declared, and a plain dict
    falls back to ``("product_code", "channel_code")``. So a workbook keyed by
    ``(product_code, channel_code, risk_class)`` routes by all three with no
    extra argument; passing ``segment_by`` explicitly overrides. Cost scales with the number of distinct segments, not the
    number of axes. ``backend`` (``"cpu"``/``"gpu"``) and ``discount_curve``
    apply to the fast path only.
    """
    if isinstance(basis, dict):
        # A SegmentedBasis from read_basis remembers its axes; a plain dict
        # defaults to (product_code, channel_code). An explicit segment_by wins.
        if segment_by is None:
            segment_by = getattr(
                basis, "segment_axes", ("product_code", "channel_code"),
            )
        if full:
            if backend != "cpu" or discount_curve is not None:
                raise ValueError(
                    "backend / discount_curve apply to the fast path "
                    "(full=False) only; measure(full=True) runs the trajectory "
                    "kernel on each segment's basis.discount_annual"
                )
            result = _measure_segmented_full(
                model_points, basis, segment_by=segment_by,
            )
        else:
            result = _measure_segmented(
                model_points, basis, backend=backend,
                discount_curve=discount_curve, segment_by=segment_by,
            )
    elif full:
        if backend != "cpu" or discount_curve is not None:
            raise ValueError(
                "backend / discount_curve apply to the fast path "
                "(full=False) only; measure(full=True) runs the trajectory "
                "kernel on basis.discount_annual"
            )
        result = _measure_full(model_points, basis)
    else:
        result = _measure_fast(
            model_points, basis, backend=backend, discount_curve=discount_curve,
        )
    # Stamp the source model points so group(m, by=[...]) can resolve axis names
    # without re-passing them (a reference, not a copy).
    return replace(result, model_points=model_points)


# ---------------------------------------------------------------------------
# In-force subsequent measurement
# ---------------------------------------------------------------------------

def value_in_force(
    model_points: ModelPoints,
    basis: Basis,
    *,
    prior_csm: FloatArray | None = None,
    lock_in_rate: float | None = None,
    period_months: int | None = None,
) -> GMMMeasurement:
    """In-force subsequent measurement (IFRS 17 Sec. 40-52).

    Each model point is valued at its valuation date -- the moment that is
    ``elapsed_months[mp]`` months after that contract's inception. The
    projection still runs from inception (so the rate lookups, the
    premium-paying window and the coverage-rule clocks all use policy
    duration); the trajectory is then sliced at ``t = elapsed_months[mp]``
    per MP, which is the present value of the **future** cash flows at the
    valuation date -- the IFRS 17 BEL / RA on subsequent measurement.

    Two modes, distinguished by whether ``prior_csm`` is supplied:

    * **Hypothetical** (``prior_csm=None``, default). The CSM returned is
      the trajectory the engine produced under the assumption "the contract
      has unfolded exactly as the current best estimate predicts since
      inception". Useful for inspecting what a freshly issued contract
      would look like at duration ``E`` under today's basis, but **not a
      production-settlement CSM** -- the real-world CSM is path-dependent
      (locked-in discount rate, accumulated unlocking and experience
      adjustments) and is not the function of current basis and duration
      alone.

    * **Settlement carry-forward** (``prior_csm`` and ``lock_in_rate``
      both given). Implements IFRS 17 Sec. 44: the prior period's closing
      CSM is accreted at the locked-in rate and released over the
      coverage units forward to the valuation date. ``prior_csm`` is the
      closing CSM at month ``elapsed_months - period_months``;
      ``lock_in_rate`` is the annual locked-in discount rate for the
      contract; ``period_months`` is the length of the period rolled
      forward (default 12). v1 covers interest accretion and
      coverage-unit release only -- assumption-change unlocking and
      experience adjustments are future work and run via
      :func:`roll_forward` with full prior and current measurements.
      Because the Sec. 44 onerous trigger needs CSM unlocking to fire
      meaningfully, ``loss_component`` is returned as zeros in this mode;
      do not interpret ``bel + ra > csm`` from a settlement call as a
      Sec. 44 loss-component recognition.

    A ``ModelPoints`` with ``elapsed_months`` all zero and ``prior_csm``
    not given reproduces the new-business ``measure(..., full=False)`` result.
    """
    settlement_mode = _validate_settlement_args(
        prior_csm, lock_in_rate, period_months,
    )
    m = _measure_full(model_points, basis)
    n_mp = m.bel.shape[0]
    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    # ``elapsed_months > term_months`` means the policy is already past
    # its original maturity; the trajectory has nothing meaningful at
    # that column (the BEL trajectory only extends to t = term). Without
    # this guard the indexing reads beyond the contract horizon and
    # returns a silent zero / garbage valuation.
    term = np.asarray(model_points.term_months, dtype=np.int64)
    over = em > term
    if np.any(over):
        bad = int(np.argmax(over))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} > "
            f"term_months[{bad}]={int(term[bad])}; the policy has run past "
            "its original maturity. value_in_force needs an as-of date "
            "within the contract horizon."
        )
    rows = np.arange(n_mp)
    bel = m.bel_path[rows, em]
    ra = m.ra_path[rows, em]
    if not settlement_mode:
        # Hypothetical: take the engine-computed CSM trajectory at t=elapsed.
        csm = m.csm_path[rows, em]
        return GMMMeasurement(
            bel=bel, ra=ra, csm=csm, loss_component=m.loss_component,
        )

    # Settlement carry-forward: roll the prior closing CSM one period over
    # the coverage units from t = em - period_months to t = em.
    prior_csm = np.asarray(prior_csm, dtype=np.float64)
    if prior_csm.shape != (n_mp,):
        raise ValueError(
            f"prior_csm must have shape ({n_mp},), got {prior_csm.shape}"
        )
    period_months = int(period_months)
    prior_t = em - period_months
    if np.any(prior_t < 0):
        bad = int(np.argmin(prior_t))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} < "
            f"period_months={period_months}; the prior closing date precedes "
            "inception, which has no CSM to carry forward"
        )
    # Each MP rolls forward from its own ``prior_t``; pack the per-MP
    # inforce segments into a single (n_mp, max_len) array (zero-padded
    # past each segment's true horizon -- zero coverage units release
    # nothing, so the padded tail does not perturb the CSM step we want).
    n_time_total = m.cashflows.inforce.shape[1]
    max_len = n_time_total - int(prior_t.min())
    col_offsets = np.arange(max_len)
    src_cols = prior_t[:, None] + col_offsets[None, :]
    mask = src_cols < n_time_total
    src_cols_safe = np.where(mask, src_cols, 0)
    inforce_seg = np.where(
        mask,
        m.cashflows.inforce[rows[:, None], src_cols_safe],
        0.0,
    )
    inforce_seg = np.ascontiguousarray(inforce_seg)
    lock_in_monthly = (1.0 + float(lock_in_rate)) ** (1.0 / 12.0) - 1.0
    monthly_rates = np.full(max_len, lock_in_monthly)
    csm_traj, _, _ = _csm_kernel(prior_csm, inforce_seg, monthly_rates)
    csm = csm_traj[:, period_months]
    # Sec. 44 loss component is left as zeros here. v1 only rolls the prior
    # CSM forward (accretion + coverage-unit release); the unlocking that
    # would actually drive the carried CSM negative -- assumption changes
    # and experience variances over the period -- is roll_forward()'s job.
    # Returning max(0, bel + ra - csm) would conflate "carried CSM is short"
    # with "true onerous recognition" and mis-signal a Sec. 44 hit.
    loss = np.zeros(n_mp, dtype=np.float64)
    return GMMMeasurement(bel=bel, ra=ra, csm=csm, loss_component=loss)


def _validate_settlement_args(
    prior_csm: FloatArray | None,
    lock_in_rate: float | None,
    period_months: int | None,
) -> bool:
    """Validate the in-force settlement triple. Returns True for settlement mode.

    ``prior_csm`` and ``lock_in_rate`` are paired -- both or neither.
    ``period_months`` is only meaningful in settlement mode; passing it in
    hypothetical mode (without prior_csm / lock_in_rate) is rejected so a
    misuse cannot silently no-op.
    """
    has_prior = prior_csm is not None
    has_lock = lock_in_rate is not None
    if has_prior != has_lock:
        raise ValueError(
            "prior_csm and lock_in_rate must both be given (settlement mode) "
            "or both omitted (hypothetical mode)"
        )
    if not has_prior and period_months is not None:
        raise ValueError(
            "period_months applies only in settlement mode "
            "(when prior_csm and lock_in_rate are given)"
        )
    if has_prior:
        p = 12 if period_months is None else int(period_months)
        if p < 1:
            raise ValueError(f"period_months must be >= 1, got {period_months}")
    return has_prior


def measure_in_force(
    model_points: ModelPoints,
    basis: Basis,
    *,
    prior_csm: FloatArray | None = None,
    lock_in_rate: float | None = None,
    period_months: int | None = None,
) -> GMMMeasurement:
    """In-force subsequent measurement -- full-trajectory variant of
    :func:`value_in_force`.

    Calls :func:`measure` to build the BEL / RA / CSM trajectories from
    inception. The two modes mirror :func:`value_in_force`:

    * **Hypothetical** (``prior_csm=None``). Returns the measure() result
      unchanged -- the CSM trajectory is the one a freshly issued contract
      would produce under the current basis. Useful for inspection.
    * **Settlement carry-forward** (``prior_csm`` and ``lock_in_rate``
      given). The CSM trajectory is re-rolled from month ``elapsed_months
      - period_months`` (where ``prior_csm`` is seated as the opening
      CSM) under the locked-in rate using the same in-force-proportional
      release that :func:`roll_forward` uses. The BEL / RA trajectories
      and the cash flow detail are unchanged -- they are forward
      projections that do not depend on the prior period's CSM.
      ``loss_component`` is returned as zeros in this mode for the same
      reason as :func:`value_in_force`: Sec. 44 onerous recognition is
      only meaningful with CSM unlocking, which v1 does not perform.

    Use this when the downstream needs a full trajectory (movement
    decomposition, period-close roll-forward) rather than just the
    valuation-date headline numbers that :func:`value_in_force` returns.
    """
    settlement_mode = _validate_settlement_args(
        prior_csm, lock_in_rate, period_months,
    )
    m = _measure_full(model_points, basis)
    if not settlement_mode:
        return m

    prior_csm = np.asarray(prior_csm, dtype=np.float64)
    n_mp = m.bel.shape[0]
    if prior_csm.shape != (n_mp,):
        raise ValueError(
            f"prior_csm must have shape ({n_mp},), got {prior_csm.shape}"
        )
    period_months = int(period_months) if period_months is not None else 12
    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    term = np.asarray(model_points.term_months, dtype=np.int64)
    over = em > term
    if np.any(over):
        bad = int(np.argmax(over))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} > "
            f"term_months[{bad}]={int(term[bad])}; the policy has run past "
            "its original maturity. roll_forward needs an as-of date within "
            "the contract horizon."
        )
    prior_t = em - period_months
    if np.any(prior_t < 0):
        bad = int(np.argmin(prior_t))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} < "
            f"period_months={period_months}; the prior closing date precedes "
            "inception, which has no CSM to carry forward"
        )

    lock_in_monthly = (1.0 + float(lock_in_rate)) ** (1.0 / 12.0) - 1.0
    csm_new = m.csm_path.copy()
    csm_accretion_new = m.csm_accretion.copy()
    csm_release_new = m.csm_release.copy()

    # Re-roll each MP's CSM trajectory from t = prior_t onwards under the
    # locked-in rate and inforce-proportional release. Pack the per-MP
    # segments into a single (n_mp, max_len) zero-padded buffer and call
    # the parallel _csm_kernel once -- a per-MP Python loop calling the
    # njit kernel one MP at a time defeats the kernel's prange(n_mp) outer
    # loop and pays the dispatch overhead n_mp times.
    n_time_total = m.cashflows.inforce.shape[1]
    rows_arr = np.arange(n_mp)
    max_len = n_time_total - int(prior_t.min())
    col_offsets = np.arange(max_len)
    src_cols = prior_t[:, None] + col_offsets[None, :]
    src_mask = src_cols < n_time_total
    src_cols_safe = np.where(src_mask, src_cols, 0)
    inforce_seg = np.where(
        src_mask,
        m.cashflows.inforce[rows_arr[:, None], src_cols_safe],
        0.0,
    )
    inforce_seg = np.ascontiguousarray(inforce_seg)
    monthly_rates = np.full(max_len, lock_in_monthly)
    csm_traj, acc, rel = _csm_kernel(prior_csm, inforce_seg, monthly_rates)

    # Scatter the per-MP segments back into the (n_mp, n_time_total+1) /
    # (n_mp, n_time_total) trajectories. csm_traj has one more column
    # (kernel returns t=0 onwards including endpoint) than the accretion /
    # release arrays.
    dst_cols_csm = prior_t[:, None] + np.arange(max_len + 1)[None, :]
    dst_mask_csm = dst_cols_csm <= n_time_total
    ii_csm = np.broadcast_to(rows_arr[:, None], dst_cols_csm.shape)[dst_mask_csm]
    jj_csm = dst_cols_csm[dst_mask_csm]
    csm_new[ii_csm, jj_csm] = csm_traj[dst_mask_csm]

    dst_cols_step = prior_t[:, None] + col_offsets[None, :]
    dst_mask_step = dst_cols_step < n_time_total
    ii_step = np.broadcast_to(rows_arr[:, None], dst_cols_step.shape)[dst_mask_step]
    jj_step = dst_cols_step[dst_mask_step]
    csm_accretion_new[ii_step, jj_step] = acc[dst_mask_step]
    csm_release_new[ii_step, jj_step] = rel[dst_mask_step]

    # See value_in_force(): Sec. 44 loss component is zeroed in settlement
    # mode v1. Unlocking and experience adjustments belong to
    # roll_forward() / Phase B v2; max(0, fcf - csm) here would mis-signal
    # a Sec. 44 hit when the only thing missing is the unlocking step.
    loss_new = np.zeros(n_mp, dtype=np.float64)

    # Headline bel/ra/csm are the as-of valuation-date values (month
    # elapsed_months per MP), matching value_in_force -- NOT column 0
    # (inception), which would ignore prior_csm entirely. The trajectory
    # fields keep the full inception-to-horizon paths.
    return GMMMeasurement(
        bel=m.bel_path[rows_arr, em],
        ra=m.ra_path[rows_arr, em],
        csm=csm_new[rows_arr, em],
        loss_component=loss_new,
        bel_path=m.bel_path,
        ra_path=m.ra_path,
        csm_path=csm_new,
        csm_accretion=csm_accretion_new,
        csm_release=csm_release_new,
        lic=m.lic,
        cashflows=m.cashflows,
        discount_bom=m.discount_bom,
        discount_mid=m.discount_mid,
    )


def measure_inforce(
    model_points: ModelPoints,
    basis: Basis,
    state: "InforceState",
    *,
    period_months: int | None = None,
    full: bool = True,
) -> GMMMeasurement:
    """In-force subsequent measurement (IFRS 17 Sec. 44) at the valuation date.

    The single entry point for settlement / period-close valuation of an
    in-force book. Each model point is valued at its ``elapsed_months``
    duration, and the prior period's closing CSM (``state.prior_csm``) is
    carried forward -- accreted at ``state.lock_in_rate`` and released over
    coverage units across ``period_months`` (default 12). The headline
    ``bel`` / ``ra`` / ``csm`` are the as-of valuation-date numbers.

    ``state`` is the :class:`InforceState` returned by
    :func:`read_inforce_policies` (it carries ``prior_csm`` and
    ``lock_in_rate``); ``model_points`` carries each contract's
    ``elapsed_months``. ``full=True`` (default) returns the BEL / RA / CSM
    trajectories and cash flows for movement analysis; ``full=False``
    returns just the headline numbers (faster).

    Sec. 44 onerous unlocking and experience adjustments are not yet folded
    in here (``loss_component`` is zero in this mode); use
    :func:`roll_forward` with prior and current measurements for the full
    movement.
    """
    if full:
        return measure_in_force(
            model_points, basis,
            prior_csm=state.prior_csm,
            lock_in_rate=state.lock_in_rate,
            period_months=period_months,
        )
    return value_in_force(
        model_points, basis,
        prior_csm=state.prior_csm,
        lock_in_rate=state.lock_in_rate,
        period_months=period_months,
    )


# ---------------------------------------------------------------------------
# Codegen specialisation -- the multi-state fast kernel
# ---------------------------------------------------------------------------
#
# The multi-state CPU fast-path (full=False) kernel is generated per StateModel: the state
# count, the full edge topology, the lump-sum flags, and which states pay
# premium or a benefit all become part of the generated Python source so
# numba's compiled inner loop has no array indirections left, only scalar
# arithmetic on register-resident occupancy. One source file lives on disk
# per unique topology (see _codegen_cache_dir), and numba caches its native
# code next to it -- compile-once per topology across Python processes.
#
# Earlier iterations of this engine kept a Markov-only closure factory plus
# hand-unrolled n_states=2 and n_states=3 kernels alongside the codegen
# path. They became dead code once the codegen path was extended to all
# n_states>=2 (commit e5e8e83) and have been removed; the git history
# carries them for anyone who wants to read the earlier shape.


def _codegen_fast_kernel_source(n_states, edge_from, edge_to, edge_lump_sum,
                                 premium_state, benefit_state,
                                 use_morbidity=True, use_annuity=True,
                                 use_disability=True, use_lae=True,
                                 use_surrender=True) -> str:
    """Generate the Python source of a fully-specialised fast kernel.

    All structural parameters (n_states, edge topology, lump-sum flags,
    premium- and benefit-paying states) are baked into the source as
    literals. The returned text is intended for ``exec`` in a namespace
    that exposes ``np``, ``njit`` and ``prange``.

    The ``use_*`` flags specialise the inner loop at source-generation time:
    a cash-flow stream the portfolio does not use (morbidity, annuity,
    disability income, LAE, surrender) has its per-cell lines omitted from
    the generated source entirely -- no runtime branch, the absent term is
    simply not there. Each flag widens the codegen cache key, so a book that
    does use the stream gets its own kernel. Skipped terms are zero, so the
    result is identical to the all-streams kernel.
    """
    n_edges = len(edge_from)
    edge_from = [int(x) for x in edge_from]
    edge_to = [int(x) for x in edge_to]
    edge_lump_sum = [bool(x) for x in edge_lump_sum]
    premium_state = [bool(x) for x in premium_state]
    benefit_state = [bool(x) for x in benefit_state]

    sum_all = " + ".join(f"occ_{i}" for i in range(n_states))
    sum_prem = " + ".join(f"occ_{i}" for i in range(n_states)
                          if premium_state[i]) or "0.0"
    sum_ben = " + ".join(f"occ_{i}" for i in range(n_states)
                         if benefit_state[i]) or "0.0"

    L: list[str] = []

    def line(indent: int, text: str) -> None:
        L.append(" " * indent + text)

    # Module-level prologue. The generated text is written to a real .py
    # file under the on-disk cache so numba's @njit(cache=True) can anchor
    # its own compile cache to that file -- without a source file numba's
    # cache silently no-ops and every Python process pays the JIT cost.
    line(0, '"""Auto-generated by fastcashflow.engine.'
            '_codegen_fast_kernel_source -- do not edit."""')
    line(0, "import numpy as np")
    line(0, "from numba import njit, prange")
    line(0, "")
    line(0, "")

    def emit_init(indent: int) -> None:
        for i in range(n_states):
            line(indent, f"occ_{i} = 0.0")
        line(indent, "if ss == 0:")
        line(indent + 4, "occ_0 = cnt")
        for i in range(1, n_states):
            line(indent, f"elif ss == {i}:")
            line(indent + 4, f"occ_{i} = cnt")

    def emit_edge_step(indent: int, scale: str = "",
                       include_lump: bool = True) -> None:
        for i in range(n_states):
            line(indent, f"occ_next_{i} = 0.0")
        for e in range(n_edges):
            ef, et, ls = edge_from[e], edge_to[e], edge_lump_sum[e]
            line(indent,
                 f"flow_{e} = occ_{ef}{scale} * edge_prob[sx, age_idx, year, {e}]")
            line(indent, f"occ_next_{et} += flow_{e}")
            if include_lump and ls:
                line(indent,
                     f"pv_disability += flow_{e} * disability_benefit[mp] * dm")
        for i in range(n_states):
            line(indent, f"occ_{i} = occ_next_{i}")

    line(0, "@njit(parallel=True, cache=True)")
    line(0, "def kernel(edge_from, edge_to, edge_prob, edge_lump_sum,")
    line(0, "           premium_state, benefit_state, start_state, "
            "issue_index, sex,")
    line(0, "           term_months, count, level_premium, single_premium,")
    line(0, "           premium_term_months, premium_frequency_months, "
            "annuity_frequency_months,")
    line(0, "           coverage_index, coverage_amount, coverage_offset, coverage_rates, "
            "coverage_risk,")
    line(0, "           coverage_is_diagnosis, maturity_benefit, "
            "annuity_payment,")
    line(0, "           disability_income, disability_benefit,")
    line(0, "           alpha_pro_rata, alpha_fixed, beta_pro_rata,")
    line(0, "           gamma_fixed, lae_pro_rata,")
    line(0, "           discount_bom, discount_mid, mortality_factor,")
    line(0, "           morbidity_factor, longevity_factor, "
            "disability_factor,")
    line(0, "           coverage_waiting, coverage_reduction_end, "
            "coverage_reduction_factor,")
    line(0, "           lapse_monthly, surrender_curve):")
    line(4, "n_mp = issue_index.shape[0]")
    line(4, "bel = np.empty(n_mp)")
    line(4, "ra = np.empty(n_mp)")
    line(4, "csm = np.empty(n_mp)")
    line(4, "loss_component = np.empty(n_mp)")

    line(4, "for mp in prange(n_mp):")
    line(8, "term = term_months[mp]")
    line(8, "premium_term = premium_term_months[mp]")
    line(8, "prem_freq = premium_frequency_months[mp]")
    line(8, "ann_freq = annuity_frequency_months[mp]")
    line(8, "age_idx = issue_index[mp]")
    line(8, "sx = sex[mp]")
    line(8, "cnt = count[mp]")
    line(8, "premium = level_premium[mp]")
    line(8, "annuity = annuity_payment[mp]")
    line(8, "c_start = coverage_offset[mp]")
    line(8, "c_end = coverage_offset[mp + 1]")
    line(8, "ss = start_state[mp]")
    line(8, "ann_prem = premium * 12.0 / prem_freq")
    emit_init(8)
    line(8, "pv_mortality = 0.0")
    line(8, "pv_morbidity = 0.0")
    line(8, "pv_disability = 0.0")
    line(8, "pv_premium = 0.0")
    line(8, "pv_expense = 0.0")
    line(8, "pv_annuity = 0.0")
    line(8, "pv_surrender = 0.0")
    line(8, "cum_premium = 0.0")
    line(8, "last_year = -1")
    line(8, "claim_rate = 0.0")
    line(8, "morb_rate = 0.0")
    line(8, "prem_due = 0")
    line(8, "ann_due = 0")
    line(8, "prem_left = premium_term")

    # Main t loop
    line(8, "for t in range(term):")
    line(12, "year = t // 12")
    line(12, "if year != last_year:")
    line(16, "claim_rate = 0.0")
    line(16, "morb_rate = 0.0")
    line(16, "for k in range(c_start, c_end):")
    line(20, "cov_idx = coverage_index[k]")
    line(20, "if coverage_is_diagnosis[cov_idx]:")
    line(24, "continue")
    line(20, "if coverage_waiting[k] != 0 or coverage_reduction_end[k] != 0:")
    line(24, "continue")
    line(20, "rate = coverage_rates[cov_idx, sx, age_idx, year] * coverage_amount[k]")
    line(20, "if coverage_risk[cov_idx] == 0:")
    line(24, "claim_rate += rate")
    line(20, "else:")
    line(24, "morb_rate += rate")
    line(16, "last_year = year")
    line(12, f"ift = {sum_all}")
    line(12, f"prem_occ = {sum_prem}")
    if use_disability:
        line(12, f"benefit_occ = {sum_ben}")
    line(12, "ds = discount_bom[t]")
    line(12, "dm = discount_mid[t]")
    line(12, "single = prem_occ * single_premium[mp] if t == 0 else 0.0")
    line(12, "if prem_due == 0 and prem_left > 0:")
    line(16, "level = prem_occ * premium")
    line(16, "prem_due = prem_freq - 1")
    line(12, "else:")
    line(16, "level = 0.0")
    line(16, "prem_due -= 1")
    line(12, "prem_left -= 1")
    line(12, "pv_premium += (level + single) * ds")
    if use_surrender:
        line(12, "cum_premium += level + single")
    line(12, "pv_mortality += ift * claim_rate * dm")
    if use_morbidity:
        line(12, "pv_morbidity += ift * morb_rate * dm")
    if use_annuity:
        line(12, "if ann_due == 0:")
        line(16, "pv_annuity += ift * annuity * ds")
        line(16, "ann_due = ann_freq - 1")
        line(12, "else:")
        line(16, "ann_due -= 1")
    if use_disability:
        line(12, "pv_disability += benefit_occ * disability_income[mp] * dm")
    if use_surrender:
        # cum_premium aggregates inforce * premium; multiplying by lapse_rate
        # alone gives the per-month surrender outflow (the count is already in
        # cum_premium, so multiplying by ift here would scale by cnt^2).
        line(12, "pv_surrender += (lapse_monthly[sx, age_idx, year]")
        line(12, "                 * cum_premium * surrender_curve[t] * dm)")
    line(12, "alpha = cnt * (alpha_pro_rata * ann_prem + alpha_fixed) if t == 0 else 0.0")
    line(12, "beta = ift * beta_pro_rata * ann_prem / 12.0 if t < premium_term else 0.0")
    line(12, "gamma = ift * gamma_fixed[t]")
    if use_lae:
        line(12, "lae = lae_pro_rata[t] * "
                "ift * (claim_rate + morb_rate)")
        line(12, "pv_expense += (alpha + beta + gamma + lae) * dm")
    else:
        line(12, "pv_expense += (alpha + beta + gamma) * dm")
    emit_edge_step(12, scale="", include_lump=True)

    line(8, f"total = {sum_all}")
    line(8, "pm = total * maturity_benefit[mp] * discount_bom[term]")

    # Coverage-rule pass
    line(8, "for k in range(c_start, c_end):")
    line(12, "cov_idx = coverage_index[k]")
    line(12, "if coverage_is_diagnosis[cov_idx]:")
    line(16, "continue")
    line(12, "wait = coverage_waiting[k]")
    line(12, "red_end = coverage_reduction_end[k]")
    line(12, "if wait == 0 and red_end == 0:")
    line(16, "continue")
    line(12, "benefit = coverage_amount[k]")
    line(12, "red_factor = coverage_reduction_factor[k]")
    line(12, "mortality_risk = coverage_risk[cov_idx] == 0")
    emit_init(12)
    line(12, "for t in range(term):")
    line(16, "year = t // 12")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, f"inf = {sum_all}")
    line(20, "contrib = (inf * coverage_rates[cov_idx, sx, age_idx, year]")
    line(20, "           * benefit * mult * discount_mid[t])")
    line(20, "if mortality_risk:")
    line(24, "pv_mortality += contrib")
    line(20, "else:")
    line(24, "pv_morbidity += contrib")
    emit_edge_step(16, scale="", include_lump=False)

    # Diagnosis pass
    line(8, "for k in range(c_start, c_end):")
    line(12, "cov_idx = coverage_index[k]")
    line(12, "if not coverage_is_diagnosis[cov_idx]:")
    line(16, "continue")
    line(12, "benefit = coverage_amount[k]")
    line(12, "wait = coverage_waiting[k]")
    line(12, "red_end = coverage_reduction_end[k]")
    line(12, "red_factor = coverage_reduction_factor[k]")
    emit_init(12)
    line(12, "d_year = -1")
    line(12, "d_rate = 0.0")
    line(12, "for t in range(term):")
    line(16, "year = t // 12")
    line(16, "if year != d_year:")
    line(20, "d_rate = coverage_rates[cov_idx, sx, age_idx, year]")
    line(20, "d_year = year")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, f"healthy = {sum_all}")
    line(20, "pv_morbidity += healthy * d_rate * benefit * mult * discount_mid[t]")
    line(16, "undiagnosed = 1.0 - d_rate")
    emit_edge_step(16, scale=" * undiagnosed", include_lump=False)

    line(8, "bel_mp = (pv_mortality + pv_morbidity + pv_disability + pm")
    line(8, "          + pv_annuity + pv_expense + pv_surrender - pv_premium)")
    line(8, "ra_mp = (mortality_factor * pv_mortality + morbidity_factor * pv_morbidity")
    line(8, "         + disability_factor * pv_disability "
            "+ longevity_factor * (pm + pv_annuity))")
    line(8, "fcf = bel_mp + ra_mp")
    line(8, "bel[mp] = bel_mp")
    line(8, "ra[mp] = ra_mp")
    line(8, "csm[mp] = max(0.0, -fcf)")
    line(8, "loss_component[mp] = max(0.0, fcf)")
    line(4, "return bel, ra, csm, loss_component")

    return "\n".join(L)


_FAST_KERNEL_CODEGEN_CACHE: dict = {}


def _codegen_cache_dir() -> Path:
    """Return the on-disk directory holding generated kernel source files.

    Honours ``XDG_CACHE_HOME`` and falls back to ``~/.cache``; the
    fastcashflow-private subdirectory is created on demand.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    cache = root / "fastcashflow" / "codegen"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Two processes racing on the same topology hash produce byte-identical
    sources (codegen is deterministic), so the final file content is the
    same regardless of who wins -- the atomic replace just avoids a
    partially-written file being observed.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def _get_fast_kernel_codegen(n_states, edge_from, edge_to, edge_lump_sum,
                              premium_state, benefit_state,
                              use_morbidity=True, use_annuity=True,
                              use_disability=True, use_lae=True,
                              use_surrender=True):
    """Return a codegen-specialised kernel for the given state machine.

    Two-level cache:

    * Process-local dict keyed on the topology -- a repeat lookup in the
      same Python process returns the already-imported function with no
      filesystem touch.
    * On-disk ``.py`` file under the codegen cache directory, named by a
      hash of the generated source. The first call for a given topology
      generates the source and writes the file atomically; ``@njit(cache
      =True)`` inside the generated module then persists the compiled
      bytecode in its ``__pycache__`` so subsequent Python processes pay
      no JIT cost. Hashing the *source* automatically invalidates the
      cache when the codegen logic itself changes.
    """
    key = (
        int(n_states),
        tuple(int(x) for x in edge_from),
        tuple(int(x) for x in edge_to),
        tuple(bool(x) for x in edge_lump_sum),
        tuple(bool(x) for x in premium_state),
        tuple(bool(x) for x in benefit_state),
        bool(use_morbidity), bool(use_annuity), bool(use_disability),
        bool(use_lae), bool(use_surrender),
    )
    cached = _FAST_KERNEL_CODEGEN_CACHE.get(key)
    if cached is not None:
        return cached

    src = _codegen_fast_kernel_source(
        n_states, edge_from, edge_to, edge_lump_sum,
        premium_state, benefit_state,
        use_morbidity=use_morbidity, use_annuity=use_annuity,
        use_disability=use_disability, use_lae=use_lae,
        use_surrender=use_surrender,
    )
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cache_path = _codegen_cache_dir() / f"fast_kernel_{digest}.py"
    if not cache_path.exists():
        _atomic_write_text(cache_path, src)
    module_name = f"_fastcashflow_codegen_{digest}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, cache_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    kernel = module.kernel
    _FAST_KERNEL_CODEGEN_CACHE[key] = kernel
    return kernel


def clear_codegen_cache(*, prune_older_than_days: float | None = None) -> int:
    """Remove generated kernel sources from the on-disk codegen cache.

    Each unique state-machine topology persists a ``fast_kernel_*.py``
    (or ``fast_kernel_sm_*.py``) source file plus its ``__pycache__``
    sidecars. Over a session that explores many topologies these
    accumulate. This helper sweeps them.

    Parameters
    ----------
    prune_older_than_days
        Only drop files whose mtime is older than this many days. ``None``
        (the default) drops everything.

    Returns
    -------
    int
        Number of files removed (sources and ``__pycache__`` artefacts).

    Notes
    -----
    Disk-only: modules already imported in the current process continue to
    work via :data:`sys.modules`. Codegen for a still-needed topology will
    transparently regenerate on the next call.
    """
    import time

    cache_dir = _codegen_cache_dir()
    cutoff = (
        time.time() - 86400.0 * prune_older_than_days
        if prune_older_than_days is not None else None
    )

    removed = 0
    for entry in cache_dir.iterdir():
        if entry.is_file() and entry.name.startswith("fast_kernel_"):
            if cutoff is None or entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        elif entry.is_dir() and entry.name == "__pycache__":
            for sub in entry.iterdir():
                if cutoff is None or sub.stat().st_mtime < cutoff:
                    sub.unlink()
                    removed += 1
    return removed


# ---------------------------------------------------------------------------
# Semi-Markov codegen (Phase (c) prototype -- cancer reincidence)
# ---------------------------------------------------------------------------
#
# When the StateModel declares any state with ``duration_max > 0`` the engine
# tracks per-cohort occupancy in that state -- ``occ[state, tau]`` where
# ``tau`` is the sojourn time (months since entering the state). Transitions
# marked ``duration_dependent=True`` then carry per-cohort rates, the natural
# way to express recovery, reincidence and exclusion-period effects.
#
# The semi-Markov codegen extends the existing codegen pattern: every cohort
# of every state becomes a scalar local (``occ_<s>_<tau>``), every edge
# unrolls per-cohort (a residual stay advances the cohort with absorbing
# semantics at the last cohort, a transient transition enters the destination
# state's cohort 0). The generated source goes through the same disk-cached
# numba compile path as the Markov codegen -- compile-once per topology.
#
# Coverage-rule and diagnosis-coverage passes are emitted as no-op stubs when
# the StateModel is semi-Markov: this prototype targets the cancer-reincidence
# product, which has no rule-bearing or diagnosis coverages on the model
# points themselves -- the reincidence benefit rides the transition lump-sum.
# Full coverage-rule / diagnosis support for semi-Markov is a follow-up.


def _codegen_fast_kernel_source_semi_markov(
    n_states, state_duration_max, edge_from, edge_to, edge_lump_sum,
    premium_state, benefit_state,
    use_annuity=True, use_lae=True, use_surrender=True,
) -> str:
    """Generate the Python source of a semi-Markov-aware fast kernel.

    Same structure as :func:`_codegen_fast_kernel_source` but with per-state
    cohort scalars and per-cohort edge processing. State ``s`` with
    ``state_duration_max[s] = D`` produces ``occ_s_0, occ_s_1, ..., occ_s_{D-1}``
    local scalars; the last cohort absorbs everyone with sojourn time at
    least ``D - 1`` months.
    """
    n_edges = len(edge_from)
    edge_from = [int(x) for x in edge_from]
    edge_to = [int(x) for x in edge_to]
    edge_lump_sum = [bool(x) for x in edge_lump_sum]
    premium_state = [bool(x) for x in premium_state]
    benefit_state = [bool(x) for x in benefit_state]
    D = [int(x) for x in state_duration_max]

    def occ(s, tau):
        return f"occ_{s}_{tau}"

    def occ_next(s, tau):
        return f"occ_next_{s}_{tau}"

    # All cohort scalars across all states -- for sums and resets.
    all_occ = [occ(s, tau) for s in range(n_states) for tau in range(D[s])]
    all_next = [occ_next(s, tau) for s in range(n_states) for tau in range(D[s])]
    sum_all = " + ".join(all_occ)
    sum_prem = " + ".join(occ(s, tau) for s in range(n_states)
                          if premium_state[s]
                          for tau in range(D[s])) or "0.0"
    sum_ben = " + ".join(occ(s, tau) for s in range(n_states)
                         if benefit_state[s]
                         for tau in range(D[s])) or "0.0"

    L: list[str] = []

    def line(indent: int, text: str) -> None:
        L.append(" " * indent + text)

    def emit_init(indent: int) -> None:
        """Initialize all cohort scalars to 0, then seat ``cnt`` based on ss."""
        for s in range(n_states):
            for tau in range(D[s]):
                line(indent, f"{occ(s, tau)} = 0.0")
        # Seating: when entering a state via seating, always cohort 0.
        line(indent, "if ss == 0:")
        line(indent + 4, f"{occ(0, 0)} = cnt")
        for s in range(1, n_states):
            line(indent, f"elif ss == {s}:")
            line(indent + 4, f"{occ(s, 0)} = cnt")

    def emit_edge_step(indent: int, include_lump: bool = True,
                       scale: str = "") -> None:
        """Emit a per-timestep advance: reset next-buffer, process every edge
        across every source cohort, then copy back.

        Edge semantics:
          - residual (edge_from[e] == edge_to[e]): cohort tau advances to tau+1,
            with cohort D-1 absorbing (long-tail).
          - transient (edge_from[e] != edge_to[e]): enters destination state's
            cohort 0.
          - exits are not represented as edges; occupancy that doesn't go to
            any next-buffer slot simply leaves the in-force set.

        ``scale`` (e.g. ``" * undiagnosed"``) multiplies every flow -- used by
        the diagnosis-coverage pass to deplete the not-yet-diagnosed
        in-force fraction along the state machine.
        """
        for s in range(n_states):
            for tau in range(D[s]):
                line(indent, f"{occ_next(s, tau)} = 0.0")
        for e in range(n_edges):
            s_from = edge_from[e]
            s_to = edge_to[e]
            is_residual = (s_from == s_to)
            ls = edge_lump_sum[e]
            for tau in range(D[s_from]):
                line(indent,
                     f"flow_{e}_{tau} = {occ(s_from, tau)}{scale} "
                     f"* edge_prob[sx, age_idx, year, {e}, {tau}]")
                if is_residual:
                    # Cohort advance; cap at D-1 (absorbing).
                    next_tau = tau + 1 if tau + 1 < D[s_from] else D[s_from] - 1
                    line(indent,
                         f"{occ_next(s_from, next_tau)} += flow_{e}_{tau}")
                else:
                    # Transient: enter destination's cohort 0.
                    line(indent,
                         f"{occ_next(s_to, 0)} += flow_{e}_{tau}")
                if include_lump and ls:
                    line(indent,
                         f"pv_disability += flow_{e}_{tau} "
                         f"* disability_benefit[mp] * dm")
        # Copy back.
        for s in range(n_states):
            for tau in range(D[s]):
                line(indent, f"{occ(s, tau)} = {occ_next(s, tau)}")

    # --- Prologue: imports + decorator ----------------------------------
    line(0, '"""Auto-generated by fastcashflow.engine.'
            '_codegen_fast_kernel_source_semi_markov -- do not edit."""')
    line(0, "import numpy as np")
    line(0, "from numba import njit, prange")
    line(0, "")
    line(0, "")
    line(0, "@njit(parallel=True, cache=True)")
    line(0, "def kernel(edge_prob, start_state, issue_index, sex,")
    line(0, "           term_months, count, level_premium, single_premium,")
    line(0, "           premium_term_months, premium_frequency_months, "
            "annuity_frequency_months,")
    line(0, "           coverage_index, coverage_amount, coverage_offset, coverage_rates, "
            "coverage_risk,")
    line(0, "           coverage_is_diagnosis, maturity_benefit, "
            "annuity_payment,")
    line(0, "           disability_income, disability_benefit,")
    line(0, "           alpha_pro_rata, alpha_fixed, beta_pro_rata,")
    line(0, "           gamma_fixed, lae_pro_rata,")
    line(0, "           discount_bom, discount_mid, mortality_factor,")
    line(0, "           morbidity_factor, longevity_factor, "
            "disability_factor,")
    line(0, "           coverage_waiting, coverage_reduction_end, "
            "coverage_reduction_factor,")
    line(0, "           lapse_monthly, surrender_curve):")
    line(4, "n_mp = issue_index.shape[0]")
    line(4, "bel = np.empty(n_mp)")
    line(4, "ra = np.empty(n_mp)")
    line(4, "csm = np.empty(n_mp)")
    line(4, "loss_component = np.empty(n_mp)")

    line(4, "for mp in prange(n_mp):")
    line(8, "term = term_months[mp]")
    line(8, "premium_term = premium_term_months[mp]")
    line(8, "prem_freq = premium_frequency_months[mp]")
    line(8, "ann_freq = annuity_frequency_months[mp]")
    line(8, "age_idx = issue_index[mp]")
    line(8, "sx = sex[mp]")
    line(8, "cnt = count[mp]")
    line(8, "premium = level_premium[mp]")
    line(8, "annuity = annuity_payment[mp]")
    line(8, "c_start = coverage_offset[mp]")
    line(8, "c_end = coverage_offset[mp + 1]")
    line(8, "ss = start_state[mp]")
    line(8, "ann_prem = premium * 12.0 / prem_freq")
    emit_init(8)
    line(8, "pv_mortality = 0.0")
    line(8, "pv_morbidity = 0.0")
    line(8, "pv_disability = 0.0")
    line(8, "pv_premium = 0.0")
    line(8, "pv_expense = 0.0")
    line(8, "pv_annuity = 0.0")
    line(8, "pv_surrender = 0.0")
    line(8, "cum_premium = 0.0")
    line(8, "last_year = -1")
    line(8, "claim_rate = 0.0")
    line(8, "morb_rate = 0.0")
    line(8, "prem_due = 0")
    line(8, "ann_due = 0")
    line(8, "prem_left = premium_term")
    # Saved per-month total in-force (sum across all cohorts). The rule
    # and diagnosis passes below reuse this trajectory instead of
    # re-running the state machine -- a 5x-10x win for semi-Markov
    # contracts that combine a deep cohort grid with rule-bearing or
    # diagnosis coverages. Same trick the detailed _project_kernel_semi_markov
    # has been using since the cohort-aware projection landed.
    line(8, "inforce_traj = np.empty(term)")

    # --- Main t loop ---------------------------------------------------
    line(8, "for t in range(term):")
    line(12, "year = t // 12")
    line(12, "if year != last_year:")
    line(16, "claim_rate = 0.0")
    line(16, "morb_rate = 0.0")
    line(16, "for k in range(c_start, c_end):")
    line(20, "cov_idx = coverage_index[k]")
    line(20, "if coverage_is_diagnosis[cov_idx]:")
    line(24, "continue          # diagnosis coverages run separately")
    line(20, "if coverage_waiting[k] != 0 or coverage_reduction_end[k] != 0:")
    line(24, "continue          # rule-bearing coverages run separately")
    line(20, "rate = coverage_rates[cov_idx, sx, age_idx, year] * coverage_amount[k]")
    line(20, "if coverage_risk[cov_idx] == 0:")
    line(24, "claim_rate += rate")
    line(20, "else:")
    line(24, "morb_rate += rate")
    line(16, "last_year = year")
    line(12, f"ift = {sum_all}")
    line(12, "inforce_traj[t] = ift")
    line(12, f"prem_occ = {sum_prem}")
    line(12, f"benefit_occ = {sum_ben}")
    line(12, "ds = discount_bom[t]")
    line(12, "dm = discount_mid[t]")
    line(12, "single = prem_occ * single_premium[mp] if t == 0 else 0.0")
    line(12, "if prem_due == 0 and prem_left > 0:")
    line(16, "level = prem_occ * premium")
    line(16, "prem_due = prem_freq - 1")
    line(12, "else:")
    line(16, "level = 0.0")
    line(16, "prem_due -= 1")
    line(12, "prem_left -= 1")
    line(12, "pv_premium += (level + single) * ds")
    if use_surrender:
        line(12, "cum_premium += level + single")
    line(12, "pv_mortality += ift * claim_rate * dm")
    line(12, "pv_morbidity += ift * morb_rate * dm")
    if use_annuity:
        line(12, "if ann_due == 0:")
        line(16, "pv_annuity += ift * annuity * ds")
        line(16, "ann_due = ann_freq - 1")
        line(12, "else:")
        line(16, "ann_due -= 1")
    line(12, "pv_disability += benefit_occ * disability_income[mp] * dm")
    if use_surrender:
        # cum_premium aggregates inforce * premium; multiplying by lapse_rate
        # alone gives the per-month surrender outflow (the count is already in
        # cum_premium, so multiplying by ift here would scale by cnt^2).
        line(12, "pv_surrender += (lapse_monthly[sx, age_idx, year]")
        line(12, "                 * cum_premium * surrender_curve[t] * dm)")
    line(12, "alpha = cnt * (alpha_pro_rata * ann_prem + alpha_fixed) if t == 0 else 0.0")
    line(12, "beta = ift * beta_pro_rata * ann_prem / 12.0 if t < premium_term else 0.0")
    line(12, "gamma = ift * gamma_fixed[t]")
    if use_lae:
        line(12, "lae = lae_pro_rata[t] * "
                "ift * (claim_rate + morb_rate)")
        line(12, "pv_expense += (alpha + beta + gamma + lae) * dm")
    else:
        line(12, "pv_expense += (alpha + beta + gamma) * dm")
    emit_edge_step(12, include_lump=True)

    line(8, f"total = {sum_all}")
    line(8, "pm = total * maturity_benefit[mp] * discount_bom[term]")

    # --- Coverage-rule pass --------------------------------------------
    # Rule-bearing non-diagnosis coverages: reuse the per-month total
    # in-force trajectory the main pass saved instead of re-running the
    # cohort-aware state machine. Each coverage becomes an O(term) scalar
    # loop -- the same trick the detailed _project_kernel_semi_markov
    # has been using all along.
    line(8, "for k in range(c_start, c_end):")
    line(12, "cov_idx = coverage_index[k]")
    line(12, "if coverage_is_diagnosis[cov_idx]:")
    line(16, "continue")
    line(12, "wait = coverage_waiting[k]")
    line(12, "red_end = coverage_reduction_end[k]")
    line(12, "if wait == 0 and red_end == 0:")
    line(16, "continue          # rule-free -- already in the main pass")
    line(12, "benefit = coverage_amount[k]")
    line(12, "red_factor = coverage_reduction_factor[k]")
    line(12, "mortality_risk = coverage_risk[cov_idx] == 0")
    line(12, "for t in range(wait, term):")
    line(16, "year = t // 12")
    line(16, "mult = red_factor if t < red_end else 1.0")
    line(16, "contrib = (inforce_traj[t] * coverage_rates[cov_idx, sx, age_idx, year]")
    line(16, "           * benefit * mult * discount_mid[t])")
    line(16, "if mortality_risk:")
    line(20, "pv_mortality += contrib")
    line(16, "else:")
    line(20, "pv_morbidity += contrib")

    # --- Diagnosis-coverage pass ---------------------------------------
    # Diagnosis coverages run off a depleting not-yet-diagnosed pool.
    # The pool's depletion (`undiagnosed`) is a scalar that multiplies the
    # saved cohort-aware in-force trajectory -- mathematically equivalent
    # to running the state machine with each flow scaled by (1 - d_rate),
    # but a single scalar loop per coverage rather than a full cohort walk.
    line(8, "for k in range(c_start, c_end):")
    line(12, "cov_idx = coverage_index[k]")
    line(12, "if not coverage_is_diagnosis[cov_idx]:")
    line(16, "continue")
    line(12, "benefit = coverage_amount[k]")
    line(12, "wait = coverage_waiting[k]")
    line(12, "red_end = coverage_reduction_end[k]")
    line(12, "red_factor = coverage_reduction_factor[k]")
    line(12, "undiagnosed = 1.0")
    line(12, "d_year = -1")
    line(12, "d_rate = 0.0")
    line(12, "for t in range(term):")
    line(16, "year = t // 12")
    line(16, "if year != d_year:")
    line(20, "d_rate = coverage_rates[cov_idx, sx, age_idx, year]")
    line(20, "d_year = year")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, "pv_morbidity += (inforce_traj[t] * undiagnosed * d_rate * benefit")
    line(20, "                 * mult * discount_mid[t])")
    line(16, "undiagnosed *= (1.0 - d_rate)")

    # --- Final output --------------------------------------------------
    line(8, "bel_mp = (pv_mortality + pv_morbidity + pv_disability + pm")
    line(8, "          + pv_annuity + pv_expense + pv_surrender - pv_premium)")
    line(8, "ra_mp = (mortality_factor * pv_mortality + morbidity_factor * pv_morbidity")
    line(8, "         + disability_factor * pv_disability "
            "+ longevity_factor * (pm + pv_annuity))")
    line(8, "fcf = bel_mp + ra_mp")
    line(8, "bel[mp] = bel_mp")
    line(8, "ra[mp] = ra_mp")
    line(8, "csm[mp] = max(0.0, -fcf)")
    line(8, "loss_component[mp] = max(0.0, fcf)")
    line(4, "return bel, ra, csm, loss_component")

    return "\n".join(L)


_FAST_KERNEL_CODEGEN_SEMI_MARKOV_CACHE: dict = {}


def _get_fast_kernel_codegen_semi_markov(
    n_states, state_duration_max, edge_from, edge_to, edge_lump_sum,
    premium_state, benefit_state,
    use_annuity=True, use_lae=True, use_surrender=True,
):
    """Return a semi-Markov codegen-specialised kernel for the given
    topology + per-state cohort counts.

    Same two-level cache pattern as :func:`_get_fast_kernel_codegen` --
    process-local dict for in-memory hits, content-addressed ``.py`` file
    on disk so ``@njit(cache=True)`` persists the compiled native code
    across processes.
    """
    key = (
        int(n_states),
        tuple(int(x) for x in state_duration_max),
        tuple(int(x) for x in edge_from),
        tuple(int(x) for x in edge_to),
        tuple(bool(x) for x in edge_lump_sum),
        tuple(bool(x) for x in premium_state),
        tuple(bool(x) for x in benefit_state),
        bool(use_annuity), bool(use_lae), bool(use_surrender),
    )
    cached = _FAST_KERNEL_CODEGEN_SEMI_MARKOV_CACHE.get(key)
    if cached is not None:
        return cached

    src = _codegen_fast_kernel_source_semi_markov(
        n_states, state_duration_max, edge_from, edge_to, edge_lump_sum,
        premium_state, benefit_state,
        use_annuity=use_annuity, use_lae=use_lae, use_surrender=use_surrender,
    )
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cache_path = _codegen_cache_dir() / f"fast_kernel_sm_{digest}.py"
    if not cache_path.exists():
        _atomic_write_text(cache_path, src)
    module_name = f"_fastcashflow_codegen_sm_{digest}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, cache_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    kernel = module.kernel
    _FAST_KERNEL_CODEGEN_SEMI_MARKOV_CACHE[key] = kernel
    return kernel


@njit(parallel=True, cache=True)
def _fast_kernel_scalar(issue_index, sex, term_months, count, level_premium,
                         single_premium, premium_term_months, premium_frequency_months,
                         annuity_frequency_months, coverage_index, coverage_amount, coverage_offset,
                         coverage_rates, coverage_risk, coverage_is_diagnosis,
                         maturity_benefit, annuity_payment,
                         alpha_pro_rata, alpha_fixed, beta_pro_rata,
                         gamma_fixed, lae_pro_rata,
                         discount_bom, discount_mid,
                         mortality_factor, morbidity_factor, longevity_factor,
                         coverage_waiting, coverage_reduction_end, coverage_reduction_factor,
                         survival_monthly, lapse_monthly, surrender_curve,
                         use_morbidity, use_annuity, use_lae, use_surrender):
    """Scalar-inforce fast path of the general codegen fast kernel
    (:func:`_codegen_fast_kernel_source`).

    Used when the in-force projection collapses to a single survival track --
    no user-supplied StateModel, no waiver inception, every model point
    seated in the active state. The in-force is carried as a scalar; the
    monthly decay is one multiply against the precomputed
    ``survival_monthly[sex, age, year] = (1 - q_monthly) * (1 - l_monthly)``
    table. Numerically identical to the general codegen kernel for this configuration
    -- the disability income, disability lump-sum and benefit-state pieces
    of the general kernel evaluate to zero here -- and recovers the
    pre-Phase(b) speed (see ``docs/tutorial/13-engine-design.md``).
    """
    n_mp = issue_index.shape[0]
    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    loss_component = np.empty(n_mp)

    for mp in prange(n_mp):
        term = term_months[mp]
        premium_term = premium_term_months[mp]
        prem_freq = premium_frequency_months[mp]
        ann_freq = annuity_frequency_months[mp]
        age_idx = issue_index[mp]
        sx = sex[mp]
        cnt = count[mp]
        premium = level_premium[mp]
        annuity = annuity_payment[mp]
        c_start = coverage_offset[mp]
        c_end = coverage_offset[mp + 1]
        inforce = cnt
        pv_mortality = 0.0  # mortality-risk claim PV (death claims)
        pv_morbidity = 0.0  # morbidity-risk claim PV (health claims)
        pv_premium = 0.0
        pv_expense = 0.0
        pv_annuity = 0.0
        pv_surrender = 0.0  # PV of surrender value (해약환급금)
        cum_premium = 0.0  # cumulative premium paid -- surrender basis
        last_year = -1
        claim_rate = 0.0
        morb_rate = 0.0
        # Counters replace modulo / less-than checks in the inner loop --
        # ``prem_due`` ticks down to the next premium-paying month,
        # ``ann_due`` to the next annuity month, and ``prem_left`` to the
        # end of the premium-paying term. Profiling shows the modulo /
        # comparison form costs ~2/3 of the inner-loop time at large
        # portfolios -- the counter form lets the compiler keep the loop
        # branch-light and 1M MP runs in ~50 ms again.
        prem_due = 0
        ann_due = 0
        prem_left = premium_term
        for t in range(term):
            year = t // 12
            if year != last_year:
                claim_rate = 0.0
                morb_rate = 0.0
                for k in range(c_start, c_end):
                    cov_idx = coverage_index[k]
                    if coverage_is_diagnosis[cov_idx]:
                        continue
                    if coverage_waiting[k] != 0 or coverage_reduction_end[k] != 0:
                        continue
                    rate = coverage_rates[cov_idx, sx, age_idx, year] * coverage_amount[k]
                    if coverage_risk[cov_idx] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year
            ds = discount_bom[t]
            dm = discount_mid[t]
            single = inforce * single_premium[mp] if t == 0 else 0.0
            if prem_due == 0 and prem_left > 0:
                level = inforce * premium
                prem_due = prem_freq - 1
            else:
                level = 0.0
                prem_due -= 1
            prem_left -= 1
            pv_premium += (level + single) * ds
            pv_mortality += inforce * claim_rate * dm
            # Streams the portfolio does not use are skipped wholesale: the
            # guards are loop-invariant per call, so the branch predicts
            # perfectly and the per-cell array gathers / multiplies of an
            # absent stream are not paid. A death-only book pays for none of
            # surrender, annuity, morbidity or LAE.
            if use_morbidity:
                pv_morbidity += inforce * morb_rate * dm
            if use_annuity:
                if ann_due == 0:
                    pv_annuity += inforce * annuity * ds
                    ann_due = ann_freq - 1
                else:
                    ann_due -= 1
            ann_prem = premium * 12.0 / prem_freq
            alpha = (cnt * (alpha_pro_rata * ann_prem + alpha_fixed)
                     if t == 0 else 0.0)
            beta = (inforce * beta_pro_rata * ann_prem / 12.0
                    if t < premium_term else 0.0)
            gamma = inforce * gamma_fixed[t]
            if use_lae:
                lae = lae_pro_rata[t] * inforce * (claim_rate + morb_rate)
                pv_expense += (alpha + beta + gamma + lae) * dm
            else:
                pv_expense += (alpha + beta + gamma) * dm
            if use_surrender:
                # cum_premium aggregates inforce * premium and is the surrender
                # basis; multiplying by lapse_rate alone gives the per-month
                # surrender outflow (the count is already in cum_premium).
                cum_premium += level + single
                pv_surrender += (lapse_monthly[sx, age_idx, year]
                                 * cum_premium * surrender_curve[t] * dm)
            inforce *= survival_monthly[sx, age_idx, year]
        pm = inforce * maturity_benefit[mp] * discount_bom[term]
        # Non-diagnosis coverages with a waiting or reduced-benefit rule:
        # rerun the survival on the same scalar track so the benefit
        # multiplier (which can change mid-year) applies cleanly.
        for k in range(c_start, c_end):
            cov_idx = coverage_index[k]
            if coverage_is_diagnosis[cov_idx]:
                continue
            wait = coverage_waiting[k]
            red_end = coverage_reduction_end[k]
            if wait == 0 and red_end == 0:
                continue
            benefit = coverage_amount[k]
            red_factor = coverage_reduction_factor[k]
            mortality_risk = coverage_risk[cov_idx] == 0
            inf = cnt
            for t in range(term):
                year = t // 12
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    contrib = (inf * coverage_rates[cov_idx, sx, age_idx, year]
                               * benefit * mult * discount_mid[t])
                    if mortality_risk:
                        pv_mortality += contrib
                    else:
                        pv_morbidity += contrib
                inf *= survival_monthly[sx, age_idx, year]
        # Diagnosis coverages: claims run off a depleting "not yet diagnosed"
        # pool, which depletes both by survival and by the diagnosis rate.
        for k in range(c_start, c_end):
            cov_idx = coverage_index[k]
            if not coverage_is_diagnosis[cov_idx]:
                continue
            benefit = coverage_amount[k]
            wait = coverage_waiting[k]
            red_end = coverage_reduction_end[k]
            red_factor = coverage_reduction_factor[k]
            healthy = cnt
            d_year = -1
            d_rate = 0.0
            for t in range(term):
                year = t // 12
                if year != d_year:
                    d_rate = coverage_rates[cov_idx, sx, age_idx, year]
                    d_year = year
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    pv_morbidity += healthy * d_rate * benefit * mult * discount_mid[t]
                healthy *= survival_monthly[sx, age_idx, year] * (1.0 - d_rate)
        bel_mp = pv_mortality + pv_morbidity + pm + pv_annuity + pv_expense + pv_surrender - pv_premium
        ra_mp = (mortality_factor * pv_mortality + morbidity_factor * pv_morbidity
                 + longevity_factor * (pm + pv_annuity))
        fcf = bel_mp + ra_mp
        bel[mp] = bel_mp
        ra[mp] = ra_mp
        csm[mp] = max(0.0, -fcf)
        loss_component[mp] = max(0.0, fcf)

    return bel, ra, csm, loss_component


def _measure_fast(
    model_points: ModelPoints,
    basis: Basis,
    *,
    backend: str = "cpu",
    discount_curve: FloatArray | None = None,
) -> GMMMeasurement:
    """Fast GMM valuation: BEL, RA and CSM per model point.

    One fused kernel; no per-month arrays are materialised. This is the
    memory-minimal, fastest path for large-scale valuation. For full cash
    flow / CSM trajectories use :func:`measure`.

    Parameters
    ----------
    backend :
        ``"cpu"`` (default) runs the numba parallel kernel across cores.
        ``"gpu"`` runs the CUDA kernel; it needs a CUDA device and is worth
        it only at large scale (kernel-launch and transfer cost is fixed).
    discount_curve :
        Optional ``(n_time,)`` array of annual discount rates -- one per
        projection month, a power-user override for the stochastic case
        where the rate must vary month by month. ``None`` (the default)
        uses the scalar or per-year curve on
        ``basis.discount_annual``; when supplied, it overrides that
        and bypasses the curves layer for the discount step.
    """
    if basis.ra_method != "confidence_level":
        raise ValueError(
            "measure(full=False) computes the confidence-level RA only; use "
            f"full=True for ra_method={basis.ra_method!r}"
        )
    # the fast path builds the rate grid at issue_class = 0 throughout; a portfolio
    # with non-zero classes would silently look up the wrong row of any
    # issue_class-bearing rate table. measure() handles it correctly.
    if np.any(model_points.issue_class != 0):
        raise NotImplementedError(
            "measure(full=False) currently evaluates rates on a single-issue-class grid "
            "(class=0). The portfolio carries non-zero issue_class values, "
            "which would land at class 0 in the fast path and produce a silently "
            "wrong BEL. Use full=True until the fast path grows per-class grid "
            "support."
        )
    if model_points.term_months.shape[0] == 0:
        raise ValueError(
            "model_points is empty (n_mp=0); measure(full=False) cannot project a "
            "zero-policy portfolio. Filter empty segments upstream."
        )
    n_time = int(model_points.term_months.max())
    n_years = (n_time + 11) // 12

    # Mortality and lapse are evaluated on a dense sex x [min, max] issue-age
    # x duration grid. Using the age range rather than the exact distinct
    # ages avoids an O(n log n) sort (np.unique): min/max and the index
    # subtraction are O(n), and the few unused ages cost nothing -- the
    # assumption grid is tiny.
    min_age = int(model_points.issue_age.min())
    max_age = int(model_points.issue_age.max())
    durations = np.arange(n_years)
    sex_grid, issue_age_grid, duration_grid = np.meshgrid(
        np.array([0, 1]), np.arange(min_age, max_age + 1), durations,
        indexing="ij",
    )
    # ``issue_class`` / ``elapsed`` axes -- passed to keep the unified
    # 5-arg rate callable shape. The dense (2, n_ages, n_year) setup grid
    # is class=0 / elapsed=0 throughout for now: tables that declare an
    # issue_class axis (a future axis-aware grid build will plug per-MP
    # class values in) are looked up at class 0; sojourn-aware tables are
    # called with the cohort dim explicitly in the semi-Markov branch
    # below. Tables without these axes broadcast over them as before.
    issue_class_grid = np.zeros_like(duration_grid)
    elapsed_grid = np.zeros_like(duration_grid)
    # Rates are supplied annual; the engine converts each to a monthly rate
    # on the constant-force basis (see basis.annual_to_monthly).
    mortality_annual_grid = basis.mortality_annual(
        sex_grid, issue_age_grid, duration_grid,
        issue_class_grid, elapsed_grid)
    mortality_grid = np.ascontiguousarray(annual_to_monthly(mortality_annual_grid))
    issue_index = (model_points.issue_age - min_age).astype(np.int64)
    lapse_grid = np.ascontiguousarray(annual_to_monthly(
        basis.lapse_annual(
            sex_grid, issue_age_grid, duration_grid,
            issue_class_grid, elapsed_grid)))
    # Fast path: when no waiver / paid-up mechanic is active and every model
    # point is seated in the active state, the in-force is a single survival
    # track. The scalar kernel carries it as one number and runs the
    # pre-Phase(b) speed path; the N-state kernel is reserved for products
    # that genuinely need an occupancy vector.
    fast_path = (backend == "cpu"
                 and basis.state_model is None
                 and basis.waiver_incidence_annual is None
                 and not np.any(model_points.state))
    if not fast_path:
        if basis.waiver_incidence_annual is None:
            waiver_grid = np.zeros_like(mortality_grid)
        else:
            waiver_grid = np.ascontiguousarray(annual_to_monthly(
                basis.waiver_incidence_annual(
                    sex_grid, issue_age_grid, duration_grid,
                    issue_class_grid, elapsed_grid)))
        # In-force state machine -- see ``statemodel.resolve_state_model``
        # for the fallback policy when ``basis.state_model`` is unset.
        # The transition rates land on the sex x age x duration grid the
        # kernel indexes.
        state_model = resolve_state_model(basis)
        semi_markov = is_semi_markov(state_model)
        if semi_markov:
            # Phase (c) path -- a state declared ``duration_max > 0`` tracks
            # per-cohort occupancy. The rate dict carries one entry per name
            # the model references; duration-dependent rates land here as
            # 4D arrays (sex, age, year, cohort), static rates as 3D.
            max_cohort = max(s.duration_max for s in state_model.states
                              if s.duration_max > 0)
            rate_dict = {"mortality": mortality_grid,
                          "lapse": lapse_grid}
            if basis.waiver_incidence_annual is not None:
                waiver_grid = np.ascontiguousarray(annual_to_monthly(
                    basis.waiver_incidence_annual(
                        sex_grid, issue_age_grid, duration_grid,
                        issue_class_grid, elapsed_grid)))
                rate_dict["waiver_incidence"] = waiver_grid
            if basis.ci_incidence_annual is not None:
                ci_inc_grid = np.ascontiguousarray(annual_to_monthly(
                    basis.ci_incidence_annual(
                        sex_grid, issue_age_grid, duration_grid,
                        issue_class_grid, elapsed_grid)))
                rate_dict["ci_incidence"] = ci_inc_grid
            if (basis.ci_reincidence_annual is not None
                    or basis.disability_recovery_annual is not None):
                # Build the (sex, age, year, cohort) grid by sweeping cohort
                # months 0..max_cohort-1. Duration-dependent rate callables
                # take the four-argument signature -- ``state_duration``
                # (the sojourn-time cohort index) lets a 면책 (exclusion)
                # window on reincidence or the duration-since-disablement
                # taper on a DI recovery rate drop straight in.
                cohort_idx = np.arange(max_cohort)
                sg4, ag4, dg4, cg4 = np.meshgrid(
                    np.array([0, 1]),
                    np.arange(min_age, max_age + 1),
                    durations,
                    cohort_idx,
                    indexing="ij",
                )
                # ``issue_class`` axis broadcast at zero on the 4D
                # sojourn grid; ``elapsed`` here is the cohort index
                # (months since entering the source state).
                ic4 = np.zeros_like(cg4)
                if basis.ci_reincidence_annual is not None:
                    rate_dict["ci_reincidence"] = np.ascontiguousarray(
                        annual_to_monthly(
                            basis.ci_reincidence_annual(
                                sg4, ag4, dg4, ic4, cg4)))
                if basis.disability_recovery_annual is not None:
                    rate_dict["disability_recovery"] = np.ascontiguousarray(
                        annual_to_monthly(
                            basis.disability_recovery_annual(
                                sg4, ag4, dg4, ic4, cg4)))
            compiled = compile_state_model_with_duration(
                state_model, rate_dict,
            )
            edge_from = compiled.edge_from
            edge_to = compiled.edge_to
            edge_lump_sum = compiled.edge_lump_sum
            n_states = compiled.n_states
            premium_state = compiled.premium_state
            benefit_state = compiled.benefit_state
            state_duration_max = compiled.state_duration_max
            # compile_state_model_with_duration returns ``edge_prob`` shape
            # ``(n_edges, sex, age, year, max_D)``. Transpose to put the
            # (sex, age, year) lookup axes outermost and the edge / cohort
            # indices last, so a per-edge per-cohort access for one
            # (sex, age, year) stays in cache.
            edge_prob = np.ascontiguousarray(
                np.transpose(compiled.edge_prob, (1, 2, 3, 0, 4)))
        else:
            # Markov path -- the rate dict mirrors the semi-Markov branch
            # above for the rates that are not duration-dependent. A custom
            # Markov topology that references ``ci_incidence`` works the same
            # way it does on the semi-Markov side, instead of hitting
            # ``compile_state_model``'s "rate not supplied" error. The two
            # 4D sojourn rates (``ci_reincidence``, ``disability_recovery``)
            # remain semi-Markov-only -- they need a cohort axis the Markov
            # kernel does not carry.
            rate_dict = {"mortality": mortality_grid,
                         "waiver_incidence": waiver_grid,
                         "lapse": lapse_grid}
            if basis.ci_incidence_annual is not None:
                ci_inc_grid = np.ascontiguousarray(annual_to_monthly(
                    basis.ci_incidence_annual(
                        sex_grid, issue_age_grid, duration_grid,
                        issue_class_grid, elapsed_grid)))
                rate_dict["ci_incidence"] = ci_inc_grid
            if model_references_rate(state_model, "lapse_paidup"):
                paidup_fn = (basis.lapse_paidup_annual
                             or basis.lapse_annual)
                rate_dict["lapse_paidup"] = np.ascontiguousarray(
                    annual_to_monthly(paidup_fn(
                        sex_grid, issue_age_grid, duration_grid,
                        issue_class_grid, elapsed_grid)))
            compiled = compile_state_model(state_model, rate_dict)
            edge_from = compiled.edge_from
            edge_to = compiled.edge_to
            edge_lump_sum = compiled.edge_lump_sum
            n_states = compiled.n_states
            premium_state = compiled.premium_state
            benefit_state = compiled.benefit_state
            # compile_state_model returns ``edge_prob`` with the edge axis
            # first -- (n_edges, sex, age, year). Transpose so the edge axis
            # is innermost: all edges for a given (sex, age, year) lookup
            # land in one cache line, ~25% faster on the multi-state hot
            # path.
            edge_prob = np.ascontiguousarray(
                np.transpose(compiled.edge_prob, (1, 2, 3, 0)))
            state_duration_max = None
        start_state = np.asarray(state_model.seating, np.int64)[model_points.state]
    # Align the basis' coverages to the order the model points were
    # built against (the one place the basis enter the coverage
    # indexing). Identity when the model points were built against this
    # same Basis; a reorder when read built them catalogue-order.
    aligned_coverages = align_coverages(
        basis.coverages, model_points.coverage_codes)
    validate_csr_codes(
        model_points.coverage_index, len(aligned_coverages),
        coverages=aligned_coverages,
        calculation_methods=model_points.calculation_methods,
    )
    coverage_is_diagnosis, coverage_risk = coverage_arrays(
        aligned_coverages, model_points.calculation_methods,
    )
    # build_coverage_rates stacks the per-coverage annual rates; the whole
    # stack is converted to monthly. mortality_annual is a separate engine
    # input (the in-force decrement); a contract's death coverage, if any,
    # lives in basis.coverages with its own rate_table -- usually the
    # same mortality table referenced from that sheet, occasionally a
    # separately calibrated death-claim experience table.
    coverage_rates = np.ascontiguousarray(annual_to_monthly(build_coverage_rates(
        [r.rate for r in aligned_coverages], sex_grid,
        issue_age_grid, duration_grid, issue_class_grid, elapsed_grid,
    )))
    # Shape contract: _fast_kernel_scalar indexes coverage_rates[cov_idx, sx, age_idx,
    # year] against the dense (sex, age, year) lookup grid. Lock the shape
    # here so a future grid refactor surfaces at this assertion rather than
    # producing a silently-broadcast wrong claim rate.
    n_ages = max_age - min_age + 1
    assert coverage_rates.shape == (
        len(aligned_coverages), 2, n_ages, n_years
    ), f"coverage_rates shape {coverage_rates.shape} != (n_cov, 2, n_ages, n_years)"

    # Expense primitives -- the five inputs every value-side kernel
    # consumes (alpha / beta / gamma scalars plus two per-month curves).
    # See projection._expense_kernel_args -- this engine path uses the
    # same helper so the item-form vs legacy dispatch is consistent across
    # the fast path (full=False) and the full path (full=True).
    from fastcashflow.projection import _expense_kernel_args
    (expense_alpha_pro_rata, expense_alpha_fixed, expense_beta_pro_rata,
     gamma_fixed, lae_pro_rata) = _expense_kernel_args(
        basis, n_time,
    )
    if discount_curve is None:
        discount_bom, discount_mid = discount_factors(basis, n_time)
    else:
        discount_curve = np.asarray(discount_curve, dtype=np.float64)
        if discount_curve.shape != (n_time,):
            raise ValueError(
                f"discount_curve must have shape ({n_time},) -- one annual "
                f"rate per projection month -- got {discount_curve.shape}"
            )
        monthly_curve = (1.0 + discount_curve) ** (1.0 / 12.0) - 1.0
        discount_bom, discount_mid = discount_factors_from_curve(monthly_curve)
    z = _norm_ppf(basis.ra_confidence)
    mortality_factor = z * basis.mortality_cv
    morbidity_factor = z * basis.morbidity_cv
    longevity_factor = z * basis.longevity_cv
    disability_factor = z * basis.disability_cv

    # A claims settlement pattern discounts claims to their payment dates;
    # scaling the coverage amounts carries that into the fused kernel.
    coverage_amount = model_points.coverage_amount
    if basis.settlement_pattern is not None:
        coverage_amount = coverage_amount * _settlement_factor(
            basis.settlement_pattern, basis.discount_monthly
        )

    # Surrender curve, padded to n_time and zero-filled when absent. Kept as
    # an always-present (n_time,) array so the kernels do not need a branch:
    # ``surrender_curve[t]`` is read once per month, and is zero whenever no
    # surrender mechanic applies.
    surr_user = basis.surrender_value_curve
    if surr_user is None:
        surrender_curve_kernel = np.zeros(n_time, dtype=np.float64)
    else:
        c = np.asarray(surr_user, dtype=np.float64)
        idx = np.minimum(np.arange(n_time), c.shape[0] - 1)
        surrender_curve_kernel = c[idx]

    # Feature flags -- skip the per-cell work of any cash-flow stream the
    # portfolio does not use. The scalar fast path branches on them; the
    # codegen path bakes them into the generated source. Cheap O(n) scans
    # once per call, off the hot loop. Shared by both paths below.
    use_surrender = bool(np.any(surrender_curve_kernel != 0.0))
    use_annuity = bool(np.any(model_points.annuity_payment != 0.0))
    use_lae = bool(np.any(lae_pro_rata != 0.0))
    use_morbidity = bool(np.any(coverage_risk != 0))
    use_disability = bool(np.any(model_points.disability_income != 0.0))

    if fast_path:
        survival_monthly = np.ascontiguousarray(
            (1.0 - mortality_grid) * (1.0 - lapse_grid)
        )
        bel, ra, csm, loss_component = _fast_kernel_scalar(
            issue_index,
            model_points.sex,
            model_points.term_months,
            model_points.count,
            model_points.level_premium,
            model_points.single_premium,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.coverage_index,
            coverage_amount,
            model_points.coverage_offset,
            coverage_rates,
            coverage_risk,
            coverage_is_diagnosis,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            expense_alpha_pro_rata,
            expense_alpha_fixed,
            expense_beta_pro_rata,
            gamma_fixed,
            lae_pro_rata,
            discount_bom,
            discount_mid,
            mortality_factor,
            morbidity_factor,
            longevity_factor,
            model_points.coverage_waiting,
            model_points.coverage_reduction_end,
            model_points.coverage_reduction_factor,
            survival_monthly,
            lapse_grid,
            surrender_curve_kernel,
            use_morbidity,
            use_annuity,
            use_lae,
            use_surrender,
        )
        return GMMMeasurement(bel=bel, ra=ra, csm=csm, loss_component=loss_component)

    # The CPU kernel takes (n_states, n_edges) via Python closure -- they are
    # not in the args tuple. The GPU kernel still takes n_states explicitly
    # as a runtime arg, so the two paths build the call list separately.
    n_edges = int(edge_from.shape[0])
    common_args = (
        edge_from,
        edge_to,
        edge_prob,
        edge_lump_sum,
        premium_state,
        benefit_state,
        start_state,
        issue_index,
        model_points.sex,
        model_points.term_months,
        model_points.count,
        model_points.level_premium,
        model_points.single_premium,
        model_points.premium_term_months,
        model_points.premium_frequency_months,
        model_points.annuity_frequency_months,
        model_points.coverage_index,
        coverage_amount,
        model_points.coverage_offset,
        coverage_rates,
        coverage_risk,
        coverage_is_diagnosis,
        model_points.maturity_benefit,
        model_points.annuity_payment,
        model_points.disability_income,
        model_points.disability_benefit,
        expense_alpha_pro_rata,
        expense_alpha_fixed,
        expense_beta_pro_rata,
        gamma_fixed,
        lae_pro_rata,
        discount_bom,
        discount_mid,
        mortality_factor,
        morbidity_factor,
        longevity_factor,
        disability_factor,
    )

    if backend == "cpu":
        if state_duration_max is not None:
            # Phase (c) semi-Markov path. The kernel takes a thinner arg
            # tuple than the Markov codegen -- the edge topology and the
            # per-state cohort counts are baked into the generated source
            # (one cache file per unique StateModel + duration shape).
            # Coverage-rule and diagnosis-coverage passes are emitted
            # alongside the main pass so contracts mixing semi-Markov
            # cohort tracking with rule-bearing or diagnosis coverages work
            # in a single measure(full=False) call.
            kernel = _get_fast_kernel_codegen_semi_markov(
                n_states, state_duration_max, edge_from, edge_to,
                edge_lump_sum, premium_state, benefit_state,
                use_annuity=use_annuity, use_lae=use_lae,
                use_surrender=use_surrender,
            )
            bel, ra, csm, loss_component = kernel(
                edge_prob, start_state, issue_index,
                model_points.sex,
                model_points.term_months,
                model_points.count,
                model_points.level_premium,
                model_points.single_premium,
                model_points.premium_term_months,
                model_points.premium_frequency_months,
                model_points.annuity_frequency_months,
                model_points.coverage_index,
                coverage_amount,
                model_points.coverage_offset,
                coverage_rates,
                coverage_risk,
                coverage_is_diagnosis,
                model_points.maturity_benefit,
                model_points.annuity_payment,
                model_points.disability_income,
                model_points.disability_benefit,
                expense_alpha_pro_rata,
                expense_alpha_fixed,
                expense_beta_pro_rata,
                gamma_fixed,
                lae_pro_rata,
                discount_bom,
                discount_mid,
                mortality_factor,
                morbidity_factor,
                longevity_factor,
                disability_factor,
                model_points.coverage_waiting,
                model_points.coverage_reduction_end,
                model_points.coverage_reduction_factor,
                lapse_grid,
                surrender_curve_kernel,
            )
        else:
            # Markov path -- every multi-state model with no duration
            # tracking. The closure factory and the hand-unrolled
            # n_states=2 / n_states=3 kernels stay in the file as a
            # readable reference but are no longer on the default path.
            kernel = _get_fast_kernel_codegen(
                n_states, edge_from, edge_to, edge_lump_sum,
                premium_state, benefit_state,
                use_morbidity=use_morbidity, use_annuity=use_annuity,
                use_disability=use_disability, use_lae=use_lae,
                use_surrender=use_surrender,
            )
            bel, ra, csm, loss_component = kernel(
                *common_args, model_points.coverage_waiting,
                model_points.coverage_reduction_end,
                model_points.coverage_reduction_factor,
                lapse_grid,
                surrender_curve_kernel,
            )
    elif backend == "gpu":
        if state_duration_max is not None:
            raise NotImplementedError(
                "measure(full=False, backend='gpu') does not support semi-Markov "
                "StateModels yet; use backend='cpu'"
            )
        if np.any(model_points.coverage_waiting) or np.any(model_points.coverage_reduction_end):
            raise ValueError(
                "measure(full=False, backend='gpu') does not support coverage waiting / "
                "reduction periods yet; use backend='cpu'"
            )
        from fastcashflow._gpu import fast_gpu
        bel, ra, csm, loss_component = fast_gpu(
            common_args[0], common_args[1], common_args[2], common_args[3],
            n_states, *common_args[4:],
            lapse_grid, surrender_curve_kernel,
        )
    else:
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")

    return GMMMeasurement(bel=bel, ra=ra, csm=csm, loss_component=loss_component)


def _measure_segmented(
    model_points: ModelPoints,
    basis: dict[tuple[str, str], Basis],
    *,
    backend: str = "cpu",
    discount_curve: FloatArray | None = None,
    segment_by=("product_code", "channel_code"),
) -> GMMMeasurement:
    """Value a multi-segment portfolio: split, value each, concatenate.

    ``basis`` is the ``{(product_code, channel_code): Basis}`` dictionary
    returned by :func:`fastcashflow.read_basis`. ``model_points``
    must carry ``product_code`` and ``channel_code`` columns identifying each row's
    segment; for each unique (product_code, channel_code) the helper masks the
    matching rows, builds a sub-:class:`~fastcashflow.ModelPoints` via
    :meth:`~fastcashflow.ModelPoints.subset`, calls ``measure(..., full=False)`` with the
    segment's ``Basis``, and writes the per-row results back to a
    single ``(n_mp,)`` :class:`GMMMeasurement`.

    ``backend`` and ``discount_curve`` flow through to ``measure(..., full=False)`` --
    declared explicitly so a typo (e.g. ``backed="gpu"``) is rejected
    here rather than reaching the kernel. A single-segment ``basis`` is
    accepted as a convenience when ``product_code`` / ``channel_code`` is
    not set.
    """
    try:
        cols = _resolve_segment_cols(model_points, segment_by)
    except KeyError:
        if len(basis) == 1:
            (basis,) = basis.values()
            return _measure_fast(
                model_points, basis,
                backend=backend, discount_curve=discount_curve,
            )
        raise ValueError(
            f"model_points has no {tuple(segment_by)} axis/axes set but the "
            f"basis has {len(basis)} segments; either set the columns or "
            "pass a single-segment basis"
        )
    basis_norm, segments = _factorise_segments(
        basis, cols, segment_by, model_points.n_mp,
    )

    n_mp = model_points.n_mp
    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    loss_component = np.empty(n_mp)
    for key, idx in segments:
        sub = model_points.subset(idx)
        val = _measure_fast(
            sub, basis_norm[key],
            backend=backend, discount_curve=discount_curve,
        )
        bel[idx] = val.bel
        ra[idx] = val.ra
        csm[idx] = val.csm
        loss_component[idx] = val.loss_component

    return GMMMeasurement(bel=bel, ra=ra, csm=csm, loss_component=loss_component)


def _resolve_segment_cols(model_points: ModelPoints, segment_by) -> list[np.ndarray]:
    """Resolve and NFC-normalise the segment-key axes (``segment_by`` names).

    Each axis is looked up via :meth:`ModelPoints.axis`, so a routing key can mix
    the segment fields (product_code / channel) and any ``attributes`` column.
    NFC-normalises so the lookup is text-identity, not byte-identity (a Korean /
    European character composed in one file and decomposed in the other compares
    unequal). Raises :class:`KeyError` if an axis is not set on the model points
    -- the caller turns that into the single-basis convenience or a clear error.
    """
    cols = []
    for name in segment_by:
        col = model_points.axis(name)
        cols.append(np.array(
            [unicodedata.normalize("NFC", str(v)) for v in col], dtype=object,
        ))
    return cols


def _factorise_segments(basis, cols, segment_by, n_mp):
    """Resolve a multi-segment portfolio to per-segment row indices.

    ``cols`` are the NFC-normalised axis arrays (one per ``segment_by`` name);
    ``basis`` is ``{key: Basis}`` keyed by a tuple of those axes in order (a bare
    value is accepted for a one-axis key). Returns ``(basis_norm, segments)``
    where ``basis_norm`` is ``basis`` re-keyed under NFC-normalised tuples and
    ``segments`` is ``[(key, idx)]`` in first-seen order -- one per segment
    present in the model points. Cost scales with the number of distinct
    segments, not the number of axes: the ``'|'``-join + ``np.unique`` is a
    single ``O(n_mp)`` factorisation regardless of how many axes the key spans.
    """
    basis_norm = {}
    for k, a in basis.items():
        parts = k if isinstance(k, tuple) else (k,)
        basis_norm[tuple(unicodedata.normalize("NFC", str(p)) for p in parts)] = a
    # Segment keys are joined with '|' for factorisation; reject codes that
    # contain that separator to keep the round-trip lossless.
    for col, name in zip(cols, segment_by):
        bad = sorted({str(v) for v in col if "|" in str(v)})
        if bad:
            raise ValueError(
                f"{name} value(s) {bad} contain the '|' character, which the "
                "segmented measure uses as the segment-key separator. Pick a "
                "different separator in your ETL or rename the offending code."
            )
    keys_arr = np.array(
        ["|".join(str(col[i]) for col in cols) for i in range(n_mp)], dtype=object,
    )
    # Preserve first-seen order so debugging output reads top-to-bottom of the
    # input (np.unique returns sorted; re-index by first occurrence).
    unique_keys, first_seen, inverse = np.unique(
        keys_arr, return_index=True, return_inverse=True,
    )
    order = np.argsort(first_seen)
    segments: list[tuple[tuple, np.ndarray]] = []
    for ord_idx in order:
        key = tuple(str(unique_keys[ord_idx]).split("|"))
        if key not in basis_norm:
            raise ValueError(
                f"segment {key!r} appears in model_points but is not in the "
                f"basis (known segments: {sorted(basis_norm)})"
            )
        idx = np.nonzero(inverse == ord_idx)[0]
        segments.append((key, idx))
    return basis_norm, segments


def _measure_segmented_full(
    model_points: ModelPoints, basis: dict[tuple[str, str], Basis],
    *, segment_by=("product_code", "channel_code"),
) -> GMMMeasurement:
    """Full multi-segment GMM measurement -- per-segment trajectories stitched.

    Each (product_code, channel_code) segment is measured under its own
    ``Basis`` via :func:`_measure_full`; the per-segment ``(n_seg, *)``
    trajectories are scattered back into one ``(n_mp, n_time+1)`` result, where
    ``n_time`` is the portfolio's longest horizon. A segment whose contracts
    mature earlier is zero-padded on the right -- a contract carries no BEL /
    RA / CSM past its term. ``discount_bom`` / ``discount_mid`` are per-MP
    ``(n_mp, ...)`` here, not the single ``(n_time+1,)`` curve of the
    single-basis path: segments discount on different curves, so the rate is a
    property of the row. The padded tail of ``discount_bom`` repeats each
    row's last factor (a flat curve -> zero forward rate) so a rate read off it
    is finite, not a 0/0.
    """
    try:
        cols = _resolve_segment_cols(model_points, segment_by)
    except KeyError:
        if len(basis) == 1:
            (basis,) = basis.values()
            return _measure_full(model_points, basis)
        raise ValueError(
            f"model_points has no {tuple(segment_by)} axis/axes set but the "
            f"basis has {len(basis)} segments; either set the columns or "
            "pass a single-segment basis"
        )
    basis_norm, segments = _factorise_segments(
        basis, cols, segment_by, model_points.n_mp,
    )
    n_mp = model_points.n_mp

    sub_results = [(idx, _measure_full(model_points.subset(idx), basis_norm[key]))
                   for key, idx in segments]
    n_time = max(m.bel_path.shape[1] - 1 for _, m in sub_results)

    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    loss_component = np.empty(n_mp)
    bel_path = np.zeros((n_mp, n_time + 1))
    ra_path = np.zeros((n_mp, n_time + 1))
    csm_path = np.zeros((n_mp, n_time + 1))
    lic = np.zeros((n_mp, n_time + 1))
    csm_accretion = np.zeros((n_mp, n_time))
    csm_release = np.zeros((n_mp, n_time))
    discount_bom = np.ones((n_mp, n_time + 1))
    discount_mid = np.ones((n_mp, n_time))

    cf_2d = ("inforce", "deaths", "premium_cf", "claim_cf", "morbidity_cf",
             "expense_cf", "annuity_cf", "disability_cf", "surrender_cf")
    cf_arrays = {name: np.zeros((n_mp, n_time)) for name in cf_2d}
    maturity_cf = np.zeros(n_mp)
    maturity_survivors = np.zeros(n_mp)

    for idx, m in sub_results:
        t = m.bel_path.shape[1] - 1
        bel[idx] = m.bel
        ra[idx] = m.ra
        csm[idx] = m.csm
        loss_component[idx] = m.loss_component
        bel_path[idx, :t + 1] = m.bel_path
        ra_path[idx, :t + 1] = m.ra_path
        csm_path[idx, :t + 1] = m.csm_path
        lic[idx, :t + 1] = m.lic
        csm_accretion[idx, :t] = m.csm_accretion
        csm_release[idx, :t] = m.csm_release
        # Per-MP discount: lay the segment's curve, then flat-fill the tail so
        # the padded months read a zero forward rate, not a 0/0.
        discount_bom[idx, :t + 1] = m.discount_bom
        discount_bom[idx, t + 1:] = m.discount_bom[-1]
        discount_mid[idx, :t] = m.discount_mid
        if t < n_time:
            discount_mid[idx, t:] = m.discount_mid[-1] if t > 0 else 1.0
        cf = m.cashflows
        for name in cf_2d:
            arr = getattr(cf, name)
            cf_arrays[name][idx, :arr.shape[1]] = arr
        maturity_cf[idx] = cf.maturity_cf
        maturity_survivors[idx] = cf.maturity_survivors

    cashflows = type(sub_results[0][1].cashflows)(
        maturity_cf=maturity_cf, maturity_survivors=maturity_survivors, **cf_arrays,
    )
    return GMMMeasurement(
        bel=bel, ra=ra, csm=csm, loss_component=loss_component,
        bel_path=bel_path, ra_path=ra_path, csm_path=csm_path,
        csm_accretion=csm_accretion, csm_release=csm_release, lic=lic,
        cashflows=cashflows, discount_bom=discount_bom, discount_mid=discount_mid,
    )
