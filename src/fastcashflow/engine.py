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
import math
import os
import sys
import unicodedata
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow._measurement_basis import (
    MEASUREMENT_BASIS_HYPOTHETICAL,
    MEASUREMENT_BASIS_INCEPTION,
    MEASUREMENT_BASIS_SETTLEMENT_CARRY,
    _inforce_marker_columns,
)
from fastcashflow.basis import (
    Basis, BasisRouter, annual_to_monthly, _single_basis, validate_factor,
    SURRENDER_VALUE_BASES,
)
from fastcashflow.curves import (
    discount_factors,
    discount_factors_from_curve,
    discount_monthly_curve,
    forward_rates,
)
from fastcashflow.numerics import (
    _cost_of_capital_ra,
    _csm_kernel,
    _carry_lic_residual,
    _norm_ppf,
    _paragraph45_csm_algebra,
    _risk_adjustment,
    _rollforward_kernel,
    _settlement_factor,
    _settlement_lic,
    _settlement_lic_discounted,
)
from fastcashflow.coverage import (
    align_coverages, build_coverage_rates, coverage_arrays, validate_csr_codes,
)
from fastcashflow.io import write_measurement, _write_measurement_columns
from fastcashflow.model_points import ModelPoints
from fastcashflow.projection import (
    Cashflows, project_cashflows, reject_account_book,
    _add_state_mortality_rates, _state_lapse_stack,
)
from fastcashflow.state_model import (
    compile_state_model,
    compile_state_model_with_duration,
    is_semi_markov,
    model_references_rate,
    needs_state_machine,
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
    # The terminal column holds the residual of claims whose settlement tail
    # runs past the horizon (stays non-zero by design, not a leak).
    cashflows: "Cashflows | None" = None
    # bom = beginning of month, mom = mid of month: discount factors for a flow
    # at the start vs the middle of each month. Shape (n_time+1,) / (n_time,)
    # for a single basis; (n_mp, n_time+1) / (n_mp, n_time) when measured under
    # a per-segment basis dict, where each row discounts on its own curve.
    discount_factor_bom: FloatArray | None = None  # beginning-of-month discount factors
    discount_factor_mid: FloatArray | None = None  # mid-of-month discount factors
    # Source model points, stamped by ``measure`` so ``group(m, by=[...])`` can
    # resolve axis names without re-passing them. A reference, not a copy; None
    # on a grouped result (its rows are groups, not model points).
    model_points: "ModelPoints | None" = None
    # The per-group composite label (one per row) on a result returned by
    # ``group`` / ``group_of_contracts``; None on a per-model-point measurement.
    group_labels: "np.ndarray | None" = None
    # The number of model points in each group, aligned with ``group_labels``.
    group_sizes: IntArray | None = None
    # Time basis of the result (see _measurement_basis): 'inception' for
    # new-business measure(); in-force results re-base the headline to the
    # valuation date while the trajectories stay on the inception axis, so
    # inception-axis consumers (group / roll_forward / report / transition /
    # plot_*) reject anything else via _require_inception.
    measurement_basis: str = MEASUREMENT_BASIS_INCEPTION

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm),
                ("loss", self.loss_component)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr("GMMMeasurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str("GMMMeasurement", self._columns())


@dataclass(frozen=True, slots=True, eq=False)
class GMMAggregate:
    """Portfolio-aggregate GMM trajectories -- the scalable ``full=True`` view.

    BEL / RA / CSM are additive across contracts, so a large book's liability
    run-off is its per-model-point trajectories *summed over the model-point
    axis*. This holds only that sum: the scalar inception totals plus the
    ``(n_time+1,)`` aggregate ``bel_path`` / ``ra_path`` / ``csm_path`` (column 0
    is inception, so ``bel == bel_path[0]``). It is what
    :func:`~fastcashflow.gmm.measure_aggregate` returns, computed in bounded
    memory so it works where a per-model-point ``measure(full=True)`` would OOM.
    """

    bel: float                   # portfolio inception BEL total
    ra: float                    # portfolio inception RA total
    csm: float                   # portfolio inception CSM total
    loss_component: float        # portfolio inception loss-component total
    bel_path: FloatArray         # (n_time+1,) -- aggregate BEL trajectory
    ra_path: FloatArray          # (n_time+1,) -- aggregate RA trajectory
    csm_path: FloatArray         # (n_time+1,) -- aggregate CSM trajectory


@write_measurement.register
def _(measurement: GMMMeasurement, path, *, ids=None):
    cols = {"bel": measurement.bel, "ra": measurement.ra,
            "csm": measurement.csm,
            "loss_component": measurement.loss_component}
    # In-force output gets marker columns so it stays distinguishable from
    # new-business output at the file boundary; inception output is unchanged.
    cols.update(_inforce_marker_columns(measurement, measurement.bel.shape[0]))
    _write_measurement_columns(cols, path, ids)


def _compute_csm(bel0, ra0, inforce, discount_monthly, discount_units=False):
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
    csm, accretion, release = _csm_kernel(csm0, inforce, discount_monthly,
                                          discount_units)
    return csm, accretion, release, loss_component


def _account_risk_adjustment(model_points, basis, proj, discount_monthly):
    """Universal-life risk adjustment -- priced on the net amount at risk.

    The insurance risk of an account-backed death leg is the mortality borne on
    the NET AMOUNT AT RISK (the death benefit above the account,
    ``deaths * max(0, face - av_mid)``) -- the account portion returns the
    policyholder's own money and bears no insurance risk -- plus expense risk,
    plus the morbidity risk of any cost-deducting rider (a fixed health benefit
    funded from the account). This BYPASSES :func:`_risk_adjustment` and its
    ``expense_cv != 0`` guard (a UL RA legitimately prices ``expense_cv``): run
    the at-risk claim, the morbidity claim and the expense through one
    roll-forward pass, then the confidence margin ``z(ra_confidence) *
    (mortality_cv*pv_nar + morbidity_cv*pv_morbidity + expense_cv*pv_expense)``,
    cost-of-capital-wrapped per ``ra_method``.
    """
    face = model_points.minimum_death_benefit
    n_mp, n_time = proj.claim_cf.shape
    zeros_t = np.zeros((n_mp, n_time))
    zeros_mp = np.zeros(n_mp)
    nar_claim = np.ascontiguousarray(
        proj.deaths * np.maximum(0.0, face[:, None] - proj.account.av_mid))
    # The annuity payout (an annuitizing UL contract, phase 2) bears longevity
    # risk -- the insurer pays the income for as long as the annuitant lives --
    # so its PV is priced through longevity_cv, alongside the at-risk mortality
    # and expense. The annuity stream rides the survival slot of the
    # roll-forward (position 6); a non-annuitizing account book has annuity_cf
    # == 0, so pv_annuity == 0 and this term vanishes (byte-identical). The
    # account maturity lump is the return of the policyholder's own balance (an
    # investment component) and bears no insurance risk, so it is deliberately
    # NOT longevity-priced (it stays out, the maturity slot is zero here).
    # The morbidity claim of a cost-deducting rider (funds from the account, but
    # pays a fixed health benefit -- not the balance) bears morbidity risk; it
    # rides the DISABILITY slot of the roll-forward (position 3, otherwise empty
    # for an account book) purely to harvest its PV. A book with no such rider
    # has morbidity_cf == 0, so pv_morbidity == 0 and the term vanishes
    # (byte-identical). expense_cf rides the morbidity slot for the same reason.
    _, pv_nar, pv_expense, pv_morbidity, pv_annuity = _rollforward_kernel(
        nar_claim, proj.expense_cf, proj.morbidity_cf, zeros_t, zeros_t,
        proj.annuity_cf, zeros_mp, zeros_t,
        model_points.contract_boundary_months, discount_monthly)
    z = _norm_ppf(basis.ra_confidence)
    confidence_margin = z * (basis.mortality_cv * pv_nar
                             + basis.morbidity_cv * pv_morbidity
                             + basis.expense_cv * pv_expense
                             + basis.longevity_cv * pv_annuity)
    if basis.ra_method == "cost_of_capital":
        return _cost_of_capital_ra(
            confidence_margin, discount_monthly, basis.cost_of_capital_rate)
    return confidence_margin


def _measure_full(model_points: ModelPoints, basis: Basis, *,
                  discount_monthly: FloatArray | None = None) -> GMMMeasurement:
    """Full GMM measurement: BEL, RA and CSM rolled forward over time.

    Returns a :class:`GMMMeasurement` carrying both the ``(n_mp,)`` inception
    headline (column 0 of each trajectory) and the ``(n_mp, n_time+1)``
    ``*_path`` trajectories. Reached by ``measure(..., full=True)``.

    ``discount_monthly`` overrides the discount / CSM-accretion curve (default: the
    locked-in ``discount_monthly_curve``). ``vfa.measure`` passes the flat
    underlying-items return here to measure a universal-life account book under
    the VFA model -- the account roll (generation) is identical to GMM, only the
    discount rate differs. The override is only used by the account path, which
    carries no ``settlement_pattern``, so the settlement factor below (keyed on
    ``basis.discount_monthly``) is never reached together with an override.
    """
    proj = project_cashflows(model_points, basis)
    claim_cf, morbidity_cf = proj.claim_cf, proj.morbidity_cf
    if discount_monthly is None:
        discount_monthly = discount_monthly_curve(basis, proj.n_time)
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
    discount_factor_bom, discount_factor_mid = discount_factors_from_curve(discount_monthly)

    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _rollforward_kernel(
        claim_cf, morbidity_cf, proj.disability_cf, proj.expense_cf,
        proj.premium_cf, proj.annuity_cf, proj.maturity_cf, proj.surrender_cf,
        model_points.contract_boundary_months, discount_monthly,
    )
    if proj.account is not None:
        # Universal-life account-backed measurement. The BEL nets the account
        # value the entity holds (fund) -- premium is the lone gross inflow
        # (counted once in the roll-forward), and the account it builds is held
        # as fund and subtracted ONCE post-PV. The RA prices the mortality risk
        # on the NET AMOUNT AT RISK (the death benefit above the account) plus
        # expense risk, bypassing the slot-RA machinery (which hard-raises on
        # expense_cv and would price mortality on the full death benefit).
        bel = bel - proj.account.fund
        ra = _account_risk_adjustment(model_points, basis, proj, discount_monthly)
    else:
        ra = _risk_adjustment(basis, pv_claims, pv_morbidity, pv_disability,
                              pv_survival, discount_monthly)
    csm, csm_accretion, csm_release, loss_component = _compute_csm(
        bel[:, 0], ra[:, 0], proj.inforce, discount_monthly,
        basis.coverage_unit_discount,
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
        discount_factor_bom=discount_factor_bom,
        discount_factor_mid=discount_factor_mid,
    )


def _require_full(measurement, entry: str) -> None:
    """Raise if a headline-only (full=False) measurement reaches a path that
    needs the ``*_path`` trajectories. Shared by report / roll_forward / group /
    transition so they give one consistent message instead of four near-copies.
    """
    if measurement.bel_path is None:
        raise ValueError(
            f"{entry} requires a full=True measurement; the trajectory fields "
            f"are None on the full=False fast path. Call measure(..., full=True)."
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
    trajectories. Those trajectories make it **memory-bound** -- several dense
    ``(n_mp, n_time+1)`` float64 arrays, on the order of ~100 KB per model point
    for a long horizon, so a million-policy ``full=True`` run needs ~100 GB and
    will OOM on a typical box. ``full=False`` is the fused, memory-minimal fast
    path -- it fills only the headline (``*_path`` are ``None``) at a few hundred
    bytes per model point, and is the right choice for large-scale valuation;
    reserve ``full=True`` for movement analysis or per-segment / chunked runs.

    ``basis`` may be a single :class:`Basis` (uniform portfolio) or a
    ``{(product, channel): Basis}`` dict; with a dict each segment is routed
    to its own basis. ``segment_by`` names the routing axes (resolved via
    :meth:`ModelPoints.axis`, so any ``attributes`` column works) and the dict
    keys are tuples of those axes in order. Left as ``None`` (the default) it is
    taken from the basis: a :class:`~fastcashflow.io.BasisRouter` from
    :func:`read_basis` carries the axes its workbook declared, and a plain dict
    falls back to ``("product", "channel")``. So a workbook keyed by
    ``(product, channel, risk_class)`` routes by all three with no
    extra argument; passing ``segment_by`` explicitly overrides. Cost scales with the number of distinct segments, not the
    number of axes. ``backend`` (``"cpu"``/``"gpu"``) and ``discount_curve``
    apply to the fast path only.
    """
    if not isinstance(basis, (Basis, BasisRouter)):
        raise TypeError(
            "basis must be a Basis or a BasisRouter (from read_basis); got "
            f"{type(basis).__name__}"
        )
    # A variable annuity payout (a finite annuity_air_annual on an annuitizing
    # row) re-floats the phase-2 income at the realised fund return; only a
    # direct-participation (VFA) discount equals that fund return and makes the
    # re-float meaningful. Under the GMM locked-in discount the fund-linked
    # payout would be valued at an unrelated rate -- reject it here rather than
    # return a meaningless number. Measure it through vfa.measure.
    if np.any(np.isfinite(model_points.annuity_air_annual)
              & (model_points.annuitization_months > 0)):
        raise NotImplementedError(
            "a variable annuity payout (a finite annuity_air_annual) is a "
            "direct-participation feature -- measure it through vfa.measure, "
            "not gmm.measure (the GMM locked-in discount cannot value a "
            "fund-linked payout that re-floats at the fund return).")
    if isinstance(basis, BasisRouter):
        # A BasisRouter remembers its axes, so measure routes without a
        # segment_by; an explicit segment_by wins.
        if segment_by is None:
            segment_by = basis.segment_axes
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


def measure_aggregate(
    model_points: ModelPoints,
    basis: "Basis | dict[tuple[str, str], Basis]",
    *,
    chunk_size: int = 200_000,
) -> GMMAggregate:
    """Portfolio-aggregate ``full=True`` measurement in bounded memory.

    ``measure(full=True)`` materialises dense ``(n_mp, n_time+1)`` trajectories
    -- ~100 KB per model point -- so a million-policy book needs ~100 GB and
    OOMs. But BEL / RA / CSM are additive across contracts, so the portfolio's
    liability run-off is the per-model-point trajectories summed over the
    model-point axis. This runs the full trajectory kernel over row-blocks of
    ``chunk_size`` model points and accumulates only that ``(n_time+1,)`` sum,
    so peak memory is ``O(chunk_size x n_time)`` regardless of ``n_mp``.

    Returns an :class:`GMMAggregate` (scalar totals + aggregate
    ``bel_path`` / ``ra_path`` / ``csm_path``). For the per-model-point detail
    (movement, in-force slicing) use :func:`measure` on a book small enough to
    hold every trajectory. ``basis`` may be a single :class:`Basis` or a
    per-segment dict, routed per chunk exactly as :func:`measure` routes it.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    n_mp = int(model_points.issue_age.shape[0])
    # The global horizon: a chunk projects only to its own boundary.max(), so
    # its (shorter) aggregate path is added to the leading slice of the global
    # one -- correct because a contract's trajectory is zero past its boundary.
    n_time = int(np.asarray(model_points.contract_boundary_months).max())
    bel_path = np.zeros(n_time + 1)
    ra_path = np.zeros(n_time + 1)
    csm_path = np.zeros(n_time + 1)
    bel = ra = csm = loss = 0.0
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        m = measure(model_points.subset(idx), basis, full=True)
        nt = m.bel_path.shape[1]
        bel_path[:nt] += m.bel_path.sum(axis=0)
        ra_path[:nt] += m.ra_path.sum(axis=0)
        csm_path[:nt] += m.csm_path.sum(axis=0)
        bel += float(m.bel.sum())
        ra += float(m.ra.sum())
        csm += float(m.csm.sum())
        loss += float(m.loss_component.sum())
    return GMMAggregate(
        bel=bel, ra=ra, csm=csm, loss_component=loss,
        bel_path=bel_path, ra_path=ra_path, csm_path=csm_path,
    )


# ---------------------------------------------------------------------------
# In-force subsequent measurement
# ---------------------------------------------------------------------------

def _inforce_rescale(m, model_points, em, rows) -> FloatArray:
    """Per-MP factor that re-bases an inception-run projection to the valuation
    date: ``count / inforce[em] = 1 / survival(0->em)``.

    The in-force projection runs from inception, so ``inforce[em] = count x
    survival(0->em)`` -- it decrements the as-of ``count`` again from inception.
    Scaling the sliced ``bel`` / ``ra`` by this factor makes the as-of in-force
    exactly the input ``count``; it is exact for every cash flow linear in the
    in-force. Where ``inforce[em]`` is zero (a fully run-off cohort) the bel is
    already zero, so the factor is 1 (a no-op).
    """
    inforce_em = m.cashflows.inforce[rows, em]
    safe = np.where(inforce_em > 0.0, inforce_em, 1.0)
    count = np.asarray(model_points.count, dtype=np.float64)
    return np.where(inforce_em > 0.0, count / safe, 1.0)


def _measure_inforce_fast(
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
    # The as-of date must lie strictly within each contract's own Sec. 34
    # boundary. The BEL / RA trajectory and the in-force only extend to
    # t = contract_boundary_months (== term_months when no boundary cut); at or
    # beyond the boundary there is no remaining coverage to value, and
    # _inforce_rescale's ``inforce[rows, em]`` would read a stale zero (em within
    # the padded width) or index out of bounds (em == the widest boundary).
    # boundary is backfilled to term in ModelPoints.__post_init__, so it is never
    # None here and is <= term_months by construction.
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    runoff = em >= boundary
    if np.any(runoff):
        bad = int(np.argmax(runoff))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} >= "
            f"contract_boundary_months[{bad}]={int(boundary[bad])} (the Sec. 34 "
            "horizon; equal to term_months when no boundary cut); the contract "
            "has no remaining coverage at the valuation date. measure_inforce "
            "needs an as-of date strictly before the contract boundary."
        )
    rows = np.arange(n_mp)
    # Re-base the inception-run projection to the valuation date (see
    # _inforce_rescale): exact for cash flows linear in the in-force.
    rescale = _inforce_rescale(m, model_points, em, rows)
    bel = m.bel_path[rows, em] * rescale
    ra = m.ra_path[rows, em] * rescale
    if not settlement_mode:
        # Hypothetical: take the engine-computed CSM trajectory at t=elapsed.
        csm = m.csm_path[rows, em]
        return GMMMeasurement(
            bel=bel, ra=ra, csm=csm, loss_component=m.loss_component,
            measurement_basis=MEASUREMENT_BASIS_HYPOTHETICAL,
        )

    # Settlement carry-forward: roll the prior closing CSM one period over
    # the coverage units from t = em - period_months to t = em.
    prior_csm = np.asarray(prior_csm, dtype=np.float64)
    if prior_csm.shape != (n_mp,):
        raise ValueError(
            f"prior_csm must have shape ({n_mp},), got {prior_csm.shape}"
        )
    period_months = int(period_months) if period_months is not None else 12
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
    csm_traj, _, _ = _csm_kernel(prior_csm, inforce_seg, monthly_rates,
                                 basis.coverage_unit_discount)
    csm = csm_traj[:, period_months]
    # Sec. 44 loss component is left as zeros here. v1 only rolls the prior
    # CSM forward (accretion + coverage-unit release); the unlocking that
    # would actually drive the carried CSM negative -- assumption changes
    # and experience variances over the period -- is roll_forward()'s job.
    # Returning max(0, bel + ra - csm) would conflate "carried CSM is short"
    # with "true onerous recognition" and mis-signal a Sec. 44 hit.
    loss = np.zeros(n_mp, dtype=np.float64)
    return GMMMeasurement(bel=bel, ra=ra, csm=csm, loss_component=loss,
                          measurement_basis=MEASUREMENT_BASIS_SETTLEMENT_CARRY)


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


def _measure_inforce_full(
    model_points: ModelPoints,
    basis: Basis,
    *,
    prior_csm: FloatArray | None = None,
    lock_in_rate: float | None = None,
    period_months: int | None = None,
) -> GMMMeasurement:
    """In-force subsequent measurement -- full-trajectory variant of
    :func:`_measure_inforce_fast`.

    Calls :func:`measure` to build the BEL / RA / CSM trajectories from
    inception. The two modes mirror :func:`_measure_inforce_fast`:

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
      reason as :func:`_measure_inforce_fast`: Sec. 44 onerous recognition is
      only meaningful with CSM unlocking, which v1 does not perform.

    Use this when the downstream needs a full trajectory (movement
    decomposition, period-close roll-forward) rather than just the
    valuation-date headline numbers that :func:`_measure_inforce_fast` returns.
    """
    settlement_mode = _validate_settlement_args(
        prior_csm, lock_in_rate, period_months,
    )
    m = _measure_full(model_points, basis)
    if not settlement_mode:
        # Hypothetical mode returns the measure() result re-tagged: the
        # trajectory is what a freshly issued contract would produce, seated
        # mid-life -- a what-if, not an inception or settlement figure.
        return replace(m, measurement_basis=MEASUREMENT_BASIS_HYPOTHETICAL)

    prior_csm = np.asarray(prior_csm, dtype=np.float64)
    n_mp = m.bel.shape[0]
    if prior_csm.shape != (n_mp,):
        raise ValueError(
            f"prior_csm must have shape ({n_mp},), got {prior_csm.shape}"
        )
    period_months = int(period_months) if period_months is not None else 12
    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    # Guard on the contract boundary, not term: the projected trajectory and
    # in-force only extend to t = contract_boundary_months (Sec. 34; == term when
    # no cut). At or beyond the boundary there is no remaining coverage, and
    # indexing there is a stale zero or an IndexError. boundary is backfilled to
    # term in ModelPoints.__post_init__ (never None, <= term).
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    runoff = em >= boundary
    if np.any(runoff):
        bad = int(np.argmax(runoff))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} >= "
            f"contract_boundary_months[{bad}]={int(boundary[bad])} (the Sec. 34 "
            "horizon; equal to term_months when no boundary cut); the contract "
            "has no remaining coverage at the valuation date. roll_forward needs "
            "an as-of date strictly before the contract boundary."
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
    csm_traj, acc, rel = _csm_kernel(prior_csm, inforce_seg, monthly_rates,
                                     basis.coverage_unit_discount)

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

    # See _measure_inforce_fast(): Sec. 44 loss component is zeroed in settlement
    # mode v1. Unlocking and experience adjustments belong to
    # roll_forward() / Phase B v2; max(0, fcf - csm) here would mis-signal
    # a Sec. 44 hit when the only thing missing is the unlocking step.
    loss_new = np.zeros(n_mp, dtype=np.float64)

    # Headline bel/ra/csm are the as-of valuation-date values (month
    # elapsed_months per MP), matching _measure_inforce_fast -- NOT column 0
    # (inception), which would ignore prior_csm entirely. The trajectory
    # fields keep the full inception-to-horizon paths.
    #
    # Re-base the in-force count to the valuation date: the projection ran from
    # inception, so inforce[em] = count x survival(0->em) -- it decremented the
    # as-of count again from inception. Scale the sliced bel / ra by
    # count / inforce[em] = 1 / survival(0->em) so the as-of inforce is exactly
    # the input count. This is exact for every cash flow linear in the in-force
    # (premium, claim, morbidity, expense, maturity, annuity); surrender uses a
    # sample-grade cum-premium base and is approximate (see measure_inforce).
    # CSM needs no rescale: its coverage-unit release is an inforce *fraction*,
    # which the uniform scale leaves unchanged.
    rescale = _inforce_rescale(m, model_points, em, rows_arr)
    return GMMMeasurement(
        bel=m.bel_path[rows_arr, em] * rescale,
        ra=m.ra_path[rows_arr, em] * rescale,
        csm=csm_new[rows_arr, em],
        loss_component=loss_new,
        bel_path=m.bel_path,
        ra_path=m.ra_path,
        csm_path=csm_new,
        csm_accretion=csm_accretion_new,
        csm_release=csm_release_new,
        lic=m.lic,
        cashflows=m.cashflows,
        discount_factor_bom=m.discount_factor_bom,
        discount_factor_mid=m.discount_factor_mid,
        measurement_basis=MEASUREMENT_BASIS_SETTLEMENT_CARRY,
    )


def _reconcile_state(model_points: ModelPoints,
                     state: "InforceState") -> "InforceState":
    """Return ``state`` row-aligned to ``model_points`` (by mp_id), after
    checking the model points were already reconciled with it.

    Two jobs in one place. (1) The measurement reads each contract's as-of
    duration / size from ``model_points``; a model_points whose elapsed_months
    / count disagree with ``state`` was not reconciled (``apply_inforce_state``)
    and is rejected, so a stale snapshot cannot borrow a fresh state's CSM.
    (2) ``state.prior_csm`` is per-MP and must enter the measurement in
    model-points order, not the state file's order -- the returned state is
    reordered by mp_id so prior_csm lines up with the rows it belongs to.
    A reconciled, same-order pair passes through unchanged."""
    from fastcashflow.model_points import align_inforce_state
    # align_inforce_state does the mp_id join (and rejects mismatched id sets)
    # and reorders every per-MP field -- crucially prior_csm -- to mp order.
    state = align_inforce_state(model_points, state)
    em_ok = np.array_equal(
        np.asarray(state.elapsed_months, dtype=np.int64),
        np.asarray(model_points.elapsed_months, dtype=np.int64),
    )
    cnt_ok = model_points.count is not None and np.array_equal(
        np.asarray(state.count, dtype=np.float64),
        np.asarray(model_points.count, dtype=np.float64),
    )
    if not (em_ok and cnt_ok):
        raise ValueError(
            "measure_inforce: model_points elapsed_months / count do not match "
            "the InforceState. Reconcile them first -- "
            "model_points = apply_inforce_state(model_points, state) -- so the "
            "as-of duration and size come from the same period-close snapshot."
        )
    return state


def measure_inforce(
    model_points: ModelPoints,
    state: "InforceState",
    basis: "Basis | dict[tuple[str, str], Basis]",
    *,
    period_months: int | None = None,
    full: bool = True,
) -> GMMMeasurement:
    """In-force diagnostic / runoff valuation at a single date.

    The diagnostic companion to :func:`fastcashflow.gmm.settle` (the
    paragraph-44 period-close settlement). Each model point is valued at its
    ``elapsed_months`` duration: the BEL / RA are full current estimates at
    that date (Sec. 40), while the CSM is a **carry-only approximation** --
    the prior period's closing CSM (``state.prior_csm``) accreted at
    ``state.lock_in_rate`` and released over coverage units across
    ``period_months`` (default 12), with no paragraph-44(c) unlocking. The
    result is stamped ``measurement_basis='settlement_carry'`` and the
    inception-only consumers (``group`` / ``group_of_contracts`` /
    ``roll_forward`` / ``report`` / ``transition`` / the plots) reject it;
    period-close balances come from ``settle``.

    The ``bel`` / ``ra`` are re-based to the valuation date: the projection runs
    from inception, so the slice is scaled by ``count / inforce[elapsed]`` to
    set the as-of in-force to the input count -- exact for every cash flow
    linear in the in-force (premium, claim, morbidity, expense, maturity,
    annuity, and the ``amount_per_policy`` / ``amount_per_unit`` surrender
    value). The one approximation is the ``cum_premium_factor`` surrender mode:
    it reconstructs the base from the *projected* cumulative premium (``lapse x
    cum_premium x factor``), which ignores premiums paid before the valuation
    date, so it is path-dependent and only sample-grade. A ``UserWarning``
    fires only in that mode (basis carries a ``cum_premium_factor`` surrender
    curve and any ``elapsed_months > 0``); the contractual ``amount_per_policy``
    / ``amount_per_unit`` curves are linear in the in-force and re-base exactly,
    so they are the production-grade surrender input and warn-free.

    ``state`` is the :class:`InforceState` returned by
    :func:`read_inforce_policies` (it carries ``prior_csm`` / ``lock_in_rate``,
    plus the ``elapsed_months`` / ``count`` reconciled onto ``model_points``).
    ``model_points`` and ``state`` must be reconciled by
    :func:`~fastcashflow.apply_inforce_state` first -- ``read_inforce_policies``
    returns the pair already reconciled; the two-file path
    (``read_model_points`` + ``read_inforce_state``) calls it explicitly. A
    model_points whose ``elapsed_months`` / ``count`` disagree with ``state``
    is rejected (a stale snapshot must not borrow a fresh state's CSM).
    ``full=True`` (default) returns the BEL / RA / CSM trajectories and cash
    flows for movement analysis; ``full=False`` returns just the headline
    numbers (faster).

    Paragraph-44(c) unlocking, experience adjustments and the loss-component
    movement live in :func:`fastcashflow.gmm.settle` (``loss_component`` is
    zero in this mode); the opening -> closing movement of a reporting period
    comes from ``settle``'s :class:`~fastcashflow.GMMSettlementMovement`, not
    from this projector.
    """
    # A mixed-model router must go through fcf.portfolio.measure_inforce --
    # silently measuring a PAA / VFA segment with the GMM kernel would return
    # a finite, plausible, wrong number. Checked before any segment is
    # measured (no-op for a bare Basis).
    _require_gmm_router(basis, entry="measure_inforce")
    # A multi-segment ``{(product, channel): Basis}`` settles the whole
    # in-force portfolio in one call: each segment is routed to its own basis
    # (its assumptions), measured, and stitched back -- no manual per-segment
    # subsetting. A single-segment dict / a bare Basis falls through.
    if isinstance(basis, BasisRouter) and len(basis.segments) > 1:
        return _measure_inforce_segmented(
            model_points, basis, state,
            period_months=period_months, full=full,
        )
    basis = _single_basis(basis, entry="measure_inforce")
    # The measurement reads each contract's as-of duration / size from
    # ``model_points``; ``state`` supplies prior_csm / lock_in_rate. Reconcile
    # the pair: reject a model_points whose elapsed / count disagree with the
    # state (a stale snapshot must not borrow a fresh state's CSM), and -- the
    # subtle part -- reorder the state to model-points order so per-MP
    # ``prior_csm`` enters in the rows it belongs to even when the state file
    # is in a different order than the policies.
    state = _reconcile_state(model_points, state)
    if (basis.surrender_value_curve is not None
            and basis.surrender_value_basis == "cum_premium_factor"
            and np.any(np.asarray(model_points.elapsed_months) > 0)):
        warnings.warn(
            "measure_inforce reconstructs the surrender value from the "
            "projected cumulative premium (lapse x cum_premium x factor), a "
            "sample-grade base that ignores premiums paid before the valuation "
            "date and reads no contractual surrender-value table. The BEL / RA "
            "are otherwise re-based to the valuation date. Supply an "
            "amount-per-policy surrender-value curve "
            "(surrender_value_basis='amount_per_policy') for a production "
            "settlement of a product whose lapse cash flow matters.",
            UserWarning,
            stacklevel=2,
        )
    if full:
        result = _measure_inforce_full(
            model_points, basis,
            prior_csm=state.prior_csm,
            lock_in_rate=state.lock_in_rate,
            period_months=period_months,
        )
    else:
        result = _measure_inforce_fast(
            model_points, basis,
            prior_csm=state.prior_csm,
            lock_in_rate=state.lock_in_rate,
            period_months=period_months,
        )
    # Stamp the source model points, as new-business measure() does, so
    # group(result, by=[...]) resolves axis names without re-passing them.
    return replace(result, model_points=model_points)


def _measure_inforce_segmented(
    model_points: ModelPoints,
    basis: dict,
    state: "InforceState",
    *,
    period_months: int | None = None,
    full: bool = True,
    segment_by=("product", "channel"),
) -> GMMMeasurement:
    """Settle a multi-segment in-force portfolio in one call.

    Each ``(product, channel)`` segment is routed to its own ``Basis`` and
    measured with :func:`measure_inforce`; the per-segment results are stitched
    into one portfolio result. The state is aligned to the model-points order
    once (the mp_id join), then sliced by the same rows as the model points, so
    each segment's ``prior_csm`` / ``lock_in_rate`` / ``count`` travel with its
    own contracts -- the in-force mirror of :func:`_measure_segmented_full`,
    with the extra ``state`` subset. (The reorder-before-subset is the subtle
    part: slicing the state in file order would hand a segment another
    contract's prior CSM.)
    """
    state = _reconcile_state(model_points, state)
    try:
        basis_norm, segments = _factorise_segments(
            basis, model_points, segment_by, model_points.n_mp,
        )
    except KeyError:
        if len(basis.segments) == 1:
            (single,) = basis.segments.values()
            return measure_inforce(model_points, state, single,
                                   period_months=period_months, full=full)
        raise ValueError(
            f"model_points has no {tuple(segment_by)} axis/axes set but the "
            f"basis has {len(basis.segments)} segments; either set the columns or "
            "pass a single-segment basis"
        )
    n_mp = model_points.n_mp
    sub_results = [
        (idx, measure_inforce(model_points.subset(idx), state.subset(idx),
                              basis_norm[key],
                              period_months=period_months, full=full))
        for key, idx in segments
    ]
    if full:
        result = _stitch_full_measurements(n_mp, sub_results)
    else:
        bel = np.empty(n_mp)
        ra = np.empty(n_mp)
        csm = np.empty(n_mp)
        loss_component = np.empty(n_mp)
        for idx, m in sub_results:
            bel[idx] = m.bel
            ra[idx] = m.ra
            csm[idx] = m.csm
            loss_component[idx] = m.loss_component
        result = GMMMeasurement(bel=bel, ra=ra, csm=csm,
                                loss_component=loss_component)
    # Re-tag: _stitch_full_measurements is shared with the new-business
    # segmented path and constructs a default ('inception') measurement.
    return replace(result, model_points=model_points,
                   measurement_basis=MEASUREMENT_BASIS_SETTLEMENT_CARRY)


def _require_scalar_lock_in_rate(state, entry: str) -> None:
    """gmm.settle / settle_aggregate value the locked-in CSM pass at a single
    flat rate; ``InforceState.lock_in_rate`` may be a per-MP array (the dataclass
    permits it), which would otherwise reach ``float(...)`` as an opaque numpy
    TypeError. Reject it with the same message settle_stream uses, so all three
    settle entry points share one clear v1-scope error."""
    lock = np.asarray(state.lock_in_rate, dtype=np.float64)
    if lock.ndim != 0:
        raise NotImplementedError(
            f"{entry}: lock_in_rate must be uniform across rows in v1; per-MP "
            "(cohort-aware) lock-in rates are a future extension"
        )


def settle(
    model_points: ModelPoints,
    state: "InforceState",
    basis: "Basis",
    *,
    period_months: int | None = None,
    premium_experience_future_fraction: float | FloatArray = 0.0,
) -> "GMMSettlementMovement":
    """Paragraph-44 subsequent-measurement settlement of a GMM in-force book.

    The opening -> closing movement over one reporting period: BEL / RA
    re-measured at current rates (B72(a)), the CSM accreted at the locked-in
    rate (44(b)/B72(b), direct compounding), adjusted for the future-service
    change measured at the locked-in rate (44(c)/B72(c) -- the gap to the
    current-rate measure is the ``finance_wedge``, insurance finance
    income/expense per B97(a), outside the CSM block), run through the
    paragraph-48/50(b) loss-component algebra, and released once at the
    period end over coverage units (44(e)/B119, em_open denominator).

    GMM carries no account value, so the expected and observed legs share
    one unit projection (``count = 1``) and differ only by scale:
    ``k_exp = prior_count / unit_inforce[em_open]`` (the on-track
    expectation) and ``k_obs = count / unit_inforce[em_close]`` (the
    observation). On-track counts make every experience line zero and the
    closing CSM telescopes to ``measure_inforce``'s monthly carry exactly.

    Premium experience (B96(a)/B97(c)): pass ``state.actual_premium`` (the
    premium cash actually received over the period) and the entity's
    ``premium_experience_future_fraction`` to split the experience adjustment
    ``actual_premium - expected_premium`` between future service (the CSM, at
    the B72(c) locked-in measure -- ``csm_premium_experience``) and
    current/past service (a P&L memo -- ``premium_experience_revenue``,
    recognised in insurance revenue, TRG 2018-09). The fraction defaults to
    0.0 (all current/past, the BC233 general rule); the standard leaves the
    split to entity judgment. The lapse-driven future-premium effect is
    already carried by the count channel, so default 0.0 avoids
    double-counting -- set the fraction above 0 only for premium received now
    for genuinely future coverage the count deviation does not capture.

    An onerous book amortises its loss component through the paragraph-50(a)/51
    incurred-service channel (``loss_component_finance`` /
    ``loss_component_amortised`` on the movement): the period's released
    claims and expenses (51a), RA release (51b) and finance (51c) are split on
    the systematic loss-component ratio ``r = loss_component_opening /
    pool_opening`` between the loss component and the LRC excluding it, running
    the loss component to zero by the end of coverage (52).

    A ``settlement_pattern`` basis is supported: the movement carries the
    liability for incurred claims (``lic_opening`` / ``claims_incurred`` /
    ``claims_paid`` / ``lic_closing``, paragraphs 40(b) / 42 / 103(b)) -- claims
    build it up as incurred and run it off over the pattern, undiscounted and at
    the expected scale, reconstructed from the projection each period.

    Within-period claims and expense experience (B97(b)/(c)) is surfaced when
    ``state.actual_claims`` / ``state.actual_expenses`` are given: the
    actual-minus-expected difference is recognised in the insurance service
    result (``claims_experience`` / ``expense_experience``, P&L memos, not the
    CSM and not a balance recursion). Absent the inputs they are zero.

    v1 scope (documented cuts, mirroring ``vfa.settle``): the closing balances
    (BEL / RA / CSM / LIC) are still built on the expected within-period run
    (the experience above is a P&L memo, not a re-derivation of the balances);
    the LIC roll is expected-scale and undiscounted (no 42(c) finance / 33-37
    discount + RA on the LIC, the same cut the measure takes); no B96(c)
    investment-component split, so the paragraph-50(a) pool includes the whole
    non-premium outflow (surrender / maturity not separated as investment
    components, B124(ii)); the RA change enters the CSM at its current measure
    (B96(d) prescribes no rate); no OCI -- the ``finance_wedge`` is the
    period's P&L line, not an accumulated-OCI state. A maturity falling inside
    the period is expected service (it seeds the unit BEL at the boundary and
    runs off through the release line), not experience.
    """
    _require_gmm_router(basis, entry="gmm.settle")
    basis = _single_basis(basis, entry="gmm.settle")
    state = _reconcile_state(model_points, state)
    _require_scalar_lock_in_rate(state, "gmm.settle")
    if state.prior_count is None:
        raise ValueError(
            "gmm.settle needs state.prior_count -- the in-force count at the "
            "opening date (the expected leg's scale and the B119 release "
            "denominator)."
        )
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period}")

    n_mp = model_points.n_mp
    prior_csm = np.asarray(state.prior_csm, dtype=np.float64)
    if np.any(prior_csm < 0.0):
        bad = int(np.argmin(prior_csm))
        raise ValueError(
            f"prior_csm[{bad}]={prior_csm[bad]} is negative; a GMM CSM is "
            "floored at zero (an onerous balance is the loss component -- "
            "pass it as prior_loss_component)."
        )
    lc_open = (np.asarray(state.prior_loss_component, dtype=np.float64)
               if state.prior_loss_component is not None
               else np.zeros(n_mp))
    both = (prior_csm > 0.0) & (lc_open > 0.0)
    if np.any(both):
        bad = int(np.argmax(both))
        raise ValueError(
            f"row {bad} carries both prior_csm={prior_csm[bad]} and "
            f"prior_loss_component={lc_open[bad]}; a group has a CSM or a "
            "loss_component, never both (paragraphs 44 / 47-52)."
        )

    em_close = np.asarray(model_points.elapsed_months, dtype=np.int64)
    em_open = em_close - period
    if np.any(em_open < 0):
        bad = int(np.argmin(em_open))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em_close[bad])} < "
            f"period_months={period}; the opening date precedes inception, "
            "which has no balances to settle from."
        )
    boundary = np.asarray(model_points.contract_boundary_months,
                          dtype=np.int64)
    if np.any(em_open >= boundary):
        bad = int(np.argmax(em_open >= boundary))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em_close[bad])} - "
            f"period_months={period} is at or past the contract boundary "
            f"({int(boundary[bad])}); the opening date must lie strictly "
            "inside the coverage period."
        )
    count = np.asarray(model_points.count, dtype=np.float64)
    final = em_close >= boundary
    if np.any(final & (count > 0.0)):
        bad = int(np.argmax(final & (count > 0.0)))
        raise ValueError(
            f"row {bad} closes at or past the contract boundary with "
            f"count={count[bad]}; a final settlement needs a zero closing "
            "snapshot (full B119 derecognition)."
        )

    # One unit projection (count = 1) carries both legs; the real scales ride
    # on k_exp / k_obs (the vfa.settle unit-count seeding, correction 6).
    unit = replace(model_points, count=np.ones(n_mp))
    m = _measure_full(unit, basis)
    cf = m.cashflows
    # The settlement movement reads claim_cf / maturity_cf / surrender_cf raw as
    # incurred / paid benefits; an account book's benefits are not priced claims,
    # so reject it (settle_aggregate funnels through here, so it is covered too).
    reject_account_book(cf, "gmm.settle")
    inforce = cf.inforce
    n_time = inforce.shape[1]
    rows = np.arange(n_mp)

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
            f"but the observed count is {count[bad]}; reconcile the snapshot "
            "(the closing date may be past the decrement horizon)."
        )
    k_obs = np.where(surv_close > 0.0,
                     count / np.where(surv_close > 0.0, surv_close, 1.0), 0.0)

    # A final settlement's expected close is ZERO, not the boundary column
    # (bel_path[boundary] seeds the maturity as still-owed; a maturity paid
    # on schedule is expected service inside the release, not experience).
    live_close = np.where(final, 0.0, 1.0)
    # A final settlement may close past the boundary (a long-matured row,
    # em_close > n_time): clamp every closing-column read -- the clamped
    # values are zeroed by live_close / k_obs / the tail's zero terminal.
    em_c = np.minimum(em_close, n_time)

    discount_monthly = forward_rates(m.discount_factor_bom)
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

    bel_o, bel_i, bel_r, bel_e, bel_c = _block(m.bel_path)
    ra_o, ra_i, ra_r, ra_e, ra_c = _block(m.ra_path)

    # paragraph 50(a)/51 incurred-service channel: as coverage is provided the
    # period's released claims and expenses (51a), RA release (51b) and finance
    # (51c) are allocated on a systematic basis between the loss component and
    # the LRC excluding it. The claims+expenses pool is the BEL path GROSS of
    # premiums -- the same backward kernel with premiums zeroed, at the current
    # rate -- so out_o/out_i/out_r are the pool's opening / interest unwind /
    # period release. The systematic basis (entity judgment, paragraph 50(a))
    # is the proportional loss-component ratio r = lc_open / pool_open: the LC
    # accretes r x the pool interest (51c) and amortises r x the pool release
    # (50(a)). A profitable book has lc_open == 0 => r == 0 => both lines
    # vanish (byte-identical to the pre-feature settle). The amortisation is the
    # paragraph-49/B123(b) loss reversal, excluded from insurance revenue; it
    # runs the LC to zero by the end of coverage (52) because r is re-derived
    # every period (at the final close the whole pool releases and lc_amortised
    # == lc carried). The future-service algebra below acts on the
    # POST-amortisation loss component.
    #
    # paragraph 51(a) allocates "claims and expenses"; investment components
    # (surrender / annuity / maturity -- amounts repaid regardless of an insured
    # event, paragraph 85 / B96(c)) are NOT claims or expenses, so the pool and
    # the amortised release exclude them (kernel run with those streams zeroed).
    zero_prem = np.zeros_like(cf.premium_cf)
    zero_ann = np.zeros_like(cf.annuity_cf)
    zero_mat = np.zeros_like(cf.maturity_cf)
    zero_surr = np.zeros_like(cf.surrender_cf)
    outflow_path = _rollforward_kernel(
        cf.claim_cf, cf.morbidity_cf, cf.disability_cf, cf.expense_cf,
        zero_prem, zero_ann, zero_mat, zero_surr,
        boundary, discount_monthly)[0]
    out_o, out_i, out_r, out_e, out_c = _block(outflow_path)
    pool_open = out_o + ra_o
    lc_ratio = np.where(pool_open > 0.0,
                        lc_open / np.where(pool_open > 0.0, pool_open, 1.0), 0.0)
    lc_finance = lc_ratio * (out_i + ra_i)          # 51(c)
    lc_amortised = lc_ratio * (out_r + ra_r)        # 50(a)/51(a)+(b)
    lc_after_incurred = lc_open + lc_finance - lc_amortised

    # B96(a)/B97(c) premium experience: actual premium received over the
    # period vs the expected (on-track) premium, summed over the same window
    # and at the same k_exp scale as the interest line. The entity's
    # future-service fraction routes the favourable(+)/unfavourable(-)
    # difference: the future part adjusts the CSM (B96(a), into the
    # paragraph-48/50(b) algebra at the locked-in measure); the current/past
    # part is a P&L memo (B97(c), insurance revenue), in no balance recursion.
    # Absent actual_premium => zero on both lines (byte-identical to the cut).
    frac = np.asarray(premium_experience_future_fraction, dtype=np.float64)
    if frac.ndim > 1 or (frac.ndim == 1 and frac.shape[0] != n_mp):
        raise ValueError(
            "premium_experience_future_fraction must be a scalar or one entry "
            f"per model point ({n_mp}), got shape {frac.shape}")
    if (not np.all(np.isfinite(frac))
            or np.any(frac < 0.0) or np.any(frac > 1.0)):
        raise ValueError(
            "premium_experience_future_fraction must be finite and lie in "
            "[0, 1] (the entity's split of the premium experience between "
            "future service -> CSM and current/past service -> P&L); got "
            f"{premium_experience_future_fraction}")
    if state.actual_premium is not None:
        exp_premium = k_exp * (cf.premium_cf[rows[:, None], cols_safe]
                               * col_ok).sum(axis=1)
        premium_experience = (np.asarray(state.actual_premium,
                                         dtype=np.float64) - exp_premium)
    else:
        premium_experience = np.zeros(n_mp)
    csm_premium_experience = frac * premium_experience
    premium_experience_revenue = (1.0 - frac) * premium_experience

    # B96(c) investment-component experience: the difference between the
    # expected and the actual investment component (surrender / annuity) that
    # becomes payable over the period. The whole difference adjusts the CSM (no
    # fraction -- B96(c) is entirely future service); the investment component
    # does not touch insurance revenue. The expected runs at the k_exp scale of
    # the within-period cash flows (only the closing count and the premium are
    # observed in v1), so this does NOT double-count the count channel. An
    # extra payout (actual > expected) is unfavourable (CSM down); retained
    # business (actual < expected) is favourable. Absent
    # actual_investment_component => zero (byte-identical).
    ic_streams = cf.surrender_cf + cf.annuity_cf
    expected_ic = k_exp * (ic_streams[rows[:, None], cols_safe]
                           * col_ok).sum(axis=1)
    if state.actual_investment_component is not None:
        actual_ic = np.asarray(state.actual_investment_component,
                               dtype=np.float64)
        csm_investment_experience = expected_ic - actual_ic
    else:
        csm_investment_experience = np.zeros(n_mp)

    # The locked-in second pass: the SAME backward kernel on the same unit
    # cash flows, at the flat locked-in rate -- exactly one extra pass (the
    # G1 gate (3) cost fact), and identical code path so a flat current
    # basis equal to the lock-in gives a zero wedge identically.
    lock = float(state.lock_in_rate)
    lock_monthly = np.full(n_time, (1.0 + lock) ** (1.0 / 12.0) - 1.0)
    bel_lock = _rollforward_kernel(
        cf.claim_cf, cf.morbidity_cf, cf.disability_cf, cf.expense_cf,
        cf.premium_cf, cf.annuity_cf, cf.maturity_cf, cf.surrender_cf,
        boundary, lock_monthly)[0]
    delta_lock = (k_obs - k_exp) * bel_lock[rows, em_c] * live_close

    # 44(c) at the locked-in rate (B72(c)); the RA change has no rate
    # prescription (B96(d)) and enters at its current measure. The wedge is
    # the current-vs-locked-in gap of the BEL delta -- B97(a), P&L.
    csm_experience_unlocking = -(delta_lock + ra_e)
    finance_wedge = -(bel_e - delta_lock)

    csm_accretion = prior_csm * ((1.0 + lock) ** (period / 12.0) - 1.0)
    accreted = prior_csm + csm_accretion
    csm_after, lc_reversed, lc_recognised, lc_closing = (
        _paragraph45_csm_algebra(
            accreted, csm_experience_unlocking + csm_premium_experience
            + csm_investment_experience,
            lc_after_incurred))

    # B119: single period-end release on the post-adjustment balance. The
    # provided units run at the expected scale over [em_open, em_close), the
    # future units at the observed scale -- the em_open-denominator fraction
    # that telescopes to the monthly carry when on-track.
    tail = np.zeros((n_mp, n_time + 1))
    if basis.coverage_unit_discount:
        # B119 discounted coverage units (accounting-policy choice): weight
        # each month's units by the cumulative locked-in discount factor before
        # the reverse-cumsum, so provided and future units are compared on a
        # common present-value basis. The fraction is invariant to the discount
        # reference point, so discounting to t=0 suffices.
        lock_m = (1.0 + lock) ** (1.0 / 12.0) - 1.0
        disc = (1.0 + lock_m) ** (-np.arange(n_time))
        weighted = inforce[:, :n_time] * disc[None, :]
        tail[:, :n_time] = np.cumsum(weighted[:, ::-1], axis=1)[:, ::-1]
    else:
        tail[:, :n_time] = np.cumsum(inforce[:, ::-1], axis=1)[:, ::-1]
    cu_provided = k_exp * (tail[rows, em_open] - tail[rows, em_c])
    cu_future = k_obs * tail[rows, em_c]
    denom = cu_provided + cu_future
    frac = np.where(denom > 0.0,
                    cu_provided / np.where(denom > 0.0, denom, 1.0), 1.0)
    csm_release = csm_after * frac
    csm_closing = csm_after - csm_release

    # Liability for incurred claims (paragraphs 40(b) / 42 / 103(b) / 37):
    # claims build it up as incurred (42(a)) and run it off over the settlement
    # pattern. The LIC is measured at fulfilment cash flows -- the discounted PV
    # of the unpaid run-off plus the risk adjustment. claims_incurred and
    # claims_paid stay NOMINAL cash amounts (claims_paid the nominal residual on
    # the undiscounted trajectory m.lic, the same reconstruction as paa.settle);
    # the discounting (42(c)) and RA (37) move only the balances, and lic_finance
    # is the reconciling residual -- the insurance finance (discount unwind) plus
    # the discounting / RA measurement effect. m.lic is the undiscounted unit
    # trajectory (all-zero when the basis has no settlement_pattern, i.e. claims
    # paid as incurred -- the LIC is zero at both dates and lic_finance is zero).
    incurred = cf.claim_cf + cf.morbidity_cf
    claims_incurred = k_exp * (incurred[rows[:, None], cols_safe]
                               * col_ok).sum(axis=1)
    claims_paid = (k_exp * m.lic[rows, em_open] + claims_incurred
                   - k_exp * m.lic[rows, em_c])
    if basis.settlement_pattern is not None:
        # discounted PV of the unpaid run-off, split by risk class for the RA
        r_lic = basis.discount_monthly
        lic_death = _settlement_lic_discounted(
            cf.claim_cf, basis.settlement_pattern, r_lic)
        lic_morb = _settlement_lic_discounted(
            cf.morbidity_cf, basis.settlement_pattern, r_lic)
        # RA on the LIC (paragraph 37): z x cv-weighted discounted LIC by risk
        # class -- the confidence-level margin, the well-defined form for the
        # short incurred-claims run-off (a cost-of-capital LIC run-off is a
        # refinement; the LIC RA was previously omitted entirely).
        z = _norm_ppf(basis.ra_confidence)
        lic_ra = z * (basis.mortality_cv * lic_death
                      + basis.morbidity_cv * lic_morb)
        lic_fcf = lic_death + lic_morb + lic_ra
        lic_opening = k_exp * lic_fcf[rows, em_open]
        lic_closing = k_exp * lic_fcf[rows, em_c]
    else:
        lic_opening = k_exp * m.lic[rows, em_open]
        lic_closing = k_exp * m.lic[rows, em_c]
    lic_finance = lic_closing - lic_opening - claims_incurred + claims_paid

    # B97(b)/(c) within-period claims and expense experience: the actual claims
    # incurred / expenses incurred over the period less the expected. The v1
    # settle otherwise assumes within-period cash flows equal expected (only the
    # closing count, premium and investment component are observed); these two
    # lines surface the remaining experience. It relates to past/current service
    # (B97) -- recognised in the insurance service result (P&L), NOT the CSM and
    # NOT a balance recursion (a memo, like premium_experience_revenue and
    # finance_wedge). Absent the inputs => zero (byte-identical). The expected
    # claims are claims_incurred (above); the expected expenses are the period
    # expense run at the expected scale.
    exp_expenses = k_exp * (cf.expense_cf[rows[:, None], cols_safe]
                            * col_ok).sum(axis=1)
    if state.actual_claims is not None:
        claims_experience = (np.asarray(state.actual_claims, dtype=np.float64)
                             - claims_incurred)
    else:
        claims_experience = np.zeros(n_mp)
    if state.actual_expenses is not None:
        expense_experience = (np.asarray(state.actual_expenses, dtype=np.float64)
                              - exp_expenses)
    else:
        expense_experience = np.zeros(n_mp)

    from fastcashflow.movement import GMMSettlementMovement
    return GMMSettlementMovement(
        bel_opening=bel_o, bel_interest=bel_i, bel_release=bel_r,
        bel_experience=bel_e, bel_closing=bel_c,
        ra_opening=ra_o, ra_interest=ra_i, ra_release=ra_r,
        ra_experience=ra_e, ra_closing=ra_c,
        csm_opening=prior_csm, csm_accretion=csm_accretion,
        csm_experience_unlocking=csm_experience_unlocking,
        csm_premium_experience=csm_premium_experience,
        csm_investment_experience=csm_investment_experience,
        finance_wedge=finance_wedge,
        premium_experience_revenue=premium_experience_revenue,
        claims_experience=claims_experience,
        expense_experience=expense_experience,
        csm_release=csm_release, csm_closing=csm_closing,
        loss_component_opening=lc_open,
        loss_component_finance=lc_finance,
        loss_component_amortised=lc_amortised,
        loss_component_reversed=lc_reversed,
        loss_component_recognised=lc_recognised,
        loss_component_closing=lc_closing,
        coverage_units_provided=cu_provided,
        coverage_units_future=cu_future,
        lic_opening=lic_opening,
        claims_incurred=claims_incurred,
        lic_finance=lic_finance,
        claims_paid=claims_paid,
        lic_closing=lic_closing,
        period_months=period, lock_in_rate=lock,
        model_points=model_points,
    )


def settle_aggregate(
    model_points: ModelPoints,
    state: "InforceState",
    basis: "Basis",
    *,
    period_months: int | None = None,
    premium_experience_future_fraction: float | FloatArray = 0.0,
    chunk_size: int = 200_000,
) -> "GMMSettlementAggregate":
    """Portfolio-total paragraph-44 settlement in bounded memory.

    :func:`settle` materialises ``(n_mp, n_time)`` projection intermediates
    -- two backward kernel passes over the whole book -- so a
    million-policy close would peak far beyond memory. Every line of the
    settlement movement is additive across contracts, so this runs
    :func:`settle` over row blocks of ``chunk_size`` model points and
    accumulates only the scalar line totals; peak memory is
    ``O(chunk_size x n_time)`` regardless of ``n_mp``.

    Returns a :class:`~fastcashflow.movement.GMMSettlementAggregate`: the
    movement's lines summed, movement-positive (``reconcile`` applies the
    display negation and reproduces the per-MP movement's table exactly).
    The aggregate cannot be chained -- ``closing_inputs()`` raises; chain
    per-MP movements instead. ``state`` joins ``model_points`` by mp_id
    once, before chunking, so a period-close file in its own row order
    never pairs one contract's rows with another's prior balances.
    """
    from fastcashflow.movement import (
        _GMM_SETTLEMENT_LINES, GMMSettlementAggregate)

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period}")
    # One global mp_id join (and stale-snapshot check) BEFORE slicing, so a
    # chunk's model points and state rows always belong to the same
    # contracts; the per-chunk settle re-checks the aligned pair (a no-op).
    state = _reconcile_state(model_points, state)
    _require_scalar_lock_in_rate(state, "gmm.settle_aggregate")
    n_mp = int(model_points.issue_age.shape[0])
    # A per-MP fraction is sliced per chunk so the aggregate equals the per-MP
    # settle sum even when the premium split varies by contract (the per-chunk
    # settle re-validates the value range / finiteness).
    pe_frac = np.asarray(premium_experience_future_fraction, dtype=np.float64)
    if pe_frac.ndim > 1 or (pe_frac.ndim == 1 and pe_frac.shape[0] != n_mp):
        raise ValueError(
            "premium_experience_future_fraction must be a scalar or one entry "
            f"per model point ({n_mp}), got shape {pe_frac.shape}")
    # Per-chunk partial sums, combined with fsum so the total does not
    # depend on the chunking (compensated summation: chunk_size is a memory
    # knob, never a numbers knob).
    parts: dict[str, list[float]] = {n: [] for n in _GMM_SETTLEMENT_LINES}
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        frac_arg = (float(pe_frac) if pe_frac.ndim == 0 else pe_frac[idx])
        mv = settle(model_points.subset(idx), state.subset(idx), basis,
                    period_months=period,
                    premium_experience_future_fraction=frac_arg)
        for name in _GMM_SETTLEMENT_LINES:
            parts[name].append(float(getattr(mv, name).sum()))
    return GMMSettlementAggregate(
        period_months=period, lock_in_rate=float(state.lock_in_rate),
        **{name: math.fsum(vals) for name, vals in parts.items()})


@dataclass(frozen=True, slots=True, eq=False)
class CSMRecognitionSchedule:
    """IFRS 17 paragraph-109 disclosure: the closing CSM allocated to maturity
    bands by expected coverage-unit recognition.

    ``band_edges_months`` are the band boundaries in months from the valuation
    date (default 12 / 36 / 60, the four-band disclosure axis); the bands are
    ``[0, e0), [e0, e1), ..., [e_last, end)``. ``csm[b]`` is the closing CSM
    expected to be recognised in band ``b`` -- allocated by each contract's
    forward coverage-unit fraction, so the bands SUM TO ``closing_csm``. It is
    an allocation of the remaining balance, not the accreted nominal release;
    the coverage-unit proxy is the in-force count, undiscounted, matching the
    B119 amortisation kernel, so the schedule tracks the actual release pattern.
    """

    band_edges_months: tuple
    csm: FloatArray              # (n_bands,) -- sums to closing_csm
    closing_csm: float

    @property
    def labels(self) -> tuple:
        """Band labels, in years when the edges are whole years."""
        def fmt(m):
            return f"{m // 12}y" if m % 12 == 0 else f"{m}m"
        edges = self.band_edges_months
        out = [f"<= {fmt(edges[0])}"]
        out += [f"{fmt(lo)} - {fmt(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]
        out.append(f"{fmt(edges[-1])} +")
        return tuple(out)


def recognition_schedule(
    model_points: ModelPoints,
    state: InforceState,
    basis,
    *,
    band_edges_months=(12, 36, 60),
    period_months: int | None = None,
) -> CSMRecognitionSchedule:
    """Paragraph-109 maturity-band disclosure for a settled GMM book.

    Allocates the closing CSM (the :func:`settle` closing of ``model_points`` /
    ``state``) to maturity bands by each contract's forward coverage-unit
    fraction, so the bands SUM TO the closing CSM -- when, in maturity terms, the
    remaining CSM is expected to be recognised in profit or loss. The coverage
    units are the in-force count from the valuation date (the B119 amortisation
    proxy, undiscounted), so the schedule matches the actual CSM release. Onerous
    contracts carry no CSM and contribute nothing.

    ``band_edges_months`` are the band boundaries in months from the valuation
    date (default 12 / 36 / 60, the four-band disclosure axis); ``period_months``
    is the settlement period (default 12), as for :func:`settle`.
    """
    edges = _validate_band_edges(band_edges_months)
    mv = settle(model_points, state, basis, period_months=period_months)
    inforce = measure(model_points, basis, full=True).cashflows.inforce
    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    return _build_recognition_schedule(
        np.asarray(mv.csm_closing, dtype=np.float64), inforce, em, boundary,
        edges)


def _validate_band_edges(band_edges_months) -> tuple:
    """Coerce / validate paragraph-109 band edges -- strictly ascending
    positive integer months from the valuation date. Shared by the GMM and VFA
    recognition schedules so the edge contract cannot drift between them."""
    edges = tuple(int(e) for e in band_edges_months)
    if (not edges or any(e <= 0 for e in edges)
            or list(edges) != sorted(set(edges))):
        raise ValueError(
            "band_edges_months must be strictly ascending positive integers "
            f"(months from the valuation date), got {band_edges_months!r}")
    return edges


def _build_recognition_schedule(csm_closing, inforce, em, boundary, edges):
    """Allocate the per-MP closing CSM to maturity bands by each contract's
    forward coverage-unit (in-force) fraction; the bands sum to the closing
    CSM. Onerous contracts (CSM <= 0) contribute nothing. Shared by the GMM
    (paragraph 44) and VFA (paragraph 45) settlement schedules -- the
    paragraph-109 allocation is identical, only the source settle differs."""
    bounds = (0,) + edges
    n_bands = len(bounds)
    band = np.zeros(n_bands)
    for i in range(csm_closing.shape[0]):
        csm_i = float(csm_closing[i])
        if csm_i <= 0.0:                  # onerous / no CSM -> nothing to recognise
            continue
        cu = inforce[i, em[i]:boundary[i]]    # forward coverage units (in-force)
        total = cu.sum()
        if total <= 0.0:                 # guarded by settle (em < boundary); belt-and-braces
            continue
        for b in range(n_bands):
            lo = bounds[b]
            hi = bounds[b + 1] if b + 1 < n_bands else cu.shape[0]
            band[b] += csm_i * cu[lo:hi].sum() / total
    return CSMRecognitionSchedule(
        band_edges_months=edges, csm=band,
        closing_csm=float(csm_closing[csm_closing > 0.0].sum()))


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
                                 use_surrender=True,
                                 surrender_is_amount=False) -> str:
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
    # State-machine lapse exits: occupancy on each state times that state's own
    # lapse rate, so surrender follows the actual lapse (a non-lapsing state
    # contributes nothing; a paid-up state lapses at its own rate). For a single
    # active state this equals ``inforce_t * lapse``, the historical formula.
    sum_lapse = " + ".join(
        f"occ_{i} * state_lapse[{i}, sx, age_idx, year]" for i in range(n_states))

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
    line(0, "           term_months, contract_boundary_months, count, premium,")
    line(0, "           premium_term_months, premium_frequency_months, "
            "annuity_frequency_months,")
    line(0, "           coverage_index, coverage_amount, coverage_offset, coverage_rates, "
            "premium_factor, annuity_factor, coverage_risk,")
    line(0, "           coverage_is_diagnosis, maturity_benefit, "
            "annuity_payment,")
    line(0, "           disability_income, disability_benefit,")
    line(0, "           alpha_pro_rata, alpha_fixed, beta_pro_rata,")
    line(0, "           gamma_fixed, lae_pro_rata,")
    line(0, "           discount_factor_bom, discount_factor_mid, mortality_factor,")
    line(0, "           morbidity_factor, longevity_factor, "
            "disability_factor,")
    line(0, "           coverage_waiting, coverage_reduction_end, "
            "coverage_reduction_factor,")
    line(0, "           lapse_monthly, state_lapse, surrender_curve, surrender_base):")
    line(4, "n_mp = issue_index.shape[0]")
    line(4, "bel = np.empty(n_mp)")
    line(4, "ra = np.empty(n_mp)")
    line(4, "csm = np.empty(n_mp)")
    line(4, "loss_component = np.empty(n_mp)")

    line(4, "for mp in prange(n_mp):")
    line(8, "term = term_months[mp]")
    line(8, "boundary = contract_boundary_months[mp]")
    line(8, "premium_term = premium_term_months[mp]")
    line(8, "prem_freq = premium_frequency_months[mp]")
    line(8, "ann_freq = annuity_frequency_months[mp]")
    line(8, "age_idx = issue_index[mp]")
    line(8, "sx = sex[mp]")
    line(8, "cnt = count[mp]")
    line(8, "prem = premium[mp]")
    line(8, "annuity = annuity_payment[mp]")
    line(8, "c_start = coverage_offset[mp]")
    line(8, "c_end = coverage_offset[mp + 1]")
    line(8, "ss = start_state[mp]")
    line(8, "pf = 1.0")
    line(8, "ann_prem = prem * 12.0 / prem_freq")
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
    line(8, "for t in range(boundary):")
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
    line(16, "pf = premium_factor[sx, age_idx, year]")
    line(16, "ann_prem = prem * pf * 12.0 / prem_freq")
    line(12, f"inforce_t = {sum_all}")
    line(12, f"prem_occ = {sum_prem}")
    if use_disability:
        line(12, f"benefit_occ = {sum_ben}")
    line(12, "ds = discount_factor_bom[t]")
    line(12, "dm = discount_factor_mid[t]")
    line(12, "if prem_due == 0 and prem_left > 0:")
    line(16, "level = prem_occ * prem * pf")
    line(16, "prem_due = prem_freq - 1")
    line(12, "else:")
    line(16, "level = 0.0")
    line(16, "prem_due -= 1")
    line(12, "prem_left -= 1")
    line(12, "pv_premium += level * ds")
    if use_surrender and not surrender_is_amount:
        line(12, "cum_premium += level")
    line(12, "pv_mortality += inforce_t * claim_rate * dm")
    if use_morbidity:
        line(12, "pv_morbidity += inforce_t * morb_rate * dm")
    if use_annuity:
        line(12, "if ann_due == 0:")
        line(16, "pv_annuity += inforce_t * annuity * annuity_factor[sx, age_idx, year] * ds")
        line(16, "ann_due = ann_freq - 1")
        line(12, "else:")
        line(16, "ann_due -= 1")
    if use_disability:
        line(12, "pv_disability += benefit_occ * disability_income[mp] * dm")
    if use_surrender:
        line(12, f"lapse_flow = {sum_lapse}")
        if surrender_is_amount:
            # amount_per_policy / amount_per_unit: surrender_curve[t] is the
            # surrender amount at duration t (per policy, or per unit of
            # surrender_base[mp]); lapse_flow is the number lapsing.
            # surrender_base is 1.0 for amount_per_policy.
            line(12, "pv_surrender += (lapse_flow * surrender_curve[t]")
            line(12, "                 * surrender_base[mp] * dm)")
        else:
            # cum_premium_factor: cum_premium aggregates inforce * premium; the
            # effective lapse fraction is lapse_flow / inforce_t (the raw rate for a
            # single state). inforce_t carries the count, so dividing it out avoids a
            # cnt^2 scaling.
            line(12, "eff_lapse = lapse_flow / inforce_t if inforce_t > 0.0 else 0.0")
            line(12, "pv_surrender += (eff_lapse")
            line(12, "                 * cum_premium * surrender_curve[t] * dm)")
    line(12, "alpha = cnt * (alpha_pro_rata * ann_prem + alpha_fixed) if t == 0 else 0.0")
    line(12, "beta = inforce_t * beta_pro_rata * ann_prem / 12.0 if t < premium_term else 0.0")
    line(12, "gamma = inforce_t * gamma_fixed[t]")
    if use_lae:
        line(12, "lae = lae_pro_rata[t] * "
                "inforce_t * (claim_rate + morb_rate)")
        line(12, "pv_expense += (alpha + beta + gamma + lae) * dm")
    else:
        line(12, "pv_expense += (alpha + beta + gamma) * dm")
    emit_edge_step(12, scale="", include_lump=True)

    line(8, f"total = {sum_all}")
    line(8, "pm = (total * maturity_benefit[mp] * discount_factor_bom[boundary]) "
         "if boundary == term else 0.0")

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
    line(12, "for t in range(boundary):")
    line(16, "year = t // 12")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, f"inforce_t = {sum_all}")
    line(20, "contrib = (inforce_t * coverage_rates[cov_idx, sx, age_idx, year]")
    line(20, "           * benefit * mult * discount_factor_mid[t])")
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
    line(12, "for t in range(boundary):")
    line(16, "year = t // 12")
    line(16, "if year != d_year:")
    line(20, "d_rate = coverage_rates[cov_idx, sx, age_idx, year]")
    line(20, "d_year = year")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, f"healthy = {sum_all}")
    line(20, "pv_morbidity += healthy * d_rate * benefit * mult * discount_factor_mid[t]")
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
                              use_surrender=True, surrender_is_amount=False):
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
        bool(use_lae), bool(use_surrender), bool(surrender_is_amount),
    )
    cached = _FAST_KERNEL_CODEGEN_CACHE.get(key)
    if cached is not None:
        return cached

    src = _codegen_fast_kernel_source(
        n_states, edge_from, edge_to, edge_lump_sum,
        premium_state, benefit_state,
        use_morbidity=use_morbidity, use_annuity=use_annuity,
        use_disability=use_disability, use_lae=use_lae,
        use_surrender=use_surrender, surrender_is_amount=surrender_is_amount,
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
# When the StateModel declares any state with ``sojourn_tracking_months > 0`` the engine
# tracks per-cohort occupancy in that state -- ``occ[state, tau]`` where
# ``tau`` is the sojourn time (months since entering the state). Transitions
# marked ``sojourn_dependent=True`` then carry per-cohort rates, the natural
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
    n_states, state_duration_max, periodic_benefit_term_months,
    edge_from, edge_to, edge_lump_sum,
    premium_state, benefit_state,
    use_annuity=True, use_lae=True, use_surrender=True,
    surrender_is_amount=False,
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
    cap = [int(x) for x in periodic_benefit_term_months]

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
    # ``periodic_benefit_term_months[s] > 0`` caps the paid cohorts: only sojourn
    # ``tau < cap`` is paid (lives past the cap stay in force, see
    # ``periodic_benefit_term_months`` on State). ``cap`` is a compile-time loop bound,
    # so the hot kernel carries no extra branch. Validation guarantees
    # ``cap < D``, so ``min`` is belt-and-suspenders.
    sum_ben = " + ".join(
        occ(s, tau) for s in range(n_states)
        if benefit_state[s]
        for tau in range(min(D[s], cap[s]) if cap[s] else D[s])) or "0.0"
    # State-machine lapse exits: each state's cohort-summed occupancy times its
    # own lapse rate (cohort-independent), so surrender follows the actual
    # lapse. Equals ``inforce_t * lapse`` for a single lapsing state.
    sum_lapse = " + ".join(
        f"({' + '.join(occ(s, tau) for tau in range(D[s]))})"
        f" * state_lapse[{s}, sx, age_idx, year]"
        for s in range(n_states) if D[s] > 0) or "0.0"

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
    line(0, "           term_months, contract_boundary_months, count, premium,")
    line(0, "           premium_term_months, premium_frequency_months, "
            "annuity_frequency_months,")
    line(0, "           coverage_index, coverage_amount, coverage_offset, coverage_rates, "
            "premium_factor, annuity_factor, coverage_risk,")
    line(0, "           coverage_is_diagnosis, maturity_benefit, "
            "annuity_payment,")
    line(0, "           disability_income, disability_benefit,")
    line(0, "           alpha_pro_rata, alpha_fixed, beta_pro_rata,")
    line(0, "           gamma_fixed, lae_pro_rata,")
    line(0, "           discount_factor_bom, discount_factor_mid, mortality_factor,")
    line(0, "           morbidity_factor, longevity_factor, "
            "disability_factor,")
    line(0, "           coverage_waiting, coverage_reduction_end, "
            "coverage_reduction_factor,")
    line(0, "           lapse_monthly, state_lapse, surrender_curve, surrender_base):")
    line(4, "n_mp = issue_index.shape[0]")
    line(4, "bel = np.empty(n_mp)")
    line(4, "ra = np.empty(n_mp)")
    line(4, "csm = np.empty(n_mp)")
    line(4, "loss_component = np.empty(n_mp)")

    line(4, "for mp in prange(n_mp):")
    line(8, "term = term_months[mp]")
    line(8, "boundary = contract_boundary_months[mp]")
    line(8, "premium_term = premium_term_months[mp]")
    line(8, "prem_freq = premium_frequency_months[mp]")
    line(8, "ann_freq = annuity_frequency_months[mp]")
    line(8, "age_idx = issue_index[mp]")
    line(8, "sx = sex[mp]")
    line(8, "cnt = count[mp]")
    line(8, "prem = premium[mp]")
    line(8, "annuity = annuity_payment[mp]")
    line(8, "c_start = coverage_offset[mp]")
    line(8, "c_end = coverage_offset[mp + 1]")
    line(8, "ss = start_state[mp]")
    line(8, "pf = 1.0")
    line(8, "ann_prem = prem * 12.0 / prem_freq")
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
    line(8, "for t in range(boundary):")
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
    line(16, "pf = premium_factor[sx, age_idx, year]")
    line(16, "ann_prem = prem * pf * 12.0 / prem_freq")
    line(12, f"inforce_t = {sum_all}")
    line(12, "inforce_traj[t] = inforce_t")
    line(12, f"prem_occ = {sum_prem}")
    line(12, f"benefit_occ = {sum_ben}")
    line(12, "ds = discount_factor_bom[t]")
    line(12, "dm = discount_factor_mid[t]")
    line(12, "if prem_due == 0 and prem_left > 0:")
    line(16, "level = prem_occ * prem * pf")
    line(16, "prem_due = prem_freq - 1")
    line(12, "else:")
    line(16, "level = 0.0")
    line(16, "prem_due -= 1")
    line(12, "prem_left -= 1")
    line(12, "pv_premium += level * ds")
    if use_surrender and not surrender_is_amount:
        line(12, "cum_premium += level")
    line(12, "pv_mortality += inforce_t * claim_rate * dm")
    line(12, "pv_morbidity += inforce_t * morb_rate * dm")
    if use_annuity:
        line(12, "if ann_due == 0:")
        line(16, "pv_annuity += inforce_t * annuity * annuity_factor[sx, age_idx, year] * ds")
        line(16, "ann_due = ann_freq - 1")
        line(12, "else:")
        line(16, "ann_due -= 1")
    line(12, "pv_disability += benefit_occ * disability_income[mp] * dm")
    if use_surrender:
        line(12, f"lapse_flow = {sum_lapse}")
        if surrender_is_amount:
            # amount_per_policy / amount_per_unit: surrender_curve[t] is the
            # surrender amount at duration t (per policy, or per unit of
            # surrender_base[mp]); lapse_flow is the number lapsing.
            # surrender_base is 1.0 for amount_per_policy.
            line(12, "pv_surrender += (lapse_flow * surrender_curve[t]")
            line(12, "                 * surrender_base[mp] * dm)")
        else:
            # cum_premium_factor: cum_premium aggregates inforce * premium; the
            # effective lapse fraction is lapse_flow / inforce_t (the raw rate for a
            # single state). inforce_t carries the count, so dividing it out avoids a
            # cnt^2 scaling.
            line(12, "eff_lapse = lapse_flow / inforce_t if inforce_t > 0.0 else 0.0")
            line(12, "pv_surrender += (eff_lapse")
            line(12, "                 * cum_premium * surrender_curve[t] * dm)")
    line(12, "alpha = cnt * (alpha_pro_rata * ann_prem + alpha_fixed) if t == 0 else 0.0")
    line(12, "beta = inforce_t * beta_pro_rata * ann_prem / 12.0 if t < premium_term else 0.0")
    line(12, "gamma = inforce_t * gamma_fixed[t]")
    if use_lae:
        line(12, "lae = lae_pro_rata[t] * "
                "inforce_t * (claim_rate + morb_rate)")
        line(12, "pv_expense += (alpha + beta + gamma + lae) * dm")
    else:
        line(12, "pv_expense += (alpha + beta + gamma) * dm")
    emit_edge_step(12, include_lump=True)

    line(8, f"total = {sum_all}")
    line(8, "pm = (total * maturity_benefit[mp] * discount_factor_bom[boundary]) "
         "if boundary == term else 0.0")

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
    line(12, "for t in range(wait, boundary):")
    line(16, "year = t // 12")
    line(16, "mult = red_factor if t < red_end else 1.0")
    line(16, "contrib = (inforce_traj[t] * coverage_rates[cov_idx, sx, age_idx, year]")
    line(16, "           * benefit * mult * discount_factor_mid[t])")
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
    line(12, "for t in range(boundary):")
    line(16, "year = t // 12")
    line(16, "if year != d_year:")
    line(20, "d_rate = coverage_rates[cov_idx, sx, age_idx, year]")
    line(20, "d_year = year")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, "pv_morbidity += (inforce_traj[t] * undiagnosed * d_rate * benefit")
    line(20, "                 * mult * discount_factor_mid[t])")
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
    n_states, state_duration_max, periodic_benefit_term_months,
    edge_from, edge_to, edge_lump_sum,
    premium_state, benefit_state,
    use_annuity=True, use_lae=True, use_surrender=True,
    surrender_is_amount=False,
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
        # ``periodic_benefit_term_months`` changes which cohorts ``sum_ben`` pays, so
        # two models that differ only by the cap MUST NOT share a kernel --
        # omitting it here would serve a stale kernel and silently mis-state
        # the disability cash flow / BEL.
        tuple(int(x) for x in periodic_benefit_term_months),
        tuple(int(x) for x in edge_from),
        tuple(int(x) for x in edge_to),
        tuple(bool(x) for x in edge_lump_sum),
        tuple(bool(x) for x in premium_state),
        tuple(bool(x) for x in benefit_state),
        bool(use_annuity), bool(use_lae), bool(use_surrender),
        bool(surrender_is_amount),
    )
    cached = _FAST_KERNEL_CODEGEN_SEMI_MARKOV_CACHE.get(key)
    if cached is not None:
        return cached

    src = _codegen_fast_kernel_source_semi_markov(
        n_states, state_duration_max, periodic_benefit_term_months,
        edge_from, edge_to, edge_lump_sum,
        premium_state, benefit_state,
        use_annuity=use_annuity, use_lae=use_lae, use_surrender=use_surrender,
        surrender_is_amount=surrender_is_amount,
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
def _fast_kernel_scalar(issue_index, sex, term_months, contract_boundary_months,
                         count, premium,
                         premium_term_months, premium_frequency_months,
                         annuity_frequency_months, coverage_index, coverage_amount, coverage_offset,
                         coverage_rates, premium_factor, annuity_factor, coverage_risk, coverage_is_diagnosis,
                         maturity_benefit, annuity_payment,
                         alpha_pro_rata, alpha_fixed, beta_pro_rata,
                         gamma_fixed, lae_pro_rata,
                         discount_factor_bom, discount_factor_mid,
                         mortality_factor, morbidity_factor, longevity_factor,
                         coverage_waiting, coverage_reduction_end, coverage_reduction_factor,
                         coverage_pays_account_balance,
                         survival_monthly, lapse_monthly, surrender_curve,
                         use_morbidity, use_annuity, use_lae, use_surrender,
                         surrender_is_amount, surrender_base,
                         mortality_monthly,
                         has_account, mp_account, account_value0, account_face,
                         account_prem_to_av, account_coi_rate, account_admin_fee,
                         account_credit, account_charge, account_gmab,
                         account_mortality_cv, account_expense_cv, account_z):
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
        boundary = contract_boundary_months[mp]
        premium_term = premium_term_months[mp]
        prem_freq = premium_frequency_months[mp]
        ann_freq = annuity_frequency_months[mp]
        age_idx = issue_index[mp]
        sx = sex[mp]
        cnt = count[mp]
        prem = premium[mp]
        annuity = annuity_payment[mp]
        c_start = coverage_offset[mp]
        c_end = coverage_offset[mp + 1]
        inforce_t = cnt
        pv_mortality = 0.0  # mortality-risk claim PV (death claims)
        pv_morbidity = 0.0  # morbidity-risk claim PV (health claims)
        pv_premium = 0.0
        pv_expense = 0.0
        pv_annuity = 0.0
        pv_surrender = 0.0  # PV of surrender value
        cum_premium = 0.0  # cumulative premium paid -- surrender basis
        last_year = -1
        claim_rate = 0.0
        morb_rate = 0.0
        # Universal-life account-value carrier (only rolled for account rows).
        # The account is a single policy's balance, rolled with the within-month
        # order verbatim from the standalone UL kernel; in-force weighting enters
        # at the death / surrender / maturity / fund aggregation, not in the roll.
        roll_av = has_account and mp_account[mp]
        a_av = account_value0[mp] if roll_av else 0.0
        face_av = account_face[mp] if roll_av else 0.0
        cr_av = account_credit[mp] if roll_av else 0.0
        half_credit_av = (1.0 + cr_av) ** 0.5
        full_credit_av = 1.0 + cr_av
        pv_account_death = 0.0  # PV of account death claim, inforce_t*q*max(av_mid,face)
        pv_account_surr = 0.0   # PV of account surrender, inforce_t*l*(1-q)*av_mid
        pv_account_nar = 0.0    # PV of net-amount-at-risk death (the UL RA base)
        av_term = 0.0           # month-end account value at the contract boundary
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
        for t in range(boundary):
            year = t // 12
            if year != last_year:
                claim_rate = 0.0
                morb_rate = 0.0
                for k in range(c_start, c_end):
                    cov_idx = coverage_index[k]
                    if coverage_is_diagnosis[cov_idx]:
                        continue
                    if coverage_pays_account_balance[cov_idx]:
                        continue          # account-backed death pays from the AV
                    if coverage_waiting[k] != 0 or coverage_reduction_end[k] != 0:
                        continue
                    rate = coverage_rates[cov_idx, sx, age_idx, year] * coverage_amount[k]
                    if coverage_risk[cov_idx] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year
            ds = discount_factor_bom[t]
            dm = discount_factor_mid[t]
            if prem_due == 0 and prem_left > 0:
                level = inforce_t * prem * premium_factor[sx, age_idx, year]
                prem_due = prem_freq - 1
            else:
                level = 0.0
                prem_due -= 1
            prem_left -= 1
            pv_premium += level * ds
            pv_mortality += inforce_t * claim_rate * dm
            if roll_av:
                # Universal-life within-month account roll (the account-roll
                # within-month order): premium in, COI on the net amount at risk,
                # admin fee, every cost-deducting rider's fixed charge, floor at
                # zero, half / full crediting. The account
                # death / surrender / NAR claims settle on the half-credited
                # av_mid, the maturity on the month-end balance.
                a_av += account_prem_to_av[mp, t]
                coi_nar = face_av - a_av
                if coi_nar < 0.0:
                    coi_nar = 0.0
                a_av -= (account_admin_fee[t] + account_coi_rate[mp, t] * coi_nar
                         + account_charge[mp, t])
                if a_av < 0.0:
                    a_av = 0.0
                av_mid_t = a_av * half_credit_av
                a_av = a_av * full_credit_av
                av_term = a_av
                # Account death pays max(av_mid, face) on the month's deaths
                # (inforce_t * q); the pays_account_balance coverage was excluded
                # from claim_rate above, so this is written ONCE here.
                q_m = mortality_monthly[sx, age_idx, year]
                deaths_m = inforce_t * q_m
                best = av_mid_t if av_mid_t > face_av else face_av
                pv_account_death += deaths_m * best * dm
                # The net amount at risk (face above the account) is the only
                # insurance-risk exposure -- the UL RA prices mortality on it.
                nar = face_av - av_mid_t
                if nar < 0.0:
                    nar = 0.0
                pv_account_nar += deaths_m * nar * dm
                # Surrender pays the account value on the mid-month lapse exits.
                # The non-maturity, non-death exit count in the single-survival
                # track is inforce_t * l * (1 - q) (the lapses net of the competing
                # death decrement) -- matching the full path's exits - deaths.
                l_m = lapse_monthly[sx, age_idx, year]
                surr_av = av_mid_t if av_mid_t > 0.0 else 0.0
                pv_account_surr += inforce_t * l_m * (1.0 - q_m) * surr_av * dm
            # Streams the portfolio does not use are skipped wholesale: the
            # guards are loop-invariant per call, so the branch predicts
            # perfectly and the per-cell array gathers / multiplies of an
            # absent stream are not paid. A death-only book pays for none of
            # surrender, annuity, morbidity or LAE.
            if use_morbidity:
                pv_morbidity += inforce_t * morb_rate * dm
            if use_annuity:
                if ann_due == 0:
                    pv_annuity += inforce_t * annuity * annuity_factor[sx, age_idx, year] * ds
                    ann_due = ann_freq - 1
                else:
                    ann_due -= 1
            ann_prem = prem * premium_factor[sx, age_idx, year] * 12.0 / prem_freq
            alpha = (cnt * (alpha_pro_rata * ann_prem + alpha_fixed)
                     if t == 0 else 0.0)
            beta = (inforce_t * beta_pro_rata * ann_prem / 12.0
                    if t < premium_term else 0.0)
            gamma = inforce_t * gamma_fixed[t]
            if use_lae:
                lae = lae_pro_rata[t] * inforce_t * (claim_rate + morb_rate)
                pv_expense += (alpha + beta + gamma + lae) * dm
            else:
                pv_expense += (alpha + beta + gamma) * dm
            if use_surrender:
                if surrender_is_amount:
                    # amount_per_policy / amount_per_unit: surrender_curve[t] is
                    # the surrender amount at duration t (per policy, or per
                    # unit of surrender_base[mp]); inforce_t * lapse_rate is the
                    # number lapsing. surrender_base is 1.0 for amount_per_policy.
                    pv_surrender += (lapse_monthly[sx, age_idx, year]
                                     * inforce_t * surrender_curve[t]
                                     * surrender_base[mp] * dm)
                else:
                    # cum_premium_factor: cum_premium aggregates inforce_t *
                    # premium and is the surrender basis; multiplying by
                    # lapse_rate alone gives the per-month surrender outflow
                    # (the count is already in cum_premium).
                    cum_premium += level
                    pv_surrender += (lapse_monthly[sx, age_idx, year]
                                     * cum_premium * surrender_curve[t] * dm)
            inforce_t *= survival_monthly[sx, age_idx, year]
        if roll_av:
            # Account maturity: survivors reaching the boundary take
            # max(matured account value, GMAB) (the maturity benefit doubles as
            # the guaranteed accumulation-benefit floor). Seeded at the boundary
            # and discounted with the boundary start-of-month factor, mirroring
            # the full path's _rollforward_kernel maturity seed.
            gmab = account_gmab[mp]
            mat_av = av_term if av_term > gmab else gmab
            pm = (inforce_t * mat_av * discount_factor_bom[boundary]
                  if boundary == term else 0.0)
        else:
            pm = (inforce_t * maturity_benefit[mp] * discount_factor_bom[boundary]
                  if boundary == term else 0.0)
        # Non-diagnosis coverages with a waiting or reduced-benefit rule:
        # rerun the survival on the same scalar track so the benefit
        # multiplier (which can change mid-year) applies cleanly.
        for k in range(c_start, c_end):
            cov_idx = coverage_index[k]
            if coverage_is_diagnosis[cov_idx]:
                continue
            if coverage_pays_account_balance[cov_idx]:
                continue          # account-backed death pays from the AV (above)
            wait = coverage_waiting[k]
            red_end = coverage_reduction_end[k]
            if wait == 0 and red_end == 0:
                continue
            benefit = coverage_amount[k]
            red_factor = coverage_reduction_factor[k]
            mortality_risk = coverage_risk[cov_idx] == 0
            inforce_t = cnt
            for t in range(boundary):
                year = t // 12
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    contrib = (inforce_t * coverage_rates[cov_idx, sx, age_idx, year]
                               * benefit * mult * discount_factor_mid[t])
                    if mortality_risk:
                        pv_mortality += contrib
                    else:
                        pv_morbidity += contrib
                inforce_t *= survival_monthly[sx, age_idx, year]
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
            for t in range(boundary):
                year = t // 12
                if year != d_year:
                    d_rate = coverage_rates[cov_idx, sx, age_idx, year]
                    d_year = year
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    pv_morbidity += healthy * d_rate * benefit * mult * discount_factor_mid[t]
                healthy *= survival_monthly[sx, age_idx, year] * (1.0 - d_rate)
        if roll_av:
            # Universal-life BEL: the slot mortality / surrender streams are
            # empty for the account leg (the pays_account_balance coverage was
            # excluded from claim_rate, and account surrender bypasses the slot
            # surrender path), so the account death / surrender / maturity PVs
            # enter here instead. The account value the entity holds (fund) is
            # netted ONCE -- only the inception fund (inforce[0] * av0 = cnt*av0)
            # survives into the as-of BEL, the premium that builds it counted
            # once via pv_premium. The RA prices mortality on the net amount at
            # risk plus expense risk, bypassing the slot RA factors.
            bel_mp = (pv_account_death + pv_account_surr + pm
                      + pv_morbidity + pv_annuity + pv_expense
                      - pv_premium - cnt * account_value0[mp])
            # A cost-deducting rider's health benefit (pv_morbidity) bears
            # morbidity risk -- priced alongside the at-risk mortality and
            # expense (morbidity_factor == account_z * morbidity_cv). Zero for an
            # account book with no such rider, so byte-identical.
            ra_mp = (account_z * (account_mortality_cv * pv_account_nar
                                  + account_expense_cv * pv_expense)
                     + morbidity_factor * pv_morbidity)
        else:
            bel_mp = pv_mortality + pv_morbidity + pm + pv_annuity + pv_expense + pv_surrender - pv_premium
            ra_mp = (mortality_factor * pv_mortality + morbidity_factor * pv_morbidity
                     + longevity_factor * (pm + pv_annuity))
        fcf = bel_mp + ra_mp
        bel[mp] = bel_mp
        ra[mp] = ra_mp
        csm[mp] = max(0.0, -fcf)
        loss_component[mp] = max(0.0, fcf)

    return bel, ra, csm, loss_component


def _portfolio_has_account(model_points: ModelPoints, basis: Basis) -> bool:
    """True when any coverage carries a universal-life account-chassis flag.

    Derived STRICTLY from the per-coverage ``funds_from_account`` /
    ``pays_account_balance`` flags read off the :class:`CoverageRate` objects
    (never ``account_value != 0``, which would wrongly flip the variable-annuity
    product onto the recursive roll). Deliberately does NOT resolve calculation
    methods -- the flags are independent of the method.
    """
    return any(getattr(r, "funds_from_account", False)
               or getattr(r, "pays_account_balance", False)
               for r in basis.coverages)


def requires_full(model_points: ModelPoints, basis: Basis) -> bool:
    """True when a book uses a mechanic the fused fast path does not apply.

    The full-only features (v1): non-zero ``issue_class`` (the fast grid is built
    at class 0), benefit escalation / step-up, a state-conditioned death benefit
    (``State.death_benefit_factor != 1``) and a deterministic transition
    (``Transition.after_sojourn_months``). A universal-life account is NOT in this
    set as of Step 4: the scalar fused kernel carries the account roll itself, so
    an account book runs on the fast path directly (the routing exceptions --
    account + state machine, account + gpu, account + discount_curve override --
    are handled in :func:`_measure_fast`, not here).
    The fast path auto-routes such a book to the full kernel instead of raising,
    so -- per segment in ``_measure_segmented`` -- only the segments that need it
    pay, and the rest stay fast. The seed of the planned portfolio-orchestrator
    FULL tier.
    """
    if np.any(model_points.issue_class != 0):
        return True
    if (model_points.coverage_step_month is not None
            and (np.any(model_points.coverage_step_month)
                 or np.any(model_points.coverage_escalation_annual))):
        return True
    sm = resolve_state_model(basis)
    if any(s.death_benefit_factor != 1.0 for s in sm.states):
        return True
    if any(tr.after_sojourn_months for s in sm.states for tr in s.transitions):
        return True
    return False


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
        # The fused fast path computes the confidence-level RA only. A
        # universal-life account book auto-routes to the full measurement (which
        # prices the cost-of-capital RA on the net amount at risk) when it can --
        # on the CPU path with no discount_curve override; otherwise a clear
        # error. A non-account book tells the user to use full=True.
        if (_portfolio_has_account(model_points, basis)
                and backend == "cpu" and discount_curve is None):
            return _measure_full(model_points, basis)
        raise ValueError(
            "measure(full=False) computes the confidence-level RA only; use "
            f"full=True for ra_method={basis.ra_method!r}"
        )
    # Full-only mechanics (issue_class != 0, benefit escalation / step-up,
    # state-conditioned death_benefit_factor, deterministic after_sojourn_months)
    # are not applied by the fused fast path. Rather than reject, route the whole
    # call to the full (trajectory) kernel -- correct, just slower for this book.
    # In _measure_segmented this runs per segment, so only the segments that need
    # it pay; the rest stay on the fast path. A gpu / discount_curve override
    # cannot ride the CPU full kernel, so that exotic combination still raises.
    if requires_full(model_points, basis):
        if backend != "cpu" or discount_curve is not None:
            raise NotImplementedError(
                "a full-path-only mechanic (issue_class / benefit escalation / "
                "state death_benefit_factor / deterministic transition) cannot be "
                "combined with backend='gpu' or a discount_curve override; use "
                "measure(full=True) on the CPU path."
            )
        return _measure_full(model_points, basis)
    # Universal-life account book. The scalar fused kernel carries the account
    # roll itself (Step 4), so a plain account book runs the fast path directly.
    # The combinations the scalar kernel does not cover still route to the
    # fund-netting full measurement: (a) a state machine -- the account roll is
    # folded only into the scalar kernel, NOT the codegen Markov / semi-Markov
    # fused sources; (b) backend='gpu' -- the CUDA account carrier is deferred;
    # (c) a discount_curve override -- the account RA / fund netting are wired
    # through the standard curve. cost_of_capital RA already routed to full via
    # the confidence-level guard at the top.
    has_account = _portfolio_has_account(model_points, basis)
    # A contract boundary shorter than the term (Sec. 34 cut) pays the boundary
    # survivors their account value as a terminal surrender; the full path's
    # exit accounting handles that, the scalar account fold (v1) does not, so
    # route a boundary-cut account book to the full measurement.
    account_boundary_cut = has_account and bool(np.any(
        model_points.contract_boundary_months < model_points.term_months))
    # The scalar fast kernel does not yet carry the annuitization phase switch
    # (a later step); route an annuitizing account book to the full measurement.
    account_annuitizing = has_account and bool(np.any(
        model_points.annuitization_months > 0))
    if has_account and (backend != "cpu" or discount_curve is not None
                        or needs_state_machine(model_points, basis)
                        or account_boundary_cut or account_annuitizing):
        if backend != "cpu" or discount_curve is not None:
            raise NotImplementedError(
                "a universal-life account book cannot be combined with "
                "backend='gpu' or a discount_curve override on the fused path; "
                "use measure(full=True) on the CPU path (the account roll is "
                "folded into the scalar fast kernel and the full kernel only)."
            )
        return _measure_full(model_points, basis)
    if model_points.term_months.shape[0] == 0:
        raise ValueError(
            "model_points is empty (n_mp=0); measure(full=False) cannot project a "
            "zero-policy portfolio. Filter empty segments upstream."
        )
    # The projection horizon is the contract boundary (defaults to the term).
    n_time = int(model_points.contract_boundary_months.max())
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
    # Premium SHAPE on the dense (sex, age, year) grid -- a multiplicative scale
    # on the level premium, the value-side mirror of project_cashflows'
    # premium_factor. NOT a rate: never annual_to_monthly (a step-up > 1.0 would
    # fail its <= 1 check). None -> all-ones (level), a structural no-op. The
    # grid is issue_class=0 / elapsed=0 (non-segmented fast path), so the factor
    # is a pure (sex, age, year) function here.
    if basis.premium_factor_annual is None:
        premium_factor_grid = np.ones_like(mortality_grid)
    else:
        premium_factor_grid = validate_factor(
            basis.premium_factor_annual(
                sex_grid, issue_age_grid, duration_grid,
                issue_class_grid, elapsed_grid),
            "premium_factor_annual", mortality_grid.shape)
    # Annuity SHAPE on the dense grid -- the survival-benefit twin of
    # premium_factor_grid (escalating annuity). Same rules: never
    # annual_to_monthly, None -> all-ones (level annuity), a no-op multiply.
    if basis.annuity_factor_annual is None:
        annuity_factor_grid = np.ones_like(mortality_grid)
    else:
        annuity_factor_grid = validate_factor(
            basis.annuity_factor_annual(
                sex_grid, issue_age_grid, duration_grid,
                issue_class_grid, elapsed_grid),
            "annuity_factor_annual", mortality_grid.shape)
    # Fast path: when no waiver / paid-up mechanic is active and every model
    # point is seated in the active state, the in-force is a single survival
    # track. The scalar kernel carries it as one number and runs the
    # pre-Phase(b) speed path; the N-state kernel is reserved for products
    # that genuinely need an occupancy vector.
    fast_path = (backend == "cpu"
                 and not needs_state_machine(model_points, basis))
    if not fast_path:
        if basis.waiver_incidence_annual is None:
            waiver_grid = np.zeros_like(mortality_grid)
        else:
            waiver_grid = np.ascontiguousarray(annual_to_monthly(
                basis.waiver_incidence_annual(
                    sex_grid, issue_age_grid, duration_grid,
                    issue_class_grid, elapsed_grid)))
        # In-force state machine -- see ``state_model.resolve_state_model``
        # for the fallback policy when ``basis.state_model`` is unset.
        # The transition rates land on the sex x age x duration grid the
        # kernel indexes.
        state_model = resolve_state_model(basis)
        semi_markov = is_semi_markov(state_model)
        if semi_markov:
            # Phase (c) path -- a state declared ``sojourn_tracking_months > 0`` tracks
            # per-cohort occupancy. The rate dict carries one entry per name
            # the model references; duration-dependent rates land here as
            # 4D arrays (sex, age, year, cohort), static rates as 3D.
            max_cohort = max(s.sojourn_tracking_months for s in state_model.states
                              if s.sojourn_tracking_months > 0)
            rate_dict = {"mortality": mortality_grid,
                          "lapse": lapse_grid}
            _add_state_mortality_rates(rate_dict, state_model, basis,
                                       sex_grid, issue_age_grid, duration_grid,
                                       issue_class_grid, elapsed_grid)
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
                # (the sojourn-time cohort index) lets an exclusion
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
            state_lapse_grid = _state_lapse_stack(state_model, rate_dict)
            edge_from = compiled.edge_from
            edge_to = compiled.edge_to
            edge_lump_sum = compiled.edge_lump_sum
            n_states = compiled.n_states
            premium_state = compiled.premium_state
            benefit_state = compiled.benefit_state
            state_duration_max = compiled.state_duration_max
            periodic_benefit_term_months = compiled.periodic_benefit_term_months
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
            _add_state_mortality_rates(rate_dict, state_model, basis,
                                       sex_grid, issue_age_grid, duration_grid,
                                       issue_class_grid, elapsed_grid)
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
            state_lapse_grid = _state_lapse_stack(state_model, rate_dict)
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
        seating = np.asarray(state_model.seating, np.int64)
        if model_points.state.size and int(model_points.state.max()) >= seating.shape[0]:
            raise ValueError(
                f"ModelPoints.state has value {int(model_points.state.max())} but "
                f"the resolved state model accepts only {seating.shape[0]} seating "
                f"states (valid 0..{seating.shape[0] - 1}); check the state column "
                "against the segment's state_model")
        start_state = seating[model_points.state]
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
    (coverage_is_diagnosis, coverage_risk,
     coverage_funds_from_account, coverage_pays_account_balance) = coverage_arrays(
        aligned_coverages, model_points.calculation_methods,
    )
    # coverage_funds_from_account / coverage_pays_account_balance are the
    # account-chassis interaction flags (all-False today; the universal-life
    # account roll folds onto them in a later step). Unused here for now.
    # build_coverage_rates stacks the per-coverage annual rates; the whole
    # stack is converted to monthly. mortality_annual is a separate engine
    # input (the in-force decrement); a contract's death coverage, if any,
    # lives in basis.coverages with its own rate_table -- usually the
    # same mortality table referenced from that sheet, occasionally a
    # separately calibrated death-claim experience table.
    coverage_rates = np.ascontiguousarray(annual_to_monthly(build_coverage_rates(
        [r.rate for r in aligned_coverages], sex_grid,
        issue_age_grid, duration_grid, issue_class_grid, elapsed_grid,
        codes=[r.code for r in aligned_coverages],
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
        discount_factor_bom, discount_factor_mid = discount_factors(basis, n_time)
    else:
        discount_curve = np.asarray(discount_curve, dtype=np.float64)
        if discount_curve.shape != (n_time,):
            raise ValueError(
                f"discount_curve must have shape ({n_time},) -- one annual "
                f"rate per projection month -- got {discount_curve.shape}"
            )
        monthly_curve = (1.0 + discount_curve) ** (1.0 / 12.0) - 1.0
        discount_factor_bom, discount_factor_mid = discount_factors_from_curve(monthly_curve)
    if basis.expense_cv != 0.0 and not has_account:
        raise NotImplementedError(
            "expense_cv is not included in the GMM / PAA risk adjustment -- only "
            "the mortality / morbidity / disability / longevity risks are priced "
            "(there is no expense-risk PV in this RA). Set expense_cv=0 for a "
            "GMM / PAA measurement. (The VFA RA does price expense_cv.)"
        )
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
    surr_mode = basis.surrender_value_basis
    if surr_user is not None and surr_mode not in SURRENDER_VALUE_BASES:
        raise ValueError(
            f"unknown surrender_value_basis {surr_mode!r}; expected one of "
            f"{SURRENDER_VALUE_BASES}."
        )
    # amount_per_policy / amount_per_unit: the curve is a surrender amount
    # applied to the in-force scalar (and, for amount_per_unit, to a per-MP
    # base). cum_premium_factor: a factor on cumulative premium. The kernels
    # branch on surrender_is_amount and multiply by surrender_base, which is
    # 1.0 for amount_per_policy / cum_premium_factor.
    surrender_is_amount = surr_mode in ("amount_per_policy", "amount_per_unit")
    if surr_user is not None and surr_mode == "amount_per_unit":
        base = model_points.surrender_base_amount
        if base is None:
            raise ValueError(
                "surrender_value_basis='amount_per_unit' requires "
                "ModelPoints.surrender_base_amount (no default base is "
                "inferred)."
            )
        surrender_base = np.asarray(base, dtype=np.float64)
    else:
        surrender_base = np.ones(model_points.n_mp, dtype=np.float64)
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

    # Universal-life account-roll inputs for the scalar fused kernel (Step 4).
    # Reuses projection._account_kernel_args -- the SAME per-policy prem_to_av /
    # COI / admin / credit arithmetic the full path folds -- so the fast path is
    # bit-identical to the full path. The helper indexes a per-MP coverage-rate
    # view (cov, mp, year); build it from the dense (cov, sex, age, year) grid at
    # each MP's (sex, age). The fast path only reaches here at issue_class 0 (the
    # requires_full guard routes class != 0 to full), so the dense-grid lookup
    # equals the per-MP rate. account_admin_fee carries gamma_fixed.
    if has_account:
        from fastcashflow.projection import _account_kernel_args
        coverage_rates_per_mp = np.ascontiguousarray(
            coverage_rates[:, model_points.sex, issue_index, :])  # (cov, mp, year)
        (acct_has, acct_mp, acct_value0, acct_face,
         acct_prem_to_av, acct_coi_rate, acct_admin_fee,
         acct_credit, acct_charge) = _account_kernel_args(
            model_points, basis, coverage_rates_per_mp,
            coverage_funds_from_account, coverage_pays_account_balance,
            gamma_fixed, n_time, n_years,
        )
        # v1 supports a homogeneous account portfolio only (every MP carries the
        # account-backed death coverage). A mixed account / plain book needs a
        # per-MP RA split; reject rather than mis-price, mirroring the full path.
        if not bool(acct_mp.all()):
            raise NotImplementedError(
                "a portfolio mixing account-backed (universal-life) and plain "
                "model points is not yet supported -- measure the account and "
                "non-account subsets separately. (Per-model-point RA splitting "
                "for mixed books is a planned follow-up.)")
        acct_gmab = np.ascontiguousarray(
            np.asarray(model_points.maturity_benefit, np.float64))
        acct_mortality_cv = float(basis.mortality_cv)
        acct_expense_cv = float(basis.expense_cv)
        acct_z = float(z)
    else:
        acct_has = False
        acct_mp = np.zeros(1, np.bool_)
        z1 = np.zeros((1, 1))
        acct_value0 = np.zeros(1)
        acct_face = np.zeros(1)
        acct_prem_to_av = z1
        acct_coi_rate = z1
        acct_admin_fee = np.zeros(1)
        acct_credit = np.zeros(1)
        acct_charge = z1
        acct_gmab = np.zeros(1)
        acct_mortality_cv = 0.0
        acct_expense_cv = 0.0
        acct_z = 0.0
    mortality_monthly = np.ascontiguousarray(mortality_grid)

    if fast_path:
        survival_monthly = np.ascontiguousarray(
            (1.0 - mortality_grid) * (1.0 - lapse_grid)
        )
        bel, ra, csm, loss_component = _fast_kernel_scalar(
            issue_index,
            model_points.sex,
            model_points.term_months,
            model_points.contract_boundary_months,
            model_points.count,
            model_points.premium,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.coverage_index,
            coverage_amount,
            model_points.coverage_offset,
            coverage_rates,
            premium_factor_grid,
            annuity_factor_grid,
            coverage_risk,
            coverage_is_diagnosis,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            expense_alpha_pro_rata,
            expense_alpha_fixed,
            expense_beta_pro_rata,
            gamma_fixed,
            lae_pro_rata,
            discount_factor_bom,
            discount_factor_mid,
            mortality_factor,
            morbidity_factor,
            longevity_factor,
            model_points.coverage_waiting,
            model_points.coverage_reduction_end,
            model_points.coverage_reduction_factor,
            coverage_pays_account_balance,
            survival_monthly,
            lapse_grid,
            surrender_curve_kernel,
            use_morbidity,
            use_annuity,
            use_lae,
            use_surrender,
            surrender_is_amount,
            surrender_base,
            mortality_monthly,
            acct_has,
            acct_mp,
            acct_value0,
            acct_face,
            acct_prem_to_av,
            acct_coi_rate,
            acct_admin_fee,
            acct_credit,
            acct_charge,
            acct_gmab,
            acct_mortality_cv,
            acct_expense_cv,
            acct_z,
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
        model_points.contract_boundary_months,
        model_points.count,
        model_points.premium,
        model_points.premium_term_months,
        model_points.premium_frequency_months,
        model_points.annuity_frequency_months,
        model_points.coverage_index,
        coverage_amount,
        model_points.coverage_offset,
        coverage_rates,
        premium_factor_grid,
        annuity_factor_grid,
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
        discount_factor_bom,
        discount_factor_mid,
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
                n_states, state_duration_max, periodic_benefit_term_months,
                edge_from, edge_to,
                edge_lump_sum, premium_state, benefit_state,
                use_annuity=use_annuity, use_lae=use_lae,
                use_surrender=use_surrender,
                surrender_is_amount=surrender_is_amount,
            )
            bel, ra, csm, loss_component = kernel(
                edge_prob, start_state, issue_index,
                model_points.sex,
                model_points.term_months,
                model_points.contract_boundary_months,
                model_points.count,
                model_points.premium,
                    model_points.premium_term_months,
                model_points.premium_frequency_months,
                model_points.annuity_frequency_months,
                model_points.coverage_index,
                coverage_amount,
                model_points.coverage_offset,
                coverage_rates,
                premium_factor_grid,
                annuity_factor_grid,
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
                discount_factor_bom,
                discount_factor_mid,
                mortality_factor,
                morbidity_factor,
                longevity_factor,
                disability_factor,
                model_points.coverage_waiting,
                model_points.coverage_reduction_end,
                model_points.coverage_reduction_factor,
                lapse_grid,
                state_lapse_grid,
                surrender_curve_kernel,
                surrender_base,
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
                surrender_is_amount=surrender_is_amount,
            )
            bel, ra, csm, loss_component = kernel(
                *common_args, model_points.coverage_waiting,
                model_points.coverage_reduction_end,
                model_points.coverage_reduction_factor,
                lapse_grid,
                state_lapse_grid,
                surrender_curve_kernel,
                surrender_base,
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
            lapse_grid, state_lapse_grid, surrender_curve_kernel, surrender_is_amount,
            surrender_base,
        )
    else:
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")

    return GMMMeasurement(bel=bel, ra=ra, csm=csm, loss_component=loss_component)


def _require_gmm_router(router, *, entry: str) -> None:
    """Reject a non-GMM router from a GMM-only entry point, before any work.

    A mixed-model portfolio must go through ``fcf.portfolio.measure``; silently
    measuring a PAA or VFA segment with the GMM kernel would return a finite,
    plausible, wrong number. Checked over the router's own segment keys
    (self-consistent, so immune to the key-normalisation _factorise_segments
    applies) up front, so no segment is measured before the mismatch is caught.
    """
    if not hasattr(router, "measurement_model_of"):
        return
    for key in router.segments:
        model = router.measurement_model_of(key)
        if model != "GMM":
            raise ValueError(
                f"segment {key!r} uses measurement_model={model!r}; {entry} "
                f"measures GMM segments only. Use fcf.portfolio.measure for a "
                f"mixed-model portfolio."
            )


def _measure_segmented(
    model_points: ModelPoints,
    basis: dict[tuple[str, str], Basis],
    *,
    backend: str = "cpu",
    discount_curve: FloatArray | None = None,
    segment_by=("product", "channel"),
) -> GMMMeasurement:
    """Value a multi-segment portfolio: split, value each, concatenate.

    ``basis`` is the ``{(product, channel): Basis}`` dictionary
    returned by :func:`fastcashflow.read_basis`. ``model_points``
    must carry ``product`` and ``channel`` columns identifying each row's
    segment; for each unique (product, channel) the helper masks the
    matching rows, builds a sub-:class:`~fastcashflow.ModelPoints` via
    :meth:`~fastcashflow.ModelPoints.subset`, calls ``measure(..., full=False)`` with the
    segment's ``Basis``, and writes the per-row results back to a
    single ``(n_mp,)`` :class:`GMMMeasurement`.

    ``backend`` and ``discount_curve`` flow through to ``measure(..., full=False)`` --
    declared explicitly so a typo (e.g. ``backed="gpu"``) is rejected
    here rather than reaching the kernel. A single-segment ``basis`` is
    accepted as a convenience when ``product`` / ``channel`` is
    not set.
    """
    _require_gmm_router(basis, entry="fcf.gmm.measure")
    try:
        basis_norm, segments = _factorise_segments(
            basis, model_points, segment_by, model_points.n_mp,
        )
    except KeyError:
        if len(basis.segments) == 1:
            (basis,) = basis.segments.values()
            return _measure_fast(
                model_points, basis,
                backend=backend, discount_curve=discount_curve,
            )
        raise ValueError(
            f"model_points has no {tuple(segment_by)} axis/axes set but the "
            f"basis has {len(basis.segments)} segments; either set the columns or "
            "pass a single-segment basis"
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


def _factorise_segments(basis, model_points: ModelPoints, segment_by, n_mp):
    """Resolve a multi-segment portfolio to per-segment row indices.

    ``basis`` is ``{key: Basis}`` keyed by a tuple of the ``segment_by`` axes in
    order (a bare value is accepted for a one-axis key). Each axis is looked up
    via :meth:`ModelPoints.axis` (so a routing key can mix the segment fields and
    any ``attributes`` column) and NFC-normalised, so the lookup is text-identity
    not byte-identity (a Korean / European character composed in one file and
    decomposed in the other would otherwise compare unequal). Raises
    :class:`KeyError` if an axis is not set -- the caller turns that into the
    single-basis convenience or a clear error.

    Returns ``(basis_norm, segments)`` where ``basis_norm`` is ``basis`` re-keyed
    under NFC-normalised tuples and ``segments`` is ``[(key, idx)]`` in first-seen
    order. The factorisation hashes the per-row axis-value *tuple* directly (one
    ``O(n_mp)`` pass), avoiding a per-row ``'|'``-join and the ``O(n_mp log
    n_mp)`` object-string sort ``np.unique`` would do -- the hot path when a
    large portfolio is split by segment.
    """
    norm = unicodedata.normalize
    basis_norm = {}
    for k, a in basis.segments.items():
        parts = k if isinstance(k, tuple) else (k,)
        basis_norm[tuple(norm("NFC", str(p)) for p in parts)] = a
    # Resolve + NFC-normalise each axis. ``axis`` raises KeyError for an unset
    # axis (caught by the caller). Object arrays so the values stay python str.
    axes = [
        np.array([norm("NFC", str(v)) for v in model_points.axis(name)],
                 dtype=object)
        for name in segment_by
    ]
    # '|' stays reserved (group_of_contracts joins labels with it); reject it so
    # a segment key still round-trips losslessly through the grouping layer.
    for col, name in zip(axes, segment_by):
        bad = sorted({v for v in col if "|" in v})
        if bad:
            raise ValueError(
                f"{name} value(s) {bad} contain the '|' character, which the "
                "grouping layer uses as the label separator. Pick a different "
                "separator in your ETL or rename the offending code."
            )
    # Hash-factorise the axis-value tuple per row, first-seen order.
    seen: dict[tuple, int] = {}
    inverse = np.empty(n_mp, dtype=np.int64)
    key_list: list[tuple] = []
    for i, key in enumerate(zip(*axes)):
        code = seen.get(key)
        if code is None:
            code = len(key_list)
            seen[key] = code
            key_list.append(key)
        inverse[i] = code
    segments: list[tuple[tuple, np.ndarray]] = []
    for code, key in enumerate(key_list):
        if key not in basis_norm:
            raise ValueError(
                f"segment {key!r} appears in model_points but is not in the "
                f"basis (known segments: {sorted(basis_norm)})"
            )
        idx = np.nonzero(inverse == code)[0]
        segments.append((key, idx))
    return basis_norm, segments


def _stitch_full_measurements(n_mp, sub_results):
    """Scatter per-segment full GMMMeasurements into one (n_mp, n_time+1) result.

    ``sub_results`` is ``[(idx, GMMMeasurement)]`` -- each segment's full
    trajectories are laid into the portfolio arrays at its rows and zero-padded
    on the right to the portfolio's longest horizon (a contract carries no BEL /
    RA / CSM past its term). ``discount_factor_bom`` / ``discount_factor_mid`` become per-MP
    2-D because segments discount on different curves; the padded tail repeats
    each row's last factor so a forward rate read off it is finite, not a 0/0.
    Shared by the new-business segmented measurement and the in-force segmented
    settlement (their per-segment results carry the identical field set).
    """
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
    discount_factor_bom = np.ones((n_mp, n_time + 1))
    discount_factor_mid = np.ones((n_mp, n_time))

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
        _carry_lic_residual(lic, idx, t, n_time, m.lic)
        csm_accretion[idx, :t] = m.csm_accretion
        csm_release[idx, :t] = m.csm_release
        # Per-MP discount: lay the segment's curve, then flat-fill the tail so
        # the padded months read a zero forward rate, not a 0/0.
        discount_factor_bom[idx, :t + 1] = m.discount_factor_bom
        discount_factor_bom[idx, t + 1:] = m.discount_factor_bom[-1]
        discount_factor_mid[idx, :t] = m.discount_factor_mid
        if t < n_time:
            discount_factor_mid[idx, t:] = m.discount_factor_mid[-1] if t > 0 else 1.0
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
        cashflows=cashflows, discount_factor_bom=discount_factor_bom, discount_factor_mid=discount_factor_mid,
    )


def _measure_segmented_full(
    model_points: ModelPoints, basis: dict[tuple[str, str], Basis],
    *, segment_by=("product", "channel"),
) -> GMMMeasurement:
    """Full multi-segment GMM measurement -- per-segment trajectories stitched.

    Each (product, channel) segment is measured under its own
    ``Basis`` via :func:`_measure_full`; the per-segment ``(n_seg, *)``
    trajectories are scattered back into one ``(n_mp, n_time+1)`` result, where
    ``n_time`` is the portfolio's longest horizon. A segment whose contracts
    mature earlier is zero-padded on the right -- a contract carries no BEL /
    RA / CSM past its term. ``discount_factor_bom`` / ``discount_factor_mid`` are per-MP
    ``(n_mp, ...)`` here, not the single ``(n_time+1,)`` curve of the
    single-basis path: segments discount on different curves, so the rate is a
    property of the row. The padded tail of ``discount_factor_bom`` repeats each
    row's last factor (a flat curve -> zero forward rate) so a rate read off it
    is finite, not a 0/0.
    """
    _require_gmm_router(basis, entry="fcf.gmm.measure")
    try:
        basis_norm, segments = _factorise_segments(
            basis, model_points, segment_by, model_points.n_mp,
        )
    except KeyError:
        if len(basis.segments) == 1:
            (basis,) = basis.segments.values()
            return _measure_full(model_points, basis)
        raise ValueError(
            f"model_points has no {tuple(segment_by)} axis/axes set but the "
            f"basis has {len(basis.segments)} segments; either set the columns or "
            "pass a single-segment basis"
        )
    # Each segment's _measure_full handles an account book correctly (fund
    # netting + NAR RA), but _stitch_full_measurements reassembles only the flat
    # trajectories -- it would drop the nested AccountTrajectory sidecar, so a
    # stitched account book would silently lose its account diagnostics. Reject
    # it until the stitch forwards the sidecar (a follow-up). The single-segment
    # convenience path above returns _measure_full directly and keeps the
    # sidecar, so this only blocks the genuine multi-segment stitch.
    if any(_portfolio_has_account(model_points, b) for b in basis_norm.values()):
        raise NotImplementedError(
            "segmented full measurement of an account-backed (universal-life) "
            "book is not yet supported -- the per-segment account trajectory is "
            "not stitched back. Measure the account segment on its own Basis.")

    n_mp = model_points.n_mp

    sub_results = [(idx, _measure_full(model_points.subset(idx), basis_norm[key]))
                   for key, idx in segments]
    return _stitch_full_measurements(n_mp, sub_results)
