"""Stochastic valuation -- the liability distribution over economic scenarios.

A deterministic run gives one liability from one assumption set. A stochastic
valuation runs the projection under many economic scenarios and reports the
*distribution* of the liability -- which feeds the percentile-based risk and
capital measures a single deterministic run cannot give.

``measure_stochastic`` takes the scenarios as input -- fastcashflow is the
engine, not an economic scenario generator -- and values each one. Running N
scenarios over millions of seriatim policies is precisely what the engine's
speed exists for: a slow engine cannot do seriatim stochastic at scale at all.

The scenario axis lives *inside* the kernel. Only the discount changes between
scenarios -- the per-month cash flows do not -- so the projection runs once
(``project_cashflows``) and a single parallel kernel sweeps every scenario,
re-discounting the shared cash flows with ``prange`` over the scenario axis.
This collapses what was one kernel dispatch per scenario into one dispatch
total, and parallelises the sweep across cores.

Each scenario is either a flat annual discount rate or a full discount-rate
curve -- one annual rate per projection month. Investment-return scenarios
for participating business are handled separately, by ``measure_tvog``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.engine import measure
from fastcashflow.model_points import ModelPoints
from fastcashflow.numerics import _norm_ppf
from fastcashflow.projection import project_cashflows


@dataclass(frozen=True, slots=True, eq=False)
class StochasticResult:
    """Per-scenario portfolio totals from a stochastic valuation.

    Each array is ``(n_scenarios,)`` -- the portfolio total of that figure
    under each scenario. Read the distribution off with :meth:`mean` and
    :meth:`percentile`, or from the arrays directly.
    """

    bel: FloatArray
    ra: FloatArray
    csm: FloatArray
    loss_component: FloatArray

    def mean(self) -> dict[str, float]:
        """The mean of each line across the scenarios."""
        return {name: float(getattr(self, name).mean())
                for name in ("bel", "ra", "csm", "loss_component")}

    def percentile(self, q: float) -> dict[str, float]:
        """The ``q``-th percentile of each line across the scenarios."""
        return {name: float(np.percentile(getattr(self, name), q))
                for name in ("bel", "ra", "csm", "loss_component")}


@njit(parallel=True, cache=True)
def _stochastic_inception_kernel(
    mortality_cf, morbidity_cf, disability_cf, expense_cf,
    premium_cf, annuity_cf, maturity_cf, surrender_cf,
    contract_boundary_months, monthly_rate_all,
    mort_factor, morb_factor, disab_factor, long_factor,
):
    """Per-scenario portfolio BEL / RA / CSM / loss from shared cash flows.

    ``monthly_rate_all`` is ``(n_scenarios, n_time)`` -- one per-month discount
    curve per scenario. The cash flows are scenario-independent; the outer
    ``prange`` runs each scenario on its own core, re-discounting the shared
    flows with that scenario's curve. The backward recursion reproduces
    :func:`fastcashflow.numerics._roll_forward_kernel` at the inception column
    (``t = 0``): premiums and annuities fall at month start, claims / expenses
    / surrender mid-month, the maturity benefit seeds ``t = boundary``. The
    recursion stops at the IFRS 17 contract boundary -- ``project_cashflows``
    truncates every cash-flow array to ``contract_boundary_months.max()``, so
    each MP must loop only over its own ``[0, boundary)`` (looping to ``term``
    reads past the array width). The RA is the confidence-level margin; CSM
    and loss component are the per-MP ``max(0, -FCF)`` / ``max(0, FCF)`` summed
    to the portfolio.
    """
    n_scen, n_time = monthly_rate_all.shape
    n_mp = mortality_cf.shape[0]
    bel_out = np.empty(n_scen)
    ra_out = np.empty(n_scen)
    csm_out = np.empty(n_scen)
    loss_out = np.empty(n_scen)

    for s in prange(n_scen):
        bel_s = 0.0
        ra_s = 0.0
        csm_s = 0.0
        loss_s = 0.0
        for mp in range(n_mp):
            boundary = contract_boundary_months[mp]
            bel_v = maturity_cf[mp]   # bel[boundary]
            pvc = 0.0                 # pv_claims (mortality risk)
            pvm = 0.0                 # pv_morbidity
            pvd = 0.0                 # pv_disability
            pvs = maturity_cf[mp]     # pv_survival (longevity risk), seeded at boundary
            for t in range(boundary - 1, -1, -1):
                mr = monthly_rate_all[s, t]
                half = (1.0 + mr) ** (-0.5)
                full = 1.0 / (1.0 + mr)
                claim = mortality_cf[mp, t]
                morb = morbidity_cf[mp, t]
                disab = disability_cf[mp, t]
                ann = annuity_cf[mp, t]
                surr = surrender_cf[mp, t]
                bel_v = (ann - premium_cf[mp, t]
                         + (claim + morb + disab + expense_cf[mp, t] + surr) * half
                         + bel_v * full)
                pvc = claim * half + pvc * full
                pvm = morb * half + pvm * full
                pvd = disab * half + pvd * full
                pvs = ann + pvs * full
            ra_mp = (mort_factor * pvc + morb_factor * pvm
                     + disab_factor * pvd + long_factor * pvs)
            fcf = bel_v + ra_mp
            bel_s += bel_v
            ra_s += ra_mp
            if fcf < 0.0:
                csm_s += -fcf
            else:
                loss_s += fcf
        bel_out[s] = bel_s
        ra_out[s] = ra_s
        csm_out[s] = csm_s
        loss_out[s] = loss_s

    return bel_out, ra_out, csm_out, loss_out


def measure_stochastic(
    model_points: ModelPoints, basis: Basis, rate_scenarios: FloatArray
) -> StochasticResult:
    """Value a portfolio under each economic scenario -- the liability distribution.

    ``rate_scenarios`` is either

    * a 1-D ``(n_scenarios,)`` array -- one flat annual discount rate per
      scenario; or
    * a 2-D ``(n_scenarios, n_time)`` array -- one discount-rate curve per
      scenario, an annual rate for each projection month.

    The portfolio total of every figure is recorded under each scenario, so
    the distribution -- mean, percentiles -- can be read from the result. The
    projection runs once and a single parallel kernel sweeps the scenario axis
    (see module docstring); the settlement-pattern and cost-of-capital paths
    fall back to a per-scenario ``measure`` loop (``full=False`` for the
    confidence-level RA, ``full=True`` for cost-of-capital, which the fast
    path does not compute). Cost-of-capital supports flat (1-D) rate_scenarios only.
    """
    rate_scenarios = np.asarray(rate_scenarios, dtype=np.float64)
    if rate_scenarios.ndim not in (1, 2):
        raise ValueError("rate_scenarios must be 1-D (flat rates) or 2-D (rate curves)")
    if rate_scenarios.size == 0:
        raise ValueError("rate_scenarios is empty; need at least one scenario")
    if not np.all(np.isfinite(rate_scenarios)):
        raise ValueError(
            "rate_scenarios must be finite (a NaN / inf rate yields a "
            "silently-NaN distribution)"
        )

    # The scenario-axis kernel re-discounts shared cash flows -- it covers the
    # confidence-level RA with no claims settlement pattern (the rate at the
    # month of incurrence in the settlement factor would otherwise have to
    # vary per scenario). Other configurations fall back to the per-scenario
    # measure(full=False) loop, which handles them correctly if more slowly.
    from fastcashflow._measurement.account import _portfolio_has_account
    if (basis.ra_method == "confidence_level"
            and basis.settlement_pattern is None
            and not _portfolio_has_account(model_points, basis)):
        # The fast inception kernel reads the benefit cash flows raw (no account
        # fund netting), so a universal-life account book skips it and falls to
        # the per-scenario fallback below, which re-measures through measure()
        # -- the account book is routed to the full measurement and netted there.
        proj = project_cashflows(model_points, basis)
        n_time = proj.mortality_cf.shape[1]
        if rate_scenarios.ndim == 2:
            if rate_scenarios.shape[1] != n_time:
                raise ValueError(
                    f"a 2-D rate_scenarios array must have {n_time} columns (the "
                    f"projection horizon), got {rate_scenarios.shape[1]}"
                )
            monthly_rate_all = (1.0 + rate_scenarios) ** (1.0 / 12.0) - 1.0
        else:
            flat = (1.0 + rate_scenarios) ** (1.0 / 12.0) - 1.0
            monthly_rate_all = np.repeat(flat[:, None], n_time, axis=1)
        monthly_rate_all = np.ascontiguousarray(monthly_rate_all)
        z = _norm_ppf(basis.ra_confidence)
        bel, ra, csm, loss_component = _stochastic_inception_kernel(
            proj.mortality_cf, proj.morbidity_cf, proj.disability_cf, proj.expense_cf,
            proj.premium_cf, proj.annuity_cf, proj.maturity_cf, proj.surrender_cf,
            np.asarray(model_points.contract_boundary_months, dtype=np.int64),
            monthly_rate_all,
            z * basis.mortality_cv, z * basis.morbidity_cv,
            z * basis.disability_cv, z * basis.longevity_cv,
        )
        return StochasticResult(bel=bel, ra=ra, csm=csm, loss_component=loss_component)

    # Fallback: settlement pattern or non-confidence-level RA. Value each
    # scenario one at a time. The confidence-level RA is available on the fast
    # path (full=False); cost-of-capital RA is not, so those rate_scenarios run on
    # the trajectory path (full=True) and read the inception headline.
    use_full = basis.ra_method != "confidence_level"
    if rate_scenarios.ndim == 2:
        # The projection horizon is the contract boundary, not the term --
        # the same width the discount curve / fast kernel use.
        n_time = int(np.asarray(model_points.contract_boundary_months).max())
        if rate_scenarios.shape[1] != n_time:
            raise ValueError(
                f"a 2-D rate_scenarios array must have {n_time} columns (the "
                f"projection horizon), got {rate_scenarios.shape[1]}"
            )
        if use_full:
            # full=True reads the discount off basis.discount_annual, which is
            # a per-year curve; a per-month scenario curve has no place to go.
            raise NotImplementedError(
                "measure_stochastic with ra_method='cost_of_capital' supports "
                "flat (1-D) discount-rate rate_scenarios only; a per-month discount "
                "curve (2-D) is supported only under the confidence-level RA. "
                "Use 1-D flat rates, or ra_method='confidence_level' for curves."
            )
    n = int(rate_scenarios.shape[0])
    bel = np.empty(n)
    ra = np.empty(n)
    csm = np.empty(n)
    loss_component = np.empty(n)
    for s in range(n):
        if rate_scenarios.ndim == 1:
            v = measure(model_points,
                        replace(basis, discount_annual=float(rate_scenarios[s])),
                        full=use_full)
        else:
            v = measure(model_points, basis, full=False, discount_curve=rate_scenarios[s])
        bel[s] = v.bel.sum()
        ra[s] = v.ra.sum()
        csm[s] = v.csm.sum()
        loss_component[s] = v.loss_component.sum()
    return StochasticResult(bel=bel, ra=ra, csm=csm, loss_component=loss_component)
