"""Engine entry points.

Two paths:

* ``measure`` -- detailed: full monthly cash flow and CSM trajectories. Use it
  for inspection, validation and movement analysis.
* ``value``   -- fast: a single fused, parallel kernel producing only the
  headline valuation (BEL, RA, CSM, loss component) per model point. It
  materialises no per-month arrays, so it is memory-minimal and the fastest
  path for large-scale valuation.

Both paths share the same arithmetic, so ``value`` reproduces ``measure``'s
headline numbers exactly (cross-checked in the tests).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.gmm import (
    _norm_ppf,
    _rollforward_kernel,
    compute_csm,
    discount_factors,
)
from fastcashflow.coverage import coverage_rates
from fastcashflow.modelpoint import ModelPointSet
from fastcashflow.projection import Cashflows, project_cashflows


# ---------------------------------------------------------------------------
# Detailed path
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Measurement:
    """Detailed measurement -- BEL, RA and CSM rolled forward over time.

    ``bel``, ``ra`` and ``csm`` are ``(n_mp, n_time+1)`` trajectories; column
    0 is the inception measurement. The CSM roll-forward decomposes as
    ``csm[:, t+1] = csm[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    """

    bel: FloatArray              # (n_mp, n_time+1) -- BEL trajectory
    ra: FloatArray               # (n_mp, n_time+1) -- RA trajectory
    csm: FloatArray              # (n_mp, n_time+1) -- CSM trajectory
    csm_accretion: FloatArray    # (n_mp, n_time)   -- CSM interest accreted each month
    csm_release: FloatArray      # (n_mp, n_time)   -- CSM released each month
    loss_component: FloatArray   # (n_mp,)          -- loss component at inception
    cashflows: Cashflows
    discount_start: FloatArray   # (n_time,) -- start-of-month discount factors
    discount_mid: FloatArray     # (n_time,) -- mid-month discount factors


def measure(mps: ModelPointSet, asmp: Assumptions) -> Measurement:
    """Detailed GMM measurement: BEL, RA and CSM rolled forward over time."""
    proj = project_cashflows(mps, asmp)
    discount_start, discount_mid = discount_factors(asmp, proj.n_time)

    bel, pv_claims, pv_survival = _rollforward_kernel(
        proj.claim_cf, proj.expense_cf, proj.premium_cf,
        proj.annuity_cf, proj.maturity_cf, mps.term_months,
        asmp.discount_monthly,
    )
    z = _norm_ppf(asmp.ra_confidence)
    ra = z * (asmp.claims_cv * pv_claims + asmp.longevity_cv * pv_survival)
    csm, csm_accretion, csm_release, loss_component = compute_csm(
        bel[:, 0], ra[:, 0], proj, asmp
    )

    return Measurement(
        bel=bel,
        ra=ra,
        csm=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        loss_component=loss_component,
        cashflows=proj,
        discount_start=discount_start,
        discount_mid=discount_mid,
    )


# ---------------------------------------------------------------------------
# Fast path -- fused, memory-minimal valuation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Valuation:
    """Headline IFRS 17 GMM valuation. Each array has shape ``(n_mp,)``."""

    bel: FloatArray             # Best Estimate of Liability
    ra: FloatArray              # Risk Adjustment
    csm: FloatArray             # CSM at initial recognition
    loss_component: FloatArray  # loss component at inception (onerous contracts)


@njit(parallel=True, cache=True)
def _value_kernel(mortality_grid, issue_index, term_months, lapse_by_year,
                  monthly_premium, single_premium, cov_kind, cov_amount,
                  cov_offset, cov_rates, maturity_benefit, annuity_payment,
                  expense_acquisition, maint_monthly, inflation,
                  discount_start, discount_mid, mortality_factor,
                  longevity_factor):
    """Fused valuation kernel -- one parallel pass, no per-month arrays.

    Per model point the in-force amount is carried as a scalar through the
    time loop while the present values are accumulated directly -- death
    claims (summed over the coverage list), premiums (level, plus the single
    premium at t=0), expenses, annuity payments and the maturity benefit.
    BEL, RA, CSM and the loss component are derived in the same pass. The RA
    adds a mortality-risk component (death claims) and a longevity-risk
    component (annuity and maturity benefits). The only memory written is the
    four ``(n_mp,)`` result arrays, so the kernel is compute-bound and scales
    near-linearly.
    """
    n_mp = issue_index.shape[0]
    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    loss_component = np.empty(n_mp)

    for mp in prange(n_mp):
        term = term_months[mp]
        ridx = issue_index[mp]
        premium = monthly_premium[mp]
        annuity = annuity_payment[mp]
        c_start = cov_offset[mp]
        c_end = cov_offset[mp + 1]
        inforce = 1.0
        pc = 0.0
        pp = 0.0
        pe = 0.0
        pa = 0.0
        last_year = -1
        claim_rate = 0.0      # aggregate claim per unit in-force, current year
        for t in range(term):
            year = t // 12
            # Coverage rates change only once a year, so the per-coverage sum
            # is rebuilt on a year boundary, not every month.
            if year != last_year:
                claim_rate = 0.0
                for k in range(c_start, c_end):
                    claim_rate += cov_rates[cov_kind[k], ridx, year] * cov_amount[k]
                last_year = year
            q = mortality_grid[ridx, year]
            ds = discount_start[t]
            dm = discount_mid[t]
            single = single_premium[mp] if t == 0 else 0.0
            pp += (inforce * premium + single) * ds
            pc += inforce * claim_rate * dm
            pa += inforce * annuity * ds
            acquisition = expense_acquisition if t == 0 else 0.0
            pe += (acquisition + inforce * maint_monthly * inflation[t]) * dm
            inforce *= (1.0 - q) * (1.0 - lapse_by_year[year])
        pm = inforce * maturity_benefit[mp] * discount_start[term]
        bel_mp = pc + pm + pa + pe - pp
        ra_mp = mortality_factor * pc + longevity_factor * (pm + pa)
        fcf = bel_mp + ra_mp
        bel[mp] = bel_mp
        ra[mp] = ra_mp
        csm[mp] = max(0.0, -fcf)
        loss_component[mp] = max(0.0, fcf)

    return bel, ra, csm, loss_component


def value(mps: ModelPointSet, asmp: Assumptions, *, backend: str = "cpu") -> Valuation:
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
    """
    n_time = int(mps.term_months.max())
    n_years = (n_time + 11) // 12
    months = np.arange(n_time)

    # Mortality and lapse are evaluated on a dense [min, max] issue-age x
    # duration grid. Using the age range rather than the exact distinct ages
    # avoids an O(n log n) sort (np.unique): min/max and the index
    # subtraction are O(n), and the few unused ages cost nothing -- the
    # assumption grid is tiny.
    min_age = int(mps.issue_age.min())
    max_age = int(mps.issue_age.max())
    durations = np.arange(n_years)
    issue_age_grid, duration_grid = np.meshgrid(
        np.arange(min_age, max_age + 1), durations, indexing="ij"
    )
    mortality_grid = np.ascontiguousarray(
        asmp.mortality_monthly(issue_age_grid, duration_grid), dtype=np.float64
    )
    issue_index = (mps.issue_age - min_age).astype(np.int64)
    lapse_by_year = np.ascontiguousarray(
        asmp.lapse_monthly(durations), dtype=np.float64
    )
    cov_rates = coverage_rates(mortality_grid)

    inflation = (1.0 + asmp.expense_inflation) ** (months / 12.0)
    discount_start, discount_mid = discount_factors(asmp, n_time)
    z = _norm_ppf(asmp.ra_confidence)
    mortality_factor = z * asmp.claims_cv
    longevity_factor = z * asmp.longevity_cv

    args = (
        mortality_grid,
        issue_index,
        mps.term_months,
        lapse_by_year,
        mps.monthly_premium,
        mps.single_premium,
        mps.cov_kind,
        mps.cov_amount,
        mps.cov_offset,
        cov_rates,
        mps.maturity_benefit,
        mps.annuity_payment,
        asmp.expense_acquisition,
        asmp.expense_maintenance_annual / 12.0,
        inflation,
        discount_start,
        discount_mid,
        mortality_factor,
        longevity_factor,
    )

    if backend == "cpu":
        bel, ra, csm, loss_component = _value_kernel(*args)
    elif backend == "gpu":
        from fastcashflow._gpu import value_gpu
        bel, ra, csm, loss_component = value_gpu(*args)
    else:
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")

    return Valuation(bel=bel, ra=ra, csm=csm, loss_component=loss_component)
