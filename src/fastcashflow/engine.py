"""Engine entry points.

Two paths:

* ``run``   -- detailed: full monthly cash flow and CSM trajectories. Use it
  for inspection, validation and movement analysis.
* ``value`` -- fast: a single fused, parallel kernel producing only the
  headline valuation (BEL, RA, CSM, loss component) per model point. It
  materialises no per-month arrays, so it is memory-minimal and the fastest
  path for large-scale valuation.

Both paths share the same arithmetic, so ``value`` reproduces ``run``'s
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
    compute_bel,
    compute_csm,
    compute_ra,
    discount_factors,
)
from fastcashflow.modelpoint import ModelPointSet
from fastcashflow.projection import CashflowProjection, project_cashflows


# ---------------------------------------------------------------------------
# Detailed path
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GMMResult:
    """Detailed GMM result.

    Per-model-point arrays have shape ``(n_mp,)`` unless stated otherwise.
    """

    bel: FloatArray              # Best Estimate of Liability
    ra: FloatArray               # Risk Adjustment
    csm0: FloatArray             # CSM at initial recognition
    loss_component: FloatArray   # loss component at inception (onerous contracts)
    csm: FloatArray              # (n_mp, n_time+1) -- CSM trajectory
    csm_release: FloatArray      # (n_mp, n_time)   -- CSM released each month
    projection: CashflowProjection
    discount: FloatArray         # (n_time,) -- monthly discount factors


def run(mps: ModelPointSet, asmp: Assumptions) -> GMMResult:
    """Detailed GMM projection: full cash flow and CSM trajectories."""
    proj = project_cashflows(mps, asmp)
    discount = discount_factors(asmp, proj.n_time)

    bel = compute_bel(proj, discount)
    ra = compute_ra(proj, discount, asmp.ra_confidence, asmp.claims_cv)
    csm = compute_csm(bel, ra, proj, asmp)

    return GMMResult(
        bel=bel,
        ra=ra,
        csm0=csm.csm[:, 0],
        loss_component=csm.loss_component,
        csm=csm.csm,
        csm_release=csm.release,
        projection=proj,
        discount=discount,
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
def _value_kernel(rates_grid, issue_index, term_months, lapse, monthly_premium,
                  sum_assured, expense_acquisition, maint_monthly,
                  inflation, discount):
    """Fused valuation kernel -- one parallel pass, no per-month arrays.

    Per model point the in-force amount is carried as a scalar through the
    time loop while the three present values are accumulated directly. The
    only memory written is the three ``(n_mp,)`` result arrays, so the
    kernel is compute-bound and scales near-linearly across cores.
    """
    n_mp = issue_index.shape[0]
    pv_claim = np.zeros(n_mp)
    pv_premium = np.zeros(n_mp)
    pv_expense = np.zeros(n_mp)

    for mp in prange(n_mp):
        term = term_months[mp]
        ridx = issue_index[mp]
        premium = monthly_premium[mp]
        sa = sum_assured[mp]
        inforce = 1.0
        pc = 0.0
        pp = 0.0
        pe = 0.0
        for t in range(term):
            q = rates_grid[ridx, t // 12]
            d = discount[t]
            pp += inforce * premium * d
            pc += inforce * q * sa * d
            acquisition = expense_acquisition if t == 0 else 0.0
            pe += (acquisition + inforce * maint_monthly * inflation[t]) * d
            inforce *= (1.0 - q) * (1.0 - lapse)
        pv_claim[mp] = pc
        pv_premium[mp] = pp
        pv_expense[mp] = pe

    return pv_claim, pv_premium, pv_expense


def value(mps: ModelPointSet, asmp: Assumptions, *, backend: str = "cpu") -> Valuation:
    """Fast GMM valuation: BEL, RA and CSM per model point.

    One fused kernel; no per-month arrays are materialised. This is the
    memory-minimal, fastest path for large-scale valuation. For full cash
    flow / CSM trajectories use :func:`run`.

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

    # Mortality depends only on the (small) set of distinct issue ages, so it
    # is evaluated on that grid -- the transcendental cost becomes negligible.
    unique_issue, issue_index = np.unique(mps.issue_age, return_inverse=True)
    age_grid = unique_issue[:, None] + np.arange(n_years)[None, :]
    rates_grid = np.ascontiguousarray(
        asmp.mortality_monthly(age_grid), dtype=np.float64
    )

    inflation = (1.0 + asmp.expense_inflation) ** (months / 12.0)
    discount = discount_factors(asmp, n_time)

    args = (
        rates_grid,
        issue_index.astype(np.int64),
        mps.term_months,
        asmp.lapse_monthly,
        mps.monthly_premium,
        mps.sum_assured,
        asmp.expense_acquisition,
        asmp.expense_maintenance_annual / 12.0,
        inflation,
        discount,
    )

    if backend == "cpu":
        pv_claim, pv_premium, pv_expense = _value_kernel(*args)
    elif backend == "gpu":
        from fastcashflow._gpu import value_pv_gpu
        pv_claim, pv_premium, pv_expense = value_pv_gpu(*args)
    else:
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")

    bel = pv_claim + pv_expense - pv_premium
    ra = _norm_ppf(asmp.ra_confidence) * asmp.claims_cv * pv_claim
    fcf = bel + ra
    return Valuation(
        bel=bel,
        ra=ra,
        csm=np.maximum(0.0, -fcf),
        loss_component=np.maximum(0.0, fcf),
    )
