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
    compute_bel,
    compute_csm,
    compute_ra,
    discount_factors,
)
from fastcashflow.modelpoint import ModelPointSet
from fastcashflow.projection import Cashflows, project_cashflows


# ---------------------------------------------------------------------------
# Detailed path
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Measurement:
    """Detailed measurement -- full cash flow and CSM trajectories.

    Per-model-point arrays have shape ``(n_mp,)`` unless stated otherwise.
    """

    bel: FloatArray              # Best Estimate of Liability
    ra: FloatArray               # Risk Adjustment
    csm0: FloatArray             # CSM at initial recognition
    loss_component: FloatArray   # loss component at inception (onerous contracts)
    csm: FloatArray              # (n_mp, n_time+1) -- CSM trajectory
    csm_release: FloatArray      # (n_mp, n_time)   -- CSM released each month
    cashflows: Cashflows
    discount: FloatArray         # (n_time,) -- monthly discount factors


def measure(mps: ModelPointSet, asmp: Assumptions) -> Measurement:
    """Detailed GMM measurement: full cash flow and CSM trajectories."""
    proj = project_cashflows(mps, asmp)
    discount = discount_factors(asmp, proj.n_time)

    bel = compute_bel(proj, discount)
    ra = compute_ra(proj, discount, asmp.ra_confidence, asmp.claims_cv)
    csm, csm_release, loss_component = compute_csm(bel, ra, proj, asmp)

    return Measurement(
        bel=bel,
        ra=ra,
        csm0=csm[:, 0],
        loss_component=loss_component,
        csm=csm,
        csm_release=csm_release,
        cashflows=proj,
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
def _value_kernel(rates_grid, issue_index, term_months, lapse_by_year,
                  monthly_premium, sum_assured, expense_acquisition,
                  maint_monthly, inflation, discount, ra_factor):
    """Fused valuation kernel -- one parallel pass, no per-month arrays.

    Per model point the in-force amount is carried as a scalar through the
    time loop while the present values are accumulated directly; BEL, RA,
    CSM and the loss component are then derived in the same pass. The only
    memory written is the four ``(n_mp,)`` result arrays, so the kernel is
    compute-bound and scales near-linearly across cores.
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
        sa = sum_assured[mp]
        inforce = 1.0
        pc = 0.0
        pp = 0.0
        pe = 0.0
        for t in range(term):
            year = t // 12
            q = rates_grid[ridx, year]
            d = discount[t]
            pp += inforce * premium * d
            pc += inforce * q * sa * d
            acquisition = expense_acquisition if t == 0 else 0.0
            pe += (acquisition + inforce * maint_monthly * inflation[t]) * d
            inforce *= (1.0 - q) * (1.0 - lapse_by_year[year])
        bel_mp = pc + pe - pp
        ra_mp = ra_factor * pc
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
    rates_grid = np.ascontiguousarray(
        asmp.mortality_monthly(issue_age_grid, duration_grid), dtype=np.float64
    )
    issue_index = (mps.issue_age - min_age).astype(np.int64)
    lapse_by_year = np.ascontiguousarray(
        asmp.lapse_monthly(durations), dtype=np.float64
    )

    inflation = (1.0 + asmp.expense_inflation) ** (months / 12.0)
    discount = discount_factors(asmp, n_time)
    ra_factor = _norm_ppf(asmp.ra_confidence) * asmp.claims_cv

    args = (
        rates_grid,
        issue_index,
        mps.term_months,
        lapse_by_year,
        mps.monthly_premium,
        mps.sum_assured,
        asmp.expense_acquisition,
        asmp.expense_maintenance_annual / 12.0,
        inflation,
        discount,
        ra_factor,
    )

    if backend == "cpu":
        bel, ra, csm, loss_component = _value_kernel(*args)
    elif backend == "gpu":
        from fastcashflow._gpu import value_gpu
        bel, ra, csm, loss_component = value_gpu(*args)
    else:
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")

    return Valuation(bel=bel, ra=ra, csm=csm, loss_component=loss_component)
