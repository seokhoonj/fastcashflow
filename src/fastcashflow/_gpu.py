"""CUDA backend for the fused valuation kernel.

Imported lazily by ``engine.value(backend="gpu")``; never loaded for CPU-only
use, so a machine without CUDA can still import and use the rest of the
package.

The kernel mirrors the CPU kernel in ``engine._value_kernel`` exactly; the
cross-check test ``test_value_gpu_matches_cpu`` guards against divergence.
"""
from __future__ import annotations

import numpy as np
from numba import cuda


@cuda.jit
def _value_cuda_kernel(rates_grid, issue_index, term_months, lapse_by_year,
                       monthly_premium, sum_assured, expense_acquisition,
                       maint_monthly, inflation, discount_start, discount_mid,
                       ra_factor, bel, ra, csm, loss_component):
    """One CUDA thread per model point; the per-month loop runs in the thread."""
    mp = cuda.grid(1)
    if mp >= issue_index.shape[0]:
        return

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
        ds = discount_start[t]
        dm = discount_mid[t]
        pp += inforce * premium * ds
        pc += inforce * q * sa * dm
        acquisition = expense_acquisition if t == 0 else 0.0
        pe += (acquisition + inforce * maint_monthly * inflation[t]) * dm
        inforce *= (1.0 - q) * (1.0 - lapse_by_year[year])
    bel_mp = pc + pe - pp
    ra_mp = ra_factor * pc
    fcf = bel_mp + ra_mp
    bel[mp] = bel_mp
    ra[mp] = ra_mp
    csm[mp] = max(0.0, -fcf)
    loss_component[mp] = max(0.0, fcf)


def value_gpu(rates_grid, issue_index, term_months, lapse_by_year,
              monthly_premium, sum_assured, expense_acquisition, maint_monthly,
              inflation, discount_start, discount_mid, ra_factor):
    """Run the fused valuation kernel on the GPU.

    Returns the four ``(n_mp,)`` valuation arrays: BEL, RA, CSM and the
    loss component.
    """
    if not cuda.is_available():
        raise RuntimeError(
            "backend='gpu' requires a CUDA device; none is available"
        )

    n_mp = issue_index.shape[0]
    d_rates = cuda.to_device(rates_grid)
    d_issue = cuda.to_device(issue_index)
    d_term = cuda.to_device(term_months)
    d_lapse = cuda.to_device(lapse_by_year)
    d_premium = cuda.to_device(monthly_premium)
    d_sum_assured = cuda.to_device(sum_assured)
    d_inflation = cuda.to_device(inflation)
    d_discount_start = cuda.to_device(discount_start)
    d_discount_mid = cuda.to_device(discount_mid)
    d_bel = cuda.device_array(n_mp, dtype=np.float64)
    d_ra = cuda.device_array(n_mp, dtype=np.float64)
    d_csm = cuda.device_array(n_mp, dtype=np.float64)
    d_loss = cuda.device_array(n_mp, dtype=np.float64)

    threads = 256
    blocks = (n_mp + threads - 1) // threads
    _value_cuda_kernel[blocks, threads](
        d_rates, d_issue, d_term, d_lapse, d_premium, d_sum_assured,
        expense_acquisition, maint_monthly, d_inflation,
        d_discount_start, d_discount_mid, ra_factor,
        d_bel, d_ra, d_csm, d_loss,
    )
    cuda.synchronize()

    return (
        d_bel.copy_to_host(),
        d_ra.copy_to_host(),
        d_csm.copy_to_host(),
        d_loss.copy_to_host(),
    )
