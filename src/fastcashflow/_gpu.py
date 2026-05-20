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
def _value_cuda_kernel(rates_grid, issue_index, term_months, lapse,
                       monthly_premium, sum_assured, expense_acquisition,
                       maint_monthly, inflation, discount,
                       pv_claim, pv_premium, pv_expense):
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


def value_pv_gpu(rates_grid, issue_index, term_months, lapse, monthly_premium,
                 sum_assured, expense_acquisition, maint_monthly,
                 inflation, discount):
    """Run the fused valuation kernel on the GPU; return the three PV arrays."""
    if not cuda.is_available():
        raise RuntimeError(
            "backend='gpu' requires a CUDA device; none is available"
        )

    n_mp = issue_index.shape[0]
    d_rates = cuda.to_device(rates_grid)
    d_issue = cuda.to_device(issue_index)
    d_term = cuda.to_device(term_months)
    d_premium = cuda.to_device(monthly_premium)
    d_sum_assured = cuda.to_device(sum_assured)
    d_inflation = cuda.to_device(inflation)
    d_discount = cuda.to_device(discount)
    d_pv_claim = cuda.device_array(n_mp, dtype=np.float64)
    d_pv_premium = cuda.device_array(n_mp, dtype=np.float64)
    d_pv_expense = cuda.device_array(n_mp, dtype=np.float64)

    threads = 256
    blocks = (n_mp + threads - 1) // threads
    _value_cuda_kernel[blocks, threads](
        d_rates, d_issue, d_term, lapse, d_premium, d_sum_assured,
        expense_acquisition, maint_monthly, d_inflation, d_discount,
        d_pv_claim, d_pv_premium, d_pv_expense,
    )
    cuda.synchronize()

    return (
        d_pv_claim.copy_to_host(),
        d_pv_premium.copy_to_host(),
        d_pv_expense.copy_to_host(),
    )
