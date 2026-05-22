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
def _value_cuda_kernel(mortality_grid, issue_index, sex, term_months, count,
                       lapse_by_year, monthly_premium, single_premium,
                       premium_term_months, cov_kind, cov_amount, cov_offset,
                       cov_rates, cov_risk,
                       cov_is_diagnosis, maturity_benefit, annuity_payment,
                       expense_acquisition, maint_monthly, inflation,
                       discount_start, discount_mid,
                       mortality_factor, morbidity_factor, longevity_factor,
                       bel, ra, csm, loss_component):
    """One CUDA thread per model point; the per-month loop runs in the thread."""
    mp = cuda.grid(1)
    if mp >= issue_index.shape[0]:
        return

    term = term_months[mp]
    premium_term = premium_term_months[mp]   # months the premium is paid
    ridx = issue_index[mp]
    sx = sex[mp]
    cnt = count[mp]
    premium = monthly_premium[mp]
    annuity = annuity_payment[mp]
    c_start = cov_offset[mp]
    c_end = cov_offset[mp + 1]
    inforce = cnt
    pc = 0.0
    pcm = 0.0
    pp = 0.0
    pe = 0.0
    pa = 0.0
    last_year = -1
    claim_rate = 0.0
    morb_rate = 0.0
    for t in range(term):
        year = t // 12
        if year != last_year:
            claim_rate = 0.0
            morb_rate = 0.0
            for k in range(c_start, c_end):
                kind = cov_kind[k]
                if cov_is_diagnosis[kind]:
                    continue
                rate = cov_rates[kind, sx, ridx, year] * cov_amount[k]
                if cov_risk[kind] == 0:
                    claim_rate += rate
                else:
                    morb_rate += rate
            last_year = year
        q = mortality_grid[sx, ridx, year]
        ds = discount_start[t]
        dm = discount_mid[t]
        single = cnt * single_premium[mp] if t == 0 else 0.0
        level = inforce * premium if t < premium_term else 0.0
        pp += (level + single) * ds
        pc += inforce * claim_rate * dm
        pcm += inforce * morb_rate * dm
        pa += inforce * annuity * ds
        acquisition = cnt * expense_acquisition if t == 0 else 0.0
        pe += (acquisition + inforce * maint_monthly * inflation[t]) * dm
        inforce *= (1.0 - q) * (1.0 - lapse_by_year[year])
    pm = inforce * maturity_benefit[mp] * discount_start[term]
    # Diagnosis coverages: claims run off a depleting "not yet diagnosed" pool.
    for k in range(c_start, c_end):
        kind = cov_kind[k]
        if not cov_is_diagnosis[kind]:
            continue
        benefit = cov_amount[k]
        healthy = cnt
        d_year = -1
        d_rate = 0.0
        surv = 0.0
        for t in range(term):
            year = t // 12
            if year != d_year:
                d_rate = cov_rates[kind, sx, ridx, year]
                surv = ((1.0 - mortality_grid[sx, ridx, year])
                        * (1.0 - lapse_by_year[year]))
                d_year = year
            pcm += healthy * d_rate * benefit * discount_mid[t]
            healthy *= surv * (1.0 - d_rate)
    bel_mp = pc + pcm + pm + pa + pe - pp
    ra_mp = (mortality_factor * pc + morbidity_factor * pcm
             + longevity_factor * (pm + pa))
    fcf = bel_mp + ra_mp
    bel[mp] = bel_mp
    ra[mp] = ra_mp
    csm[mp] = max(0.0, -fcf)
    loss_component[mp] = max(0.0, fcf)


def value_gpu(mortality_grid, issue_index, sex, term_months, count, lapse_by_year,
              monthly_premium, single_premium, premium_term_months, cov_kind,
              cov_amount,
              cov_offset, cov_rates, cov_risk, cov_is_diagnosis,
              maturity_benefit, annuity_payment, expense_acquisition,
              maint_monthly, inflation, discount_start, discount_mid,
              mortality_factor, morbidity_factor, longevity_factor):
    """Run the fused valuation kernel on the GPU.

    Returns the four ``(n_mp,)`` valuation arrays: BEL, RA, CSM and the
    loss component.
    """
    if not cuda.is_available():
        raise RuntimeError(
            "backend='gpu' requires a CUDA device; none is available"
        )

    n_mp = issue_index.shape[0]
    d_mortality = cuda.to_device(mortality_grid)
    d_issue = cuda.to_device(issue_index)
    d_sex = cuda.to_device(sex)
    d_term = cuda.to_device(term_months)
    d_count = cuda.to_device(count)
    d_lapse = cuda.to_device(lapse_by_year)
    d_premium = cuda.to_device(monthly_premium)
    d_single = cuda.to_device(single_premium)
    d_premium_term = cuda.to_device(premium_term_months)
    d_cov_kind = cuda.to_device(cov_kind)
    d_cov_amount = cuda.to_device(cov_amount)
    d_cov_offset = cuda.to_device(cov_offset)
    d_cov_rates = cuda.to_device(cov_rates)
    d_cov_risk = cuda.to_device(cov_risk)
    d_cov_is_diagnosis = cuda.to_device(cov_is_diagnosis)
    d_maturity = cuda.to_device(maturity_benefit)
    d_annuity = cuda.to_device(annuity_payment)
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
        d_mortality, d_issue, d_sex, d_term, d_count, d_lapse, d_premium,
        d_single, d_premium_term, d_cov_kind, d_cov_amount, d_cov_offset,
        d_cov_rates, d_cov_risk,
        d_cov_is_diagnosis, d_maturity, d_annuity, expense_acquisition,
        maint_monthly, d_inflation, d_discount_start, d_discount_mid,
        mortality_factor, morbidity_factor, longevity_factor,
        d_bel, d_ra, d_csm, d_loss,
    )
    cuda.synchronize()

    return (
        d_bel.copy_to_host(),
        d_ra.copy_to_host(),
        d_csm.copy_to_host(),
        d_loss.copy_to_host(),
    )
