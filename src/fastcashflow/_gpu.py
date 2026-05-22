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

from fastcashflow.modelpoints import STATE_ACTIVE


@cuda.jit
def _value_cuda_kernel(mortality_grid, waiver_grid, issue_index, sex,
                       term_months, count, state, lapse_by_year,
                       monthly_premium, single_premium, premium_term_months,
                       cov_kind, cov_amount, cov_offset, cov_rates, cov_risk,
                       cov_is_diagnosis, maturity_benefit, annuity_payment,
                       expense_acquisition, maint_monthly, inflation,
                       discount_start, discount_mid,
                       mortality_factor, morbidity_factor, longevity_factor,
                       bel, ra, csm, loss_component):
    """One CUDA thread per model point; the per-month loop runs in the thread.

    The in-force amount is carried as two scalars -- an active track (paying
    premium) and a waiver track (premium waived, coverage continuing); the
    waiver-inception rate moves a fraction of the active track each month.
    """
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
    # Two in-force tracks -- active (paying premium) and waiver (premium
    # waived, coverage continuing). The input state seats the count.
    if state[mp] == STATE_ACTIVE:
        act = cnt
        wav = 0.0
    else:
        act = 0.0
        wav = cnt
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
        w = waiver_grid[sx, ridx, year]
        lapse = lapse_by_year[year]
        ds = discount_start[t]
        dm = discount_mid[t]
        total = act + wav
        single = act * single_premium[mp] if t == 0 else 0.0
        level = act * premium if t < premium_term else 0.0
        pp += (level + single) * ds
        pc += total * claim_rate * dm
        pcm += total * morb_rate * dm
        pa += total * annuity * ds
        acquisition = cnt * expense_acquisition if t == 0 else 0.0
        pe += (acquisition + total * maint_monthly * inflation[t]) * dm
        # Active loses mortality, waiver-inception and lapse; the waiver
        # track loses mortality only and gains the inceptions.
        act_next = act * (1.0 - q) * (1.0 - w) * (1.0 - lapse)
        wav = wav * (1.0 - q) + act * (1.0 - q) * w
        act = act_next
    pm = (act + wav) * maturity_benefit[mp] * discount_start[term]
    # Diagnosis coverages: claims run off a depleting "not yet diagnosed"
    # pool, carried over the two in-force tracks.
    for k in range(c_start, c_end):
        kind = cov_kind[k]
        if not cov_is_diagnosis[kind]:
            continue
        benefit = cov_amount[k]
        if state[mp] == STATE_ACTIVE:
            h_act = cnt
            h_wav = 0.0
        else:
            h_act = 0.0
            h_wav = cnt
        d_year = -1
        d_rate = 0.0
        for t in range(term):
            year = t // 12
            if year != d_year:
                d_rate = cov_rates[kind, sx, ridx, year]
                d_year = year
            pcm += (h_act + h_wav) * d_rate * benefit * discount_mid[t]
            q = mortality_grid[sx, ridx, year]
            w = waiver_grid[sx, ridx, year]
            lapse = lapse_by_year[year]
            undiag = 1.0 - d_rate
            h_act_next = h_act * undiag * (1.0 - q) * (1.0 - w) * (1.0 - lapse)
            h_wav = (h_wav * undiag * (1.0 - q)
                     + h_act * undiag * (1.0 - q) * w)
            h_act = h_act_next
    bel_mp = pc + pcm + pm + pa + pe - pp
    ra_mp = (mortality_factor * pc + morbidity_factor * pcm
             + longevity_factor * (pm + pa))
    fcf = bel_mp + ra_mp
    bel[mp] = bel_mp
    ra[mp] = ra_mp
    csm[mp] = max(0.0, -fcf)
    loss_component[mp] = max(0.0, fcf)


def value_gpu(mortality_grid, waiver_grid, issue_index, sex, term_months,
              count, state, lapse_by_year, monthly_premium, single_premium,
              premium_term_months, cov_kind, cov_amount, cov_offset, cov_rates,
              cov_risk, cov_is_diagnosis, maturity_benefit, annuity_payment,
              expense_acquisition, maint_monthly, inflation, discount_start,
              discount_mid, mortality_factor, morbidity_factor,
              longevity_factor):
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
    d_waiver = cuda.to_device(waiver_grid)
    d_issue = cuda.to_device(issue_index)
    d_sex = cuda.to_device(sex)
    d_term = cuda.to_device(term_months)
    d_count = cuda.to_device(count)
    d_state = cuda.to_device(state)
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
        d_mortality, d_waiver, d_issue, d_sex, d_term, d_count, d_state,
        d_lapse, d_premium, d_single, d_premium_term, d_cov_kind,
        d_cov_amount, d_cov_offset, d_cov_rates, d_cov_risk,
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
