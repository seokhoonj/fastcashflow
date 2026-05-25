"""CUDA backend for the fused valuation kernel.

Imported lazily by ``engine.value(backend="gpu")``; never loaded for CPU-only
use, so a machine without CUDA can still import and use the rest of the
package.

The kernel mirrors the CPU kernel in ``engine._value_kernel`` exactly; the
cross-check test ``test_value_gpu_matches_cpu`` guards against divergence.
"""
from __future__ import annotations

import numpy as np
from numba import cuda, float64

# Per-thread occupancy scratch is a fixed-size CUDA local array; a state
# model with more transient states than this cannot run on the GPU backend.
MAX_STATES = 8


@cuda.jit
def _value_cuda_kernel(edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
                       premium_state, benefit_state, start_state, issue_index,
                       sex, term_months, count, level_premium, single_premium,
                       premium_term_months, premium_frequency, annuity_frequency,
                       coverage_kind, coverage_amount, coverage_offset, cov_rates, cov_risk,
                       cov_is_diagnosis, maturity_benefit, annuity_payment,
                       disability_income, disability_benefit,
                       alpha_pct, alpha_flat, beta_pct,
                       gamma_inflated_monthly,
                       discount_start, discount_mid,
                       mortality_factor, morbidity_factor, longevity_factor,
                       disability_factor, lapse_monthly, surrender_curve,
                       bel, ra, csm, loss_component):
    """One CUDA thread per model point; the per-month loop runs in the thread.

    In-force is an occupancy vector over ``n_states`` transient states,
    advanced each month along the transition edges. Premium accrues on the
    states flagged in ``premium_state``; claims, expenses and survival
    benefits on the total occupancy; disability income on the
    ``benefit_state`` occupancy, and a lump-sum transition on its flow.
    """
    mp = cuda.grid(1)
    if mp >= issue_index.shape[0]:
        return

    n_edges = edge_from.shape[0]
    term = term_months[mp]
    premium_term = premium_term_months[mp]
    prem_freq = premium_frequency[mp]
    ann_freq = annuity_frequency[mp]
    ridx = issue_index[mp]
    sx = sex[mp]
    cnt = count[mp]
    premium = level_premium[mp]
    annuity = annuity_payment[mp]
    c_start = coverage_offset[mp]
    c_end = coverage_offset[mp + 1]
    ss = start_state[mp]
    occ = cuda.local.array(MAX_STATES, float64)
    occ_next = cuda.local.array(MAX_STATES, float64)
    for s in range(n_states):
        occ[s] = 0.0
    occ[ss] = cnt
    pc = 0.0
    pcm = 0.0
    pd = 0.0
    pp = 0.0
    pe = 0.0
    pa = 0.0
    ps = 0.0
    cum_premium = 0.0
    last_year = -1
    claim_rate = 0.0
    morb_rate = 0.0
    for t in range(term):
        year = t // 12
        if year != last_year:
            claim_rate = 0.0
            morb_rate = 0.0
            for k in range(c_start, c_end):
                kind = coverage_kind[k]
                if cov_is_diagnosis[kind]:
                    continue
                rate = cov_rates[kind, sx, ridx, year] * coverage_amount[k]
                if cov_risk[kind] == 0:
                    claim_rate += rate
                else:
                    morb_rate += rate
            last_year = year
        ift = 0.0
        prem_occ = 0.0
        benefit_occ = 0.0
        for s in range(n_states):
            ift += occ[s]
            if premium_state[s]:
                prem_occ += occ[s]
            if benefit_state[s]:
                benefit_occ += occ[s]
        ds = discount_start[t]
        dm = discount_mid[t]
        single = prem_occ * single_premium[mp] if t == 0 else 0.0
        level = (prem_occ * premium
                 if (t < premium_term and t % prem_freq == 0) else 0.0)
        pp += (level + single) * ds
        cum_premium += level + single
        pc += ift * claim_rate * dm
        pcm += ift * morb_rate * dm
        if t % ann_freq == 0:
            pa += ift * annuity * ds
        pd += benefit_occ * disability_income[mp] * dm
        ann_prem = level_premium[mp] * 12.0 / prem_freq
        alpha = (cnt * (alpha_pct * ann_prem + alpha_flat)
                 if t == 0 else 0.0)
        beta = (ift * beta_pct * ann_prem / 12.0
                if t < premium_term else 0.0)
        gamma = ift * gamma_inflated_monthly[t]
        pe += (alpha + beta + gamma) * dm
        ps += (ift * lapse_monthly[sx, ridx, year]
               * cum_premium * surrender_curve[t] * dm)
        for s in range(n_states):
            occ_next[s] = 0.0
        for e in range(n_edges):
            flow = occ[edge_from[e]] * edge_prob[sx, ridx, year, e]
            occ_next[edge_to[e]] += flow
            if edge_lump_sum[e]:
                pd += flow * disability_benefit[mp] * dm
        for s in range(n_states):
            occ[s] = occ_next[s]
    total = 0.0
    for s in range(n_states):
        total += occ[s]
    pm = total * maturity_benefit[mp] * discount_start[term]
    # Diagnosis coverages: claims run off a depleting "not yet diagnosed"
    # occupancy, carried over the transient states.
    for k in range(c_start, c_end):
        kind = coverage_kind[k]
        if not cov_is_diagnosis[kind]:
            continue
        benefit = coverage_amount[k]
        for s in range(n_states):
            occ[s] = 0.0
        occ[ss] = cnt
        d_year = -1
        d_rate = 0.0
        for t in range(term):
            year = t // 12
            if year != d_year:
                d_rate = cov_rates[kind, sx, ridx, year]
                d_year = year
            healthy = 0.0
            for s in range(n_states):
                healthy += occ[s]
            pcm += healthy * d_rate * benefit * discount_mid[t]
            undiag = 1.0 - d_rate
            for s in range(n_states):
                occ_next[s] = 0.0
            for e in range(n_edges):
                occ_next[edge_to[e]] += (occ[edge_from[e]] * undiag
                                         * edge_prob[sx, ridx, year, e])
            for s in range(n_states):
                occ[s] = occ_next[s]
    bel_mp = pc + pcm + pd + pm + pa + pe + ps - pp
    ra_mp = (mortality_factor * pc + morbidity_factor * pcm
             + disability_factor * pd + longevity_factor * (pm + pa))
    fcf = bel_mp + ra_mp
    bel[mp] = bel_mp
    ra[mp] = ra_mp
    csm[mp] = max(0.0, -fcf)
    loss_component[mp] = max(0.0, fcf)


def value_gpu(edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
              premium_state, benefit_state, start_state, issue_index, sex,
              term_months, count, level_premium, single_premium,
              premium_term_months, premium_frequency, annuity_frequency,
              coverage_kind, coverage_amount, coverage_offset, cov_rates, cov_risk,
              cov_is_diagnosis, maturity_benefit, annuity_payment,
              disability_income, disability_benefit,
              alpha_pct, alpha_flat, beta_pct,
              gamma_inflated_monthly, discount_start, discount_mid,
              mortality_factor, morbidity_factor, longevity_factor,
              disability_factor, lapse_monthly, surrender_curve):
    """Run the fused valuation kernel on the GPU.

    Returns the four ``(n_mp,)`` valuation arrays: BEL, RA, CSM and the
    loss component.
    """
    if not cuda.is_available():
        raise RuntimeError(
            "backend='gpu' requires a CUDA device; none is available"
        )
    if n_states > MAX_STATES:
        raise ValueError(
            f"the GPU backend supports up to {MAX_STATES} states; "
            f"the state model has {n_states}"
        )

    n_mp = issue_index.shape[0]
    d_edge_from = cuda.to_device(edge_from)
    d_edge_to = cuda.to_device(edge_to)
    d_edge_prob = cuda.to_device(edge_prob)
    d_edge_lump = cuda.to_device(edge_lump_sum)
    d_premium_state = cuda.to_device(premium_state)
    d_benefit_state = cuda.to_device(benefit_state)
    d_start_state = cuda.to_device(start_state)
    d_issue = cuda.to_device(issue_index)
    d_sex = cuda.to_device(sex)
    d_term = cuda.to_device(term_months)
    d_count = cuda.to_device(count)
    d_premium = cuda.to_device(level_premium)
    d_single = cuda.to_device(single_premium)
    d_premium_term = cuda.to_device(premium_term_months)
    d_premium_freq = cuda.to_device(premium_frequency)
    d_annuity_freq = cuda.to_device(annuity_frequency)
    d_cov_kind = cuda.to_device(coverage_kind)
    d_cov_amount = cuda.to_device(coverage_amount)
    d_cov_offset = cuda.to_device(coverage_offset)
    d_cov_rates = cuda.to_device(cov_rates)
    d_cov_risk = cuda.to_device(cov_risk)
    d_cov_is_diagnosis = cuda.to_device(cov_is_diagnosis)
    d_maturity = cuda.to_device(maturity_benefit)
    d_annuity = cuda.to_device(annuity_payment)
    d_disability_income = cuda.to_device(disability_income)
    d_disability_benefit = cuda.to_device(disability_benefit)
    d_gamma_inflated = cuda.to_device(gamma_inflated_monthly)
    d_discount_start = cuda.to_device(discount_start)
    d_discount_mid = cuda.to_device(discount_mid)
    d_lapse_monthly = cuda.to_device(lapse_monthly)
    d_surrender_curve = cuda.to_device(surrender_curve)
    d_bel = cuda.device_array(n_mp, dtype=np.float64)
    d_ra = cuda.device_array(n_mp, dtype=np.float64)
    d_csm = cuda.device_array(n_mp, dtype=np.float64)
    d_loss = cuda.device_array(n_mp, dtype=np.float64)

    threads = 256
    blocks = (n_mp + threads - 1) // threads
    _value_cuda_kernel[blocks, threads](
        d_edge_from, d_edge_to, d_edge_prob, d_edge_lump, n_states,
        d_premium_state, d_benefit_state, d_start_state, d_issue, d_sex,
        d_term, d_count, d_premium, d_single, d_premium_term, d_premium_freq,
        d_annuity_freq, d_cov_kind, d_cov_amount, d_cov_offset, d_cov_rates,
        d_cov_risk, d_cov_is_diagnosis, d_maturity, d_annuity,
        d_disability_income, d_disability_benefit,
        alpha_pct, alpha_flat, beta_pct,
        d_gamma_inflated, d_discount_start,
        d_discount_mid, mortality_factor, morbidity_factor, longevity_factor,
        disability_factor, d_lapse_monthly, d_surrender_curve,
        d_bel, d_ra, d_csm, d_loss,
    )
    cuda.synchronize()

    return (
        d_bel.copy_to_host(),
        d_ra.copy_to_host(),
        d_csm.copy_to_host(),
        d_loss.copy_to_host(),
    )
