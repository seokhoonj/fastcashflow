"""CUDA backend for the fused valuation kernel.

Imported lazily by ``engine.value(backend="gpu")``; never loaded for CPU-only
use, so a machine without CUDA can still import and use the rest of the
package.

The kernel mirrors the CPU codegen fast kernel
(``gmm._codegen._get_markov_kernel``) exactly; the
cross-check test ``test_fast_gpu_matches_cpu`` guards against divergence.
"""
from __future__ import annotations

import numpy as np
from numba import cuda, float64

# Per-thread occupancy scratch is a fixed-size CUDA local array; a state
# model with more transient states than this cannot run on the GPU backend.
MAX_STATES = 8


@cuda.jit
def _value_cuda_kernel(edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
                       state_pays_premium, state_pays_benefit, start_state, issue_index,
                       sex, term_months, contract_boundary_months, count, premium,
                       premium_term_months, premium_frequency_months, annuity_frequency_months,
                       coverage_index, coverage_amount, coverage_offset, coverage_rates,
                       premium_factor, annuity_factor, coverage_risk,
                       coverage_is_diagnosis, maturity_benefit, annuity_payment,
                       disability_income, disability_benefit,
                       acquisition_premium, acquisition_per_policy, maintenance_premium,
                       maintenance_per_policy, lae,
                       discount_factor_bom, discount_factor_mid,
                       mortality_factor, morbidity_factor, longevity_factor,
                       disability_factor, lapse_monthly, state_lapse, surrender_curve,
                       surrender_is_amount, surrender_base,
                       bel, ra, csm, loss_component):
    """One CUDA thread per model point; the per-month loop runs in the thread.

    In-force is an occupancy vector over ``n_states`` transient states,
    advanced each month along the transition edges. Premium accrues on the
    states flagged in ``state_pays_premium``; claims, expenses and survival
    benefits on the total occupancy; disability income on the
    ``state_pays_benefit`` occupancy, and a lump-sum transition on its flow.
    """
    mp = cuda.grid(1)
    if mp >= issue_index.shape[0]:
        return

    n_edges = edge_from.shape[0]
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
    ss = start_state[mp]
    occ = cuda.local.array(MAX_STATES, float64)
    occ_next = cuda.local.array(MAX_STATES, float64)
    for s in range(n_states):
        occ[s] = 0.0
    occ[ss] = cnt
    pv_mortality = 0.0
    pv_morbidity = 0.0
    pv_disability = 0.0
    pv_premium = 0.0
    pv_expense = 0.0
    pv_annuity = 0.0
    pv_surrender = 0.0
    cum_premium = 0.0
    last_year = -1
    claim_rate = 0.0
    morb_rate = 0.0
    for t in range(boundary):
        year = t // 12
        if year != last_year:
            claim_rate = 0.0
            morb_rate = 0.0
            for k in range(c_start, c_end):
                cov_idx = coverage_index[k]
                if coverage_is_diagnosis[cov_idx]:
                    continue
                rate = coverage_rates[cov_idx, sx, age_idx, year] * coverage_amount[k]
                if coverage_risk[cov_idx] == 0:
                    claim_rate += rate
                else:
                    morb_rate += rate
            last_year = year
        inforce_t = 0.0
        prem_occ = 0.0
        benefit_occ = 0.0
        for s in range(n_states):
            inforce_t += occ[s]
            if state_pays_premium[s]:
                prem_occ += occ[s]
            if state_pays_benefit[s]:
                benefit_occ += occ[s]
        discount_factor_bom_t = discount_factor_bom[t]
        discount_factor_mid_t = discount_factor_mid[t]
        level = (prem_occ * prem * premium_factor[sx, age_idx, year]
                 if (t < premium_term and t % prem_freq == 0) else 0.0)
        pv_premium += level * discount_factor_bom_t
        cum_premium += level
        pv_mortality += inforce_t * claim_rate * discount_factor_mid_t
        pv_morbidity += inforce_t * morb_rate * discount_factor_mid_t
        if t % ann_freq == 0:
            pv_annuity += inforce_t * annuity * annuity_factor[sx, age_idx, year] * discount_factor_bom_t
        pv_disability += benefit_occ * disability_income[mp] * discount_factor_mid_t
        ann_prem = premium[mp] * premium_factor[sx, age_idx, year] * 12.0 / prem_freq
        acquisition_expense = (cnt * (acquisition_premium * ann_prem + acquisition_per_policy)
                 if t == 0 else 0.0)
        maintenance_premium_expense = (inforce_t * maintenance_premium[t] * ann_prem / 12.0
                if t < premium_term else 0.0)
        maintenance_per_policy_expense = inforce_t * maintenance_per_policy[t]
        lae_expense = lae[t] * inforce_t * (claim_rate + morb_rate)
        pv_expense += (acquisition_expense + maintenance_premium_expense + maintenance_per_policy_expense + lae_expense) * discount_factor_mid_t
        lapse_flow = 0.0
        for s in range(n_states):
            lapse_flow += occ[s] * state_lapse[s, sx, age_idx, year]
        if surrender_is_amount:
            # amount_per_policy / amount_per_unit: surrender_curve[t] is the
            # surrender amount at duration t (per policy, or per unit of
            # surrender_base[mp]); lapse_flow is the state-machine number lapsing.
            pv_surrender += (lapse_flow * surrender_curve[t]
                             * surrender_base[mp] * discount_factor_mid_t)
        else:
            # cum_premium aggregates inforce * premium; the effective lapse
            # fraction is lapse_flow / inforce_t (the raw rate for a single state).
            eff_lapse = lapse_flow / inforce_t if inforce_t > 0.0 else 0.0
            pv_surrender += (eff_lapse
                             * cum_premium * surrender_curve[t] * discount_factor_mid_t)
        for s in range(n_states):
            occ_next[s] = 0.0
        for e in range(n_edges):
            flow = occ[edge_from[e]] * edge_prob[sx, age_idx, year, e]
            occ_next[edge_to[e]] += flow
            if edge_lump_sum[e]:
                pv_disability += flow * disability_benefit[mp] * discount_factor_mid_t
        for s in range(n_states):
            occ[s] = occ_next[s]
    total = 0.0
    for s in range(n_states):
        total += occ[s]
    pm = (total * maturity_benefit[mp] * discount_factor_bom[boundary]
          if boundary == term else 0.0)
    # Diagnosis coverages: claims run off a depleting "not yet diagnosed"
    # occupancy, carried over the transient states.
    for k in range(c_start, c_end):
        cov_idx = coverage_index[k]
        if not coverage_is_diagnosis[cov_idx]:
            continue
        benefit = coverage_amount[k]
        for s in range(n_states):
            occ[s] = 0.0
        occ[ss] = cnt
        d_year = -1
        d_rate = 0.0
        for t in range(boundary):
            year = t // 12
            if year != d_year:
                d_rate = coverage_rates[cov_idx, sx, age_idx, year]
                d_year = year
            healthy = 0.0
            for s in range(n_states):
                healthy += occ[s]
            pv_morbidity += healthy * d_rate * benefit * discount_factor_mid[t]
            undiagnosed = 1.0 - d_rate
            for s in range(n_states):
                occ_next[s] = 0.0
            for e in range(n_edges):
                occ_next[edge_to[e]] += (occ[edge_from[e]] * undiagnosed
                                         * edge_prob[sx, age_idx, year, e])
            for s in range(n_states):
                occ[s] = occ_next[s]
    bel_mp = (pv_mortality + pv_morbidity + pv_disability + pm
              + pv_annuity + pv_expense + pv_surrender - pv_premium)
    ra_mp = (mortality_factor * pv_mortality + morbidity_factor * pv_morbidity
             + disability_factor * pv_disability
             + longevity_factor * (pm + pv_annuity))
    fcf = bel_mp + ra_mp
    bel[mp] = bel_mp
    ra[mp] = ra_mp
    csm[mp] = max(0.0, -fcf)
    loss_component[mp] = max(0.0, fcf)


def fast_gpu(edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
              state_pays_premium, state_pays_benefit, start_state, issue_index, sex,
              term_months, contract_boundary_months, count, premium,
              premium_term_months, premium_frequency_months, annuity_frequency_months,
              coverage_index, coverage_amount, coverage_offset, coverage_rates,
              premium_factor, annuity_factor, coverage_risk,
              coverage_is_diagnosis, maturity_benefit, annuity_payment,
              disability_income, disability_benefit,
              acquisition_premium, acquisition_per_policy, maintenance_premium,
              maintenance_per_policy, lae,
              discount_factor_bom, discount_factor_mid,
              mortality_factor, morbidity_factor, longevity_factor,
              disability_factor, lapse_monthly, state_lapse, surrender_curve,
              surrender_is_amount=False, surrender_base=None):
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
    d_state_pays_premium = cuda.to_device(state_pays_premium)
    d_state_pays_benefit = cuda.to_device(state_pays_benefit)
    d_start_state = cuda.to_device(start_state)
    d_issue = cuda.to_device(issue_index)
    d_sex = cuda.to_device(sex)
    d_term = cuda.to_device(term_months)
    d_boundary = cuda.to_device(contract_boundary_months)
    d_count = cuda.to_device(count)
    d_premium = cuda.to_device(premium)
    d_premium_term = cuda.to_device(premium_term_months)
    d_premium_freq = cuda.to_device(premium_frequency_months)
    d_annuity_freq = cuda.to_device(annuity_frequency_months)
    d_coverage_index = cuda.to_device(coverage_index)
    d_cov_amount = cuda.to_device(coverage_amount)
    d_cov_offset = cuda.to_device(coverage_offset)
    d_coverage_rates = cuda.to_device(coverage_rates)
    d_premium_factor = cuda.to_device(premium_factor)
    d_annuity_factor = cuda.to_device(annuity_factor)
    d_coverage_risk = cuda.to_device(coverage_risk)
    d_coverage_is_diagnosis = cuda.to_device(coverage_is_diagnosis)
    d_maturity = cuda.to_device(maturity_benefit)
    d_annuity = cuda.to_device(annuity_payment)
    d_disability_income = cuda.to_device(disability_income)
    d_disability_benefit = cuda.to_device(disability_benefit)
    d_gamma_fixed = cuda.to_device(maintenance_per_policy)
    d_lae_pro_rata = cuda.to_device(lae)
    d_discount_factor_bom = cuda.to_device(discount_factor_bom)
    d_discount_factor_mid = cuda.to_device(discount_factor_mid)
    d_lapse_monthly = cuda.to_device(lapse_monthly)
    d_state_lapse = cuda.to_device(state_lapse)
    d_surrender_curve = cuda.to_device(surrender_curve)
    if surrender_base is None:
        surrender_base = np.ones(n_mp, dtype=np.float64)
    d_surrender_base = cuda.to_device(np.asarray(surrender_base, dtype=np.float64))
    d_bel = cuda.device_array(n_mp, dtype=np.float64)
    d_ra = cuda.device_array(n_mp, dtype=np.float64)
    d_csm = cuda.device_array(n_mp, dtype=np.float64)
    d_loss = cuda.device_array(n_mp, dtype=np.float64)

    threads = 256
    blocks = (n_mp + threads - 1) // threads
    _value_cuda_kernel[blocks, threads](
        d_edge_from, d_edge_to, d_edge_prob, d_edge_lump, n_states,
        d_state_pays_premium, d_state_pays_benefit, d_start_state, d_issue, d_sex,
        d_term, d_boundary, d_count, d_premium, d_premium_term, d_premium_freq,
        d_annuity_freq, d_coverage_index, d_cov_amount, d_cov_offset, d_coverage_rates,
        d_premium_factor,
        d_annuity_factor,
        d_coverage_risk, d_coverage_is_diagnosis, d_maturity, d_annuity,
        d_disability_income, d_disability_benefit,
        acquisition_premium, acquisition_per_policy, maintenance_premium,
        d_gamma_fixed, d_lae_pro_rata, d_discount_factor_bom,
        d_discount_factor_mid, mortality_factor, morbidity_factor, longevity_factor,
        disability_factor, d_lapse_monthly, d_state_lapse, d_surrender_curve,
        surrender_is_amount, d_surrender_base,
        d_bel, d_ra, d_csm, d_loss,
    )
    cuda.synchronize()

    return (
        d_bel.copy_to_host(),
        d_ra.copy_to_host(),
        d_csm.copy_to_host(),
        d_loss.copy_to_host(),
    )
