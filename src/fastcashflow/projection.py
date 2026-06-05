"""Monthly cash flow projection -- the BaseProj layer.

Sign convention (liability perspective, used consistently across the engine):

    premium_cf  : insurer INFLOW  -- reduces the insurance liability
    claim_cf    : insurer OUTFLOW -- DEATH-pattern claims (priced via mortality_cv)
    morbidity_cf: insurer OUTFLOW -- MORBIDITY-pattern claims (priced via morbidity_cv)
    expense_cf  : insurer OUTFLOW -- increases the insurance liability
    annuity_cf  : insurer OUTFLOW -- increases the insurance liability
    maturity_cf : insurer OUTFLOW -- increases the insurance liability

Getting this convention consistent everywhere is the single most error-prone
part of a GMM engine, so it is stated once here and never re-decided.

Timing convention (monthly steps, month ``t`` spans ``[t, t+1)``):

    inforce[t]  : policies in force at the START of month t
    premium     : level premium charged at the start of month t, on
                  inforce[t], every premium_frequency months it is in force
                  (the single premium, if any, is added at t = 0)
    annuity     : paid at the start of month t, on inforce[t], every
                  annuity_frequency months
    deaths[t]   : occur during month t -- inforce[t] * monthly mortality
    lapses      : occur during month t, on the mortality survivors
    claim       : sum of the policy's coverages for events during month t;
                  death claims decrement, health claims do not (a health
                  claim leaves the policy in force -- multiple-occurrence)
    expense     : acquisition at t = 0; maintenance every in-force month
    maturity    : maturity benefit at time = term, paid to the survivors

Two layers: a compiled, parallel kernel (``_project_kernel``) runs the raw
time loop; a Pythonic wrapper (``project_cashflows``) prepares its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray
from fastcashflow.basis import (
    Basis, annual_to_monthly, derive_expense_components,
)
from fastcashflow.coverage import (
    align_coverages, build_coverage_rates, coverage_arrays, validate_csr_codes,
)
from fastcashflow.curves import inflation_index
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.statemodel import (
    compile_state_model,
    compile_state_model_with_duration,
    is_semi_markov,
    model_references_rate,
    resolve_state_model,
)


@dataclass(frozen=True, slots=True)
class Cashflows:
    """Projected cash flows.

    The per-month arrays are shaped ``(n_mp, n_time)``; ``maturity_cf`` is
    ``(n_mp,)`` -- one payment per policy, at that policy's term. Death and
    health claims are kept apart -- ``claim_cf`` and ``morbidity_cf`` -- so
    the Risk Adjustment can price the two risks separately; ``disability_cf``
    is a third risk class -- the income paid while in a benefit state plus
    the on-transition lump sum.
    """

    inforce: FloatArray       # policies in force at the start of each month
    deaths: FloatArray        # deaths during each month
    premium_cf: FloatArray    # premium inflow per month (single premium at t=0)
    claim_cf: FloatArray      # DEATH-pattern claim outflow per month (priced via mortality_cv)
    morbidity_cf: FloatArray  # MORBIDITY-pattern claim outflow per month (priced via morbidity_cv)
    expense_cf: FloatArray    # expense outflow per month
    annuity_cf: FloatArray    # annuity (survival income) outflow per month
    disability_cf: FloatArray # disability income + lump-sum outflow per month
    maturity_cf: FloatArray   # (n_mp,) maturity benefit, paid at time = term
    maturity_survivors: FloatArray  # (n_mp,) in-force reaching term (the maturity exit count)
    surrender_cf: FloatArray  # surrender value (해약환급금) paid on lapse

    @property
    def n_time(self) -> int:
        """Number of monthly projection steps."""
        return int(self.inforce.shape[1])


def _expense_kernel_args(
    basis: Basis, n_time: int,
) -> tuple[float, float, float, FloatArray, FloatArray]:
    """Return the five expense primitives the kernels take.

    Projects ``Basis.expense_items`` onto the kernel-side inputs,
    threading ``Basis.expense_inflation`` through the recurring
    rows via :func:`fastcashflow.curves.inflation_index`:

    - ``alpha_pro_rata``, ``alpha_fixed``, ``beta_pro_rata`` -- scalars used at
      ``t=0`` (alpha) and every premium-paying month (beta).
    - ``gamma_fixed`` -- ``(n_time,)`` per-policy monthly maintenance
      amount (with global inflation baked in).
    - ``lae_pro_rata`` -- ``(n_time,)`` LAE fraction applied each
      month to ``(claim + morbidity + disability)`` (with global
      inflation baked in).

    An empty ``expense_items`` produces five zeros -- the no-expense
    basis -- so the kernel can run unchanged.
    """
    return derive_expense_components(
        basis.expense_items, n_time, inflation_index(basis, n_time),
    )


@njit(parallel=True, cache=True)
def _project_kernel(state_mortality, state_lapse, edge_from, edge_to, edge_prob, edge_lump_sum,
                    n_states, premium_state, benefit_state, start_state,
                    term_months, contract_boundary_months, count, premium,
                    premium_term_months, premium_frequency_months, annuity_frequency_months,
                    coverage_index, coverage_amount, coverage_offset, coverage_waiting,
                    coverage_reduction_end, coverage_reduction_factor, coverage_rates,
                    coverage_risk, coverage_is_diagnosis, maturity_benefit,
                    annuity_payment, disability_income, disability_benefit,
                    alpha_pro_rata, alpha_fixed, beta_pro_rata,
                    gamma_fixed, lae_pro_rata, n_time):
    """Compiled, parallel time-loop kernel -- raw numpy arrays and scalars only.

    The model-point axis is the independent (outer) loop, run in parallel
    across cores; the time axis is the sequential (inner) loop, because the
    in-force recursion depends on the previous month.

    In-force is an occupancy vector over ``n_states`` transient states. Each
    month it is advanced along the transition edges: edge ``e`` carries
    ``edge_prob[e, mp, year]`` of the occupancy from state ``edge_from[e]``
    to ``edge_to[e]``. Premium accrues on the states flagged in
    ``premium_state``; claims, expenses and survival benefits on the total
    occupancy; disability income on the ``benefit_state`` occupancy, and a
    lump-sum transition pays on the flow it carries. The transition
    probabilities are composed by the caller, so the kernel itself is
    state-machine-agnostic.

    A policy's claim is the sum over its coverage list: coverage ``k`` pays
    ``coverage_amount[k]`` at rate ``coverage_rates[coverage_index[k], mp, year]``, summed
    into the mortality or morbidity total by the coverage's risk class. Coverage
    rates change only once a year, so the per-coverage sum is rebuilt on a
    year boundary. The maturity benefit is paid to the in-force survivors at
    time = term.
    """
    n_mp = state_mortality.shape[1]
    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))
    premium_cf = np.zeros((n_mp, n_time))
    claim_cf = np.zeros((n_mp, n_time))
    morbidity_cf = np.zeros((n_mp, n_time))
    expense_cf = np.zeros((n_mp, n_time))
    annuity_cf = np.zeros((n_mp, n_time))
    disability_cf = np.zeros((n_mp, n_time))
    lapse_flow = np.zeros((n_mp, n_time))   # state-machine lapse exits, for surrender
    maturity_cf = np.zeros(n_mp)
    maturity_survivors = np.zeros(n_mp)

    n_edges = edge_from.shape[0]
    for mp in prange(n_mp):
        term = term_months[mp]
        boundary = contract_boundary_months[mp]  # Sec. 34 horizon (<= term)
        premium_term = premium_term_months[mp]   # months the premium is paid
        prem_freq = premium_frequency_months[mp]        # months between premiums
        ann_freq = annuity_frequency_months[mp]         # months between annuity payouts
        cnt = count[mp]
        c_start = coverage_offset[mp]
        c_end = coverage_offset[mp + 1]
        # In-force occupancy over the transient states; the input state
        # seats the model point's count on its starting state.
        occ = np.zeros(n_states)
        occ_next = np.zeros(n_states)
        occ[start_state[mp]] = cnt
        last_year = -1
        claim_rate = 0.0      # aggregate mortality claim per unit in-force
        morb_rate = 0.0       # aggregate morbidity claim per unit in-force
        for t in range(boundary):
            year = t // 12
            ift = 0.0         # total in-force
            prem_occ = 0.0    # in-force on the premium-paying states
            benefit_occ = 0.0 # in-force on the benefit-paying states
            deaths_acc = 0.0  # state-conditional death count
            lapse_acc = 0.0   # state-conditional lapse count (surrender)
            for s in range(n_states):
                ift += occ[s]
                deaths_acc += occ[s] * state_mortality[s, mp, year]
                lapse_acc += occ[s] * state_lapse[s, mp, year]
                if premium_state[s]:
                    prem_occ += occ[s]
                if benefit_state[s]:
                    benefit_occ += occ[s]
            inforce[mp, t] = ift
            lapse_flow[mp, t] = lapse_acc
            if year != last_year:
                claim_rate = 0.0
                morb_rate = 0.0
                for k in range(c_start, c_end):
                    cov_idx = coverage_index[k]
                    if coverage_is_diagnosis[cov_idx]:
                        continue          # diagnosis coverages run separately
                    if coverage_waiting[k] != 0 or coverage_reduction_end[k] != 0:
                        continue          # rule-bearing coverages run separately
                    rate = coverage_rates[cov_idx, mp, year] * coverage_amount[k]
                    if coverage_risk[cov_idx] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year
            deaths[mp, t] = deaths_acc
            level = (prem_occ * premium[mp]
                     if (t < premium_term and t % prem_freq == 0) else 0.0)
            premium_cf[mp, t] = level
            claim_cf[mp, t] = ift * claim_rate
            morbidity_cf[mp, t] = ift * morb_rate
            annuity_cf[mp, t] = (ift * annuity_payment[mp]
                                 if t % ann_freq == 0 else 0.0)
            disability_cf[mp, t] = benefit_occ * disability_income[mp]
            # Expense: alpha / beta / gamma maintenance plus LAE on the
            # month's claim + morbidity total. Dispatched from
            # Basis.expense_items by basis (alpha_pro_rata /
            # alpha_fixed / beta_pro_rata / gamma_fixed / lae_pro_rata).
            ann_prem = premium[mp] * 12.0 / prem_freq
            alpha = (cnt * (alpha_pro_rata * ann_prem + alpha_fixed)
                     if t == 0 else 0.0)
            beta = (ift * beta_pro_rata * ann_prem / 12.0
                    if t < premium_term else 0.0)
            gamma = ift * gamma_fixed[t]
            # LAE applies to mortality + morbidity claims only --
            # disability income is a periodic annuity-like benefit, lump
            # sums are one-off transitions, and conflating either with
            # LAE would double-count. Add a dedicated basis later if the
            # practice ever needs it.
            lae = lae_pro_rata[t] * (
                claim_cf[mp, t] + morbidity_cf[mp, t])
            expense_cf[mp, t] = alpha + beta + gamma + lae
            # Advance the occupancy along the transition edges; a lump-sum
            # transition pays its benefit on the occupancy it carries.
            for s in range(n_states):
                occ_next[s] = 0.0
            for e in range(n_edges):
                flow = occ[edge_from[e]] * edge_prob[e, mp, year]
                occ_next[edge_to[e]] += flow
                if edge_lump_sum[e]:
                    disability_cf[mp, t] += flow * disability_benefit[mp]
            if t + 1 == term:
                total_next = 0.0
                for s in range(n_states):
                    total_next += occ_next[s]
                maturity_cf[mp] = total_next * maturity_benefit[mp]
                maturity_survivors[mp] = total_next
            for s in range(n_states):
                occ[s] = occ_next[s]

        # Non-diagnosis coverages carrying a waiting or reduced-benefit rule
        # run per month here, not in the year-aggregated rate above, because
        # the benefit multiplier can change partway through a year.
        for k in range(c_start, c_end):
            cov_idx = coverage_index[k]
            if coverage_is_diagnosis[cov_idx]:
                continue
            wait = coverage_waiting[k]
            red_end = coverage_reduction_end[k]
            if wait == 0 and red_end == 0:
                continue          # rule-free -- already in the aggregate
            benefit = coverage_amount[k]
            red_factor = coverage_reduction_factor[k]
            mortality_risk = coverage_risk[cov_idx] == 0
            for t in range(wait, boundary):
                mult = red_factor if t < red_end else 1.0
                amt = (inforce[mp, t] * coverage_rates[cov_idx, mp, t // 12]
                       * benefit * mult)
                if mortality_risk:
                    claim_cf[mp, t] += amt
                else:
                    morbidity_cf[mp, t] += amt

        # Diagnosis coverages pay once on first diagnosis, so each one's
        # claims run off a "not yet diagnosed" fraction of the in-force that
        # the diagnosis rate depletes (on top of mortality and lapse).
        for k in range(c_start, c_end):
            cov_idx = coverage_index[k]
            if not coverage_is_diagnosis[cov_idx]:
                continue
            benefit = coverage_amount[k]
            wait = coverage_waiting[k]
            red_end = coverage_reduction_end[k]
            red_factor = coverage_reduction_factor[k]
            undiagnosed = 1.0   # fraction of the in-force still undiagnosed
            d_year = -1
            d_rate = 0.0
            for t in range(boundary):
                year = t // 12
                if year != d_year:
                    d_rate = coverage_rates[cov_idx, mp, year]
                    d_year = year
                # A waiting period suppresses the payment, not the diagnosis:
                # the not-yet-diagnosed pool depletes either way.
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    morbidity_cf[mp, t] += (inforce[mp, t] * undiagnosed
                                            * d_rate * benefit * mult)
                undiagnosed *= (1.0 - d_rate)

    return (inforce, deaths, premium_cf, claim_cf, morbidity_cf, expense_cf,
            annuity_cf, disability_cf, lapse_flow, maturity_cf, maturity_survivors)


@njit(parallel=True, cache=True)
def _project_kernel_semi_markov(
    state_mortality, state_lapse, edge_from, edge_to, edge_prob, edge_lump_sum,
    n_states, state_duration_max, state_offset, benefit_max_months,
    premium_state, benefit_state, start_state,
    term_months, contract_boundary_months, count, premium,
    premium_term_months, premium_frequency_months, annuity_frequency_months,
    coverage_index, coverage_amount, coverage_offset, coverage_waiting,
    coverage_reduction_end, coverage_reduction_factor, coverage_rates,
    coverage_risk, coverage_is_diagnosis,
    maturity_benefit, annuity_payment, disability_income, disability_benefit,
    alpha_pro_rata, alpha_fixed, beta_pro_rata,
    gamma_fixed, lae_pro_rata, n_time,
):
    """Detailed semi-Markov projection -- main pass only.

    Cohort-aware analogue of :func:`_project_kernel`. State ``s`` has
    ``state_duration_max[s] = D`` monthly cohorts, indexed via the flat
    occupancy vector at ``state_offset[s] + tau`` for ``tau in 0..D-1``.
    Transitions whose rate is duration_dependent carry per-cohort
    probabilities through ``edge_prob``'s trailing axis (shape
    ``(n_edges, n_mp, n_year, max_D)``).

    The residual stay edge (``edge_from == edge_to``) advances each cohort
    to ``tau + 1``, with the last cohort absorbing the long tail. A
    transient transition enters the destination state's cohort 0.

    Coverage-rule and diagnosis-coverage passes are not emitted -- the
    semi-Markov prototype rejects model points carrying either, so the
    main pass alone is the full projection for the supported cases.
    """
    n_mp = state_mortality.shape[1]
    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))
    premium_cf = np.zeros((n_mp, n_time))
    claim_cf = np.zeros((n_mp, n_time))
    morbidity_cf = np.zeros((n_mp, n_time))
    expense_cf = np.zeros((n_mp, n_time))
    annuity_cf = np.zeros((n_mp, n_time))
    disability_cf = np.zeros((n_mp, n_time))
    lapse_flow = np.zeros((n_mp, n_time))   # state-machine lapse exits, for surrender
    maturity_cf = np.zeros(n_mp)
    maturity_survivors = np.zeros(n_mp)

    n_edges = edge_from.shape[0]
    total_cohorts = state_offset[n_states]

    for mp in prange(n_mp):
        term = term_months[mp]
        boundary = contract_boundary_months[mp]  # Sec. 34 horizon (<= term)
        premium_term = premium_term_months[mp]
        prem_freq = premium_frequency_months[mp]
        ann_freq = annuity_frequency_months[mp]
        cnt = count[mp]
        c_start = coverage_offset[mp]
        c_end = coverage_offset[mp + 1]

        # Flat per-mp occupancy. cohort tau of state s lives at
        # state_offset[s] + tau. Seating goes to cohort 0 of the start state.
        occ = np.zeros(total_cohorts)
        occ_next = np.zeros(total_cohorts)
        occ[state_offset[start_state[mp]]] = cnt

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
                        continue          # diagnosis coverages run separately
                    if coverage_waiting[k] != 0 or coverage_reduction_end[k] != 0:
                        continue          # rule-bearing coverages run separately
                    rate = coverage_rates[cov_idx, mp, year] * coverage_amount[k]
                    if coverage_risk[cov_idx] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year

            ift = 0.0
            prem_occ = 0.0
            benefit_occ = 0.0
            deaths_acc = 0.0  # state-conditional death count
            lapse_acc = 0.0   # state-conditional lapse count (surrender)
            for s in range(n_states):
                s_off = state_offset[s]
                D = state_duration_max[s]
                state_sum = 0.0
                for tau in range(D):
                    state_sum += occ[s_off + tau]
                ift += state_sum
                deaths_acc += state_sum * state_mortality[s, mp, year]
                lapse_acc += state_sum * state_lapse[s, mp, year]
                if premium_state[s]:
                    prem_occ += state_sum
                if benefit_state[s]:
                    cap = benefit_max_months[s]
                    if cap > 0:
                        # Pay only the cohorts still within the cap; lives
                        # past it stay in force but stop receiving income.
                        ben_sum = 0.0
                        for tau in range(cap):
                            ben_sum += occ[s_off + tau]
                        benefit_occ += ben_sum
                    else:
                        benefit_occ += state_sum

            inforce[mp, t] = ift
            lapse_flow[mp, t] = lapse_acc
            deaths[mp, t] = deaths_acc
            level = (prem_occ * premium[mp]
                     if (t < premium_term and t % prem_freq == 0) else 0.0)
            premium_cf[mp, t] = level
            claim_cf[mp, t] = ift * claim_rate
            morbidity_cf[mp, t] = ift * morb_rate
            annuity_cf[mp, t] = (ift * annuity_payment[mp]
                                  if t % ann_freq == 0 else 0.0)
            disability_cf[mp, t] = benefit_occ * disability_income[mp]
            # Expense: same dispatch as _project_kernel (see its comment).
            ann_prem = premium[mp] * 12.0 / prem_freq
            alpha = (cnt * (alpha_pro_rata * ann_prem + alpha_fixed)
                     if t == 0 else 0.0)
            beta = (ift * beta_pro_rata * ann_prem / 12.0
                    if t < premium_term else 0.0)
            gamma = ift * gamma_fixed[t]
            # LAE applies to mortality + morbidity claims only --
            # disability income is a periodic annuity-like benefit, lump
            # sums are one-off transitions, and conflating either with
            # LAE would double-count. Add a dedicated basis later if the
            # practice ever needs it.
            lae = lae_pro_rata[t] * (
                claim_cf[mp, t] + morbidity_cf[mp, t])
            expense_cf[mp, t] = alpha + beta + gamma + lae

            for i in range(total_cohorts):
                occ_next[i] = 0.0
            for e in range(n_edges):
                s_from = edge_from[e]
                s_to = edge_to[e]
                D_from = state_duration_max[s_from]
                src_off = state_offset[s_from]
                is_residual = s_from == s_to
                if is_residual:
                    for tau in range(D_from):
                        flow = occ[src_off + tau] * edge_prob[e, mp, year, tau]
                        next_tau = tau + 1 if tau + 1 < D_from else D_from - 1
                        occ_next[src_off + next_tau] += flow
                        if edge_lump_sum[e]:
                            disability_cf[mp, t] += flow * disability_benefit[mp]
                else:
                    dst_off = state_offset[s_to]
                    for tau in range(D_from):
                        flow = occ[src_off + tau] * edge_prob[e, mp, year, tau]
                        occ_next[dst_off] += flow
                        if edge_lump_sum[e]:
                            disability_cf[mp, t] += flow * disability_benefit[mp]

            if t + 1 == term:
                total_next = 0.0
                for i in range(total_cohorts):
                    total_next += occ_next[i]
                maturity_cf[mp] = total_next * maturity_benefit[mp]
                maturity_survivors[mp] = total_next

            for i in range(total_cohorts):
                occ[i] = occ_next[i]

        # Coverage-rule pass -- non-diagnosis coverages with a waiting or
        # reduction period. The benefit multiplier can change partway
        # through a year, so we walk per-month and apply it to the
        # saved total in-force. Cohort tracking is unnecessary here:
        # the multiplier rides the same in-force trajectory the main
        # pass already produced.
        for k in range(c_start, c_end):
            cov_idx = coverage_index[k]
            if coverage_is_diagnosis[cov_idx]:
                continue
            wait = coverage_waiting[k]
            red_end = coverage_reduction_end[k]
            if wait == 0 and red_end == 0:
                continue          # rule-free -- already in the main pass
            benefit = coverage_amount[k]
            red_factor = coverage_reduction_factor[k]
            mortality_risk = coverage_risk[cov_idx] == 0
            for t in range(wait, boundary):
                mult = red_factor if t < red_end else 1.0
                amt = (inforce[mp, t] * coverage_rates[cov_idx, mp, t // 12]
                       * benefit * mult)
                if mortality_risk:
                    claim_cf[mp, t] += amt
                else:
                    morbidity_cf[mp, t] += amt

        # Diagnosis-coverage pass -- claims run off a depleting "not yet
        # diagnosed" pool that drops by (1 - d_rate) each month. The pool
        # multiplies the cohort-aware in-force from the main pass.
        for k in range(c_start, c_end):
            cov_idx = coverage_index[k]
            if not coverage_is_diagnosis[cov_idx]:
                continue
            benefit = coverage_amount[k]
            wait = coverage_waiting[k]
            red_end = coverage_reduction_end[k]
            red_factor = coverage_reduction_factor[k]
            undiagnosed = 1.0   # fraction of the in-force still undiagnosed
            d_year = -1
            d_rate = 0.0
            for t in range(boundary):
                year = t // 12
                if year != d_year:
                    d_rate = coverage_rates[cov_idx, mp, year]
                    d_year = year
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    morbidity_cf[mp, t] += (inforce[mp, t] * undiagnosed
                                            * d_rate * benefit * mult)
                undiagnosed *= (1.0 - d_rate)

    return (inforce, deaths, premium_cf, claim_cf, morbidity_cf, expense_cf,
            annuity_cf, disability_cf, lapse_flow, maturity_cf, maturity_survivors)


def _add_state_mortality_rates(rate_dict, state_model, basis, sex_grid,
                               issue_age_grid, duration_grid,
                               issue_class_grid, elapsed_grid):
    """Add each state's distinct mortality decrement rate to ``rate_dict``.

    A state may carry its own in-force mortality under ``State.mortality_rate``
    (default ``"mortality"``) -- a post-diagnosis state with elevated death.
    Each distinct non-default name is read from ``basis.state_mortality_annual``
    (a name -> callable dict), falling back to the global ``mortality_annual``
    when the name is absent, so declaring the state without a table preserves
    behaviour.
    """
    table = basis.state_mortality_annual or {}
    for rname in {s.mortality_rate for s in state_model.states}:
        if rname == "mortality" or rname in rate_dict:
            continue
        mort_fn = table.get(rname) or basis.mortality_annual
        rate_dict[rname] = np.ascontiguousarray(annual_to_monthly(
            mort_fn(sex_grid, issue_age_grid, duration_grid,
                    issue_class_grid, elapsed_grid)))


# Transition rates that remove occupancy from the in-force set as a lapse (the
# surrender trigger). A state may carry at most one; the surrender value is paid
# on ``occupancy x this rate``, so a non-lapsing state (e.g. WAIVER) contributes
# nothing and a paid-up state lapses at its own ``lapse_paidup`` rate.
_LAPSE_RATES = ("lapse", "lapse_paidup")


def _state_lapse_stack(state_model, rate_dict):
    """Per-state monthly lapse rate, ``(n_states, n_mp, n_year)``.

    Mirrors the per-state mortality stack: each state's surrender count is
    ``occ[state] x state_lapse[state]``, so surrender follows the actual
    state-machine lapse (WAIVER does not lapse; paid-up lapses at
    ``lapse_paidup``) instead of a single global rate applied to the total
    in-force. A state with no lapse transition contributes a zero row.
    """
    zero = np.zeros_like(rate_dict["lapse"])
    rows = []
    for s in state_model.states:
        rname = None
        for tr in s.transitions:
            if tr.rate in _LAPSE_RATES:
                rname = tr.rate
                break
        rows.append(rate_dict[rname] if rname is not None else zero)
    return np.ascontiguousarray(np.stack(rows))


def project_cashflows(model_points: ModelPoints, basis: Basis) -> Cashflows:
    """Project cash flows for every model point.

    The Pythonic wrapper: it extracts raw arrays from the inputs and
    evaluates the basis. Mortality, lapse and the coverage rates are
    evaluated on the per-policy-year grid, not the full ``(n_mp, n_time)``
    grid -- all change only once a year, so this is an identical result for
    a twelfth of the work.
    """
    if model_points.term_months.shape[0] == 0:
        raise ValueError(
            "model_points is empty (n_mp=0); measure() cannot project a "
            "zero-policy portfolio. Filter empty segments upstream."
        )
    # The projection horizon is the contract boundary (Sec. 34), which
    # defaults to ``term_months`` -- so a book with no boundary cut sizes the
    # arrays exactly as before. A shorter boundary trims both the loop and the
    # array width.
    n_time = int(model_points.contract_boundary_months.max())  # months 0 .. n_time-1
    n_years = (n_time + 11) // 12
    durations = np.arange(n_years)

    sex_grid, _ = np.meshgrid(model_points.sex, durations, indexing="ij")
    issue_age_grid, duration_grid = np.meshgrid(
        model_points.issue_age, durations, indexing="ij"
    )
    issue_class_grid, _ = np.meshgrid(
        model_points.issue_class, durations, indexing="ij"
    )
    # ``elapsed`` axis -- carried only by semi-Markov sojourn-aware rates.
    # The standard (non-cohort) setup grid is elapsed=0 throughout: tables
    # without the axis broadcast over it (no effect), tables that declare
    # it are looked up at elapsed=0 here (a future cohort-aware pass plugs
    # the per-MP per-cohort elapsed values in).
    elapsed_grid = np.zeros_like(duration_grid)
    # Rates are supplied annual; the engine converts each to a monthly rate
    # on the constant-force basis (see basis.annual_to_monthly).
    mortality_annual = basis.mortality_annual(
        sex_grid, issue_age_grid, duration_grid,
        issue_class_grid, elapsed_grid)
    mortality = np.ascontiguousarray(annual_to_monthly(mortality_annual))
    if basis.waiver_incidence_annual is None:
        waiver = np.zeros_like(mortality)
    else:
        waiver = np.ascontiguousarray(annual_to_monthly(
            basis.waiver_incidence_annual(
                sex_grid, issue_age_grid, duration_grid,
                issue_class_grid, elapsed_grid)))
    lapse = np.ascontiguousarray(annual_to_monthly(
        basis.lapse_annual(
            sex_grid, issue_age_grid, duration_grid,
            issue_class_grid, elapsed_grid)))
    # Align the basis' coverages to the order the model points were
    # built against, so coverage_index integers index the right rate row.
    # Reading the portfolio never had to know this order -- it is resolved
    # here, the one place the basis enter. (Identity when the model
    # points were built against this same Basis.)
    aligned_coverages = align_coverages(
        basis.coverages, model_points.coverage_codes)
    validate_csr_codes(
        model_points.coverage_index, len(aligned_coverages),
        coverages=aligned_coverages,
        calculation_methods=model_points.calculation_methods,
    )
    coverage_is_diagnosis, coverage_risk = coverage_arrays(
        aligned_coverages, model_points.calculation_methods,
    )
    # build_coverage_rates stacks the per-coverage annual rates; the whole
    # stack is converted to monthly. mortality_annual above is the separate
    # in-force decrement input; a death coverage's claim payout is driven
    # by its own rate_table from basis.coverages.
    coverage_rates = np.ascontiguousarray(annual_to_monthly(build_coverage_rates(
        [r.rate for r in aligned_coverages],
        sex_grid, issue_age_grid, duration_grid,
        issue_class_grid, elapsed_grid,
    )))
    # Shape contract: _project_kernel / _project_kernel_semi_markov index
    # coverage_rates[coverage_index[k], mp, year]. Lock the shape here so a future
    # change to the grid construction surfaces at this assertion rather than
    # silently broadcasting into a wrong claim rate.
    assert coverage_rates.shape == (
        len(aligned_coverages), len(model_points.issue_age), n_years
    ), f"coverage_rates shape {coverage_rates.shape} != (n_cov, n_mp, n_years)"
    # Expense primitives -- the five inputs the kernel consumes. Honours
    # Basis.expense_items when set, otherwise the legacy alpha / beta
    # / gamma / expense_inflation scalars (see _expense_kernel_args).
    (expense_alpha_pro_rata, expense_alpha_fixed, expense_beta_pro_rata,
     gamma_fixed, lae_pro_rata) = _expense_kernel_args(
        basis, n_time,
    )

    # In-force state machine -- see ``statemodel.resolve_state_model`` for
    # the fallback policy when ``basis.state_model`` is unset.
    state_model = resolve_state_model(basis)
    start_state = np.asarray(state_model.seating, np.int64)[model_points.state]

    if is_semi_markov(state_model):
        # Phase (c) detailed projection. Build the rate dict the cohort-
        # aware compile expects: static rates stay (n_mp, n_year); the
        # duration-dependent reincidence rate carries an extra cohort
        # axis of length ``max_cohort`` -- the largest ``duration_max``
        # across the tracked states. Coverage-rule and diagnosis-coverage
        # passes ride the cohort-aware main pass: rule benefits scale the
        # saved per-month total in-force, diagnosis pools multiply that
        # same trajectory by a per-coverage depletion fraction.
        max_cohort = max(s.duration_max for s in state_model.states
                          if s.duration_max > 0)
        rate_dict = {"mortality": mortality, "lapse": lapse}
        _add_state_mortality_rates(rate_dict, state_model, basis,
                                   sex_grid, issue_age_grid, duration_grid,
                                   issue_class_grid, elapsed_grid)
        if basis.waiver_incidence_annual is not None:
            rate_dict["waiver_incidence"] = waiver
        if basis.ci_incidence_annual is not None:
            ci_inc = np.ascontiguousarray(annual_to_monthly(
                basis.ci_incidence_annual(
                    sex_grid, issue_age_grid, duration_grid,
                    issue_class_grid, elapsed_grid)))
            rate_dict["ci_incidence"] = ci_inc
        if (basis.ci_reincidence_annual is not None
                or basis.disability_recovery_annual is not None):
            # Broadcast (n_mp, 1, 1) sex + (n_mp, 1, 1) age +
            # (1, n_year, 1) duration + (1, 1, max_cohort) cohort to
            # (n_mp, n_year, max_cohort). Sojourn-aware rate callables
            # share the unified 5-arg signature; the ``elapsed`` axis
            # carries the cohort (months since entering the source state),
            # and ``issue_class`` is broadcast at zero on this setup grid.
            sex_4d = model_points.sex.reshape(-1, 1, 1)
            age_4d = model_points.issue_age.reshape(-1, 1, 1)
            dur_4d = np.arange(n_years).reshape(1, -1, 1)
            coh_4d = np.arange(max_cohort).reshape(1, 1, -1)
            ic_4d = np.zeros_like(coh_4d)
            if basis.ci_reincidence_annual is not None:
                rate_dict["ci_reincidence"] = np.ascontiguousarray(
                    annual_to_monthly(
                        basis.ci_reincidence_annual(
                            sex_4d, age_4d, dur_4d, ic_4d, coh_4d)))
            if basis.disability_recovery_annual is not None:
                rate_dict["disability_recovery"] = np.ascontiguousarray(
                    annual_to_monthly(
                        basis.disability_recovery_annual(
                            sex_4d, age_4d, dur_4d, ic_4d, coh_4d)))
        compiled = compile_state_model_with_duration(state_model, rate_dict)
        edge_from = compiled.edge_from
        edge_to = compiled.edge_to
        edge_prob = compiled.edge_prob
        edge_lump_sum = compiled.edge_lump_sum
        n_states = compiled.n_states
        premium_state = compiled.premium_state
        benefit_state = compiled.benefit_state
        state_duration_max = compiled.state_duration_max
        benefit_max_months = compiled.benefit_max_months
        # compile_state_model_with_duration returns ``edge_prob`` shape
        # ``(n_edges, n_mp, n_year, max_D)`` -- already in the layout the
        # detailed kernel reads (edge axis outer, cohort axis inner).
        state_offset = np.zeros(n_states + 1, dtype=np.int64)
        state_offset[1:] = np.cumsum(state_duration_max)
        # Per-state mortality stack (n_states, n_mp, n_year) -- each state's
        # in-force death decrement so the death-count reporter splits by state.
        state_mortality = np.stack(
            [rate_dict[s.mortality_rate] for s in state_model.states])
        state_lapse = _state_lapse_stack(state_model, rate_dict)
        (inforce, deaths, premium_cf, claim_cf, morbidity_cf, expense_cf,
         annuity_cf, disability_cf, lapse_flow,
         maturity_cf, maturity_survivors) = _project_kernel_semi_markov(
            state_mortality, state_lapse, edge_from, edge_to, edge_prob, edge_lump_sum,
            n_states, state_duration_max, state_offset, benefit_max_months,
            premium_state, benefit_state, start_state,
            model_points.term_months,
            model_points.contract_boundary_months,
            model_points.count,
            model_points.premium,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.coverage_index,
            model_points.coverage_amount,
            model_points.coverage_offset,
            model_points.coverage_waiting,
            model_points.coverage_reduction_end,
            model_points.coverage_reduction_factor,
            coverage_rates,
            coverage_risk,
            coverage_is_diagnosis,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            model_points.disability_income,
            model_points.disability_benefit,
            expense_alpha_pro_rata,
            expense_alpha_fixed,
            expense_beta_pro_rata,
            gamma_fixed,
            lae_pro_rata,
            n_time,
        )
    else:
        # Markov path -- mirror the semi-Markov branch above for the rates
        # that are not duration-dependent. A custom Markov topology that
        # references ``ci_incidence`` works the same way it does on the
        # semi-Markov side; the two 4D sojourn rates (``ci_reincidence``,
        # ``disability_recovery``) remain semi-Markov-only.
        rate_dict = {"mortality": mortality, "waiver_incidence": waiver,
                     "lapse": lapse}
        _add_state_mortality_rates(rate_dict, state_model, basis,
                                   sex_grid, issue_age_grid, duration_grid,
                                   issue_class_grid, elapsed_grid)
        if basis.ci_incidence_annual is not None:
            ci_inc = np.ascontiguousarray(annual_to_monthly(
                basis.ci_incidence_annual(
                    sex_grid, issue_age_grid, duration_grid,
                    issue_class_grid, elapsed_grid)))
            rate_dict["ci_incidence"] = ci_inc
        if model_references_rate(state_model, "lapse_paidup"):
            paidup_fn = (basis.lapse_paidup_annual
                         or basis.lapse_annual)
            rate_dict["lapse_paidup"] = np.ascontiguousarray(annual_to_monthly(
                paidup_fn(sex_grid, issue_age_grid, duration_grid,
                          issue_class_grid, elapsed_grid)))
        compiled = compile_state_model(state_model, rate_dict)
        edge_from = compiled.edge_from
        edge_to = compiled.edge_to
        edge_prob = compiled.edge_prob
        edge_lump_sum = compiled.edge_lump_sum
        n_states = compiled.n_states
        premium_state = compiled.premium_state
        benefit_state = compiled.benefit_state
        state_mortality = np.stack(
            [rate_dict[s.mortality_rate] for s in state_model.states])
        state_lapse = _state_lapse_stack(state_model, rate_dict)
        (inforce, deaths, premium_cf, claim_cf, morbidity_cf, expense_cf,
         annuity_cf, disability_cf, lapse_flow, maturity_cf, maturity_survivors) = _project_kernel(
            state_mortality,
            state_lapse,
            edge_from,
            edge_to,
            edge_prob,
            edge_lump_sum,
            n_states,
            premium_state,
            benefit_state,
            start_state,
            model_points.term_months,
            model_points.contract_boundary_months,
            model_points.count,
            model_points.premium,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.coverage_index,
            model_points.coverage_amount,
            model_points.coverage_offset,
            model_points.coverage_waiting,
            model_points.coverage_reduction_end,
            model_points.coverage_reduction_factor,
            coverage_rates,
            coverage_risk,
            coverage_is_diagnosis,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            model_points.disability_income,
            model_points.disability_benefit,
            expense_alpha_pro_rata,
            expense_alpha_fixed,
            expense_beta_pro_rata,
            gamma_fixed,
            lae_pro_rata,
            n_time,
        )
    # Surrender value (해약환급금) -- post-projection compute. ``lapse_flow``
    # is the per-month state-machine lapse exit count (occupancy on each state
    # times that state's own lapse rate), so the surrender follows the actual
    # lapse: a non-lapsing WAIVER state pays no surrender, and a paid-up state
    # lapses at ``lapse_paidup``, not the active rate. For a single active
    # state ``lapse_flow == inforce x lapse``, the historical formula.
    # ``surrender_value_curve = None`` falls back to zero, the historical
    # "lapse silently removes" behaviour.
    surrender_cf = np.zeros_like(expense_cf)
    curve = basis.surrender_value_curve
    if curve is not None:
        # Curve held flat past its end; clip lookup to its length. ``t`` here
        # is the absolute policy duration (the projection runs from
        # inception), so the in-force slice at ``elapsed`` reads
        # ``curve[elapsed + future_t]`` for free.
        c = np.asarray(curve, dtype=np.float64)
        idx = np.minimum(np.arange(n_time), c.shape[0] - 1)
        value = c[idx]
        mode = basis.surrender_value_basis
        if mode == "cum_premium_factor":
            # Sample-grade: a factor on cumulative premium. ``cum_premium``
            # aggregates inforce * premium each month; the effective lapse
            # fraction is ``lapse_flow / inforce`` (the raw rate for a single
            # state). Not linear in the as-of in-force (cum_premium is
            # path-dependent on pre-valuation premiums), so the in-force
            # rescale is inexact here.
            cum_premium = np.cumsum(premium_cf, axis=1)
            inforce_safe = np.where(inforce > 0.0, inforce, 1.0)
            surrender_cf = (lapse_flow / inforce_safe) * cum_premium * value
        elif mode == "amount_per_policy":
            # Contractual per-policy amount at policy-duration t. The number
            # lapsing in month t is ``lapse_flow[t]``; each pays ``value[t]``.
            # Linear in the in-force, so the in-force ``count / inforce[elapsed]``
            # rescale re-bases it exactly.
            surrender_cf = lapse_flow * value
        elif mode == "amount_per_unit":
            # Same as amount_per_policy, scaled by the per-MP base amount
            # (sum insured / basic premium / ...). Explicit -- no default base.
            base = model_points.surrender_base_amount
            if base is None:
                raise ValueError(
                    "surrender_value_basis='amount_per_unit' requires "
                    "ModelPoints.surrender_base_amount (no default base is "
                    "inferred)."
                )
            surrender_cf = (lapse_flow * value
                            * np.asarray(base, dtype=np.float64)[:, None])
        else:
            raise ValueError(
                f"unknown surrender_value_basis {mode!r}; expected "
                "'cum_premium_factor', 'amount_per_policy', or "
                "'amount_per_unit'."
            )
    return Cashflows(
        inforce=inforce,
        deaths=deaths,
        premium_cf=premium_cf,
        claim_cf=claim_cf,
        morbidity_cf=morbidity_cf,
        expense_cf=expense_cf,
        annuity_cf=annuity_cf,
        disability_cf=disability_cf,
        maturity_cf=maturity_cf,
        maturity_survivors=maturity_survivors,
        surrender_cf=surrender_cf,
    )
