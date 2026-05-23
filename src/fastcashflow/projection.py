"""Monthly cash flow projection -- the BaseProj layer.

Sign convention (liability perspective, used consistently across the engine):

    premium_cf  : insurer INFLOW  -- reduces the insurance liability
    claim_cf    : insurer OUTFLOW -- death claims (mortality risk)
    morbidity_cf: insurer OUTFLOW -- health claims (morbidity risk)
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

import warnings

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions, annual_to_monthly
from fastcashflow.coverage import coverage_arrays, coverage_rates
from fastcashflow.curves import inflation_index, maintenance_monthly_curve
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.statemodel import (
    WAIVER_MODEL,
    compile_state_model,
    compile_state_model_with_duration,
    is_semi_markov,
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
    claim_cf: FloatArray      # death-benefit outflow per month (mortality risk)
    morbidity_cf: FloatArray  # health-benefit outflow per month (morbidity risk)
    expense_cf: FloatArray    # expense outflow per month
    annuity_cf: FloatArray    # annuity (survival income) outflow per month
    disability_cf: FloatArray # disability income + lump-sum outflow per month
    maturity_cf: FloatArray   # (n_mp,) maturity benefit, paid at time = term

    @property
    def n_time(self) -> int:
        """Number of monthly projection steps."""
        return int(self.inforce.shape[1])


@njit(parallel=True, cache=True)
def _project_kernel(mortality, edge_from, edge_to, edge_prob, edge_lump_sum,
                    n_states, premium_state, benefit_state, start_state,
                    term_months, count, level_premium, single_premium,
                    premium_term_months, premium_frequency, annuity_frequency,
                    cov_kind, cov_amount, cov_offset, cov_waiting,
                    cov_reduction_end, cov_reduction_factor, cov_rates,
                    cov_risk, cov_is_diagnosis, maturity_benefit,
                    annuity_payment, disability_income, disability_benefit,
                    expense_acquisition, maint_inflated_monthly, n_time):
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
    ``cov_amount[k]`` at rate ``cov_rates[cov_kind[k], mp, year]``, summed
    into the mortality or morbidity total by the kind's risk class. Coverage
    rates change only once a year, so the per-coverage sum is rebuilt on a
    year boundary. The maturity benefit is paid to the in-force survivors at
    time = term.
    """
    n_mp = mortality.shape[0]
    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))
    premium_cf = np.zeros((n_mp, n_time))
    claim_cf = np.zeros((n_mp, n_time))
    morbidity_cf = np.zeros((n_mp, n_time))
    expense_cf = np.zeros((n_mp, n_time))
    annuity_cf = np.zeros((n_mp, n_time))
    disability_cf = np.zeros((n_mp, n_time))
    maturity_cf = np.zeros(n_mp)

    n_edges = edge_from.shape[0]
    for mp in prange(n_mp):
        term = term_months[mp]
        premium_term = premium_term_months[mp]   # months the premium is paid
        prem_freq = premium_frequency[mp]        # months between premiums
        ann_freq = annuity_frequency[mp]         # months between annuity payouts
        cnt = count[mp]
        c_start = cov_offset[mp]
        c_end = cov_offset[mp + 1]
        # In-force occupancy over the transient states; the input state
        # seats the model point's count on its starting state.
        occ = np.zeros(n_states)
        occ_next = np.zeros(n_states)
        occ[start_state[mp]] = cnt
        last_year = -1
        claim_rate = 0.0      # aggregate mortality claim per unit in-force
        morb_rate = 0.0       # aggregate morbidity claim per unit in-force
        for t in range(term):
            ift = 0.0         # total in-force
            prem_occ = 0.0    # in-force on the premium-paying states
            benefit_occ = 0.0 # in-force on the benefit-paying states
            for s in range(n_states):
                ift += occ[s]
                if premium_state[s]:
                    prem_occ += occ[s]
                if benefit_state[s]:
                    benefit_occ += occ[s]
            inforce[mp, t] = ift
            year = t // 12
            if year != last_year:
                claim_rate = 0.0
                morb_rate = 0.0
                for k in range(c_start, c_end):
                    kind = cov_kind[k]
                    if cov_is_diagnosis[kind]:
                        continue          # diagnosis coverages run separately
                    if cov_waiting[k] != 0 or cov_reduction_end[k] != 0:
                        continue          # rule-bearing coverages run separately
                    rate = cov_rates[kind, mp, year] * cov_amount[k]
                    if cov_risk[kind] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year
            q = mortality[mp, year]
            deaths[mp, t] = ift * q
            single = prem_occ * single_premium[mp] if t == 0 else 0.0
            level = (prem_occ * level_premium[mp]
                     if (t < premium_term and t % prem_freq == 0) else 0.0)
            premium_cf[mp, t] = level + single
            claim_cf[mp, t] = ift * claim_rate
            morbidity_cf[mp, t] = ift * morb_rate
            annuity_cf[mp, t] = (ift * annuity_payment[mp]
                                 if t % ann_freq == 0 else 0.0)
            disability_cf[mp, t] = benefit_occ * disability_income[mp]
            acquisition = cnt * expense_acquisition if t == 0 else 0.0
            expense_cf[mp, t] = acquisition + ift * maint_inflated_monthly[t]
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
            for s in range(n_states):
                occ[s] = occ_next[s]

        # Non-diagnosis coverages carrying a waiting or reduced-benefit rule
        # run per month here, not in the year-aggregated rate above, because
        # the benefit multiplier can change partway through a year.
        for k in range(c_start, c_end):
            kind = cov_kind[k]
            if cov_is_diagnosis[kind]:
                continue
            wait = cov_waiting[k]
            red_end = cov_reduction_end[k]
            if wait == 0 and red_end == 0:
                continue          # rule-free -- already in the aggregate
            benefit = cov_amount[k]
            red_factor = cov_reduction_factor[k]
            mortality_risk = cov_risk[kind] == 0
            for t in range(wait, term):
                mult = red_factor if t < red_end else 1.0
                amt = (inforce[mp, t] * cov_rates[kind, mp, t // 12]
                       * benefit * mult)
                if mortality_risk:
                    claim_cf[mp, t] += amt
                else:
                    morbidity_cf[mp, t] += amt

        # Diagnosis coverages pay once on first diagnosis, so each one's
        # claims run off a "not yet diagnosed" fraction of the in-force that
        # the diagnosis rate depletes (on top of mortality and lapse).
        for k in range(c_start, c_end):
            kind = cov_kind[k]
            if not cov_is_diagnosis[kind]:
                continue
            benefit = cov_amount[k]
            wait = cov_waiting[k]
            red_end = cov_reduction_end[k]
            red_factor = cov_reduction_factor[k]
            frac = 1.0          # fraction of the in-force still undiagnosed
            d_year = -1
            d_rate = 0.0
            for t in range(term):
                year = t // 12
                if year != d_year:
                    d_rate = cov_rates[kind, mp, year]
                    d_year = year
                # A waiting period suppresses the payment, not the diagnosis:
                # the not-yet-diagnosed pool depletes either way.
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    morbidity_cf[mp, t] += (inforce[mp, t] * frac * d_rate
                                            * benefit * mult)
                frac *= (1.0 - d_rate)

    return (inforce, deaths, premium_cf, claim_cf, morbidity_cf, expense_cf,
            annuity_cf, disability_cf, maturity_cf)


@njit(parallel=True, cache=True)
def _project_kernel_semi_markov(
    mortality, edge_from, edge_to, edge_prob, edge_lump_sum,
    n_states, state_duration_max, state_offset,
    premium_state, benefit_state, start_state,
    term_months, count, level_premium, single_premium,
    premium_term_months, premium_frequency, annuity_frequency,
    cov_kind, cov_amount, cov_offset, cov_waiting,
    cov_reduction_end, cov_reduction_factor, cov_rates,
    cov_risk, cov_is_diagnosis,
    maturity_benefit, annuity_payment, disability_income, disability_benefit,
    expense_acquisition, maint_inflated_monthly, n_time,
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
    n_mp = mortality.shape[0]
    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))
    premium_cf = np.zeros((n_mp, n_time))
    claim_cf = np.zeros((n_mp, n_time))
    morbidity_cf = np.zeros((n_mp, n_time))
    expense_cf = np.zeros((n_mp, n_time))
    annuity_cf = np.zeros((n_mp, n_time))
    disability_cf = np.zeros((n_mp, n_time))
    maturity_cf = np.zeros(n_mp)

    n_edges = edge_from.shape[0]
    total_cohorts = state_offset[n_states]

    for mp in prange(n_mp):
        term = term_months[mp]
        premium_term = premium_term_months[mp]
        prem_freq = premium_frequency[mp]
        ann_freq = annuity_frequency[mp]
        cnt = count[mp]
        c_start = cov_offset[mp]
        c_end = cov_offset[mp + 1]

        # Flat per-mp occupancy. cohort tau of state s lives at
        # state_offset[s] + tau. Seating goes to cohort 0 of the start state.
        occ = np.zeros(total_cohorts)
        occ_next = np.zeros(total_cohorts)
        occ[state_offset[start_state[mp]]] = cnt

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
                        continue          # diagnosis coverages run separately
                    if cov_waiting[k] != 0 or cov_reduction_end[k] != 0:
                        continue          # rule-bearing coverages run separately
                    rate = cov_rates[kind, mp, year] * cov_amount[k]
                    if cov_risk[kind] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year

            ift = 0.0
            prem_occ = 0.0
            benefit_occ = 0.0
            for s in range(n_states):
                s_off = state_offset[s]
                D = state_duration_max[s]
                state_sum = 0.0
                for tau in range(D):
                    state_sum += occ[s_off + tau]
                ift += state_sum
                if premium_state[s]:
                    prem_occ += state_sum
                if benefit_state[s]:
                    benefit_occ += state_sum

            inforce[mp, t] = ift
            q = mortality[mp, year]
            deaths[mp, t] = ift * q
            single = prem_occ * single_premium[mp] if t == 0 else 0.0
            level = (prem_occ * level_premium[mp]
                     if (t < premium_term and t % prem_freq == 0) else 0.0)
            premium_cf[mp, t] = level + single
            claim_cf[mp, t] = ift * claim_rate
            morbidity_cf[mp, t] = ift * morb_rate
            annuity_cf[mp, t] = (ift * annuity_payment[mp]
                                  if t % ann_freq == 0 else 0.0)
            disability_cf[mp, t] = benefit_occ * disability_income[mp]
            acquisition = cnt * expense_acquisition if t == 0 else 0.0
            expense_cf[mp, t] = acquisition + ift * maint_inflated_monthly[t]

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

            for i in range(total_cohorts):
                occ[i] = occ_next[i]

        # Coverage-rule pass -- non-diagnosis coverages with a waiting or
        # reduction period. The benefit multiplier can change partway
        # through a year, so we walk per-month and apply it to the
        # saved total in-force. Cohort tracking is unnecessary here:
        # the multiplier rides the same in-force trajectory the main
        # pass already produced.
        for k in range(c_start, c_end):
            kind = cov_kind[k]
            if cov_is_diagnosis[kind]:
                continue
            wait = cov_waiting[k]
            red_end = cov_reduction_end[k]
            if wait == 0 and red_end == 0:
                continue          # rule-free -- already in the main pass
            benefit = cov_amount[k]
            red_factor = cov_reduction_factor[k]
            mortality_risk = cov_risk[kind] == 0
            for t in range(wait, term):
                mult = red_factor if t < red_end else 1.0
                amt = (inforce[mp, t] * cov_rates[kind, mp, t // 12]
                       * benefit * mult)
                if mortality_risk:
                    claim_cf[mp, t] += amt
                else:
                    morbidity_cf[mp, t] += amt

        # Diagnosis-coverage pass -- claims run off a depleting "not yet
        # diagnosed" pool that drops by (1 - d_rate) each month. The pool
        # multiplies the cohort-aware in-force from the main pass.
        for k in range(c_start, c_end):
            kind = cov_kind[k]
            if not cov_is_diagnosis[kind]:
                continue
            benefit = cov_amount[k]
            wait = cov_waiting[k]
            red_end = cov_reduction_end[k]
            red_factor = cov_reduction_factor[k]
            frac = 1.0
            d_year = -1
            d_rate = 0.0
            for t in range(term):
                year = t // 12
                if year != d_year:
                    d_rate = cov_rates[kind, mp, year]
                    d_year = year
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    morbidity_cf[mp, t] += (inforce[mp, t] * frac
                                            * d_rate * benefit * mult)
                frac *= (1.0 - d_rate)

    return (inforce, deaths, premium_cf, claim_cf, morbidity_cf, expense_cf,
            annuity_cf, disability_cf, maturity_cf)


def project_cashflows(model_points: ModelPoints, assumptions: Assumptions) -> Cashflows:
    """Project cash flows for every model point.

    The Pythonic wrapper: it extracts raw arrays from the inputs and
    evaluates the assumptions. Mortality, lapse and the coverage rates are
    evaluated on the per-policy-year grid, not the full ``(n_mp, n_time)``
    grid -- all change only once a year, so this is an identical result for
    a twelfth of the work.
    """
    n_time = int(model_points.term_months.max())     # months 0 .. n_time-1
    n_years = (n_time + 11) // 12
    durations = np.arange(n_years)

    sex_grid, _ = np.meshgrid(model_points.sex, durations, indexing="ij")
    issue_age_grid, duration_grid = np.meshgrid(
        model_points.issue_age, durations, indexing="ij"
    )
    # Rates are supplied annual; the engine converts each to a monthly rate
    # on the constant-force basis (see assumptions.annual_to_monthly).
    mortality_annual = assumptions.mortality_annual(
        sex_grid, issue_age_grid, duration_grid)
    mortality = np.ascontiguousarray(annual_to_monthly(mortality_annual))
    if assumptions.waiver_incidence_annual is None:
        waiver = np.zeros_like(mortality)
    else:
        waiver = np.ascontiguousarray(annual_to_monthly(
            assumptions.waiver_incidence_annual(
                sex_grid, issue_age_grid, duration_grid)))
    lapse = np.ascontiguousarray(annual_to_monthly(
        assumptions.lapse_annual(sex_grid, issue_age_grid, duration_grid)))
    cov_is_diagnosis, cov_risk = coverage_arrays(assumptions.riders)
    # coverage_rates stacks the annual mortality and rider rates; the whole
    # stack is converted to monthly. Slab 0 is the monthly mortality above.
    cov_rates = np.ascontiguousarray(annual_to_monthly(coverage_rates(
        mortality_annual, [r.rate for r in assumptions.riders],
        sex_grid, issue_age_grid, duration_grid,
    )))
    maint_inflated_monthly = (maintenance_monthly_curve(assumptions, n_time)
                              * inflation_index(assumptions, n_time))

    # In-force state machine -- the StateModel composes the transition edges
    # the generic occupancy recursion advances; the kernel carries no state
    # set of its own. A product overrides it through assumptions.state_model.
    if assumptions.state_model is None:
        warnings.warn(
            "measure() defaulting to WAIVER_MODEL because "
            "waiver_incidence_annual is set or model_points.state has "
            "non-zero entries. Set assumptions.state_model explicitly "
            "(e.g. STATE_MODELS['WAIVER']) -- the implicit fallback is "
            "deprecated and will be removed in a future major version.",
            DeprecationWarning, stacklevel=2,
        )
        state_model = WAIVER_MODEL
    else:
        state_model = assumptions.state_model
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
        if assumptions.waiver_incidence_annual is not None:
            rate_dict["waiver_incidence"] = waiver
        if assumptions.ci_incidence_annual is not None:
            ci_inc = np.ascontiguousarray(annual_to_monthly(
                assumptions.ci_incidence_annual(
                    sex_grid, issue_age_grid, duration_grid)))
            rate_dict["ci_incidence"] = ci_inc
        if (assumptions.ci_reincidence_annual is not None
                or assumptions.disability_recovery_annual is not None):
            # Broadcast (n_mp, 1, 1) sex + (n_mp, 1, 1) age +
            # (1, n_year, 1) duration + (1, 1, max_cohort) cohort to
            # (n_mp, n_year, max_cohort). Duration-dependent rate
            # callables share the four-argument signature: the cohort
            # axis is months since entering the source state.
            sex_4d = model_points.sex.reshape(-1, 1, 1)
            age_4d = model_points.issue_age.reshape(-1, 1, 1)
            dur_4d = np.arange(n_years).reshape(1, -1, 1)
            coh_4d = np.arange(max_cohort).reshape(1, 1, -1)
            if assumptions.ci_reincidence_annual is not None:
                rate_dict["ci_reincidence"] = np.ascontiguousarray(
                    annual_to_monthly(
                        assumptions.ci_reincidence_annual(
                            sex_4d, age_4d, dur_4d, coh_4d)))
            if assumptions.disability_recovery_annual is not None:
                rate_dict["disability_recovery"] = np.ascontiguousarray(
                    annual_to_monthly(
                        assumptions.disability_recovery_annual(
                            sex_4d, age_4d, dur_4d, coh_4d)))
        (edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
         premium_state, benefit_state,
         state_duration_max) = compile_state_model_with_duration(
            state_model, rate_dict,
        )
        # compile_state_model_with_duration returns ``edge_prob`` shape
        # ``(n_edges, n_mp, n_year, max_D)`` -- already in the layout the
        # detailed kernel reads (edge axis outer, cohort axis inner).
        state_offset = np.zeros(n_states + 1, dtype=np.int64)
        state_offset[1:] = np.cumsum(state_duration_max)
        (inforce, deaths, premium_cf, claim_cf, morbidity_cf, expense_cf,
         annuity_cf, disability_cf,
         maturity_cf) = _project_kernel_semi_markov(
            mortality, edge_from, edge_to, edge_prob, edge_lump_sum,
            n_states, state_duration_max, state_offset,
            premium_state, benefit_state, start_state,
            model_points.term_months,
            model_points.count,
            model_points.level_premium,
            model_points.single_premium,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.cov_kind,
            model_points.cov_amount,
            model_points.cov_offset,
            model_points.cov_waiting,
            model_points.cov_reduction_end,
            model_points.cov_reduction_factor,
            cov_rates,
            cov_risk,
            cov_is_diagnosis,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            model_points.disability_income,
            model_points.disability_benefit,
            assumptions.expense_acquisition,
            maint_inflated_monthly,
            n_time,
        )
    else:
        (edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
         premium_state, benefit_state) = compile_state_model(
            state_model,
            {"mortality": mortality, "waiver_incidence": waiver,
             "lapse": lapse},
        )
        (inforce, deaths, premium_cf, claim_cf, morbidity_cf, expense_cf,
         annuity_cf, disability_cf, maturity_cf) = _project_kernel(
            mortality,
            edge_from,
            edge_to,
            edge_prob,
            edge_lump_sum,
            n_states,
            premium_state,
            benefit_state,
            start_state,
            model_points.term_months,
            model_points.count,
            model_points.level_premium,
            model_points.single_premium,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.cov_kind,
            model_points.cov_amount,
            model_points.cov_offset,
            model_points.cov_waiting,
            model_points.cov_reduction_end,
            model_points.cov_reduction_factor,
            cov_rates,
            cov_risk,
            cov_is_diagnosis,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            model_points.disability_income,
            model_points.disability_benefit,
            assumptions.expense_acquisition,
            maint_inflated_monthly,
            n_time,
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
    )
