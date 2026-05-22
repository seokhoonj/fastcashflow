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
    premium     : charged at the start of month t, on inforce[t]
                  (the single premium, if any, is added at t = 0)
    annuity     : paid at the start of month t, on inforce[t]
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
from fastcashflow.assumptions import Assumptions
from fastcashflow.coverage import coverage_arrays, coverage_rates
from fastcashflow.modelpoints import STATE_ACTIVE, ModelPoints


@dataclass(frozen=True, slots=True)
class Cashflows:
    """Projected cash flows.

    The per-month arrays are shaped ``(n_mp, n_time)``; ``maturity_cf`` is
    ``(n_mp,)`` -- one payment per policy, at that policy's term. Death and
    health claims are kept apart -- ``claim_cf`` and ``morbidity_cf`` -- so
    the Risk Adjustment can price the two risks separately.
    """

    inforce: FloatArray      # policies in force at the start of each month
    deaths: FloatArray       # deaths during each month
    premium_cf: FloatArray   # premium inflow per month (single premium at t=0)
    claim_cf: FloatArray     # death-benefit outflow per month (mortality risk)
    morbidity_cf: FloatArray # health-benefit outflow per month (morbidity risk)
    expense_cf: FloatArray   # expense outflow per month
    annuity_cf: FloatArray   # annuity (survival income) outflow per month
    maturity_cf: FloatArray  # (n_mp,) maturity benefit, paid at time = term

    @property
    def n_time(self) -> int:
        """Number of monthly projection steps."""
        return int(self.inforce.shape[1])


@njit(parallel=True, cache=True)
def _project_kernel(mortality, edge_from, edge_to, edge_prob, n_states,
                    premium_state, start_state, term_months, count,
                    monthly_premium, single_premium, premium_term_months,
                    cov_kind, cov_amount, cov_offset, cov_waiting,
                    cov_reduction_end, cov_reduction_factor, cov_rates,
                    cov_risk, cov_is_diagnosis, maturity_benefit,
                    annuity_payment, expense_acquisition, maint_monthly,
                    inflation, n_time):
    """Compiled, parallel time-loop kernel -- raw numpy arrays and scalars only.

    The model-point axis is the independent (outer) loop, run in parallel
    across cores; the time axis is the sequential (inner) loop, because the
    in-force recursion depends on the previous month.

    In-force is an occupancy vector over ``n_states`` transient states. Each
    month it is advanced along the transition edges: edge ``e`` carries
    ``edge_prob[e, mp, year]`` of the occupancy from state ``edge_from[e]``
    to ``edge_to[e]``. Premium accrues on the states flagged in
    ``premium_state``; claims, expenses and survival benefits on the total
    occupancy. The transition probabilities are composed by the caller, so
    the kernel itself is state-machine-agnostic.

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
    maturity_cf = np.zeros(n_mp)

    n_edges = edge_from.shape[0]
    for mp in prange(n_mp):
        term = term_months[mp]
        premium_term = premium_term_months[mp]   # months the premium is paid
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
            for s in range(n_states):
                ift += occ[s]
                if premium_state[s]:
                    prem_occ += occ[s]
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
            level = prem_occ * monthly_premium[mp] if t < premium_term else 0.0
            premium_cf[mp, t] = level + single
            claim_cf[mp, t] = ift * claim_rate
            morbidity_cf[mp, t] = ift * morb_rate
            annuity_cf[mp, t] = ift * annuity_payment[mp]
            acquisition = cnt * expense_acquisition if t == 0 else 0.0
            expense_cf[mp, t] = acquisition + ift * maint_monthly * inflation[t]
            # Advance the occupancy along the transition edges.
            for s in range(n_states):
                occ_next[s] = 0.0
            for e in range(n_edges):
                occ_next[edge_to[e]] += (occ[edge_from[e]]
                                         * edge_prob[e, mp, year])
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
            annuity_cf, maturity_cf)


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
    months = np.arange(n_time)
    durations = np.arange(n_years)

    sex_grid, _ = np.meshgrid(model_points.sex, durations, indexing="ij")
    issue_age_grid, duration_grid = np.meshgrid(
        model_points.issue_age, durations, indexing="ij"
    )
    mortality = np.ascontiguousarray(
        assumptions.mortality_monthly(sex_grid, issue_age_grid, duration_grid),
        dtype=np.float64,
    )
    if assumptions.waiver_inception_monthly is None:
        waiver = np.zeros_like(mortality)
    else:
        waiver = np.ascontiguousarray(
            assumptions.waiver_inception_monthly(
                sex_grid, issue_age_grid, duration_grid),
            dtype=np.float64,
        )
    lapse_by_year = np.ascontiguousarray(
        assumptions.lapse_monthly(durations), dtype=np.float64
    )
    cov_is_diagnosis, cov_risk = coverage_arrays(assumptions.riders)
    cov_rates = coverage_rates(
        mortality, [r.rate for r in assumptions.riders], sex_grid, issue_age_grid,
        duration_grid,
    )
    inflation = (1.0 + assumptions.expense_inflation) ** (months / 12.0)

    # Waiver model -- the in-force state machine. Two transient states
    # (0 active, 1 waiver); the monthly transition probabilities are composed
    # here, so the kernel runs a generic occupancy recursion. (a-2 is this
    # 2-state instance; phase (b) opens the state set up.)
    surv = 1.0 - mortality
    lapse_grid = lapse_by_year[None, :]
    edge_from = np.array([0, 0, 1], dtype=np.int64)
    edge_to = np.array([0, 1, 1], dtype=np.int64)
    edge_prob = np.ascontiguousarray(np.stack((
        surv * (1.0 - waiver) * (1.0 - lapse_grid),   # 0 -> 0  active stays
        surv * waiver,                                # 0 -> 1  active -> waiver
        surv,                                         # 1 -> 1  waiver stays
    )))
    n_states = 2
    premium_state = np.array([True, False])
    start_state = np.where(model_points.state == STATE_ACTIVE,
                           0, 1).astype(np.int64)

    (inforce, deaths, premium_cf, claim_cf, morbidity_cf, expense_cf,
     annuity_cf, maturity_cf) = _project_kernel(
        mortality,
        edge_from,
        edge_to,
        edge_prob,
        n_states,
        premium_state,
        start_state,
        model_points.term_months,
        model_points.count,
        model_points.monthly_premium,
        model_points.single_premium,
        model_points.premium_term_months,
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
        assumptions.expense_acquisition,
        assumptions.expense_maintenance_annual / 12.0,
        inflation,
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
        maturity_cf=maturity_cf,
    )
