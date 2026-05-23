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
from fastcashflow.assumptions import Assumptions, annual_to_monthly
from fastcashflow.curves import (
    discount_factors,
    discount_factors_from_curve,
    discount_monthly_curve,
    inflation_index,
    maintenance_monthly_curve,
)
from fastcashflow.numerics import (
    _cost_of_capital_ra,
    _csm_kernel,
    _norm_ppf,
    _rollforward_kernel,
    _settlement_factor,
    _settlement_lic,
)
from fastcashflow.coverage import coverage_arrays, coverage_rates
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows
from fastcashflow.statemodel import WAIVER_MODEL, compile_state_model


# ---------------------------------------------------------------------------
# Detailed path
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Measurement:
    """Detailed measurement -- BEL, RA and CSM rolled forward over time.

    ``bel``, ``ra`` and ``csm`` are ``(n_mp, n_time+1)`` trajectories; column
    0 is the inception measurement. The CSM roll-forward decomposes as
    ``csm[:, t+1] = csm[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    ``lic`` is the liability for incurred claims -- zero unless a claims
    settlement pattern is set, which also discounts claims to their payment
    dates in the BEL.
    """

    bel: FloatArray              # (n_mp, n_time+1) -- BEL trajectory
    ra: FloatArray               # (n_mp, n_time+1) -- RA trajectory
    csm: FloatArray              # (n_mp, n_time+1) -- CSM trajectory
    csm_accretion: FloatArray    # (n_mp, n_time)   -- CSM interest accreted each month
    csm_release: FloatArray      # (n_mp, n_time)   -- CSM released each month
    loss_component: FloatArray   # (n_mp,)          -- loss component at inception
    lic: FloatArray              # (n_mp, n_time+1) -- liability for incurred claims
    cashflows: Cashflows
    discount_start: FloatArray   # (n_time,) -- start-of-month discount factors
    discount_mid: FloatArray     # (n_time,) -- mid-month discount factors


def _compute_csm(bel0, ra0, inforce, monthly_rate):
    """CSM at initial recognition (Sec. 38) and deterministic roll-forward (Sec. 44).

    Pure-array orchestration: fulfilment cash flows ``FCF = BEL + RA``,
    initial CSM = ``max(0, -FCF)``, loss component = ``max(0, FCF)``, then
    the CSM is rolled forward in :func:`_csm_kernel` (interest accretion at
    the locked-in monthly rate, release proportional to coverage units --
    in-force here).

    ``inforce`` is ``(n_mp, n_time)`` (the coverage-unit series), ``bel0`` /
    ``ra0`` are ``(n_mp,)``. Returns
    ``(csm, accretion, release, loss_component)``.
    """
    fcf = bel0 + ra0
    csm0 = np.maximum(0.0, -fcf)
    loss_component = np.maximum(0.0, fcf)
    csm, accretion, release = _csm_kernel(csm0, inforce, monthly_rate)
    return csm, accretion, release, loss_component


def measure(model_points: ModelPoints, assumptions: Assumptions) -> Measurement:
    """Detailed GMM measurement: BEL, RA and CSM rolled forward over time."""
    proj = project_cashflows(model_points, assumptions)
    claim_cf, morbidity_cf = proj.claim_cf, proj.morbidity_cf
    monthly_rate = discount_monthly_curve(assumptions, proj.n_time)
    if assumptions.settlement_pattern is None:
        lic = np.zeros((claim_cf.shape[0], proj.n_time + 1))
    else:
        lic = _settlement_lic(claim_cf + morbidity_cf, assumptions.settlement_pattern)
        # Claims are paid over the pattern, not at incurrence -- discount
        # them to their payment dates in the fulfilment cash flows. With a
        # discount curve we use the in-year scalar (Sec. 40 / B71 -- the
        # rate at the month of incurrence is the right reference); the
        # full-curve treatment would require a time-varying settlement
        # factor inside the kernel, deferred.
        factor = _settlement_factor(assumptions.settlement_pattern, assumptions.discount_monthly)
        claim_cf = claim_cf * factor
        morbidity_cf = morbidity_cf * factor
    discount_start, discount_mid = discount_factors_from_curve(monthly_rate)

    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _rollforward_kernel(
        claim_cf, morbidity_cf, proj.disability_cf, proj.expense_cf,
        proj.premium_cf, proj.annuity_cf, proj.maturity_cf,
        model_points.term_months, monthly_rate,
    )
    z = _norm_ppf(assumptions.ra_confidence)
    cl_margin = z * (assumptions.mortality_cv * pv_claims
                     + assumptions.morbidity_cv * pv_morbidity
                     + assumptions.disability_cv * pv_disability
                     + assumptions.longevity_cv * pv_survival)
    if assumptions.ra_method == "confidence_level":
        ra = cl_margin
    elif assumptions.ra_method == "cost_of_capital":
        ra = _cost_of_capital_ra(
            cl_margin, monthly_rate, assumptions.cost_of_capital_rate
        )
    else:
        raise ValueError(
            "ra_method must be 'confidence_level' or 'cost_of_capital', "
            f"got {assumptions.ra_method!r}"
        )
    csm, csm_accretion, csm_release, loss_component = _compute_csm(
        bel[:, 0], ra[:, 0], proj.inforce, monthly_rate,
    )

    return Measurement(
        bel=bel,
        ra=ra,
        csm=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        loss_component=loss_component,
        lic=lic,
        cashflows=proj,
        discount_start=discount_start,
        discount_mid=discount_mid,
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


def _make_value_kernel(N_STATES, N_EDGES):
    """Generate the fused valuation kernel for a specific (n_states, n_edges).

    The state count and edge count enter via the Python closure, not as
    runtime parameters. numba treats integer closure variables as
    compile-time constants, so the per-state and per-edge inner loops
    unroll, ``occ[s]`` lookups become register operations and the kernel
    runs roughly twice as fast as a generic n_states variant.

    Per model point the in-force is an occupancy vector over ``N_STATES``
    transient states, advanced each month along ``N_EDGES`` transition
    edges -- edge ``e`` carries ``edge_prob[sex, age, year, e]`` of the
    occupancy from ``edge_from[e]`` to ``edge_to[e]``. The edge axis is
    innermost so all edges for a given (sex, age, year) lookup land in one
    cache line. Premium accrues on the states flagged in ``premium_state``;
    claims, expenses and survival benefits on the total occupancy. The
    present values are accumulated directly and BEL, RA, CSM and the loss
    component derived in the same pass. The RA sums a mortality-risk
    component (death claims), a morbidity-risk component (health claims), a
    disability-risk component (disability income and the lump sum) and a
    longevity-risk component (annuity and maturity benefits). The only
    memory written is the four ``(n_mp,)`` result arrays, so the kernel is
    compute-bound and scales near-linearly.
    """
    @njit(parallel=True, cache=True)
    def kernel(edge_from, edge_to, edge_prob, edge_lump_sum,
               premium_state, benefit_state, start_state, issue_index, sex,
               term_months, count, level_premium, single_premium,
               premium_term_months, premium_frequency, annuity_frequency,
               cov_kind, cov_amount, cov_offset, cov_rates, cov_risk,
               cov_is_diagnosis, maturity_benefit, annuity_payment,
               disability_income, disability_benefit,
               expense_acquisition, maint_inflated_monthly,
               discount_start, discount_mid, mortality_factor,
               morbidity_factor, longevity_factor, disability_factor,
               cov_waiting, cov_reduction_end, cov_reduction_factor):
        n_mp = issue_index.shape[0]
        bel = np.empty(n_mp)
        ra = np.empty(n_mp)
        csm = np.empty(n_mp)
        loss_component = np.empty(n_mp)

        for mp in prange(n_mp):
            term = term_months[mp]
            premium_term = premium_term_months[mp]   # months the premium is paid
            prem_freq = premium_frequency[mp]        # months between premiums
            ann_freq = annuity_frequency[mp]         # months between annuity payouts
            ridx = issue_index[mp]
            sx = sex[mp]
            cnt = count[mp]
            premium = level_premium[mp]
            annuity = annuity_payment[mp]
            c_start = cov_offset[mp]
            c_end = cov_offset[mp + 1]
            ss = start_state[mp]
            occ = np.zeros(N_STATES)
            occ_next = np.zeros(N_STATES)
            pc = 0.0          # PV of death claims (mortality risk)
            pcm = 0.0         # PV of health claims (morbidity risk)
            pd = 0.0          # PV of disability income + lump sum
            pp = 0.0
            pe = 0.0
            pa = 0.0
            # Main pass -- the rule-free, non-diagnosis coverages.
            occ[ss] = cnt
            last_year = -1
            claim_rate = 0.0  # aggregate mortality claim per unit in-force
            morb_rate = 0.0   # aggregate morbidity claim per unit in-force
            # Counters in place of the per-timestep modulo / less-than checks.
            prem_due = 0
            ann_due = 0
            prem_left = premium_term
            for t in range(term):
                year = t // 12
                # Coverage rates change only once a year, so the per-coverage sum
                # is rebuilt on a year boundary, not every month.
                if year != last_year:
                    claim_rate = 0.0
                    morb_rate = 0.0
                    for k in range(c_start, c_end):
                        kind = cov_kind[k]
                        if cov_is_diagnosis[kind]:
                            continue          # diagnosis coverages run separately
                        if cov_waiting[k] != 0 or cov_reduction_end[k] != 0:
                            continue          # rule-bearing coverages run separately
                        rate = cov_rates[kind, sx, ridx, year] * cov_amount[k]
                        if cov_risk[kind] == 0:
                            claim_rate += rate
                        else:
                            morb_rate += rate
                    last_year = year
                ift = 0.0          # total in-force
                prem_occ = 0.0     # in-force on the premium-paying states
                benefit_occ = 0.0  # in-force on the benefit-paying states
                for s in range(N_STATES):
                    ift += occ[s]
                    if premium_state[s]:
                        prem_occ += occ[s]
                    if benefit_state[s]:
                        benefit_occ += occ[s]
                ds = discount_start[t]
                dm = discount_mid[t]
                single = prem_occ * single_premium[mp] if t == 0 else 0.0
                if prem_due == 0 and prem_left > 0:
                    level = prem_occ * premium
                    prem_due = prem_freq - 1
                else:
                    level = 0.0
                    prem_due -= 1
                prem_left -= 1
                pp += (level + single) * ds
                pc += ift * claim_rate * dm
                pcm += ift * morb_rate * dm
                if ann_due == 0:
                    pa += ift * annuity * ds
                    ann_due = ann_freq - 1
                else:
                    ann_due -= 1
                pd += benefit_occ * disability_income[mp] * dm
                acquisition = cnt * expense_acquisition if t == 0 else 0.0
                pe += (acquisition + ift * maint_inflated_monthly[t]) * dm
                # Advance the occupancy; a lump-sum transition pays on its flow.
                for s in range(N_STATES):
                    occ_next[s] = 0.0
                for e in range(N_EDGES):
                    flow = occ[edge_from[e]] * edge_prob[sx, ridx, year, e]
                    occ_next[edge_to[e]] += flow
                    if edge_lump_sum[e]:
                        pd += flow * disability_benefit[mp] * dm
                for s in range(N_STATES):
                    occ[s] = occ_next[s]
            total = 0.0
            for s in range(N_STATES):
                total += occ[s]
            pm = total * maturity_benefit[mp] * discount_start[term]
            # Non-diagnosis coverages with a waiting or reduced-benefit rule: a
            # per-coverage pass, re-deriving the occupancy month by month so the
            # benefit multiplier (which can change mid-year) applies cleanly.
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
                for s in range(N_STATES):
                    occ[s] = 0.0
                occ[ss] = cnt
                for t in range(term):
                    year = t // 12
                    if t >= wait:
                        mult = red_factor if t < red_end else 1.0
                        inf = 0.0
                        for s in range(N_STATES):
                            inf += occ[s]
                        contrib = (inf * cov_rates[kind, sx, ridx, year]
                                   * benefit * mult * discount_mid[t])
                        if mortality_risk:
                            pc += contrib
                        else:
                            pcm += contrib
                    for s in range(N_STATES):
                        occ_next[s] = 0.0
                    for e in range(N_EDGES):
                        occ_next[edge_to[e]] += (occ[edge_from[e]]
                                                 * edge_prob[sx, ridx, year, e])
                    for s in range(N_STATES):
                        occ[s] = occ_next[s]
            # Diagnosis coverages pay once on first diagnosis, so each one's
            # claims run off a depleting "not yet diagnosed" occupancy -- a
            # separate pass over the time axis, into the morbidity PV.
            for k in range(c_start, c_end):
                kind = cov_kind[k]
                if not cov_is_diagnosis[kind]:
                    continue
                benefit = cov_amount[k]
                wait = cov_waiting[k]
                red_end = cov_reduction_end[k]
                red_factor = cov_reduction_factor[k]
                # The not-yet-diagnosed occupancy -- depletes by the diagnosis
                # rate on top of the ordinary transitions.
                for s in range(N_STATES):
                    occ[s] = 0.0
                occ[ss] = cnt
                d_year = -1
                d_rate = 0.0
                for t in range(term):
                    year = t // 12
                    if year != d_year:
                        d_rate = cov_rates[kind, sx, ridx, year]
                        d_year = year
                    # A waiting period suppresses the payment, not the diagnosis:
                    # the not-yet-diagnosed pool depletes either way.
                    if t >= wait:
                        mult = red_factor if t < red_end else 1.0
                        healthy = 0.0
                        for s in range(N_STATES):
                            healthy += occ[s]
                        pcm += healthy * d_rate * benefit * mult * discount_mid[t]
                    undiag = 1.0 - d_rate
                    for s in range(N_STATES):
                        occ_next[s] = 0.0
                    for e in range(N_EDGES):
                        occ_next[edge_to[e]] += (occ[edge_from[e]] * undiag
                                                 * edge_prob[sx, ridx, year, e])
                    for s in range(N_STATES):
                        occ[s] = occ_next[s]
            bel_mp = pc + pcm + pd + pm + pa + pe - pp
            ra_mp = (mortality_factor * pc + morbidity_factor * pcm
                     + disability_factor * pd + longevity_factor * (pm + pa))
            fcf = bel_mp + ra_mp
            bel[mp] = bel_mp
            ra[mp] = ra_mp
            csm[mp] = max(0.0, -fcf)
            loss_component[mp] = max(0.0, fcf)

        return bel, ra, csm, loss_component

    return kernel


_VALUE_KERNEL_CACHE: dict[tuple[int, int], object] = {}


def _get_value_kernel(n_states: int, n_edges: int):
    """Return the value kernel specialised for ``(n_states, n_edges)``.

    First call for a new ``(N, E)`` pair triggers a numba compile (a few
    seconds); subsequent calls reuse the cached function. With ``cache=True``
    on the inner kernel, numba also persists the compile to disk so a fresh
    Python process pays no compile cost on repeat runs.
    """
    key = (int(n_states), int(n_edges))
    cached = _VALUE_KERNEL_CACHE.get(key)
    if cached is None:
        cached = _make_value_kernel(*key)
        _VALUE_KERNEL_CACHE[key] = cached
    return cached


@njit(parallel=True, cache=True)
def _value_kernel_scalar(issue_index, sex, term_months, count, level_premium,
                         single_premium, premium_term_months, premium_frequency,
                         annuity_frequency, cov_kind, cov_amount, cov_offset,
                         cov_rates, cov_risk, cov_is_diagnosis,
                         maturity_benefit, annuity_payment, expense_acquisition,
                         maint_inflated_monthly, discount_start, discount_mid,
                         mortality_factor, morbidity_factor, longevity_factor,
                         cov_waiting, cov_reduction_end, cov_reduction_factor,
                         survival_monthly):
    """Scalar-inforce fast path of :func:`_value_kernel`.

    Used when the in-force projection collapses to a single survival track --
    no user-supplied StateModel, no waiver inception, every model point
    seated in the active state. The in-force is carried as a scalar; the
    monthly decay is one multiply against the precomputed
    ``survival_monthly[sex, age, year] = (1 - q_monthly) * (1 - l_monthly)``
    table. Numerically identical to ``_value_kernel`` for this configuration
    -- the disability income, disability lump-sum and benefit-state pieces
    of the general kernel evaluate to zero here -- and recovers the
    pre-Phase(b) speed (see ``docs/tutorial/13-why-fast.md``).
    """
    n_mp = issue_index.shape[0]
    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    loss_component = np.empty(n_mp)

    for mp in prange(n_mp):
        term = term_months[mp]
        premium_term = premium_term_months[mp]
        prem_freq = premium_frequency[mp]
        ann_freq = annuity_frequency[mp]
        ridx = issue_index[mp]
        sx = sex[mp]
        cnt = count[mp]
        premium = level_premium[mp]
        annuity = annuity_payment[mp]
        c_start = cov_offset[mp]
        c_end = cov_offset[mp + 1]
        inforce = cnt
        pc = 0.0          # PV of death claims (mortality risk)
        pcm = 0.0         # PV of health claims (morbidity risk)
        pp = 0.0
        pe = 0.0
        pa = 0.0
        last_year = -1
        claim_rate = 0.0
        morb_rate = 0.0
        # Counters replace modulo / less-than checks in the inner loop --
        # ``prem_due`` ticks down to the next premium-paying month,
        # ``ann_due`` to the next annuity month, and ``prem_left`` to the
        # end of the premium-paying term. Profiling shows the modulo /
        # comparison form costs ~2/3 of the inner-loop time at large
        # portfolios -- the counter form lets the compiler keep the loop
        # branch-light and 1M MP runs in ~50 ms again.
        prem_due = 0
        ann_due = 0
        prem_left = premium_term
        for t in range(term):
            year = t // 12
            if year != last_year:
                claim_rate = 0.0
                morb_rate = 0.0
                for k in range(c_start, c_end):
                    kind = cov_kind[k]
                    if cov_is_diagnosis[kind]:
                        continue
                    if cov_waiting[k] != 0 or cov_reduction_end[k] != 0:
                        continue
                    rate = cov_rates[kind, sx, ridx, year] * cov_amount[k]
                    if cov_risk[kind] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year
            ds = discount_start[t]
            dm = discount_mid[t]
            single = inforce * single_premium[mp] if t == 0 else 0.0
            if prem_due == 0 and prem_left > 0:
                level = inforce * premium
                prem_due = prem_freq - 1
            else:
                level = 0.0
                prem_due -= 1
            prem_left -= 1
            pp += (level + single) * ds
            pc += inforce * claim_rate * dm
            pcm += inforce * morb_rate * dm
            if ann_due == 0:
                pa += inforce * annuity * ds
                ann_due = ann_freq - 1
            else:
                ann_due -= 1
            acquisition = cnt * expense_acquisition if t == 0 else 0.0
            pe += (acquisition + inforce * maint_inflated_monthly[t]) * dm
            inforce *= survival_monthly[sx, ridx, year]
        pm = inforce * maturity_benefit[mp] * discount_start[term]
        # Non-diagnosis coverages with a waiting or reduced-benefit rule:
        # rerun the survival on the same scalar track so the benefit
        # multiplier (which can change mid-year) applies cleanly.
        for k in range(c_start, c_end):
            kind = cov_kind[k]
            if cov_is_diagnosis[kind]:
                continue
            wait = cov_waiting[k]
            red_end = cov_reduction_end[k]
            if wait == 0 and red_end == 0:
                continue
            benefit = cov_amount[k]
            red_factor = cov_reduction_factor[k]
            mortality_risk = cov_risk[kind] == 0
            inf = cnt
            for t in range(term):
                year = t // 12
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    contrib = (inf * cov_rates[kind, sx, ridx, year]
                               * benefit * mult * discount_mid[t])
                    if mortality_risk:
                        pc += contrib
                    else:
                        pcm += contrib
                inf *= survival_monthly[sx, ridx, year]
        # Diagnosis coverages: claims run off a depleting "not yet diagnosed"
        # pool, which depletes both by survival and by the diagnosis rate.
        for k in range(c_start, c_end):
            kind = cov_kind[k]
            if not cov_is_diagnosis[kind]:
                continue
            benefit = cov_amount[k]
            wait = cov_waiting[k]
            red_end = cov_reduction_end[k]
            red_factor = cov_reduction_factor[k]
            healthy = cnt
            d_year = -1
            d_rate = 0.0
            for t in range(term):
                year = t // 12
                if year != d_year:
                    d_rate = cov_rates[kind, sx, ridx, year]
                    d_year = year
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    pcm += healthy * d_rate * benefit * mult * discount_mid[t]
                healthy *= survival_monthly[sx, ridx, year] * (1.0 - d_rate)
        bel_mp = pc + pcm + pm + pa + pe - pp
        ra_mp = (mortality_factor * pc + morbidity_factor * pcm
                 + longevity_factor * (pm + pa))
        fcf = bel_mp + ra_mp
        bel[mp] = bel_mp
        ra[mp] = ra_mp
        csm[mp] = max(0.0, -fcf)
        loss_component[mp] = max(0.0, fcf)

    return bel, ra, csm, loss_component


def value(
    model_points: ModelPoints,
    assumptions: Assumptions,
    *,
    backend: str = "cpu",
    discount_curve: FloatArray | None = None,
) -> Valuation:
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
    discount_curve :
        Optional ``(n_time,)`` array of annual discount rates -- one per
        projection month, a power-user override for the stochastic case
        where the rate must vary month by month. ``None`` (the default)
        uses the scalar or per-year curve on
        ``assumptions.discount_annual``; when supplied, it overrides that
        and bypasses the curves layer for the discount step.
    """
    if assumptions.ra_method != "confidence_level":
        raise ValueError(
            "value() computes the confidence-level RA only; use measure() "
            f"for ra_method={assumptions.ra_method!r}"
        )
    n_time = int(model_points.term_months.max())
    n_years = (n_time + 11) // 12

    # Mortality and lapse are evaluated on a dense sex x [min, max] issue-age
    # x duration grid. Using the age range rather than the exact distinct
    # ages avoids an O(n log n) sort (np.unique): min/max and the index
    # subtraction are O(n), and the few unused ages cost nothing -- the
    # assumption grid is tiny.
    min_age = int(model_points.issue_age.min())
    max_age = int(model_points.issue_age.max())
    durations = np.arange(n_years)
    sex_grid, issue_age_grid, duration_grid = np.meshgrid(
        np.array([0, 1]), np.arange(min_age, max_age + 1), durations,
        indexing="ij",
    )
    # Rates are supplied annual; the engine converts each to a monthly rate
    # on the constant-force basis (see assumptions.annual_to_monthly).
    mortality_annual_grid = assumptions.mortality_annual(
        sex_grid, issue_age_grid, duration_grid)
    mortality_grid = np.ascontiguousarray(annual_to_monthly(mortality_annual_grid))
    issue_index = (model_points.issue_age - min_age).astype(np.int64)
    lapse_grid = np.ascontiguousarray(annual_to_monthly(
        assumptions.lapse_annual(sex_grid, issue_age_grid, duration_grid)))
    # Fast path: when no waiver / paid-up mechanic is active and every model
    # point is seated in the active state, the in-force is a single survival
    # track. The scalar kernel carries it as one number and runs the
    # pre-Phase(b) speed path; the N-state kernel is reserved for products
    # that genuinely need an occupancy vector.
    fast_path = (backend == "cpu"
                 and assumptions.state_model is None
                 and assumptions.waiver_inception_annual is None
                 and not np.any(model_points.state))
    if not fast_path:
        if assumptions.waiver_inception_annual is None:
            waiver_grid = np.zeros_like(mortality_grid)
        else:
            waiver_grid = np.ascontiguousarray(annual_to_monthly(
                assumptions.waiver_inception_annual(
                    sex_grid, issue_age_grid, duration_grid)))
        # In-force state machine -- the StateModel composes the transition
        # edges for the generic occupancy recursion (see
        # fastcashflow.statemodel). The rates are on the sex x age x duration
        # grid the kernel indexes.
        state_model = assumptions.state_model or WAIVER_MODEL
        (edge_from, edge_to, edge_prob, edge_lump_sum, n_states, premium_state,
         benefit_state) = compile_state_model(
            state_model,
            {"mortality": mortality_grid, "waiver_inception": waiver_grid,
             "lapse": lapse_grid},
        )
        # compile_state_model returns ``edge_prob`` with the edge axis first
        # -- (n_edges, sex, age, year). Transpose so the edge axis is
        # innermost: all edges for a given (sex, age, year) lookup land in
        # one cache line, ~25% faster on the multi-state hot path.
        edge_prob = np.ascontiguousarray(np.transpose(edge_prob, (1, 2, 3, 0)))
        start_state = np.asarray(state_model.seating, np.int64)[model_points.state]
    cov_is_diagnosis, cov_risk = coverage_arrays(assumptions.riders)
    # coverage_rates stacks the annual mortality and rider rates; the whole
    # stack is converted to monthly. Slab 0 is the monthly mortality above.
    cov_rates = np.ascontiguousarray(annual_to_monthly(coverage_rates(
        mortality_annual_grid, [r.rate for r in assumptions.riders], sex_grid,
        issue_age_grid, duration_grid,
    )))

    maint_inflated_monthly = (maintenance_monthly_curve(assumptions, n_time)
                              * inflation_index(assumptions, n_time))
    if discount_curve is None:
        discount_start, discount_mid = discount_factors(assumptions, n_time)
    else:
        discount_curve = np.asarray(discount_curve, dtype=np.float64)
        if discount_curve.shape != (n_time,):
            raise ValueError(
                f"discount_curve must have shape ({n_time},) -- one annual "
                f"rate per projection month -- got {discount_curve.shape}"
            )
        monthly_curve = (1.0 + discount_curve) ** (1.0 / 12.0) - 1.0
        discount_start, discount_mid = discount_factors_from_curve(monthly_curve)
    z = _norm_ppf(assumptions.ra_confidence)
    mortality_factor = z * assumptions.mortality_cv
    morbidity_factor = z * assumptions.morbidity_cv
    longevity_factor = z * assumptions.longevity_cv
    disability_factor = z * assumptions.disability_cv

    # A claims settlement pattern discounts claims to their payment dates;
    # scaling the coverage amounts carries that into the fused kernel.
    cov_amount = model_points.cov_amount
    if assumptions.settlement_pattern is not None:
        cov_amount = cov_amount * _settlement_factor(
            assumptions.settlement_pattern, assumptions.discount_monthly
        )

    if fast_path:
        survival_monthly = np.ascontiguousarray(
            (1.0 - mortality_grid) * (1.0 - lapse_grid)
        )
        bel, ra, csm, loss_component = _value_kernel_scalar(
            issue_index,
            model_points.sex,
            model_points.term_months,
            model_points.count,
            model_points.level_premium,
            model_points.single_premium,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.cov_kind,
            cov_amount,
            model_points.cov_offset,
            cov_rates,
            cov_risk,
            cov_is_diagnosis,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            assumptions.expense_acquisition,
            maint_inflated_monthly,
            discount_start,
            discount_mid,
            mortality_factor,
            morbidity_factor,
            longevity_factor,
            model_points.cov_waiting,
            model_points.cov_reduction_end,
            model_points.cov_reduction_factor,
            survival_monthly,
        )
        return Valuation(bel=bel, ra=ra, csm=csm, loss_component=loss_component)

    # The CPU kernel takes (n_states, n_edges) via Python closure -- they are
    # not in the args tuple. The GPU kernel still takes n_states explicitly
    # as a runtime arg, so the two paths build the call list separately.
    n_edges = int(edge_from.shape[0])
    common_args = (
        edge_from,
        edge_to,
        edge_prob,
        edge_lump_sum,
        premium_state,
        benefit_state,
        start_state,
        issue_index,
        model_points.sex,
        model_points.term_months,
        model_points.count,
        model_points.level_premium,
        model_points.single_premium,
        model_points.premium_term_months,
        model_points.premium_frequency_months,
        model_points.annuity_frequency_months,
        model_points.cov_kind,
        cov_amount,
        model_points.cov_offset,
        cov_rates,
        cov_risk,
        cov_is_diagnosis,
        model_points.maturity_benefit,
        model_points.annuity_payment,
        model_points.disability_income,
        model_points.disability_benefit,
        assumptions.expense_acquisition,
        maint_inflated_monthly,
        discount_start,
        discount_mid,
        mortality_factor,
        morbidity_factor,
        longevity_factor,
        disability_factor,
    )

    if backend == "cpu":
        kernel = _get_value_kernel(n_states, n_edges)
        bel, ra, csm, loss_component = kernel(
            *common_args, model_points.cov_waiting, model_points.cov_reduction_end,
            model_points.cov_reduction_factor,
        )
    elif backend == "gpu":
        if np.any(model_points.cov_waiting) or np.any(model_points.cov_reduction_end):
            raise ValueError(
                "value(backend='gpu') does not support coverage waiting / "
                "reduction periods yet; use backend='cpu'"
            )
        from fastcashflow._gpu import value_gpu
        bel, ra, csm, loss_component = value_gpu(
            common_args[0], common_args[1], common_args[2], common_args[3],
            n_states, *common_args[4:],
        )
    else:
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")

    return Valuation(bel=bel, ra=ra, csm=csm, loss_component=loss_component)


def value_segmented(
    model_points: ModelPoints,
    basis: dict[tuple[str, str], Assumptions],
    **kwargs,
) -> Valuation:
    """Value a multi-segment portfolio: split, value each, concatenate.

    ``basis`` is the ``{(product, channel): Assumptions}`` dictionary
    returned by :func:`fastcashflow.read_assumptions`. ``model_points``
    must carry ``product`` and ``channel`` columns identifying each row's
    segment; for each unique (product, channel) the helper masks the
    matching rows, builds a sub-:class:`~fastcashflow.ModelPoints` via
    :meth:`~fastcashflow.ModelPoints.subset`, calls :func:`value` with the
    segment's ``Assumptions``, and writes the per-row results back to a
    single ``(n_mp,)`` :class:`Valuation`.

    Extra keyword arguments (``backend``, ``discount_curve``) flow through
    to :func:`value`. A single-segment ``basis`` is accepted as a
    convenience when ``product`` / ``channel`` is not set.
    """
    if model_points.product is None or model_points.channel is None:
        if len(basis) == 1:
            (assumptions,) = basis.values()
            return value(model_points, assumptions, **kwargs)
        raise ValueError(
            "model_points has no 'product'/'channel' set but the basis has "
            f"{len(basis)} segments; either set the columns or pass a "
            "single-segment basis"
        )

    product = model_points.product
    channel = model_points.channel
    n_mp = model_points.n_mp

    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    loss_component = np.empty(n_mp)

    # Run each unique (product, channel) once -- order is stable on the
    # first-seen index, so debugging output reads top-to-bottom of the input.
    seen: set[tuple] = set()
    keys: list[tuple] = []
    for p, c in zip(product, channel):
        key = (str(p), str(c))
        if key not in seen:
            seen.add(key)
            keys.append(key)

    for key in keys:
        if key not in basis:
            raise ValueError(
                f"segment {key!r} appears in model_points but is not in the "
                f"basis (known segments: {sorted(basis)})"
            )
        mask = np.fromiter(
            ((str(p), str(c)) == key for p, c in zip(product, channel)),
            dtype=bool, count=n_mp,
        )
        idx = np.nonzero(mask)[0]
        sub = model_points.subset(idx)
        val = value(sub, basis[key], **kwargs)
        bel[idx] = val.bel
        ra[idx] = val.ra
        csm[idx] = val.csm
        loss_component[idx] = val.loss_component

    return Valuation(bel=bel, ra=ra, csm=csm, loss_component=loss_component)
