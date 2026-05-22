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
from fastcashflow.gmm import (
    _norm_ppf,
    _rollforward_kernel,
    _settlement_factor,
    _settlement_lic,
    compute_csm,
    discount_factors,
    discount_factors_from_curve,
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


def _cost_of_capital_ra(cl_margin, monthly_rate, coc_rate):
    """Cost-of-capital RA -- the cost of holding the confidence-level margin
    as non-financial-risk capital over the contract's run-off.

    The capital required at each future month is taken as the confidence-
    level margin there; the RA at month ``t`` is the cost-of-capital rate
    times the present value, at ``t``, of that capital over months ``t``
    onward.
    """
    full = 1.0 / (1.0 + monthly_rate)
    cap_pv = np.empty_like(cl_margin)
    cap_pv[:, -1] = cl_margin[:, -1]
    for t in range(cl_margin.shape[1] - 2, -1, -1):
        cap_pv[:, t] = cl_margin[:, t] + full * cap_pv[:, t + 1]
    return coc_rate * cap_pv


def measure(model_points: ModelPoints, assumptions: Assumptions) -> Measurement:
    """Detailed GMM measurement: BEL, RA and CSM rolled forward over time."""
    proj = project_cashflows(model_points, assumptions)
    claim_cf, morbidity_cf = proj.claim_cf, proj.morbidity_cf
    if assumptions.settlement_pattern is None:
        lic = np.zeros((claim_cf.shape[0], proj.n_time + 1))
    else:
        lic = _settlement_lic(claim_cf + morbidity_cf, assumptions.settlement_pattern)
        # Claims are paid over the pattern, not at incurrence -- discount
        # them to their payment dates in the fulfilment cash flows.
        factor = _settlement_factor(assumptions.settlement_pattern, assumptions.discount_monthly)
        claim_cf = claim_cf * factor
        morbidity_cf = morbidity_cf * factor
    discount_start, discount_mid = discount_factors(assumptions, proj.n_time)

    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _rollforward_kernel(
        claim_cf, morbidity_cf, proj.disability_cf, proj.expense_cf,
        proj.premium_cf, proj.annuity_cf, proj.maturity_cf,
        model_points.term_months, assumptions.discount_monthly,
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
            cl_margin, assumptions.discount_monthly, assumptions.cost_of_capital_rate
        )
    else:
        raise ValueError(
            "ra_method must be 'confidence_level' or 'cost_of_capital', "
            f"got {assumptions.ra_method!r}"
        )
    csm, csm_accretion, csm_release, loss_component = compute_csm(
        bel[:, 0], ra[:, 0], proj, assumptions
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


@njit(parallel=True, cache=True)
def _value_kernel(edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
                  premium_state, benefit_state, start_state, issue_index, sex,
                  term_months, count, level_premium, single_premium,
                  premium_term_months, premium_frequency, annuity_frequency,
                  cov_kind, cov_amount, cov_offset, cov_rates, cov_risk,
                  cov_is_diagnosis, maturity_benefit, annuity_payment,
                  disability_income, disability_benefit,
                  expense_acquisition, maint_monthly, inflation,
                  discount_start, discount_mid, mortality_factor,
                  morbidity_factor, longevity_factor, disability_factor,
                  cov_waiting, cov_reduction_end, cov_reduction_factor):
    """Fused valuation kernel -- one parallel pass, no per-month arrays.

    Per model point the in-force is an occupancy vector over ``n_states``
    transient states, advanced each month along the transition edges -- edge
    ``e`` carries ``edge_prob[e, sex, age, year]`` of the occupancy from
    ``edge_from[e]`` to ``edge_to[e]``. Premium accrues on the states
    flagged in ``premium_state``; claims, expenses and survival benefits on
    the total occupancy. The present values are accumulated directly and
    BEL, RA, CSM and the loss component derived in the same pass. The RA
    sums a mortality-risk component (death claims), a morbidity-risk
    component (health claims), a disability-risk component (disability
    income and the lump sum) and a longevity-risk component (annuity and
    maturity benefits). The only memory written is the four ``(n_mp,)``
    result arrays, so the kernel is compute-bound and scales near-linearly.
    """
    n_mp = issue_index.shape[0]
    n_edges = edge_from.shape[0]
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
        occ = np.zeros(n_states)
        occ_next = np.zeros(n_states)
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
            pc += ift * claim_rate * dm
            pcm += ift * morb_rate * dm
            if t % ann_freq == 0:
                pa += ift * annuity * ds
            pd += benefit_occ * disability_income[mp] * dm
            acquisition = cnt * expense_acquisition if t == 0 else 0.0
            pe += (acquisition + ift * maint_monthly * inflation[t]) * dm
            # Advance the occupancy; a lump-sum transition pays on its flow.
            for s in range(n_states):
                occ_next[s] = 0.0
            for e in range(n_edges):
                flow = occ[edge_from[e]] * edge_prob[e, sx, ridx, year]
                occ_next[edge_to[e]] += flow
                if edge_lump_sum[e]:
                    pd += flow * disability_benefit[mp] * dm
            for s in range(n_states):
                occ[s] = occ_next[s]
        total = 0.0
        for s in range(n_states):
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
            for s in range(n_states):
                occ[s] = 0.0
            occ[ss] = cnt
            for t in range(term):
                year = t // 12
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    inf = 0.0
                    for s in range(n_states):
                        inf += occ[s]
                    contrib = (inf * cov_rates[kind, sx, ridx, year]
                               * benefit * mult * discount_mid[t])
                    if mortality_risk:
                        pc += contrib
                    else:
                        pcm += contrib
                for s in range(n_states):
                    occ_next[s] = 0.0
                for e in range(n_edges):
                    occ_next[edge_to[e]] += (occ[edge_from[e]]
                                             * edge_prob[e, sx, ridx, year])
                for s in range(n_states):
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
                # A waiting period suppresses the payment, not the diagnosis:
                # the not-yet-diagnosed pool depletes either way.
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    healthy = 0.0
                    for s in range(n_states):
                        healthy += occ[s]
                    pcm += healthy * d_rate * benefit * mult * discount_mid[t]
                undiag = 1.0 - d_rate
                for s in range(n_states):
                    occ_next[s] = 0.0
                for e in range(n_edges):
                    occ_next[edge_to[e]] += (occ[edge_from[e]] * undiag
                                             * edge_prob[e, sx, ridx, year])
                for s in range(n_states):
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
        projection month, a locked-in rate curve that replaces the flat
        ``assumptions.discount_annual``. ``None`` uses the flat rate.
    """
    if assumptions.ra_method != "confidence_level":
        raise ValueError(
            "value() computes the confidence-level RA only; use measure() "
            f"for ra_method={assumptions.ra_method!r}"
        )
    n_time = int(model_points.term_months.max())
    n_years = (n_time + 11) // 12
    months = np.arange(n_time)

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
    if assumptions.waiver_inception_annual is None:
        waiver_grid = np.zeros_like(mortality_grid)
    else:
        waiver_grid = np.ascontiguousarray(annual_to_monthly(
            assumptions.waiver_inception_annual(
                sex_grid, issue_age_grid, duration_grid)))
    issue_index = (model_points.issue_age - min_age).astype(np.int64)
    lapse_by_year = np.ascontiguousarray(annual_to_monthly(
        assumptions.lapse_annual(durations)))
    # In-force state machine -- the StateModel composes the transition edges
    # for the generic occupancy recursion (see fastcashflow.statemodel). The
    # rates are on the sex x age x duration grid the kernel indexes.
    state_model = assumptions.state_model or WAIVER_MODEL
    (edge_from, edge_to, edge_prob, edge_lump_sum, n_states, premium_state,
     benefit_state) = compile_state_model(
        state_model,
        {"mortality": mortality_grid, "waiver_inception": waiver_grid,
         "lapse": lapse_by_year[None, None, :]},
    )
    start_state = np.asarray(state_model.seating, np.int64)[model_points.state]
    cov_is_diagnosis, cov_risk = coverage_arrays(assumptions.riders)
    # coverage_rates stacks the annual mortality and rider rates; the whole
    # stack is converted to monthly. Slab 0 is the monthly mortality above.
    cov_rates = np.ascontiguousarray(annual_to_monthly(coverage_rates(
        mortality_annual_grid, [r.rate for r in assumptions.riders], sex_grid,
        issue_age_grid, duration_grid,
    )))

    inflation = (1.0 + assumptions.expense_inflation) ** (months / 12.0)
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

    args = (
        edge_from,
        edge_to,
        edge_prob,
        edge_lump_sum,
        n_states,
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
        assumptions.expense_maintenance_annual / 12.0,
        inflation,
        discount_start,
        discount_mid,
        mortality_factor,
        morbidity_factor,
        longevity_factor,
        disability_factor,
    )

    if backend == "cpu":
        bel, ra, csm, loss_component = _value_kernel(
            *args, model_points.cov_waiting, model_points.cov_reduction_end,
            model_points.cov_reduction_factor,
        )
    elif backend == "gpu":
        if np.any(model_points.cov_waiting) or np.any(model_points.cov_reduction_end):
            raise ValueError(
                "value(backend='gpu') does not support coverage waiting / "
                "reduction periods yet; use backend='cpu'"
            )
        from fastcashflow._gpu import value_gpu
        bel, ra, csm, loss_component = value_gpu(*args)
    else:
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")

    return Valuation(bel=bel, ra=ra, csm=csm, loss_component=loss_component)
