"""Monthly cash flow projection -- the BaseProj layer.

Sign convention (liability perspective, used consistently across the engine):

    premium_cf  : insurer INFLOW  -- reduces the insurance liability
    mortality_cf    : insurer OUTFLOW -- DEATH-pattern claims (priced via mortality_cv)
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

from fastcashflow._typing import BoolArray, FloatArray, IntArray
from fastcashflow.basis import (
    Basis, annual_to_monthly, derive_expense_components, validate_factor,
    SURRENDER_VALUE_BASES,
)
from fastcashflow.coverage import (
    align_coverages, build_coverage_rates, coverage_arrays, validate_csr_codes,
)
from fastcashflow.curves import inflation_index
from fastcashflow.model_points import ModelPoints
from fastcashflow.multistate import (
    compile_model,
    compile_model_with_duration,
    is_semi_markov,
    model_references_rate,
    resolve_model,
)

# Public surface of the ``fastcashflow.projection`` namespace: the raw
# cash-flow projection entry point and its two result types. A non-IFRS17
# user (pricing, ALM, experience study) projects the flows here and values
# them against a curve from ``fastcashflow.curves``, without the measurement
# layer (BEL / RA / CSM).
__all__ = ["project_cashflows", "Cashflows", "AccountTrajectory"]


@dataclass(frozen=True, slots=True)
class AccountTrajectory:
    """Universal-life account-value diagnostics -- a nested sidecar on
    :class:`Cashflows`, populated only when the portfolio carries an
    account-referencing coverage (``has_account``).

    The benefits the account drives stay in-band on :class:`Cashflows`
    (``mortality_cf`` / ``surrender_cf`` / ``maturity_cf``) so the BEL measurement
    inherits them with no new parameter; this object localises the
    account-state arrays that the flat ``(n_mp, n_time)`` stitch loops would
    otherwise trip on (``av`` / ``fund`` carry the extra month-end column).
    A single nested field is invisible to those flat-FloatArray loops -- they
    enumerate the known flat fields and skip it.

    ``nar`` is NOT stored; it is recomputed as ``max(0, face - av_mid)`` where
    the net amount at risk is needed (the RA on the death leg).
    """

    av: FloatArray      # (n_mp, n_time+1) account value at month start (col 0 = av0)
    av_mid: FloatArray  # (n_mp, n_time)   half-month-credited AV (death / lapse + NAR base)
    coi: FloatArray     # (n_mp, n_time)   cost-of-insurance charged (diagnostic)
    fund: FloatArray    # (n_mp, n_time+1) inforce-weighted AV held = inforce_pad * av
    # Per-policy account charge flows (n_mp, n_time) for the UL entity net-liability
    # decomposition (the asset-liability gap): premium credited net of load, admin
    # fee drawn, cost-deducting rider charge drawn. None on books without the roll.
    prem_to_av: FloatArray | None = None
    admin_charge: FloatArray | None = None
    account_charge: FloatArray | None = None
    # (n_mp, n_time) per-policy account surrender value, max(0, av_mid *
    # (1 - surr_charge_rate)) -- the figure a surrender exit is paid, and what
    # ``_measurement.inforce.inforce_surrender_value`` reads for the mass-lapse surrender strain.
    surr_value: FloatArray | None = None


@dataclass(frozen=True, slots=True)
class StateTrace:
    """Compiled state-machine handles for the opt-in per-state reserve.

    Attached to :class:`Cashflows` only when ``project_cashflows(...,
    emit_state=True)`` is asked for it (the per-state reserve path). It carries
    the exact compiled edge list the projection ran, so a caller can replay the
    per-state occupancy and value each state without recompiling the model or
    re-evaluating rates (which would risk diverging from the projection). Markov
    only -- the semi-Markov path leaves ``Cashflows.state_trace`` ``None``.

    ``edge_prob`` is ``(n_edges, n_mp, n_year)`` -- the transition probability of
    each edge, constant within a policy year. ``state_pays_premium`` /
    ``state_pays_benefit`` / ``death_benefit_factor`` are the ``(n_states,)`` per-state
    flags and the death-benefit multiplier the projection weighted occupancy by.
    ``has_premium_term_move`` flags a deterministic at-premium-term transition
    (active -> paid-up): the edge list alone does not carry it, so an edge-only
    occupancy replay would misplace occupancy -- the per-state reserve rejects it.

    ``state_death_exit`` / ``state_lapse`` are the ``(n_states, n_mp, n_year)``
    per-state absorbing-exit probabilities (death, lapse) -- NOT edges (they leave
    the in-force set), so the per-state sum-at-risk synthesizes a death / lapse
    transition for each state whose exit probability is non-zero. ``state_names``
    labels the transient states for the transition descriptors. ``edge_lump_sum``
    ``(n_edges,)`` flags the inter-state edges that pay a lump on transition (the
    ``ModelPoints.disability_benefit`` amount). ``death_face`` ``(n_mp,)`` is the
    per-unit level death benefit (sum of the plain death-risk coverage amounts);
    ``has_death_coverage_rules`` flags a rule-bearing death coverage (waiting /
    reduction / step / escalation / term) whose benefit varies in time -- the
    sum-at-risk (which assumes a level death benefit) rejects it in v1.
    """
    edge_from: IntArray
    edge_to: IntArray
    edge_prob: FloatArray
    edge_lump_sum: BoolArray
    n_states: int
    start_state: IntArray
    count: FloatArray
    state_pays_premium: BoolArray
    state_pays_benefit: BoolArray
    death_benefit_factor: FloatArray
    has_premium_term_move: bool
    state_death_exit: FloatArray
    state_lapse: FloatArray
    state_names: tuple[str, ...]
    death_face: FloatArray
    has_death_coverage_rules: bool


@dataclass(frozen=True, slots=True)
class Cashflows:
    """Projected cash flows.

    The per-month arrays are shaped ``(n_mp, n_time)``; ``maturity_cf`` is
    ``(n_mp,)`` -- one payment per policy, at that policy's term. Death and
    health claims are kept apart -- ``mortality_cf`` and ``morbidity_cf`` -- so
    the Risk Adjustment can price the two risks separately; ``disability_cf``
    is a third risk class -- the income paid while in a benefit state plus
    the on-transition lump sum.

    Field-name convention (applies across the engine): a ``_cf`` suffix marks a
    per-month money flow (the signed cash flow in or out that month); a
    ``_path`` suffix marks a running stock / balance trajectory (``lrc_path`` /
    ``csm_path`` / ``ra_path`` on the measure results); no suffix marks a count
    (``inforce`` / ``deaths`` -- policies / lives, not money). The scalar an
    in-force loop carries for the current month is spelled ``inforce_t``.
    """

    inforce: FloatArray       # policies in force at the start of each month
    deaths: FloatArray        # deaths during each month
    premium_cf: FloatArray    # premium inflow per month (single premium at t=0)
    mortality_cf: FloatArray  # DEATH-pattern claim outflow per month (priced via mortality_cv)
    morbidity_cf: FloatArray  # MORBIDITY-pattern claim outflow per month (priced via morbidity_cv)
    expense_cf: FloatArray    # expense outflow per month
    annuity_cf: FloatArray    # annuity (survival income) outflow per month
    disability_cf: FloatArray # disability income + lump-sum outflow per month
    maturity_cf: FloatArray   # (n_mp,) maturity benefit, paid at time = term
    maturity_survivors: FloatArray  # (n_mp,) in-force reaching term (the maturity exit count)
    surrender_cf: FloatArray  # surrender value paid on lapse
    # Universal-life account diagnostics -- None for every non-account
    # portfolio (the flat-array stitch loops skip the nested object).
    account: "AccountTrajectory | None" = None
    # The guaranteed (certain) part of annuity_cf -- the payments inside a
    # certain-and-life guarantee window, paid regardless of survival. None when
    # no model point carries a guarantee period. Used to remove the certain
    # payments from the longevity Risk Adjustment (they bear no longevity risk).
    annuity_certain_cf: "FloatArray | None" = None
    # Compiled state-machine handles for the opt-in per-state reserve
    # (project_cashflows(..., emit_state=True)); None on every ordinary
    # projection and on the semi-Markov path.
    state_trace: "StateTrace | None" = None

    @property
    def n_time(self) -> int:
        """Number of monthly projection steps."""
        return int(self.inforce.shape[1])


def reject_account_book(cashflows: "Cashflows | None", entry: str) -> None:
    """Raise if a projection carries a universal-life account (the Step-3.5 gate).

    Measure paths that read the benefit cash flows (``mortality_cf`` /
    ``surrender_cf`` / ``maturity_cf``) RAW -- without netting the account
    ``fund`` or splitting the net amount at risk -- would double-count a
    universal-life account book (the account benefit is the policyholder's own
    money, not a priced claim). Until each such path grows account support,
    reject it rather than mis-measure. Measure the contract through
    :func:`fastcashflow.gmm.measure` / :func:`fastcashflow.vfa.measure` instead.
    """
    if cashflows is not None and cashflows.account is not None:
        raise NotImplementedError(
            f"{entry} does not yet support a universal-life account book -- the "
            "account is not netted on this path, so the benefit cash flows would "
            "be double-counted. Measure the contract through gmm.measure / "
            "vfa.measure instead.")


def _expense_kernel_args(
    basis: Basis, n_time: int,
) -> tuple[float, float, float, FloatArray, FloatArray, float, float]:
    """Return the seven expense primitives the kernels take.

    Projects ``Basis.expense_items`` onto the kernel-side inputs,
    threading ``Basis.expense_inflation`` through the recurring
    rows via :func:`fastcashflow.curves.inflation_index`:

    - ``acquisition_premium``, ``acquisition_per_policy``, ``maintenance_premium``
      -- scalars used at ``t=0`` (acquisition) and every premium-paying month
      (maintenance on premium).
    - ``maintenance_per_policy`` -- ``(n_time,)`` per-policy monthly maintenance
      amount (with global inflation baked in).
    - ``lae`` -- ``(n_time,)`` Loss-Adjustment-Expense fraction applied each
      month to ``(claim + morbidity + disability)`` (with global
      inflation baked in).
    - ``maintenance_surrender_value`` / ``maintenance_face`` -- scalar annual
      rates charged each in-force month on the in-force surrender value / the
      policy's sum assured. Both bases are built in :func:`project_cashflows`
      post-projection (they need the in-force path), so these scalars are the
      only kernel-side inputs.

    An empty ``expense_items`` produces seven zeros -- the no-expense
    basis -- so the kernel can run unchanged.
    """
    return derive_expense_components(
        basis.expense_items, n_time, inflation_index(basis, n_time),
    )


def _account_kernel_args(
    model_points, basis, coverage_rates, coverage_funds_from_account,
    coverage_pays_account_balance, maintenance_per_policy, n_time, n_years,
):
    """Build the per-policy universal-life account-roll inputs for the kernel.

    Returns ``(has_account, mp_account, account_value0, account_face,
    prem_to_av, coi_rate_m, admin_fee, credit, account_charge)`` -- per-policy
    scalars / arrays the kernel rolls the account value with. ``coi_rate_m`` is
    the NAR-priced death-leg rate; ``account_charge`` is the level per-month
    charge of every cost-deducting rider (funds from the account, fixed
    benefit). The roll is NOT in-force weighted
    (it tracks a single policy's account); in-force enters at the fund / benefit
    aggregation. ``has_account`` is derived STRICTLY from the coverage flags
    (NEVER ``account_value != 0``).

    The COI rate the account is charged is the monthly rate of the
    ``funds_from_account`` coverage each MP carries (its ``rate_table`` is
    ``coi_annual``); the admin fee deducted from the account is the per-policy
    monthly ``maintenance_per_policy`` (NOT in-force weighted); crediting is the declared
    ``investment_return`` floored at each contract's ``minimum_crediting_rate``.
    """
    from fastcashflow._measurement.tvog import credited_monthly_rate

    n_mp = model_points.issue_age.shape[0]
    # Per-MP gate: an MP carries the account roll iff one of its coverages is
    # account-referencing (funds_from_account or pays_account_balance). A term
    # row in a mixed book stays untouched.
    account_cov = coverage_funds_from_account | coverage_pays_account_balance
    mp_account = np.zeros(n_mp, np.bool_)
    # The death-leg COI is the net-amount-at-risk charge of the account-backed
    # death coverage -- the one that BOTH funds from the account AND pays the
    # account balance (max(av, face)). A cost-deducting rider funds from the
    # account but does NOT pay the balance; it charges a fixed amount, summed
    # into ``account_charge`` below, never into this NAR-priced rate.
    coi_cov_of_mp = np.full(n_mp, -1, np.int64)  # the death-leg coverage index per MP
    if model_points.coverage_index.size and account_cov.any():
        cov_idx = model_points.coverage_index
        offset = model_points.coverage_offset
        for mp in range(n_mp):
            for k in range(offset[mp], offset[mp + 1]):
                ci = cov_idx[k]
                if account_cov[ci]:
                    mp_account[mp] = True
                if (coverage_funds_from_account[ci]
                        and coverage_pays_account_balance[ci]):
                    coi_cov_of_mp[mp] = ci
    has_account = bool(mp_account.any())
    if not has_account:
        # Strict no-op: a 1-wide stub for every kernel-required array; the
        # kernel never reads them (has_account=False short-circuits the roll).
        z1 = np.zeros((1, 1))
        return (False, mp_account, np.zeros(1), np.zeros(1),
                z1, z1, np.zeros(1), np.zeros(1), z1, z1)

    year_of_month = np.arange(n_time) // 12

    # Per-policy premium credited to the account, net of the load. premium_cf
    # in the kernel stays GROSS (the BEL inflow); only the load-net amount
    # builds the account.
    if basis.premium_factor_annual is None:
        premium_factor = np.ones((n_mp, n_years))
    else:
        sex_grid, _ = np.meshgrid(model_points.sex, np.arange(n_years),
                                  indexing="ij")
        issue_age_grid, duration_grid = np.meshgrid(
            model_points.issue_age, np.arange(n_years), indexing="ij")
        issue_class_grid, _ = np.meshgrid(model_points.issue_class,
                                          np.arange(n_years), indexing="ij")
        elapsed_grid = np.zeros_like(duration_grid)
        premium_factor = validate_factor(
            basis.premium_factor_annual(
                sex_grid, issue_age_grid, duration_grid,
                issue_class_grid, elapsed_grid),
            "premium_factor_annual", (n_mp, n_years))
    pf_m = premium_factor[:, year_of_month]               # (n_mp, n_time)
    t_idx = np.arange(n_time)[None, :]
    premium_term = model_points.premium_term_months[:, None]
    prem_freq = model_points.premium_frequency_months[:, None]
    paying = (t_idx < premium_term) & (t_idx % prem_freq == 0)
    prem_to_av = np.ascontiguousarray(
        model_points.premium[:, None] * pf_m * paying * (1.0 - basis.premium_load))

    # COI monthly charge rate per MP, year-expanded to (n_mp, n_time). The rate
    # is the account-backed death leg's own monthly rate (funds AND pays); an MP
    # with no such leg charges zero NAR-COI (e.g. a savings-only account whose
    # only account charge is a cost-deducting rider).
    coi_rate_m = np.zeros((n_mp, n_time))
    for mp in range(n_mp):
        ci = coi_cov_of_mp[mp]
        if ci >= 0:
            coi_rate_m[mp] = coverage_rates[ci, mp, year_of_month]
    coi_rate_m = np.ascontiguousarray(coi_rate_m)

    # Fixed per-month account charge of every cost-deducting rider (funds from
    # the account, does NOT pay the balance). Each such coverage draws
    # ``rate * amount`` from the account; its benefit is paid normally on the
    # claim side (it is NOT excluded from the claim accumulators, only the
    # pays_account_balance death leg is). The charge is NOT net-amount-at-risk
    # priced -- the benefit is a fixed sum, not the account -- so it is summed
    # here into a level charge, separate from the NAR-priced ``coi_rate_m``.
    account_charge = np.zeros((n_mp, n_time))
    if model_points.coverage_index.size:
        cov_idx = model_points.coverage_index
        cov_amt = model_points.coverage_amount
        offset = model_points.coverage_offset
        for mp in range(n_mp):
            for k in range(offset[mp], offset[mp + 1]):
                ci = cov_idx[k]
                if (coverage_funds_from_account[ci]
                        and not coverage_pays_account_balance[ci]):
                    account_charge[mp] += (
                        coverage_rates[ci, mp, year_of_month] * cov_amt[k])
    account_charge = np.ascontiguousarray(account_charge)

    # Admin fee = per-policy monthly maintenance_per_policy (NOT inforce_t-weighted). Crediting
    # = declared return floored at each contract's guarantee.
    admin_fee = np.ascontiguousarray(np.asarray(maintenance_per_policy, np.float64))
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    credit = np.ascontiguousarray(
        credited_monthly_rate(r_m, model_points.minimum_crediting_rate))
    account_value0 = np.ascontiguousarray(
        np.asarray(model_points.account_value, np.float64))
    account_face = np.ascontiguousarray(
        np.asarray(model_points.minimum_death_benefit, np.float64))

    # Surrender charge rate per policy year, year-expanded to (n_mp, n_time). The
    # account surrender value is av_mid * (1 - surr_charge_rate); a fraction in
    # [0, 1], NOT a decrement (never annual_to_monthly). None -> 0 (full account
    # value paid), so an account book with no charge stays bit-identical.
    if basis.surrender_charge_annual is None:
        surr_charge_rate = np.zeros((n_mp, n_time))
    else:
        sex_grid, _ = np.meshgrid(model_points.sex, np.arange(n_years),
                                  indexing="ij")
        issue_age_grid, duration_grid = np.meshgrid(
            model_points.issue_age, np.arange(n_years), indexing="ij")
        issue_class_grid, _ = np.meshgrid(model_points.issue_class,
                                          np.arange(n_years), indexing="ij")
        elapsed_grid = np.zeros_like(duration_grid)
        sc_year = validate_factor(
            basis.surrender_charge_annual(
                sex_grid, issue_age_grid, duration_grid,
                issue_class_grid, elapsed_grid),
            "surrender_charge_annual", (n_mp, n_years))
        surr_charge_rate = sc_year[:, year_of_month]          # (n_mp, n_time)
    surr_charge_rate = np.ascontiguousarray(surr_charge_rate)
    return (True, mp_account, account_value0, account_face,
            prem_to_av, coi_rate_m, admin_fee, credit, account_charge,
            surr_charge_rate)


@njit(cache=True)
def _benefit_factor(t, year, red_factor, red_end, step_month, step_factor,
                    esc, cap):
    """The per-coverage benefit multiplier at month ``t`` (policy ``year``).

    Three independent, composable shapes, all neutral by default:
    reduction (factor < 1 before ``red_end``), annual escalation (the
    benefit compounds at ``esc`` per year, capped at ``cap``
    x base when ``cap > 0``), and a single step-up (factor from
    ``step_month`` on). The reduction term is the historical expression
    verbatim, so a contract with no escalation / step is bit-identical.
    """
    m = red_factor if t < red_end else 1.0
    if esc != 0.0:
        e = (1.0 + esc) ** year
        if cap > 0.0 and e > cap:
            e = cap
        m *= e
    if step_month != 0 and t >= step_month:
        m *= step_factor
    return m


@njit(parallel=True, cache=True)
def _project_kernel(state_death_exit, state_lapse, state_death_benefit_factor,
                    state_premium_term_to,
                    edge_from, edge_to, edge_prob, edge_lump_sum,
                    n_states, state_pays_premium, state_pays_benefit, start_state,
                    term_months, contract_boundary_months, count, premium, premium_factor, annuity_factor,
                    premium_term_months, premium_frequency_months, annuity_frequency_months,
                    coverage_index, coverage_amount, coverage_offset, coverage_waiting,
                    coverage_reduction_end, coverage_reduction_factor,
    coverage_step_month, coverage_step_factor,
    coverage_escalation_annual, coverage_escalation_cap, coverage_term, coverage_rates,
                    coverage_risk, coverage_is_diagnosis,
                    coverage_pays_account_balance, maturity_benefit,
                    annuity_payment, disability_income, disability_benefit,
                    acquisition_premium, acquisition_per_policy, maintenance_premium,
                    maintenance_per_policy, lae,
                    has_account, mp_account, account_value0, account_face,
                    account_prem_to_av, account_coi_rate, account_admin_fee,
                    account_credit, account_charge,
                    annuitization_months, annuitization_rate,
                    annuity_air_monthly, min_accumulation_benefit,
                    annuity_start_months, annuity_term_months,
                    annuity_guarantee_months, n_time):
    """Compiled, parallel time-loop kernel -- raw numpy arrays and scalars only.

    The model-point axis is the independent (outer) loop, run in parallel
    across cores; the time axis is the sequential (inner) loop, because the
    in-force recursion depends on the previous month.

    In-force is an occupancy vector over ``n_states`` transient states. Each
    month it is advanced along the transition edges: edge ``e`` carries
    ``edge_prob[e, mp, year]`` of the occupancy from state ``edge_from[e]``
    to ``edge_to[e]``. Premium accrues on the states flagged in
    ``state_pays_premium``; claims, expenses and survival benefits on the total
    occupancy; disability income on the ``state_pays_benefit`` occupancy, and a
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
    n_mp = state_death_exit.shape[1]
    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))
    premium_cf = np.zeros((n_mp, n_time))
    mortality_cf = np.zeros((n_mp, n_time))
    morbidity_cf = np.zeros((n_mp, n_time))
    expense_cf = np.zeros((n_mp, n_time))
    annuity_cf = np.zeros((n_mp, n_time))
    annuity_certain_cf = np.zeros((n_mp, n_time))  # the guaranteed (certain) part
    disability_cf = np.zeros((n_mp, n_time))
    lapse_flow = np.zeros((n_mp, n_time))   # state-machine lapse exits, for surrender
    maturity_cf = np.zeros(n_mp)
    maturity_survivors = np.zeros(n_mp)
    # Universal-life account-value trajectory. Sized densely only when the
    # portfolio carries an account coverage; otherwise a 1-wide stub the
    # caller discards (numba needs a concrete array either branch).
    av_rows = n_mp if has_account else 1
    av = np.zeros((av_rows, n_time + 1))
    av_mid = np.zeros((av_rows, n_time))
    coi_av = np.zeros((av_rows, n_time))
    # Per-policy account charge flows emitted alongside coi_av for the UL entity
    # net-liability decomposition (the asset-liability gap): the premium credited
    # to the account (net of load), the admin fee drawn, and the cost-deducting
    # rider charge drawn. (n_mp, n_time) like coi_av; in-force weighting is applied
    # downstream. COI is coi_av; credited interest and the account-value pass-through
    # net against the held fund and are not emitted.
    prem_to_av_out = np.zeros((av_rows, n_time))
    admin_out = np.zeros((av_rows, n_time))
    account_charge_out = np.zeros((av_rows, n_time))

    n_edges = edge_from.shape[0]
    for mp in prange(n_mp):
        term = term_months[mp]
        boundary = contract_boundary_months[mp]  # paragraph 34 horizon (<= term)
        premium_term = premium_term_months[mp]   # months the premium is paid
        prem_freq = premium_frequency_months[mp]        # months between premiums
        ann_freq = annuity_frequency_months[mp]         # months between annuity payouts
        # Plain-annuity payout schedule (annuitizing books reject these upstream).
        ann_start = annuity_start_months[mp]    # deferred payout start (0 = inception)
        ann_term = annuity_term_months[mp]      # term-certain payout count (0 = life)
        ann_guar = annuity_guarantee_months[mp]  # certain-and-life window (0 = pure life)
        cnt = count[mp]
        c_start = coverage_offset[mp]
        c_end = coverage_offset[mp + 1]
        # Per-policy universal-life account-value roll (only for MPs carrying an
        # account-referencing coverage). The roll is a single policy's account;
        # in-force weighting enters at the fund / benefit aggregation, not here.
        # Within-month order is verbatim from the standalone UL kernel:
        #   a += prem_to_av; coi_nar = max(0, face - a); coi = coi_rate * coi_nar;
        #   a -= admin_fee + coi; if a < 0: a = 0;
        #   av_mid = a*(1+cr)^0.5; a = a*(1+cr).
        roll_av = has_account and mp_account[mp]
        A_annz = annuitization_months[mp]
        # An MP annuitizes iff it is account-backed and carries a conversion
        # month: at A_annz the account accumulation stops and the balance buys a
        # survival-contingent income (phase 2). A_annz == 0 -> the ordinary
        # account (a maturity lump), every existing book byte-identical.
        annuitizing = roll_av and A_annz > 0
        # Variable payout: a finite per-MP AIR re-floats the phase-2 income by
        # (1+fund)/(1+air) each elapsed month; NaN keeps a fixed GAO payout. The
        # fund rate is the account's own credited rate.
        air_m = annuity_air_monthly[mp]
        # Variable iff the AIR is a finite rate (NaN = fixed). isfinite, not
        # "not isnan", so a stray inf cannot reach the re-float (validation
        # rejects inf upstream; this is the matching kernel-side guard).
        variable_payout = annuitizing and np.isfinite(air_m)
        converted_balance = 0.0
        locked_annuity_payment = 0.0
        if roll_av:
            a_av = account_value0[mp]
            av[mp, 0] = a_av
            face_av = account_face[mp]
            cr_av = account_credit[mp]
            # Mid-month (death / lapse) credit is GEOMETRIC half: (1+cr)^0.5, the
            # unique h with h*h = (1+cr) so two half-months compose to the full
            # month, and consistent with the geometric mid-month discount
            # (1+r)^-(t+0.5). A deliberate, validated choice -- do NOT "simplify"
            # it to the linear 1 + 0.5*cr (which overshoots a full month by
            # 0.25*cr^2 and is not self-consistent).
            half_credit = (1.0 + cr_av) ** 0.5
            full_credit = 1.0 + cr_av
            # Accumulation stops at the conversion month for an annuitizing MP
            # (the balance is spent to buy the annuity); otherwise it rolls to
            # the projection horizon as before.
            roll_end = A_annz if annuitizing else boundary
            for t in range(roll_end):
                a_av += account_prem_to_av[mp, t]
                coi_nar = face_av - a_av   # net amount at risk (face above the account)
                if coi_nar < 0.0:
                    coi_nar = 0.0
                c_av = account_coi_rate[mp, t] * coi_nar
                coi_av[mp, t] = c_av
                prem_to_av_out[mp, t] = account_prem_to_av[mp, t]
                admin_out[mp, t] = account_admin_fee[t]
                account_charge_out[mp, t] = account_charge[mp, t]
                # The death-leg NAR-COI plus every cost-deducting rider's fixed
                # charge are both drawn from the account this month.
                a_av -= account_admin_fee[t] + c_av + account_charge[mp, t]
                if a_av < 0.0:
                    a_av = 0.0
                av_mid[mp, t] = a_av * half_credit
                a_av = a_av * full_credit
                av[mp, t + 1] = a_av
            if annuitizing:
                # Conversion: the balance carried into month A (the month-A
                # start = month A-1 end, no month-A credit), floored at the GMAB
                # (minimum_accumulation_benefit). The GAO rate locks the level
                # payment once -- a guaranteed annuity, not re-credited.
                converted_balance = av[mp, A_annz]
                gmab_acc = min_accumulation_benefit[mp]
                if gmab_acc > converted_balance:
                    converted_balance = gmab_acc
                locked_annuity_payment = converted_balance * annuitization_rate[mp]
        # In-force occupancy over the transient states; the input state
        # seats the model point's count on its starting state.
        occ = np.zeros(n_states)
        occ_next = np.zeros(n_states)
        occ[start_state[mp]] = cnt
        last_year = -1
        claim_rate = 0.0      # aggregate mortality claim per unit in-force
        morb_rate = 0.0       # aggregate morbidity claim per unit in-force
        cnt0_annuity = 0.0    # in-force at the payout-start month (guarantee base)
        for t in range(boundary):
            year = t // 12
            in_payout = annuitizing and t >= A_annz
            inforce_t = 0.0   # total in-force
            dclaim_occ = 0.0  # death-benefit-weighted in-force (claim base)
            prem_occ = 0.0    # in-force on the premium-paying states
            benefit_occ = 0.0 # in-force on the benefit-paying states
            deaths_acc = 0.0  # state-conditional death count
            lapse_acc = 0.0   # state-conditional lapse count (surrender)
            for s in range(n_states):
                inforce_t += occ[s]
                dclaim_occ += occ[s] * state_death_benefit_factor[s]
                deaths_acc += occ[s] * state_death_exit[s, mp, year]
                lapse_acc += occ[s] * state_lapse[s, mp, year]
                if state_pays_premium[s]:
                    prem_occ += occ[s]
                if state_pays_benefit[s]:
                    benefit_occ += occ[s]
            inforce[mp, t] = inforce_t
            # No surrender in the payout phase (a life annuity in payment cannot
            # lapse); phase 1 reports the actual state-machine lapse exit.
            lapse_flow[mp, t] = 0.0 if in_payout else lapse_acc
            if year != last_year:
                claim_rate = 0.0
                morb_rate = 0.0
                for k in range(c_start, c_end):
                    cov_idx = coverage_index[k]
                    if coverage_is_diagnosis[cov_idx]:
                        continue          # diagnosis coverages run separately
                    if coverage_pays_account_balance[cov_idx]:
                        continue          # account-backed death pays from the AV
                    if (coverage_waiting[k] != 0 or coverage_reduction_end[k] != 0
                            or coverage_step_month[k] != 0
                            or coverage_escalation_annual[k] != 0.0
                            or coverage_term[k] != 0):
                        continue          # rule-bearing coverages run separately
                    rate = coverage_rates[cov_idx, mp, year] * coverage_amount[k]
                    if coverage_risk[cov_idx] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year
            deaths[mp, t] = deaths_acc
            level = (prem_occ * premium[mp] * premium_factor[mp, year]
                     if (t < premium_term and t % prem_freq == 0) else 0.0)
            premium_cf[mp, t] = level
            mortality_cf[mp, t] = dclaim_occ * claim_rate
            if roll_av and not in_payout:
                # Account-backed death pays max(account value, face) on the
                # occupancy deaths -- written ONCE here, the pays_account_balance
                # coverage having been excluded from claim_rate above. The
                # account portion returns the policyholder's own money; the face
                # tops it up where it exceeds the account (the net amount at
                # risk). Added to any non-account death claim already accrued.
                av_m = av_mid[mp, t]
                fa = account_face[mp]
                mortality_cf[mp, t] += deaths_acc * (av_m if av_m > fa else fa)
            morbidity_cf[mp, t] = inforce_t * morb_rate
            # Capture the in-force at the payout-start month -- the guaranteed
            # payments are paid on this count regardless of later survival.
            if (not annuitizing) and t == ann_start:
                cnt0_annuity = inforce_t
            if annuitizing:
                # Phase 2 (t >= A): the guaranteed annuity, paid annuity-due from
                # the conversion month on the surviving in-force, every
                # annuity_frequency_months. Phase 1 pays nothing here. A variable
                # payout re-floats the level by ((1+fund)/(1+air))^k,
                # k = t - A (the annuity-unit value); a fixed payout keeps the
                # locked level (variable_payout False -> the multiplier is 1).
                if in_payout and (t - A_annz) % ann_freq == 0:
                    pay = locked_annuity_payment
                    if variable_payout:
                        pay *= ((1.0 + account_credit[mp])
                                / (1.0 + air_m)) ** (t - A_annz)
                    annuity_cf[mp, t] = inforce_t * pay
                else:
                    annuity_cf[mp, t] = 0.0
            else:
                # Plain annuity with the deferred-start / term-certain /
                # guaranteed-period schedule. k = months since the payout start;
                # all-zero fields reduce to today's level-from-inception income.
                if t < ann_start:
                    annuity_cf[mp, t] = 0.0                  # before the start
                else:
                    k = t - ann_start
                    if ann_term > 0 and k >= ann_term:
                        annuity_cf[mp, t] = 0.0              # term-certain exhausted
                    elif k % ann_freq == 0:
                        base = annuity_payment[mp] * annuity_factor[mp, year]
                        if k < ann_guar:
                            # Guarantee window: certain payment on the payout-start
                            # survivor count, regardless of later survival. Tracked
                            # separately so the longevity RA can exclude it.
                            certain = cnt0_annuity * base
                            annuity_cf[mp, t] = certain
                            annuity_certain_cf[mp, t] = certain
                        else:
                            annuity_cf[mp, t] = inforce_t * base
                    else:
                        annuity_cf[mp, t] = 0.0
            disability_cf[mp, t] = benefit_occ * disability_income[mp]
            # Expense: alpha / beta / gamma maintenance plus LAE on the
            # month's claim + morbidity total. Dispatched from
            # Basis.expense_items by basis (acquisition_premium /
            # acquisition_per_policy / maintenance_premium / maintenance_per_policy / lae).
            ann_prem = premium[mp] * premium_factor[mp, year] * 12.0 / prem_freq
            acquisition_expense = (
                cnt * (acquisition_premium * ann_prem + acquisition_per_policy)
                if t == 0 else 0.0)
            maintenance_premium_expense = (
                inforce_t * maintenance_premium[t] * ann_prem / 12.0
                if t < premium_term else 0.0)
            maintenance_per_policy_expense = inforce_t * maintenance_per_policy[t]
            # LAE applies to claim + morbidity claims only --
            # disability income is a periodic annuity-like benefit, lump
            # sums are one-off transitions, and conflating either with
            # LAE would double-count. Add a dedicated basis later if the
            # practice ever needs it.
            lae_expense = lae[t] * (
                mortality_cf[mp, t] + morbidity_cf[mp, t])
            expense_cf[mp, t] = (acquisition_expense + maintenance_premium_expense
                                 + maintenance_per_policy_expense + lae_expense)
            # Advance the occupancy along the transition edges; a lump-sum
            # transition pays its benefit on the occupancy it carries.
            for s in range(n_states):
                occ_next[s] = 0.0
            if in_payout:
                # Phase 2 occupancy: mortality-only decrement (survivors stay,
                # deaths leave). Lapse and inter-state transitions are suppressed
                # -- a life annuity in payment does not lapse. Single active
                # state in v1; a multi-state book freezes transitions here.
                for s in range(n_states):
                    occ_next[s] = occ[s] * (1.0 - state_death_exit[s, mp, year])
            else:
                for e in range(n_edges):
                    flow = occ[edge_from[e]] * edge_prob[e, mp, year]
                    occ_next[edge_to[e]] += flow
                    if edge_lump_sum[e]:
                        disability_cf[mp, t] += flow * disability_benefit[mp]
            # Maturity lump: skipped for an annuitizing MP (the balance was
            # already converted to the annuity at A; paying a lump too would
            # double-count).
            if t + 1 == term and not annuitizing:
                total_next = 0.0
                for s in range(n_states):
                    total_next += occ_next[s]
                if roll_av:
                    # Account maturity: survivors * max(matured av, GMAB). The
                    # matured account value is the month-end balance at the term
                    # (av[mp, term] = av[mp, t+1]); account_face carries no GMAB,
                    # so maturity_benefit doubles as the guaranteed accumulation
                    # benefit floor here.
                    av_term = av[mp, t + 1]
                    gmab = maturity_benefit[mp]
                    maturity_cf[mp] = total_next * (
                        av_term if av_term > gmab else gmab)
                else:
                    maturity_cf[mp] = total_next * maturity_benefit[mp]
                maturity_survivors[mp] = total_next
            # Calendar-keyed deterministic transition: when the premium-paying
            # period ends (entering month ``premium_term``), the source state's
            # occupancy moves prob-1 to its at_premium_term destination -- the
            # active -> paid-up relabel. Per-MP timing (premium_term varies by
            # model point); applied to the start-of-next-month occupancy so the
            # destination state holds it from month ``premium_term`` onward.
            # dest -2 leaves the in-force set (to=None cover-end at premium_term).
            # The move reads a SNAPSHOT of the pre-move occupancy (reusing ``occ``
            # as scratch -- it is overwritten just below) so it is independent of
            # state order and safe against chained destinations: each state moves
            # only its own original occupancy, never occupancy it just received.
            if premium_term > 0 and t + 1 == premium_term:
                for s in range(n_states):
                    occ[s] = occ_next[s]            # snapshot of pre-move occupancy
                for s in range(n_states):
                    dto = state_premium_term_to[s]
                    if dto == -1 or dto == s:
                        continue                    # no transition / self -> stay
                    occ_next[s] -= occ[s]           # remove this state's own occupancy
                    if dto >= 0:
                        occ_next[dto] += occ[s]      # ... add it to the destination
                    # dto == -2: leaves the in-force set (not added anywhere)
            for s in range(n_states):
                occ[s] = occ_next[s]

        # Non-diagnosis coverages carrying a waiting or reduced-benefit rule
        # run per month here, not in the year-aggregated rate above, because
        # the benefit multiplier can change partway through a year.
        for k in range(c_start, c_end):
            cov_idx = coverage_index[k]
            if coverage_is_diagnosis[cov_idx]:
                continue
            if coverage_pays_account_balance[cov_idx]:
                continue          # account-backed death pays from the AV (above)
            wait = coverage_waiting[k]
            red_end = coverage_reduction_end[k]
            step_month = coverage_step_month[k]
            step_factor = coverage_step_factor[k]
            esc = coverage_escalation_annual[k]
            cap = coverage_escalation_cap[k]
            if (wait == 0 and red_end == 0 and step_month == 0 and esc == 0.0
                    and coverage_term[k] == 0):
                continue          # rule-free -- already in the aggregate
            benefit = coverage_amount[k]
            red_factor = coverage_reduction_factor[k]
            mortality_risk = coverage_risk[cov_idx] == 0
            cov_end = boundary if coverage_term[k] == 0 else min(boundary, coverage_term[k])
            for t in range(wait, cov_end):
                mult = _benefit_factor(t, t // 12, red_factor, red_end,
                                       step_month, step_factor, esc, cap)
                amt = (inforce[mp, t] * coverage_rates[cov_idx, mp, t // 12]
                       * benefit * mult)
                if mortality_risk:
                    mortality_cf[mp, t] += amt
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
            step_month = coverage_step_month[k]
            step_factor = coverage_step_factor[k]
            esc = coverage_escalation_annual[k]
            cap = coverage_escalation_cap[k]
            red_factor = coverage_reduction_factor[k]
            undiagnosed = 1.0   # fraction of the in-force still undiagnosed
            d_year = -1
            d_rate = 0.0
            cov_end = boundary if coverage_term[k] == 0 else min(boundary, coverage_term[k])
            for t in range(cov_end):
                year = t // 12
                if year != d_year:
                    d_rate = coverage_rates[cov_idx, mp, year]
                    d_year = year
                # A waiting period suppresses the payment, not the diagnosis:
                # the not-yet-diagnosed pool depletes either way.
                if t >= wait:
                    mult = _benefit_factor(t, t // 12, red_factor, red_end,
                                       step_month, step_factor, esc, cap)
                    morbidity_cf[mp, t] += (inforce[mp, t] * undiagnosed
                                            * d_rate * benefit * mult)
                undiagnosed *= (1.0 - d_rate)

    return (inforce, deaths, premium_cf, mortality_cf, morbidity_cf, expense_cf,
            annuity_cf, annuity_certain_cf, disability_cf, lapse_flow,
            maturity_cf, maturity_survivors,
            av, av_mid, coi_av,
            prem_to_av_out, admin_out, account_charge_out)


@njit(parallel=True, cache=True)
def _project_kernel_semi_markov(
    state_death_exit, state_lapse, state_death_benefit_factor,
    state_det_at, state_det_to, state_det_lump,
    edge_from, edge_to, edge_prob, edge_lump_sum,
    n_states, state_duration_max, state_offset, periodic_benefit_term_months,
    state_pays_premium, state_pays_benefit, start_state,
    term_months, contract_boundary_months, count, premium, premium_factor, annuity_factor,
    premium_term_months, premium_frequency_months, annuity_frequency_months,
    coverage_index, coverage_amount, coverage_offset, coverage_waiting,
    coverage_reduction_end, coverage_reduction_factor,
    coverage_step_month, coverage_step_factor,
    coverage_escalation_annual, coverage_escalation_cap, coverage_term, coverage_rates,
    coverage_risk, coverage_is_diagnosis,
    maturity_benefit, annuity_payment, disability_income, disability_benefit,
    acquisition_premium, acquisition_per_policy, maintenance_premium,
    maintenance_per_policy, lae, n_time,
):
    """Detailed semi-Markov projection -- main pass only.

    Cohort-aware analogue of :func:`_project_kernel`. State ``s`` has
    ``state_duration_max[s] = D`` monthly cohorts, indexed via the flat
    occupancy vector at ``state_offset[s] + tau`` for ``tau in 0..D-1``.
    Transitions whose rate is sojourn_dependent carry per-cohort
    probabilities through ``edge_prob``'s trailing axis (shape
    ``(n_edges, n_mp, n_year, max_D)``).

    The residual stay edge (``edge_from == edge_to``) advances each cohort
    to ``tau + 1``, with the last cohort absorbing the long tail. A
    transient transition enters the destination state's cohort 0.

    Coverage-rule and diagnosis-coverage passes are not emitted -- the
    semi-Markov prototype rejects model points carrying either, so the
    main pass alone is the full projection for the supported cases.
    """
    n_mp = state_death_exit.shape[1]
    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))
    premium_cf = np.zeros((n_mp, n_time))
    mortality_cf = np.zeros((n_mp, n_time))
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
        boundary = contract_boundary_months[mp]  # paragraph 34 horizon (<= term)
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
                    if (coverage_waiting[k] != 0 or coverage_reduction_end[k] != 0
                            or coverage_step_month[k] != 0
                            or coverage_escalation_annual[k] != 0.0
                            or coverage_term[k] != 0):
                        continue          # rule-bearing coverages run separately
                    rate = coverage_rates[cov_idx, mp, year] * coverage_amount[k]
                    if coverage_risk[cov_idx] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year

            inforce_t = 0.0
            dclaim_occ = 0.0  # death-benefit-weighted in-force (claim base)
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
                inforce_t += state_sum
                dclaim_occ += state_sum * state_death_benefit_factor[s]
                deaths_acc += state_sum * state_death_exit[s, mp, year]
                lapse_acc += state_sum * state_lapse[s, mp, year]
                if state_pays_premium[s]:
                    prem_occ += state_sum
                if state_pays_benefit[s]:
                    cap = periodic_benefit_term_months[s]
                    if cap > 0:
                        # Pay only the cohorts still within the cap; lives
                        # past it stay in force but stop receiving income.
                        ben_sum = 0.0
                        for tau in range(cap):
                            ben_sum += occ[s_off + tau]
                        benefit_occ += ben_sum
                    else:
                        benefit_occ += state_sum

            inforce[mp, t] = inforce_t
            lapse_flow[mp, t] = lapse_acc
            deaths[mp, t] = deaths_acc
            level = (prem_occ * premium[mp] * premium_factor[mp, year]
                     if (t < premium_term and t % prem_freq == 0) else 0.0)
            premium_cf[mp, t] = level
            mortality_cf[mp, t] = dclaim_occ * claim_rate
            morbidity_cf[mp, t] = inforce_t * morb_rate
            annuity_cf[mp, t] = (inforce_t * annuity_payment[mp] * annuity_factor[mp, year]
                                  if t % ann_freq == 0 else 0.0)
            disability_cf[mp, t] = benefit_occ * disability_income[mp]
            # Expense: same dispatch as _project_kernel (see its comment).
            ann_prem = premium[mp] * premium_factor[mp, year] * 12.0 / prem_freq
            acquisition_expense = (
                cnt * (acquisition_premium * ann_prem + acquisition_per_policy)
                if t == 0 else 0.0)
            maintenance_premium_expense = (
                inforce_t * maintenance_premium[t] * ann_prem / 12.0
                if t < premium_term else 0.0)
            maintenance_per_policy_expense = inforce_t * maintenance_per_policy[t]
            # LAE applies to claim + morbidity claims only --
            # disability income is a periodic annuity-like benefit, lump
            # sums are one-off transitions, and conflating either with
            # LAE would double-count. Add a dedicated basis later if the
            # practice ever needs it.
            lae_expense = lae[t] * (
                mortality_cf[mp, t] + morbidity_cf[mp, t])
            expense_cf[mp, t] = (acquisition_expense + maintenance_premium_expense
                                 + maintenance_per_policy_expense + lae_expense)

            for i in range(total_cohorts):
                occ_next[i] = 0.0
            for e in range(n_edges):
                s_from = edge_from[e]
                s_to = edge_to[e]
                D_from = state_duration_max[s_from]
                src_off = state_offset[s_from]
                is_residual = s_from == s_to
                if is_residual:
                    # Residual (stay) edge advances each cohort tau -> tau+1. A
                    # deterministic transition (Transition.after_sojourn_months=K)
                    # rides this gate: a cohort advancing INTO sojourn >= K is
                    # routed prob-1 to its destination instead of advancing.
                    #   det_to >= 0  -> route survivors to dest cohort 0 (to=state)
                    #   det_to <  0  -> leave the in-force set (to=None; the
                    #                   historical exit -- no occ_next write)
                    # The route arm and the advance arm are mutually exclusive on
                    # ``next_tau >= K``, so flow is written exactly once (no
                    # double-count); to=None is byte-identical to the old exit.
                    K = state_det_at[s_from]
                    dto = state_det_to[s_from]
                    dlump = state_det_lump[s_from]
                    for tau in range(D_from):
                        flow = occ[src_off + tau] * edge_prob[e, mp, year, tau]
                        next_tau = tau + 1 if tau + 1 < D_from else D_from - 1
                        if K > 0 and next_tau >= K:
                            if dto >= 0:
                                occ_next[state_offset[dto]] += flow
                            if dlump:
                                disability_cf[mp, t] += flow * disability_benefit[mp]
                        else:
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
            step_month = coverage_step_month[k]
            step_factor = coverage_step_factor[k]
            esc = coverage_escalation_annual[k]
            cap = coverage_escalation_cap[k]
            if (wait == 0 and red_end == 0 and step_month == 0 and esc == 0.0
                    and coverage_term[k] == 0):
                continue          # rule-free -- already in the main pass
            benefit = coverage_amount[k]
            red_factor = coverage_reduction_factor[k]
            mortality_risk = coverage_risk[cov_idx] == 0
            cov_end = boundary if coverage_term[k] == 0 else min(boundary, coverage_term[k])
            for t in range(wait, cov_end):
                mult = _benefit_factor(t, t // 12, red_factor, red_end,
                                       step_month, step_factor, esc, cap)
                amt = (inforce[mp, t] * coverage_rates[cov_idx, mp, t // 12]
                       * benefit * mult)
                if mortality_risk:
                    mortality_cf[mp, t] += amt
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
            step_month = coverage_step_month[k]
            step_factor = coverage_step_factor[k]
            esc = coverage_escalation_annual[k]
            cap = coverage_escalation_cap[k]
            red_factor = coverage_reduction_factor[k]
            undiagnosed = 1.0   # fraction of the in-force still undiagnosed
            d_year = -1
            d_rate = 0.0
            cov_end = boundary if coverage_term[k] == 0 else min(boundary, coverage_term[k])
            for t in range(cov_end):
                year = t // 12
                if year != d_year:
                    d_rate = coverage_rates[cov_idx, mp, year]
                    d_year = year
                if t >= wait:
                    mult = _benefit_factor(t, t // 12, red_factor, red_end,
                                       step_month, step_factor, esc, cap)
                    morbidity_cf[mp, t] += (inforce[mp, t] * undiagnosed
                                            * d_rate * benefit * mult)
                undiagnosed *= (1.0 - d_rate)

    return (inforce, deaths, premium_cf, mortality_cf, morbidity_cf, expense_cf,
            annuity_cf, disability_cf, lapse_flow, maturity_cf, maturity_survivors)


def _add_state_mortality_rates(rate_dict, state_machine, basis, sex_grid,
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
    for rname in {s.mortality_rate for s in state_machine.states}:
        if rname == "mortality" or rname in rate_dict:
            continue
        mort_fn = table.get(rname) or basis.mortality_annual
        rate_dict[rname] = np.ascontiguousarray(annual_to_monthly(
            mort_fn(sex_grid, issue_age_grid, duration_grid,
                    issue_class_grid, elapsed_grid)))


# Transition rates that remove occupancy from the in-force set as a lapse (the
# surrender trigger). A state may carry at most one; the surrender value is paid
# on ``occupancy x this rate``, so a paid-up state lapses at its own
# ``lapse_paidup`` rate and a waiver state at its own ``lapse_waiver`` rate
# (which defaults to 0 -- a non-lapsing waiver -- unless a rate is set).
_LAPSE_RATES = ("lapse", "lapse_paidup", "lapse_waiver")


def _state_lapse_stack(state_machine, rate_dict):
    """Per-state monthly lapse rate, ``(n_states, n_mp, n_year)``.

    Mirrors the per-state mortality stack: each state's surrender count is
    ``occ[state] x state_lapse[state]``, so surrender follows the actual
    state-machine lapse (WAIVER does not lapse; paid-up lapses at
    ``lapse_paidup``) instead of a single global rate applied to the total
    in-force. A state with no lapse transition contributes a zero row.
    """
    zero = np.zeros_like(rate_dict["lapse"])
    rows = []
    for s in state_machine.states:
        rname = None
        for tr in s.transitions:
            if tr.rate in _LAPSE_RATES:
                rname = tr.rate
                break
        rows.append(rate_dict[rname] if rname is not None else zero)
    return np.ascontiguousarray(np.stack(rows))


def project_cashflows(model_points: ModelPoints, basis: Basis,
                      *, lapse_scale: FloatArray | None = None,
                      emit_state: bool = False) -> Cashflows:
    """Project cash flows for every model point.

    The Pythonic wrapper: it extracts raw arrays from the inputs and
    evaluates the basis. Mortality, lapse and the coverage rates are
    evaluated on the per-policy-year grid, not the full ``(n_mp, n_time)``
    grid -- all change only once a year, so this is an identical result for
    a twelfth of the work.

    ``lapse_scale`` is an optional per-policy-year multiplier on the lapse
    decrement, shape ``(n_mp, n_years)`` -- the seam for a dynamic lapse whose
    elasticity is resolved to a fixed multiplier array UP FRONT (e.g. an
    account-value moneyness path). It scales the per-state lapse stack before
    the kernel runs, so the hot loop sees only the already-scaled lapse array
    (no per-step callback). ``None`` (the default) leaves the lapse untouched.

    ``emit_state`` (default ``False``) attaches a :class:`StateTrace` to the
    returned :class:`Cashflows` on the Markov path -- the compiled edge list and
    per-state flags for the opt-in per-state reserve. It does not change any cash
    flow the projection returns; the semi-Markov path leaves it ``None``.
    """
    if model_points.term_months.shape[0] == 0:
        raise ValueError(
            "model_points is empty (n_mp=0); measure() cannot project a "
            "zero-policy portfolio. Filter empty segments upstream."
        )
    # The projection horizon is the contract boundary (paragraph 34), which
    # defaults to ``term_months`` -- so a book with no boundary cut sizes the
    # arrays exactly as before. A shorter boundary trims both the loop and the
    # array width.
    n_time = int(model_points.contract_boundary_months.max())  # months 0 .. n_time-1
    n_years = (n_time + 11) // 12
    durations = np.arange(n_years)

    n_mp = model_points.term_months.shape[0]
    if lapse_scale is not None:
        lapse_scale = np.asarray(lapse_scale, dtype=np.float64)
        if lapse_scale.shape != (n_mp, n_years):
            raise ValueError(
                f"lapse_scale must be (n_mp, n_years) = ({n_mp}, {n_years}); "
                f"got {lapse_scale.shape}")

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
    # Dynamic-lapse seam: scale the per-policy-year monthly lapse before it is
    # compiled into the state-machine transition edges (and the surrender-flow
    # stack), so BOTH the in-force decrement and the reported surrender follow
    # the scaled lapse consistently. The factor array is resolved up front (e.g.
    # an account-value moneyness path), so the hot kernel sees only the scaled
    # rate -- no per-step callback. ``lapse_paidup`` (below) takes the same scale.
    if lapse_scale is not None:
        lapse = np.ascontiguousarray(lapse * lapse_scale)
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
    (coverage_is_diagnosis, coverage_risk,
     coverage_funds_from_account, coverage_pays_account_balance) = coverage_arrays(
        aligned_coverages, model_points.calculation_methods,
    )
    # coverage_funds_from_account / coverage_pays_account_balance are the
    # account-chassis interaction flags (all-False today; the universal-life
    # account roll folds onto them in a later step). Unused here for now.
    # build_coverage_rates stacks the per-coverage annual rates; the whole
    # stack is converted to monthly. mortality_annual above is the separate
    # in-force decrement input; a death coverage's claim payout is driven
    # by its own rate_table from basis.coverages.
    coverage_rates = np.ascontiguousarray(annual_to_monthly(build_coverage_rates(
        [r.rate for r in aligned_coverages],
        sex_grid, issue_age_grid, duration_grid,
        issue_class_grid, elapsed_grid,
        codes=[r.code for r in aligned_coverages],
    )))
    # Shape contract: _project_kernel / _project_kernel_semi_markov index
    # coverage_rates[coverage_index[k], mp, year]. Lock the shape here so a future
    # change to the grid construction surfaces at this assertion rather than
    # silently broadcasting into a wrong claim rate.
    assert coverage_rates.shape == (
        len(aligned_coverages), len(model_points.issue_age), n_years
    ), f"coverage_rates shape {coverage_rates.shape} != (n_cov, n_mp, n_years)"
    # Premium SHAPE -- a multiplicative factor on the level premium, per
    # (sex, issue_age, duration) year grid. NOT a rate: it is never converted
    # to monthly (a step-up factor > 1.0 would fail annual_to_monthly's <= 1
    # check). None -> all-ones (level premium), a structural no-op multiply.
    if basis.premium_factor_annual is None:
        premium_factor = np.ones((len(model_points.issue_age), n_years))
    else:
        premium_factor = validate_factor(
            basis.premium_factor_annual(
                sex_grid, issue_age_grid, duration_grid,
                issue_class_grid, elapsed_grid),
            "premium_factor_annual", (len(model_points.issue_age), n_years))
    # Annuity SHAPE -- the survival-benefit twin of premium_factor (escalating
    # annuity). Same multiplicative-scale rules: never annual_to_monthly, None
    # -> all-ones (level annuity), a structural no-op multiply.
    if basis.annuity_factor_annual is None:
        annuity_factor = np.ones((len(model_points.issue_age), n_years))
    else:
        annuity_factor = validate_factor(
            basis.annuity_factor_annual(
                sex_grid, issue_age_grid, duration_grid,
                issue_class_grid, elapsed_grid),
            "annuity_factor_annual", (len(model_points.issue_age), n_years))
    # Expense primitives, named (category)_(base). The surrender-value and face
    # maintenance components are added post-projection (they ride the in-force
    # surrender value / sum assured, known only after the time loop).
    (expense_acquisition_premium, expense_acquisition_per_policy, expense_maintenance_premium,
     maintenance_per_policy, lae, expense_maintenance_surrender_value,
     expense_maintenance_face) = _expense_kernel_args(
        basis, n_time,
    )

    # Universal-life account-roll inputs (per-policy scalars / arrays). All
    # all-False / stub for a non-account portfolio -- a strict no-op.
    (has_account, mp_account, account_value0, account_face,
     account_prem_to_av, account_coi_rate, account_admin_fee,
     account_credit, account_charge, account_surr_charge) = _account_kernel_args(
        model_points, basis, coverage_rates, coverage_funds_from_account,
        coverage_pays_account_balance, maintenance_per_policy, n_time, n_years,
    )
    # v1 supports a homogeneous account portfolio (every model point carries the
    # account-backed death coverage). A MIXED book -- some account rows, some
    # plain protection rows -- needs a per-model-point risk-adjustment split
    # (NAR-priced RA on the account rows, slot RA on the rest); until that lands
    # the account RA path would mis-price the non-account rows, so reject mixed
    # input rather than mis-measure. Measure the account and non-account subsets
    # separately.
    if has_account and not bool(mp_account.all()):
        raise NotImplementedError(
            "a portfolio mixing account-backed (universal-life) and plain "
            "model points is not yet supported -- measure the account and "
            "non-account subsets separately. (Per-model-point RA splitting for "
            "mixed books is a planned follow-up.)")
    # Universal-life annuitization cross-checks: an annuitizing MP needs an
    # account to convert (the coverage flags, known only here) and must leave at
    # least one payout month inside the projection horizon (the boundary).
    annz = model_points.annuitization_months > 0
    if np.any(annz):
        if not has_account or np.any(annz & ~mp_account):
            raise ValueError(
                "annuitization_months is set on a model point with no "
                "account-backed coverage -- there is no balance to convert. An "
                "annuitizing contract must carry a funds_from_account / "
                "pays_account_balance coverage.")
        if np.any(annz & (model_points.annuitization_months
                          >= model_points.contract_boundary_months)):
            raise ValueError(
                "annuitization_months must be < contract_boundary_months (the "
                "projection runs to the boundary, so the conversion month must "
                "leave at least one payout month inside the horizon).")
    # Variable payout: the per-MP monthly assumed interest rate (AIR).
    # NaN where the payout is fixed (the default) -- NaN propagates through the
    # power so finite-AIR rows convert and fixed rows stay NaN; the kernel
    # re-floats only the finite-AIR annuitizing rows.
    annuity_air_monthly = np.ascontiguousarray(
        (1.0 + model_points.annuity_air_annual) ** (1.0 / 12.0) - 1.0)
    # A universal-life account book pays its benefits (the account value) at the
    # exit, not over a settlement pattern; the measurement's settlement factor is
    # keyed on basis.discount_monthly (the GMM in-year rate), which would also be
    # wrong under VFA. Reject a settlement_pattern on an account book rather than
    # mis-discount it (a follow-up if a UL claim-settlement lag is ever needed).
    if has_account and basis.settlement_pattern is not None:
        raise NotImplementedError(
            "a settlement_pattern is not supported on a universal-life account "
            "book (the account benefit settles at exit, not over a pattern).")

    # In-force state machine -- see ``multistate.resolve_model`` for
    # the fallback policy when ``basis.state_machine`` is unset.
    state_machine = resolve_model(basis)
    seating = np.asarray(state_machine.seating, np.int64)
    if model_points.state.size and int(model_points.state.max()) >= seating.shape[0]:
        raise ValueError(
            f"ModelPoints.state has value {int(model_points.state.max())} but the "
            f"resolved state model accepts only {seating.shape[0]} seating states "
            f"(valid 0..{seating.shape[0] - 1}); check the state column against the "
            "segment's state_machine")
    start_state = seating[model_points.state]

    # A state-conditioned death benefit (death_benefit_factor) weights only the
    # AGGREGATE death claim (the rule-free, non-diagnosis pass that pays off the
    # death-weighted occupancy). The per-month coverage-rule pass and the
    # diagnosis pass pay off plain in-force, so combining the factor with a
    # rule-bearing or diagnosis DEATH-risk coverage would weight one death
    # claim and not the other -- inconsistent. Reject that mix in v1.
    if (model_points.coverage_index is not None
            and model_points.coverage_index.shape[0] > 0
            and any(s.death_benefit_factor != 1.0
                    for s in state_machine.states)):
        cov_idx_k = model_points.coverage_index
        death_k = coverage_risk[cov_idx_k] == 0
        diag_k = coverage_is_diagnosis[cov_idx_k]
        rule_k = ((model_points.coverage_waiting != 0)
                  | (model_points.coverage_reduction_end != 0)
                  | (model_points.coverage_step_month != 0)
                  | (model_points.coverage_escalation_annual != 0.0))
        if np.any(death_k & (diag_k | rule_k)):
            raise ValueError(
                "state-conditioned death benefit (State.death_benefit_factor) "
                "is not supported together with a rule-bearing (waiting / "
                "reduction / step / escalation) or diagnosis DEATH-risk "
                "coverage in v1: the rule and diagnosis death passes pay off "
                "plain in-force, so the per-state factor would weight one "
                "death claim and not the other. Use a plain death coverage."
            )

    uses_annuity_forms = bool(np.any(
        (model_points.annuity_start_months > 0)
        | (model_points.annuity_term_months > 0)
        | (model_points.annuity_guarantee_months > 0)))
    if uses_annuity_forms and is_semi_markov(state_machine):
        raise NotImplementedError(
            "the deferred / term-certain / guaranteed-period annuity payout forms "
            "are supported on the Markov projection path only (v1); this portfolio "
            "resolves to a semi-Markov state model.")
    # The guaranteed (certain) annuity stream -- the Markov kernel sets it; it
    # stays None on the semi-Markov path (which rejects the forms above).
    annuity_certain_cf = None
    # The opt-in per-state reserve handles -- built on the Markov path below when
    # emit_state is asked for; left None otherwise (incl. the semi-Markov path).
    state_trace = None
    if has_account and is_semi_markov(state_machine):
        # The account roll is folded into the Markov kernel only (v1). A plain
        # UL contract has no state model, so it routes to the Markov path; a
        # semi-Markov account product is a deferred kernel step.
        raise NotImplementedError(
            "universal-life account roll is supported on the Markov projection "
            "path only (v1); this portfolio resolves to a semi-Markov state "
            "model. Account-on-semi-Markov is a later step."
        )
    if is_semi_markov(state_machine):
        # Phase (c) detailed projection. Build the rate dict the cohort-
        # aware compile expects: static rates stay (n_mp, n_year); the
        # duration-dependent reincidence rate carries an extra cohort
        # axis of length ``max_cohort`` -- the largest ``sojourn_tracking_months``
        # across the tracked states. Coverage-rule and diagnosis-coverage
        # passes ride the cohort-aware main pass: rule benefits scale the
        # saved per-month total in-force, diagnosis pools multiply that
        # same trajectory by a per-coverage depletion fraction.
        max_cohort = max(s.sojourn_tracking_months for s in state_machine.states
                          if s.sojourn_tracking_months > 0)
        rate_dict = {"mortality": mortality, "lapse": lapse}
        _add_state_mortality_rates(rate_dict, state_machine, basis,
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
        compiled = compile_model_with_duration(state_machine, rate_dict)
        edge_from = compiled.edge_from
        edge_to = compiled.edge_to
        edge_prob = compiled.edge_prob
        edge_lump_sum = compiled.edge_lump_sum
        n_states = compiled.n_states
        state_pays_premium = compiled.state_pays_premium
        state_pays_benefit = compiled.state_pays_benefit
        state_duration_max = compiled.state_duration_max
        periodic_benefit_term_months = compiled.periodic_benefit_term_months
        # compile_model_with_duration returns ``edge_prob`` shape
        # ``(n_edges, n_mp, n_year, max_D)`` -- already in the layout the
        # detailed kernel reads (edge axis outer, cohort axis inner).
        state_offset = np.zeros(n_states + 1, dtype=np.int64)
        state_offset[1:] = np.cumsum(state_duration_max)
        # Per-state exact death-exit stack (n_states, n_mp, n_year) -- each
        # state's in-force death exit (survive x mortality) so the death-count
        # reporter respects the within-month competing-risk order.
        state_death_exit = compiled.state_death_exit
        state_lapse = _state_lapse_stack(state_machine, rate_dict)
        state_death_benefit_factor = compiled.state_death_benefit_factor
        state_det_at = compiled.state_det_at
        state_det_to = compiled.state_det_to
        state_det_lump = compiled.state_det_lump
        (inforce, deaths, premium_cf, mortality_cf, morbidity_cf, expense_cf,
         annuity_cf, disability_cf, lapse_flow,
         maturity_cf, maturity_survivors) = _project_kernel_semi_markov(
            state_death_exit, state_lapse,
            state_death_benefit_factor, state_det_at, state_det_to, state_det_lump,
            edge_from, edge_to, edge_prob, edge_lump_sum,
            n_states, state_duration_max, state_offset, periodic_benefit_term_months,
            state_pays_premium, state_pays_benefit, start_state,
            model_points.term_months,
            model_points.contract_boundary_months,
            model_points.count,
            model_points.premium,
            premium_factor,
            annuity_factor,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.coverage_index,
            model_points.coverage_amount,
            model_points.coverage_offset,
            model_points.coverage_waiting,
            model_points.coverage_reduction_end,
            model_points.coverage_reduction_factor,
            model_points.coverage_step_month,
            model_points.coverage_step_factor,
            model_points.coverage_escalation_annual,
            model_points.coverage_escalation_cap,
            model_points.coverage_term,
            coverage_rates,
            coverage_risk,
            coverage_is_diagnosis,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            model_points.disability_income,
            model_points.disability_benefit,
            expense_acquisition_premium,
            expense_acquisition_per_policy,
            expense_maintenance_premium,
            maintenance_per_policy,
            lae,
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
        _add_state_mortality_rates(rate_dict, state_machine, basis,
                                   sex_grid, issue_age_grid, duration_grid,
                                   issue_class_grid, elapsed_grid)
        if basis.ci_incidence_annual is not None:
            ci_inc = np.ascontiguousarray(annual_to_monthly(
                basis.ci_incidence_annual(
                    sex_grid, issue_age_grid, duration_grid,
                    issue_class_grid, elapsed_grid)))
            rate_dict["ci_incidence"] = ci_inc
        if model_references_rate(state_machine, "lapse_paidup"):
            paidup_fn = (basis.lapse_paidup_annual
                         or basis.lapse_annual)
            paidup = annual_to_monthly(
                paidup_fn(sex_grid, issue_age_grid, duration_grid,
                          issue_class_grid, elapsed_grid))
            if lapse_scale is not None:
                paidup = paidup * lapse_scale
            rate_dict["lapse_paidup"] = np.ascontiguousarray(paidup)
        if model_references_rate(state_machine, "lapse_waiver"):
            # Unlike lapse_paidup (falls back to the active lapse), the waiver
            # state defaults to NO lapse -- a waived contract holds free cover,
            # so anti-selection keeps it in force. A 0 rate preserves the
            # pure-waiver behaviour; set ``lapse_waiver_annual`` (typically low)
            # to model the residual waived-state surrender.
            if basis.lapse_waiver_annual is None:
                waiver_lapse = np.zeros_like(rate_dict["lapse"])
            else:
                waiver_lapse = annual_to_monthly(
                    basis.lapse_waiver_annual(
                        sex_grid, issue_age_grid, duration_grid,
                        issue_class_grid, elapsed_grid))
                if lapse_scale is not None:
                    waiver_lapse = waiver_lapse * lapse_scale
            rate_dict["lapse_waiver"] = np.ascontiguousarray(waiver_lapse)
        compiled = compile_model(state_machine, rate_dict)
        edge_from = compiled.edge_from
        edge_to = compiled.edge_to
        edge_prob = compiled.edge_prob
        edge_lump_sum = compiled.edge_lump_sum
        n_states = compiled.n_states
        state_pays_premium = compiled.state_pays_premium
        state_pays_benefit = compiled.state_pays_benefit
        state_death_exit = compiled.state_death_exit
        state_lapse = _state_lapse_stack(state_machine, rate_dict)
        state_death_benefit_factor = compiled.state_death_benefit_factor
        (inforce, deaths, premium_cf, mortality_cf, morbidity_cf, expense_cf,
         annuity_cf, annuity_certain_cf, disability_cf, lapse_flow,
         maturity_cf, maturity_survivors,
         av, av_mid, coi_av,
         prem_to_av_out, admin_out, account_charge_out) = _project_kernel(
            state_death_exit,
            state_lapse,
            state_death_benefit_factor,
            compiled.state_premium_term_to,
            edge_from,
            edge_to,
            edge_prob,
            edge_lump_sum,
            n_states,
            state_pays_premium,
            state_pays_benefit,
            start_state,
            model_points.term_months,
            model_points.contract_boundary_months,
            model_points.count,
            model_points.premium,
            premium_factor,
            annuity_factor,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.coverage_index,
            model_points.coverage_amount,
            model_points.coverage_offset,
            model_points.coverage_waiting,
            model_points.coverage_reduction_end,
            model_points.coverage_reduction_factor,
            model_points.coverage_step_month,
            model_points.coverage_step_factor,
            model_points.coverage_escalation_annual,
            model_points.coverage_escalation_cap,
            model_points.coverage_term,
            coverage_rates,
            coverage_risk,
            coverage_is_diagnosis,
            coverage_pays_account_balance,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            model_points.disability_income,
            model_points.disability_benefit,
            expense_acquisition_premium,
            expense_acquisition_per_policy,
            expense_maintenance_premium,
            maintenance_per_policy,
            lae,
            has_account,
            mp_account,
            account_value0,
            account_face,
            account_prem_to_av,
            account_coi_rate,
            account_admin_fee,
            account_credit,
            account_charge,
            model_points.annuitization_months,
            model_points.annuitization_rate,
            annuity_air_monthly,
            model_points.minimum_accumulation_benefit,
            model_points.annuity_start_months,
            model_points.annuity_term_months,
            model_points.annuity_guarantee_months,
            n_time,
        )
        if emit_state:
            # Per-state reserve handles: the exact compiled edge list (so a
            # caller replays the same occupancy) plus the per-state flags the
            # kernel weighted by. state_premium_term_to != -1 marks the
            # deterministic active -> paid-up move the edge list does not carry.
            # death_face: the per-unit LEVEL death benefit (sum of the plain
            # death-risk coverage amounts per policy); rule-bearing death
            # coverages are flagged so the (level-benefit) sum-at-risk rejects
            # them. Reuses the projection's own coverage_risk -- no reconstruction.
            death_face = np.zeros(n_mp)
            has_death_rules = False
            ci = model_points.coverage_index
            if ci is not None and ci.shape[0] > 0:
                plain_death = ((coverage_risk[ci] == 0)
                               & ~coverage_is_diagnosis[ci]
                               & ~coverage_pays_account_balance[ci])
                mp_of_k = np.repeat(np.arange(n_mp),
                                    np.diff(model_points.coverage_offset))
                death_face = np.bincount(
                    mp_of_k, weights=np.where(plain_death, model_points.coverage_amount, 0.0),
                    minlength=n_mp)
                rule_k = plain_death & (
                    (model_points.coverage_waiting != 0)
                    | (model_points.coverage_reduction_end != 0)
                    | (model_points.coverage_step_month != 0)
                    | (model_points.coverage_escalation_annual != 0.0)
                    | (model_points.coverage_term != 0))
                has_death_rules = bool(np.any(rule_k))
            state_trace = StateTrace(
                edge_from=edge_from,
                edge_to=edge_to,
                edge_prob=edge_prob,
                edge_lump_sum=edge_lump_sum,
                n_states=n_states,
                start_state=start_state,
                count=model_points.count,
                state_pays_premium=state_pays_premium,
                state_pays_benefit=state_pays_benefit,
                death_benefit_factor=state_death_benefit_factor,
                has_premium_term_move=bool(
                    np.any(compiled.state_premium_term_to != -1)),
                state_death_exit=state_death_exit,
                state_lapse=state_lapse,
                state_names=tuple(s.name for s in state_machine.states),
                death_face=death_face,
                has_death_coverage_rules=has_death_rules,
            )
    # Surrender value -- post-projection compute. ``lapse_flow``
    # is the per-month state-machine lapse exit count (occupancy on each state
    # times that state's own lapse rate), so the surrender follows the actual
    # lapse: a non-lapsing WAIVER state pays no surrender, and a paid-up state
    # lapses at ``lapse_paidup``, not the active rate. For a single active
    # state ``lapse_flow == inforce x lapse``, the historical formula.
    # ``surrender_value_curve = None`` falls back to zero, the historical
    # "lapse silently removes" behaviour.
    # maintenance_surrender_value expense rides the in-force surrender value, so it
    # needs a surrender_value_curve and is undefined on an account-backed book
    # (the account fund_fee already charges the account value -- a second charge
    # would double-count). Guard both at measure time rather than silently
    # emitting a zero / double-counted expense leg.
    if expense_maintenance_surrender_value != 0.0:
        if basis.surrender_value_curve is None:
            raise ValueError(
                "a 'maintenance_surrender_value' ExpenseItem requires "
                "Basis.surrender_value_curve (the surrender value it charges "
                "on); none is set, so the expense base would be silently zero."
            )
        if has_account:
            raise ValueError(
                "a 'maintenance_surrender_value' ExpenseItem is not supported on an "
                "account-backed (universal-life / VFA) book -- the account "
                "fund_fee already charges the account value. Remove the item or "
                "measure the account business without it."
            )
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
        # ``inforce_sv`` is the surrender value held by the in-force survivors
        # each month (per-policy surrender value x in-force) -- the base the
        # maintenance_surrender_value maintenance expense charges. It mirrors each
        # surrender mode's per-policy value but weights by ``inforce`` (the
        # begin-of-month survivors), not ``lapse_flow`` (the month's exits).
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
            # per-policy SV = (cum_premium / inforce) * value, so the in-force
            # aggregate cancels the inforce divisor: cum_premium * value.
            inforce_sv = cum_premium * value
        elif mode == "amount_per_policy":
            # Contractual per-policy amount at policy-duration t. The number
            # lapsing in month t is ``lapse_flow[t]``; each pays ``value[t]``.
            # Linear in the in-force, so the in-force ``count / inforce[elapsed]``
            # rescale re-bases it exactly.
            surrender_cf = lapse_flow * value
            inforce_sv = inforce * value
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
            base_col = np.asarray(base, dtype=np.float64)[:, None]
            surrender_cf = lapse_flow * value * base_col
            inforce_sv = inforce * value * base_col
        else:
            raise ValueError(
                f"unknown surrender_value_basis {mode!r}; expected one of "
                f"{SURRENDER_VALUE_BASES}."
            )
        # maintenance_surrender_value maintenance: rate/12 of the in-force
        # surrender value each month, added to the expense leg. No inflation --
        # the surrender-value curve already carries its own growth.
        if expense_maintenance_surrender_value != 0.0:
            expense_cf += (expense_maintenance_surrender_value / 12.0) * inforce_sv
    # maintenance_face maintenance: rate/12 of the policy's sum assured (the
    # main coverage's amount, flagged by coverage_is_main) on every in-force
    # month. Inflated like maintenance_per_policy -- the sum assured is level,
    # so the maintenance rate inflates. Independent of the surrender curve.
    if expense_maintenance_face != 0.0:
        is_main = model_points.coverage_is_main
        if not np.any(is_main):
            raise ValueError(
                "a (maintenance, face) ExpenseItem charges on the policy's sum "
                "assured -- the main coverage's amount -- but no coverage is "
                "flagged as the main contract. Set ModelPoints.coverage_is_main."
            )
        main_amount = model_points.coverage_amount * is_main
        cov_mp = np.repeat(np.arange(n_mp), np.diff(model_points.coverage_offset))
        face_amount = np.bincount(cov_mp, weights=main_amount, minlength=n_mp)
        expense_cf += ((expense_maintenance_face / 12.0)
                       * inflation_index(basis, n_time)
                       * face_amount[:, None] * inforce)
    account = None
    if has_account:
        # Account-backed surrender overrides the curve-based surrender for the
        # account rows: a surrender pays max(0, av_mid * (1 - surr_charge_rate))
        # per lapse exit. The lapse count is the per-month NON-maturity,
        # non-death exit (``exits - deaths``, with the maturing survivors removed
        # at the term) -- the actual lapses net of the within-month competing
        # risk, NOT the raw rate-weighted ``lapse_flow``. Term rows in a mixed
        # book keep their curve-based surrender (mp_account False -> untouched).
        n_mp_ = inforce.shape[0]
        inforce_pad = np.concatenate([inforce, np.zeros((n_mp_, 1))], axis=1)
        exits = inforce_pad[:, :-1] - inforce_pad[:, 1:]
        non_maturity_exits = exits - deaths
        boundary_idx = model_points.contract_boundary_months - 1
        within = (model_points.term_months - 1) <= boundary_idx
        term_idx = np.where(within, model_points.term_months - 1, boundary_idx)
        rows = np.arange(n_mp_)
        non_maturity_exits[rows, term_idx] -= np.where(
            within, maturity_survivors, 0.0)
        # Surrender pays the account value net of the surrender charge:
        # max(0, av_mid * (1 - surr_charge_rate)). surr_charge_rate is 0 with no
        # charge. surr_value is the per-policy figure; the flow weights it by the
        # lapse exit count, and inforce_surrender_value reads it (re-based to the
        # as-of count) for the mass-lapse surrender strain.
        surr_value = np.maximum(0.0, av_mid * (1.0 - account_surr_charge))
        acct_surr = non_maturity_exits * surr_value
        surrender_cf = np.where(mp_account[:, None], acct_surr, surrender_cf)
        # The entity holds the in-force-weighted account value (the VFA fund).
        fund = inforce_pad * av
        account = AccountTrajectory(
            av=av, av_mid=av_mid, coi=coi_av, fund=fund,
            prem_to_av=prem_to_av_out, admin_charge=admin_out,
            account_charge=account_charge_out, surr_value=surr_value)
    return Cashflows(
        inforce=inforce,
        deaths=deaths,
        premium_cf=premium_cf,
        mortality_cf=mortality_cf,
        morbidity_cf=morbidity_cf,
        expense_cf=expense_cf,
        annuity_cf=annuity_cf,
        disability_cf=disability_cf,
        maturity_cf=maturity_cf,
        maturity_survivors=maturity_survivors,
        surrender_cf=surrender_cf,
        account=account,
        annuity_certain_cf=annuity_certain_cf,
        state_trace=state_trace,
    )
