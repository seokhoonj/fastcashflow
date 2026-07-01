"""Model-neutral valuation core -- project cash flows into the BEL / RA bundle.

The prefix every GMM-family model shares: :func:`valued_projection` projects the
cash flows and rolls them into a :class:`ValuedProjection` (BEL / RA trajectories
plus discount context, NO CSM, no model identity). Each model then assembles its
own measurement (CSM / LRC) on top. Extracted from the GMM engine so VFA / PAA /
reinsurance share this core directly rather than borrowing from a GMM module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.curves import (
    discount_factors_from_curve,
    discount_monthly_curve,
)
from fastcashflow._numerics import (
    _cost_of_capital_ra,
    _forward_occupancy_kernel,
    _norm_ppf,
    _state_reserve_kernel,
    _risk_adjustment,
    _roll_forward_kernel,
    _settlement_factor,
    _settlement_lic,
)
from fastcashflow.model_points import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows


def _account_risk_adjustment(model_points, basis, proj, discount_monthly):
    """Universal-life risk adjustment -- priced on the net amount at risk.

    The insurance risk of an account-backed death leg is the mortality borne on
    the NET AMOUNT AT RISK (the death benefit above the account,
    ``deaths * max(0, face - av_mid)``) -- the account portion returns the
    policyholder's own money and bears no insurance risk -- plus expense risk,
    plus the morbidity risk of any cost-deducting rider (a fixed health benefit
    funded from the account). This BYPASSES :func:`_risk_adjustment` and its
    ``expense_cv != 0`` guard (a UL RA legitimately prices ``expense_cv``): run
    the at-risk claim, the morbidity claim and the expense through one
    roll-forward pass, then the confidence margin ``z(ra_confidence) *
    (mortality_cv*pv_nar + morbidity_cv*pv_morbidity + expense_cv*pv_expense)``,
    cost-of-capital-wrapped per ``ra_method``.
    """
    face = model_points.minimum_death_benefit
    n_mp, n_time = proj.mortality_cf.shape
    zeros_t = np.zeros((n_mp, n_time))
    zeros_mp = np.zeros(n_mp)
    nar_claim = np.ascontiguousarray(
        proj.deaths * np.maximum(0.0, face[:, None] - proj.account.av_mid))
    # The annuity payout (an annuitizing UL contract, phase 2) bears longevity
    # risk -- the insurer pays the income for as long as the annuitant lives --
    # so its PV is priced through longevity_cv, alongside the at-risk mortality
    # and expense. The annuity stream rides the survival slot of the
    # roll-forward (position 6); a non-annuitizing account book has annuity_cf
    # == 0, so pv_annuity == 0 and this term vanishes (byte-identical). The
    # account maturity lump is the return of the policyholder's own balance (an
    # investment component) and bears no insurance risk, so it is deliberately
    # NOT longevity-priced (it stays out, the maturity slot is zero here).
    # The morbidity claim of a cost-deducting rider (funds from the account, but
    # pays a fixed health benefit -- not the balance) bears morbidity risk; it
    # rides the DISABILITY slot of the roll-forward (position 3, otherwise empty
    # for an account book) purely to harvest its PV. A book with no such rider
    # has morbidity_cf == 0, so pv_morbidity == 0 and the term vanishes
    # (byte-identical). expense_cf rides the morbidity slot for the same reason.
    _, pv_nar, pv_expense, pv_morbidity, pv_annuity = _roll_forward_kernel(
        nar_claim, proj.expense_cf, proj.morbidity_cf, zeros_t, zeros_t,
        proj.annuity_cf, zeros_mp, zeros_t,
        model_points.contract_boundary_months, discount_monthly)
    z = _norm_ppf(basis.ra_confidence)
    confidence_margin = z * (basis.mortality_cv * pv_nar
                             + basis.morbidity_cv * pv_morbidity
                             + basis.expense_cv * pv_expense
                             + basis.longevity_cv * pv_annuity)
    if basis.ra_method == "cost_of_capital":
        return _cost_of_capital_ra(
            confidence_margin, discount_monthly, basis.cost_of_capital_rate)
    return confidence_margin


@dataclass(frozen=True, slots=True, eq=False)
class ValuedProjection:
    """Neutral valuation bundle -- the model-agnostic prefix of a full
    measurement.

    The projected cash flows valued into BEL / RA trajectories plus the discount
    context, with NO CSM and NO model identity. Produced by
    :func:`valued_projection` (downstream of the cash-flow projection). Each
    model's full measurement assembles its own result from this bundle plus its
    own CSM / LRC machinery, so no model borrows another's measurement
    container. ``bel`` / ``ra`` are the ``(n_mp,)`` inception headline (column 0
    of the trajectories).
    """

    bel_path: FloatArray              # (n_mp, n_time+1) -- BEL trajectory
    ra_path: FloatArray               # (n_mp, n_time+1) -- RA trajectory
    lic_path: FloatArray              # (n_mp, n_time+1) -- liability for incurred claims
    discount_factor_bom: FloatArray   # beginning-of-month discount factors
    discount_factor_mid: FloatArray   # mid-of-month discount factors
    discount_monthly: FloatArray      # per-month discount / CSM-accretion rate curve
    cashflows: Cashflows              # the underlying projection
    # Opt-in per-state policy value V^i(t), (n_mp, n_states, n_time+1); None
    # unless valued_projection(..., state_reserve=True) was asked for it.
    state_reserve: FloatArray | None = None
    # Opt-in per-transition sum at risk S^ij + V^j - V^i, (n_mp, n_transition,
    # n_time+1), with its transition descriptors; None unless sum_at_risk=True.
    sum_at_risk: FloatArray | None = None
    transitions: "tuple | None" = None

    @property
    def bel(self) -> FloatArray:
        return self.bel_path[:, 0]

    @property
    def ra(self) -> FloatArray:
        return self.ra_path[:, 0]


def _safe_div(num: FloatArray, den: FloatArray) -> FloatArray:
    """Per-unit flow = aggregate flow / its occupancy base, 0 where the base is 0.

    Every aggregate flow this decomposes is zero wherever its occupancy base is
    (no premium with nobody on a premium state, no income with nobody disabled),
    so ``0 / 0 -> 0`` loses nothing and avoids a spurious NaN.
    """
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0.0)


def _state_reserve(model_points: ModelPoints, proj: Cashflows,
                      mortality_cf: FloatArray, morbidity_cf: FloatArray,
                      bel_path: FloatArray,
                      discount_monthly: FloatArray) -> FloatArray:
    """Per-state policy value ``V^i(t)`` for a Markov book (the opt-in path).

    The aggregate roll-forward (:func:`_roll_forward_kernel`) is a single-life
    recursion on occupancy-weighted cash flows, so the per-state value satisfies
    ``sum_i occ_i(t) V^i(t) == bel[t]`` whenever each aggregate flow is split
    onto its true occupancy base and occupancy evolves by the same transition
    probabilities. This replays the occupancy from the compiled edge list, splits
    every flow (premium onto the premium states, the death claim by the
    per-state death-benefit factor, disability income onto the benefit states,
    morbidity / expense / surrender / maturity across the in-force pool), and
    values each state by the backward :func:`_state_reserve_kernel`.

    ``mortality_cf`` / ``morbidity_cf`` are the SETTLEMENT-ADJUSTED claim arrays
    the caller fed the aggregate roll-forward (not the raw projection claims), so
    the per-state values reconcile to ``bel_path`` exactly. Markov, no account,
    no annuity payout, no deterministic premium-term move -- v1 scope; anything
    else raises. Runs only on the full path (never inside ``value()``).
    """
    st = proj.state_trace
    if st is None:
        raise NotImplementedError(
            "state_reserve is supported on the Markov projection path only "
            "(v1); this portfolio resolves to a semi-Markov state model.")
    if proj.account is not None:
        raise NotImplementedError(
            "state_reserve does not yet support a universal-life account "
            "book (v1); the per-state value would need to net the account fund.")
    if st.has_premium_term_move:
        raise NotImplementedError(
            "state_reserve does not yet support a deterministic "
            "at-premium-term transition (active -> paid-up) (v1): the edge-list "
            "occupancy replay does not carry that calendar move.")
    if np.any(proj.annuity_cf != 0.0):
        raise NotImplementedError(
            "state_reserve does not yet support an annuity payout (v1): a "
            "survival / guaranteed annuity is not attributable per resident "
            "unit by the in-force pool split.")

    n_mp, n_time = proj.premium_cf.shape
    boundary = model_points.contract_boundary_months
    factor = st.death_benefit_factor            # (n_states,)

    occ = _forward_occupancy_kernel(
        st.edge_from, st.edge_to, st.edge_prob, st.n_states,
        st.start_state, st.count, boundary, n_time)   # (n_mp, n_states, n_time+1)
    occ_t = occ[:, :, :n_time]                         # (n_mp, n_states, n_time)

    # The replayed occupancy must match the projection's own in-force exactly;
    # a mismatch means an unsupported mechanic moved occupancy off the edge list.
    inforce_st = occ_t.sum(axis=1)                     # (n_mp, n_time)
    if not np.allclose(inforce_st, proj.inforce, rtol=1e-9, atol=1e-9):
        raise NotImplementedError(
            "state_reserve occupancy replay diverged from the projection "
            "in-force -- the portfolio uses a mechanic the per-state edge-list "
            "replay does not carry (v1 supports plain Markov transitions).")

    prem_occ = occ_t[:, st.premium_state, :].sum(axis=1)     # (n_mp, n_time)
    benefit_occ = occ_t[:, st.benefit_state, :].sum(axis=1)
    dclaim_occ = (occ_t * factor[None, :, None]).sum(axis=1)

    prem_unit = _safe_div(proj.premium_cf, prem_occ)         # per premium-state unit
    claim_unit = _safe_div(mortality_cf, dclaim_occ)         # per factor-weighted unit
    morb_unit = _safe_div(morbidity_cf, inforce_st)          # per in-force unit
    dis_unit = _safe_div(proj.disability_cf, benefit_occ)    # per benefit-state unit
    exp_unit = _safe_div(proj.expense_cf, inforce_st)
    surr_unit = _safe_div(proj.surrender_cf, inforce_st)

    prem_state = st.premium_state[None, :, None]
    benefit_state = st.benefit_state[None, :, None]
    # Beginning-of-month per-unit flow: premium is an inflow (reduces the value);
    # v1 carries no annuity payout, so the BOM leg is premium only.
    flow_bom = -np.where(prem_state, prem_unit[:, None, :], 0.0)
    # Mid-month per-unit outflow: death claim (weighted by the state factor),
    # morbidity + expense + surrender across the pool, disability on the benefit
    # states. Summed over states weighted by occupancy this reconstructs each
    # aggregate flow exactly.
    flow_mid = (factor[None, :, None] * claim_unit[:, None, :]
                + morb_unit[:, None, :]
                + np.where(benefit_state, dis_unit[:, None, :], 0.0)
                + exp_unit[:, None, :]
                + surr_unit[:, None, :])
    flow_bom = np.ascontiguousarray(flow_bom)
    flow_mid = np.ascontiguousarray(flow_mid)

    # Boundary seed: the maturity benefit per surviving unit, uniform across
    # states (states that cannot reach the boundary hold zero occupancy there).
    rows = np.arange(n_mp)
    inforce_b = occ.sum(axis=1)[rows, boundary]              # (n_mp,)
    mat_unit = _safe_div(proj.maturity_cf, inforce_b)        # (n_mp,)
    v_boundary = np.ascontiguousarray(
        np.repeat(mat_unit[:, None], st.n_states, axis=1))   # (n_mp, n_states)

    reserve = _state_reserve_kernel(
        flow_bom, flow_mid, v_boundary,
        st.edge_from, st.edge_to, st.edge_prob, boundary, discount_monthly)

    # Belt-and-suspenders: the per-state values must reconcile to bel_path.
    parity = (occ * reserve).sum(axis=1)                     # (n_mp, n_time+1)
    if not np.allclose(parity, bel_path, rtol=1e-7,
                       atol=1e-6 * max(1.0, float(np.abs(bel_path).max()))):
        raise AssertionError(
            "state_reserve parity check failed: sum_i occ_i V^i != bel_path "
            f"(max abs diff {np.max(np.abs(parity - bel_path)):.3e}).")
    return reserve, occ


@dataclass(frozen=True, slots=True)
class TransitionRisk:
    """Descriptor for one transition on the ``sum_at_risk`` axis.

    ``from_state`` / ``to_state`` are transient-state indices; ``to_state`` is
    ``None`` for an absorbing exit (death / lapse leave the in-force set).
    ``kind`` is ``"death"``, ``"lapse"`` or ``"transfer"`` (an inter-state edge).
    ``from_name`` / ``to_name`` label them for display (``to_name`` is the
    destination state name, or ``"death"`` / ``"lapse"``).
    """
    from_state: int
    to_state: "int | None"
    kind: str
    from_name: str
    to_name: str


def _sum_at_risk(model_points: ModelPoints, proj: Cashflows,
                 reserve: FloatArray, occ: FloatArray):
    """Per-transition sum at risk ``S^ij + V^j - V^i`` (Markov book, opt-in).

    The net exposure if a transition fires: the benefit paid on the edge plus the
    reserve of the destination state minus the reserve released by the source
    state, all per unit in the source state. Three transition kinds are
    enumerated per model point:

    * **death** exit from each state with a death decrement -- ``S`` is the death
      coverage SUM ASSURED per unit (``death_face x death_benefit_factor``),
      ``V^j = 0`` (death leaves the in-force set), so it is the classic net amount
      at risk ``sum_assured - V^i``. This is the exposure per death, independent of
      how often a death fires; it assumes the death coverage insures the modeled
      in-force death decrement (the usual case, coverage rate = mortality
      decrement). If a book deliberately decouples the death coverage rate from
      the in-force decrement, the reported figure is the per-claim sum-assured
      NAR, not a per-decrement expected loss.
    * **lapse** exit from each state with a lapse decrement -- ``S`` is the cash
      surrender value per unit (derived from ``surrender_cf`` over the actual
      lapse exits), ``V^j = 0``, so ``csv - V^i``.
    * **transfer** -- each compiled inter-state edge ``i -> j`` -- ``S`` is the
      transition lump (``disability_benefit`` when the edge pays one, else 0),
      ``V^j - V^i`` the reserve jump.

    Reserves are taken at the same month ``t`` as the source (``V^i(t)``), so
    ``sum_at_risk[t]`` reads "if this transition fires around month ``t``". Death
    / lapse benefits are level in v1; a rule-bearing death coverage is rejected.
    Returns ``(sum_at_risk (n_mp, n_transition, n_time+1), transitions)``.
    """
    st = proj.state_trace
    if st.has_death_coverage_rules:
        raise NotImplementedError(
            "sum_at_risk assumes a level death benefit (v1); this book has a "
            "rule-bearing death coverage (waiting / reduction / step / "
            "escalation / term) whose benefit varies in time.")
    ns = st.n_states
    n_mp, _, n_timep1 = reserve.shape
    n_time = n_timep1 - 1
    occ_t = occ[:, :, :n_time]

    # per-unit death benefit per state = level face x per-state factor
    death_benefit = st.death_face[:, None] * st.death_benefit_factor[None, :]  # (n_mp, n_states)

    # cash surrender value per unit (t) = surrender_cf / actual lapse exits, so it
    # reflects exactly what the projection paid (no curve reconstruction).
    year_idx = np.arange(n_time) // 12
    lapse_rate = np.transpose(st.state_lapse[:, :, year_idx], (1, 0, 2))  # (n_mp, n_states, n_time)
    lapse_exits = (occ_t * lapse_rate).sum(axis=1)                        # (n_mp, n_time)
    csv_unit = _safe_div(proj.surrender_cf, lapse_exits)                  # (n_mp, n_time)
    csv_pad = np.concatenate([csv_unit, np.zeros((n_mp, 1))], axis=1)     # (n_mp, n_time+1)

    lump = (np.asarray(model_points.disability_benefit, dtype=float)
            if model_points.disability_benefit is not None else np.zeros(n_mp))

    death_any = st.state_death_exit.any(axis=(1, 2))   # (n_states,)
    lapse_any = st.state_lapse.any(axis=(1, 2))
    names = st.state_names

    sar_rows, transitions = [], []
    for i in range(ns):                                 # death exits
        if death_any[i]:
            sar_rows.append(death_benefit[:, i:i + 1] - reserve[:, i, :])
            transitions.append(TransitionRisk(i, None, "death", names[i], "death"))
    for i in range(ns):                                 # lapse exits
        if lapse_any[i]:
            sar_rows.append(csv_pad - reserve[:, i, :])
            transitions.append(TransitionRisk(i, None, "lapse", names[i], "lapse"))
    for e in range(st.edge_from.shape[0]):              # inter-state transfers
        i, j = int(st.edge_from[e]), int(st.edge_to[e])
        if i == j:
            continue
        s = lump[:, None] if st.edge_lump_sum[e] else 0.0
        sar_rows.append(s + reserve[:, j, :] - reserve[:, i, :])
        transitions.append(TransitionRisk(i, j, "transfer", names[i], names[j]))

    sum_at_risk = np.stack(sar_rows, axis=1)            # (n_mp, n_transition, n_time+1)
    return sum_at_risk, tuple(transitions)


def valued_projection(model_points: ModelPoints, basis: Basis, *,
                      discount_monthly: FloatArray | None = None,
                      lapse_scale: FloatArray | None = None,
                      state_reserve: bool = False,
                      sum_at_risk: bool = False) -> ValuedProjection:
    """Value a cash-flow projection into the neutral BEL / RA bundle.

    The model-agnostic core of a full measurement: project the cash flows, then
    roll them forward into the BEL trajectory and price the RA, returning a
    :class:`ValuedProjection` (no CSM, no model identity). This is the prefix
    every GMM-family model shares; each model then adds its own CSM / LRC on top.

    ``discount_monthly`` overrides the discount / CSM-accretion curve (default:
    the locked-in ``discount_monthly_curve``). ``vfa.measure`` passes the flat
    underlying-items return here to value a universal-life account book under the
    VFA model -- the account roll (generation) is identical to GMM, only the
    discount rate differs. The override is only used by the account path, which
    carries no ``settlement_pattern``, so the settlement factor below (keyed on
    ``basis.discount_monthly``) is never reached together with an override.

    ``state_reserve`` (default ``False``) also computes the per-state policy
    value ``V^i(t)`` and attaches it to the bundle -- Markov books only (v1); see
    :func:`_state_reserve`. ``sum_at_risk`` (default ``False``) additionally
    computes the per-transition sum at risk (and implies ``state_reserve``, which
    it is built from). Both leave every other field byte-identical.
    """
    need_state = state_reserve or sum_at_risk
    proj = project_cashflows(model_points, basis, lapse_scale=lapse_scale,
                             emit_state=need_state)
    mortality_cf, morbidity_cf = proj.mortality_cf, proj.morbidity_cf
    if discount_monthly is None:
        discount_monthly = discount_monthly_curve(basis, proj.n_time)
    if basis.settlement_pattern is None:
        lic_path = np.zeros((mortality_cf.shape[0], proj.n_time + 1))
    else:
        lic_path = _settlement_lic(mortality_cf + morbidity_cf, basis.settlement_pattern)
        # Claims are paid over the pattern, not at incurrence -- discount
        # them to their payment dates in the fulfilment cash flows. With a
        # discount curve we use the in-year scalar (paragraph 40 / B71 -- the
        # rate at the month of incurrence is the right reference); the
        # full-curve treatment would require a time-varying settlement
        # factor inside the kernel, deferred.
        factor = _settlement_factor(basis.settlement_pattern, basis.discount_monthly)
        mortality_cf = mortality_cf * factor
        morbidity_cf = morbidity_cf * factor
    discount_factor_bom, discount_factor_mid = discount_factors_from_curve(discount_monthly)

    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _roll_forward_kernel(
        mortality_cf, morbidity_cf, proj.disability_cf, proj.expense_cf,
        proj.premium_cf, proj.annuity_cf, proj.maturity_cf, proj.surrender_cf,
        model_points.contract_boundary_months, discount_monthly,
    )
    if proj.account is not None:
        # Universal-life account-backed measurement. The BEL nets the account
        # value the entity holds (fund) -- premium is the lone gross inflow
        # (counted once in the roll-forward), and the account it builds is held
        # as fund and subtracted ONCE post-PV. The RA prices the mortality risk
        # on the NET AMOUNT AT RISK (the death benefit above the account) plus
        # expense risk, bypassing the slot-RA machinery (which hard-raises on
        # expense_cv and would price mortality on the full death benefit).
        bel = bel - proj.account.fund
        ra = _account_risk_adjustment(model_points, basis, proj, discount_monthly)
    else:
        pv_survival_ra = pv_survival
        ac = proj.annuity_certain_cf
        if ac is not None and np.any(ac != 0.0):
            # The guaranteed (certain) annuity payments are paid regardless of
            # survival, so they carry no longevity risk -- remove their PV from
            # the survival PV that feeds the longevity RA (the BEL still includes
            # them via the full annuity_cf). A second roll-forward over the
            # certain stream alone (everything else zero) yields its PV in the
            # pv_survival slot, with the kernel's exact start-of-month discount.
            zero = np.zeros_like(ac)
            zero_mat = np.zeros(ac.shape[0])
            _, _, _, _, pv_certain = _roll_forward_kernel(
                zero, zero, zero, zero, zero, ac, zero_mat, zero,
                model_points.contract_boundary_months, discount_monthly,
            )
            pv_survival_ra = pv_survival - pv_certain
        ra = _risk_adjustment(basis, pv_claims, pv_morbidity, pv_disability,
                              pv_survival_ra, discount_monthly)
    rbs = sar = transitions = None
    if need_state:
        # The settlement-adjusted claim arrays (mortality_cf / morbidity_cf) are
        # the ones fed to the roll-forward above, so the per-state values
        # reconcile to this exact bel.
        rbs, occ = _state_reserve(model_points, proj, mortality_cf, morbidity_cf,
                                  bel, discount_monthly)
        if sum_at_risk:
            sar, transitions = _sum_at_risk(model_points, proj, rbs, occ)
    return ValuedProjection(
        bel_path=bel,
        ra_path=ra,
        lic_path=lic_path,
        discount_factor_bom=discount_factor_bom,
        discount_factor_mid=discount_factor_mid,
        discount_monthly=discount_monthly,
        cashflows=proj,
        state_reserve=rbs,
        sum_at_risk=sar,
        transitions=transitions,
    )
