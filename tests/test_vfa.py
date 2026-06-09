"""VFA validation -- the Variable Fee Approach for account-value contracts.

The account value grows at the underlying-items return less the variable
fee. The benefit on every exit is the account value, so the entity's profit
is the variable fee it keeps -- which is the inception CSM.
"""
import fastcashflow as fcf
import numpy as np
import pytest

from fastcashflow import ExpenseItem, ModelPoints, report
from conftest import annual_from_monthly as _annual, make_death_basis


Q = 0.002          # flat monthly mortality
LAPSE = 0.004      # flat monthly lapse


def _basis(**overrides):
    kw = dict(
        mortality_q       = Q,
        lapse_q           = LAPSE,
        discount_annual   = 0.03,
        ra_confidence     = 0.75,
        mortality_cv      = 0.10,
        investment_return = 0.06,
        fund_fee          = 0.015,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


def test_vfa_account_value_and_csm_hand_calc():
    """Account value grows at (1+r)(1-f); CSM is the entity's unearned fee."""
    basis = _basis()
    av0, term = 1e8, 60
    res = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0), basis
    )

    r_m = 1.06 ** (1 / 12) - 1
    f_m = 1.015 ** (1 / 12) - 1
    growth = (1 + r_m) * (1 - f_m)
    assert np.allclose(res.account_value_path[0], av0 * growth ** np.arange(term + 1))

    # Mid-month decrements (deaths + lapses during month t) pay the
    # start-of-month account value av[t]; maturity survivors reach time term and
    # are paid the matured value av[term] (one more month of growth). Benefits
    # discount at r, which with the account-value growth collapses to (1 - f)^t.
    surv = (1 - Q) * (1 - LAPSE)
    t = np.arange(term)
    decrements = surv ** t * (1 - surv)             # deaths + lapses in month t -> av[t]@t
    maturity = surv ** term                          # survivors -> matured av[term]@term
    pv_benefits = av0 * (np.sum(decrements * (1 - f_m) ** t)
                         + maturity * (1 - f_m) ** term)
    bel = pv_benefits - av0
    assert np.isclose(res.bel_path[0, 0], bel)
    assert np.isclose(res.csm_path[0, 0], max(0.0, -bel))


def test_gmdb_floor_on_death_hand_calc():
    """A GMDB lifts each death payout from the account value to the floor.

    With zero return and zero fee the account value stays flat at ``av0`` and
    r=0 means no discounting, so the BEL increase from the guarantee is the
    total death decrement times the per-death excess ``(gmdb - av0)``.
    Surrender and maturity exits are unaffected (still pay the account value).
    """
    basis = _basis(investment_return=0.0, fund_fee=0.0)
    av0, gmdb, term = 1000.0, 1200.0, 60
    base = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0), basis
    )
    floored = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           minimum_death_benefit=gmdb), basis
    )
    surv = (1 - Q) * (1 - LAPSE)
    deaths = surv ** np.arange(term) * Q             # monthly death decrement
    expected_delta = deaths.sum() * (gmdb - av0)
    assert np.isclose(floored.bel_path[0, 0] - base.bel_path[0, 0], expected_delta)

    # A floor below the account value never bites -- max(AV, gmdb) == AV.
    low = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           minimum_death_benefit=500.0), basis
    )
    assert np.isclose(low.bel_path[0, 0], base.bel_path[0, 0])


def test_gmab_floor_at_maturity_hand_calc():
    """A GMAB lifts the maturity payout from the account value to the floor.

    With zero return and zero fee the account value stays flat at ``av0`` and
    r=0 means no discounting, so the BEL increase from the guarantee is the
    in-force surviving to term times the per-survivor excess ``(gmab - av0)``.
    Death and surrender exits are unaffected (still pay the account value).
    """
    basis = _basis(investment_return=0.0, fund_fee=0.0)
    av0, gmab, term = 1000.0, 1200.0, 60
    base = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0), basis
    )
    floored = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           minimum_accumulation_benefit=gmab), basis
    )
    surv = (1 - Q) * (1 - LAPSE)
    maturity_survivors = surv ** term                # in-force reaching term
    expected_delta = maturity_survivors * (gmab - av0)
    assert np.isclose(floored.bel_path[0, 0] - base.bel_path[0, 0], expected_delta)

    # A floor below the account value never bites -- max(AV, gmab) == AV.
    low = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           minimum_accumulation_benefit=500.0), basis
    )
    assert np.isclose(low.bel_path[0, 0], base.bel_path[0, 0])


def test_gmab_floor_strikes_the_matured_account_value_hand_calc():
    """The GMAB floors the *matured* account value (after the final month's
    growth), paid at time ``term`` -- not the value one month earlier.

    The flat-account test above (zero return, zero fee) cannot see this: there
    ``av[term-1] == av[term]`` and r=0 means no discounting. Here a non-zero
    return and fee make ``av[term-1] != av[term]``, so the guarantee excess must
    strike ``av[term]`` and discount to time ``term``. The GMAB-vs-no-GMAB BEL
    delta isolates the guarantee cost: the account-value payout, fee, deaths,
    expenses and RA are identical in both runs and cancel.
    """
    r, f = 0.06, 0.012
    basis = _basis(investment_return=r, fund_fee=f)
    av0, gmab, term = 1000.0, 1200.0, 24
    base = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0), basis
    )
    floored = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           minimum_accumulation_benefit=gmab), basis
    )

    r_m = (1 + r) ** (1 / 12) - 1
    f_m = (1 + f) ** (1 / 12) - 1
    growth = (1 + r_m) * (1 - f_m)
    av_term = av0 * growth ** term                    # matured value at time term
    surv = (1 - Q) * (1 - LAPSE)
    maturity_survivors = surv ** term                 # in-force reaching term
    # excess struck on the matured value, discounted to maturity (time term)
    expected_delta = maturity_survivors * (gmab - av_term) * (1 + r_m) ** (-term)
    delta = floored.bel_path[0, 0] - base.bel_path[0, 0]
    assert np.isclose(delta, expected_delta)

    # The off-by-one would strike av[term-1] and discount to time term-1;
    # confirm the engine reports the matured-value figure, not that one.
    av_prev = av0 * growth ** (term - 1)
    wrong_delta = maturity_survivors * (gmab - av_prev) * (1 + r_m) ** (-(term - 1))
    assert not np.isclose(delta, wrong_delta)


def test_gmab_binding_pays_exactly_the_guarantee_at_maturity_hand_calc():
    """A binding GMAB pays the maturity survivor exactly ``gmab`` at time term.

    The base account-value payout (the matured av[term]) and the floor top-up
    share one maturity date and value, so they sum to max(av[term], gmab) with
    no one-month gap. With no decrements a single survivor reaches term, so the
    BEL is the PV of paying ``gmab`` at time term less the account value held.
    """
    r, f = 0.06, 0.012
    basis = _basis(mortality_q=0.0, lapse_q=0.0, mortality_cv=0.0,
                   investment_return=r, fund_fee=f)
    av0, gmab, term = 1000.0, 2000.0, 24       # gmab far above any av -> binds
    res = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           minimum_accumulation_benefit=gmab), basis)
    r_m = (1 + r) ** (1 / 12) - 1
    bel = gmab * (1 + r_m) ** (-term) - av0     # exactly gmab @ time term, less the fund
    assert np.isclose(res.bel_path[0, 0], bel)


def test_gmab_lic_uses_the_nominal_top_up_not_the_discounted_pv():
    """Under a settlement pattern the GMAB top-up enters the LIC at its nominal
    incurred amount, not the present-value figure the BEL path uses.

    The BEL discounts the term-1-column top-up the extra month to time term, but
    that discount must not leak into the LIC, which settles the *incurred*
    benefit undiscounted. Regression for the benefit_cf / LIC coupling.
    """
    from dataclasses import replace
    r, f = 0.06, 0.012
    pattern = np.array([0.6, 0.4])
    basis = replace(_basis(investment_return=r, fund_fee=f),
                    settlement_pattern=pattern)
    av0, gmab, term = 1000.0, 1200.0, 24
    base = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0), basis)
    floored = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           minimum_accumulation_benefit=gmab), basis)
    delta_lic = floored.lic[0] - base.lic[0]

    r_m = (1 + r) ** (1 / 12) - 1
    f_m = (1 + f) ** (1 / 12) - 1
    growth = (1 + r_m) * (1 - f_m)
    av_term = av0 * growth ** term
    surv = (1 - Q) * (1 - LAPSE)
    nominal_excess = surv ** term * (gmab - av_term)
    # LIC right after incurral (end of the maturity month, index term): the
    # incurred top-up less the first settlement instalment, undiscounted.
    expected_jump = nominal_excess * (1 - pattern[0])
    assert np.isclose(delta_lic[term], expected_jump)
    # The pre-fix code stored the PV-discounted top-up in benefit_cf, which would
    # understate this LIC jump by 1 / (1 + r_m).
    assert not np.isclose(delta_lic[term], expected_jump / (1 + r_m))


def test_floor_tvog_zero_under_flat_scenarios():
    """A flat scenario set (every path = the central return) adds no TVOG.

    With no return volatility the mean cost equals the central (intrinsic)
    cost, so the GMDB/GMAB floor time value is zero and the measurement
    matches the deterministic run.
    """
    basis = _basis(investment_return=0.04, fund_fee=0.01)
    av0, term = 1000.0, 36
    mp = ModelPoints.single(40, 0.0, term, account_value=av0,
                            minimum_death_benefit=1100.0,
                            minimum_accumulation_benefit=1100.0)
    deterministic = fcf.vfa.measure(mp, basis)
    r_m = 1.04 ** (1 / 12) - 1
    flat = np.full((8, term), r_m)
    stochastic = fcf.vfa.measure(mp, basis, return_scenarios=flat)
    assert np.allclose(stochastic.time_value, 0.0, atol=1e-6)
    assert np.isclose(stochastic.bel_path[0, 0], deterministic.bel_path[0, 0])


def test_floor_tvog_matches_independent_reimplementation():
    """The GMDB/GMAB floor time value equals an explicit per-scenario reimpl.

    The floors are put options on the account value; their time value is the
    mean put cost over the scenarios less the central put cost. Re-derive that
    with a plain scenario loop and check the engine agrees. (The sign is not
    constrained: discounting at the underlying return -- the VFA basis, not a
    risk-neutral measure -- lets a deep in-the-money floor have negative time
    value, since volatility mostly lets scenarios escape the floor here.)
    """
    from fastcashflow.projection import project_cashflows
    from fastcashflow.tvog import guarantee_floor_time_value

    basis = _basis(investment_return=0.04, fund_fee=0.0)
    av0, gmdb, gmab, term = 1000.0, 1100.0, 1100.0, 24
    mp = ModelPoints.single(40, 0.0, term, account_value=av0,
                            minimum_death_benefit=gmdb,
                            minimum_accumulation_benefit=gmab)
    proj = project_cashflows(mp, basis)
    deaths, ms = proj.deaths[0], float(proj.maturity_survivors[0])

    rng = np.random.default_rng(0)
    r_m = 1.04 ** (1 / 12) - 1
    scen = r_m + 0.02 * rng.standard_normal((300, term))

    tv = guarantee_floor_time_value(
        account_value=mp.account_value, deaths=proj.deaths,
        maturity_survivors=proj.maturity_survivors,
        term_index=mp.term_months - 1,
        minimum_death_benefit=mp.minimum_death_benefit,
        minimum_accumulation_benefit=mp.minimum_accumulation_benefit,
        minimum_crediting_rate=0.0, fund_fee=0.0, investment_return=0.04,
        return_scenarios=scen,
    )

    def put_cost(returns):
        credit = np.maximum(returns, 0.0)          # g_credit = 0
        a = np.empty(term); a[0] = 1.0
        a[1:] = np.cumprod((1 + credit))[:-1]      # fee = 0
        d = np.empty(term); d[0] = 1.0
        d[1:] = np.cumprod(1.0 / (1 + returns))[:-1]
        av = av0 * a
        c = (deaths * np.maximum(0.0, gmdb - av) * d).sum()
        # GMAB strikes the matured value at time term -- one month past the
        # width-term path (the final month's growth / discount applied).
        a_term = a[term - 1] * (1 + credit[term - 1])
        d_term = d[term - 1] / (1 + returns[term - 1])
        c += ms * max(0.0, gmab - av0 * a_term) * d_term
        return c

    cost_s = np.array([put_cost(scen[s]) for s in range(scen.shape[0])])
    expected = cost_s.mean() - put_cost(np.full(term, r_m))
    assert np.isclose(tv[0], expected)
    assert not np.isclose(tv[0], 0.0)               # the floor does real work


def test_vfa_zero_fee_gives_no_profit():
    """With no variable fee the contract is a pure pass-through -- no CSM."""
    res = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, 60, account_value=1e8),
        _basis(fund_fee=0.0),
    )
    assert np.isclose(res.csm_path[0, 0], 0.0, atol=1.0)   # ~0 vs a 1e8 contract
    assert np.isclose(res.variable_fee[0], 0.0)


def test_vfa_csm_releases_over_the_term():
    """The CSM builds at inception and releases to zero over the term."""
    res = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, 120, account_value=1e8), _basis()
    )
    assert res.csm_path[0, 0] > 0.0
    assert np.isclose(res.csm_path[0, -1], 0.0)
    step = res.csm_path[0, :-1] + res.csm_accretion[0] - res.csm_release[0]
    assert np.allclose(step, res.csm_path[0, 1:])


def test_vfa_variable_fee_scales_with_the_fee():
    """A larger fund fee leaves the entity a larger variable fee and CSM."""
    small = fcf.vfa.measure(ModelPoints.single(40, 0.0, 60, account_value=1e8),
                        _basis(fund_fee=0.01))
    large = fcf.vfa.measure(ModelPoints.single(40, 0.0, 60, account_value=1e8),
                        _basis(fund_fee=0.03))
    assert large.variable_fee[0] > small.variable_fee[0] > 0.0
    assert large.csm_path[0, 0] > small.csm_path[0, 0] > 0.0


def test_vfa_onerous_when_expenses_exceed_the_fee():
    """Heavy acquisition expense makes the contract onerous."""
    profitable = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, 60, account_value=1e8), _basis()
    )
    onerous = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, 60, account_value=1e8),
        _basis(expense_items=(
            ExpenseItem("acquisition", "alpha_fixed", 10_000_000.0),
        )),
    )
    assert np.isclose(profitable.loss_component[0], 0.0)
    assert onerous.loss_component[0] > 0.0


def _return_paths(annual: float, vol: float, n: int, n_time: int, seed: int):
    """N monthly return paths centred on the central monthly return."""
    r_m = (1.0 + annual) ** (1.0 / 12.0) - 1.0
    return r_m + np.random.default_rng(seed).normal(0.0, vol, size=(n, n_time))


def test_vfa_tvog_folds_into_bel_and_reduces_csm():
    """Return scenarios fold the guarantee's time value into the BEL."""
    basis = _basis(investment_return=0.05)
    mp = ModelPoints.single(40, 0.0, 120,
                             account_value=1e8, minimum_crediting_rate=0.05)
    scenarios = _return_paths(0.05, vol=0.008, n=2000, n_time=120, seed=7)

    plain = fcf.vfa.measure(mp, basis)
    stoch = fcf.vfa.measure(mp, basis, scenarios)
    assert np.allclose(plain.time_value, 0.0)          # no scenarios -> no TVOG
    assert stoch.time_value[0] > 0.0
    # the TVOG raises the liability -- it is carried in time_value
    assert (stoch.bel_path[0, 0] + stoch.time_value[0]
            > plain.bel_path[0, 0] + plain.time_value[0])
    assert stoch.csm_path[0, 0] < plain.csm_path[0, 0]           # the CSM absorbs it


def test_vfa_large_tvog_turns_the_contract_onerous():
    """A guarantee time value beyond the unearned fee makes the contract onerous."""
    basis = _basis(investment_return=0.05)
    mp = ModelPoints.single(40, 0.0, 120,
                             account_value=1e8, minimum_crediting_rate=0.05)
    scenarios = _return_paths(0.05, vol=0.03, n=2000, n_time=120, seed=8)

    plain = fcf.vfa.measure(mp, basis)
    stoch = fcf.vfa.measure(mp, basis, scenarios)
    assert np.isclose(plain.loss_component[0], 0.0)
    assert stoch.loss_component[0] > 0.0
    assert np.isclose(stoch.csm_path[0, 0], 0.0)


def test_vfa_tvog_matches_measure_tvog():
    """The TVOG folded into measure_vfa equals the stand-alone measure_tvog."""
    basis = _basis(investment_return=0.04)
    mp = ModelPoints.single(40, 0.0, 120,
                             account_value=1e8, minimum_crediting_rate=0.045)
    scenarios = _return_paths(0.04, vol=0.012, n=1500, n_time=120, seed=9)

    folded = fcf.vfa.measure(mp, basis, scenarios).time_value.sum()
    standalone = fcf.vfa.tvog(mp, basis, scenarios).time_value
    assert np.isclose(folded, standalone)


def test_vfa_scenarios_with_per_mp_varying_guarantee_is_rejected():
    """Per-MP varying minimum_crediting_rate with stochastic return scenarios
    is not supported in v1 -- the time-value pass is portfolio-level."""
    basis = _basis(investment_return=0.04)
    mp = ModelPoints(
        issue_age=np.array([40, 45]),
        premium=np.array([0.0, 0.0]),
        term_months=np.array([120, 120]),
        account_value=np.array([1e8, 1e8]),
        minimum_crediting_rate=np.array([0.04, 0.05]),
    )
    with pytest.raises(NotImplementedError, match="per-MP varying"):
        fcf.vfa.measure(mp, basis, np.full((10, 120), 0.003))


def test_vfa_rejects_degenerate_return_scenarios():
    """Empty, non-finite, or <= -100% monthly returns are rejected at both
    stochastic entry points -- they would otherwise produce NaN CSM or a
    sign-flipped discount silently."""
    term = 60
    mp = ModelPoints.single(40, 0.0, term, account_value=1e8,
                            minimum_crediting_rate=0.02)
    basis = _basis(investment_return=0.04)

    empty = np.empty((0, term))
    non_finite = np.full((4, term), np.inf)
    ruin = np.full((4, term), -1.0)              # a -100% monthly return

    for scen in (empty, non_finite, ruin):
        with pytest.raises(ValueError):
            fcf.vfa.measure(mp, basis, return_scenarios=scen)
        with pytest.raises(ValueError):
            fcf.vfa.tvog(mp, basis, scen)


def test_tvog_boundary_cut_contract_does_not_index_out_of_bounds():
    """A contract whose term runs past the scenario horizon (a boundary cut)
    has no maturity, but the GMAB maturity index must stay in range. Regression
    for the stochastic-path out-of-bounds read on the GMAB column.
    """
    basis = _basis(investment_return=0.04, fund_fee=0.0)
    mp = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([0.0]),
        term_months=np.array([24]),
        contract_boundary_months=np.array([12], dtype=np.int64),
        account_value=np.array([1000.0]),
        minimum_crediting_rate=np.array([0.02]),
        minimum_accumulation_benefit=np.array([1100.0]),
    )
    r_m = 1.04 ** (1 / 12) - 1
    scen = np.full((4, 12), r_m)        # width = horizon (the 12-month boundary)
    res = fcf.vfa.measure(mp, basis, return_scenarios=scen)
    # No maturity within the boundary -> the GMAB adds no time value, but the
    # call must complete without an out-of-bounds read.
    assert np.isfinite(res.time_value).all()
    assert np.isclose(res.time_value[0], 0.0, atol=1e-6)


def test_vfa_ra_zero_without_expense_cv():
    """With no expense_cv the VFA RA is zero -- the v1 default."""
    res = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, 60, account_value=1e8),
        _basis(expense_items=(
            ExpenseItem("maintenance", "gamma_fixed", 120_000.0),
        )),
    )
    assert np.allclose(res.ra, 0.0)


def test_vfa_ra_scales_with_expense_cv():
    """The VFA RA is a confidence-level margin linear in the expense CV."""
    mp = ModelPoints.single(40, 0.0, 60, account_value=1e8)
    _g120k = (ExpenseItem("maintenance", "gamma_fixed", 120_000.0),)
    r1 = fcf.vfa.measure(mp, _basis(expense_items=_g120k, expense_cv=0.10))
    r2 = fcf.vfa.measure(mp, _basis(expense_items=_g120k, expense_cv=0.20))
    assert r1.ra_path[0, 0] > 0.0
    assert np.isclose(r2.ra_path[0, 0], 2.0 * r1.ra_path[0, 0])


def test_vfa_ra_reduces_the_csm():
    """The RA is part of the fulfilment cash flows, so it reduces the CSM."""
    mp = ModelPoints.single(40, 0.0, 60, account_value=1e8)
    _g120k = (ExpenseItem("maintenance", "gamma_fixed", 120_000.0),)
    no_ra = fcf.vfa.measure(mp, _basis(expense_items=_g120k, expense_cv=0.0))
    with_ra = fcf.vfa.measure(mp, _basis(expense_items=_g120k, expense_cv=0.30))
    assert with_ra.csm_path[0, 0] < no_ra.csm_path[0, 0]


def test_load_sample_vfa_is_measurable():
    """The bundled VFA sample measures, and its uniform credit rate lets the
    stochastic time-value pass run."""
    mp = fcf.samples.model_points(template="vfa")
    basis = fcf.samples.basis(template="vfa")
    m = fcf.vfa.measure(mp, basis)
    assert m.csm_path[:, 0].sum() > 0.0          # the variable fee is unearned profit
    assert np.allclose(m.loss_component, 0.0)

    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    scen = r_m + np.random.default_rng(0).normal(
        0.0, 0.01, size=(64, int(mp.term_months.max())))
    tvog = fcf.vfa.tvog(mp, basis, scen)
    assert tvog.time_value != 0.0           # the guarantees carry a time value


def test_vfa_report_releases_the_ra_into_revenue():
    """The report releases the VFA RA into insurance revenue."""
    basis = _basis(expense_items=(
        ExpenseItem("maintenance", "gamma_fixed", 120_000.0),
    ), expense_cv=0.25)
    m = fcf.vfa.measure(ModelPoints.single(40, 0.0, 60, account_value=1e8), basis)
    rep = report(m)
    ra_in_revenue = (rep.insurance_revenue - rep.insurance_service_expense
                     - m.csm_release)
    assert np.isclose(ra_in_revenue[0].sum(), m.ra_path[0, 0])


# ---------------------------------------------------------------------------
# full=False headline contract (the chunked-portfolio building block) + guards
# ---------------------------------------------------------------------------
def _vfa_mp():
    return ModelPoints.single(40, 0.0, 60, account_value=1e8)


def test_vfa_full_false_matches_full_headline():
    """full=False fills the same headline (bel / ra / csm / variable_fee /
    time_value / loss_component) as full=True and leaves the trajectories,
    account value and cash flows None."""
    basis = _basis()
    mp = _vfa_mp()
    full = fcf.vfa.measure(mp, basis)
    head = fcf.vfa.measure(mp, basis, full=False)
    for f in ("bel", "ra", "csm", "variable_fee", "time_value", "loss_component"):
        assert np.allclose(getattr(head, f), getattr(full, f)), f
    assert head.bel_path is None and head.ra_path is None and head.csm_path is None
    assert head.account_value_path is None and head.csm_accretion is None
    assert head.lic is None and head.cashflows is None


def test_vfa_headline_only_rejected_by_consumers():
    """A headline-only VFA measurement gives a clear error in group / roll /
    report rather than crashing on a None trajectory."""
    head = fcf.vfa.measure(_vfa_mp(), _basis(), full=False)
    with pytest.raises(ValueError, match="full=True"):
        fcf.roll_forward(head)
    with pytest.raises(ValueError, match="full=True"):
        fcf.report(head)
    with pytest.raises(ValueError, match="full measurement"):
        fcf.group(head, np.zeros(1, dtype=int))


def test_vfa_measure_inforce_reproduces_inception_slice_when_av_is_modelled():
    """VFA in-force consistency anchor: when the observed account value at the
    valuation date equals the modelled value (av0 * growth^em) and the carried
    prior_csm is the inception csm0, the in-force BEL is the inception trajectory
    slice re-based by count / inforce[em], and the carried CSM (accreted at the
    underlying return + coverage-unit release from t=0) equals the inception
    csm_path[em]. This pins the re-anchor + re-base + return-accretion carry."""
    import fastcashflow as fcf

    basis = _basis()
    av0, term, em = 1e8, 60, 12
    inc = fcf.vfa.measure(ModelPoints.single(40, 0.0, term, account_value=av0), basis)

    av_modelled = inc.account_value_path[0, em]      # observed == modelled here
    state = fcf.InforceState(
        mp_id=np.array(["X"]), elapsed_months=np.array([em]),
        count=np.array([1.0]), prior_csm=np.array([inc.csm_path[0, 0]]),
        lock_in_rate=0.0, account_value=np.array([av_modelled]))
    mp = fcf.apply_inforce_state(
        ModelPoints(issue_age=np.array([40]), premium=np.array([0.0]),
                    term_months=np.array([60]), account_value=np.array([av0]),
                    mp_id=np.array(["X"])),
        state)

    v = fcf.vfa.measure_inforce(mp, state, basis, period_months=12)
    rescale = 1.0 / inc.cashflows.inforce[0, em]
    assert np.isclose(v.bel[0], inc.bel_path[0, em] * rescale)
    assert np.isclose(v.csm[0], inc.csm_path[0, em])      # carried csm0 -> csm_path[em]
    assert np.isclose(v.loss_component[0], 0.0)           # deferred
    assert np.isclose(v.time_value[0], 0.0)               # intrinsic only


def test_vfa_measure_inforce_uses_observed_account_value():
    """A higher observed fund than the modelled one raises the in-force variable
    fee (fee is a share of the fund) -- confirming the observed AV, not the
    modelled path, drives the result."""
    import fastcashflow as fcf

    basis = _basis()
    av0, term, em = 1e8, 60, 12
    inc = fcf.vfa.measure(ModelPoints.single(40, 0.0, term, account_value=av0), basis)
    av_modelled = inc.account_value_path[0, em]
    mp0 = ModelPoints(issue_age=np.array([40]), premium=np.array([0.0]),
                      term_months=np.array([60]), account_value=np.array([av0]),
                      mp_id=np.array(["X"]))

    def run(obs):
        state = fcf.InforceState(
            mp_id=np.array(["X"]), elapsed_months=np.array([em]),
            count=np.array([1.0]), prior_csm=np.array([inc.csm_path[0, 0]]),
            lock_in_rate=0.0, account_value=np.array([obs]))
        return fcf.vfa.measure_inforce(fcf.apply_inforce_state(mp0, state), state, basis)

    base = run(av_modelled)
    higher = run(av_modelled * 1.5)
    assert higher.variable_fee[0] > base.variable_fee[0]


def test_vfa_measure_inforce_requires_account_value():
    """A VFA in-force needs the observed fund value; a state without
    account_value is rejected (GMM/PAA states have None)."""
    import fastcashflow as fcf
    basis = _basis()
    mp0 = ModelPoints(issue_age=np.array([40]), premium=np.array([0.0]),
                      term_months=np.array([60]), account_value=np.array([1e8]),
                      mp_id=np.array(["X"]))
    state = fcf.InforceState(
        mp_id=np.array(["X"]), elapsed_months=np.array([12]),
        count=np.array([1.0]), prior_csm=np.array([0.0]), lock_in_rate=0.0)
    with pytest.raises(ValueError, match="account_value"):
        fcf.vfa.measure_inforce(fcf.apply_inforce_state(mp0, state), state, basis)


def test_vfa_measure_inforce_nonzero_prior_t_carries_inception_csm():
    """Nonzero prior_t (em=24, period=12 -> prior_t=12): carrying the inception
    closing CSM at month 12 forward one period (accrete at the return + release
    by coverage units) reproduces the inception csm_path[24]. Pins the
    inforce-segment offset (the carry must start at prior_t, not em)."""
    import fastcashflow as fcf

    basis = _basis()
    av0, term, em, period = 1e8, 60, 24, 12
    inc = fcf.vfa.measure(ModelPoints.single(40, 0.0, term, account_value=av0), basis)
    mp0 = ModelPoints(issue_age=np.array([40]), premium=np.array([0.0]),
                      term_months=np.array([60]), account_value=np.array([av0]),
                      mp_id=np.array(["X"]))
    state = fcf.InforceState(
        mp_id=np.array(["X"]), elapsed_months=np.array([em]),
        count=np.array([1.0]), prior_csm=np.array([inc.csm_path[0, em - period]]),
        lock_in_rate=0.0, account_value=np.array([inc.account_value_path[0, em]]))
    v = fcf.vfa.measure_inforce(fcf.apply_inforce_state(mp0, state), state, basis,
                                period_months=period)
    assert np.isclose(v.csm[0], inc.csm_path[0, em])
    assert np.isclose(v.bel[0], inc.bel_path[0, em] / inc.cashflows.inforce[0, em])


def test_vfa_measure_inforce_bel_scales_with_count():
    """The re-base uses count / inforce[em]: doubling the as-of count doubles the
    BEL (and the variable fee), while the carried CSM -- an absolute amount in
    state.prior_csm, released by a count-invariant coverage-unit fraction -- is
    unchanged."""
    import fastcashflow as fcf

    basis = _basis()
    av0, term, em = 1e8, 60, 12
    inc = fcf.vfa.measure(ModelPoints.single(40, 0.0, term, account_value=av0), basis)
    mp0 = ModelPoints(issue_age=np.array([40]), premium=np.array([0.0]),
                      term_months=np.array([60]), account_value=np.array([av0]),
                      mp_id=np.array(["X"]))

    def run(count):
        state = fcf.InforceState(
            mp_id=np.array(["X"]), elapsed_months=np.array([em]),
            count=np.array([float(count)]), prior_csm=np.array([inc.csm_path[0, 0]]),
            lock_in_rate=0.0, account_value=np.array([inc.account_value_path[0, em]]))
        return fcf.vfa.measure_inforce(fcf.apply_inforce_state(mp0, state), state, basis)

    one, two = run(1), run(2)
    assert np.isclose(two.bel[0], 2.0 * one.bel[0])
    assert np.isclose(two.variable_fee[0], 2.0 * one.variable_fee[0])
    assert np.isclose(two.csm[0], one.csm[0])     # CSM is the absolute carried amount


def test_vfa_measure_handles_boundary_before_term():
    """A contract whose Sec. 34 boundary cuts before the term: the projection
    runs only to the boundary, the GMAB maturity (at the term, beyond the
    horizon) is not applied, and nothing indexes out of bounds (regression for
    the term_idx clamp in _vfa_project)."""
    import fastcashflow as fcf
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([0.0]),
        term_months=np.array([60]), contract_boundary_months=np.array([24]),
        account_value=np.array([1e8]),
        minimum_accumulation_benefit=np.array([1e9]))   # huge GMAB, must NOT apply
    res = fcf.vfa.measure(mp, _basis())
    assert res.bel_path.shape[1] == 25                  # n_time = boundary 24 (+1)
    assert np.all(np.isfinite(res.bel_path))


def test_vfa_measure_inforce_rejects_as_of_at_boundary():
    """An as-of date at (or beyond) a contract's own Sec. 34 boundary is rejected
    -- no remaining coverage to value -- rather than indexing a dead in-force
    column."""
    import fastcashflow as fcf
    mp0 = ModelPoints(
        issue_age=np.array([40]), premium=np.array([0.0]),
        term_months=np.array([24]), account_value=np.array([1e8]),
        mp_id=np.array(["X"]))
    state = fcf.InforceState(
        mp_id=np.array(["X"]), elapsed_months=np.array([24]),
        count=np.array([1.0]), prior_csm=np.array([0.0]), lock_in_rate=0.0,
        account_value=np.array([1e8]))
    with pytest.raises(ValueError, match="no remaining coverage|contract_boundary_months"):
        fcf.vfa.measure_inforce(fcf.apply_inforce_state(mp0, state), state, _basis())


def test_vfa_measure_inforce_mixed_book_judges_each_contract_on_its_own_boundary():
    """Per-contract horizons (not the portfolio-wide max): a short-boundary
    contract sat next to a long one is (P1) measured identically to being valued
    alone -- the long contract's horizon does not contaminate its GMAB
    eligibility -- and (P2) rejected at its own boundary even though the long
    contract makes the portfolio horizon larger."""
    import fastcashflow as fcf
    basis = _basis()
    # A: term 60 but the Sec. 34 boundary cuts at 24, with a huge GMAB that must
    # NOT apply (the maturity is past the boundary). B: a full 60-month contract,
    # so the portfolio horizon n_time = 60 > A's boundary 24.
    a_alone = ModelPoints(
        issue_age=np.array([40]), premium=np.array([0.0]),
        term_months=np.array([60]), contract_boundary_months=np.array([24]),
        account_value=np.array([1e8]),
        minimum_accumulation_benefit=np.array([1e12]))
    mixed = ModelPoints(
        issue_age=np.array([40, 40]), premium=np.array([0.0, 0.0]),
        term_months=np.array([60, 60]),
        contract_boundary_months=np.array([24, 60]),
        account_value=np.array([1e8, 1e8]),
        minimum_accumulation_benefit=np.array([1e12, 0.0]),
        mp_id=np.array(["A", "B"]))

    # (P1) A's inception measurement is identical alone and inside the mixed book
    # -- the GMAB does not leak in off B's longer horizon.
    bel_alone = fcf.vfa.measure(a_alone, basis).bel[0]
    bel_mixed_a = fcf.vfa.measure(mixed, basis).bel[0]
    assert np.isclose(bel_alone, bel_mixed_a)

    # (P2) valuing the mixed book with A at its own boundary (24) is rejected,
    # even though B makes the portfolio horizon 60.
    state = fcf.InforceState(
        mp_id=np.array(["A", "B"]), elapsed_months=np.array([24, 12]),
        count=np.array([1.0, 1.0]), prior_csm=np.array([0.0, 0.0]),
        lock_in_rate=0.0, account_value=np.array([1e8, 1e8]))
    with pytest.raises(ValueError, match="no remaining coverage|contract_boundary_months"):
        fcf.vfa.measure_inforce(fcf.apply_inforce_state(mixed, state), state, basis)


def test_vfa_measure_inforce_csm_basis_is_carry_only_and_guarded(tmp_path):
    """measure_inforce tags its result csm_basis='carry_only', and the
    accounting-output entry points (roll_forward / report / group /
    group_of_contracts / write_measurement) refuse it -- a carry-only in-force
    CSM cannot be silently consumed as a paragraph-45 settlement figure. The
    inception measurement (projected_runoff) flows through unguarded."""
    import fastcashflow as fcf
    basis = _basis()
    av0, term, em = 1e8, 60, 12
    inc = fcf.vfa.measure(ModelPoints.single(40, 0.0, term, account_value=av0), basis)
    assert inc.csm_basis == "projected_runoff"

    mp0 = ModelPoints(issue_age=np.array([40]), premium=np.array([0.0]),
                      term_months=np.array([60]), account_value=np.array([av0]),
                      mp_id=np.array(["X"]))
    state = fcf.InforceState(
        mp_id=np.array(["X"]), elapsed_months=np.array([em]),
        count=np.array([1.0]), prior_csm=np.array([inc.csm_path[0, 0]]),
        lock_in_rate=0.0, account_value=np.array([inc.account_value_path[0, em]]))
    carry = fcf.vfa.measure_inforce(fcf.apply_inforce_state(mp0, state), state, basis)
    assert carry.csm_basis == "carry_only"

    for op in (lambda: fcf.roll_forward(carry),
               lambda: fcf.report(carry),
               lambda: fcf.group(carry, by="product"),
               lambda: fcf.group_of_contracts(carry),
               lambda: fcf.write_measurement(carry, tmp_path / "carry.csv")):
        with pytest.raises(ValueError, match="carry.only"):
            op()

    # the inception measurement remains usable by the same entry points
    fcf.report(inc)
    fcf.roll_forward(inc)


def test_vfa_project_exposes_guarantee_excess_and_expense_pv():
    """_vfa_project exposes the guarantee-excess PV (G) and expense PV (E) for
    the paragraph-45 settlement movement's future-service term (c) = -(dG+dE+dRA).
    G[:,0] must equal the BEL increase from adding the GMDB (with r=0, f=0 it is
    the total death decrement times the per-death excess gmdb-av0)."""
    from fastcashflow._vfa import _vfa_project
    basis = _basis(investment_return=0.0, fund_fee=0.0)
    av0, gmdb, term = 1000.0, 1200.0, 60
    mp = ModelPoints.single(40, 0.0, term, account_value=av0,
                            minimum_death_benefit=gmdb)
    p = _vfa_project(mp, basis)
    base = fcf.vfa.measure(ModelPoints.single(40, 0.0, term, account_value=av0),
                           basis)
    floored = fcf.vfa.measure(mp, basis)
    assert np.isclose(p.guarantee_excess_pv[0, 0],
                      floored.bel_path[0, 0] - base.bel_path[0, 0])
    # hand value: total death decrement * (gmdb - av0)
    surv = (1 - Q) * (1 - LAPSE)
    deaths = surv ** np.arange(term) * Q
    assert np.isclose(p.guarantee_excess_pv[0, 0], deaths.sum() * (gmdb - av0))
    assert p.expense_pv.shape == p.bel.shape       # E exposed as a trajectory
