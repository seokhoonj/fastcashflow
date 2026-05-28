"""VFA validation -- the Variable Fee Approach for account-value contracts.

The account value grows at the underlying-items return less the variable
fee. The benefit on every exit is the account value, so the entity's profit
is the variable fee it keeps -- which is the inception CSM.
"""
import numpy as np
import pytest

from fastcashflow import (
    ExpenseItem, ModelPoints, load_sample_vfa_assumptions,
    load_sample_vfa_model_points, measure_tvog, measure_vfa, report,
)
from conftest import annual_from_monthly as _annual, make_death_assumptions


Q = 0.002          # flat monthly mortality
LAPSE = 0.004      # flat monthly lapse


def _assumptions(**overrides):
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
    return make_death_assumptions(**kw)


def test_vfa_account_value_and_csm_hand_calc():
    """Account value grows at (1+r)(1-f); CSM is the entity's unearned fee."""
    asmp = _assumptions()
    av0, term = 1e8, 60
    res = measure_vfa(
        ModelPoints.single(40, 0.0, term, account_value=av0), asmp
    )

    r_m = 1.06 ** (1 / 12) - 1
    f_m = 1.015 ** (1 / 12) - 1
    growth = (1 + r_m) * (1 - f_m)
    assert np.allclose(res.account_value[0], av0 * growth ** np.arange(term + 1))

    # every exit pays the account value; benefits discount at r, which with
    # the account-value growth collapses to (1 - f)^t
    surv = (1 - Q) * (1 - LAPSE)
    inforce = surv ** np.arange(term)
    exits = np.empty(term)
    exits[:-1] = inforce[:-1] - inforce[1:]
    exits[-1] = inforce[-1]
    pv_benefits = av0 * np.sum(exits * (1 - f_m) ** np.arange(term))
    bel = pv_benefits - av0
    assert np.isclose(res.bel[0, 0], bel)
    assert np.isclose(res.csm[0, 0], max(0.0, -bel))


def test_gmdb_floor_on_death_hand_calc():
    """A GMDB lifts each death payout from the account value to the floor.

    With zero return and zero fee the account value stays flat at ``av0`` and
    r=0 means no discounting, so the BEL increase from the guarantee is the
    total death decrement times the per-death excess ``(gdb - av0)``.
    Surrender and maturity exits are unaffected (still pay the account value).
    """
    asmp = _assumptions(investment_return=0.0, fund_fee=0.0)
    av0, gdb, term = 1000.0, 1200.0, 60
    base = measure_vfa(
        ModelPoints.single(40, 0.0, term, account_value=av0), asmp
    )
    floored = measure_vfa(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           guaranteed_death_benefit=gdb), asmp
    )
    surv = (1 - Q) * (1 - LAPSE)
    deaths = surv ** np.arange(term) * Q             # monthly death decrement
    expected_delta = deaths.sum() * (gdb - av0)
    assert np.isclose(floored.bel[0, 0] - base.bel[0, 0], expected_delta)

    # A floor below the account value never bites -- max(AV, gdb) == AV.
    low = measure_vfa(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           guaranteed_death_benefit=500.0), asmp
    )
    assert np.isclose(low.bel[0, 0], base.bel[0, 0])


def test_gmab_floor_at_maturity_hand_calc():
    """A GMAB lifts the maturity payout from the account value to the floor.

    With zero return and zero fee the account value stays flat at ``av0`` and
    r=0 means no discounting, so the BEL increase from the guarantee is the
    in-force surviving to term times the per-survivor excess ``(gab - av0)``.
    Death and surrender exits are unaffected (still pay the account value).
    """
    asmp = _assumptions(investment_return=0.0, fund_fee=0.0)
    av0, gab, term = 1000.0, 1200.0, 60
    base = measure_vfa(
        ModelPoints.single(40, 0.0, term, account_value=av0), asmp
    )
    floored = measure_vfa(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           guaranteed_accumulation_benefit=gab), asmp
    )
    surv = (1 - Q) * (1 - LAPSE)
    maturity_survivors = surv ** term                # in-force reaching term
    expected_delta = maturity_survivors * (gab - av0)
    assert np.isclose(floored.bel[0, 0] - base.bel[0, 0], expected_delta)

    # A floor below the account value never bites -- max(AV, gab) == AV.
    low = measure_vfa(
        ModelPoints.single(40, 0.0, term, account_value=av0,
                           guaranteed_accumulation_benefit=500.0), asmp
    )
    assert np.isclose(low.bel[0, 0], base.bel[0, 0])


def test_floor_tvog_zero_under_flat_scenarios():
    """A flat scenario set (every path = the central return) adds no TVOG.

    With no return volatility the mean cost equals the central (intrinsic)
    cost, so the GMDB/GMAB floor time value is zero and the measurement
    matches the deterministic run.
    """
    asmp = _assumptions(investment_return=0.04, fund_fee=0.01)
    av0, term = 1000.0, 36
    mp = ModelPoints.single(40, 0.0, term, account_value=av0,
                            guaranteed_death_benefit=1100.0,
                            guaranteed_accumulation_benefit=1100.0)
    deterministic = measure_vfa(mp, asmp)
    r_m = 1.04 ** (1 / 12) - 1
    flat = np.full((8, term), r_m)
    stochastic = measure_vfa(mp, asmp, return_scenarios=flat)
    assert np.allclose(stochastic.time_value, 0.0, atol=1e-6)
    assert np.isclose(stochastic.bel[0, 0], deterministic.bel[0, 0])


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

    asmp = _assumptions(investment_return=0.04, fund_fee=0.0)
    av0, gdb, gab, term = 1000.0, 1100.0, 1100.0, 24
    mp = ModelPoints.single(40, 0.0, term, account_value=av0,
                            guaranteed_death_benefit=gdb,
                            guaranteed_accumulation_benefit=gab)
    proj = project_cashflows(mp, asmp)
    deaths, ms = proj.deaths[0], float(proj.maturity_survivors[0])

    rng = np.random.default_rng(0)
    r_m = 1.04 ** (1 / 12) - 1
    scen = r_m + 0.02 * rng.standard_normal((300, term))

    tv = guarantee_floor_time_value(
        account_value=mp.account_value, deaths=proj.deaths,
        maturity_survivors=proj.maturity_survivors,
        term_index=mp.term_months - 1,
        guaranteed_death_benefit=mp.guaranteed_death_benefit,
        guaranteed_accumulation_benefit=mp.guaranteed_accumulation_benefit,
        guaranteed_credit_rate=0.0, fund_fee=0.0, investment_return=0.04,
        return_scenarios=scen,
    )

    def put_cost(returns):
        credit = np.maximum(returns, 0.0)          # g_credit = 0
        a = np.empty(term); a[0] = 1.0
        a[1:] = np.cumprod((1 + credit))[:-1]      # fee = 0
        d = np.empty(term); d[0] = 1.0
        d[1:] = np.cumprod(1.0 / (1 + returns))[:-1]
        av = av0 * a
        c = (deaths * np.maximum(0.0, gdb - av) * d).sum()
        c += ms * max(0.0, gab - av0 * a[term - 1]) * d[term - 1]
        return c

    cost_s = np.array([put_cost(scen[s]) for s in range(scen.shape[0])])
    expected = cost_s.mean() - put_cost(np.full(term, r_m))
    assert np.isclose(tv[0], expected)
    assert not np.isclose(tv[0], 0.0)               # the floor does real work


def test_vfa_zero_fee_gives_no_profit():
    """With no variable fee the contract is a pure pass-through -- no CSM."""
    res = measure_vfa(
        ModelPoints.single(40, 0.0, 60, account_value=1e8),
        _assumptions(fund_fee=0.0),
    )
    assert np.isclose(res.csm[0, 0], 0.0, atol=1.0)   # ~0 vs a 1e8 contract
    assert np.isclose(res.variable_fee[0], 0.0)


def test_vfa_csm_releases_over_the_term():
    """The CSM builds at inception and releases to zero over the term."""
    res = measure_vfa(
        ModelPoints.single(40, 0.0, 120, account_value=1e8), _assumptions()
    )
    assert res.csm[0, 0] > 0.0
    assert np.isclose(res.csm[0, -1], 0.0)
    step = res.csm[0, :-1] + res.csm_accretion[0] - res.csm_release[0]
    assert np.allclose(step, res.csm[0, 1:])


def test_vfa_variable_fee_scales_with_the_fee():
    """A larger fund fee leaves the entity a larger variable fee and CSM."""
    small = measure_vfa(ModelPoints.single(40, 0.0, 60, account_value=1e8),
                        _assumptions(fund_fee=0.01))
    large = measure_vfa(ModelPoints.single(40, 0.0, 60, account_value=1e8),
                        _assumptions(fund_fee=0.03))
    assert large.variable_fee[0] > small.variable_fee[0] > 0.0
    assert large.csm[0, 0] > small.csm[0, 0] > 0.0


def test_vfa_onerous_when_expenses_exceed_the_fee():
    """Heavy acquisition expense makes the contract onerous."""
    profitable = measure_vfa(
        ModelPoints.single(40, 0.0, 60, account_value=1e8), _assumptions()
    )
    onerous = measure_vfa(
        ModelPoints.single(40, 0.0, 60, account_value=1e8),
        _assumptions(expense_items=(
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
    asmp = _assumptions(investment_return=0.05)
    mp = ModelPoints.single(40, 0.0, 120,
                             account_value=1e8, guaranteed_credit_rate=0.05)
    scenarios = _return_paths(0.05, vol=0.008, n=2000, n_time=120, seed=7)

    plain = measure_vfa(mp, asmp)
    stoch = measure_vfa(mp, asmp, scenarios)
    assert np.allclose(plain.time_value, 0.0)          # no scenarios -> no TVOG
    assert stoch.time_value[0] > 0.0
    # the TVOG raises the liability -- it is carried in time_value
    assert (stoch.bel[0, 0] + stoch.time_value[0]
            > plain.bel[0, 0] + plain.time_value[0])
    assert stoch.csm[0, 0] < plain.csm[0, 0]           # the CSM absorbs it


def test_vfa_large_tvog_turns_the_contract_onerous():
    """A guarantee time value beyond the unearned fee makes the contract onerous."""
    asmp = _assumptions(investment_return=0.05)
    mp = ModelPoints.single(40, 0.0, 120,
                             account_value=1e8, guaranteed_credit_rate=0.05)
    scenarios = _return_paths(0.05, vol=0.03, n=2000, n_time=120, seed=8)

    plain = measure_vfa(mp, asmp)
    stoch = measure_vfa(mp, asmp, scenarios)
    assert np.isclose(plain.loss_component[0], 0.0)
    assert stoch.loss_component[0] > 0.0
    assert np.isclose(stoch.csm[0, 0], 0.0)


def test_vfa_tvog_matches_measure_tvog():
    """The TVOG folded into measure_vfa equals the stand-alone measure_tvog."""
    asmp = _assumptions(investment_return=0.04)
    mp = ModelPoints.single(40, 0.0, 120,
                             account_value=1e8, guaranteed_credit_rate=0.045)
    scenarios = _return_paths(0.04, vol=0.012, n=1500, n_time=120, seed=9)

    folded = measure_vfa(mp, asmp, scenarios).time_value.sum()
    standalone = measure_tvog(mp, asmp, scenarios).time_value
    assert np.isclose(folded, standalone)


def test_vfa_scenarios_with_per_mp_varying_guarantee_is_rejected():
    """Per-MP varying guaranteed_credit_rate with stochastic return scenarios
    is not supported in v1 -- the time-value pass is portfolio-level."""
    asmp = _assumptions(investment_return=0.04)
    mp = ModelPoints(
        issue_age=np.array([40, 45]),
        level_premium=np.array([0.0, 0.0]),
        term_months=np.array([120, 120]),
        account_value=np.array([1e8, 1e8]),
        guaranteed_credit_rate=np.array([0.04, 0.05]),
    )
    with pytest.raises(NotImplementedError, match="per-MP varying"):
        measure_vfa(mp, asmp, np.full((10, 120), 0.003))


def test_vfa_ra_zero_without_expense_cv():
    """With no expense_cv the VFA RA is zero -- the v1 default."""
    res = measure_vfa(
        ModelPoints.single(40, 0.0, 60, account_value=1e8),
        _assumptions(expense_items=(
            ExpenseItem("maintenance", "gamma_fixed", 120_000.0),
        )),
    )
    assert np.allclose(res.ra, 0.0)


def test_vfa_ra_scales_with_expense_cv():
    """The VFA RA is a confidence-level margin linear in the expense CV."""
    mp = ModelPoints.single(40, 0.0, 60, account_value=1e8)
    _g120k = (ExpenseItem("maintenance", "gamma_fixed", 120_000.0),)
    r1 = measure_vfa(mp, _assumptions(expense_items=_g120k, expense_cv=0.10))
    r2 = measure_vfa(mp, _assumptions(expense_items=_g120k, expense_cv=0.20))
    assert r1.ra[0, 0] > 0.0
    assert np.isclose(r2.ra[0, 0], 2.0 * r1.ra[0, 0])


def test_vfa_ra_reduces_the_csm():
    """The RA is part of the fulfilment cash flows, so it reduces the CSM."""
    mp = ModelPoints.single(40, 0.0, 60, account_value=1e8)
    _g120k = (ExpenseItem("maintenance", "gamma_fixed", 120_000.0),)
    no_ra = measure_vfa(mp, _assumptions(expense_items=_g120k, expense_cv=0.0))
    with_ra = measure_vfa(mp, _assumptions(expense_items=_g120k, expense_cv=0.30))
    assert with_ra.csm[0, 0] < no_ra.csm[0, 0]


def test_load_sample_vfa_is_measurable():
    """The bundled VFA sample measures, and its uniform credit rate lets the
    stochastic time-value pass run."""
    mp = load_sample_vfa_model_points()
    asmp = load_sample_vfa_assumptions()
    m = measure_vfa(mp, asmp)
    assert m.csm[:, 0].sum() > 0.0          # the variable fee is unearned profit
    assert np.allclose(m.loss_component, 0.0)

    r_m = (1.0 + asmp.investment_return) ** (1.0 / 12.0) - 1.0
    scen = r_m + np.random.default_rng(0).normal(
        0.0, 0.01, size=(64, int(mp.term_months.max())))
    tvog = measure_tvog(mp, asmp, scen)
    assert tvog.time_value != 0.0           # the guarantees carry a time value


def test_vfa_report_releases_the_ra_into_revenue():
    """The report releases the VFA RA into insurance revenue."""
    asmp = _assumptions(expense_items=(
        ExpenseItem("maintenance", "gamma_fixed", 120_000.0),
    ), expense_cv=0.25)
    m = measure_vfa(ModelPoints.single(40, 0.0, 60, account_value=1e8), asmp)
    rep = report(m)
    ra_in_revenue = (rep.insurance_revenue - rep.insurance_service_expense
                     - m.csm_release)
    assert np.isclose(ra_in_revenue[0].sum(), m.ra[0, 0])
