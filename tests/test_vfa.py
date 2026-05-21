"""VFA validation -- the Variable Fee Approach for account-value contracts.

The account value grows at the underlying-items return less the variable
fee. The benefit on every exit is the account value, so the entity's profit
is the variable fee it keeps -- which is the inception CSM.
"""
import numpy as np
import pytest

from fastcashflow import Assumptions, ModelPointSet, measure_tvog, measure_vfa, report

Q = 0.002          # flat monthly mortality
LAPSE = 0.004      # flat monthly lapse


def _assumptions(**overrides) -> Assumptions:
    base = dict(
        mortality_monthly=lambda sex, issue_age, duration: np.full(issue_age.shape, Q),
        lapse_monthly=lambda duration: np.full(duration.shape, LAPSE),
        discount_annual=0.03,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
        investment_return=0.06,
        fund_fee=0.015,
    )
    base.update(overrides)
    return Assumptions(**base)


def test_vfa_account_value_and_csm_hand_calc():
    """Account value grows at (1+r)(1-f); CSM is the entity's unearned fee."""
    asmp = _assumptions()
    av0, term = 1e8, 60
    res = measure_vfa(
        ModelPointSet.single(40, 0.0, 0.0, term, account_value=av0), asmp
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


def test_vfa_zero_fee_gives_no_profit():
    """With no variable fee the contract is a pure pass-through -- no CSM."""
    res = measure_vfa(
        ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8),
        _assumptions(fund_fee=0.0),
    )
    assert np.isclose(res.csm[0, 0], 0.0, atol=1.0)   # ~0 vs a 1e8 contract
    assert np.isclose(res.variable_fee[0], 0.0)


def test_vfa_csm_releases_over_the_term():
    """The CSM builds at inception and releases to zero over the term."""
    res = measure_vfa(
        ModelPointSet.single(40, 0.0, 0.0, 120, account_value=1e8), _assumptions()
    )
    assert res.csm[0, 0] > 0.0
    assert np.isclose(res.csm[0, -1], 0.0)
    step = res.csm[0, :-1] + res.csm_accretion[0] - res.csm_release[0]
    assert np.allclose(step, res.csm[0, 1:])


def test_vfa_variable_fee_scales_with_the_fee():
    """A larger fund fee leaves the entity a larger variable fee and CSM."""
    small = measure_vfa(ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8),
                        _assumptions(fund_fee=0.01))
    large = measure_vfa(ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8),
                        _assumptions(fund_fee=0.03))
    assert large.variable_fee[0] > small.variable_fee[0] > 0.0
    assert large.csm[0, 0] > small.csm[0, 0] > 0.0


def test_vfa_onerous_when_expenses_exceed_the_fee():
    """Heavy acquisition expense makes the contract onerous."""
    profitable = measure_vfa(
        ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8), _assumptions()
    )
    onerous = measure_vfa(
        ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8),
        _assumptions(expense_acquisition=10_000_000.0),
    )
    assert np.isclose(profitable.loss_component[0], 0.0)
    assert onerous.loss_component[0] > 0.0


def _return_paths(annual: float, vol: float, n: int, n_time: int, seed: int):
    """N monthly return paths centred on the central monthly return."""
    r_m = (1.0 + annual) ** (1.0 / 12.0) - 1.0
    return r_m + np.random.default_rng(seed).normal(0.0, vol, size=(n, n_time))


def test_vfa_tvog_folds_into_bel_and_reduces_csm():
    """Return scenarios fold the guarantee's time value into the BEL."""
    asmp = _assumptions(investment_return=0.05, guaranteed_credit_rate=0.05)
    mp = ModelPointSet.single(40, 0.0, 0.0, 120, account_value=1e8)
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
    asmp = _assumptions(investment_return=0.05, guaranteed_credit_rate=0.05)
    mp = ModelPointSet.single(40, 0.0, 0.0, 120, account_value=1e8)
    scenarios = _return_paths(0.05, vol=0.03, n=2000, n_time=120, seed=8)

    plain = measure_vfa(mp, asmp)
    stoch = measure_vfa(mp, asmp, scenarios)
    assert np.isclose(plain.loss_component[0], 0.0)
    assert stoch.loss_component[0] > 0.0
    assert np.isclose(stoch.csm[0, 0], 0.0)


def test_vfa_tvog_matches_measure_tvog():
    """The TVOG folded into measure_vfa equals the stand-alone measure_tvog."""
    asmp = _assumptions(investment_return=0.04, guaranteed_credit_rate=0.045)
    mp = ModelPointSet.single(40, 0.0, 0.0, 120, account_value=1e8)
    scenarios = _return_paths(0.04, vol=0.012, n=1500, n_time=120, seed=9)

    folded = measure_vfa(mp, asmp, scenarios).time_value.sum()
    standalone = measure_tvog(mp, asmp, scenarios).time_value
    assert np.isclose(folded, standalone)


def test_vfa_scenarios_without_a_guarantee_is_rejected():
    """Passing return scenarios with no guarantee set is an error."""
    asmp = _assumptions(investment_return=0.04)        # no guaranteed_credit_rate
    mp = ModelPointSet.single(40, 0.0, 0.0, 120, account_value=1e8)
    with pytest.raises(ValueError, match="guarantee"):
        measure_vfa(mp, asmp, np.full((10, 120), 0.003))


def test_vfa_ra_zero_without_expense_cv():
    """With no expense_cv the VFA RA is zero -- the v1 default."""
    res = measure_vfa(
        ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8),
        _assumptions(expense_maintenance_annual=120_000.0),
    )
    assert np.allclose(res.ra, 0.0)


def test_vfa_ra_scales_with_expense_cv():
    """The VFA RA is a confidence-level margin linear in the expense CV."""
    mp = ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8)
    r1 = measure_vfa(mp, _assumptions(expense_maintenance_annual=120_000.0,
                                      expense_cv=0.10))
    r2 = measure_vfa(mp, _assumptions(expense_maintenance_annual=120_000.0,
                                      expense_cv=0.20))
    assert r1.ra[0, 0] > 0.0
    assert np.isclose(r2.ra[0, 0], 2.0 * r1.ra[0, 0])


def test_vfa_ra_reduces_the_csm():
    """The RA is part of the fulfilment cash flows, so it reduces the CSM."""
    mp = ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8)
    no_ra = measure_vfa(mp, _assumptions(expense_maintenance_annual=120_000.0,
                                         expense_cv=0.0))
    with_ra = measure_vfa(mp, _assumptions(expense_maintenance_annual=120_000.0,
                                           expense_cv=0.30))
    assert with_ra.csm[0, 0] < no_ra.csm[0, 0]


def test_vfa_report_releases_the_ra_into_revenue():
    """The disclosure releases the VFA RA into insurance revenue."""
    asmp = _assumptions(expense_maintenance_annual=120_000.0, expense_cv=0.25)
    m = measure_vfa(ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8), asmp)
    rep = report(m)
    ra_in_revenue = (rep.insurance_revenue - rep.insurance_service_expense
                     - m.csm_release)
    assert np.isclose(ra_in_revenue[0].sum(), m.ra[0, 0])
