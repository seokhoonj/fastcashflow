"""VFA validation -- the Variable Fee Approach for account-value contracts.

The account value grows at the underlying-items return less the variable
fee. The benefit on every exit is the account value, so the entity's profit
is the variable fee it keeps -- which is the inception CSM.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPointSet, measure_vfa

Q = 0.002          # flat monthly mortality
LAPSE = 0.004      # flat monthly lapse


def _assumptions(**overrides) -> Assumptions:
    base = dict(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, Q),
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
    assert np.isclose(res.bel[0], bel)
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
