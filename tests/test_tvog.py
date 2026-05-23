"""TVOG validation -- the time value of a VFA minimum-rate guarantee.

The account is credited ``max(return, guarantee)``. A deterministic run sees
only the intrinsic value of that guarantee; ``measure_tvog`` values it over
many return scenarios and recovers the time value -- the cost the convex
``max`` adds once returns are allowed to vary.
"""
import numpy as np
import pytest

from fastcashflow import Assumptions, ModelPoints, measure_tvog


def _annual(m: float) -> float:
    """Convert a monthly rate to its annual equivalent so the engine converts back."""
    return 1.0 - (1.0 - m) ** 12


def _assumptions(**overrides) -> Assumptions:
    base = dict(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.002)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.004)),
        discount_annual=0.03,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
        fund_fee=0.015,
    )
    base.update(overrides)
    return Assumptions(**base)


def _contract(term: int = 120) -> ModelPoints:
    return ModelPoints.single(40, 0.0, 0.0, term, account_value=1e8)


def _return_paths(annual_return: float, vol: float, n: int, n_time: int, seed: int):
    """N monthly return paths, centred on the central monthly return."""
    r_m = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
    rng = np.random.default_rng(seed)
    return r_m + rng.normal(0.0, vol, size=(n, n_time))


def test_tvog_positive_from_return_volatility():
    """An at-the-money guarantee has a positive time value -- the Jensen gap."""
    term = 120
    asmp = _assumptions(investment_return=0.04, guaranteed_credit_rate=0.04)
    scenarios = _return_paths(0.04, vol=0.015, n=3000, n_time=term, seed=1)
    res = measure_tvog(_contract(term), asmp, scenarios)
    assert res.time_value > 0.0
    assert res.total_value > res.intrinsic_value


def test_tvog_decomposition():
    """total_value = intrinsic value + time value = the mean guarantee cost."""
    term = 120
    asmp = _assumptions(investment_return=0.03, guaranteed_credit_rate=0.05)
    scenarios = _return_paths(0.03, vol=0.012, n=1000, n_time=term, seed=3)
    res = measure_tvog(_contract(term), asmp, scenarios)
    assert np.isclose(res.total_value, res.intrinsic_value + res.time_value)
    assert np.isclose(res.total_value, res.guarantee_cost.mean())
    # the guarantee (5%) is in the money even deterministically
    assert res.intrinsic_value > 0.0


def test_tvog_zero_when_every_scenario_is_central():
    """With no return volatility the time value vanishes; intrinsic value remains."""
    term = 120
    asmp = _assumptions(investment_return=0.02, guaranteed_credit_rate=0.05)
    r_m = (1.02) ** (1.0 / 12.0) - 1.0
    scenarios = np.full((50, term), r_m)
    res = measure_tvog(_contract(term), asmp, scenarios)
    assert np.isclose(res.time_value, 0.0, atol=1.0)   # ~0 vs a 1e8 contract
    assert res.intrinsic_value > 0.0


def test_tvog_deep_out_of_the_money_is_nearly_zero():
    """A guarantee far below every scenario return costs almost nothing."""
    term = 120
    asmp = _assumptions(investment_return=0.06, guaranteed_credit_rate=-0.20)
    scenarios = _return_paths(0.06, vol=0.005, n=500, n_time=term, seed=2)
    res = measure_tvog(_contract(term), asmp, scenarios)
    assert abs(res.total_value) < 1.0          # the guarantee never bites


def test_tvog_requires_a_guarantee():
    """measure_tvog needs guaranteed_credit_rate to be set."""
    asmp = _assumptions(investment_return=0.04, guaranteed_credit_rate=None)
    with pytest.raises(ValueError, match="guaranteed_credit_rate"):
        measure_tvog(_contract(120), asmp, np.full((10, 120), 0.003))


def test_tvog_rejects_wrong_horizon():
    """The scenario width must match the projection horizon."""
    asmp = _assumptions(investment_return=0.04, guaranteed_credit_rate=0.04)
    with pytest.raises(ValueError, match="columns"):
        measure_tvog(_contract(120), asmp, np.full((10, 7), 0.003))
