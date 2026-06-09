"""TVOG validation -- the time value of a VFA minimum-rate guarantee.

The account is credited ``max(return, guarantee)``. A deterministic run sees
only the intrinsic value of that guarantee; ``measure_tvog`` values it over
many return scenarios and recovers the time value -- the cost the convex
``max`` adds once returns are allowed to vary.
"""
import fastcashflow as fcf
import numpy as np
import pytest

from fastcashflow import ModelPoints
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


def _basis(**overrides):
    kw = dict(
        mortality_q     = 0.002,
        lapse_q         = 0.004,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = 0.10,
        fund_fee        = 0.015,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


def _contract(term: int = 120, g: float = 0.0) -> ModelPoints:
    return ModelPoints.single(
        40, 0.0, term, account_value=1e8, minimum_crediting_rate=g,
        calculation_methods=PATTERNS,
    )


def _return_paths(annual_return: float, vol: float, n: int, n_time: int, seed: int):
    """N monthly return paths, centred on the central monthly return."""
    r_m = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
    rng = np.random.default_rng(seed)
    return r_m + rng.normal(0.0, vol, size=(n, n_time))


def test_tvog_positive_from_return_volatility():
    """An at-the-money guarantee has a positive time value -- the Jensen gap."""
    term = 120
    basis = _basis(investment_return=0.04)
    scenarios = _return_paths(0.04, vol=0.015, n=3000, n_time=term, seed=1)
    res = fcf.vfa.tvog(_contract(term, g=0.04), basis, scenarios)
    assert res.time_value > 0.0
    assert res.total_value > res.intrinsic_value


def test_tvog_decomposition():
    """total_value = intrinsic value + time value = the mean guarantee cost."""
    term = 120
    basis = _basis(investment_return=0.03)
    scenarios = _return_paths(0.03, vol=0.012, n=1000, n_time=term, seed=3)
    res = fcf.vfa.tvog(_contract(term, g=0.05), basis, scenarios)
    assert np.isclose(res.total_value, res.intrinsic_value + res.time_value)
    assert np.isclose(res.total_value, res.guarantee_cost.mean())
    # the guarantee (5%) is in the money even deterministically
    assert res.intrinsic_value > 0.0


def test_tvog_zero_when_every_scenario_is_central():
    """With no return volatility the time value vanishes; intrinsic value remains."""
    term = 120
    basis = _basis(investment_return=0.02)
    r_m = (1.02) ** (1.0 / 12.0) - 1.0
    scenarios = np.full((50, term), r_m)
    res = fcf.vfa.tvog(_contract(term, g=0.05), basis, scenarios)
    assert np.isclose(res.time_value, 0.0, atol=1.0)   # ~0 vs a 1e8 contract
    assert res.intrinsic_value > 0.0


def test_tvog_floor_below_returns_is_nearly_zero():
    """A 0% floor the returns never breach (a high central return, tight
    scenarios) costs almost nothing -- the guarantee never bites."""
    term = 120
    basis = _basis(investment_return=0.06)
    scenarios = _return_paths(0.06, vol=0.001, n=500, n_time=term, seed=2)
    res = fcf.vfa.tvog(_contract(term, g=0.0), basis, scenarios)
    assert abs(res.total_value) < 1.0          # the floor never bites


def test_tvog_requires_a_guarantee():
    """measure_tvog rejects a contract with no crediting guarantee (the
    NO_GUARANTEE_RATE sentinel); a 0.0 rate is a real 0% floor and is valued."""
    basis = _basis(investment_return=0.04)
    scen = np.full((10, 120), 0.003)
    with pytest.raises(ValueError, match="guarantee"):
        fcf.vfa.tvog(_contract(120, g=fcf.NO_GUARANTEE_RATE), basis, scen)
    # a 0% floor is a real guarantee -- accepted, not rejected
    res = fcf.vfa.tvog(_contract(120, g=0.0), basis, scen)
    assert np.isfinite(res.total_value)


def test_tvog_rejects_wrong_horizon():
    """The scenario width must match the projection horizon."""
    basis = _basis(investment_return=0.04)
    with pytest.raises(ValueError, match="columns"):
        fcf.vfa.tvog(_contract(120, g=0.04), basis, np.full((10, 7), 0.003))


def test_tvog_weights_no_guarantee_is_exact_zero_even_at_extreme_returns():
    """With no crediting guarantee the credit-rate time value is identically
    zero, and the short-circuit returns exact zeros rather than routing through
    the over/underflowing cumulative growth/discount products (a near-ruin but
    valid return path would otherwise give 0 * inf = NaN)."""
    from fastcashflow.tvog import tvog_weights
    term = 240
    extreme = np.full((8, term), -0.95)        # valid (> -1) but near-ruin
    w = tvog_weights(minimum_crediting_rate=fcf.NO_GUARANTEE_RATE,
                     fund_fee=0.015, investment_return=0.04,
                     return_scenarios=extreme)
    assert np.all(np.isfinite(w))
    assert np.all(w == 0.0)


def test_tvog_weights_rejects_nonfinite_rate():
    """A non-finite crediting rate is rejected at the helper boundary -- the
    scalar TVOG helpers bypass the ModelPoints finite check, so the domain
    validator must reject it itself."""
    from fastcashflow.tvog import tvog_weights
    with pytest.raises(ValueError, match="finite"):
        tvog_weights(minimum_crediting_rate=np.nan, fund_fee=0.015,
                     investment_return=0.04,
                     return_scenarios=np.full((4, 12), 0.003))
