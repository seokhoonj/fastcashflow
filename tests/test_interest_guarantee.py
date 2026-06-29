"""Traditional interest-rate-guarantee TVOG -- hand-calc anchors.

A general-account contract credits its reserve at a minimum guaranteed rate
``i_g``; when the earned rate falls below ``i_g`` the entity funds the shortfall.
``pricing.interest_tvog`` prices that guarantee over earned-rate
scenarios and splits the cost into intrinsic value (the central-path cost) and
time value (the extra the convex ``max(i_g - r, 0)`` adds once rates vary).

The anchors: a zero-volatility scenario set collapses to the intrinsic value; a
guarantee that never bites costs nothing; a tiny case is re-derived by hand from
the real statutory reserve; the decomposition is an identity; and a risk-neutral
ESG scenario set with a forward central path shows a positive time value.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import pricing


def _endow(term: int = 24):
    return fcf.ModelPoints.single(
        issue_age=40, premium=0.0, term_months=term,
        benefits={"DEATH": 100_000_000.0}, maturity_benefit=100_000_000.0,
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})


def _stat(rate: float = 0.025):
    return fcf.Basis(mortality_annual=0.01, lapse_annual=0.0, discount_annual=rate,
                     ra_confidence=0.75, mortality_cv=0.0,
                     coverages=(fcf.CoverageRate("DEATH", 0.01),))


def _n_time(endow, stat):
    V, _ = pricing.statutory_reserve(endow, stat)
    return V.shape[1] - 1


def test_zero_volatility_collapses_to_intrinsic():
    """A scenario set that is the central path repeated (no volatility) has zero
    time value -- the whole cost is the intrinsic value."""
    endow, stat = _endow(), _stat()
    n = _n_time(endow, stat)
    central = np.linspace(0.01, 0.04, n)          # dips below i_g early -> intrinsic > 0
    scen = np.tile(central, (5, 1))
    res = pricing.interest_tvog(endow, stat, scen, central_rates=central)
    assert np.isclose(res.time_value, 0.0, atol=1e-9)
    assert np.isclose(res.total_value, res.intrinsic_value)
    assert res.intrinsic_value > 0.0              # the central path does bite
    assert np.allclose(res.guarantee_cost, res.guarantee_cost[0])


def test_no_cost_when_earned_above_guarantee():
    """If every scenario earns at or above i_g, the guarantee never bites."""
    endow, stat = _endow(), _stat(0.02)
    n = _n_time(endow, stat)
    scen = np.full((4, n), 0.05)
    res = pricing.interest_tvog(endow, stat, scen,
                                          central_rates=np.full(n, 0.05))
    assert np.all(res.guarantee_cost == 0.0)
    assert res.total_value == 0.0
    assert res.time_value == 0.0


def test_tiny_hand_case():
    """A 3-month case, re-derived by hand from the real statutory reserve.

    Scenario 0 stays at 5% (above i_g = 2.5%, no cost); scenario 1 dips to 1% in
    month 1 only. The month-1 shortfall ``i_g_m - r_m`` accrues on the reserve
    held over month 1 and discounts at the scenario's own rate to the end of that
    month -- computed here with explicit scalars (no cumprod / matmul)."""
    endow, stat = _endow(term=3), _stat(0.025)
    V, _ = pricing.statutory_reserve(endow, stat)
    Vt = V.sum(axis=0)                            # (4,) portfolio reserve, V[0..3]

    s0 = np.full(3, 0.05)
    s1 = np.array([0.05, 0.01, 0.05])             # only month 1 below i_g
    scen = np.vstack([s0, s1])
    central = np.full(3, 0.05)                    # >= i_g everywhere -> intrinsic 0
    res = pricing.interest_tvog(endow, stat, scen, central_rates=central)

    ig_m = (1.0 + 0.025) ** (1.0 / 12.0) - 1.0
    rm_hi = (1.0 + 0.05) ** (1.0 / 12.0) - 1.0
    rm_lo = (1.0 + 0.01) ** (1.0 / 12.0) - 1.0
    # end-of-month discount to the close of month 1: 1/(1+r0) * 1/(1+r1)
    discount_to_eom1 = (1.0 / (1.0 + rm_hi)) / (1.0 + rm_lo)
    cost1 = discount_to_eom1 * (ig_m - rm_lo) * Vt[1]

    assert res.guarantee_cost[0] == 0.0
    assert np.isclose(res.guarantee_cost[1], cost1)
    assert np.isclose(res.intrinsic_value, 0.0)
    assert np.isclose(res.time_value, res.guarantee_cost.mean())


def test_decomposition_identity():
    """total_value == intrinsic + time_value == mean(guarantee_cost)."""
    endow, stat = _endow(), _stat()
    n = _n_time(endow, stat)
    rng = np.random.default_rng(0)
    scen = 0.025 + rng.normal(0.0, 0.02, size=(50, n))
    res = pricing.interest_tvog(endow, stat, scen,
                                          central_rates=np.full(n, 0.025))
    assert np.isclose(res.total_value, res.intrinsic_value + res.time_value)
    assert np.isclose(res.total_value, float(res.guarantee_cost.mean()))
    assert res.time_value >= 0.0                  # Jensen gap of a convex payoff


def test_forward_central_path_with_esg():
    """A risk-neutral ESG scenario set with the forward central path shows a
    positive time value when the guarantee can bite (the curve starts below i_g)."""
    endow, stat = _endow(term=120), _stat(0.025)
    n = _n_time(endow, stat)
    maturities = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0])
    rates = np.array([0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.030])
    scen = fcf.esg.simulate(
        maturities, rates, ufr=0.035, alpha=0.1, mean_reversion=0.05,
        rate_vol=0.015, equity_vol=0.0, correlation=0.0,
        n_scenarios=2000, n_time=n, seed=1)
    res = pricing.interest_tvog(
        endow, stat, scen.rates, initial_prices=scen.initial_prices)
    assert res.time_value > 0.0
    assert res.total_value >= res.intrinsic_value


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------

def test_requires_central_or_initial_prices():
    endow, stat = _endow(), _stat()
    n = _n_time(endow, stat)
    with pytest.raises(ValueError, match="central_rates or initial_prices"):
        pricing.interest_tvog(endow, stat, np.full((3, n), 0.03))


def test_rejects_horizon_mismatch():
    endow, stat = _endow(), _stat()
    n = _n_time(endow, stat)
    with pytest.raises(ValueError, match="columns"):
        pricing.interest_tvog(
            endow, stat, np.full((3, n + 5), 0.03), central_rates=np.full(n, 0.02))


def test_rejects_router():
    with pytest.raises(NotImplementedError, match="single Basis"):
        pricing.interest_tvog(
            _endow(), fcf.samples.basis("gmm"), np.full((2, 24), 0.03),
            central_rates=np.full(24, 0.02))
