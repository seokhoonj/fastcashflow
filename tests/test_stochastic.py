"""Stochastic-valuation validation -- the liability distribution over scenarios.

Each scenario is a discount rate; ``value_stochastic`` values the portfolio
under every scenario with the fused kernel and records the portfolio totals,
so the distribution -- mean, percentiles -- can be read off.
"""
from dataclasses import replace

import numpy as np
import pytest

from fastcashflow import ExpenseItem, ModelPoints, value, value_stochastic
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_assumptions


def _assumptions():
    return make_death_assumptions(
        mortality_q       = 0.001,
        lapse_q           = 0.01,
        discount_annual   = 0.03,
        expense_inflation = 0.02,
        expense_items     = (
            ExpenseItem("acquisition",  "alpha_fixed",    200_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  60_000.0),
        ),
        ra_confidence     = 0.75,
        mortality_cv      = 0.10,
    )


def _portfolio(n: int = 200) -> ModelPoints:
    rng = np.random.default_rng(6)
    return ModelPoints(
        issue_age=rng.integers(30, 55, n),
        benefits={0: rng.integers(20, 90, n) * 1_000_000},
        level_premium=rng.integers(8, 20, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        benefit_patterns=PATTERNS,
    )


def test_stochastic_matches_value_per_scenario():
    """Each scenario's total equals value() run at that discount rate."""
    mps, asmp = _portfolio(), _assumptions()
    rates = np.array([0.02, 0.03, 0.05])
    res = value_stochastic(mps, asmp, rates)
    for s, rate in enumerate(rates):
        v = value(mps, replace(asmp, discount_annual=float(rate)))
        assert np.isclose(res.bel[s], v.bel.sum())
        assert np.isclose(res.ra[s], v.ra.sum())
        assert np.isclose(res.csm[s], v.csm.sum())
        assert np.isclose(res.loss_component[s], v.loss_component.sum())


def test_stochastic_produces_a_distribution():
    """Different discount scenarios give a genuine spread of liabilities."""
    res = value_stochastic(_portfolio(), _assumptions(),
                           np.linspace(0.01, 0.06, 40))
    assert res.bel.std() > 0.0
    assert res.csm.std() > 0.0


def test_stochastic_summary_statistics():
    """mean() and percentile() read the distribution off the scenario totals."""
    res = value_stochastic(_portfolio(), _assumptions(),
                           np.linspace(0.01, 0.06, 40))
    assert np.isclose(res.mean()["bel"], res.bel.mean())
    assert np.isclose(res.percentile(75)["csm"], np.percentile(res.csm, 75))


def test_stochastic_single_scenario():
    """One scenario reproduces a plain value() at that rate."""
    mps, asmp = _portfolio(), _assumptions()
    res = value_stochastic(mps, asmp, np.array([0.04]))
    v = value(mps, replace(asmp, discount_annual=0.04))
    assert res.bel.shape == (1,)
    assert np.isclose(res.bel[0], v.bel.sum())


def test_value_constant_curve_matches_the_flat_rate():
    """value() with a constant discount curve reproduces the flat-rate run."""
    mps, asmp = _portfolio(), _assumptions()
    n_time = int(mps.term_months.max())
    flat = value(mps, replace(asmp, discount_annual=0.04))
    curve = value(mps, asmp, discount_curve=np.full(n_time, 0.04))
    assert np.allclose(flat.bel, curve.bel, rtol=1e-9)
    assert np.allclose(flat.csm, curve.csm, rtol=1e-9)


def test_stochastic_accepts_rate_curves():
    """A 2-D scenarios array is read as one discount-rate curve per scenario."""
    mps, asmp = _portfolio(), _assumptions()
    n_time = int(mps.term_months.max())
    rng = np.random.default_rng(8)
    curves = 0.03 + rng.normal(0.0, 0.005, size=(20, n_time))
    res = value_stochastic(mps, asmp, curves)
    assert res.bel.shape == (20,)
    assert res.bel.std() > 0.0


def test_stochastic_rising_curve_differs_from_flat():
    """A sloped curve gives a different liability than a flat rate."""
    mps, asmp = _portfolio(), _assumptions()
    n_time = int(mps.term_months.max())
    rising = np.linspace(0.01, 0.06, n_time).reshape(1, n_time)
    flat = np.array([float(rising.mean())])
    res_curve = value_stochastic(mps, asmp, rising)
    res_flat = value_stochastic(mps, asmp, flat)
    assert not np.isclose(res_curve.bel[0], res_flat.bel[0])


def test_stochastic_curve_rejects_wrong_width():
    """A 2-D scenarios array must be as wide as the projection horizon."""
    mps, asmp = _portfolio(), _assumptions()
    with pytest.raises(ValueError, match="columns"):
        value_stochastic(mps, asmp, np.full((5, 7), 0.03))
