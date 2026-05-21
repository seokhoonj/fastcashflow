"""Stochastic-valuation validation -- the liability distribution over scenarios.

Each scenario is a discount rate; ``value_stochastic`` values the portfolio
under every scenario with the fused kernel and records the portfolio totals,
so the distribution -- mean, percentiles -- can be read off.
"""
from dataclasses import replace

import numpy as np

from fastcashflow import Assumptions, ModelPointSet, value, value_stochastic


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.001),
        lapse_monthly=lambda duration: np.full(duration.shape, 0.01),
        discount_annual=0.03,
        expense_acquisition=200_000.0,
        expense_maintenance_annual=60_000.0,
        expense_inflation=0.02,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )


def _portfolio(n: int = 200) -> ModelPointSet:
    rng = np.random.default_rng(6)
    return ModelPointSet(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(20, 90, n) * 1_000_000,
        monthly_premium=rng.integers(8, 20, n) * 10_000,
        term_months=rng.integers(60, 180, n),
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
