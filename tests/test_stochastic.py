"""Stochastic-valuation validation -- the liability distribution over scenarios.

Each scenario is a discount rate; ``measure_stochastic`` values the portfolio
under every scenario with the fused kernel and records the portfolio totals,
so the distribution -- mean, percentiles -- can be read off.
"""
import fastcashflow as fcf
from dataclasses import replace

import numpy as np
import pytest

from fastcashflow import ExpenseItem, ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


def _basis():
    return make_death_basis(
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
        benefits={"DEATH": rng.integers(20, 90, n) * 1_000_000},
        premium=rng.integers(8, 20, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        calculation_methods=PATTERNS,
    )


def test_stochastic_matches_value_per_scenario():
    """Each scenario's total equals measure() run at that discount rate."""
    mps, basis = _portfolio(), _basis()
    rates = np.array([0.02, 0.03, 0.05])
    res = fcf.gmm.stochastic(mps, basis, rates)
    for s, rate in enumerate(rates):
        v = measure(mps, replace(basis, discount_annual=float(rate)), full=False)
        assert np.isclose(res.bel[s], v.bel.sum())
        assert np.isclose(res.ra[s], v.ra.sum())
        assert np.isclose(res.csm[s], v.csm.sum())
        assert np.isclose(res.loss_component[s], v.loss_component.sum())


def test_stochastic_produces_a_distribution():
    """Different discount scenarios give a genuine spread of liabilities."""
    res = fcf.gmm.stochastic(_portfolio(), _basis(),
                           np.linspace(0.01, 0.06, 40))
    assert res.bel.std() > 0.0
    assert res.csm.std() > 0.0


def test_stochastic_summary_statistics():
    """mean() and percentile() read the distribution off the scenario totals."""
    res = fcf.gmm.stochastic(_portfolio(), _basis(),
                           np.linspace(0.01, 0.06, 40))
    assert np.isclose(res.mean()["bel"], res.bel.mean())
    assert np.isclose(res.percentile(75)["csm"], np.percentile(res.csm, 75))


def test_stochastic_single_scenario():
    """One scenario reproduces a plain measure() at that rate."""
    mps, basis = _portfolio(), _basis()
    res = fcf.gmm.stochastic(mps, basis, np.array([0.04]))
    v = measure(mps, replace(basis, discount_annual=0.04), full=False)
    assert res.bel.shape == (1,)
    assert np.isclose(res.bel[0], v.bel.sum())


def test_value_constant_curve_matches_the_flat_rate():
    """measure() with a constant discount curve reproduces the flat-rate run."""
    mps, basis = _portfolio(), _basis()
    n_time = int(mps.term_months.max())
    flat = measure(mps, replace(basis, discount_annual=0.04), full=False)
    curve = measure(mps, basis, discount_curve=np.full(n_time, 0.04), full=False)
    assert np.allclose(flat.bel, curve.bel, rtol=1e-9)
    assert np.allclose(flat.csm, curve.csm, rtol=1e-9)


def test_stochastic_accepts_rate_curves():
    """A 2-D scenarios array is read as one discount-rate curve per scenario."""
    mps, basis = _portfolio(), _basis()
    n_time = int(mps.term_months.max())
    rng = np.random.default_rng(8)
    curves = 0.03 + rng.normal(0.0, 0.005, size=(20, n_time))
    res = fcf.gmm.stochastic(mps, basis, curves)
    assert res.bel.shape == (20,)
    assert res.bel.std() > 0.0


def test_stochastic_rising_curve_differs_from_flat():
    """A sloped curve gives a different liability than a flat rate."""
    mps, basis = _portfolio(), _basis()
    n_time = int(mps.term_months.max())
    rising = np.linspace(0.01, 0.06, n_time).reshape(1, n_time)
    flat = np.array([float(rising.mean())])
    res_curve = fcf.gmm.stochastic(mps, basis, rising)
    res_flat = fcf.gmm.stochastic(mps, basis, flat)
    assert not np.isclose(res_curve.bel[0], res_flat.bel[0])


def test_stochastic_curve_rejects_wrong_width():
    """A 2-D scenarios array must be as wide as the projection horizon."""
    mps, basis = _portfolio(), _basis()
    with pytest.raises(ValueError, match="columns"):
        fcf.gmm.stochastic(mps, basis, np.full((5, 7), 0.03))


def test_stochastic_settlement_pattern_fallback_matches_value():
    """A claims settlement pattern routes to the per-scenario fallback, which
    must still equal measure() at each discount rate."""
    mps = _portfolio()
    basis = replace(_basis(), settlement_pattern=np.array([0.5, 0.3, 0.2]))
    rates = np.array([0.02, 0.03, 0.05])
    res = fcf.gmm.stochastic(mps, basis, rates)
    for s, rate in enumerate(rates):
        v = measure(mps, replace(basis, discount_annual=float(rate)), full=False)
        assert np.isclose(res.bel[s], v.bel.sum())
        assert np.isclose(res.csm[s], v.csm.sum())


def _boundary_portfolio(n: int = 50) -> ModelPoints:
    """A book whose contract boundary cuts well short of the term -- the case
    that read past the cash-flow array width before the boundary fix."""
    rng = np.random.default_rng(11)
    term = rng.integers(120, 240, n)
    return ModelPoints(
        issue_age=rng.integers(30, 55, n),
        benefits={"DEATH": rng.integers(20, 90, n) * 1_000_000},
        premium=rng.integers(8, 20, n) * 10_000,
        term_months=term,
        contract_boundary_months=term // 3,        # cut at a third of the term
        calculation_methods=PATTERNS,
    )


def test_stochastic_with_contract_boundary_matches_value_per_scenario():
    """Regression (P0-1): the scenario kernel must loop to the contract
    boundary, not the term. With boundary < term it previously read past the
    truncated cash-flow arrays inside @njit and returned nan / garbage."""
    mps, basis = _boundary_portfolio(), _basis()
    rates = np.array([0.02, 0.03, 0.05])
    res = fcf.gmm.stochastic(mps, basis, rates)
    assert np.all(np.isfinite(res.bel))          # was nan / 1e271 before the fix
    for s, rate in enumerate(rates):
        v = measure(mps, replace(basis, discount_annual=float(rate)), full=False)
        assert np.isclose(res.bel[s], v.bel.sum())
        assert np.isclose(res.ra[s], v.ra.sum())
        assert np.isclose(res.csm[s], v.csm.sum())


def test_stochastic_cost_of_capital_matches_full_per_scenario():
    """Regression (P1-3): cost-of-capital RA is not on the fast path, so the
    fallback must value each flat scenario with measure(full=True) rather than
    full=False (which rejects COC). Previously this raised ValueError."""
    mps = _portfolio()
    basis = replace(_basis(), ra_method="cost_of_capital",
                    cost_of_capital_rate=0.06)
    rates = np.array([0.02, 0.03, 0.05])
    res = fcf.gmm.stochastic(mps, basis, rates)
    for s, rate in enumerate(rates):
        v = measure(mps, replace(basis, discount_annual=float(rate)), full=True)
        assert np.isclose(res.bel[s], v.bel.sum())
        assert np.isclose(res.ra[s], v.ra.sum())
        assert np.isclose(res.csm[s], v.csm.sum())


def test_stochastic_cost_of_capital_rejects_rate_curves():
    """COC + a per-month discount curve (2-D) has no full-path home for the
    curve -- raise a clear NotImplementedError, not a misleading ValueError."""
    mps = _portfolio()
    basis = replace(_basis(), ra_method="cost_of_capital",
                    cost_of_capital_rate=0.06)
    n_time = int(np.asarray(mps.contract_boundary_months).max())
    with pytest.raises(NotImplementedError, match="cost_of_capital"):
        fcf.gmm.stochastic(mps, basis, np.full((3, n_time), 0.03))
