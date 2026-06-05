"""Pricing validation -- solve_premium against its profitability targets.

Each solved premium is fed back through measure() and checked to reproduce
the target it was solved for.
"""
import numpy as np
import pytest

from fastcashflow import ExpenseItem, ModelPoints, solve_premium
from fastcashflow.gmm import measure
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


def _assumptions():
    return make_death_basis(
        mortality_q       = 0.0008,
        lapse_q           = 0.01,
        discount_annual   = 0.03,
        expense_inflation = 0.02,
        expense_items     = (
            ExpenseItem("acquisition",  "alpha_fixed",    200_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  36_000.0),
        ),
        ra_confidence     = 0.80,
        mortality_cv      = 0.10,
    )


def _portfolio(n: int = 300) -> ModelPoints:
    rng = np.random.default_rng(8)
    return ModelPoints(
        issue_age=rng.integers(30, 55, n),
        benefits={0: rng.integers(20, 100, n) * 1_000_000},
        premium=np.zeros(n),          # ignored by solve_premium
        term_months=rng.integers(60, 180, n),
        calculation_methods=PATTERNS,
    )


def _priced(mps: ModelPoints, premium) -> ModelPoints:
    return ModelPoints(
        issue_age=mps.issue_age,
        coverage_index=mps.coverage_index,
        coverage_amount=mps.coverage_amount,
        coverage_offset=mps.coverage_offset,
        premium=premium,
        term_months=mps.term_months,
        calculation_methods=mps.calculation_methods,
    )


def test_break_even_premium():
    """The break-even premium yields zero CSM and zero loss component."""
    mps, basis = _portfolio(), _assumptions()
    premium = solve_premium(mps, basis, break_even=True)

    v = measure(_priced(mps, premium), basis, full=False)
    assert np.allclose(v.csm, 0.0, atol=1.0)
    assert np.allclose(v.loss_component, 0.0, atol=1.0)


def test_target_csm_premium():
    """Solving for an absolute CSM reproduces that CSM."""
    mps, basis = _portfolio(), _assumptions()
    target = 500_000.0
    premium = solve_premium(mps, basis, csm=target)

    v = measure(_priced(mps, premium), basis, full=False)
    assert np.allclose(v.csm, target)


def test_target_margin_premium():
    """Solving for a profit margin yields CSM / PV(premiums) == margin."""
    mps, basis = _portfolio(), _assumptions()
    m = 0.15
    premium = solve_premium(mps, basis, margin=m)
    v = measure(_priced(mps, premium), basis, full=False)

    # PV(premiums) = premium * B, with B from the linear FCF relation
    at_zero = measure(_priced(mps, np.zeros(mps.n_mp)), basis, full=False)
    at_one = measure(_priced(mps, np.ones(mps.n_mp)), basis, full=False)
    b = (at_zero.bel + at_zero.ra) - (at_one.bel + at_one.ra)
    pv_premiums = premium * b
    assert np.allclose(v.csm / pv_premiums, m)


def test_invalid_target():
    """Zero, multiple or out-of-range targets are rejected."""
    mps, basis = _portfolio(50), _assumptions()
    with pytest.raises(ValueError, match="exactly one target"):
        solve_premium(mps, basis)
    with pytest.raises(ValueError, match="exactly one target"):
        solve_premium(mps, basis, break_even=True, margin=0.1)
    with pytest.raises(ValueError, match="margin must be in"):
        solve_premium(mps, basis, margin=1.5)
