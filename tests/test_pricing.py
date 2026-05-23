"""Pricing validation -- solve_premium against its profitability targets.

Each solved premium is fed back through value() and checked to reproduce
the target it was solved for.
"""
import numpy as np
import pytest

from fastcashflow import Assumptions, ModelPoints, solve_premium, value


def _annual(m):
    """Convert a monthly rate to the equivalent annual rate the engine expects."""
    return 1.0 - (1.0 - m) ** 12


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.0008)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.01)),
        discount_annual=0.03,
        expense_acquisition=200_000.0,
        expense_maintenance_annual=36_000.0,
        expense_inflation=0.02,
        ra_confidence=0.80,
        mortality_cv=0.10,
    )


def _portfolio(n: int = 300) -> ModelPoints:
    rng = np.random.default_rng(8)
    return ModelPoints(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(20, 100, n) * 1_000_000,
        level_premium=np.zeros(n),          # ignored by solve_premium
        term_months=rng.integers(60, 180, n),
    )


def _priced(mps: ModelPoints, premium) -> ModelPoints:
    return ModelPoints(
        issue_age=mps.issue_age,
        death_benefit=mps.death_benefit,
        level_premium=premium,
        term_months=mps.term_months,
    )


def test_break_even_premium():
    """The break-even premium yields zero CSM and zero loss component."""
    mps, asmp = _portfolio(), _assumptions()
    premium = solve_premium(mps, asmp, break_even=True)

    v = value(_priced(mps, premium), asmp)
    assert np.allclose(v.csm, 0.0, atol=1.0)
    assert np.allclose(v.loss_component, 0.0, atol=1.0)


def test_target_csm_premium():
    """Solving for an absolute CSM reproduces that CSM."""
    mps, asmp = _portfolio(), _assumptions()
    target = 500_000.0
    premium = solve_premium(mps, asmp, csm=target)

    v = value(_priced(mps, premium), asmp)
    assert np.allclose(v.csm, target)


def test_target_margin_premium():
    """Solving for a profit margin yields CSM / PV(premiums) == margin."""
    mps, asmp = _portfolio(), _assumptions()
    m = 0.15
    premium = solve_premium(mps, asmp, margin=m)
    v = value(_priced(mps, premium), asmp)

    # PV(premiums) = premium * B, with B from the linear FCF relation
    at_zero = value(_priced(mps, np.zeros(mps.n_mp)), asmp)
    at_one = value(_priced(mps, np.ones(mps.n_mp)), asmp)
    b = (at_zero.bel + at_zero.ra) - (at_one.bel + at_one.ra)
    pv_premiums = premium * b
    assert np.allclose(v.csm / pv_premiums, m)


def test_invalid_target():
    """Zero, multiple or out-of-range targets are rejected."""
    mps, asmp = _portfolio(50), _assumptions()
    with pytest.raises(ValueError, match="exactly one target"):
        solve_premium(mps, asmp)
    with pytest.raises(ValueError, match="exactly one target"):
        solve_premium(mps, asmp, break_even=True, margin=0.1)
    with pytest.raises(ValueError, match="margin must be in"):
        solve_premium(mps, asmp, margin=1.5)
