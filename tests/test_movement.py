"""Period-close roll-forward validation -- the expected-basis analysis of change.

``roll_forward`` slices a GMM measurement into reporting-period movements.
Each period must reconcile (opening + movements = closing) and consecutive
periods must chain.
"""
import numpy as np
import pytest

from fastcashflow import Assumptions, ModelPointSet, measure, roll_forward


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


def _portfolio(n: int = 50) -> ModelPointSet:
    rng = np.random.default_rng(4)
    return ModelPointSet(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(20, 90, n) * 1_000_000,
        monthly_premium=rng.integers(8, 20, n) * 10_000,
        term_months=np.full(n, 120),
    )


def test_roll_forward_period_count():
    """A 120-month horizon in yearly periods gives ten movements."""
    periods = roll_forward(measure(_portfolio(), _assumptions()), period_months=12)
    assert len(periods) == 10


def test_roll_forward_csm_reconciles():
    """Each period: opening + accretion - release = closing CSM."""
    for p in roll_forward(measure(_portfolio(), _assumptions()), 12):
        assert np.allclose(
            p.csm_opening + p.csm_accretion - p.csm_release, p.csm_closing
        )


def test_roll_forward_bel_and_ra_reconcile():
    """Each period: opening + interest - release = closing, for BEL and RA."""
    for p in roll_forward(measure(_portfolio(), _assumptions()), 12):
        assert np.allclose(
            p.bel_opening + p.bel_interest - p.bel_release, p.bel_closing
        )
        assert np.allclose(
            p.ra_opening + p.ra_interest - p.ra_release, p.ra_closing
        )


def test_roll_forward_periods_chain():
    """Each period's closing balances are the next period's opening balances."""
    periods = roll_forward(measure(_portfolio(), _assumptions()), 12)
    for prev, nxt in zip(periods, periods[1:]):
        assert prev.month_end == nxt.month_start
        assert np.allclose(prev.csm_closing, nxt.csm_opening)
        assert np.allclose(prev.bel_closing, nxt.bel_opening)
        assert np.allclose(prev.ra_closing, nxt.ra_opening)


def test_roll_forward_opening_is_inception():
    """The first period opens at the inception measurement."""
    m = measure(_portfolio(), _assumptions())
    first = roll_forward(m, 12)[0]
    assert first.month_start == 0
    assert np.allclose(first.csm_opening, m.csm[:, 0])
    assert np.allclose(first.bel_opening, m.bel[:, 0])


def test_roll_forward_runs_off_to_zero():
    """The final period closes the contract -- CSM and BEL run off to zero."""
    last = roll_forward(measure(_portfolio(), _assumptions()), 12)[-1]
    assert np.allclose(last.csm_closing, 0.0, atol=1.0)
    assert np.allclose(last.bel_closing, 0.0, atol=1.0)


def test_roll_forward_uneven_last_period():
    """A horizon not divisible by the period gives a short final period."""
    periods = roll_forward(measure(_portfolio(), _assumptions()), period_months=7)
    assert periods[-1].month_end == 120
    assert sum(p.month_end - p.month_start for p in periods) == 120


def test_roll_forward_rejects_bad_period():
    """A non-positive period length is an error."""
    with pytest.raises(ValueError, match="period_months"):
        roll_forward(measure(_portfolio(), _assumptions()), 0)
