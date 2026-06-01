"""IFRS 17 transition validation -- the fair value approach.

The CSM at transition is the fair value less the fulfilment cash flows,
floored at zero; any shortfall falls into the loss component.
"""
import numpy as np
import pytest

from fastcashflow import ExpenseItem, ModelPoints, roll_forward, transition
from fastcashflow.gmm import measure
from conftest import annual_from_monthly as _annual, make_death_assumptions


def _assumptions():
    return make_death_assumptions(
        mortality_q       = 0.002,
        lapse_q           = 0.01,
        discount_annual   = 0.03,
        expense_inflation = 0.02,
        expense_items     = (
            ExpenseItem("acquisition",  "alpha_fixed",    100_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  60_000.0),
        ),
        ra_confidence     = 0.75,
        mortality_cv      = 0.10,
    )


def _portfolio(n: int = 50) -> ModelPoints:
    rng = np.random.default_rng(8)
    return ModelPoints(
        issue_age=rng.integers(30, 55, n),
        benefits={0: rng.integers(20, 90, n) * 1_000_000},
        level_premium=rng.integers(8, 20, n) * 10_000,
        term_months=np.full(n, 120),
    )


def test_transition_csm_is_fair_value_less_fcf():
    """The transition CSM is the fair value less the fulfilment cash flows."""
    m = measure(_portfolio(), _assumptions())
    fcf0 = m.bel_path[:, 0] + m.ra_path[:, 0]
    t = transition(m, fcf0 + 1_000_000.0)
    assert np.allclose(t.csm_path[:, 0], 1_000_000.0)
    assert np.allclose(t.loss_component, 0.0)


def test_transition_below_fair_value_is_onerous():
    """A fair value below the fulfilment cash flows gives a loss component."""
    m = measure(_portfolio(), _assumptions())
    fcf0 = m.bel_path[:, 0] + m.ra_path[:, 0]
    t = transition(m, fcf0 - 500_000.0)
    assert np.allclose(t.csm_path[:, 0], 0.0)
    assert np.allclose(t.loss_component, 500_000.0)


def test_transition_csm_reconciles():
    """The transition CSM trajectory reconciles."""
    m = measure(_portfolio(), _assumptions())
    t = transition(m, m.bel_path[:, 0] + m.ra_path[:, 0] + 500_000.0)
    assert np.allclose(
        t.csm_path[:, :-1] + t.csm_accretion - t.csm_release, t.csm_path[:, 1:]
    )


def test_transition_composes_with_roll_forward():
    """A transitioned measurement flows into the period-close roll-forward."""
    m = measure(_portfolio(), _assumptions())
    t = transition(m, m.bel_path[:, 0] + m.ra_path[:, 0] + 1_000_000.0)
    periods = roll_forward(t, 12)
    assert np.allclose(periods[0].csm_opening, t.csm_path[:, 0])


def test_transition_rejects_wrong_length():
    """fair_value must have one entry per measurement row."""
    m = measure(_portfolio(), _assumptions())
    with pytest.raises(ValueError, match="one entry per row"):
        transition(m, np.array([1.0, 2.0]))
