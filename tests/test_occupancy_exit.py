"""F7 -- true occupancy exit (``State.exit_after_months``).

A cohort whose sojourn in a state reaches ``exit_after_months`` months leaves the
in-force set (a ``to=None`` transition by sojourn). Semi-Markov only -- it
needs duration tracking. Distinct from ``benefit_max_months``, which stops the
payment but keeps the lives in force. Full path only.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import State, Transition, StateModel
from fastcashflow.basis import Basis
from fastcashflow.modelpoints import ModelPoints

_ZERO = lambda s, a, d: np.full(np.shape(a), 0.0)   # no death / lapse


def _model(exit_after_months, cap=0, duration_max=8):
    """active + disabled (benefit) with a sojourn exit; no decrements, so a
    life seated in ``disabled`` stays until the exit boundary."""
    return StateModel(states=(
        State("active", premium=True, transitions=(
            Transition("mortality"), Transition("lapse"))),
        State("disabled", benefit=True, duration_max=duration_max,
              benefit_max_months=cap, exit_after_months=exit_after_months,
              transitions=(Transition("mortality"),)),
    ), seating=(0, 1))


def _seated_mp(term=12):
    return ModelPoints(
        issue_age=np.array([50], dtype=np.int64),
        benefits={0: np.array([0.0])},
        premium=np.array([0.0]),
        term_months=np.array([term], dtype=np.int64),
        disability_income=np.array([100.0]),
        state=np.array([1], dtype=np.int64),          # seated in disabled
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})


def _basis(exit_after_months, cap=0, duration_max=8):
    return Basis(
        mortality_annual=_ZERO, lapse_annual=_ZERO,
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        state_model=_model(exit_after_months, cap, duration_max),
        coverages=(fcf.CoverageRate("DEATH", _ZERO),))


# ---------------------------------------------------------------------------
# F7.1 -- hand calc: exit_after_months=3 keeps the lives 3 months, then drops them
# ---------------------------------------------------------------------------
def test_exit_after_drops_cohort_at_boundary():
    m = fcf.gmm.measure(_seated_mp(12), _basis(3), full=True)
    ife = m.cashflows.inforce[0]
    # present months 0,1,2 (3 months of sojourn); gone from month 3 on
    assert np.allclose(ife[:3], 1.0)
    assert np.allclose(ife[3:], 0.0)


# ---------------------------------------------------------------------------
# F7.2 -- exit_after_months is NOT benefit_max_months: cap stops pay, exit ends cover
# ---------------------------------------------------------------------------
def test_exit_distinct_from_benefit_cap():
    """benefit_max_months=3 stops the payment but the lives stay in force;
    exit_after_months=3 removes them. The two mechanisms must be distinguishable."""
    cap_only = fcf.gmm.measure(_seated_mp(12), _basis(0, cap=3), full=True)
    exit_only = fcf.gmm.measure(_seated_mp(12), _basis(3), full=True)
    assert np.allclose(cap_only.cashflows.inforce[0], 1.0)     # lives stay
    assert np.allclose(exit_only.cashflows.inforce[0][3:], 0.0)  # lives leave


# ---------------------------------------------------------------------------
# F7.3 -- pay the cap, then exit: cap=exit=3 pays 3 months AND drops at t=3
# ---------------------------------------------------------------------------
def test_pay_then_exit_compose():
    m = fcf.gmm.measure(_seated_mp(12), _basis(3, cap=3), full=True)
    dis = m.cashflows.disability_cf[0]
    ife = m.cashflows.inforce[0]
    assert np.allclose(dis[:3], 100.0)          # paid 3 months
    assert np.allclose(dis[3:], 0.0)
    assert np.allclose(ife[:3], 1.0)
    assert np.allclose(ife[3:], 0.0)            # then exits


# ---------------------------------------------------------------------------
# F7.4 -- off-by-one: exit_after_months == duration_max RAISES (strict guard)
# ---------------------------------------------------------------------------
def test_exit_after_must_be_below_duration_max():
    with pytest.raises(ValueError, match="must exceed exit_after_months"):
        State("x", duration_max=4, exit_after_months=4)
    with pytest.raises(ValueError, match="must exceed exit_after_months"):
        State("x", duration_max=3, exit_after_months=4)
    # the tightest valid stack (duration_max just above exit_after_months) works
    s = State("x", duration_max=4, exit_after_months=3)
    assert s.exit_after_months == 3


def test_exit_after_tightest_valid_drops():
    m = fcf.gmm.measure(_seated_mp(12), _basis(3, duration_max=4), full=True)
    assert np.allclose(m.cashflows.inforce[0][3:], 0.0)


# ---------------------------------------------------------------------------
# F7.5 -- exit_after_months >= benefit_max_months (pay the cap, then exit)
# ---------------------------------------------------------------------------
def test_exit_after_below_cap_rejected():
    with pytest.raises(ValueError, match="must be >= benefit_max_months"):
        State("x", benefit=True, duration_max=8, benefit_max_months=4,
              exit_after_months=3)


# ---------------------------------------------------------------------------
# F7.6 -- exit_after_months needs sojourn tracking (semi-Markov)
# ---------------------------------------------------------------------------
def test_exit_after_needs_duration_max():
    # duration_max == 0 (Markov state) with exit_after_months fails the strict guard
    with pytest.raises(ValueError, match="must exceed exit_after_months"):
        State("x", exit_after_months=3)


def test_exit_after_rejected_on_markov_compile():
    """A Markov model (no duration-tracked states) that somehow carries an
    exit_after_months is rejected at compile -- exit needs sojourn tracking."""
    from fastcashflow.statemodel import compile_state_model
    # build a Markov state and inject exit_after_months past __post_init__ (which
    # would otherwise force duration_max); use a duration-free sibling to keep
    # the model Markov so compile_state_model is the path taken.
    model = StateModel(states=(
        State("a", premium=True, transitions=(Transition("mortality"),)),
    ), seating=(0,))
    # forge a state with exit_after_months but duration_max 0 by bypassing validation
    object.__setattr__(model.states[0], "exit_after_months", 3)
    with pytest.raises(ValueError, match="semi-Markov only"):
        compile_state_model(model, {"mortality": np.zeros((1, 1))})


# ---------------------------------------------------------------------------
# F7.7 -- the fast path rejects exit_after_months (selective)
# ---------------------------------------------------------------------------
def test_fast_path_rejects_exit_after():
    with pytest.raises(NotImplementedError, match="exit_after_months"):
        fcf.gmm.measure(_seated_mp(12), _basis(3), full=False)
    # a benefit_max_months-only model still runs fast
    fast = fcf.gmm.measure(_seated_mp(12), _basis(0, cap=3), full=False)
    full = fcf.gmm.measure(_seated_mp(12), _basis(0, cap=3), full=True)
    assert fast.bel[0] == pytest.approx(full.bel[0], rel=1e-9)
