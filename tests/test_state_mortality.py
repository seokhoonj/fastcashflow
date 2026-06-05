"""GAP A -- state-conditional mortality (``State.mortality_rate``).

A state routes its in-force death decrement to a named rate; a post-diagnosis
state (post-cancer death) can carry an elevated mortality supplied via
``Basis.state_mortality_annual``. Default (``"mortality"``) keeps the global
decrement, so existing models are unchanged.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import State, Transition, StateModel
from fastcashflow.basis import Basis
from fastcashflow.modelpoints import ModelPoints

_FLAT = lambda v: (lambda s, a, d: np.full(np.shape(a), v))
_ZERO = _FLAT(0.0)


def _two_state(post_rate_name):
    """healthy(0) + post(1); post routes its mortality to ``post_rate_name``."""
    return StateModel(states=(
        State("healthy", premium=True, transitions=(
            Transition("mortality"), Transition("lapse"))),
        State("post", transitions=(Transition("mortality"),),
              mortality_rate=post_rate_name),
    ), seating=(0, 1))


def _seated_post_mp(term=12):
    return ModelPoints(
        issue_age=np.array([50], dtype=np.int64),
        benefits={0: np.array([0.0])},
        premium=np.array([0.0]),
        term_months=np.array([term], dtype=np.int64),
        state=np.array([1], dtype=np.int64),          # seated in post
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})


def _basis(post_rate_name, state_mort=None):
    return Basis(
        mortality_annual=_FLAT(0.10), lapse_annual=_ZERO,
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        state_mortality_annual=state_mort,
        state_model=_two_state(post_rate_name),
        coverages=(fcf.CoverageRate("DEATH", _FLAT(0.10)),))


# ---------------------------------------------------------------------------
# A2 -- the post state decrements at its own (elevated) mortality
# ---------------------------------------------------------------------------
def test_state_conditional_decrement():
    # post mortality 0.30/yr vs global 0.10/yr; seated in post, no lapse.
    # constant-force monthly survival ** 12 == (1 - annual), so inforce[12]
    # is exactly 1 - q.
    elevated = fcf.gmm.measure(
        _seated_post_mp(13),
        _basis("dth_post", {"dth_post": _FLAT(0.30)}))
    control = fcf.gmm.measure(
        _seated_post_mp(13), _basis("mortality"))    # post uses global 0.10
    ife = elevated.cashflows.inforce[0]
    ifc = control.cashflows.inforce[0]
    assert ife[12] == pytest.approx(0.70, abs=1e-9)
    assert ifc[12] == pytest.approx(0.90, abs=1e-9)
    assert ife[12] < ifc[12]                          # elevated decays faster


# ---------------------------------------------------------------------------
# A1 -- a named rate with no table falls back to the global mortality
# ---------------------------------------------------------------------------
def test_named_rate_without_table_falls_back():
    fallback = fcf.gmm.measure(
        _seated_post_mp(12), _basis("dth_post", state_mort=None))
    control = fcf.gmm.measure(_seated_post_mp(12), _basis("mortality"))
    assert np.allclose(fallback.cashflows.inforce[0], control.cashflows.inforce[0])


# ---------------------------------------------------------------------------
# A3 -- detailed (measure) and fused (value) agree under elevated mortality
# ---------------------------------------------------------------------------
def test_state_mortality_detailed_matches_fused():
    mp = _seated_post_mp(120)
    # give the post state a death benefit so BEL is non-trivial
    mp = ModelPoints(
        issue_age=np.array([50], dtype=np.int64),
        benefits={0: np.array([100_000.0])},
        premium=np.array([0.0]),
        term_months=np.array([120], dtype=np.int64),
        state=np.array([1], dtype=np.int64),
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
    basis = Basis(
        mortality_annual=_FLAT(0.10), lapse_annual=_ZERO,
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        state_mortality_annual={"dth_post": _FLAT(0.30)},
        state_model=_two_state("dth_post"),
        coverages=(fcf.CoverageRate("DEATH", _FLAT(0.30)),))
    detailed = fcf.gmm.measure(mp, basis, full=True)
    fused = fcf.gmm.measure(mp, basis, full=False)
    assert fused.bel[0] == pytest.approx(detailed.bel[0], rel=1e-9)
    assert detailed.bel[0] > 0.0                       # a real death-claim BEL
