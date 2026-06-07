"""GAP A -- state-conditional mortality (``State.mortality_rate_name``).

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
              mortality_rate_name=post_rate_name),
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


# ---------------------------------------------------------------------------
# A4 -- the death-count reporter splits by state (no split-brain)
# ---------------------------------------------------------------------------
def test_deaths_reporter_is_state_aware():
    elevated = fcf.gmm.measure(
        _seated_post_mp(13), _basis("dth_post", {"dth_post": _FLAT(0.30)}))
    control = fcf.gmm.measure(_seated_post_mp(13), _basis("mortality"))
    # seated in post: month-0 deaths follow the post (0.30) rate, not global.
    post_m = 1 - (1 - 0.30) ** (1 / 12)
    glob_m = 1 - (1 - 0.10) ** (1 / 12)
    assert elevated.cashflows.deaths[0][0] == pytest.approx(post_m, abs=1e-9)
    assert control.cashflows.deaths[0][0] == pytest.approx(glob_m, abs=1e-9)


# ---------------------------------------------------------------------------
# AB1 -- GAP A (elevated mortality) and GAP B (benefit cap) compose
# ---------------------------------------------------------------------------
def test_state_mortality_and_benefit_cap_compose():
    # disabled state: elevated mortality 0.20 AND a 3-month benefit cap.
    model = StateModel(states=(
        State("active", premium=True, transitions=(
            Transition("mortality"), Transition("lapse"))),
        State("disabled", benefit=True, sojourn_tracking_months=8, periodic_benefit_term_months=3,
              mortality_rate_name="dth_dis", transitions=(Transition("mortality"),)),
    ), seating=(0, 1))
    basis = Basis(
        mortality_annual=_FLAT(0.10), lapse_annual=_ZERO,
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        state_mortality_annual={"dth_dis": _FLAT(0.20)},
        state_model=model,
        coverages=(fcf.CoverageRate("DEATH", _FLAT(0.10)),))
    mp = ModelPoints(
        issue_age=np.array([55], dtype=np.int64),
        benefits={0: np.array([0.0])},
        premium=np.array([0.0]),
        term_months=np.array([24], dtype=np.int64),
        disability_income=np.array([100.0]),
        state=np.array([1], dtype=np.int64),
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
    m = fcf.gmm.measure(mp, basis)
    cf = m.cashflows.disability_cf[0]
    surv = (1 - 0.20) ** (1 / 12)                      # monthly survival at 0.20
    # benefit paid only the first 3 months, on the elevated-mortality decay
    assert cf[0] == pytest.approx(100.0)
    assert cf[1] == pytest.approx(100.0 * surv, abs=1e-6)
    assert cf[2] == pytest.approx(100.0 * surv ** 2, abs=1e-6)
    assert np.allclose(cf[3:], 0.0)                    # capped
    # detailed and fused agree with both features on
    assert (fcf.gmm.measure(mp, basis, full=False).bel[0]
            == pytest.approx(m.bel[0], rel=1e-9))
