"""F6 -- state-conditioned death benefit (``State.death_benefit_factor``).

A state scales the death-coverage benefit paid for the lives residing in it,
occupancy-weighted: ``claim = (sum_s occ[s]*factor[s]) * claim_rate``. It
multiplies the benefit AMOUNT, not the decrement, so the death count is
unchanged. Default ``1.0`` is bit-identical to today. Full path only -- the
fast and VFA paths reject a non-default factor.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import State, Transition, StateModel
from fastcashflow.basis import Basis
from fastcashflow.model_points import ModelPoints

_FLAT = lambda v: (lambda s, a, d: np.full(np.shape(a), v))
_ZERO = _FLAT(0.0)


def _two_state(factor):
    """healthy(0) + post(1); the post state pays ``factor`` x the death benefit."""
    return StateModel(states=(
        State("healthy", pays_premium=True, transitions=(
            Transition("mortality"), Transition("lapse"))),
        State("post", transitions=(Transition("mortality"),),
              death_benefit_factor=factor),
    ), seating=(0, 1))


def _seated_mp(state, term=12, benefit=100_000.0):
    return ModelPoints(
        issue_age=np.array([50], dtype=np.int64),
        benefits={"DEATH": np.array([benefit])},
        premium=np.array([0.0]),
        term_months=np.array([term], dtype=np.int64),
        state=np.array([state], dtype=np.int64),
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})


def _basis(factor):
    return Basis(
        mortality_annual=_FLAT(0.10), lapse_annual=_ZERO,
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        state_model=_two_state(factor),
        coverages=(fcf.CoverageRate("DEATH", _FLAT(0.10)),))


# ---------------------------------------------------------------------------
# F6.1 -- hand calc: seated in post, factor 2.0 doubles the death claim exactly
# ---------------------------------------------------------------------------
def test_post_factor_doubles_death_claim():
    """Lives wholly in the post state (factor 2.0): every month's death claim
    is exactly 2x the factor-1.0 claim, and the BEL doubles."""
    mp = _seated_mp(1)                       # seated in post
    f1 = fcf.gmm.measure(mp, _basis(1.0), full=True)
    f2 = fcf.gmm.measure(mp, _basis(2.0), full=True)
    c1 = f1.cashflows.mortality_cf[0]
    c2 = f2.cashflows.mortality_cf[0]
    assert np.all(c1 > 0.0)
    assert np.allclose(c2, 2.0 * c1, rtol=1e-12)
    assert f2.bel[0] == pytest.approx(2.0 * f1.bel[0], rel=1e-12)


# ---------------------------------------------------------------------------
# F6.2 -- occupancy-weighting: a factor on a state with no occupancy is inert
# ---------------------------------------------------------------------------
def test_factor_is_occupancy_weighted():
    """The post-state factor only weights lives in post. Seated in healthy
    (which never reaches post here -- the death transition exits), the claim is
    identical with or without the factor: it applies per occupancy, not
    globally."""
    mp = _seated_mp(0)                       # seated in healthy, factor 2 on post
    healthy = fcf.gmm.measure(mp, _basis(2.0), full=True).cashflows.mortality_cf[0]
    control = fcf.gmm.measure(mp, _basis(1.0), full=True).cashflows.mortality_cf[0]
    assert np.allclose(healthy, control, rtol=1e-12)


# ---------------------------------------------------------------------------
# F6.3 -- default factor 1.0 is bit-identical (no factor field == factor 1.0)
# ---------------------------------------------------------------------------
def test_default_factor_is_bit_identical():
    """A model that declares no factor and one that sets 1.0 give the SAME
    claim element-for-element (==, not approx)."""
    mp = _seated_mp(1)
    plain = StateModel(states=(
        State("healthy", pays_premium=True, transitions=(
            Transition("mortality"), Transition("lapse"))),
        State("post", transitions=(Transition("mortality"),)),   # no factor
    ), seating=(0, 1))
    b_plain = Basis(
        mortality_annual=_FLAT(0.10), lapse_annual=_ZERO,
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        state_model=plain, coverages=(fcf.CoverageRate("DEATH", _FLAT(0.10)),))
    a = fcf.gmm.measure(mp, b_plain, full=True).cashflows.mortality_cf[0]
    b = fcf.gmm.measure(mp, _basis(1.0), full=True).cashflows.mortality_cf[0]
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# F6.4 -- no split-brain: the death count is unchanged by the benefit factor
# ---------------------------------------------------------------------------
def test_deaths_count_unchanged_by_factor():
    """The factor scales money, not bodies -- the deaths reporter is identical
    at factor 1.0 and 2.0."""
    mp = _seated_mp(1)
    d1 = fcf.gmm.measure(mp, _basis(1.0), full=True).cashflows.deaths[0]
    d2 = fcf.gmm.measure(mp, _basis(2.0), full=True).cashflows.deaths[0]
    assert np.array_equal(d1, d2)


# ---------------------------------------------------------------------------
# F6.5 -- the fast path auto-routes a non-default factor to the full kernel
# ---------------------------------------------------------------------------
def test_fast_path_auto_routes_factor():
    mp = _seated_mp(1)
    # full=False auto-routes a death_benefit_factor book to full -- equal, not raise
    fast = fcf.gmm.measure(mp, _basis(2.0), full=False)
    full = fcf.gmm.measure(mp, _basis(2.0), full=True)
    assert fast.bel[0] == pytest.approx(full.bel[0], rel=1e-9)
    # a factor-1.0 model still runs the genuine fast path
    fast1 = fcf.gmm.measure(mp, _basis(1.0), full=False)
    full1 = fcf.gmm.measure(mp, _basis(1.0), full=True)
    assert fast1.bel[0] == pytest.approx(full1.bel[0], rel=1e-9)


# ---------------------------------------------------------------------------
# F6.6 -- validation: a negative factor is rejected at construction
# ---------------------------------------------------------------------------
def test_negative_factor_rejected():
    with pytest.raises(ValueError, match="death_benefit_factor must be"):
        State("x", death_benefit_factor=-1.0)


# ---------------------------------------------------------------------------
# F6.6b -- the VFA path rejects a non-default factor (the silent-wrong hole)
# ---------------------------------------------------------------------------
def test_vfa_path_rejects_factor():
    """vfa.measure pays the GMDB/GMAB floor on the occupancy decrement, which
    never reads the GMM death-claim factor -- a factor here would be silently
    ignored, so it is rejected."""
    vfa_basis = Basis(
        mortality_annual=_FLAT(0.10), lapse_annual=_ZERO, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        investment_return=0.06, fund_fee=0.015,
        state_model=_two_state(2.0),
        coverages=(fcf.CoverageRate("DEATH", _FLAT(0.10)),))
    mp = ModelPoints.single(40, 0.0, 60, account_value=1e6)
    with pytest.raises(NotImplementedError, match="death_benefit"):
        fcf.vfa.measure(mp, vfa_basis)


# ---------------------------------------------------------------------------
# F6.7 -- F6 mixed with a rule-bearing DEATH coverage is rejected
# ---------------------------------------------------------------------------
def test_factor_with_rule_death_coverage_rejected():
    """A waiting-period (rule-bearing) DEATH coverage pays off plain in-force,
    so combining it with a per-state factor would weight one death claim and
    not the other -- rejected in v1."""
    mp = ModelPoints(
        issue_age=np.array([50], dtype=np.int64),
        premium=np.array([0.0]),
        term_months=np.array([24], dtype=np.int64),
        state=np.array([1], dtype=np.int64),
        coverage_index=np.array([0], dtype=np.int64),
        coverage_amount=np.array([100_000.0]),
        coverage_offset=np.array([0, 1], dtype=np.int64),
        coverage_waiting=np.array([3], dtype=np.int64),         # rule-bearing
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
    with pytest.raises(ValueError, match="state-conditioned death benefit"):
        fcf.gmm.measure(mp, _basis(2.0), full=True)
