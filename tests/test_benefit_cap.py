"""GAP B -- sojourn-bounded monthly benefit (``State.periodic_benefit_term_months``).

A benefit state with ``periodic_benefit_term_months = cap`` pays its monthly
``disability_income`` only while a cohort's sojourn ``tau < cap``; the lives
stay in force past the cap but stop being paid (a guaranteed-payout LTC /
dementia annuity). ``cap = 0`` is unbounded (the historical behaviour).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import State, Transition, StateModel
from fastcashflow.basis import Basis
from fastcashflow.model_points import ModelPoints

_ZERO = lambda s, a, d: np.full(np.shape(a), 0.0)   # no death / lapse


def _capped_model(cap):
    """active + disabled (benefit) with a sojourn cap; no decrements, so a
    life seated in ``disabled`` stays there and only the cap stops payment."""
    return StateModel(states=(
        State("active", pays_premium=True, transitions=(
            Transition("mortality"), Transition("lapse"))),
        State("disabled", pays_periodic_benefit=True, sojourn_tracking_months=8, periodic_benefit_term_months=cap,
              transitions=(Transition("mortality"),)),
    ), seating=(0, 1))


def _seated_mp(term=12):
    return ModelPoints(
        issue_age=np.array([50], dtype=np.int64),
        benefits={"DEATH": np.array([0.0])},
        premium=np.array([0.0]),
        term_months=np.array([term], dtype=np.int64),
        disability_income=np.array([100.0]),
        state=np.array([1], dtype=np.int64),          # seated in disabled
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})


def _basis(cap):
    return Basis(
        mortality_annual=_ZERO, lapse_annual=_ZERO,
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        state_model=_capped_model(cap),
        coverages=(fcf.CoverageRate("DEATH", _ZERO),))


# ---------------------------------------------------------------------------
# B1 -- hand calc: cap=3 pays exactly the first 3 months, then stops
# ---------------------------------------------------------------------------
def test_benefit_cap_pays_only_within_cap():
    m = fcf.gmm.measure(_seated_mp(term=12), _basis(3))
    cf = m.cashflows.disability_cf[0]
    # tau 0,1,2 paid (100 each), tau>=3 not paid; lives never leave (no decr)
    assert np.allclose(cf[:3], 100.0)
    assert np.allclose(cf[3:], 0.0)
    # discount 0, no other cash flow -> BEL = 3 * 100
    assert m.bel[0] == pytest.approx(300.0)


def test_benefit_cap_unbounded_pays_every_month():
    m = fcf.gmm.measure(_seated_mp(term=12), _basis(0))    # cap=0 unbounded
    cf = m.cashflows.disability_cf[0]
    assert np.allclose(cf, 100.0)                           # all 12 months
    assert m.bel[0] == pytest.approx(1200.0)


def test_benefit_cap_detailed_matches_fused():
    """measure() (detailed) and value() (fused codegen) agree on the capped
    BEL -- both kernel paths apply the cap."""
    mp, basis = _seated_mp(term=12), _basis(3)
    detailed = fcf.gmm.measure(mp, basis, full=True)
    fused = fcf.gmm.measure(mp, basis, full=False)
    assert fused.bel[0] == pytest.approx(detailed.bel[0])
    assert fused.bel[0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# B2 -- monotonicity: a higher cap pays at least as much
# ---------------------------------------------------------------------------
def test_benefit_cap_bel_monotone_in_cap():
    bels = [fcf.gmm.measure(_seated_mp(12), _basis(c)).bel[0] for c in (1, 2, 3)]
    assert bels == [pytest.approx(100.0), pytest.approx(200.0),
                    pytest.approx(300.0)]
    assert bels[0] < bels[1] < bels[2]


# ---------------------------------------------------------------------------
# B3 -- validation
# ---------------------------------------------------------------------------
def test_cap_requires_benefit_state():
    with pytest.raises(ValueError, match="requires pays_periodic_benefit=True"):
        State("x", pays_periodic_benefit=False, sojourn_tracking_months=5, periodic_benefit_term_months=2)


def test_cap_must_be_below_duration_max():
    with pytest.raises(ValueError, match="must exceed the deterministic sojourn boundary"):
        State("x", pays_periodic_benefit=True, sojourn_tracking_months=3, periodic_benefit_term_months=3)


def test_cap_auto_derives_tracking():
    # Omitting sojourn_tracking_months auto-derives one guard cohort past the cap.
    s = State("x", pays_periodic_benefit=True, periodic_benefit_term_months=36)
    assert s.sojourn_tracking_months == 37


def test_cap_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        State("x", pays_periodic_benefit=True, sojourn_tracking_months=5, periodic_benefit_term_months=-1)
