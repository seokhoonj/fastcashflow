"""Deterministic transitions -- ``Transition(after_sojourn_months=K, to=...)``.

A cohort advancing into sojourn ``K`` is routed with probability one: ``to=None``
removes it from the in-force set (a cover that ends after a fixed term), ``to=X``
moves it to another state (a guaranteed conversion). Distinct from
``periodic_benefit_term_months`` (which stops the payment but keeps the lives in
force). Semi-Markov only (needs sojourn tracking, auto-derived). Full path only.

Hand-calc timing: a life seated in ``disabled`` at month 0 is at sojourn 0; the
kernel pays the month-t cash flows on start-of-month occupancy, then advances
sojourn tau -> tau+1. So the advance INTO sojourn K happens in month ``K-1``'s
loop: with K=3 the life is paid for months 0,1,2 and routed at the month-2
advance, gone (or moved) from month 3 on.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.multistate import State, Transition, Model
from fastcashflow.basis import Basis
from fastcashflow.model_points import ModelPoints

_ZERO = lambda s, a, d: np.full(np.shape(a), 0.0)   # no death / lapse decrement


def _model(*, after=0, to=None, cap=0, lump=False, benefit=True,
           sojourn_tracking_months=8, recovery=None):
    """active + disabled; a life seated in ``disabled`` stays (no decrement)
    until a deterministic transition routes it. Optional ``recovery`` rate adds
    a competing disabled->active decrement (for the no-double-count test)."""
    disabled_trs = [Transition("mortality")]
    if recovery is not None:
        disabled_trs.append(Transition("recovery", to="active"))
    if after:
        disabled_trs.append(Transition(after_sojourn_months=after, to=to,
                                       pays_lump_sum=lump))
    return Model(states=(
        State("active", pays_premium=True, transitions=(
            Transition("mortality"), Transition("lapse"))),
        State("disabled", pays_periodic_benefit=benefit,
              sojourn_tracking_months=sojourn_tracking_months,
              periodic_benefit_term_months=cap, transitions=tuple(disabled_trs)),
    ), seating=(0, 1))


def _seated_mp(term=12, lump_amount=0.0):
    return ModelPoints(
        issue_age=np.array([50], dtype=np.int64),
        benefits={"DEATH": np.array([0.0])},
        premium=np.array([0.0]),
        term_months=np.array([term], dtype=np.int64),
        disability_income=np.array([100.0]),
        disability_benefit=np.array([lump_amount]),
        state=np.array([1], dtype=np.int64),          # seated in disabled
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})


def _basis(model):
    return Basis(
        mortality_annual=_ZERO, lapse_annual=_ZERO,
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        state_machine=model, coverages=(fcf.CoverageRate("DEATH", _ZERO),))


# ---------------------------------------------------------------------------
# T1 -- exit (to=None) drops the cohort at the boundary (regression anchor)
# ---------------------------------------------------------------------------
def test_t1_exit_drops_cohort_at_boundary():
    m = fcf.gmm.measure(_seated_mp(12), _basis(_model(after=3, to=None)), full=True)
    ife = m.cashflows.inforce[0]
    assert np.allclose(ife[:3], 1.0)     # present sojourn 0,1,2
    assert np.allclose(ife[3:], 0.0)     # routed out at the month-2 advance


# ---------------------------------------------------------------------------
# T2 -- the boundary uses ``>=`` (advance target): no re-accretion ever
# ---------------------------------------------------------------------------
def test_t2_no_reaccretion_after_boundary():
    m = fcf.gmm.measure(_seated_mp(24), _basis(_model(after=3, to=None)), full=True)
    ife = m.cashflows.inforce[0]
    assert np.allclose(ife[3:], 0.0)     # gone for the whole horizon, never returns


# ---------------------------------------------------------------------------
# T3/T4 -- deterministic MOVE (to="active"): conserved, no double-count
# ---------------------------------------------------------------------------
def test_t4_deterministic_move_conserves_inforce():
    m = fcf.gmm.measure(_seated_mp(12),
                        _basis(_model(after=3, to="active")), full=True)
    ife = m.cashflows.inforce[0]
    dis = m.cashflows.disability_cf[0]
    # total in-force is conserved at 1 every month (the life MOVES, not exits) --
    # if the routed flow were double-counted in-force would exceed 1.
    assert np.allclose(ife, 1.0)
    # the periodic benefit (disabled-only) stops once the life is in active
    assert np.allclose(dis[:3], 100.0)   # paid while in disabled (sojourn 0,1,2)
    assert np.allclose(dis[3:], 0.0)     # in active from month 3 -> no benefit


# ---------------------------------------------------------------------------
# T5 -- cap vs exit independence (pay the cap, keep cover, then exit later)
# ---------------------------------------------------------------------------
def test_t5_cap_and_exit_independent():
    m = fcf.gmm.measure(_seated_mp(12),
                        _basis(_model(after=4, to=None, cap=2)), full=True)
    ife = m.cashflows.inforce[0]
    dis = m.cashflows.disability_cf[0]
    assert np.allclose(dis[:2], 100.0)   # paid sojourn 0,1 (cap=2)
    assert np.allclose(dis[2:], 0.0)     # cap stops the payment...
    assert np.allclose(ife[:4], 1.0)     # ...but the lives stay in force
    assert np.allclose(ife[4:], 0.0)     # cover ends at the exit boundary (4)


def test_t5_auto_derives_tracking():
    # cap 2, exit 4 -> boundary 4 -> auto sojourn_tracking_months 5
    s = State("disabled", pays_periodic_benefit=True, periodic_benefit_term_months=2,
              transitions=(Transition("mortality"),
                           Transition(after_sojourn_months=4, to=None)))
    assert s.sojourn_tracking_months == 5


# ---------------------------------------------------------------------------
# T6 -- lump on the deterministic exit fires on the routed flow at the boundary
# ---------------------------------------------------------------------------
def test_t6_lump_on_exit():
    mp = _seated_mp(12, lump_amount=5000.0)
    m = fcf.gmm.measure(mp, _basis(_model(after=3, to=None, lump=True,
                                          benefit=False)), full=True)
    dis = m.cashflows.disability_cf[0]
    # benefit=False -> no periodic income; the only disability cash flow is the
    # lump, fired on the routed flow (=1.0) at the month-2 advance into sojourn 3.
    assert np.isclose(dis[2], 5000.0)
    assert np.allclose(np.delete(dis, 2), 0.0)


# ---------------------------------------------------------------------------
# Validation -- the API makes silently-wrong compositions unexpressible
# ---------------------------------------------------------------------------
def test_transition_rate_vs_deterministic_mutual_exclusion():
    with pytest.raises(ValueError, match="carries no rate"):
        Transition(rate="recovery", after_sojourn_months=5)
    with pytest.raises(ValueError, match="needs a rate"):
        Transition()
    with pytest.raises(ValueError, match="already keyed"):
        Transition(after_sojourn_months=5, sojourn_dependent=True)


def test_at_most_one_deterministic_transition_per_state():
    with pytest.raises(ValueError, match="at most one deterministic transition"):
        State("x", transitions=(Transition(after_sojourn_months=3, to=None),
                                Transition(after_sojourn_months=4, to=None)))


def test_deterministic_must_clear_the_cap():
    with pytest.raises(ValueError, match=">= "):
        State("x", pays_periodic_benefit=True, periodic_benefit_term_months=6,
              transitions=(Transition(after_sojourn_months=3, to=None),))


def test_explicit_tracking_at_or_below_boundary_rejected():
    with pytest.raises(ValueError, match="must exceed the deterministic sojourn boundary"):
        State("x", sojourn_tracking_months=3,
              transitions=(Transition(after_sojourn_months=3, to=None),))


def test_after_sojourn_auto_derives_tracking():
    s = State("x", transitions=(Transition(after_sojourn_months=6, to=None),))
    assert s.sojourn_tracking_months == 7        # boundary 6 + 1 guard cohort


# ---------------------------------------------------------------------------
# Path restrictions -- deterministic transitions are full-path only in v1
# ---------------------------------------------------------------------------
def test_fast_path_auto_routes_deterministic_transition():
    # full=False auto-routes a deterministic-transition book to the full kernel
    # (no longer raises) -- byte-identical to full=True.
    mp, b = _seated_mp(12), _basis(_model(after=3, to=None))
    fast = fcf.gmm.measure(mp, b, full=False)
    full = fcf.gmm.measure(mp, b, full=True)
    assert np.isclose(float(fast.bel[0]), float(full.bel[0]))
    # a periodic_benefit_term_months-only model still runs the genuine fast path
    cap_model = _model(cap=3)
    fast2 = fcf.gmm.measure(_seated_mp(12), _basis(cap_model), full=False)
    full2 = fcf.gmm.measure(_seated_mp(12), _basis(cap_model), full=True)
    assert np.isclose(float(fast2.bel[0]), float(full2.bel[0]))


def test_markov_compile_rejects_deterministic_transition():
    from fastcashflow.multistate import compile_model
    # force a Markov compile path on a model carrying a deterministic transition
    # by bypassing the auto-derive (which would make it semi-Markov)
    model = Model(states=(
        State("a", pays_premium=True, transitions=(Transition("mortality"),)),
    ), seating=(0,))
    object.__setattr__(model.states[0], "transitions",
                       (Transition("mortality"), Transition(after_sojourn_months=3, to=None)))
    with pytest.raises(ValueError, match="semi-Markov only"):
        compile_model(model, {"mortality": np.zeros((1, 1)), "lapse": np.zeros((1, 1))})
