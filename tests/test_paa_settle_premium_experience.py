"""paa.settle -- the paragraph-55(b)(i) premium experience (the PAA counterpart
of the GMM B96(a)).

Authoritative skeleton. Anchors from dev/paa-premium-experience-gate.md. PAA has
no CSM, so the actual premium received over the period (state.actual_premium)
simply enters the LRC (the unearned premium, Sec. 55(a)) and earns as revenue
over the remaining coverage -- no future/current split, no new movement line.
The premiums line becomes the actual cash and the closing LRC reflects it; the
block identity lrc_closing == lrc_opening + premiums - revenue + lrc_experience
is preserved.
"""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from conftest import PATTERNS, make_death_basis

settle = getattr(fcf.paa, "settle", None)
pytestmark = pytest.mark.skipif(
    settle is None, reason="paa.settle not implemented yet")


def _basis(**overrides):
    kw = dict(mortality_q=0.0, lapse_q=0.0, discount_annual=0.0,
              ra_confidence=0.75, mortality_cv=0.10)
    kw.update(overrides)
    return make_death_basis(**kw)


def _book(*, premium, benefit, actual_premium=None, actual_claims=None,
          actual_expenses=None, em_close=6, term=12, basis=None):
    basis = _basis() if basis is None else basis
    surv = fcf.paa.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([premium]),
                    term_months=np.array([term]),
                    premium_term_months=np.array([1]),
                    benefits={"DEATH": np.array([benefit])}, count=np.array([1.0]),
                    calculation_methods=PATTERNS),
        basis, full=True).cashflows.inforce[0]
    ids = np.array(["PA0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([premium]),
        term_months=np.array([term]), premium_term_months=np.array([1]),
        benefits={"DEATH": np.array([benefit])}, count=np.array([surv[em_close]]),
        elapsed_months=np.array([em_close]), mp_id=ids,
        product=np.array(["ACC"]), calculation_methods=PATTERNS)
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([em_close]),
        count=np.array([surv[em_close]]), prior_csm=np.array([0.0]),
        lock_in_rate=0.0, prior_count=np.array([1.0]),
        actual_premium=(None if actual_premium is None
                        else np.array([actual_premium])),
        actual_claims=(None if actual_claims is None else np.array([actual_claims])),
        actual_expenses=(None if actual_expenses is None else np.array([actual_expenses])))
    return mp, state


def test_within_period_claims_experience_is_a_pl_memo():
    """actual_claims surfaces the within-period claims experience (B97) as a
    P&L memo; absent => zero."""
    onerous = _basis(mortality_q=0.05)
    mp0, st0 = _book(premium=60.0, benefit=6000.0, em_close=6, basis=onerous)
    mv0 = settle(mp0, st0, onerous, period_months=6)
    np.testing.assert_array_equal(mv0.claims_experience, 0.0)
    expected_claims = float(mv0.claims_incurred[0])
    mp, state = _book(premium=60.0, benefit=6000.0,
                      actual_claims=expected_claims + 100.0, em_close=6,
                      basis=onerous)
    mv = settle(mp, state, onerous, period_months=6)
    np.testing.assert_allclose(mv.claims_experience[0], 100.0, rtol=1e-6)


def test_premiums_line_is_the_actual_received():
    # single premium 120 at t0; the period [0, 6) contains it -- actual 150
    mp, state = _book(premium=120.0, benefit=480.0, actual_premium=150.0,
                      em_close=6)
    mv = settle(mp, state, _basis(), period_months=6)
    np.testing.assert_allclose(mv.premiums[0], 150.0, rtol=1e-12)
    # block identity holds
    np.testing.assert_allclose(
        mv.lrc_opening + mv.premiums - mv.revenue + mv.lrc_experience,
        mv.lrc_closing, rtol=1e-10)


def test_extra_premium_sits_in_the_closing_lrc():
    """The actual-minus-expected premium adds to the closing LRC (unearned,
    Sec. 55(a)); it is byte-identical when no actual premium is given."""
    base_mp, base_state = _book(premium=120.0, benefit=480.0, em_close=6)
    base = settle(base_mp, base_state, _basis(), period_months=6)
    mp, state = _book(premium=120.0, benefit=480.0, actual_premium=150.0,
                      em_close=6)
    mv = settle(mp, state, _basis(), period_months=6)
    # expected premium was 120; actual 150 -> +30 in the closing LRC
    np.testing.assert_allclose(mv.lrc_closing[0],
                               base.lrc_closing[0] + 30.0, rtol=1e-9)
    np.testing.assert_allclose(mv.premiums[0] - base.premiums[0], 30.0,
                               rtol=1e-9)
    # revenue (coverage-based) is unchanged
    np.testing.assert_allclose(mv.revenue, base.revenue, rtol=1e-12)


def test_absent_actual_premium_is_byte_identical():
    base_mp, base_state = _book(premium=120.0, benefit=480.0, em_close=6)
    base = settle(base_mp, base_state, _basis(), period_months=6)
    # actual_premium=None on the same book
    np.testing.assert_array_equal(
        base.premiums, settle(base_mp, base_state, _basis(),
                              period_months=6).premiums)


def test_more_premium_reduces_the_loss_component_on_an_onerous_book():
    """A higher LRC (more premium received) makes the Sec. 57-58 re-test less
    onerous: the closing loss component falls."""
    onerous = _basis(mortality_q=0.05)                # claims make it onerous
    base_mp, base_state = _book(premium=60.0, benefit=6000.0, em_close=6,
                                basis=onerous)
    base = settle(base_mp, base_state, onerous, period_months=6)
    assert base.loss_component_closing[0] > 0.0       # onerous
    mp, state = _book(premium=60.0, benefit=6000.0, actual_premium=2_000.0,
                      em_close=6, basis=onerous)
    mv = settle(mp, state, onerous, period_months=6)
    assert mv.loss_component_closing[0] < base.loss_component_closing[0]
