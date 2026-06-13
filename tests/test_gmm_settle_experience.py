"""gmm.settle -- within-period claims and expense experience (Sec. B97(b)/(c)).

Authoritative skeleton. The v1 settle assumes within-period cash flows equal
expected (only the closing count, premium and investment component are
observed). state.actual_claims / state.actual_expenses surface the remaining
experience: the actual-minus-expected difference relates to past/current
service (B97), recognised in the insurance service result (P&L) as
claims_experience / expense_experience -- NOT the CSM, NOT a balance recursion
(a memo, like premium_experience_revenue). Zero unless the inputs are given.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, InforceState, ModelPoints)
from fastcashflow.movement import GMMSettlementMovement

settle = getattr(fcf.gmm, "settle", None)
_HAS = (settle is not None
        and "claims_experience" in GMMSettlementMovement.__dataclass_fields__)
pytestmark = pytest.mark.skipif(
    not _HAS, reason="gmm.settle within-period experience not implemented yet")

CM = {"DEATH": CalculationMethod.DEATH}


def _flat(v):
    return lambda s, ia, d: np.full(d.shape, v, dtype=np.float64)


def _basis():
    return Basis(
        mortality_annual=_flat(0.012), lapse_annual=_flat(0.05),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat(0.012)),))


def _book(*, actual_claims=None, actual_expenses=None, em_open=12, period=12,
          term=36, scale=1000.0, prior_csm=5_000.0):
    basis = _basis()
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([term]), benefits={0: np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=CM),
        basis, full=True).cashflows.inforce[0]
    em_close = em_open + period
    ids = np.array(["P0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([term]), benefits={0: np.array([1e6])},
        count=np.array([scale * surv[em_close]]),
        elapsed_months=np.array([em_close]), mp_id=ids,
        product=np.array(["A"]), calculation_methods=CM)
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([em_close]),
        count=np.array([scale * surv[em_close]]), prior_csm=np.array([prior_csm]),
        lock_in_rate=0.03, prior_count=np.array([scale * surv[em_open]]),
        actual_claims=(None if actual_claims is None else np.array([actual_claims])),
        actual_expenses=(None if actual_expenses is None else np.array([actual_expenses])))
    return mp, state, basis


def test_absent_is_zero_and_balances_unchanged():
    mp, state, basis = _book()
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_array_equal(mv.claims_experience, 0.0)
    np.testing.assert_array_equal(mv.expense_experience, 0.0)
    # baseline balances
    base_csm = float(mv.csm_closing[0])
    base_bel = float(mv.bel_closing[0])
    mp2, st2, _ = _book(actual_claims=999_999.0, actual_expenses=5_000.0)
    mv2 = settle(mp2, st2, basis, period_months=12)
    # the experience is a P&L memo -- it does NOT move the CSM / BEL / RA
    np.testing.assert_allclose(mv2.csm_closing[0], base_csm, rtol=1e-12)
    np.testing.assert_allclose(mv2.bel_closing[0], base_bel, rtol=1e-12)


def test_claims_experience_is_actual_minus_expected():
    # probe expected claims (actual=0 -> claims_experience = -expected)
    mp0, st0, basis = _book(actual_claims=0.0)
    mv0 = settle(mp0, st0, basis, period_months=12)
    expected_claims = -float(mv0.claims_experience[0])
    assert expected_claims > 0.0
    # equals the LIC claims_incurred (the expected within-period claims)
    np.testing.assert_allclose(expected_claims, float(mv0.claims_incurred[0]),
                               rtol=1e-12)
    # actual above expected -> positive experience (more claims than expected)
    mp, state, _ = _book(actual_claims=expected_claims + 40_000.0)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.claims_experience[0], 40_000.0, rtol=1e-7)


def test_expense_experience_is_actual_minus_expected():
    mp0, st0, basis = _book(actual_expenses=0.0)
    expected_exp = -float(settle(mp0, st0, basis,
                                 period_months=12).expense_experience[0])
    # a flat-rate book with no expense items has zero expected expense
    mp, state, _ = _book(actual_expenses=expected_exp + 7_500.0)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.expense_experience[0], 7_500.0, rtol=1e-7)


def test_csm_recursion_excludes_the_experience_memos():
    """claims/expense experience are P&L memos -- they are not in the CSM
    recursion (B97: past/current service does not adjust the CSM)."""
    mp, state, basis = _book(actual_claims=200_000.0, actual_expenses=3_000.0)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(
        mv.csm_opening + mv.csm_accretion + mv.csm_experience_unlocking
        + mv.csm_premium_experience + mv.csm_investment_experience
        - mv.loss_component_reversed + mv.loss_component_recognised
        - mv.csm_release, mv.csm_closing, rtol=1e-10)


def test_reconciliation_and_aggregate_carry_the_lines():
    mp, state, basis = _book(actual_claims=200_000.0, actual_expenses=3_000.0)
    mv = settle(mp, state, basis, period_months=12)
    recon = fcf.reconcile([mv])[0]
    np.testing.assert_allclose(recon.claims_experience,
                               float(mv.claims_experience.sum()), rtol=1e-10)
    np.testing.assert_allclose(recon.expense_experience,
                               float(mv.expense_experience.sum()), rtol=1e-10)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    np.testing.assert_allclose(agg.claims_experience,
                               float(mv.claims_experience.sum()), rtol=1e-9)
