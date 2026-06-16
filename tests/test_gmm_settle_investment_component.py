"""gmm.settle -- B96(c) investment-component separation.

Authoritative skeleton. Anchors from dev/b96c-investment-component-gate.md and
the hand-calc oracle dev/scratch_b96c_gate.py.

Gap A -- the B96(c) experience adjustment: the expected less the actual
investment component (surrender / annuity repayments) payable over the period,
the WHOLE difference into the CSM (no fraction). ``csm_investment_experience =
expected_ic - actual_ic``; an extra payout (actual > expected) is unfavourable.

Gap B -- the paragraph-50(a) LC allocation pool EXCLUDES the investment-
component streams (surrender / maturity / annuity), per 51(a) / 85. A
pure-protection book has none, so those numbers are unchanged; a book with
surrender values amortises its loss component against the claims+expenses pool
only, and paragraph 52 still runs the loss component to zero.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, InforceState, ModelPoints)
from fastcashflow.movement import GMMSettlementMovement

settle = getattr(fcf.gmm, "settle", None)
_HAS_B96C = (settle is not None
             and "csm_investment_experience"
             in GMMSettlementMovement.__dataclass_fields__)
pytestmark = pytest.mark.skipif(
    not _HAS_B96C, reason="gmm.settle B96(c) not implemented yet")

CM = {"DEATH": CalculationMethod.DEATH}


def _flat(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis(*, discount=0.03, surrender=False):
    kw = {}
    if surrender:
        kw.update(surrender_value_curve=np.full(48, 30_000.0),
                  surrender_value_basis="amount_per_policy")
    return Basis(
        mortality_annual=_flat(0.012), lapse_annual=_flat(0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat(0.012)),), **kw)


def _book(basis, *, em_open=12, period=12, term=48, scale=1000.0,
          prior_csm=5_000.0, lc_open=0.0, actual_ic=None, final=False):
    em_close = term if final else em_open + period
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=CM),
        basis, full=True).cashflows.inforce[0]
    prior_count = scale * surv[em_open]
    surv_c = surv[em_close] if em_close < surv.shape[0] else 0.0
    count_close = 0.0 if final else scale * surv_c
    ids = np.array(["P0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
        count=np.array([count_close]), elapsed_months=np.array([em_close]),
        mp_id=ids, product=np.array(["A"]), calculation_methods=CM)
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([em_close]),
        count=np.array([count_close]), prior_csm=np.array([prior_csm]),
        lock_in_rate=basis.discount_annual, prior_count=np.array([prior_count]),
        prior_loss_component=(np.array([lc_open]) if lc_open else None),
        actual_investment_component=(None if actual_ic is None
                                     else np.array([actual_ic])))
    return mp, state


# ---------------------------------------------------------------------------
# Gap A -- the experience adjustment
# ---------------------------------------------------------------------------

def test_absent_actual_ic_is_zero_and_csm_recursion_holds():
    basis = _basis(surrender=True)
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_array_equal(mv.csm_investment_experience, 0.0)
    np.testing.assert_allclose(
        mv.csm_opening + mv.csm_accretion + mv.csm_experience_unlocking
        + mv.csm_premium_experience + mv.csm_investment_experience
        - mv.loss_component_reversed + mv.loss_component_recognised
        - mv.csm_release, mv.csm_closing, rtol=1e-10)


def test_more_ic_paid_than_expected_is_unfavourable():
    """actual > expected investment component => CSM down (an extra deposit
    payout); actual < expected => CSM up (retained business)."""
    basis = _basis(surrender=True)
    # Probe the expected IC: with actual = 0, csm_investment_experience equals
    # the whole expected (expected - 0).
    mp0, st0 = _book(basis, actual_ic=0.0)
    expected_ic = float(settle(mp0, st0, basis,
                               period_months=12).csm_investment_experience[0])
    assert expected_ic > 0.0
    # actual above expected -> unfavourable (negative).
    mp_hi, st_hi = _book(basis, actual_ic=expected_ic + 50_000.0)
    mv_hi = settle(mp_hi, st_hi, basis, period_months=12)
    np.testing.assert_allclose(mv_hi.csm_investment_experience, -50_000.0,
                               rtol=1e-7)
    # actual below expected -> favourable (positive).
    mp_lo, st_lo = _book(basis, actual_ic=expected_ic - 50_000.0)
    mv_lo = settle(mp_lo, st_lo, basis, period_months=12)
    np.testing.assert_allclose(mv_lo.csm_investment_experience, 50_000.0,
                               rtol=1e-7)


def test_investment_experience_is_outside_the_three_term_tie():
    """The B96(c) leg is a new future-service change with no BEL/RA
    counterpart, so the GMM three-term cross-identity still holds."""
    basis = _basis(surrender=True)
    mp, state = _book(basis, actual_ic=200_000.0)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(
        mv.csm_experience_unlocking + mv.finance_wedge,
        -(mv.bel_experience + mv.ra_experience), rtol=1e-9)


# ---------------------------------------------------------------------------
# Gap B -- the LC pool excludes investment components
# ---------------------------------------------------------------------------

def test_pure_protection_lc_allocation_is_unchanged_by_b96c():
    """A book with no investment-component streams (no surrender / maturity /
    annuity) amortises its loss component exactly as before -- the IC exclusion
    is a no-op when there is no IC."""
    basis = _basis(surrender=False)
    mp, state = _book(basis, lc_open=40_000.0, prior_csm=0.0)
    mv = settle(mp, state, basis, period_months=12)
    assert np.all(mv.loss_component_amortised > 0.0)
    # reconciliation identity of the loss component still holds
    np.testing.assert_allclose(
        mv.loss_component_opening + mv.loss_component_finance
        - mv.loss_component_amortised - mv.loss_component_reversed
        + mv.loss_component_recognised, mv.loss_component_closing, rtol=1e-10)


def test_paragraph52_still_runs_to_zero_with_surrender_values():
    """With investment-component (surrender) streams present and excluded from
    the pool, the final settlement still lands the loss component at zero."""
    basis = _basis(surrender=True)
    mp, state = _book(basis, em_open=36, period=12, term=48,
                      lc_open=40_000.0, prior_csm=0.0, final=True)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.loss_component_closing, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# the line threads through the reconciliation and the aggregate
# ---------------------------------------------------------------------------

def test_reconciliation_and_aggregate_carry_the_line():
    basis = _basis(surrender=True)
    mp, state = _book(basis, actual_ic=200_000.0)
    mv = settle(mp, state, basis, period_months=12)
    recon = fcf.reconcile([mv])[0]
    assert hasattr(recon, "csm_investment_experience")
    np.testing.assert_allclose(
        recon.csm_investment_experience,
        float(mv.csm_investment_experience.sum()), rtol=1e-10)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    np.testing.assert_allclose(
        agg.csm_investment_experience,
        float(mv.csm_investment_experience.sum()), rtol=1e-9)
