"""reinsurance loss-recovery component -- IFRS 17 paragraphs 66A-66B.

Authoritative skeleton. Anchors from dev/loss-recovery-66ab-gate.md and the
hand-calc dev/scratch_loss_recovery_gate.py, which validates the inception CSM
adjustment against IASB AP2C (Dec 2019) Appendix A Examples 1-3:

    inception (66A):  csm_after = csm0 - loss_recovery_component
    loss_recovery   = underlying_loss_component x claim_recovery_%   (B95B/B119D)
    subsequent (B119F): a separate tracked balance, amortised in lock-step with
                        the underlying loss component (reversals in P&L), the
                        CSM is NOT re-adjusted.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from conftest import make_death_basis, PATTERNS

QS = fcf.reinsurance.QuotaShare
settle = getattr(fcf.reinsurance, "settle", None)
_HAS_LR = (settle is not None
           and hasattr(fcf.reinsurance.measure(
               ModelPoints.single(40, 400_000.0, 240, benefits={0: 1e8},
                                  calculation_methods=PATTERNS),
               make_death_basis(mortality_q=0.002, lapse_q=0.005,
                                discount_annual=0.03, ra_confidence=0.75,
                                mortality_cv=0.10),
               treaty=QS(0.4)), "loss_recovery_component"))
pytestmark = pytest.mark.skipif(
    not _HAS_LR, reason="reinsurance loss-recovery (66A-66B) not implemented yet")


def _basis():
    return make_death_basis(mortality_q=0.002, lapse_q=0.005,
                            discount_annual=0.03, ra_confidence=0.75,
                            mortality_cv=0.10)


def _unit(basis, *, premium=400_000.0):
    return ModelPoints.single(40, premium, 240, benefits={0: 1e8},
                              calculation_methods=PATTERNS)


# ---------------------------------------------------------------------------
# measure (66A): csm_after = csm0 - loss_recovery
# ---------------------------------------------------------------------------

def test_measure_csm_is_reduced_by_the_loss_recovery():
    basis, treaty = _basis(), QS(0.4)
    unit = _unit(basis)
    base = fcf.reinsurance.measure(unit, basis, treaty=treaty)
    underlying_loss = 1_000_000.0
    lr = fcf.reinsurance.measure(unit, basis, treaty=treaty,
                                 underlying_loss_component=underlying_loss)
    # csm_after = csm0 - underlying_loss x cession (AP2C: csm0 - LR)
    expected_recovery = underlying_loss * 0.4
    np.testing.assert_allclose(lr.loss_recovery_component[0], expected_recovery,
                               rtol=1e-12)
    np.testing.assert_allclose(lr.csm[0], base.csm[0] - expected_recovery,
                               rtol=1e-9, atol=1e-3)


def test_measure_absent_underlying_is_byte_identical():
    basis, treaty = _basis(), QS(0.4)
    unit = _unit(basis)
    base = fcf.reinsurance.measure(unit, basis, treaty=treaty)
    np.testing.assert_array_equal(base.loss_recovery_component, 0.0)


def test_recovery_percentage_override():
    basis, treaty = _basis(), QS(0.4)
    unit = _unit(basis)
    lr = fcf.reinsurance.measure(unit, basis, treaty=treaty,
                                 underlying_loss_component=1_000_000.0,
                                 recovery_percentage=0.25)   # not the cession
    np.testing.assert_allclose(lr.loss_recovery_component[0], 250_000.0,
                               rtol=1e-12)


# ---------------------------------------------------------------------------
# settle (B119F): the loss-recovery tracking lines, lock-step amortisation
# ---------------------------------------------------------------------------

def _settle_book(*, elapsed=36, period=12, cession=0.4):
    basis, treaty = _basis(), QS(cession)
    unit = _unit(basis)
    m = fcf.reinsurance.measure(unit, basis, treaty=treaty)
    surv = m.cashflows.inforce[0]
    csm_seed = float(m.csm_path[0, elapsed - period])
    scale = 1000.0
    ids = np.array(["R0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([400_000.0]),
        term_months=np.array([240]), benefits={0: np.array([1e8])},
        count=np.array([scale * surv[elapsed]]),
        elapsed_months=np.array([elapsed]), mp_id=ids,
        calculation_methods=PATTERNS)
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([elapsed]),
        count=np.array([scale * surv[elapsed]]), prior_csm=np.array([csm_seed * scale]),
        lock_in_rate=basis.discount_annual,
        prior_count=np.array([scale * surv[elapsed - period]]))
    return mp, state, basis, treaty


def test_settle_loss_recovery_identity_and_lock_step():
    mp, state, basis, treaty = _settle_book()
    # the underlying loss amortises 200,000 -> 150,000 over the period
    mv = settle(mp, state, basis, treaty=treaty, period_months=12,
                underlying_loss_opening=200_000.0,
                underlying_loss_closing=150_000.0)
    np.testing.assert_allclose(mv.loss_recovery_opening[0], 200_000.0 * 0.4)
    np.testing.assert_allclose(mv.loss_recovery_closing[0], 150_000.0 * 0.4)
    # underlying loss fell -> the recovery reverses in P&L
    np.testing.assert_allclose(mv.loss_recovery_reversed[0],
                               (200_000.0 - 150_000.0) * 0.4)
    np.testing.assert_array_equal(mv.loss_recovery_recognised, 0.0)
    np.testing.assert_allclose(
        mv.loss_recovery_opening + mv.loss_recovery_recognised
        - mv.loss_recovery_reversed, mv.loss_recovery_closing, rtol=1e-12)


def test_settle_csm_recursion_is_unchanged_by_loss_recovery():
    """The loss-recovery is a separate balance -- the four-term CSM recursion
    and the three-term cross-tie are unaffected."""
    mp, state, basis, treaty = _settle_book()
    mv = settle(mp, state, basis, treaty=treaty, period_months=12,
                underlying_loss_opening=200_000.0,
                underlying_loss_closing=150_000.0)
    np.testing.assert_allclose(
        mv.csm_opening + mv.csm_accretion + mv.csm_experience_unlocking
        - mv.csm_release, mv.csm_closing, rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(
        mv.csm_experience_unlocking + mv.finance_wedge,
        -(mv.bel_experience + mv.ra_experience), rtol=1e-9, atol=1e-3)


def test_settle_absent_is_byte_identical():
    mp, state, basis, treaty = _settle_book()
    mv = settle(mp, state, basis, treaty=treaty, period_months=12)
    for nm in ("loss_recovery_opening", "loss_recovery_recognised",
               "loss_recovery_reversed", "loss_recovery_closing"):
        np.testing.assert_array_equal(getattr(mv, nm), 0.0)


def test_reconciliation_and_aggregate_carry_the_lines():
    mp, state, basis, treaty = _settle_book()
    mv = settle(mp, state, basis, treaty=treaty, period_months=12,
                underlying_loss_opening=200_000.0,
                underlying_loss_closing=150_000.0)
    recon = fcf.reconcile([mv])[0]
    assert hasattr(recon, "loss_recovery_closing")
    np.testing.assert_allclose(
        recon.loss_recovery_reversed, -float(mv.loss_recovery_reversed.sum()),
        rtol=1e-12)
    agg = fcf.reinsurance.settle_aggregate(
        mp, state, basis, treaty=treaty, period_months=12,
        underlying_loss_opening=200_000.0, underlying_loss_closing=150_000.0)
    np.testing.assert_allclose(agg.loss_recovery_closing,
                               float(mv.loss_recovery_closing.sum()), rtol=1e-9)
