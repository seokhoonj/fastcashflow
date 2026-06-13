"""vfa.settle -- the liability-for-incurred-claims (LIC) block under a
settlement_pattern basis (paragraphs 40(b) / 42 / 103(b)).

Authoritative skeleton. The VFA mirror of test_gmm_settle_lic.py: vfa.settle
previously rejected a settlement_pattern basis; this adds the four LIC movement
lines, built from the VFA benefit_cf incurred stream, entirely expected-scale
with claims_paid the residual::

    lic_closing == lic_opening + claims_incurred - claims_paid
"""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CoverageRate, ExpenseItem, InforceState, ModelPoints
from fastcashflow.movement import VFASettlementMovement

settle = getattr(fcf.vfa, "settle", None)
_HAS_LIC = (settle is not None
            and "lic_closing" in VFASettlementMovement.__dataclass_fields__)
pytestmark = pytest.mark.skipif(
    not _HAS_LIC,
    reason="vfa.settle LIC block not implemented yet")

PATTERN = np.array([0.6, 0.3, 0.1])


def _basis(*, settlement_pattern=None):
    death_fn = lambda s, ia, d: np.full(s.shape, 0.012)
    return Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=0.05, ra_confidence=0.75, mortality_cv=0.0,
        expense_cv=0.10, investment_return=0.05, fund_fee=0.015,
        expense_items=(ExpenseItem("maintenance", "gamma_fixed", 1_000.0),),
        settlement_pattern=settlement_pattern,
        coverages=(CoverageRate("DEATH", death_fn),),
    )


def _growth(basis, mp):
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0
    return (1.0 + r_m) * (1.0 - f_m)


def _book(basis, *, em_open=6, period=6):
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6)
    em_close = em_open + period
    m0 = fcf.vfa.measure(mp0, basis)
    inforce = m0.cashflows.inforce
    rows = np.arange(1)
    growth = _growth(basis, mp0)
    av_open = np.asarray(mp0.account_value, dtype=np.float64) * growth ** em_open
    av_close = av_open * growth ** period
    inforce_pad = np.concatenate([inforce, np.zeros((1, 1))], axis=1)
    count_open = inforce[rows, em_open]
    boundary = np.asarray(mp0.contract_boundary_months)
    count_close = inforce_pad[rows, np.minimum(em_close, boundary)]
    ids = np.array(["P0"])
    mp = replace(mp0, mp_id=ids,
                 elapsed_months=np.full(1, em_close, dtype=np.int64),
                 count=np.asarray(count_close, dtype=np.float64))
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(1, em_close, dtype=np.int64),
        count=np.asarray(count_close, dtype=np.float64),
        prior_csm=m0.csm_path[rows, em_open], lock_in_rate=0.0,
        account_value=np.asarray(av_close, dtype=np.float64),
        prior_count=np.asarray(count_open, dtype=np.float64),
        prior_account_value=av_open)
    return mp, state


def test_settlement_pattern_basis_is_accepted_and_block_reconciles():
    basis = _basis(settlement_pattern=PATTERN)
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=6)
    for nm in ("lic_opening", "claims_incurred", "claims_paid", "lic_closing"):
        assert hasattr(mv, nm)
    np.testing.assert_allclose(
        mv.lic_opening + mv.claims_incurred - mv.claims_paid,
        mv.lic_closing, rtol=1e-10)


def test_settlement_pattern_leaves_an_outstanding_lic():
    basis = _basis(settlement_pattern=PATTERN)
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=6)
    assert np.all(mv.lic_opening > 0.0)
    assert np.all(mv.lic_closing > 0.0)
    assert np.all(mv.claims_incurred > 0.0)


def test_no_pattern_has_zero_lic_and_claims_paid_equals_incurred():
    basis = _basis(settlement_pattern=None)
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=6)
    np.testing.assert_array_equal(mv.lic_opening, 0.0)
    np.testing.assert_array_equal(mv.lic_closing, 0.0)
    np.testing.assert_allclose(mv.claims_paid, mv.claims_incurred, rtol=1e-12)


def test_reconciliation_and_aggregate_carry_the_lic_lines():
    basis = _basis(settlement_pattern=PATTERN)
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=6)
    recon = fcf.reconcile([mv])[0]
    np.testing.assert_allclose(
        recon.claims_paid, -float(mv.claims_paid.sum()), rtol=1e-10)
    np.testing.assert_allclose(
        recon.lic_closing, float(mv.lic_closing.sum()), rtol=1e-10)
    agg = fcf.vfa.settle_aggregate(mp, state, basis, period_months=6)
    np.testing.assert_allclose(
        agg.lic_closing, float(mv.lic_closing.sum()), rtol=1e-9)
    np.testing.assert_allclose(
        agg.claims_incurred, float(mv.claims_incurred.sum()), rtol=1e-9)
