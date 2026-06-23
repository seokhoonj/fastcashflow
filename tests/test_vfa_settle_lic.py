"""vfa.settle -- the liability-for-incurred-claims (LIC) block under a
settlement_pattern basis (paragraphs 40(b) / 42(c) / 103(b)).

The VFA mirror of test_gmm_settle_lic.py: the LIC is built from the VFA
benefit_cf incurred stream and measured at fulfilment cash flows -- the
discounted PV of the unpaid run-off (42(c)). It carries NO risk adjustment (the
VFA RA prices expense risk only, the benefit risk sitting in the variable fee).
claims_incurred / claims_paid stay nominal; lic_finance is the reconciling
residual::

    lic_closing == lic_opening + claims_incurred + lic_finance - claims_paid
"""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CoverageRate, ExpenseItem, InforceState, ModelPoints
from fastcashflow.vfa import SettlementMovement

settle = getattr(fcf.vfa, "settle", None)
_HAS_LIC = (settle is not None
            and "lic_closing" in SettlementMovement.__dataclass_fields__)
pytestmark = pytest.mark.skipif(
    not _HAS_LIC,
    reason="vfa.settle LIC block not implemented yet")

PATTERN = np.array([0.6, 0.3, 0.1])


def _basis(*, settlement_pattern=None, discount_annual=0.05):
    death_fn = lambda s, ia, d: np.full(s.shape, 0.012)
    return Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=discount_annual, ra_confidence=0.75, mortality_cv=0.0,
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
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6,
                             calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
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
    for nm in ("lic_opening", "claims_incurred", "lic_finance",
               "claims_paid", "lic_closing"):
        assert hasattr(mv, nm)
    np.testing.assert_allclose(
        mv.lic_opening + mv.claims_incurred + mv.lic_finance - mv.claims_paid,
        mv.lic_closing, rtol=1e-10)


def test_settlement_pattern_leaves_an_outstanding_lic():
    basis = _basis(settlement_pattern=PATTERN)
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=6)
    assert np.all(mv.lic_opening > 0.0)
    assert np.all(mv.lic_closing > 0.0)
    assert np.all(mv.claims_incurred > 0.0)


def test_lic_is_discounted_below_the_nominal_balance():
    """The VFA LIC is the discounted PV of the benefit run-off (no RA), so on a
    positive discount it sits strictly below the undiscounted nominal balance."""
    nodisc = _basis(settlement_pattern=PATTERN, discount_annual=0.0)
    disc = _basis(settlement_pattern=PATTERN, discount_annual=0.05)
    mp_n, st_n = _book(nodisc)
    mp_d, st_d = _book(disc)
    mv_n = settle(mp_n, st_n, nodisc, period_months=6)
    mv_d = settle(mp_d, st_d, disc, period_months=6)
    assert np.all(mv_d.lic_opening < mv_n.lic_opening)
    np.testing.assert_array_equal(mv_n.lic_finance, 0.0)   # r=0 => no unwind


def test_no_pattern_has_zero_lic_and_claims_paid_equals_incurred():
    basis = _basis(settlement_pattern=None)
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=6)
    np.testing.assert_array_equal(mv.lic_opening, 0.0)
    np.testing.assert_array_equal(mv.lic_closing, 0.0)
    np.testing.assert_array_equal(mv.lic_finance, 0.0)
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


def test_lic_discounts_at_the_ifrs_rate_not_the_fund_return():
    """The incurred-claims LIC discounts at the IFRS discount curve
    (discount_monthly from discount_annual), NOT the underlying-items return r_m
    the rest of the VFA path uses. An incurred claim is a fixed, determined
    amount that no longer varies with the fund (B74). Off-diagonal: vary
    discount_annual with investment_return held fixed (0.05); the LIC opening
    must move, and a higher rate must lower it."""
    pat = np.array([0.25, 0.25, 0.25, 0.25])     # a longer tail makes it visible
    lo = _basis(settlement_pattern=pat, discount_annual=0.02)
    hi = _basis(settlement_pattern=pat, discount_annual=0.10)
    mv_lo = settle(*_book(lo), lo, period_months=6)
    mv_hi = settle(*_book(hi), hi, period_months=6)
    # depends on discount_annual -> discounts at the IFRS rate, not r_m (fixed)
    assert not np.isclose(mv_lo.lic_opening[0], mv_hi.lic_opening[0])
    assert mv_hi.lic_opening[0] < mv_lo.lic_opening[0]   # higher rate, lower LIC
