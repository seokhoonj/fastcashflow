"""vfa.settle -- the paragraph-50(a)-52 loss-component allocation and the
B96(c) investment-component experience (the VFA mirrors of the gmm.settle
features).

For VFA the investment component IS the account value, so the two features are
unified: the paragraph-50(a) pool is the guarantee excess + expenses (the
claims+expenses pool, inherently excluding the account-value investment
component), and the B96(c) experience is on the account value returned on exits
(benefit_cf - guarantee_excess_cf).
"""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CoverageRate, ExpenseItem, InforceState, ModelPoints
from fastcashflow.movement import VFASettlementMovement

settle = getattr(fcf.vfa, "settle", None)
_HAS = (settle is not None
        and "csm_investment_experience"
        in VFASettlementMovement.__dataclass_fields__
        and "loss_component_amortised"
        in VFASettlementMovement.__dataclass_fields__)
pytestmark = pytest.mark.skipif(
    not _HAS, reason="vfa.settle 50(a)-52 / B96(c) not implemented yet")


def _basis():
    death_fn = lambda s, ia, d: np.full(s.shape, 0.012)
    return Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=0.05, ra_confidence=0.75, mortality_cv=0.0,
        expense_cv=0.10, investment_return=0.05, fund_fee=0.015,
        expense_items=(ExpenseItem("maintenance", "gamma_fixed", 1_000.0),),
        coverages=(CoverageRate("DEATH", death_fn),))


def _growth(basis):
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0
    return (1.0 + r_m) * (1.0 - f_m)


def _book(basis, *, em_open=6, period=6, term=24, lc_open=0.0, prior_csm=None,
          actual_ic=None, final=False):
    # a GMAB crediting guarantee ABOVE the fund return gives a non-zero
    # guarantee excess -> a real claims+expenses pool for the 50(a) allocation.
    mp0 = ModelPoints.single(40, 100.0, term, account_value=1e6,
                             minimum_crediting_rate=0.08)
    em_close = term if final else em_open + period
    m0 = fcf.vfa.measure(mp0, basis)
    inforce = m0.cashflows.inforce
    rows = np.arange(1)
    growth = _growth(basis)
    av_open = np.asarray(mp0.account_value, dtype=np.float64) * growth ** em_open
    av_close = np.zeros(1) if final else av_open * growth ** period
    pad = np.concatenate([inforce, np.zeros((1, 1))], axis=1)
    count_open = inforce[rows, em_open]
    boundary = np.asarray(mp0.contract_boundary_months)
    count_close = (np.zeros(1) if final
                   else pad[rows, np.minimum(em_close, boundary)])
    ids = np.array(["P0"])
    mp = replace(mp0, mp_id=ids,
                 elapsed_months=np.full(1, em_close, dtype=np.int64),
                 count=np.asarray(count_close, dtype=np.float64))
    pc = (m0.csm_path[rows, em_open] if prior_csm is None
          else np.array([prior_csm]))
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(1, em_close, dtype=np.int64),
        count=np.asarray(count_close, dtype=np.float64),
        prior_csm=pc, lock_in_rate=0.0,
        account_value=np.asarray(av_close, dtype=np.float64),
        prior_count=np.asarray(count_open, dtype=np.float64),
        prior_account_value=av_open,
        prior_loss_component=(np.array([lc_open]) if lc_open else None),
        actual_investment_component=(None if actual_ic is None
                                     else np.array([actual_ic])))
    return mp, state


# ---------------------------------------------------------------------------
# paragraph 50(a)-52 loss-component allocation
# ---------------------------------------------------------------------------

def test_lc_reconciliation_identity_with_the_two_channels():
    basis = _basis()
    mp, state = _book(basis, lc_open=20_000.0, prior_csm=0.0)
    mv = settle(mp, state, basis, period_months=6)
    np.testing.assert_allclose(
        mv.loss_component_opening + mv.loss_component_finance
        - mv.loss_component_amortised - mv.loss_component_reversed
        + mv.loss_component_recognised, mv.loss_component_closing, rtol=1e-10)
    assert np.all(mv.loss_component_amortised > 0.0)
    assert np.all(mv.loss_component_finance > 0.0)


def test_paragraph52_runs_to_zero_at_final_settlement():
    basis = _basis()
    mp, state = _book(basis, em_open=18, period=6, term=24, lc_open=20_000.0,
                      prior_csm=0.0, final=True)
    mv = settle(mp, state, basis, period_months=6)
    np.testing.assert_allclose(mv.loss_component_closing, 0.0, atol=1e-6)


def test_profitable_book_has_zero_lc_channel_lines():
    basis = _basis()
    mp, state = _book(basis, lc_open=0.0)   # prior_csm from the measure (> 0)
    mv = settle(mp, state, basis, period_months=6)
    np.testing.assert_array_equal(mv.loss_component_finance, 0.0)
    np.testing.assert_array_equal(mv.loss_component_amortised, 0.0)


# ---------------------------------------------------------------------------
# B96(c) investment-component experience
# ---------------------------------------------------------------------------

def test_absent_actual_ic_zero_and_csm_recursion_holds():
    basis = _basis()
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=6)
    np.testing.assert_array_equal(mv.csm_investment_experience, 0.0)
    np.testing.assert_allclose(
        mv.csm_opening + mv.csm_accretion + mv.csm_fv_share
        + mv.csm_future_service + mv.csm_premium_experience
        + mv.csm_investment_experience - mv.loss_component_reversed
        + mv.loss_component_recognised - mv.csm_release, mv.csm_closing,
        rtol=1e-9, atol=1e-9)


def test_investment_experience_sign_and_cross_tie():
    basis = _basis()
    # probe the expected IC (actual = 0 => csm_investment_experience == expected)
    mp0, st0 = _book(basis, actual_ic=0.0)
    expected_ic = float(settle(mp0, st0, basis,
                               period_months=6).csm_investment_experience[0])
    assert expected_ic > 0.0
    mp_hi, st_hi = _book(basis, actual_ic=expected_ic + 10_000.0)
    mv_hi = settle(mp_hi, st_hi, basis, period_months=6)
    np.testing.assert_allclose(mv_hi.csm_investment_experience, -10_000.0,
                               rtol=1e-6)
    # the new leg is outside the VFA cross-tie
    np.testing.assert_allclose(
        mv_hi.csm_fv_share + mv_hi.csm_future_service,
        -(mv_hi.bel_experience + mv_hi.ra_experience), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# reconciliation + aggregate carry the new lines
# ---------------------------------------------------------------------------

def test_reconciliation_and_aggregate_carry_the_new_lines():
    basis = _basis()
    mp, state = _book(basis, lc_open=20_000.0, prior_csm=0.0,
                      actual_ic=200_000.0)
    mv = settle(mp, state, basis, period_months=6)
    recon = fcf.reconcile([mv])[0]
    for nm in ("csm_investment_experience", "loss_component_finance",
               "loss_component_amortised"):
        assert hasattr(recon, nm)
    np.testing.assert_allclose(
        recon.loss_component_amortised,
        -float(mv.loss_component_amortised.sum()), rtol=1e-10)
    agg = fcf.vfa.settle_aggregate(mp, state, basis, period_months=6)
    np.testing.assert_allclose(
        agg.csm_investment_experience,
        float(mv.csm_investment_experience.sum()), rtol=1e-9)
    np.testing.assert_allclose(
        agg.loss_component_amortised,
        float(mv.loss_component_amortised.sum()), rtol=1e-9)
