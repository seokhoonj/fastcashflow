"""vfa.settle -- the B96(a)/B97(c) premium experience split (the VFA mirror of
the gmm.settle premium experience).

Authoritative skeleton. ``state.actual_premium`` less the expected premium is
split by ``premium_experience_future_fraction`` between future service (the CSM,
``csm_premium_experience``) and current/past service (a P&L memo,
``premium_experience_revenue``). The future leg is a NEW future-service change
with no BEL/RA counterpart, so it enters the paragraph-45 algebra ON TOP of the
``x = -(bel_experience + ra_experience)`` change but stays OUTSIDE the VFA
cross-tie ``csm_fv_share + csm_future_service == x``. Both lines are zero unless
actual_premium is given (byte-identical to the pre-feature settle).
"""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CoverageRate, ExpenseItem, InforceState, ModelPoints
from fastcashflow.movement import VFASettlementMovement

settle = getattr(fcf.vfa, "settle", None)
_HAS_PE = (settle is not None
           and "csm_premium_experience" in VFASettlementMovement.__dataclass_fields__)
pytestmark = pytest.mark.skipif(
    not _HAS_PE, reason="vfa.settle premium experience not implemented yet")


def _basis():
    death_fn = lambda s, ia, d: np.full(s.shape, 0.012)
    return Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=0.05, ra_confidence=0.75, mortality_cv=0.0,
        expense_cv=0.10, investment_return=0.05, fund_fee=0.015,
        expense_items=(ExpenseItem("maintenance", "gamma_fixed", 1_000.0),),
        coverages=(CoverageRate("DEATH", death_fn),),
    )


def _growth(basis):
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0
    return (1.0 + r_m) * (1.0 - f_m)


def _book(basis, *, em_open=6, period=6, actual_premium=None):
    mp0 = ModelPoints.single(40, 100.0, 24, account_value=1e6)
    em_close = em_open + period
    m0 = fcf.vfa.measure(mp0, basis)
    inforce = m0.cashflows.inforce
    rows = np.arange(1)
    growth = _growth(basis)
    av_open = np.asarray(mp0.account_value, dtype=np.float64) * growth ** em_open
    av_close = av_open * growth ** period
    pad = np.concatenate([inforce, np.zeros((1, 1))], axis=1)
    count_open = inforce[rows, em_open]
    boundary = np.asarray(mp0.contract_boundary_months)
    count_close = pad[rows, np.minimum(em_close, boundary)]
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
        prior_account_value=av_open,
        actual_premium=(None if actual_premium is None
                        else np.asarray([actual_premium], dtype=np.float64)))
    return mp, state


def test_absent_actual_premium_zeroes_both_lines():
    basis = _basis()
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=6)
    np.testing.assert_array_equal(mv.csm_premium_experience, 0.0)
    np.testing.assert_array_equal(mv.premium_experience_revenue, 0.0)


def test_favourable_premium_all_to_csm_when_fraction_one():
    basis = _basis()
    # a strongly favourable premium experience (more received than expected)
    mp, state = _book(basis, actual_premium=5_000.0)
    mv = settle(mp, state, basis, period_months=6,
                premium_experience_future_fraction=1.0)
    assert np.all(mv.csm_premium_experience > 0.0)
    np.testing.assert_array_equal(mv.premium_experience_revenue, 0.0)


def test_fraction_zero_routes_all_to_revenue():
    basis = _basis()
    mp, state = _book(basis, actual_premium=5_000.0)
    mv = settle(mp, state, basis, period_months=6,
                premium_experience_future_fraction=0.0)
    np.testing.assert_array_equal(mv.csm_premium_experience, 0.0)
    assert np.all(mv.premium_experience_revenue > 0.0)


def test_premium_experience_is_outside_the_vfa_cross_tie():
    """The cross-tie csm_fv_share + csm_future_service == -(bel_exp + ra_exp)
    holds WITHOUT the premium-experience leg (it is a new future-service change
    with no BEL/RA counterpart)."""
    basis = _basis()
    mp, state = _book(basis, actual_premium=5_000.0)
    mv = settle(mp, state, basis, period_months=6,
                premium_experience_future_fraction=0.6)
    np.testing.assert_allclose(
        mv.csm_fv_share + mv.csm_future_service,
        -(mv.bel_experience + mv.ra_experience), rtol=1e-9, atol=1e-9)


def test_csm_closing_recursion_includes_premium_experience():
    basis = _basis()
    mp, state = _book(basis, actual_premium=5_000.0)
    mv = settle(mp, state, basis, period_months=6,
                premium_experience_future_fraction=0.6)
    np.testing.assert_allclose(
        mv.csm_closing,
        mv.csm_opening + mv.csm_accretion + mv.csm_fv_share
        + mv.csm_future_service + mv.csm_premium_experience
        - mv.loss_component_reversed + mv.loss_component_recognised
        - mv.csm_release, rtol=1e-9, atol=1e-9)


def test_reconciliation_carries_premium_experience():
    basis = _basis()
    mp, state = _book(basis, actual_premium=5_000.0)
    mv = settle(mp, state, basis, period_months=6,
                premium_experience_future_fraction=0.6)
    recon = fcf.reconcile([mv])[0]
    assert hasattr(recon, "csm_premium_experience")
    np.testing.assert_allclose(
        recon.csm_premium_experience, float(mv.csm_premium_experience.sum()),
        rtol=1e-10)
