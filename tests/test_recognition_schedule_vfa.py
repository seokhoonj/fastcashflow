"""vfa.recognition_schedule -- the paragraph-109 maturity-band API for VFA.

The VFA counterpart of gmm.recognition_schedule: allocates the vfa.settle
closing CSM to maturity bands by each contract's forward coverage-unit (in
force) fraction, so the bands sum to the closing CSM. Shares the allocation
helper with the GMM schedule (only the source settle differs).
"""
import numpy as np
import pytest
from dataclasses import replace

import fastcashflow as fcf
from fastcashflow import (
    Basis, CoverageRate, ExpenseItem, InforceState, ModelPoints)

recognition_schedule = getattr(fcf.vfa, "recognition_schedule", None)
pytestmark = pytest.mark.skipif(
    recognition_schedule is None,
    reason="vfa.recognition_schedule not implemented yet")


def _basis(*, investment_return=0.05, fund_fee=0.015):
    death_fn = lambda s, ia, d: np.full(s.shape, 0.012)
    return Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=investment_return, ra_confidence=0.75,
        mortality_cv=0.0, expense_cv=0.10,
        investment_return=investment_return, fund_fee=fund_fee,
        expense_items=(ExpenseItem("maintenance", "gamma_fixed", 1_000.0),),
        coverages=(CoverageRate("DEATH", death_fn),))


def _growth(b, mp):
    r_m = (1.0 + b.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + b.fund_fee) ** (1.0 / 12.0) - 1.0
    return (1.0 + r_m) * (1.0 - f_m)


def _book(*, n=3, em_open=24, period=12):
    """A profitable VFA book (csm_closing > 0): no guarantee, on-track."""
    basis = _basis()
    em_close = em_open + period
    mp0 = ModelPoints(
        issue_age=np.full(n, 40), premium=np.zeros(n),
        term_months=np.full(n, 120), account_value=np.full(n, 1.0e6),
        product=np.full(n, "VA"), benefits={0: np.zeros(n)}, count=np.ones(n))
    m0 = fcf.vfa.measure(mp0, basis)
    inforce = m0.cashflows.inforce
    rows = np.arange(n)
    g = _growth(basis, mp0)
    av_open = mp0.account_value * g ** em_open
    av_close = av_open * g ** period
    prior_csm = m0.csm_path[rows, em_open]
    scale = np.array([1000.0, 500.0, 2000.0])[:n]
    ids = np.array([f"V{i}" for i in range(n)])
    count_open = inforce[rows, em_open] * scale
    count_close = inforce[rows, em_close] * scale
    mp = replace(mp0, mp_id=ids,
                 elapsed_months=np.full(n, em_close, dtype=np.int64),
                 count=count_close)
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(n, em_close, dtype=np.int64),
        count=count_close, prior_csm=prior_csm * scale, lock_in_rate=0.0,
        account_value=av_close, prior_count=count_open,
        prior_account_value=av_open)
    return mp, state, basis


def test_bands_sum_to_closing_csm():
    mp, state, basis = _book()
    sched = recognition_schedule(mp, state, basis, period_months=12)
    np.testing.assert_allclose(float(sched.csm.sum()), sched.closing_csm,
                               rtol=1e-10)
    assert sched.closing_csm > 0.0
    assert len(sched.csm) == len(sched.labels)


def test_configurable_edges():
    mp, state, basis = _book()
    sched = recognition_schedule(mp, state, basis, period_months=12,
                                 band_edges_months=(24, 60))
    assert sched.band_edges_months == (24, 60)
    assert len(sched.csm) == 3
    np.testing.assert_allclose(float(sched.csm.sum()), sched.closing_csm,
                               rtol=1e-10)


def test_matches_settle_closing_csm():
    mp, state, basis = _book()
    sched = recognition_schedule(mp, state, basis, period_months=12)
    mv = fcf.vfa.settle(mp, state, basis, period_months=12)
    pos = float(np.asarray(mv.csm_closing)[np.asarray(mv.csm_closing) > 0].sum())
    np.testing.assert_allclose(sched.closing_csm, pos, rtol=1e-10)


def test_bad_edges_rejected():
    mp, state, basis = _book()
    with pytest.raises(ValueError, match="ascending|positive"):
        recognition_schedule(mp, state, basis, band_edges_months=(36, 12))
