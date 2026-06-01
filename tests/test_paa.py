"""PAA validation -- the Premium Allocation Approach measurement.

The PAA measures the Liability for Remaining Coverage as an unearned
premium: premiums build it up, insurance revenue (allocated by coverage
units) releases it. Total revenue equals total premium, so the service
result is just premiums less claims and expenses -- the underwriting profit.
"""
import fastcashflow as fcf
import numpy as np
import pytest

from fastcashflow import ExpenseItem, ModelPoints
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_assumptions


Q = 0.002          # flat monthly mortality
LAPSE = 0.005      # flat monthly lapse


def _assumptions(**overrides):
    kw = dict(
        mortality_q     = Q,
        lapse_q         = LAPSE,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = 0.10,
    )
    kw.update(overrides)
    return make_death_assumptions(**kw)


def test_paa_revenue_equals_total_premium():
    """Insurance revenue recognised over the contract equals total premium."""
    res = fcf.paa.measure(ModelPoints.single(40, 50_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS), _assumptions())
    assert np.isclose(res.revenue.sum(), res.cashflows.premium_cf.sum())


def test_paa_lrc_hand_calc():
    """Single-premium contract -- the LRC is the textbook pro-rata UPR."""
    basis = _assumptions()
    single, term = 1_000_000.0, 12
    res = fcf.paa.measure(
        ModelPoints.single(40, 0.0, term, benefits={0: 1e8}, single_premium=single, calculation_methods=PATTERNS), basis
    )

    # straight-line earning: the premium spread evenly over the coverage period
    assert np.allclose(res.revenue[0], single / term)
    # LRC = premium * remaining coverage / total coverage (unearned premium)
    lrc = np.empty(term + 1)
    lrc[0] = 0.0
    lrc[1:] = single * (term - np.arange(1, term + 1)) / term
    assert np.allclose(res.lrc_path[0], lrc)
    assert np.isclose(res.lrc_path[0, -1], 0.0)     # fully earned by the term end


def test_paa_lrc_builds_and_releases():
    """The LRC builds from zero and releases back to zero over the term."""
    res = fcf.paa.measure(ModelPoints.single(35, 40_000.0, 24, benefits={0: 5e7}, calculation_methods=PATTERNS), _assumptions())
    assert np.isclose(res.lrc_path[0, 0], 0.0)        # builds from zero
    assert np.isclose(res.lrc_path[0, -1], 0.0)       # releases back to zero
    assert np.all(res.lrc_path[0] >= -1e-6)           # a liability, never negative
    assert res.lrc_path[0].max() > 0.0                # genuinely non-trivial between


def test_paa_service_result_is_the_underwriting_profit():
    """Total service result = premiums - claims - expenses."""
    basis = _assumptions(expense_items=(
        ExpenseItem("acquisition",  "alpha_fixed",    100_000.0),
        ExpenseItem("maintenance",  "gamma_fixed",  12_000.0),
    ))
    res = fcf.paa.measure(ModelPoints.single(45, 60_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS), basis)
    cf = res.cashflows
    profit = (cf.premium_cf.sum() - cf.claim_cf.sum()
              - cf.morbidity_cf.sum() - cf.expense_cf.sum())
    assert np.isclose(res.service_result.sum(), profit)


def test_paa_onerous_contract_carries_a_loss():
    """A contract whose claims exceed its premiums is flagged onerous."""
    profitable = fcf.paa.measure(
        ModelPoints.single(40, 500_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS), _assumptions()
    )
    onerous = fcf.paa.measure(
        ModelPoints.single(40, 1_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS), _assumptions()
    )
    assert np.allclose(profitable.loss_component, 0.0)
    assert onerous.loss_component[0] > 0.0


def test_paa_revenue_basis_claims():
    """B126(b): revenue allocated by the expected timing of incurred claims."""
    basis = _assumptions(expense_items=(
        ExpenseItem("acquisition", "alpha_fixed", 500_000.0),
    ))
    mps = ModelPoints.single(40, 50_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS)
    by_time = fcf.paa.measure(mps, basis, revenue_basis="time")
    by_claims = fcf.paa.measure(mps, basis, revenue_basis="claims")

    total_premium = by_claims.cashflows.premium_cf.sum()
    assert np.isclose(by_claims.revenue.sum(), total_premium)   # still totals premium

    se = by_claims.service_expense[0]
    assert np.allclose(by_claims.revenue[0], total_premium * se / se.sum())
    # the t=0 acquisition spike makes the claims basis differ from passage of time
    assert not np.allclose(by_time.revenue[0], by_claims.revenue[0])


def test_paa_rejects_unknown_revenue_basis():
    """An unrecognised revenue basis is an error."""
    with pytest.raises(ValueError, match="revenue_basis"):
        fcf.paa.measure(ModelPoints.single(40, 50_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS),
                    _assumptions(), revenue_basis="weekly")
