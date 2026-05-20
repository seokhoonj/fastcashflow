"""Phase 1 validation -- Risk Adjustment and expense cash flows."""
import numpy as np

from fastcashflow import Assumptions, ModelPointSet, measure
from fastcashflow.gmm import _norm_ppf


def _assumptions(**overrides) -> Assumptions:
    """Build an Assumptions with simple defaults, overridable per test."""
    base = dict(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.01),
        lapse_monthly=lambda duration: np.full(duration.shape, 0.02),
        discount_annual=0.0,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        claims_cv=0.0,
    )
    base.update(overrides)
    return Assumptions(**base)


def test_norm_ppf_known_quantiles():
    """The inverse normal CDF against known standard-normal quantiles."""
    assert np.isclose(_norm_ppf(0.50), 0.0, atol=1e-9)
    assert np.isclose(_norm_ppf(0.75), 0.6744897501960817, atol=1e-6)
    assert np.isclose(_norm_ppf(0.90), 1.2815515655446004, atol=1e-6)
    assert np.isclose(_norm_ppf(0.95), 1.6448536269514722, atol=1e-6)
    assert np.isclose(_norm_ppf(0.99), 2.3263478740408408, atol=1e-6)


def test_risk_adjustment():
    """RA = z(confidence) * claims_cv * PV(claims), hand-checked."""
    res = measure(
        ModelPointSet.single(
            issue_age=40, sum_assured=1_000_000.0,
            monthly_premium=12_000.0, term_months=2,
        ),
        _assumptions(ra_confidence=0.75, claims_cv=0.20),
    )
    # zero discount; PV(claims) = 10000 + 9702 = 19702 (see test_phase0)
    pv_claims = 19702.0
    z_75 = 0.6744897501960817
    assert np.isclose(res.ra[0], z_75 * 0.20 * pv_claims)


def test_expenses():
    """Acquisition (t=0) and maintenance expense, hand-checked."""
    res = measure(
        ModelPointSet.single(
            issue_age=40, sum_assured=1_000_000.0,
            monthly_premium=12_000.0, term_months=2,
        ),
        _assumptions(
            expense_acquisition=500.0,
            expense_maintenance_annual=120.0,   # 10 per month
            expense_inflation=0.0,
        ),
    )
    inforce = [1.0, 0.99 * 0.98]
    # expense_cf[0] = acquisition + maintenance = 1*500 + 1*(120/12) = 510
    # expense_cf[1] = maintenance only         = 0.9702*(120/12)     = 9.702
    assert np.isclose(res.cashflows.expense_cf[0, 0], 510.0)
    assert np.isclose(res.cashflows.expense_cf[0, 1], 9.702)

    # BEL = PV(claims) + PV(expenses) - PV(premiums)
    pv_claims = 19702.0
    pv_expenses = 510.0 + 9.702
    pv_premiums = 12_000.0 + inforce[1] * 12_000.0
    assert np.isclose(res.bel[0], pv_claims + pv_expenses - pv_premiums)


def test_expense_inflation():
    """Maintenance expense grows with inflation; acquisition does not recur."""
    res = measure(
        ModelPointSet.single(
            issue_age=40, sum_assured=1_000_000.0,
            monthly_premium=12_000.0, term_months=13,
        ),
        _assumptions(
            mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.0),
            lapse_monthly=lambda duration: np.full(duration.shape, 0.0),
            expense_maintenance_annual=120.0,   # 10 per month
            expense_inflation=0.06,
        ),
    )
    # no mortality/lapse -> in force stays 1.0
    # maintenance[t] = 1.0 * 10 * (1.06)^(t/12)
    assert np.isclose(res.cashflows.expense_cf[0, 0], 10.0)
    assert np.isclose(res.cashflows.expense_cf[0, 12], 10.0 * 1.06)
