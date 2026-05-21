"""Disclosure validation -- the IFRS 17 report assembled from a GMM measurement.

The report turns a measurement into the insurance service result, its
build-up (revenue and service expense) and the CSM analysis of change. The
checks here are identities -- the CSM waterfall reconciles, and the service
result equals revenue less expense -- plus that the whole CSM releases.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPointSet, measure, report


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.001),
        lapse_monthly=lambda duration: np.full(duration.shape, 0.01),
        discount_annual=0.03,
        expense_acquisition=200_000.0,
        expense_maintenance_annual=60_000.0,
        expense_inflation=0.02,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )


def _portfolio(n: int = 300) -> ModelPointSet:
    rng = np.random.default_rng(4)
    return ModelPointSet(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(20, 90, n) * 1_000_000,
        monthly_premium=rng.integers(5, 18, n) * 10_000,
        term_months=rng.integers(60, 180, n),
    )


def test_report_csm_analysis_of_change_reconciles():
    """The CSM waterfall: opening + accretion - release = closing."""
    res = report(measure(_portfolio(), _assumptions()))
    assert np.allclose(
        res.csm_opening + res.csm_accretion - res.csm_release, res.csm_closing
    )


def test_report_service_result_is_revenue_less_expense():
    """Service result = revenue - service expense, and revenue grosses it up."""
    res = report(measure(_portfolio(), _assumptions()))
    assert np.allclose(
        res.insurance_service_result,
        res.insurance_revenue - res.insurance_service_expense,
    )
    assert np.allclose(
        res.insurance_revenue,
        res.insurance_service_expense + res.insurance_service_result,
    )


def test_report_csm_fully_releases_with_non_negative_profit():
    """A profitable contract releases its whole CSM, earning profit each month."""
    res = report(measure(ModelPointSet.single(40, 1e8, 150_000.0, 120), _assumptions()))
    assert res.csm_opening[0, 0] > 0.0                  # there is a CSM
    assert np.isclose(res.csm_closing[0, -1], 0.0)      # all released by term end
    assert np.all(res.insurance_service_result[0] >= -1e-6)   # profit emerges >= 0


def test_report_annual_totals_match_the_monthly_sum():
    """annual() portfolio-and-year totals reconcile with the monthly figures."""
    res = report(measure(_portfolio(), _assumptions()))
    ann = res.annual()
    assert np.isclose(
        ann["insurance_service_result"].sum(), res.insurance_service_result.sum()
    )
    assert np.isclose(ann["csm_release"].sum(), res.csm_release.sum())
    assert np.isclose(ann["insurance_revenue"].sum(), res.insurance_revenue.sum())
