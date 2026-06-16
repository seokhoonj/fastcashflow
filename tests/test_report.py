"""Report validation -- the IFRS 17 report assembled from a GMM measurement.

The report turns a measurement into the insurance service result, its
build-up (revenue and service expense) and the CSM analysis of change. The
checks here are identities -- the CSM waterfall reconciles, and the service
result equals revenue less expense -- plus that the whole CSM releases.
"""
import fastcashflow as fcf
import numpy as np
import pytest

from conftest import annual_from_monthly as _annual, PATTERNS
from fastcashflow import Basis, ExpenseItem, ModelPoints, report, CoverageRate
from fastcashflow.gmm import measure


def _basis() -> Basis:
    return Basis(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.001)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.01)),
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    200_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  60_000.0),
        ),
        ra_confidence=0.75,
        mortality_cv=0.10,
        investment_return=0.06,
        fund_fee=0.015,
        coverages=(CoverageRate("DEATH", lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.001))),),
    )


def _portfolio(n: int = 300) -> ModelPoints:
    rng = np.random.default_rng(4)
    return ModelPoints(
        issue_age=rng.integers(30, 55, n),
        benefits={"DEATH": rng.integers(20, 90, n) * 1_000_000},
        premium=rng.integers(5, 18, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        calculation_methods=PATTERNS,
    )


def test_report_csm_analysis_of_change_reconciles():
    """The CSM waterfall: opening + accretion - release = closing."""
    res = report(measure(_portfolio(), _basis()))
    assert np.allclose(
        res.csm_opening + res.csm_accretion - res.csm_release, res.csm_closing
    )


def test_report_service_result_is_revenue_less_expense():
    """Service result = revenue - service expense, and revenue grosses it up."""
    res = report(measure(_portfolio(), _basis()))
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
    res = report(measure(ModelPoints.single(40, 150_000.0, 120, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS), _basis()))
    assert res.csm_opening[0, 0] > 0.0                  # there is a CSM
    assert np.isclose(res.csm_closing[0, -1], 0.0)      # all released by term end
    assert np.all(res.insurance_service_result[0] >= -1e-6)   # profit emerges >= 0


def test_report_annual_totals_match_the_monthly_sum():
    """annual() portfolio-and-year totals reconcile with the monthly figures."""
    res = report(measure(_portfolio(), _basis()))
    ann = res.annual()
    assert np.isclose(
        ann["insurance_service_result"].sum(), res.insurance_service_result.sum()
    )
    assert np.isclose(ann["csm_release"].sum(), res.csm_release.sum())
    assert np.isclose(ann["insurance_revenue"].sum(), res.insurance_revenue.sum())


def test_report_str_renders_the_annual_table():
    """str(report) shows the annual table -- title, labels, year-1 figure."""
    res = report(measure(_portfolio(), _basis()))
    text = str(res)
    ann = res.annual()
    assert "IFRS 17 report" in text
    assert "Insurance revenue" in text
    assert "Year 1" in text
    assert f"{ann['insurance_revenue'][0]:,.0f}" in text


def test_report_handles_paa():
    """report() accepts a PAA measurement -- which has no CSM."""
    m = fcf.paa.measure(ModelPoints.single(40, 50_000.0, 12, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS), _basis())
    res = report(m)
    assert np.allclose(res.insurance_revenue, m.revenue)
    assert np.allclose(res.insurance_service_result, m.service_result)
    assert np.allclose(res.csm_opening, 0.0)
    assert np.allclose(res.csm_release, 0.0)


def test_report_handles_vfa():
    """report() accepts a VFA measurement -- the result is the CSM release."""
    m = fcf.vfa.measure(
        ModelPoints.single(40, 0.0, 60, account_value=1e8,
                           calculation_methods=PATTERNS), _basis()
    )
    res = report(m)
    assert np.allclose(
        res.csm_opening + res.csm_accretion - res.csm_release, res.csm_closing
    )
    assert np.allclose(res.insurance_service_result, res.csm_release)


def test_report_loss_component():
    """The loss component is zero when profitable, positive when onerous."""
    profitable = report(measure(
        ModelPoints.single(40, 150_000.0, 120, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS), _basis()))
    onerous = report(measure(
        ModelPoints.single(40, 1_000.0, 120, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS), _basis()))
    assert np.allclose(profitable.loss_component, 0.0)
    assert onerous.loss_component[0] > 0.0


def test_report_rejects_unknown_measurement():
    """A non-measurement input is an error."""
    with pytest.raises(TypeError, match="GMM, PAA, VFA or reinsurance"):
        report(object())


def test_report_finance_expense_is_curve_aware():
    """With a per-year discount curve, finance_expense uses the per-month
    forward rate at every month -- not just the first month's rate."""
    a = Basis(
        mortality_annual=lambda sex, ia, dur: np.full(ia.shape, _annual(0.001)),
        lapse_annual=lambda sex, ia, dur: np.full(dur.shape, _annual(0.01)),
        discount_annual=np.array([0.01, 0.05, 0.05, 0.05, 0.05,
                                  0.05, 0.05, 0.05, 0.05, 0.05]),
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    200_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  60_000.0),
        ),
        ra_confidence=0.75,
        mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", lambda sex, ia, dur: np.full(ia.shape, _annual(0.001))),),
    )
    m = measure(ModelPoints.single(40, 50_000.0, 120, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS), a)
    r = report(m)
    ds = m.discount_bom
    rate = ds[:-1] / ds[1:] - 1.0
    expected = rate * (m.bel_path[0, :-1] + m.ra_path[0, :-1]) + m.csm_accretion[0]
    assert np.allclose(r.insurance_finance_expense[0], expected)
    # And the rate genuinely varies across the curve break -- without the
    # fix, finance_expense would be off by the year-0 vs year-1 spread.
    assert not np.isclose(rate[0], rate[12])


# ---------------------------------------------------------------------------
# Finance-expense disaggregation by source (IFRS 17 B130-B136): BEL / RA / CSM
# ---------------------------------------------------------------------------
def test_report_finance_expense_disaggregates_by_source():
    """The single finance line splits into BEL / RA / CSM components, each its
    own formula, and they sum back to the aggregate."""
    a = Basis(
        mortality_annual=lambda sex, ia, dur: np.full(ia.shape, _annual(0.001)),
        lapse_annual=lambda sex, ia, dur: np.full(dur.shape, _annual(0.01)),
        discount_annual=np.array([0.01, 0.05, 0.05, 0.05, 0.05,
                                  0.05, 0.05, 0.05, 0.05, 0.05]),
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH",
                   lambda sex, ia, dur: np.full(ia.shape, _annual(0.001))),))
    m = measure(ModelPoints.single(40, 150_000.0, 120, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS), a)
    r = report(m)
    ds = m.discount_bom
    rate = ds[:-1] / ds[1:] - 1.0
    # each component is exactly its own source's interest unwind
    assert np.allclose(r.bel_finance_expense[0], rate * m.bel_path[0, :-1])
    assert np.allclose(r.ra_finance_expense[0],  rate * m.ra_path[0, :-1])
    assert np.allclose(r.csm_finance_expense[0], m.csm_accretion[0])
    # the CSM component is genuinely non-trivial here (a profitable contract)
    assert np.any(r.csm_finance_expense[0] > 0.0)
    # and they sum back to the aggregate (a rounding step, not byte-equal)
    np.testing.assert_allclose(
        r.bel_finance_expense + r.ra_finance_expense + r.csm_finance_expense,
        r.insurance_finance_expense, atol=1e-9, rtol=0)


def test_report_finance_disaggregation_identity_all_models():
    """The sum identity holds for GMM, PAA and VFA. The VFA leg is the
    load-bearing guard: VFA's finance line is the CSM accretion (not zero), so
    its CSM component must carry the whole line."""
    gmm = report(measure(_portfolio(), _basis()))
    paa = report(fcf.paa.measure(
        ModelPoints.single(40, 50_000.0, 12, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS), _basis()))
    vfa = report(fcf.vfa.measure(
        ModelPoints.single(40, 0.0, 60, account_value=1e8,
                           calculation_methods=PATTERNS), _basis()))
    for r in (gmm, paa, vfa):
        np.testing.assert_allclose(
            r.bel_finance_expense + r.ra_finance_expense + r.csm_finance_expense,
            r.insurance_finance_expense, atol=1e-9, rtol=0)
    # PAA: all three components are zero (LRC held undiscounted)
    assert np.allclose(paa.bel_finance_expense, 0.0)
    assert np.allclose(paa.csm_finance_expense, 0.0)
    # VFA: the finance line sits wholly on the CSM component
    assert np.allclose(vfa.csm_finance_expense, vfa.insurance_finance_expense)
    assert np.allclose(vfa.bel_finance_expense, 0.0)


def test_report_finance_disaggregation_segmented_broadcast():
    """A segmented full measurement has a per-MP (2-D) discount curve; the
    three-way identity holds at every (mp, t) cell."""
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis()
    r = report(fcf.gmm.measure(mp, basis))
    assert r.insurance_finance_expense.ndim == 2
    np.testing.assert_allclose(
        r.bel_finance_expense + r.ra_finance_expense + r.csm_finance_expense,
        r.insurance_finance_expense, atol=1e-9, rtol=0)
