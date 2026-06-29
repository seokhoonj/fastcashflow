"""Reporting layer: report.by_period -- period bucketing.

by_period generalises annual() to arbitrary reporting-period lengths and to a
calendar basis (each cohort shifted by its inception month). The reports are
built directly with chosen (n_mp, n_time) arrays so the bucketing is hand-checked
in isolation from the measurement pipeline.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.reporting.report import Report


def _report(revenue, **over):
    """A Report whose lines are all `revenue` unless overridden -- enough to
    exercise the bucketing, which treats every flow line the same way."""
    revenue = np.asarray(revenue, dtype=float)
    n_mp = revenue.shape[0]
    zeros2d = np.zeros_like(revenue)
    fields = dict(
        insurance_revenue=revenue,
        insurance_service_expense=zeros2d, insurance_service_result=zeros2d,
        insurance_finance_expense=zeros2d, bel_finance_expense=zeros2d,
        ra_finance_expense=zeros2d, csm_finance_expense=zeros2d,
        loss_component=np.zeros(n_mp), csm_opening=zeros2d, csm_accretion=zeros2d,
        csm_release=zeros2d, csm_closing=zeros2d)
    fields.update(over)
    return Report(**fields)


def test_elapsed_buckets_sum_mps_then_chunk_time():
    # 2 MPs, 4 months; period_months=2 -> 2 periods of [m0,m1] and [m2,m3]
    revenue = np.array([[1.0, 2.0, 3.0, 4.0],
                        [10.0, 20.0, 30.0, 40.0]])
    out = _report(revenue).by_period(2)
    # period 0 = (1+2) + (10+20) = 33; period 1 = (3+4) + (30+40) = 77
    np.testing.assert_allclose(out["insurance_revenue"], [33.0, 77.0])


def test_calendar_basis_shifts_each_cohort_by_inception_month():
    revenue = np.array([[1.0, 2.0, 3.0, 4.0],
                        [10.0, 20.0, 30.0, 40.0]])
    # MP0 inception at calendar 0 -> cal months 0,1,2,3
    # MP1 inception at calendar 1 -> cal months 1,2,3,4
    # period_months=2 -> periods {0,1},{2,3},{4}
    out = _report(revenue).by_period(2, basis="calendar",
                                     inception_month=[0, 1])
    # p0: MP0 cal0(1) cal1(2) + MP1 cal1(10)           = 13
    # p1: MP0 cal2(3) cal3(4) + MP1 cal2(20) cal3(30)  = 57
    # p2: MP1 cal4(40)                                  = 40
    np.testing.assert_allclose(out["insurance_revenue"], [13.0, 57.0, 40.0])


def test_loss_component_lands_in_the_inception_period():
    revenue = np.zeros((2, 6))
    lc = np.array([100.0, 7.0])
    # calendar: MP0 inception cal 0 -> period 0; MP1 inception cal 3 -> period 1.
    # n_periods follows the 6-month flow (offset 3 + 5 = cal 8 -> 3 periods).
    out = _report(revenue, loss_component=lc).by_period(
        3, basis="calendar", inception_month=[0, 3])
    np.testing.assert_allclose(out["loss_component"], [100.0, 7.0, 0.0])
    # elapsed: both inceptions at policy-month 0 -> period 0
    out_e = _report(revenue, loss_component=lc).by_period(3)
    np.testing.assert_allclose(out_e["loss_component"], [107.0, 0.0])


def test_ragged_last_period_is_padded():
    # 5 months, period_months=2 -> 3 periods, last holds a single month
    revenue = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    out = _report(revenue).by_period(2)
    np.testing.assert_allclose(out["insurance_revenue"], [2.0, 2.0, 1.0])


def test_by_period_12_matches_annual_for_shared_lines():
    """by_period(12, elapsed) reproduces annual() on the lines they share --
    bit for bit: the elapsed basis sums across model points before chunking, the
    same order annual() uses, so floating-point cancellation cannot diverge."""
    rng = np.random.default_rng(0)
    revenue = rng.normal(size=(4, 30))
    rep = _report(revenue, insurance_service_result=revenue * 0.5)
    ann = rep.annual()
    per = rep.by_period(12)
    for line in ("insurance_revenue", "insurance_service_result"):
        np.testing.assert_array_equal(per[line], ann[line])


def test_elapsed_rejects_inception_month():
    with pytest.raises(ValueError, match="elapsed"):
        _report(np.zeros((1, 4))).by_period(2, inception_month=[0])


def test_calendar_requires_inception_month():
    with pytest.raises(ValueError, match="calendar"):
        _report(np.zeros((1, 4))).by_period(2, basis="calendar")


def test_calendar_rejects_wrong_length_offsets():
    with pytest.raises(ValueError, match="inception_month"):
        _report(np.zeros((2, 4))).by_period(
            2, basis="calendar", inception_month=[0, 1, 2])


def test_rejects_non_positive_period():
    with pytest.raises(ValueError, match="period_months"):
        _report(np.zeros((1, 4))).by_period(0)


def test_unknown_basis_rejected():
    with pytest.raises(ValueError, match="basis"):
        _report(np.zeros((1, 4))).by_period(2, basis="quarterly")


def _reins_report(premium, **over):
    premium = np.asarray(premium, dtype=float)
    zeros2d = np.zeros_like(premium)
    fields = dict(
        reinsurance_premium_allocated=premium, amounts_recovered=zeros2d,
        reinsurance_service_result=zeros2d, ra_release=zeros2d,
        reinsurance_finance_expense=zeros2d, bel_finance_expense=zeros2d,
        ra_finance_expense=zeros2d, csm_finance_expense=zeros2d,
        csm_opening=zeros2d, csm_accretion=zeros2d, csm_release=zeros2d,
        csm_closing=zeros2d)
    fields.update(over)
    return fcf.reinsurance.Report(**fields)


def test_reinsurance_by_period_buckets_and_has_no_loss_component():
    premium = np.array([[1.0, 2.0, 3.0, 4.0]])
    recovered = np.array([[5.0, 5.0, 5.0, 5.0]])
    out = _reins_report(premium, amounts_recovered=recovered).by_period(2)
    np.testing.assert_allclose(out["reinsurance_premium_allocated"], [3.0, 7.0])
    np.testing.assert_allclose(out["amounts_recovered"], [10.0, 10.0])
    assert "loss_component" not in out


def test_by_period_runs_on_a_real_measurement():
    """Smoke: a real GMM report buckets without error and ties to annual()."""
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis()
    rep = fcf.report(fcf.gmm.measure(mp, basis, full=True))
    per = rep.by_period(12)
    ann = rep.annual()
    np.testing.assert_allclose(per["insurance_revenue"], ann["insurance_revenue"])
