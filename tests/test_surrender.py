"""Surrender value (해약환급금) cash flow tests.

The engine post-computes surrender_cf as
``lapse_flow * cum_premium * surrender_value_curve[t]``. These tests
verify the formula end-to-end against hand calculation.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPoints, measure


def _flat_rate(value):
    """3-arg rate callable that returns a constant for any (sex, age, dur)."""
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis(lapse_rate, surrender_curve):
    return Assumptions(
        mortality_annual=_flat_rate(0.0),
        lapse_annual=_flat_rate(lapse_rate),
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.0,
        surrender_value_curve=surrender_curve,
    )


def test_surrender_cf_zero_when_curve_is_none():
    """``surrender_value_curve=None`` reproduces the historical behaviour:
    lapse silently removes the contract, no surrender cash flow."""
    mp = ModelPoints.single(
        issue_age=40, death_benefit=100_000_000.0,
        level_premium=50_000.0, term_months=12,
    )
    asmp = _basis(lapse_rate=0.1, surrender_curve=None)
    m = measure(mp, asmp)
    assert np.all(m.cashflows.surrender_cf == 0.0)


def test_surrender_cf_hand_calc_single_period():
    """A single MP, deterministic lapse, constant factor -- check that
    surrender_cf at month 0 equals
    ``lapse_flow * cum_premium * factor[0]``."""
    mp = ModelPoints.single(
        issue_age=40, death_benefit=100_000_000.0,
        level_premium=10_000.0, term_months=12,
    )
    # 12% annual lapse -> ~1.06% monthly under constant-force conversion
    # (engine handles annual_to_monthly conversion internally).
    lapse_annual = 0.12
    # Flat 50% factor at every month
    curve = np.full(13, 0.5)
    asmp = _basis(lapse_rate=lapse_annual, surrender_curve=curve)

    m = measure(mp, asmp)
    # At t=0, premium_cf = 10000, cum_premium[0] = 10000
    # inforce[0] = 1.0, lapse_monthly = 1 - (1 - 0.12)**(1/12)
    lapse_monthly = 1.0 - (1.0 - lapse_annual) ** (1.0 / 12.0)
    lapse_flow_0 = 1.0 * lapse_monthly
    cum_prem_0 = 10_000.0
    expected_surrender_0 = lapse_flow_0 * cum_prem_0 * 0.5

    assert np.isclose(m.cashflows.surrender_cf[0, 0], expected_surrender_0)


def test_surrender_cf_accumulates_with_cum_premium():
    """As cumulative premium grows, surrender_cf at the same factor grows
    proportionally."""
    mp = ModelPoints.single(
        issue_age=40, death_benefit=100_000_000.0,
        level_premium=10_000.0, term_months=24,
    )
    asmp = _basis(lapse_rate=0.05, surrender_curve=np.full(25, 0.8))
    m = measure(mp, asmp)
    # cum_premium grows linearly with months paid. The ratio of surrender_cf
    # at month 11 vs month 0 should equal the ratio of cum_premium x inforce
    # (factor is flat so it cancels).
    cum_prem = np.cumsum(m.cashflows.premium_cf[0])
    expected_ratio = (m.cashflows.inforce[0, 11] * cum_prem[11]) / (
        m.cashflows.inforce[0, 0] * cum_prem[0]
    )
    actual_ratio = m.cashflows.surrender_cf[0, 11] / m.cashflows.surrender_cf[0, 0]
    assert np.isclose(actual_ratio, expected_ratio)


def test_surrender_cf_widens_bel():
    """Enabling surrender increases BEL (more outflow on lapse)."""
    mp = ModelPoints.single(
        issue_age=40, death_benefit=100_000_000.0,
        level_premium=10_000.0, term_months=24,
    )
    asmp_no_surr = _basis(lapse_rate=0.05, surrender_curve=None)
    asmp_with_surr = _basis(
        lapse_rate=0.05, surrender_curve=np.full(25, 1.0))
    bel_no = measure(mp, asmp_no_surr).bel[0, 0]
    bel_with = measure(mp, asmp_with_surr).bel[0, 0]
    # BEL is the present value of future outflows minus premiums.
    # Adding a positive surrender outflow strictly increases BEL.
    assert bel_with > bel_no
