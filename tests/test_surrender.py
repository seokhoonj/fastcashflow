"""Surrender value (해약환급금) cash flow tests.

The engine post-computes surrender_cf as
``lapse_flow * cum_premium * surrender_value_curve[t]``. These tests
verify the formula end-to-end against hand calculation.
"""
import numpy as np

from fastcashflow import Basis, ModelPoints, CoverageRate
from fastcashflow.gmm import measure
from conftest import PATTERNS

def _flat_rate(value):
    """3-arg rate callable that returns a constant for any (sex, age, dur)."""
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis(lapse_rate, surrender_curve):
    return Basis(
        mortality_annual=_flat_rate(0.0),
        lapse_annual=_flat_rate(lapse_rate),
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.0,
        surrender_value_curve=surrender_curve,
        coverages=(CoverageRate("DEATH", _flat_rate(0.0)),),
    )


def test_surrender_cf_zero_when_curve_is_none():
    """``surrender_value_curve=None`` reproduces the historical behaviour:
    lapse silently removes the contract, no surrender cash flow."""
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        level_premium=50_000.0, term_months=12,
        calculation_methods=PATTERNS,
    )
    asmp = _basis(lapse_rate=0.1, surrender_curve=None)
    m = measure(mp, asmp)
    assert np.all(m.cashflows.surrender_cf == 0.0)


def test_surrender_cf_hand_calc_single_period():
    """A single MP, deterministic lapse, constant factor -- check that
    surrender_cf at month 0 equals
    ``lapse_flow * cum_premium * factor[0]``."""
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        level_premium=10_000.0, term_months=12,
        calculation_methods=PATTERNS,
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
    proportionally. surrender_cf[t] = lapse_rate * cum_premium[t] * factor,
    so with flat lapse and flat factor the ratio is just the cum_premium
    ratio (the inforce-decay is already absorbed into cum_premium, which
    aggregates inforce * premium each month)."""
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        level_premium=10_000.0, term_months=24,
        calculation_methods=PATTERNS,
    )
    asmp = _basis(lapse_rate=0.05, surrender_curve=np.full(25, 0.8))
    m = measure(mp, asmp)
    cum_prem = np.cumsum(m.cashflows.premium_cf[0])
    expected_ratio = cum_prem[11] / cum_prem[0]
    actual_ratio = m.cashflows.surrender_cf[0, 11] / m.cashflows.surrender_cf[0, 0]
    assert np.isclose(actual_ratio, expected_ratio)


def test_surrender_cf_widens_bel():
    """Enabling surrender increases BEL (more outflow on lapse)."""
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        level_premium=10_000.0, term_months=24,
        calculation_methods=PATTERNS,
    )
    asmp_no_surr = _basis(lapse_rate=0.05, surrender_curve=None)
    asmp_with_surr = _basis(
        lapse_rate=0.05, surrender_curve=np.full(25, 1.0))
    bel_no = measure(mp, asmp_no_surr).bel_path[0, 0]
    bel_with = measure(mp, asmp_with_surr).bel_path[0, 0]
    # BEL is the present value of future outflows minus premiums.
    # Adding a positive surrender outflow strictly increases BEL.
    assert bel_with > bel_no
