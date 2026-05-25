"""Curve-shaped discount / inflation / maintenance assumptions.

``Assumptions.discount_annual`` / ``expense_inflation`` /
``gamma_flat`` accept either a flat scalar (historical
behaviour) or a per-year 1-D array (`(n_years,)`) for a time-varying basis
locked in at initial recognition. The engine expands either form to a
per-month curve via :mod:`fastcashflow.curves`. The flat scalar must
reproduce the previous result exactly; the curve form is validated by hand
against a small two-year contract.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPoints, measure, value
from fastcashflow.curves import (
    discount_monthly_curve,
    inflation_index,
    gamma_monthly_curve,
)


def _flat_asmp(**overrides) -> Assumptions:
    """Minimal flat-basis assumptions; everything else off."""
    base = dict(
        mortality_annual=lambda s, ia, d: np.zeros_like(s, dtype=np.float64),
        lapse_annual=lambda s, ia, d: np.zeros_like(s, dtype=np.float64),
        discount_annual=0.05,
        alpha_flat=0.0,
        gamma_flat=12_000.0,
        expense_inflation=0.02,
        ra_confidence=0.75,
        mortality_cv=0.0,
    )
    base.update(overrides)
    return Assumptions(**base)


# ---------------------------------------------------------------------------
# curves.py helpers -- direct unit checks
# ---------------------------------------------------------------------------

def test_scalar_discount_reproduces_flat_curve():
    """``discount_annual = 0.05`` gives a flat per-month curve of length n_time."""
    asmp = _flat_asmp(discount_annual=0.05)
    curve = discount_monthly_curve(asmp, 24)
    expected = (1.05) ** (1.0 / 12.0) - 1.0
    assert curve.shape == (24,)
    assert np.allclose(curve, expected)


def test_per_year_discount_curve_steps_at_year_boundary():
    """A 2-year ``[0.03, 0.05]`` curve gives the year-0 monthly rate for months
    0..11 and the year-1 rate for months 12..23."""
    asmp = _flat_asmp(discount_annual=np.array([0.03, 0.05]))
    curve = discount_monthly_curve(asmp, 24)
    m0 = (1.03) ** (1.0 / 12.0) - 1.0
    m1 = (1.05) ** (1.0 / 12.0) - 1.0
    assert np.allclose(curve[:12], m0)
    assert np.allclose(curve[12:], m1)


def test_per_year_discount_holds_flat_past_curve_end():
    """A short curve is held flat at its last value past the end."""
    asmp = _flat_asmp(discount_annual=np.array([0.03, 0.05]))
    curve = discount_monthly_curve(asmp, 36)            # 3 years, 2-year curve
    m1 = (1.05) ** (1.0 / 12.0) - 1.0
    assert np.allclose(curve[24:], m1)                   # year 2 -> held at year-1 value


def test_scalar_inflation_index_matches_continuous_compound():
    """Scalar ``expense_inflation`` -- ``(1+i)^(t/12)`` -- bit-equivalent path."""
    asmp = _flat_asmp(expense_inflation=0.02)
    idx = inflation_index(asmp, 24)
    expected = (1.02) ** (np.arange(24) / 12.0)
    assert np.allclose(idx, expected)


def test_per_year_inflation_compounds_across_years():
    """A 2-year ``[0.02, 0.05]`` inflation curve compounds:

    * month 0: 1.0
    * month 6: ``(1.02)^(6/12)``
    * month 12: ``(1.02)^1`` (full year-0 factor)
    * month 18: ``1.02 * (1.05)^(6/12)``
    * month 24: ``1.02 * 1.05`` (had the array extended)
    """
    asmp = _flat_asmp(expense_inflation=np.array([0.02, 0.05]))
    idx = inflation_index(asmp, 24)
    assert idx[0] == 1.0
    assert np.isclose(idx[6], (1.02) ** (6 / 12))
    assert np.isclose(idx[12], 1.02)
    assert np.isclose(idx[18], 1.02 * (1.05) ** (6 / 12))


def test_per_year_maintenance_steps_at_year_boundary():
    """A 2-year ``[12000, 18000]`` maintenance curve gives monthly 1000 / 1500."""
    asmp = _flat_asmp(gamma_flat=np.array([12_000.0, 18_000.0]))
    monthly = gamma_monthly_curve(asmp, 24)
    assert np.allclose(monthly[:12], 1_000.0)
    assert np.allclose(monthly[12:], 1_500.0)


# ---------------------------------------------------------------------------
# End-to-end -- BEL with a non-flat discount curve, hand-validated
# ---------------------------------------------------------------------------

def test_bel_with_curve_discount_matches_hand_calc():
    """A 2-year single-payment contract with all decrements zero, a flat
    expense, and a stepped discount curve. BEL = PV(expense over 24
    months). Hand calc: 1,000 per month, discounted at the curve's stepped
    monthly rates.
    """
    asmp = _flat_asmp(
        # zero everything except maintenance + discount
        expense_inflation=0.0,
        gamma_flat=12_000.0,       # 1,000 / month, flat
        discount_annual=np.array([0.03, 0.05]),    # 3% year 0, 5% year 1
    )
    mp = ModelPoints.single(issue_age=40, death_benefit=0.0,
                            level_premium=0.0, term_months=24, count=1)
    m = measure(mp, asmp)

    # Hand calc: expense 1,000 per month, discount mid-month at the
    # per-month rate (constant-force conversion of the per-year rate).
    m0 = (1.03) ** (1.0 / 12.0) - 1.0
    m1 = (1.05) ** (1.0 / 12.0) - 1.0
    months = np.arange(24)
    rates = np.where(months < 12, m0, m1)
    discount_start = np.concatenate(([1.0], np.cumprod(1.0 / (1.0 + rates))))
    discount_mid = discount_start[:-1] / np.sqrt(1.0 + rates)
    expected_bel = (1_000.0 * discount_mid).sum()

    assert np.isclose(m.bel[0, 0], expected_bel)


def test_bel_value_matches_measure_with_curve_discount():
    """`value()` and `measure()` agree on BEL for a non-flat discount curve too."""
    asmp = _flat_asmp(
        expense_inflation=0.0,
        gamma_flat=12_000.0,
        discount_annual=np.array([0.03, 0.05, 0.06]),
    )
    mp = ModelPoints.single(issue_age=40, death_benefit=100_000.0,
                            level_premium=1_000.0, term_months=36, count=1)
    m = measure(mp, asmp).bel[0, 0]
    v = value(mp, asmp).bel[0]
    assert np.isclose(m, v)


def test_csm_accretes_at_curve_rate():
    """A 2-year curve discount accretes the CSM at the per-month curve rate,
    not at a single scalar. Two segments give different accretion factors."""
    # Profitable contract: premium covers expenses with margin -> positive CSM
    asmp = _flat_asmp(
        expense_inflation=0.0,
        gamma_flat=12_000.0,
        discount_annual=np.array([0.03, 0.10]),   # step UP at year 1
    )
    mp = ModelPoints.single(issue_age=40, death_benefit=0.0,
                            level_premium=5_000.0, term_months=24, count=1)
    m = measure(mp, asmp)
    csm_open = m.csm[0, 0]
    csm_close_year0 = m.csm[0, 12]
    csm_close_year1 = m.csm[0, 24]

    # The year-1 monthly rate is higher than year-0; so the accretion in
    # year 1 should outpace what a flat 3% would give.
    assert csm_open > 0          # profitable
    assert csm_close_year0 > 0
    assert csm_close_year1 >= 0  # may be ~0 after full release
    # Year 1 accretion sum should be > year 0 accretion sum (higher rate,
    # similar opening balance order of magnitude).
    acc_year0 = m.csm_accretion[0, :12].sum()
    acc_year1 = m.csm_accretion[0, 12:24].sum()
    assert acc_year1 > acc_year0
