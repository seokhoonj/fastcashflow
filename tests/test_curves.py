"""Curve-shaped discount basis.

``Basis.discount_annual`` accepts either a flat scalar (historical
behaviour) or a per-year 1-D array (``(n_years,)``) for a time-varying
basis locked in at initial recognition. The engine expands either form
to a per-month curve via :mod:`fastcashflow.curves`. The flat scalar
must reproduce the previous result exactly; the curve form is validated
by hand against a small two-year contract.
"""
import numpy as np

from fastcashflow import Basis, CalculationMethod, ExpenseItem, ModelPoints, CoverageRate
from fastcashflow.gmm import measure
from fastcashflow.curves import discount_monthly_curve


def _flat_basis(**overrides) -> Basis:
    """Minimal flat-basis basis; everything else off."""
    base = dict(
        mortality_annual=lambda s, ia, d: np.zeros_like(s, dtype=np.float64),
        lapse_annual=lambda s, ia, d: np.zeros_like(s, dtype=np.float64),
        discount_annual=0.05,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("maintenance", "gamma_fixed", 12_000.0),
        ),
        ra_confidence=0.75,
        mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", lambda s, ia, d: np.zeros_like(s, dtype=np.float64)),),
    )
    base.update(overrides)
    return Basis(**base)


# ---------------------------------------------------------------------------
# curves.py helpers -- direct unit checks
# ---------------------------------------------------------------------------

def test_scalar_discount_reproduces_flat_curve():
    """``discount_annual = 0.05`` gives a flat per-month curve of length n_time."""
    basis = _flat_basis(discount_annual=0.05)
    curve = discount_monthly_curve(basis, 24)
    expected = (1.05) ** (1.0 / 12.0) - 1.0
    assert curve.shape == (24,)
    assert np.allclose(curve, expected)


def test_per_year_discount_curve_steps_at_year_boundary():
    """A 2-year ``[0.03, 0.05]`` curve gives the year-0 monthly rate for months
    0..11 and the year-1 rate for months 12..23."""
    basis = _flat_basis(discount_annual=np.array([0.03, 0.05]))
    curve = discount_monthly_curve(basis, 24)
    m0 = (1.03) ** (1.0 / 12.0) - 1.0
    m1 = (1.05) ** (1.0 / 12.0) - 1.0
    assert np.allclose(curve[:12], m0)
    assert np.allclose(curve[12:], m1)


def test_per_year_discount_holds_flat_past_curve_end():
    """A short curve is held flat at its last value past the end."""
    basis = _flat_basis(discount_annual=np.array([0.03, 0.05]))
    curve = discount_monthly_curve(basis, 36)            # 3 years, 2-year curve
    m1 = (1.05) ** (1.0 / 12.0) - 1.0
    assert np.allclose(curve[24:], m1)                   # year 2 -> held at year-1 value


# ---------------------------------------------------------------------------
# End-to-end -- BEL with a non-flat discount curve, hand-validated
# ---------------------------------------------------------------------------

def test_bel_with_curve_discount_matches_hand_calc():
    """A 2-year single-payment contract with all decrements zero, a flat
    expense, and a stepped discount curve. BEL = PV(expense over 24
    months). Hand calc: 1,000 per month, discounted at the curve's stepped
    monthly rates.
    """
    basis = _flat_basis(
        # zero everything except maintenance + discount
        expense_inflation=0.0,
        expense_items=(
            ExpenseItem("maintenance", "gamma_fixed", 12_000.0),
        ),
        discount_annual=np.array([0.03, 0.05]),    # 3% year 0, 5% year 1
    )
    mp = ModelPoints.single(issue_age=40, benefits={"DEATH": 0.0},
                            calculation_methods={"DEATH": CalculationMethod.DEATH},
                            premium=0.0, term_months=24, count=1)
    m = measure(mp, basis)

    # Hand calc: expense 1,000 per month, discount mid-month at the
    # per-month rate (constant-force conversion of the per-year rate).
    m0 = (1.03) ** (1.0 / 12.0) - 1.0
    m1 = (1.05) ** (1.0 / 12.0) - 1.0
    months = np.arange(24)
    rates = np.where(months < 12, m0, m1)
    discount_factor_bom = np.concatenate(([1.0], np.cumprod(1.0 / (1.0 + rates))))
    discount_factor_mid = discount_factor_bom[:-1] / np.sqrt(1.0 + rates)
    expected_bel = (1_000.0 * discount_factor_mid).sum()

    assert np.isclose(m.bel_path[0, 0], expected_bel)


def test_bel_value_matches_measure_with_curve_discount():
    """`measure()` and `measure()` agree on BEL for a non-flat discount curve too."""
    basis = _flat_basis(
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("maintenance", "gamma_fixed", 12_000.0),
        ),
        discount_annual=np.array([0.03, 0.05, 0.06]),
    )
    mp = ModelPoints.single(issue_age=40, benefits={"DEATH": 100_000.0},
                            calculation_methods={"DEATH": CalculationMethod.DEATH},
                            premium=1_000.0, term_months=36, count=1)
    m = measure(mp, basis).bel_path[0, 0]
    v = measure(mp, basis, full=False).bel[0]
    assert np.isclose(m, v)


def test_csm_accretes_at_curve_rate():
    """A 2-year curve discount accretes the CSM at the per-month curve rate,
    not at a single scalar. Two segments give different accretion factors."""
    # Profitable contract: premium covers expenses with margin -> positive CSM
    basis = _flat_basis(
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("maintenance", "gamma_fixed", 12_000.0),
        ),
        discount_annual=np.array([0.03, 0.10]),   # step UP at year 1
    )
    mp = ModelPoints.single(issue_age=40, benefits={"DEATH": 0.0},
                            calculation_methods={"DEATH": CalculationMethod.DEATH},
                            premium=5_000.0, term_months=24, count=1)
    m = measure(mp, basis)
    csm_open = m.csm_path[0, 0]
    csm_close_year0 = m.csm_path[0, 12]
    csm_close_year1 = m.csm_path[0, 24]

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
