"""Smith-Wilson discount-curve construction.

The Smith-Wilson curve fits the observed zero-coupon rates exactly and
extrapolates the long end toward the ultimate forward rate (UFR). The anchors:
the fit is exact at the observed maturities (a property of the construction), the
forward rate converges to the UFR, and a single-instrument case matches a hand
calculation. The same call serves any currency -- only the inputs differ.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow._smith_wilson import (
    smith_wilson, smith_wilson_prices, smith_wilson_alpha)


# Korean government-bond curve (ECOS, observed) -- the won risk-free input.
KR_MAT = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0])
KR_RATE = np.array([0.0310, 0.0355, 0.0368, 0.0390, 0.0408, 0.0410])
KR_UFR, KR_ALPHA = 0.0405, 0.10


def test_exact_fit_reproduces_observed_rates():
    # The curve passes through every observed point exactly (a defining property
    # of Smith-Wilson -- the fit, not an approximation).
    p = smith_wilson_prices(KR_MAT, KR_RATE, ufr=KR_UFR, alpha=KR_ALPHA, target=KR_MAT)
    implied = p ** (-1.0 / KR_MAT) - 1.0
    np.testing.assert_allclose(implied, KR_RATE, atol=1e-12)


def test_public_curve_hits_observed_integer_years():
    # The (years,) spot curve, indexed by year, equals the observed rate at each
    # observed integer maturity.
    curve = smith_wilson(KR_MAT, KR_RATE, ufr=KR_UFR, alpha=KR_ALPHA, years=120)
    assert curve.shape == (120,)
    for u, r in zip(KR_MAT, KR_RATE):
        np.testing.assert_allclose(curve[int(u) - 1], r, atol=1e-12)


def test_forward_rate_converges_to_ufr():
    # Past the last liquid point the one-year forward rate converges to the UFR.
    t = np.array([100.0, 101.0, 119.0, 120.0])
    p = smith_wilson_prices(KR_MAT, KR_RATE, ufr=KR_UFR, alpha=KR_ALPHA, target=t)
    fwd_100 = p[0] / p[1] - 1.0       # 1y forward at 100
    fwd_119 = p[2] / p[3] - 1.0       # 1y forward at 119
    np.testing.assert_allclose([fwd_100, fwd_119], KR_UFR, atol=1e-4)


def test_single_instrument_hand_calc():
    # N = 1: solve zeta by hand and reprice. P(u) must equal the observed price,
    # and the implementation must reproduce the hand value.
    u, r, ufr, alpha = 10.0, 0.04, 0.0405, 0.10
    w = np.log1p(ufr)
    m = (1.0 + r) ** (-u)
    mu = np.exp(-w * u)
    w11 = np.exp(-w * 2 * u) * (
        alpha * u - 0.5 * np.exp(-alpha * u) * (np.exp(alpha * u) - np.exp(-alpha * u)))
    zeta = (m - mu) / w11
    p_hand = np.exp(-w * u) + w11 * zeta
    p_impl = smith_wilson_prices(
        np.array([u]), np.array([r]), ufr=ufr, alpha=alpha, target=np.array([u]))[0]
    assert np.isclose(p_hand, m)            # reprices the instrument
    assert np.isclose(p_impl, p_hand)       # implementation == hand


def test_curve_feeds_basis_discount_annual():
    # The constructed curve is a drop-in Basis.discount_annual (a per-year curve).
    curve = smith_wilson(KR_MAT, KR_RATE, ufr=KR_UFR, alpha=KR_ALPHA, years=100)
    basis = fcf.Basis(
        mortality_annual=0.004, lapse_annual=0.03, discount_annual=curve,
        ra_confidence=0.75, mortality_cv=0.1,
        coverages=(fcf.CoverageRate("DEATH", 0.004),))
    mp = fcf.ModelPoints.single(
        issue_age=40, premium=1_000.0, term_months=120,
        benefits={"DEATH": 100_000.0},
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
    m = fcf.gmm.measure(mp, basis, full=False)
    assert np.isfinite(np.atleast_1d(m.bel)).all()


def test_currency_agnostic_same_call():
    # The same function builds a foreign-currency curve -- only the inputs change.
    usd_mat = np.array([1.0, 5.0, 10.0, 30.0])
    usd_rate = np.array([0.0362, 0.0418, 0.0447, 0.0497])
    curve = smith_wilson(usd_mat, usd_rate, ufr=0.036, alpha=0.12, years=120)
    assert curve.shape == (120,)
    # exact fit holds for this currency too
    for u, r in zip(usd_mat, usd_rate):
        np.testing.assert_allclose(curve[int(u) - 1], r, atol=1e-12)


def _fwd_at(t, *, alpha, h=1e-3):
    # Continuous forward at maturity t from a central difference of the fitted
    # log-prices: f(t) = -d ln P / dt. A black-box check of the solver's target.
    p = smith_wilson_prices(KR_MAT, KR_RATE, ufr=KR_UFR, alpha=alpha,
                            target=np.array([t - h, t + h]))
    return (np.log(p[0]) - np.log(p[1])) / (2.0 * h)


def test_alpha_flat_at_ufr_returns_floor():
    # When the observed rates already sit at the UFR the fit is flat (zeta = 0) and
    # the forward equals the UFR at any alpha, so the smallest qualifying alpha is
    # the floor -- an analytic anchor independent of the matrix solve.
    mat = np.array([1.0, 5.0, 10.0, 20.0])
    rate = np.full(mat.shape, KR_UFR)
    a = smith_wilson_alpha(mat, rate, ufr=KR_UFR, convergence_point=60.0, alpha_min=0.05)
    assert a == 0.05


def test_alpha_makes_forward_reach_ufr_at_cp():
    # The defining property: at the solved alpha the fitted forward at the
    # convergence point equals the UFR to within the tolerance.
    cp = 60.0
    a = smith_wilson_alpha(KR_MAT, KR_RATE, ufr=KR_UFR, convergence_point=cp,
                           tolerance=1e-4)
    assert 0.05 <= a <= 1.0
    fwd_annual = np.exp(_fwd_at(cp, alpha=a)) - 1.0
    assert np.isclose(fwd_annual, KR_UFR, atol=2e-4)


def test_alpha_is_the_smallest_qualifying_value():
    # Halving the solved alpha moves the forward at the convergence point farther
    # from the UFR -- confirming the solver found the boundary, not an interior point.
    cp = 60.0
    a = smith_wilson_alpha(KR_MAT, KR_RATE, ufr=KR_UFR, convergence_point=cp,
                           tolerance=1e-4)
    ew = np.exp(np.log1p(KR_UFR))
    gap_solved = abs(np.exp(_fwd_at(cp, alpha=a)) - ew)
    gap_half = abs(np.exp(_fwd_at(cp, alpha=a * 0.5)) - ew)
    assert gap_half > gap_solved


def test_alpha_bisects_when_floor_insufficient():
    # A long end well below a high UFR cannot converge at the floor, so the solver
    # bisects to a strictly higher alpha that sits on the tolerance boundary (the
    # forward at the convergence point is within -- but no tighter than -- the tol).
    cp, ufr, tol = 40.0, 0.05, 1e-4
    a = smith_wilson_alpha(KR_MAT, KR_RATE, ufr=ufr, convergence_point=cp, tolerance=tol)
    assert a > 0.05                                        # floor did not suffice
    p = smith_wilson_prices(KR_MAT, KR_RATE, ufr=ufr, alpha=a,
                            target=np.array([cp - 1e-3, cp + 1e-3]))
    fwd_annual = np.exp((np.log(p[0]) - np.log(p[1])) / 2e-3) - 1.0
    assert abs(fwd_annual - ufr) <= 2 * tol               # within tolerance at the boundary


def test_alpha_rejects_bad_input():
    with pytest.raises(ValueError):                       # cp not past last maturity
        smith_wilson_alpha(KR_MAT, KR_RATE, ufr=KR_UFR, convergence_point=20.0)
    with pytest.raises(ValueError):                       # non-positive cp
        smith_wilson_alpha(KR_MAT, KR_RATE, ufr=KR_UFR, convergence_point=0.0)
    with pytest.raises(ValueError):                       # bad alpha bracket
        smith_wilson_alpha(KR_MAT, KR_RATE, ufr=KR_UFR, convergence_point=60.0,
                           alpha_min=0.5, alpha_max=0.5)


def test_rejects_bad_input():
    with pytest.raises(ValueError):                       # mismatched lengths
        smith_wilson(np.array([1.0, 2.0]), np.array([0.03]), ufr=0.04, alpha=0.1)
    with pytest.raises(ValueError):                       # non-positive maturity
        smith_wilson(np.array([0.0]), np.array([0.03]), ufr=0.04, alpha=0.1)
    with pytest.raises(ValueError):                       # non-positive alpha
        smith_wilson(KR_MAT, KR_RATE, ufr=0.04, alpha=0.0)
    with pytest.raises(ValueError):                       # empty
        smith_wilson(np.array([]), np.array([]), ufr=0.04, alpha=0.1)
