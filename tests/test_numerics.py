"""Unit tests for ``numerics`` primitives -- the shared pure-array helpers.

These cover edge cases of the helpers themselves, independent of the
engine / PAA / VFA orchestration that calls them.
"""
import math

import numpy as np
import pytest

from fastcashflow.numerics import _norm_ppf, _settlement_factor


def test_settlement_factor_scalar_immediate():
    """Immediate settlement gives factor 1 -- no discounting in any month."""
    pattern = np.array([1.0])
    assert _settlement_factor(pattern, 0.005) == pytest.approx(1.0)


def test_settlement_factor_scalar_two_month_lag():
    """Two-month lag: half discounted one month, half two."""
    pattern = np.array([0.5, 0.5])
    r = 0.01
    expected = 0.5 + 0.5 / (1.0 + r)
    assert _settlement_factor(pattern, r) == pytest.approx(expected)


def test_settlement_factor_constant_curve_agrees_with_scalar():
    """A constant rate curve produces the same factor as the scalar form
    in every month."""
    pattern = np.array([0.4, 0.3, 0.2, 0.1])
    r = 0.0075
    curve = np.full(24, r)
    scalar = _settlement_factor(pattern, r)
    vector = _settlement_factor(pattern, curve)
    assert vector.shape == (24,)
    assert np.allclose(vector, scalar)


def test_settlement_factor_curve_varies_by_month():
    """A step-up curve makes the factor smaller at earlier months -- the
    run-off starting there sees more of the high-rate tail."""
    pattern = np.array([0.5, 0.5])
    # month 0 sees rates [0.01, 0.10]; month 5 sees [0.10, 0.10].
    curve = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.10])
    factor = _settlement_factor(pattern, curve)
    # At month 0, second cf discounted by 1.01; at month 5, by 1.10.
    expected_t0 = 0.5 + 0.5 / 1.01
    expected_t5 = 0.5 + 0.5 / 1.10
    assert factor[0] == pytest.approx(expected_t0)
    assert factor[5] == pytest.approx(expected_t5)
    assert factor[0] > factor[5]


def test_settlement_factor_rejects_invalid_curve_shape():
    """A higher-dimensional curve is not supported."""
    with pytest.raises(ValueError, match="scalar or a 1-D curve"):
        _settlement_factor(np.array([1.0]), np.ones((2, 2)))


def test_norm_ppf_extreme_tail_full_precision():
    """One Halley step ties the inverse CDF to machine precision in the
    extreme tail where the rational approximation alone degrades."""
    for p in (1e-8, 1e-10, 1e-12):
        x = _norm_ppf(p)
        # Round-trip: Phi(x) should agree with p to far better than ~1e-9.
        cdf = 0.5 * math.erfc(-x / math.sqrt(2.0))
        assert abs(cdf - p) < 1e-15 + 1e-15 * abs(p)


def test_norm_ppf_typical_quantiles():
    """Standard table values reproduce to 1e-9, the headline accuracy."""
    # Source: Phi^{-1} at common quantiles.
    cases = {
        0.5:    0.0,
        0.75:   0.6744897501960817,
        0.9:    1.2815515655446004,
        0.95:   1.6448536269514722,
        0.975:  1.959963984540054,
        0.99:   2.3263478740408408,
        0.995:  2.5758293035489004,
        0.999:  3.090232306167813,
    }
    for p, want in cases.items():
        assert abs(_norm_ppf(p) - want) < 1e-9
