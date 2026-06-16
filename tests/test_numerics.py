"""Unit tests for ``numerics`` primitives -- the shared pure-array helpers.

These cover edge cases of the helpers themselves, independent of the
engine / PAA / VFA orchestration that calls them.
"""
import math

import numpy as np
import pytest

from fastcashflow.numerics import (
    _carry_lic_residual, _norm_ppf, _settlement_factor, _settlement_lic)


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


def test_settlement_lic_holds_tail_beyond_horizon():
    """A settlement tail past the horizon parks its residual at the terminal LIC.

    Hand calc: incurred = [10000, 9900, 9801] (a 3-month horizon), pattern
    [0.2]*5 (5 lags, so every claim's settlement runs past month 2). The
    within-horizon paid each month is 0.2 * (running incurred): col0 = 2000,
    col1 = 0.2*19900 = 3980, col2 = 0.2*29701 = 5940.2. The LIC is the
    undiscounted cumsum of (incurred - paid):
        lic_path[:, 1:] = cumsum([8000, 5920, 3860.8]) = [8000, 13920, 17780.8].
    The terminal 17780.8 == total incurred (29701) - paid-within (11920.2):
    the unpaid tail is HELD, not dropped. lic_path[:, 0] is 0 (no double-count with
    the BEL, which values the full claim via _settlement_factor = 1 at r = 0).
    """
    incurred = np.array([[10000.0, 9900.0, 9801.0]])
    pattern = np.full(5, 0.2)
    lic_path = _settlement_lic(incurred, pattern)
    np.testing.assert_allclose(lic_path, [[0.0, 8000.0, 13920.0, 17780.8]])
    assert lic_path[0, -1] == pytest.approx(29701.0 - 11920.2)
    assert lic_path[0, 0] == 0.0
    # The BEL values the full incurred claim separately -- factor 1 at r = 0.
    assert _settlement_factor(pattern, 0.0) == pytest.approx(1.0)


def test_settlement_lic_pattern_longer_than_book():
    """A pattern with more lags than the horizon must not crash or go negative.

    incurred = [10000, 9900, 9801] (n_time = 3), pattern [0.1]*10 (10 lags).
    Only k = 0, 1, 2 clear the ``k < n_time`` guard, so paid = [1000, 1990,
    2970.1]; the terminal LIC = 29701 - 5960.1 = 23740.9. Without the guard the
    stop index incurred[:, :n_time - k] would go negative and broadcast-error.
    """
    incurred = np.array([[10000.0, 9900.0, 9801.0]])
    lic_path = _settlement_lic(incurred, np.full(10, 0.1))
    assert lic_path.shape == (1, 4)
    assert lic_path[0, -1] == pytest.approx(23740.9)
    assert np.all(lic_path >= 0.0)


def test_settlement_lic_runs_off_to_zero_within_horizon():
    """A tail that fits inside the horizon settles to a zero terminal LIC.

    The same [0.2]*5 run-off, but on an 8-month horizon that fully absorbs it:
    every incurred claim is paid before the horizon ends, so nothing is parked.
    """
    incurred = np.zeros((1, 8))
    incurred[0, :3] = [10000.0, 9900.0, 9801.0]
    lic_path = _settlement_lic(incurred, np.full(5, 0.2))
    assert lic_path[0, -1] == 0.0                  # analytic terminal -> exact zero, no dust


def test_settlement_lic_tiny_tail_weight_preserved_exactly():
    """A small settlement weight leaves a small but EXACT terminal residual.

    The terminal is the analytic tail, not a thresholded heuristic, so a
    0.0001 weight settling past the horizon is preserved -- it is not mistaken
    for float dust and dropped. A claim of 1e6 at the last month (horizon 3)
    with pattern [0.9999, 0.0001] leaves 1e6 * 0.0001 = 100 outstanding.
    """
    incurred = np.array([[0.0, 0.0, 1_000_000.0]])
    lic_path = _settlement_lic(incurred, np.array([0.9999, 0.0001]))
    assert lic_path[0, -1] == pytest.approx(100.0, rel=1e-12)
    assert lic_path[0, -1] > 0.0


def test_carry_lic_residual_holds_residual_flat():
    """The stitch helper carries each segment's exact terminal residual flat.

    Two segment rows on a local horizon t = 2, scattered into a global horizon
    5. Row 0 has a genuine outstanding residual (17780.8); row 1 fully settled
    (exact zero from _settlement_lic). Each is carried flat across the tail.
    """
    seg = np.array([[0.0, 8000.0, 17780.8],   # genuine outstanding residual
                    [0.0, 5000.0, 0.0]])       # fully settled -> exact zero
    lic_path = np.zeros((2, 6))
    lic_path[[0, 1], :3] = seg
    _carry_lic_residual(lic_path, np.array([0, 1]), 2, 5, seg)
    np.testing.assert_allclose(lic_path[0, 3:], 17780.8)      # genuine residual held flat
    assert np.all(lic_path[1, 3:] == 0.0)                     # fully-settled pads exact zero


def test_carry_lic_residual_noop_at_horizon_segment():
    """The horizon-defining segment (t == n_time) is laid in directly -- no tail."""
    seg = np.array([[0.0, 1.0, 2.0, 3.0]])
    lic_path = np.zeros((1, 4))
    lic_path[[0], :4] = seg
    before = lic_path.copy()
    _carry_lic_residual(lic_path, np.array([0]), 3, 3, seg)   # t == n_time
    np.testing.assert_array_equal(lic_path, before)


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
