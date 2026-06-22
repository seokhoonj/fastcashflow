"""Nelson-Siegel-Svensson curve -- evaluate (hand-calc) and least-squares fit."""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.nelson_siegel import _ns_loadings


_PARAMS = dict(beta0=0.04, beta1=-0.02, beta2=0.01, beta3=0.005,
               lambda1=2.0, lambda2=5.0)


def test_loadings_limit_at_zero():
    """f1(0) = 1, f2(0) = 0 (the 0/0 limit is taken explicitly)."""
    f1, f2 = _ns_loadings(np.array([0.0, 1.0]), lam=2.0)
    assert np.isclose(f1[0], 1.0)
    assert np.isclose(f2[0], 0.0)


def test_evaluate_hand_calc():
    """y(tau=2) computed by hand from the loadings.

    tau=lambda1=2 -> x=1: f1=(1-e^-1)=0.632121, f2=f1-e^-1=0.264241.
    lambda2 term tau=2,lam=5 -> x=0.4: f1=(1-e^-0.4)/0.4=0.824200, f2=0.153890.
    y = 0.04 - 0.02*0.632121 + 0.01*0.264241 + 0.005*0.153890 = 0.0307695.
    """
    y = fcf.nelson_siegel_svensson(2.0, **_PARAMS)
    assert np.isclose(float(y), 0.0307695, atol=1e-6)


def test_short_and_long_limits():
    """y(0) = beta0 + beta1 (short rate); y(inf) -> beta0 (long rate)."""
    y0 = float(fcf.nelson_siegel_svensson(1e-9, **_PARAMS))
    ylong = float(fcf.nelson_siegel_svensson(1e6, **_PARAMS))
    assert np.isclose(y0, _PARAMS["beta0"] + _PARAMS["beta1"], atol=1e-6)
    assert np.isclose(ylong, _PARAMS["beta0"], atol=1e-6)


def test_scalar_in_scalar_out_and_vector():
    assert np.ndim(fcf.nelson_siegel_svensson(5.0, **_PARAMS)) == 0
    v = fcf.nelson_siegel_svensson(np.array([1.0, 5.0, 10.0]), **_PARAMS)
    assert v.shape == (3,)


def test_nelson_siegel_is_svensson_with_zero_second_hump():
    taus = np.array([0.5, 2.0, 10.0, 30.0])
    ns = fcf.nelson_siegel(taus, beta0=0.04, beta1=-0.02, beta2=0.01, lambda1=2.0)
    nss = fcf.nelson_siegel_svensson(taus, beta0=0.04, beta1=-0.02, beta2=0.01,
                                     beta3=0.0, lambda1=2.0, lambda2=1.0)
    assert np.allclose(ns, nss)


def test_negative_lambda_rejected():
    with pytest.raises(ValueError, match="positive"):
        fcf.nelson_siegel_svensson(5.0, beta0=0.04, beta1=0.0, beta2=0.0,
                                   beta3=0.0, lambda1=-1.0, lambda2=5.0)


def test_fit_recovers_a_synthetic_curve():
    """Yields generated from a known NSS curve are reproduced by the fit to a tiny
    error (exact-NSS data -> the separable LS + grid-zoom nails the shape)."""
    taus = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    y = fcf.nelson_siegel_svensson(taus, **_PARAMS)
    ns = fcf.fit_nelson_siegel_svensson(taus, y)
    fitted = ns(taus)
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    assert rmse < 1e-5
    # the level/short anchors land on the generating curve too
    assert np.isclose(ns.long_rate, _PARAMS["beta0"], atol=1e-3)


def test_fit_smooths_through_noise():
    """A noisy humped curve is fitted with a small (non-zero) residual; the fit
    smooths rather than interpolates."""
    rng = np.random.default_rng(0)
    taus = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    clean = fcf.nelson_siegel_svensson(taus, **_PARAMS)
    noisy = clean + rng.normal(0.0, 2e-4, taus.shape)
    ns = fcf.fit_nelson_siegel_svensson(taus, noisy)
    resid = ns(taus) - noisy
    assert float(np.sqrt(np.mean(resid ** 2))) < 1e-3       # close, but not exact
    assert float(np.max(np.abs(ns(taus) - clean))) < 1e-3   # recovers the signal


def test_evaluate_rejects_negative_maturity():
    with pytest.raises(ValueError, match="non-negative"):
        fcf.nelson_siegel_svensson(np.array([1.0, -2.0]), **_PARAMS)


def test_fit_rejects_degenerate_grid_or_zoom():
    taus = np.array([0.5, 1, 2, 5, 10], dtype=float)
    y = fcf.nelson_siegel_svensson(taus, **_PARAMS)
    with pytest.raises(ValueError, match="grid"):
        fcf.fit_nelson_siegel_svensson(taus, y, grid=1)
    with pytest.raises(ValueError, match="zoom_rounds"):
        fcf.fit_nelson_siegel_svensson(taus, y, zoom_rounds=0)
    with pytest.raises(ValueError, match="non-negative"):
        fcf.fit_nelson_siegel_svensson(np.array([-1.0, 1, 2, 5]), y[:4])


def test_fit_requires_four_points_and_matching_shapes():
    with pytest.raises(ValueError, match="at least 4"):
        fcf.fit_nelson_siegel_svensson(np.array([1.0, 5.0, 10.0]),
                                       np.array([0.03, 0.035, 0.04]))
    with pytest.raises(ValueError, match="same length"):
        fcf.fit_nelson_siegel_svensson(np.array([1.0, 2, 3, 4]),
                                       np.array([0.03, 0.035, 0.04]))
