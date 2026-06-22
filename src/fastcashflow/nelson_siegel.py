"""Nelson-Siegel-Svensson yield-curve model -- a parametric spot-rate curve and
its least-squares calibration to observed yields.

The Svensson (1994) extension of Nelson-Siegel (1987) writes the spot rate at
maturity ``tau`` (years) as four economically-readable factors::

    y(tau) = beta0
           + beta1 * f1(tau, lambda1)
           + beta2 * f2(tau, lambda1)
           + beta3 * f2(tau, lambda2)

    f1(tau, lam) = (1 - exp(-tau/lam)) / (tau/lam)          # slope loading
    f2(tau, lam) = f1(tau, lam) - exp(-tau/lam)             # curvature loading

``beta0`` is the long-term level (``y -> beta0`` as ``tau -> inf``); ``beta0 +
beta1`` is the instantaneous short rate (``tau -> 0``); ``beta2`` / ``beta3`` are
the magnitudes of two humps whose locations are set by the decays ``lambda1`` /
``lambda2``. Dropping the ``beta3`` / ``lambda2`` term recovers plain
Nelson-Siegel (see :func:`nelson_siegel`).

A compact, smooth alternative to Smith-Wilson (:mod:`fastcashflow.smith_wilson`):
where Smith-Wilson interpolates exactly through the inputs and extrapolates to an
ultimate forward rate, Nelson-Siegel-Svensson *smooths* a parametric shape
through them (4 / 6 parameters), so it is the natural tool for fitting a noisy
quoted curve or summarising a curve in a few factors. The model is
convention-agnostic: ``y`` is returned in the same compounding as the ``yields``
it was fitted to.

Calibration (:func:`fit_nelson_siegel_svensson`) is a separable least squares --
for fixed decays the loadings are constant so the betas are an ordinary linear
solve; the two decays are found by a deterministic grid-and-zoom search (no
third-party optimiser). Original implementation; the model is from the
Nelson-Siegel (1987) and Svensson (1994) papers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray


def _ns_loadings(tau: FloatArray, lam: float) -> tuple[FloatArray, FloatArray]:
    """The slope and curvature loadings ``(f1, f2)`` at decay ``lam``. The
    ``tau -> 0`` limit is ``f1 = 1``, ``f2 = 0`` (taken explicitly, the formula is
    ``0/0`` there)."""
    x = np.asarray(tau, dtype=np.float64) / lam
    ex = np.exp(-x)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(x > 0.0, (1.0 - ex) / x, 1.0)
    f2 = f1 - ex
    return f1, f2


def _design(tau: FloatArray, lambda1: float, lambda2: float) -> FloatArray:
    """The ``(n, 4)`` factor-loading matrix ``[1, f1(l1), f2(l1), f2(l2)]`` -- the
    linear part of the model for fixed decays."""
    f1a, f2a = _ns_loadings(tau, lambda1)
    _f1b, f2b = _ns_loadings(tau, lambda2)
    return np.column_stack([np.ones_like(f1a), f1a, f2a, f2b])


@dataclass(frozen=True, slots=True)
class NelsonSiegelSvensson:
    """A calibrated Nelson-Siegel-Svensson curve. Call it at maturities (years) to
    get spot rates; ``beta3 == 0`` is the plain Nelson-Siegel curve."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lambda1: float
    lambda2: float

    def __call__(self, maturities: FloatArray) -> FloatArray:
        """Spot rate(s) at ``maturities`` (years), same compounding as the fit."""
        return nelson_siegel_svensson(
            maturities, beta0=self.beta0, beta1=self.beta1, beta2=self.beta2,
            beta3=self.beta3, lambda1=self.lambda1, lambda2=self.lambda2)

    @property
    def short_rate(self) -> float:
        """The instantaneous short rate ``beta0 + beta1`` (``tau -> 0``)."""
        return self.beta0 + self.beta1

    @property
    def long_rate(self) -> float:
        """The asymptotic long rate ``beta0`` (``tau -> inf``)."""
        return self.beta0


def nelson_siegel_svensson(maturities: FloatArray, *, beta0: float, beta1: float,
                           beta2: float, beta3: float, lambda1: float,
                           lambda2: float) -> FloatArray:
    """Evaluate the Nelson-Siegel-Svensson spot rate at ``maturities`` (years).

    ``lambda1`` / ``lambda2`` must be positive. The two curvature loadings become
    collinear as ``lambda1 -> lambda2`` (the model is then over-parameterised);
    that is a fit concern, not an evaluation one. Returns a scalar for a scalar
    maturity, else an array."""
    if lambda1 <= 0.0 or lambda2 <= 0.0:
        raise ValueError("lambda1 and lambda2 must be positive")
    tau = np.asarray(maturities, dtype=np.float64)
    if np.any(tau < 0.0):
        raise ValueError("maturities must be non-negative")
    f1a, f2a = _ns_loadings(tau, lambda1)
    _f1b, f2b = _ns_loadings(tau, lambda2)
    y = beta0 + beta1 * f1a + beta2 * f2a + beta3 * f2b
    return y


def nelson_siegel(maturities: FloatArray, *, beta0: float, beta1: float,
                  beta2: float, lambda1: float) -> FloatArray:
    """Evaluate the plain Nelson-Siegel spot rate (the Svensson second hump set to
    zero) at ``maturities`` (years)."""
    return nelson_siegel_svensson(
        maturities, beta0=beta0, beta1=beta1, beta2=beta2, beta3=0.0,
        lambda1=lambda1, lambda2=1.0)


def _betas_and_sse(tau, yields, weights, lambda1, lambda2):
    """For fixed decays, the least-squares betas and the (weighted) SSE -- the
    inner linear solve of the separable fit."""
    design = _design(tau, lambda1, lambda2)
    if weights is None:
        beta, *_ = np.linalg.lstsq(design, yields, rcond=None)
        resid = design @ beta - yields
        return beta, float(resid @ resid)
    sw = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(design * sw[:, None], yields * sw, rcond=None)
    resid = design @ beta - yields
    return beta, float((weights * resid * resid).sum())


def fit_nelson_siegel_svensson(
    maturities: FloatArray, yields: FloatArray, *,
    weights: FloatArray | None = None,
    lambda_bounds: tuple[float, float] = (0.1, 30.0),
    grid: int = 25, zoom_rounds: int = 5,
) -> NelsonSiegelSvensson:
    """Calibrate a Nelson-Siegel-Svensson curve to observed ``(maturities,
    yields)`` by least squares.

    Separable fit: for each decay pair the betas are an ordinary linear solve, so
    only the two decays ``lambda1 < lambda2`` are searched -- on a log-spaced grid
    over ``lambda_bounds`` (years), refined by ``zoom_rounds`` of shrinking the
    window around the best pair. Deterministic (no random restarts), no
    third-party optimiser. ``weights`` (optional, one per maturity) weights the
    squared residuals -- e.g. emphasise liquid tenors. Needs at least 4 points
    (the model has 4 linear parameters).

    Returns the fitted :class:`NelsonSiegelSvensson`; reading ``ns(maturities)``
    back gives the smoothed curve.
    """
    tau = np.asarray(maturities, dtype=np.float64)
    y = np.asarray(yields, dtype=np.float64)
    if tau.ndim != 1 or tau.shape != y.shape:
        raise ValueError("maturities and yields must be 1-D and the same length")
    if tau.shape[0] < 4:
        raise ValueError("need at least 4 points to fit the 4 linear parameters")
    if np.any(tau < 0.0):
        raise ValueError("maturities must be non-negative")
    if grid < 2:
        raise ValueError("grid must be >= 2 (need at least two decay candidates)")
    if zoom_rounds < 1:
        raise ValueError("zoom_rounds must be >= 1")
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    lo, hi = lambda_bounds
    if not 0.0 < lo < hi:
        raise ValueError("lambda_bounds must satisfy 0 < lo < hi")

    best = None                                           # (sse, lambda1, lambda2, beta)
    log_lo, log_hi = np.log(lo), np.log(hi)
    for _ in range(zoom_rounds):
        candidates = np.exp(np.linspace(log_lo, log_hi, grid))
        for i, lambda1 in enumerate(candidates):
            for lambda2 in candidates[i + 1:]:           # enforce lambda1 < lambda2
                beta, sse = _betas_and_sse(tau, y, w, lambda1, lambda2)
                if best is None or sse < best[0]:
                    best = (sse, lambda1, lambda2, beta)
        # Zoom: shrink the (log) search window around the best decays.
        _sse, l1, l2, _beta = best
        half = 0.5 * (log_hi - log_lo) / (grid - 1) * 2.0
        log_lo = np.log(min(l1, l2)) - half
        log_hi = np.log(max(l1, l2)) + half

    _sse, lambda1, lambda2, beta = best
    return NelsonSiegelSvensson(
        beta0=float(beta[0]), beta1=float(beta[1]), beta2=float(beta[2]),
        beta3=float(beta[3]), lambda1=float(lambda1), lambda2=float(lambda2))
