"""Smith-Wilson discount-curve construction.

Builds a discount / spot-rate curve from a set of observed zero-coupon rates by
the Smith-Wilson method -- the standard technique for fitting the liquid part of
a yield curve and extrapolating the illiquid long end toward an ultimate forward
rate (UFR). It is the method the Korean responsibility-reserve external-
verification guideline prescribes for the insurance-liability discount curve
(Smith-Wilson for the par-to-spot conversion and the interpolation segment) and
the method EIOPA uses for the Solvency II risk-free curve; the same algorithm
serves the won curve and a foreign-currency curve -- only the inputs (observed
rates, last liquid point, UFR, alpha) differ by currency.

The curve is determined by, for ``N`` observed maturities ``u`` with prices
``m``,

    P(t) = exp(-w*t) + sum_j zeta_j * W(t, u_j)

where ``w = ln(1 + UFR)`` is the continuous ultimate forward rate, ``P`` is the
discount function (``P(u_i) = m_i`` is reproduced exactly), and the symmetric
Wilson kernel is

    W(t, u) = exp(-w*(t+u)) * ( a*min(t,u)
                                - 0.5 * exp(-a*max(t,u))
                                       * (exp(a*min(t,u)) - exp(-a*min(t,u))) )

with ``a`` the convergence-speed parameter ``alpha``. The ``zeta`` vector solves
the linear system ``m - mu = W @ zeta`` (``mu_i = exp(-w*u_i)``,
``W_ij = W(u_i, u_j)``). The forward rate converges to the UFR as ``t`` grows.

``alpha`` is an input (chosen outside the model, as EIOPA and the Korean
guideline both publish it per currency); a liquidity premium, if any, is added by
the caller to the observed rates or the returned spot curve -- this builds the
risk-free curve only. From scratch -- the formulae are the Smith-Wilson paper's
(A. Smith, T. Wilson, "Fitting Yield Curves with Long Term Constraints", 2001),
no third-party code is adapted.
"""
from __future__ import annotations

import numpy as np

from fastcashflow._typing import FloatArray


def _wilson(t: FloatArray, u: FloatArray, w: float, alpha: float) -> FloatArray:
    """The symmetric Wilson kernel ``W(t, u)`` broadcast over ``t`` x ``u``.

    ``t`` and ``u`` are 1-D; returns ``(len(t), len(u))``. ``w`` is the continuous
    UFR ``ln(1 + UFR)``.
    """
    ti = np.asarray(t, dtype=np.float64)[:, None]
    uj = np.asarray(u, dtype=np.float64)[None, :]
    lo = np.minimum(ti, uj)
    hi = np.maximum(ti, uj)
    return np.exp(-w * (ti + uj)) * (
        alpha * lo
        - 0.5 * np.exp(-alpha * hi) * (np.exp(alpha * lo) - np.exp(-alpha * lo))
    )


def smith_wilson_prices(
    maturities: FloatArray,
    rates: FloatArray,
    *,
    ufr: float,
    alpha: float,
    target: FloatArray,
) -> FloatArray:
    """Smith-Wilson discount factors ``P(t)`` at the ``target`` maturities.

    ``maturities`` / ``rates`` are the ``N`` observed zero-coupon maturities (in
    years) and their annual-compounded spot rates. ``ufr`` is the annual ultimate
    forward rate, ``alpha`` the convergence speed. Returns ``P`` at each ``target``
    maturity -- the discount factor ``(1 + spot)**(-t)``. ``P`` reproduces the
    observed prices exactly at the input maturities.
    """
    u = np.asarray(maturities, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    if u.ndim != 1 or r.ndim != 1 or u.shape != r.shape:
        raise ValueError("maturities and rates must be 1-D arrays of equal length")
    if u.size == 0:
        raise ValueError("at least one observed maturity is required")
    if np.any(u <= 0):
        raise ValueError("maturities must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if ufr <= -1.0:
        raise ValueError("ufr must be greater than -1")

    w = np.log1p(ufr)                                  # continuous UFR
    m = (1.0 + r) ** (-u)                              # observed prices
    mu = np.exp(-w * u)                                # (N,)
    wmat = _wilson(u, u, w, alpha)                     # (N, N)
    zeta = np.linalg.solve(wmat, m - mu)               # (N,)

    t = np.asarray(target, dtype=np.float64)
    return np.exp(-w * t) + _wilson(t, u, w, alpha) @ zeta


def smith_wilson(
    maturities: FloatArray,
    rates: FloatArray,
    *,
    ufr: float,
    alpha: float,
    years: int = 120,
) -> FloatArray:
    """Annual spot-rate curve by Smith-Wilson, at integer years ``1 .. years``.

    Fits the observed zero-coupon ``rates`` at ``maturities`` (years) and
    extrapolates to the ultimate forward rate ``ufr`` at convergence speed
    ``alpha``. Returns a ``(years,)`` array of annual spot rates -- the
    :attr:`~fastcashflow.Basis.discount_annual` curve (year ``y`` in entry
    ``y - 1``). The curve passes through the observed rates exactly at integer
    input maturities and its forward rate converges to ``ufr``.

    A liquidity premium, if the basis carries one, is added by the caller (to the
    observed ``rates`` or the returned spot curve); this returns the risk-free
    curve only. The same call serves any currency -- pass that currency's observed
    rates, last liquid point (the largest ``maturities`` entry), ``ufr`` and
    ``alpha``.
    """
    if years < 1:
        raise ValueError("years must be at least 1")
    t = np.arange(1, years + 1, dtype=np.float64)
    p = smith_wilson_prices(maturities, rates, ufr=ufr, alpha=alpha, target=t)
    return p ** (-1.0 / t) - 1.0


def _convergence_forward(
    maturities: FloatArray, rates: FloatArray, *, w: float, alpha: float, cp: float,
) -> float:
    """Continuous instantaneous forward rate ``f(cp) = -P'(cp)/P(cp)`` of the fit.

    ``w = ln(1 + UFR)``. ``cp`` (the convergence point) is assumed BEYOND the last
    observed maturity, so the Wilson kernel's ``min(cp, u_i)`` is the observed
    tenor ``u_i``; the discount function and its derivative then collapse to the
    closed forms ``P(cp) = exp(-w*cp) * (1 + alpha*X - exp(-alpha*cp)*Y)`` and
    ``P'(cp) = -w*P(cp) + exp(-w*cp)*alpha*exp(-alpha*cp)*Y`` with
    ``X = sum(u_i * mu_i * zeta_i)`` and ``Y = sum(sinh(alpha*u_i) * mu_i * zeta_i)``
    (``mu_i = exp(-w*u_i)``). This is the rate the convergence criterion targets.
    """
    u = np.asarray(maturities, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    m = (1.0 + r) ** (-u)                          # observed prices
    mu = np.exp(-w * u)
    zeta = np.linalg.solve(_wilson(u, u, w, alpha), m - mu)
    X = float(np.sum(u * mu * zeta))
    Y = float(np.sum(np.sinh(alpha * u) * mu * zeta))
    e = np.exp(-alpha * cp)
    p = np.exp(-w * cp) * (1.0 + alpha * X - e * Y)
    dp = -w * p + np.exp(-w * cp) * alpha * e * Y
    return float(-dp / p)


def smith_wilson_alpha(
    maturities: FloatArray,
    rates: FloatArray,
    *,
    ufr: float,
    convergence_point: float,
    tolerance: float = 1e-4,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
    max_iter: int = 100,
) -> float:
    """Solve the convergence-speed ``alpha`` from the long-end target.

    Returns the smallest ``alpha >= alpha_min`` for which the fitted forward rate
    at ``convergence_point`` (years) reaches the ultimate forward rate ``ufr`` to
    within ``tolerance``, compared as ``|(1 + f) - (1 + ufr)|`` -- the criterion
    the EIOPA and Korean risk-free-rate technical documentation use. ``alpha`` is
    otherwise an input to :func:`smith_wilson`; this derives it from the last-
    observed-term / convergence-point / UFR triple the supervisor publishes, so the
    curve is pinned by those alone (the published per-currency ``alpha`` IS this
    solver's output).

    The forward rate at the convergence point moves monotonically closer to the UFR
    as ``alpha`` rises (faster convergence), so the criterion holds on an upper
    interval of ``alpha`` and the smallest qualifying value is found by bisection.
    When already met at ``alpha_min`` the floor is returned; when not met even at
    ``alpha_max`` the latter is returned (the closest achievable). From the
    algorithm (A. Smith, T. Wilson, 2001) -- no third-party code is adapted.
    """
    if convergence_point <= 0:
        raise ValueError("convergence_point must be positive")
    if alpha_min <= 0 or alpha_max <= alpha_min:
        raise ValueError("require 0 < alpha_min < alpha_max")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    u_max = float(np.asarray(maturities, dtype=np.float64).max())
    if convergence_point <= u_max:
        raise ValueError(
            "convergence_point must be beyond the last observed maturity "
            f"({u_max}); the long-end forward is only defined past the fit")
    w = np.log1p(ufr)

    def gap(alpha: float) -> float:
        f = _convergence_forward(maturities, rates, w=w, alpha=alpha, cp=convergence_point)
        return abs(np.exp(f) - np.exp(w))          # |(1 + f) - (1 + ufr)|, level space

    if gap(alpha_min) <= tolerance:
        return alpha_min
    if gap(alpha_max) > tolerance:
        return alpha_max                           # cannot converge tighter; best effort
    lo, hi = alpha_min, alpha_max                  # gap(lo) > tol >= gap(hi): bisect the boundary
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if gap(mid) <= tolerance:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break
    return hi
