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
