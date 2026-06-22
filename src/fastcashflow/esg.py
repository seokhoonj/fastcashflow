"""Risk-neutral economic scenario generator (ESG).

A minimal, market-consistent generator that produces the scenarios the
measurement engine consumes -- it does not value anything itself. Two factors:

* a Hull-White one-factor (HW1F) short rate, calibrated so the model reprices the
  initial risk-free curve (a :func:`~fastcashflow.smith_wilson_prices` discount
  curve); ``.rates`` feeds ``gmm.stochastic`` (annual rate per month).
* a correlated lognormal (geometric Brownian motion) fund return whose
  risk-neutral drift is the short rate; ``.returns`` feeds ``vfa.measure`` /
  ``measure_tvog`` (monthly simple return).

Correctness is by no-arbitrage, not hand calculation: under the risk-neutral
measure the Monte-Carlo average of the stochastic discount factor reprices the
zero-coupon bond, and the discounted fund value is a martingale --
:meth:`EconomicScenarios.martingale_error` reports both.

Scope (v1): HW1F + lognormal, calibrated to the Smith-Wilson curve, with the
mean reversion / volatilities supplied by the caller (not fitted to a swaption
surface), antithetic variance reduction. Out of scope: a real-world measure,
multi-factor / stochastic-vol models, and quasi-Monte-Carlo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray
from fastcashflow._smith_wilson import smith_wilson_prices

_DT = 1.0 / 12.0                    # one month in years
_SQRT_DT = _DT ** 0.5


@dataclass(frozen=True, slots=True, eq=False)
class EconomicScenarios:
    """Correlated risk-neutral scenarios from :func:`simulate`.

    ``rates`` is ``(n_scenarios, n_time)`` -- the HW1F short rate as an **annual**
    rate for each projection month, the ``scenarios`` input to ``gmm.stochastic``.
    ``returns`` is ``(n_scenarios, n_time)`` -- the lognormal fund **monthly simple
    return**, the ``return_scenarios`` input to ``vfa.measure`` / ``measure_tvog``.
    The fields are used directly; no further unit conversion is needed.
    """

    rates: FloatArray             # (n_scen, n_time) annual short rate per month
    returns: FloatArray           # (n_scen, n_time) monthly fund return
    short_rate: FloatArray        # (n_scen, n_time) continuous short rate r_t
    initial_prices: FloatArray    # (n_time+1,) P(0,t) the model is calibrated to

    def martingale_error(self):
        """The two no-arbitrage check errors, as relative deviations.

        Returns ``(bond_error, equity_error)``:

        * ``bond_error`` -- the worst relative deviation, over all maturities, of
          the Monte-Carlo zero-coupon price ``mean_s exp(-sum_t r/12)`` from the
          calibrated ``P(0, T)``. Zero under perfect calibration / infinite paths.
        * ``equity_error`` -- the relative deviation of the discounted terminal
          fund value ``mean_s [ prod_t (1 + return) * exp(-sum_t r/12) ]`` from 1
          (the fund is a tradeable, so its discounted value is a martingale).
        """
        disc = np.exp(-np.cumsum(self.short_rate, axis=1) * _DT)   # (n_scen, n_time)
        mc_price = disc.mean(axis=0)                               # (n_time,)
        p0 = self.initial_prices[1:]                              # P(0, 1..n_time)
        bond_error = float(np.max(np.abs(mc_price / p0 - 1.0)))
        fund = np.cumprod(1.0 + self.returns, axis=1)             # S_t / S_0
        equity_error = float(np.abs((fund * disc).mean(axis=0)[-1] - 1.0))
        return bond_error, equity_error


def _initial_prices(maturities, rates, ufr, alpha, n_time):
    """``P(0, t)`` on the monthly grid, shape ``(n_time + 1,)`` (``P[0] = 1``),
    from the Smith-Wilson discount curve."""
    tau = np.arange(n_time + 1) * _DT
    prices = np.empty(n_time + 1)
    prices[0] = 1.0
    prices[1:] = smith_wilson_prices(maturities, rates, ufr=ufr, alpha=alpha,
                                     target=tau[1:])
    return prices


def _hw_drift(prices, a, sigma, n_time):
    """The HW1F deterministic drift ``alpha_i`` calibrated so the *discrete*-step
    model reprices every ``P(0, T)`` EXACTLY (only Monte-Carlo noise remains, with
    no monthly-discretisation bias).

    The mean-zero OU factor has ``x_i = vol_step * sum_{j<i} decay^{i-1-j} Z_j``,
    so ``Var(sum_{i<T} x_i) = vol_step^2 / (1-decay)^2 * sum_{m=1}^{T-1}
    (1-decay^m)^2``. The discrete bond ``E[exp(-dt sum_{i<T}(x_i+alpha_i))] =
    exp(-dt A_T + 0.5 dt^2 Var_T)`` with ``A_T = sum_{i<T} alpha_i``; setting it to
    ``P(0,T)`` gives ``A_T = 0.5 dt Var_T - ln P(0,T)/dt`` and ``alpha_i =
    A_{i+1} - A_i``.
    """
    decay = np.exp(-a * _DT)
    vol_step = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * _DT)) / (2.0 * a))
    m = np.arange(1, n_time)
    cum_g = np.concatenate(([0.0], np.cumsum((1.0 - decay ** m) ** 2)))  # S_0..S_{T-1}
    var_t = (vol_step * vol_step) / ((1.0 - decay) ** 2) * cum_g         # Var_{T=k+1}
    a_t = 0.5 * _DT * var_t - np.log(prices[1:]) / _DT                   # A_T, T=1..n_time
    a_full = np.concatenate(([0.0], a_t))                               # A_0..A_{n_time}
    return a_full[1:] - a_full[:-1]                                     # alpha_0..alpha_{n_time-1}


def _normals(n_scenarios, n_time, seed, antithetic):
    """Two independent standard-normal matrices ``(n_scenarios, n_time)``,
    deterministic in the seed. With ``antithetic`` the second half of each path
    set is the negation of the first (a variance-reduction pairing); an odd count
    drops the one spare mirror path. Two 2-D draws (not one 3-D array) keep the
    correlation step a plain elementwise combination."""
    rng = np.random.default_rng(seed)
    if antithetic:
        half = (n_scenarios + 1) // 2
        za = rng.standard_normal((half, n_time))
        zb = rng.standard_normal((half, n_time))
        return (np.concatenate([za, -za])[:n_scenarios],
                np.concatenate([zb, -zb])[:n_scenarios])
    return (rng.standard_normal((n_scenarios, n_time)),
            rng.standard_normal((n_scenarios, n_time)))


@njit(parallel=True, cache=True)
def _ou_short_rate(z_rate, drift, decay, vol_step):
    """The HW1F short rate ``r_i = x_i + alpha_i`` from the rate innovations.

    ``x`` is the exact mean-zero Ornstein-Uhlenbeck step ``x_{i+1} = x_i * decay
    + vol_step * z_i`` (``x_0 = 0``). The scenario axis is independent, so it runs
    in parallel (``prange``); the time axis is the sequential recursion. The
    arithmetic is identical to the plain numpy loop -- this only removes the
    per-step Python overhead and intermediates. Returns ``(n_scenarios, n_time)``.
    """
    n_scen, n_time = z_rate.shape
    short_rate = np.empty((n_scen, n_time))
    for s in prange(n_scen):
        x = 0.0
        for i in range(n_time):
            short_rate[s, i] = x + drift[i]
            x = x * decay + vol_step * z_rate[s, i]
    return short_rate


def simulate(
    maturities: FloatArray,
    rates: FloatArray,
    *,
    ufr: float,
    alpha: float,
    mean_reversion: float,
    rate_vol: float,
    equity_vol: float,
    correlation: float,
    n_scenarios: int,
    n_time: int,
    seed: int,
    antithetic: bool = True,
) -> EconomicScenarios:
    """Simulate correlated risk-neutral short-rate and fund-return scenarios.

    ``maturities`` / ``rates`` / ``ufr`` / ``alpha`` are the Smith-Wilson curve the
    short rate is calibrated to (so it reprices that curve). ``mean_reversion``
    (``a``) and ``rate_vol`` (``sigma``) parameterise the HW1F short rate;
    ``equity_vol`` and ``correlation`` (the rate/equity Brownian correlation) the
    fund return. Returns ``n_scenarios`` paths over ``n_time`` months; ``seed``
    makes the draw deterministic.
    """
    a = float(mean_reversion)
    sigma = float(rate_vol)
    sig_s = float(equity_vol)
    rho = float(correlation)
    n_scenarios = int(n_scenarios)
    n_time = int(n_time)
    if a <= 0.0:
        raise ValueError("mean_reversion must be positive")
    if sigma < 0.0 or sig_s < 0.0:
        raise ValueError("rate_vol and equity_vol must be >= 0")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    if n_scenarios < 1 or n_time < 1:
        raise ValueError("n_scenarios and n_time must be >= 1")
    if not np.all(np.isfinite([a, sigma, sig_s, rho, float(ufr), float(alpha)])):
        raise ValueError(
            "mean_reversion / rate_vol / equity_vol / correlation / ufr / alpha "
            "must be finite (a NaN would propagate silently to every scenario)")
    if not (np.all(np.isfinite(np.asarray(maturities, float)))
            and np.all(np.isfinite(np.asarray(rates, float)))):
        raise ValueError("maturities and rates must be finite")

    prices = _initial_prices(maturities, rates, ufr, alpha, n_time)
    drift = _hw_drift(prices, a, sigma, n_time)            # alpha_i, (n_time,)

    # Two correlated Brownian factors per step. Correlate the equity factor
    # explicitly (rho * rate + sqrt(1 - rho^2) * independent), reusing its own
    # buffer -- lighter and clearer than a 3-D array plus a Cholesky matmul.
    z_rate, z_eq = _normals(n_scenarios, n_time, seed, antithetic)
    z_eq *= np.sqrt(1.0 - rho * rho)
    z_eq += rho * z_rate

    # Exact Ornstein-Uhlenbeck short rate (the scenario axis runs in parallel).
    decay = np.exp(-a * _DT)
    vol_step = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * _DT)) / (2.0 * a))
    short_rate = _ou_short_rate(z_rate, drift, decay, vol_step)

    # Annual rate per month -- the engine's (1+r)^(1/12) discount then matches the
    # HW continuous month discount exp(-r/12). expm1 is exact near zero.
    annual = np.expm1(short_rate)
    # Lognormal fund return; risk-neutral drift = short rate, so the discounted
    # fund value is a martingale. exp(.) - 1 keeps every return > -1.
    fund_return = np.expm1((short_rate - 0.5 * sig_s * sig_s) * _DT
                           + sig_s * _SQRT_DT * z_eq)

    return EconomicScenarios(rates=annual, returns=fund_return,
                             short_rate=short_rate, initial_prices=prices)


def hull_white_rates(
    maturities: FloatArray,
    rates: FloatArray,
    *,
    ufr: float,
    alpha: float,
    mean_reversion: float,
    rate_vol: float,
    n_scenarios: int,
    n_time: int,
    seed: int,
    antithetic: bool = True,
) -> FloatArray:
    """The HW1F annual short-rate scenarios alone (the ``gmm.stochastic`` input).

    A thin wrapper over :func:`simulate` (zero equity vol / correlation) returning
    only ``(n_scenarios, n_time)`` annual rates -- for a fixed-income book with no
    fund-linked guarantee.
    """
    return simulate(
        maturities, rates, ufr=ufr, alpha=alpha, mean_reversion=mean_reversion,
        rate_vol=rate_vol, equity_vol=0.0, correlation=0.0,
        n_scenarios=n_scenarios, n_time=n_time, seed=seed, antithetic=antithetic,
    ).rates


__all__ = ["EconomicScenarios", "simulate", "hull_white_rates"]
