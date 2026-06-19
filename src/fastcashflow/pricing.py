"""Pricing -- premium solving and profit testing.

Premium solving exploits that fulfilment cash flows are linear in the premium:
claims, expenses and the in-force run-off do not depend on it, so
``FCF = A - premium * B``. Two valuations pin down ``A`` and ``B``, and the
premium that meets a profitability target then has a closed form -- no iteration.

Profit testing (re-exported from :mod:`fastcashflow.profit`) adds the value and
emergence of new business: the present-value metrics (``nbv``, ``profit_margin``),
the per-period ``signature``, and the rate metrics (``irr``, ``break_even_year``).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.basis import Basis, BasisRouter
from fastcashflow.curves import discount_monthly_curve
from fastcashflow.engine import measure
from fastcashflow.model_points import ModelPoints
from fastcashflow.profit import (
    ProfitSignature, break_even_year, irr, nbv, profit_margin, signature,
)
from fastcashflow.tvog import TVOGResult

__all__ = ["solve_premium", "statutory_reserve", "statutory_profit_signature",
           "interest_guarantee_tvog", "ProfitSignature", "TVOGResult",
           "nbv", "profit_margin", "signature", "irr", "break_even_year"]


def _with_premium(model_points: ModelPoints, premium: float) -> ModelPoints:
    """A copy of ``model_points`` with every level premium set to ``premium``.

    Every other field -- including the payment frequency -- is carried over
    unchanged, so the two valuations that pin down the premium see the same
    contract bar the premium itself.
    """
    return replace(
        model_points, premium=np.full(model_points.n_mp, premium)
    )


def solve_premium(
    model_points: ModelPoints,
    basis: Basis,
    *,
    break_even: bool = False,
    margin: float | None = None,
    csm: float | None = None,
) -> FloatArray:
    """Solve the level premium that meets a profitability target.

    Exactly one target must be given:

    * ``break_even`` -- the lowest non-onerous premium (FCF = 0, zero CSM).
    * ``margin``     -- a profit margin, ``CSM / PV(premiums) = margin``
      (e.g. ``0.10`` for 10%); must satisfy ``0 <= margin < 1``.
    * ``csm``        -- an absolute target CSM (profit) per model point.

    Every product field of ``model_points`` is used as given -- only
    ``premium`` is ignored, since it is the unknown being solved for.
    Returns the solved premium per model point, shape ``(n_mp,)``.
    """
    chosen = (break_even, margin is not None, csm is not None)
    if sum(chosen) != 1:
        raise ValueError(
            "specify exactly one target: break_even, margin or csm"
        )
    if margin is not None and not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must be in [0, 1), got {margin}")

    # FCF is linear in the premium -- FCF = A - premium * B -- so two
    # valuations (premium 0 and 1) pin the line down exactly. The fast path
    # computes the confidence-level RA only; cost-of-capital RA needs the
    # trajectory path (the inception headline is identical either way). A dict
    # (segmented) basis takes the trajectory path if any segment uses it.
    bases = (basis.segments.values()
             if isinstance(basis, BasisRouter) else (basis,))
    use_full = any(b.ra_method != "confidence_level" for b in bases)
    at_zero = measure(_with_premium(model_points, 0.0), basis, full=use_full)
    at_one = measure(_with_premium(model_points, 1.0), basis, full=use_full)
    a = at_zero.bel + at_zero.ra
    b = a - (at_one.bel + at_one.ra)

    zero_sens = np.abs(b) < 1e-12
    if np.any(zero_sens):
        raise ValueError(
            "solve_premium: FCF is insensitive to the premium for "
            f"{int(zero_sens.sum())} model point(s) -- cannot solve. "
            "Check that premium enters the cash flows (non-zero "
            "premium term and payment frequency)."
        )

    if break_even:
        return a / b
    if margin is not None:
        return a / (b * (1.0 - margin))
    return (csm + a) / b


def statutory_reserve(
    model_points: ModelPoints,
    statutory_basis: Basis,
) -> tuple[FloatArray, FloatArray]:
    """The net-level-premium (NLP) reserve trajectory on a locked statutory basis.

    Computed by projection -- the engine's backward present value IS the
    prospective reserve, so **no commutation functions** (Dx / Nx / Mx) are
    needed: the net premium is the break-even premium (it funds the benefits with
    no margin), and the reserve at each month is the BEL carrying that net premium.

    Returns ``(reserve, net_premium)``. ``reserve`` is ``(n_mp, n_time+1)`` -- the
    cohort prospective reserve, column 0 approximately zero (the net premium makes
    the issue value nil); ``net_premium`` is ``(n_mp,)``.

    ``statutory_basis`` is the locked reserving basis (its mortality / interest /
    lapse). For a pure NLP reserve use a deterministic one (``mortality_cv = 0``,
    no expense loading); a gross-premium or expense-loaded reserve follows from
    putting those into the basis.
    """
    net = solve_premium(model_points, statutory_basis, break_even=True)
    m = measure(replace(model_points, premium=net), statutory_basis)
    return m.bel_path, net


def statutory_profit_signature(
    model_points: ModelPoints,
    pricing_basis: Basis,
    statutory_basis: Basis,
    *,
    period_months: int = 12,
    earned_rate: float | None = None,
) -> ProfitSignature:
    """The traditional / statutory profit signature.

    Holds the net-level-premium reserve ``V`` on ``statutory_basis`` and lets the
    profit emerge on the ``pricing_basis`` experience (the actual gross premium in
    ``model_points`` and the best-estimate decrements). Per month, matching the
    engine's within-month discount convention (premium / annuity beginning of
    month, claims and expenses mid-month):

        profit_t = (V_t + premium_t - annuity_t)(1 + i)
                   - outgo_t (1 + i)^0.5 - V_{t+1}

    with ``i`` the monthly earned rate (``earned_rate`` if given, else the pricing
    discount). On a run where the pricing basis equals the statutory basis and the
    premium is the net premium this is identically zero (the reserve is
    self-financing); profit emerges from the premium loading (gross over net) and
    the interest spread (earned over the valuation rate). The result feeds
    :func:`irr` / :func:`break_even_year` once the day-0 strain is prepended.

    v1 assumes the pricing and statutory bases share decrements (mortality /
    lapse) -- only the valuation interest differs; a reserve re-based onto
    different decrements is a follow-up. Pass a single :class:`Basis` (not a
    router) for each.
    """
    if isinstance(pricing_basis, BasisRouter) or isinstance(statutory_basis, BasisRouter):
        raise NotImplementedError(
            "statutory_profit_signature takes a single Basis for each argument "
            "(profit testing is per product); resolve the router per segment.")
    reserve, _ = statutory_reserve(model_points, statutory_basis)
    m = measure(model_points, pricing_basis)             # actual gross premium
    cf = m.cashflows
    n_time = cf.premium_cf.shape[1]
    V = reserve.sum(axis=0)                               # portfolio cohort reserve
    if earned_rate is None:
        i = discount_monthly_curve(pricing_basis, n_time)   # (n_time,)
    else:
        i = np.full(n_time, (1.0 + earned_rate) ** (1.0 / 12.0) - 1.0)
    premium = cf.premium_cf.sum(axis=0)
    annuity = cf.annuity_cf.sum(axis=0)
    outgo = (cf.mortality_cf + cf.morbidity_cf + cf.disability_cf
             + cf.expense_cf + cf.surrender_cf).sum(axis=0)
    profit_m = ((V[:n_time] + premium - annuity) * (1.0 + i)
                - outgo * (1.0 + i) ** 0.5 - V[1:n_time + 1])
    # Aggregate the monthly profit into reporting periods.
    n_periods = (n_time + period_months - 1) // period_months
    pad = n_periods * period_months
    profit = np.zeros(pad)
    profit[:n_time] = profit_m
    profit = profit.reshape(n_periods, period_months).sum(axis=1)
    month_end = (np.minimum(np.arange(1, n_periods + 1) * period_months, n_time)
                 ).astype(np.int64)
    return ProfitSignature(period_months=period_months, month_end=month_end,
                           profit=profit)


def _validate_rate_block(rates: FloatArray, name: str) -> FloatArray:
    """A ``(n_path, n_time)`` annual-rate block the TVOG kernel can discount.

    Reject a non-2-D / empty / non-finite block, or any annual rate ``<= -100%``
    (which sign-flips the ``1 / (1 + r_m)`` discount and returns a
    plausible-looking wrong number). Returns the validated float array.
    """
    a = np.asarray(rates, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_scenarios, n_time)")
    if a.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one scenario row")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    if np.any(a <= -1.0):
        raise ValueError(
            f"{name} must be greater than -1 -- an annual rate of -100% or "
            "worse is invalid")
    return a


def _forward_central_path(initial_prices: FloatArray) -> FloatArray:
    """The deterministic central rate path -- the one-month forward curve implied
    by ``P(0, t)``, as an annual rate per month ``(n_time,)``.

    ``f(t) = (P(0, t+1) / P(0, t)) ** (-12) - 1`` is the exact inverse of the
    engine's ``(1 + annual) ** (1/12)`` month discount, so ``(1 + f) ** (1/12) =
    P(0, t) / P(0, t+1)``. Under a risk-neutral generator the forward path is the
    zero-volatility short-rate path that reproduces the same ``P(0, t)``, so it is
    the no-volatility baseline whose only gap to the scenario mean is the
    guarantee's convexity (the time value).
    """
    p = np.asarray(initial_prices, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] < 2:
        raise ValueError(
            "initial_prices must be a 1-D array of length n_time+1 (P(0,t), P[0]=1)")
    if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("initial_prices must be finite and strictly positive")
    one_month_disc = p[1:] / p[:-1]                    # P(0,t+1)/P(0,t), (n_time,)
    return one_month_disc ** (-12.0) - 1.0             # annual forward per month


def interest_guarantee_tvog(
    model_points: ModelPoints,
    statutory_basis: Basis,
    rate_scenarios: FloatArray,
    *,
    guaranteed_rate: float | None = None,
    central_rates: FloatArray | None = None,
    initial_prices: FloatArray | None = None,
) -> TVOGResult:
    """The cost of a traditional minimum interest-rate guarantee, split into
    intrinsic value and time value (TVOG).

    A general-account (GMM) traditional / interest-sensitive contract credits the
    policy reserve at a minimum guaranteed rate ``i_g``. When the company's earned
    investment rate ``r`` falls below ``i_g`` the company funds the shortfall on
    the reserve. Because ``max(i_g - r, 0)`` is convex, a deterministic projection
    at the central rate sees only the *intrinsic value*; the extra cost from rate
    volatility -- the *time value* -- appears only over many scenarios.

    This composes two pieces: the net-level-premium reserve ``V_t`` from
    :func:`statutory_reserve` (which accrues at ``i_g``), and the earned-rate
    scenarios ``rate_scenarios`` (e.g. ``fastcashflow.esg.simulate(...).rates``).
    Per scenario ``s`` the guarantee cost is the present value of the funded
    shortfall::

        cost_s = sum_t  D_s(t) * max(i_g_m - r_m[s, t], 0) * V_t

    with monthly rates ``i_g_m = (1 + i_g) ** (1/12) - 1`` and likewise ``r_m``,
    ``V_t`` the portfolio reserve held at the start of month ``t``, and ``D_s(t)``
    the end-of-month stochastic discount along the scenario's own short rate (the
    shortfall rides the reserve's full-month interest credit). The scenario mean
    of ``D_s`` reprices ``P(0, t+1)``, so the measure stays risk-neutral.

    Parameters
    ----------
    rate_scenarios
        ``(n_scenarios, n_time)`` annual earned rate per projection month, where
        ``n_time = statutory_reserve(...)[0].shape[1] - 1`` (the contract-boundary
        horizon, NOT the term -- the same horizon convention as the projection).
    guaranteed_rate
        The minimum guaranteed annual rate ``i_g``. Defaults to
        ``statutory_basis.discount_annual`` when that is a scalar; pass it
        explicitly if the statutory basis uses a per-year discount curve (v1 takes
        a scalar ``i_g``).
    central_rates
        ``(n_time,)`` annual central path for the intrinsic value. If omitted, the
        forward path implied by ``initial_prices`` is used; exactly one of the two
        must be given (no silent scenario-mean fallback).
    initial_prices
        ``(n_time+1,)`` ``P(0, t)`` the scenarios were calibrated to (e.g.
        ``EconomicScenarios.initial_prices``), used to derive the forward central
        path when ``central_rates`` is omitted.

    Returns
    -------
    TVOGResult
        ``guarantee_cost`` is the ``(n_scenarios,)`` cost distribution;
        ``intrinsic_value`` the central-path cost; ``time_value`` the TVOG
        (``mean(cost) - intrinsic``); ``total_value`` their sum. All are ``>= 0``:
        the guarantee is a cost to the entity. Net it by ADDING ``total_value`` to
        the fulfilment cash flows / BEL (subtracting from CSM / NBV);
        ``intrinsic_value`` is the part a deterministic central-rate valuation
        already captures, ``time_value`` the extra only stochastic scenarios show.
    """
    if isinstance(statutory_basis, BasisRouter):
        raise NotImplementedError(
            "interest_guarantee_tvog takes a single Basis (the guarantee is per "
            "product); resolve the router per segment.")

    reserve, _ = statutory_reserve(model_points, statutory_basis)
    V = reserve.sum(axis=0)                            # (n_time+1,) portfolio reserve
    n_time = V.shape[0] - 1
    Vt = V[:n_time]                                    # reserve held over each month

    rates = _validate_rate_block(rate_scenarios, "rate_scenarios")
    if rates.shape[1] != n_time:
        raise ValueError(
            f"rate_scenarios must have {n_time} columns (the contract-boundary "
            f"horizon = statutory_reserve(...)[0].shape[1] - 1), got "
            f"{rates.shape[1]}")

    if guaranteed_rate is None:
        ig = statutory_basis.discount_annual
        if np.ndim(ig) != 0:
            raise ValueError(
                "statutory_basis.discount_annual is a per-year curve; pass an "
                "explicit scalar guaranteed_rate (v1 takes a scalar i_g)")
        guaranteed_rate = float(ig)
    ig_m = (1.0 + guaranteed_rate) ** (1.0 / 12.0) - 1.0

    if central_rates is not None:
        central = np.asarray(central_rates, dtype=np.float64)
        if central.ndim != 1 or central.shape[0] != n_time:
            raise ValueError(
                f"central_rates must be 1-D of length {n_time} (n_time), got "
                f"shape {central.shape}")
        if not np.all(np.isfinite(central)) or np.any(central <= -1.0):
            raise ValueError("central_rates must be finite and greater than -1")
    elif initial_prices is not None:
        central = _forward_central_path(initial_prices)
        if central.shape[0] != n_time:
            raise ValueError(
                f"initial_prices implies a horizon of {central.shape[0]} months, "
                f"but the reserve horizon is {n_time}")
    else:
        raise ValueError(
            "supply central_rates or initial_prices to define the deterministic "
            "central path (no scenario-mean fallback)")

    def _cost_block(block: FloatArray) -> FloatArray:
        """PV of the funded shortfall for each row of an annual-rate block."""
        r_m = (1.0 + block) ** (1.0 / 12.0) - 1.0
        shortfall = np.maximum(ig_m - r_m, 0.0)
        ones = np.ones((block.shape[0], 1))
        bom = np.concatenate(
            [ones, np.cumprod(1.0 / (1.0 + r_m), axis=1)[:, :-1]], axis=1)
        discount = bom / (1.0 + r_m)                   # end-of-month, (n_path, n_time)
        return (discount * shortfall) @ Vt             # (n_path,)

    guarantee_cost = _cost_block(rates)
    intrinsic_value = float(_cost_block(central[None, :])[0])
    time_value = float(guarantee_cost.mean() - intrinsic_value)
    return TVOGResult(guarantee_cost=guarantee_cost,
                      intrinsic_value=intrinsic_value, time_value=time_value)
