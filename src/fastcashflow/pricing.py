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

__all__ = ["solve_premium", "statutory_reserve", "statutory_profit_signature",
           "ProfitSignature", "nbv", "profit_margin", "signature", "irr",
           "break_even_year"]


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
