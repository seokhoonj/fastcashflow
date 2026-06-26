"""Assets and the solvency balance sheet -- the asset side of the ratio.

The solvency ratio is available capital over the required capital. fastcashflow
computes the liability side (the BEL and the SCR); this module adds a STATIC
(t=0) asset valuation so the ratio is computable: a portfolio's market value,
available capital (assets less liabilities), a NET interest-rate SCR (assets and
liabilities re-priced together under a curve shock), and the assembled ratio.

The SCR is an instantaneous shock-and-revalue, so a t=0 valuation is enough -- a
full dynamic asset projection (rolling, reinvestment) is not needed and is out of
scope. This module sits above :mod:`fastcashflow.alm` (it prices bonds) and
:mod:`fastcashflow.solvency` (it consumes the liability SCR); it adds no new
regulatory numbers.

The asset-side SCR modules are the market risk (interest / equity / property / FX /
asset concentration) and the credit risk (bond default + downgrade). Both regimes
are calibrated: K-ICS from the handbook (tables 22 / 29-31 / 23-24) and Solvency II
from the Delegated Regulation (Article 176 spread, Article 188 currency, Articles 184-187
concentration).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow._duration import DurationResult, _BP
from fastcashflow.alm import net_liability_cashflows


@dataclass(frozen=True, slots=True)
class Equity:
    """An equity holding carried at a given market value (asset-positive).

    ``risk_type`` selects the market-risk shock (``"developed"`` or ``"emerging"``
    market listed equity); the shock magnitude is the regime's calibration.
    ``currency`` (ISO code, "KRW" for domestic) drives the FX SCR. ``issuer``
    (counterparty) and ``credit_rating`` group exposures for the concentration
    SCR."""

    market_value: float
    risk_type: str = "developed"
    currency: str = "KRW"
    issuer: str = ""
    credit_rating: str = "AA"


@dataclass(frozen=True, slots=True)
class Property:
    """A property holding carried at a given market value. ``currency`` (ISO code)
    drives the FX SCR; property contributes to the property concentration SCR."""

    market_value: float
    currency: str = "KRW"


@dataclass(frozen=True, slots=True)
class Cash:
    """A cash holding, carried at face (curve-insensitive). ``currency`` (ISO code)
    drives the FX SCR. ``issuer`` (the deposit counterparty) and ``credit_rating``
    group exposures for the concentration SCR."""

    market_value: float
    currency: str = "KRW"
    issuer: str = ""
    credit_rating: str = "AA"


Holding = "Bond | Equity | Property | Cash"


@dataclass(frozen=True, slots=True)
class Portfolio:
    """An immutable set of holdings. Bonds are priced off the discount curve;
    equity / property / cash carry a given market value."""

    holdings: tuple


def holding_value(holding, discount_annual) -> float:
    """Market value of one holding -- a :class:`~fastcashflow.assets.Bond` priced at
    the curve, otherwise the holding's carried ``market_value``."""
    if isinstance(holding, Bond):
        return bond_value(holding, discount_annual)
    return float(holding.market_value)


def portfolio_value(portfolio: Portfolio, discount_annual) -> float:
    """Total market value of the portfolio at the given discount curve."""
    return float(sum(holding_value(h, discount_annual) for h in portfolio.holdings))


def available_capital(asset_portfolio_value: float, bel: float,
                      risk_margin: float) -> float:
    """Available capital (own funds) -- assets less liabilities on the prudential
    balance sheet: ``asset_portfolio_value - (bel + risk_margin)``. The liability is the
    technical provision (best estimate plus risk margin). Positive = solvent
    surplus. (Other balance-sheet liabilities, if any, are the caller's to net
    out of the portfolio value.)"""
    return asset_portfolio_value - (bel + risk_margin)


def portfolio_cashflows(portfolio: Portfolio, n_months: int) -> FloatArray:
    """Project the portfolio's asset cash flows onto a monthly grid.

    Returns ``(n_months + 1,)`` -- the cash received at each month ``0 .. n_months``
    (month 0 normally zero). Each :class:`~fastcashflow.assets.Bond` contributes its
    coupons and final redemption (:func:`~fastcashflow.assets.bond_cashflows`),
    placed at month ``round(time_years x 12)``; cash flows beyond ``n_months`` are
    dropped. Equity, property and cash carry NO scheduled cash flow in v1 (they
    are stocks held at market value, not scheduled streams) -- dividends, rent and
    cash interest are future work.

    This is the asset-side counterpart to
    :func:`fastcashflow.alm.net_liability_cashflows`; the two share the monthly
    grid, so the asset-liability cash-flow gap is their difference."""
    if n_months <= 0:
        raise ValueError(f"n_months must be positive, got {n_months}")
    flow = np.zeros(n_months + 1, dtype=np.float64)
    for holding in portfolio.holdings:
        if not isinstance(holding, Bond):
            continue                                  # v1: only bonds have scheduled CFs
        times, amounts = bond_cashflows(holding)
        months = np.rint(np.asarray(times) * 12.0).astype(np.int64)
        for m, amt in zip(months, amounts):
            if 1 <= m <= n_months:
                flow[m] += float(amt)
    return flow


def portfolio_value_path(portfolio: Portfolio, n_months: int,
                     discount_annual) -> FloatArray:
    """Market value of the still-held portfolio at each month ``0 .. n_months`` under
    run-off (no new business, no reinvestment).

    Each :class:`~fastcashflow.assets.Bond` is revalued on its REMAINING (future) cash
    flows -- as coupons and the redemption pay out, the bond amortises away (its
    value at month ``t`` is the present value, re-based to ``t``, of the cash flows
    dated after ``t``). Equity, property and cash are held flat at market value (v1:
    no scheduled run-off). Returns ``(n_months + 1,)``.

    This is the asset STOCK still on the book -- the cap on a forced sale
    (:func:`liquidate`), distinct from the reinvested-cash account
    (:func:`reinvest` / :func:`liquidate`) that carries the cash the run-off throws
    off. At month 0 it equals :func:`portfolio_value`; once every bond has
    matured it is just the flat (equity / property / cash) holdings."""
    if n_months <= 0:
        raise ValueError(f"n_months must be positive, got {n_months}")
    val_years = np.arange(n_months + 1, dtype=np.float64) / 12.0
    df_val = _annual_df(val_years, discount_annual)         # DF from 0 to each month
    flat = sum(float(h.market_value) for h in portfolio.holdings
               if not isinstance(h, Bond))
    path = np.full(n_months + 1, flat, dtype=np.float64)
    for holding in portfolio.holdings:
        if not isinstance(holding, Bond):
            continue
        times, amounts = bond_cashflows(holding)
        cf_months = np.rint(np.asarray(times) * 12.0).astype(np.int64)
        pv_cf = np.asarray(amounts) * _annual_df(times, discount_annual)
        paid = np.zeros(n_months + 2, dtype=np.float64)     # pv of CFs paid AT each month
        np.add.at(paid, np.minimum(cf_months, n_months + 1), pv_cf)
        remaining = float(pv_cf.sum()) - np.cumsum(paid)[:n_months + 1]
        path += remaining / df_val                          # re-base the remaining PV to t
    return path


def _annual_forward_curve(rate_scenarios: FloatArray) -> FloatArray:
    """Per-year annual-forward discount curve from a monthly annual-rate path.

    ``rate_scenarios`` is ``(n_scenarios, n_time)`` -- the annual rate at each
    projection month, the discount path a stochastic liability uses (the month is
    discounted by ``(1 + rate)^(1/12)``). The per-year forward for year ``j`` is the
    geometric product of that year's twelve monthly growth factors, annualised, so
    ``_annual_df`` reproduces the liability's cumulative discount factor at the year
    grid EXACTLY (a partial final year is annualised by its month count). Returns
    ``(n_scenarios, n_years)``."""
    rs = np.asarray(rate_scenarios, dtype=np.float64)
    growth = (1.0 + rs) ** (1.0 / 12.0)                     # monthly growth factor
    n_scen, n_time = rs.shape
    n_years = (n_time + 11) // 12
    c = np.empty((n_scen, n_years))
    for j in range(n_years):
        block = growth[:, 12 * j: 12 * j + 12]             # this year's months (<=12)
        c[:, j] = np.prod(block, axis=1) ** (12.0 / block.shape[1]) - 1.0
    return c


def portfolio_value_by_scenario(portfolio: Portfolio,
                            rate_scenarios: FloatArray) -> FloatArray:
    """Market value of ``portfolio`` under each rate scenario -- the co-moving asset
    leg of a stochastic solvency distribution.

    A 1-D ``(n_scenarios,)`` array is one flat annual rate per scenario; a 2-D
    ``(n_scenarios, n_time)`` array is the monthly annual-rate discount path,
    bootstrapped to a per-year annual-forward curve (:func:`_annual_forward_curve`)
    so a bond's discounting matches the liability's cumulative discount factor at the
    year grid. Bonds reprice on the curve; equity / property / cash are held flat (a
    rate move is the entity's interest exposure, the asset side of the duration gap).
    Returns ``(n_scenarios,)``."""
    rs = np.asarray(rate_scenarios, dtype=np.float64)
    if rs.ndim == 1:
        return np.array([portfolio_value(portfolio, float(r)) for r in rs])
    if rs.ndim != 2:
        raise ValueError("rate_scenarios must be 1-D (flat rates) or 2-D (curves)")
    curves = _annual_forward_curve(rs)
    return np.array([portfolio_value(portfolio, curves[s])
                     for s in range(rs.shape[0])])


@dataclass(frozen=True, slots=True)
class CashflowGap:
    """The asset-liability cash-flow ladder on the shared monthly grid.

    ``asset_cf`` / ``liability_cf`` are ``(n_time + 1,)`` -- the cash the asset
    portfolio receives and the net cash the liability pays at each month, both
    undiscounted. ``net_cf`` is their difference (positive = a month with a cash
    surplus to reinvest, negative = a shortfall to fund) and ``cumulative_net`` its
    running total (the funding position carried to each month, before any
    reinvestment return -- the running balance a static matching view inspects)."""

    asset_cf: FloatArray
    liability_cf: FloatArray

    @property
    def net_cf(self) -> FloatArray:
        return self.asset_cf - self.liability_cf

    @property
    def cumulative_net(self) -> FloatArray:
        return np.cumsum(self.net_cf)


def cashflow_gap(portfolio: Portfolio, measurement) -> CashflowGap:
    """The month-by-month asset-liability cash-flow gap.

    Nets the projected asset cash flows (:func:`portfolio_cashflows`) against
    the net liability cash flows (:func:`fastcashflow.alm.net_liability_cashflows`)
    on the engine's monthly grid. The liability's begin-of-month flows
    (``annuity - premium`` and maturity benefits) and mid-month flows (death /
    morbidity / disability / expense / surrender claims) are folded into one
    outflow per month; the asset bonds' coupons and redemptions land at their
    month boundaries. The result is the undiscounted liquidity ladder -- where the
    book throws off surplus cash and where it must find cash -- the foundation for
    the reinvestment / rollover trajectory to come.

    Requires a ``full=True`` measurement (it carries the cash flows); account-value
    (universal-life) books are rejected, as in ``net_liability_cashflows``."""
    flow_bom, flow_mid = net_liability_cashflows(measurement)
    n_time = flow_mid.shape[0]
    liability_cf = flow_bom.copy()
    liability_cf[:n_time] += flow_mid
    asset_cf = portfolio_cashflows(portfolio, n_time)
    return CashflowGap(asset_cf=asset_cf, liability_cf=liability_cf)


@dataclass(frozen=True, slots=True)
class ReinvestmentResult:
    """The reinvestment-account roll-forward of a cash-flow gap.

    ``balance`` is ``(n_time + 1,)`` -- the account balance at each month end:
    positive = surplus cash reinvested at the new-money rate, negative = a shortfall
    funded (borrowed) at the funding rate. ``interest`` is the return credited
    (charged) on the carried balance each month and ``net_cf`` echoes the gap's net
    cash flow added each month (``interest`` and ``net_cf`` reconcile the balance:
    ``balance[m] = balance[m-1] + interest[m] + net_cf[m]``).

    This is the GAP account only -- the surplus reinvested / shortfall funded. The
    held bonds' own returns are already in their coupon / redemption cash flows
    (inside ``net_cf``), so the account does NOT re-credit a yield on bond
    principal; ``opening_balance`` carries any initial cash cushion outside the
    bonds."""

    balance: FloatArray
    interest: FloatArray
    net_cf: FloatArray

    @property
    def closing_balance(self) -> float:
        """The account balance at the horizon (surplus if positive)."""
        return float(self.balance[-1])


def _monthly_factors(rate, n: int) -> FloatArray:
    """Per-month growth factors ``(1 + annual)^(1/12)`` for months ``1 .. n`` (index
    0 unused), from a flat scalar annual rate or a per-month annual-rate array (the
    monthly rate held flat past the array end). Annual compounding matches the
    engine's discounting convention."""
    r = np.asarray(rate, dtype=np.float64)
    out = np.ones(n + 1, dtype=np.float64)
    if r.ndim == 0:
        out[1:] = (1.0 + float(r)) ** (1.0 / 12.0)
        return out
    if r.shape[0] < n:
        r = np.concatenate([r, np.full(n - r.shape[0], r[-1])])
    out[1:] = (1.0 + r[:n]) ** (1.0 / 12.0)
    return out


def reinvest(gap: CashflowGap, *, reinvest_rate, funding_rate=None,
             opening_balance: float = 0.0) -> ReinvestmentResult:
    """Roll the cash-flow gap forward, reinvesting surplus and funding shortfall.

    Walks the monthly ``gap.net_cf`` ladder: each month the balance carried from the
    prior month earns one month of the ``reinvest_rate`` (on a non-negative balance)
    or pays the ``funding_rate`` (on a negative balance), then the month's net cash
    flow lands at month end (it earns no return in the month it arrives). The
    month-0 net flow (the inception premium) seeds the starting balance together
    with ``opening_balance`` (default 0 -- a pure gap account); at a zero rate the
    balance is then exactly the gap's ``cumulative_net``.

    ``reinvest_rate`` / ``funding_rate`` are ANNUAL rates, a flat scalar or a
    per-month path (the new-money curve -- the reinvestment-risk lever); the monthly
    factor is ``(1 + annual)^(1/12)``. ``funding_rate`` defaults to
    ``reinvest_rate`` (symmetric); a higher funding rate models a borrowing spread
    over the reinvestment yield.

    Returns a :class:`ReinvestmentResult`; ``closing_balance`` is the horizon
    surplus (positive) or accumulated funding cost (negative)."""
    net_cf = np.asarray(gap.net_cf, dtype=np.float64)
    n = net_cf.shape[0] - 1
    inv_f = _monthly_factors(reinvest_rate, n)
    fund_f = inv_f if funding_rate is None else _monthly_factors(funding_rate, n)
    balance = np.empty(n + 1, dtype=np.float64)
    interest = np.zeros(n + 1, dtype=np.float64)
    balance[0] = opening_balance + net_cf[0]
    for m in range(1, n + 1):
        prev = balance[m - 1]
        factor = inv_f[m] if prev >= 0.0 else fund_f[m]
        interest[m] = prev * (factor - 1.0)
        balance[m] = prev + interest[m] + net_cf[m]
    return ReinvestmentResult(balance=balance, interest=interest, net_cf=net_cf)


@dataclass(frozen=True, slots=True)
class LiquidationResult:
    """The forced-sale roll-forward of a cash-flow gap under a sell-to-fund policy.

    ``balance`` is ``(n_time + 1,)`` -- the cash account at each month end, floored
    at zero (a shortfall is met by selling assets, not by borrowing).
    ``forced_sale`` is the cash raised by selling each month (zero when the carried
    surplus covers the outflow) and ``realized_loss`` is the loss that sale
    crystallises at the stressed haircut. ``unfunded`` is the cash a shortfall
    needed but could NOT raise because the asset stock was exhausted (zero unless a
    finite ``available_assets`` cap was supplied) -- an insolvency signal.
    ``total_realized_loss`` / ``total_unfunded`` sum them: the cost of being a
    forced seller in a stressed market and the shortfall the assets could not cover
    -- the asset-side bite of the lapse<->rate interaction."""

    balance: FloatArray
    forced_sale: FloatArray
    realized_loss: FloatArray
    unfunded: FloatArray

    @property
    def total_realized_loss(self) -> float:
        """The realized loss over the horizon (the forced-sale cost)."""
        return float(self.realized_loss.sum())

    @property
    def total_unfunded(self) -> float:
        """The cash shortfall the asset stock could not cover (0 if uncapped)."""
        return float(self.unfunded.sum())


def liquidate(gap: CashflowGap, *, haircut: float, reinvest_rate=0.0,
              opening_balance: float = 0.0,
              available_assets=None) -> LiquidationResult:
    """Roll the cash-flow gap forward, meeting shortfalls by selling assets.

    The counterpart to :func:`reinvest` under a different liquidity policy: where
    ``reinvest`` borrows to fund a deficit, ``liquidate`` sells assets to floor the
    cash account at zero and recognises the loss of selling into a stressed market.
    Each month the carried surplus earns one month of ``reinvest_rate`` and the net
    cash flow lands; if the account would go negative, that shortfall is the cash
    raised by a forced sale and ``haircut * shortfall`` is the realized loss (the
    haircut is the loss per unit of cash raised -- the depressed-price discount).

    ``available_assets`` (optional, ``(n_time + 1,)`` -- e.g. from
    :func:`portfolio_value_path`) caps the forced sale at the FAIR-VALUE asset stock
    still on the book. Raising ``s`` of cash consumes ``s * (1 + haircut)`` of stock
    (the haircut loss is destroyed fair value too), so a finite stock raises at most
    ``stock / (1 + haircut)`` cash; any shortfall beyond that is ``unfunded`` (the
    book is insolvent for it). ``None`` (the default) assumes assets are always
    available -- the historical behaviour, ``unfunded`` all zero.

    ``haircut`` is the seam (the stressed-liquidation discount, e.g. 0.10); a
    forced sale under a wider stress carries a deeper haircut."""
    net_cf = np.asarray(gap.net_cf, dtype=np.float64)
    n = net_cf.shape[0] - 1
    grow = _monthly_factors(reinvest_rate, n)
    avail = None if available_assets is None else np.asarray(available_assets, np.float64)
    balance = np.empty(n + 1, dtype=np.float64)
    forced_sale = np.zeros(n + 1, dtype=np.float64)
    realized_loss = np.zeros(n + 1, dtype=np.float64)
    unfunded = np.zeros(n + 1, dtype=np.float64)
    sold_fv = 0.0                                     # cumulative FAIR VALUE of stock sold

    def settle(m: int, bal: float) -> float:
        nonlocal sold_fv
        # available_assets is fair value; raising `sell` cash at the haircut consumes
        # sell * (1 + haircut) of fair value (the loss is destroyed stock too), so the
        # cash a finite stock can raise is capacity / (1 + haircut).
        need = -bal
        if avail is None:
            sell = need
        else:
            capacity = max(0.0, float(avail[m]) - sold_fv)
            sell = min(need, capacity / (1.0 + haircut))
        forced_sale[m] = sell
        realized_loss[m] = sell * haircut
        sold_fv += sell * (1.0 + haircut)
        bal += sell
        if bal < 0.0:                                 # stock exhausted -> uncovered
            unfunded[m] = -bal
            bal = 0.0
        return bal

    bal = opening_balance + net_cf[0]
    if bal < 0.0:                                     # a shortfall already at inception
        bal = settle(0, bal)
    balance[0] = bal
    for m in range(1, n + 1):
        bal = balance[m - 1] * grow[m] + net_cf[m]
        if bal < 0.0:
            bal = settle(m, bal)
        balance[m] = bal
    return LiquidationResult(balance=balance, forced_sale=forced_sale,
                             realized_loss=realized_loss, unfunded=unfunded)


# ---------------------------------------------------------------------------
# Bonds -- the asset side's interest-rate sensitivity (single-sign cash flows,
# so the textbook Macaulay / Modified duration applies cleanly).
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Bond:
    """A fixed-coupon bullet bond. ``coupon_rate`` is the annual coupon as a
    fraction of ``face``; ``frequency`` is the number of coupons per year.

    ``credit_rating`` (external / S&P scale: AAA, AA, A, BBB, BB, B, CCC, D, or
    "unrated") and ``exposure_class`` ("corporate", "public", "securitisation")
    drive the credit-risk SCR (:func:`fastcashflow.solvency.credit_scr`); ``currency`` (ISO
    code, "KRW" for domestic) drives the FX SCR (:func:`fastcashflow.solvency.fx_scr`);
    ``issuer`` (counterparty name) groups exposures for the concentration SCR
    (:func:`fastcashflow.solvency.concentration_scr`). None of these affect the price or
    duration; the market value is in the reporting currency."""

    face: float
    coupon_rate: float
    maturity_years: float
    frequency: int = 1
    credit_rating: str = "AA"
    exposure_class: str = "corporate"
    currency: str = "KRW"
    issuer: str = ""


def bond_cashflows(bond: Bond) -> tuple[FloatArray, FloatArray]:
    """The bond's ``(times_years, amounts)`` -- a coupon at each period and the
    face repaid with the final coupon."""
    n = int(round(bond.maturity_years * bond.frequency))
    times = np.arange(1, n + 1, dtype=np.float64) / bond.frequency
    coupon = bond.face * bond.coupon_rate / bond.frequency
    amounts = np.full(n, coupon, dtype=np.float64)
    amounts[-1] += bond.face
    return times, amounts


def effective_maturity(bond: Bond) -> float:
    """The cash-flow-weighted average maturity ``sum(t * CF_t) / sum(CF_t)``
    (K-ICS effective maturity, undiscounted as written in the standard). Used to
    pick the credit-risk maturity bucket. A coupon bond's effective maturity is
    shorter than its final maturity (early coupons pull the weight in)."""
    t, a = bond_cashflows(bond)
    total = float(a.sum())
    return float((t * a).sum() / total) if total > 0.0 else 0.0


def _annual_df(times: FloatArray, discount_annual) -> FloatArray:
    """Annual-compounding discount factors at ``times`` (years) for a flat scalar
    rate or a per-year rate array (the spot, year by year, held flat past its
    end). Constant-force monthly discounting agrees with this at the year grid."""
    times = np.asarray(times, dtype=np.float64)
    c = np.asarray(discount_annual, dtype=np.float64)
    if c.ndim == 0:
        return (1.0 + float(c)) ** (-times)
    n_max = int(np.ceil(times.max())) if times.size else 0
    rates = np.array([c[min(k, c.shape[0] - 1)] for k in range(n_max)])
    cum = np.concatenate([[0.0], np.cumsum(np.log1p(rates))])   # cum[n] = sum_{k<n} ln(1+c_k)
    floor = np.floor(times).astype(np.int64)
    frac = times - floor
    last_ln = np.array([np.log1p(c[min(k, c.shape[0] - 1)]) for k in floor])
    return np.exp(-(cum[floor] + frac * last_ln))


def bond_value(bond: Bond, discount_annual) -> float:
    """Market value of the bond -- its cash flows discounted at the curve."""
    t, a = bond_cashflows(bond)
    return float((a * _annual_df(t, discount_annual)).sum())


def _bond_irr(times: FloatArray, amounts: FloatArray, pv: float) -> float:
    """The flat annual yield reproducing ``pv`` (bisection; price falls in yield).

    The bracket ``(-0.99, 100)`` contains any realistic bond yield -- a
    positive-cash-flow bond has price ``-> +inf`` as the yield approaches -100%
    and ``-> 0`` as it grows, so the root is always inside. Raises if the price is
    not bracketed (e.g. non-positive or non-monotone cash flows)."""
    lo, hi = -0.99, 100.0

    def f(y: float) -> float:
        return float((amounts * (1.0 + y) ** (-times)).sum()) - pv

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            "bond yield is not bracketed in (-0.99, 100) -- check the bond cash "
            "flows and price")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < 1e-10 or (hi - lo) < 1e-13:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def bond_duration(bond: Bond, discount_annual) -> DurationResult:
    """The bond's market value, Macaulay / Modified duration, DV01 and convexity.
    Macaulay is the present-value-weighted time; Modified is ``Macaulay / (1 + y)``
    with ``y`` the flat-equivalent yield; DV01 is ``Modified * value * 1bp`` (the
    value drop per +1bp); convexity is the yield-based
    ``sum(t(t+1) CF_t (1+y)^-(t+2)) / value`` in years^2."""
    t, a = bond_cashflows(bond)
    pv_t = a * _annual_df(t, discount_annual)
    pv = float(pv_t.sum())
    macaulay = float((t * pv_t).sum() / pv)
    y = _bond_irr(t, a, pv)
    modified = macaulay / (1.0 + y)
    convexity = float((t * (t + 1.0) * a * (1.0 + y) ** (-(t + 2.0))).sum() / pv)
    return DurationResult(pv=pv, macaulay=macaulay, modified=modified,
                          dv01=modified * pv * _BP, convexity=convexity)


__all__ = [
    "Equity", "Property", "Cash", "Bond", "Portfolio",
    "holding_value", "portfolio_value", "available_capital",
    "portfolio_cashflows", "portfolio_value_path", "portfolio_value_by_scenario",
    "bond_cashflows", "bond_value", "bond_duration", "effective_maturity",
    "CashflowGap", "cashflow_gap",
    "ReinvestmentResult", "reinvest", "LiquidationResult", "liquidate",
]
