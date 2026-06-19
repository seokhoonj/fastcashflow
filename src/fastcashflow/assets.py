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
asset concentration) and the credit risk (bond default + downgrade). Credit, FX and
concentration risk are charged for K-ICS only; the Solvency II spread / counterparty,
currency and concentration calibrations are separate and not encoded here (deferred),
so those charges are zero under Solvency II.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fastcashflow.alm import Bond, bond_value, effective_maturity
from fastcashflow.basis import Basis
from fastcashflow.engine import measure
from fastcashflow.model_points import ModelPoints
from fastcashflow.solvency import RegimeSpec, required_capital


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
class AssetPortfolio:
    """An immutable set of holdings. Bonds are priced off the discount curve;
    equity / property / cash carry a given market value."""

    holdings: tuple


def holding_value(holding, discount_annual) -> float:
    """Market value of one holding -- a :class:`~fastcashflow.alm.Bond` priced at
    the curve, otherwise the holding's carried ``market_value``."""
    if isinstance(holding, Bond):
        return bond_value(holding, discount_annual)
    return float(holding.market_value)


def portfolio_value(portfolio: AssetPortfolio, discount_annual) -> float:
    """Total market value of the portfolio at the given discount curve."""
    return float(sum(holding_value(h, discount_annual) for h in portfolio.holdings))


def available_capital(portfolio_value: float, bel: float,
                      risk_margin: float) -> float:
    """Available capital (own funds) -- assets less liabilities on the prudential
    balance sheet: ``portfolio_value - (bel + risk_margin)``. The liability is the
    technical provision (best estimate plus risk margin). Positive = solvent
    surplus. (Other balance-sheet liabilities, if any, are the caller's to net
    out of the portfolio value.)"""
    return portfolio_value - (bel + risk_margin)


def net_interest_scr(portfolio: AssetPortfolio, model_points: ModelPoints,
                     basis: Basis, *, interest_curves: tuple) -> float:
    """The net interest-rate SCR -- the worst loss in own funds (assets less
    liabilities) over the regime's up / down curve shocks.

    A rate rise lowers BOTH the asset value (bonds) and the BEL; the capital is
    the fall in net asset value ``NAV(base) - NAV(stress)``, where ``NAV(c) =
    portfolio_value(c) - BEL(c)``. Both sides re-price on the SAME shocked curve
    (the shock's ``Stress`` rebuilds ``basis.discount_annual``, which prices the
    bonds and the liability alike). The worst of the up / down shocks is taken,
    floored at zero. A duration-matched book gives ~ 0 -- the immunised gap.

    ``interest_curves`` is the regime's tuple of interest-rate stresses
    (``RegimeSpec.interest_curves``); pass a non-empty tuple (the assembler
    handles a regime with no curves)."""
    base_nav = (portfolio_value(portfolio, basis.discount_annual)
                - float(measure(model_points, basis, full=False).bel.sum()))
    worst = 0.0
    for stress in interest_curves:
        mp_s, basis_s = stress.apply(model_points, basis)
        stress_nav = (portfolio_value(portfolio, basis_s.discount_annual)
                      - float(measure(mp_s, basis_s, full=False).bel.sum()))
        worst = max(worst, base_nav - stress_nav)
    return worst


# ---------------------------------------------------------------------------
# Asset-side market-risk SCR (equity / property shocks, factor x market value).
# Primary-source calibration (K-ICS handbook Ch.4 -- developed equity -35%,
# emerging -48%, property -25%; market sub-risk correlation table 19 with 0.25
# off-diagonals; top-level life <-> market 0.25 from table 3. Solvency II uses the
# same equity / property magnitudes; its top-level inter-module matrix is in the
# Directive (Annex IV point 1) and is not extractable here, so the top-level
# aggregation falls back to a simple sum -- no diversification credit).
# ---------------------------------------------------------------------------

# Market sub-risks ordered (interest, equity, property) for the correlation axis.
# K-ICS table 19: market sub-risk correlation -- interest / equity / property / FX
# / asset concentration. Note equity <-> FX is NEGATIVE 0.25 (the standard's
# triangle mark): a won spike tends to coincide with foreign selling that drops
# equities, so they diversify. Asset concentration is independent (correlation 0
# with every other sub-risk) -- it is each holding's own idiosyncratic risk.
_MARKET_CORRELATION = np.array([
    [1.00,  0.25, 0.25,  0.25, 0.00],
    [0.25,  1.00, 0.25, -0.25, 0.00],
    [0.25,  0.25, 1.00,  0.25, 0.00],
    [0.25, -0.25, 0.25,  1.00, 0.00],
    [0.00,  0.00, 0.00,  0.00, 1.00],
])

# K-ICS equity price-fall shocks by type (handbook 4-3): developed listed 35%,
# emerging listed 48%, infrastructure 20%, long-term holdings 20%, other 49%,
# preferred 35% (the table-20 unrated/other default; full rating differentiation is
# a follow-up). The type-level amounts aggregate at the 0.75 inter-type correlation.
_EQUITY_SHOCKS = {
    "developed": 0.35, "emerging": 0.48, "infrastructure": 0.20,
    "long_term": 0.20, "other": 0.49, "preferred": 0.35,
}
_EQUITY_TYPE_CORR = 0.75       # handbook 4-3.da.(4): inter-equity-type correlation

_MARKET_CALIBRATION = {
    "K-ICS": {
        "equity_shocks": _EQUITY_SHOCKS,
        "property_shock": 0.25,
        "market_correlation": _MARKET_CORRELATION,
        "insurance_market_corr": 0.25,     # table 3 (life-long-term <-> market)
    },
    "Solvency II": {
        "equity_shocks": _EQUITY_SHOCKS,
        "property_shock": 0.25,
        "market_correlation": _MARKET_CORRELATION,
        "insurance_market_corr": None,     # top-level matrix not extracted -> simple sum
    },
}


def _market_cal(regime):
    try:
        return _MARKET_CALIBRATION[regime.name]
    except KeyError:
        raise ValueError(
            f"no market-risk calibration for regime {regime.name!r} "
            f"(known: {sorted(_MARKET_CALIBRATION)})")


def equity_scr(portfolio: AssetPortfolio, regime) -> float:
    """The equity market-risk SCR -- the per-type amounts (each type's holdings'
    market value times its price-fall shock) aggregated at the 0.75 inter-type
    correlation (handbook 4-3). Types: developed / emerging listed, infrastructure,
    long_term, other, preferred. Raises on an unknown type."""
    shocks = _market_cal(regime)["equity_shocks"]
    by_type: dict[str, float] = {}
    for h in portfolio.holdings:
        if isinstance(h, Equity):
            if h.risk_type not in shocks:
                raise ValueError(
                    f"unknown equity risk_type {h.risk_type!r} for regime "
                    f"{regime.name!r}; known: {sorted(shocks)}")
            by_type[h.risk_type] = (by_type.get(h.risk_type, 0.0)
                                    + h.market_value * shocks[h.risk_type])
    amounts = [a for a in by_type.values() if a > 0.0]    # losing types only
    n = len(amounts)
    if n == 0:
        return 0.0
    a = np.array(amounts, dtype=np.float64)
    R = np.full((n, n), _EQUITY_TYPE_CORR)
    np.fill_diagonal(R, 1.0)
    return float(np.sqrt(max(0.0, a @ R @ a)))


def property_scr(portfolio: AssetPortfolio, regime) -> float:
    """The property market-risk SCR -- property market value times the regime's
    price-fall shock."""
    shock = _market_cal(regime)["property_shock"]
    total = sum(h.market_value * shock
                for h in portfolio.holdings if isinstance(h, Property))
    return max(0.0, float(total))


# ---------------------------------------------------------------------------
# FX risk -- a market sub-risk on net foreign-currency exposure. K-ICS (table 22):
# shock each currency vs the won, sum the net-asset-value LOSSES (declining
# currencies only) under a won-up (rates fall) and a won-down scenario through a
# 0.5 inter-currency correlation, and take the worse of the two. FX derivative
# price volatility is a further term, taken as 0 here. Solvency II currency risk
# is a separate calibration (a flat 25% shock) and is deferred. Holding values are
# in the reporting currency (won); the currency tag only selects the shock.
# ---------------------------------------------------------------------------

_FX_SHOCK_KRW = {              # K-ICS table 22, won-base currency shock (percent)
    "AUD": 30, "BRL": 50, "CAD": 25, "CHF": 40, "CLP": 30, "CNY": 25,
    "COP": 35, "CZK": 35, "DKK": 35, "EUR": 35, "GBP": 30, "HKD": 25,
    "HUF": 40, "IDR": 40, "ILS": 30, "INR": 25, "JPY": 40, "MXN": 30,
    "MYR": 25, "NOK": 35, "NZD": 35, "PEN": 25, "PHP": 25, "PLN": 35,
    "RON": 35, "RUB": 40, "SAR": 25, "SEK": 35, "SGD": 20, "THB": 25,
    "TRY": 55, "TWD": 20, "USD": 25, "ZAR": 45,
}

_FX_CALIBRATION = {"K-ICS": _FX_SHOCK_KRW, "Solvency II": None}

_FX_CORRELATION = 0.5          # table 22: inter-currency correlation (declining only)


def _fx_aggregate(losses) -> float:
    """Aggregate per-currency NAV losses through the 0.5 inter-currency
    correlation: ``sqrt(L^T R L)`` with R = 0.5 off-diagonal, 1 on it."""
    vals = [v for v in losses.values() if v > 0.0]
    n = len(vals)
    if n == 0:
        return 0.0
    L = np.array(vals, dtype=np.float64)
    R = np.full((n, n), _FX_CORRELATION)
    np.fill_diagonal(R, 1.0)
    return float(np.sqrt(max(0.0, L @ R @ L)))


def fx_scr(portfolio: AssetPortfolio, regime, discount_annual) -> float:
    """The FX-risk SCR -- the worse of the won-up and won-down currency shocks on
    the net foreign-currency exposure, aggregated through the 0.5 correlation.

    Each non-won holding's market value is netted by currency; under the won-up
    (foreign-down) scenario a net-long currency loses ``shock x exposure`` (only
    losing currencies are summed), and symmetrically for the won-down scenario on
    net-short currencies. The SCR is the larger aggregate. Returns 0 for Solvency
    II (its currency-risk calibration is deferred)."""
    shocks = _FX_CALIBRATION.get(regime.name)
    if shocks is None:
        return 0.0
    exposure = {}
    for h in portfolio.holdings:
        cur = getattr(h, "currency", "KRW")
        if cur == "KRW":
            continue
        if cur not in shocks:
            raise ValueError(
                f"unknown currency {cur!r}; known: {sorted(shocks)}")
        exposure[cur] = exposure.get(cur, 0.0) + holding_value(h, discount_annual)
    down = {c: shocks[c] / 100.0 * e for c, e in exposure.items() if e > 0.0}
    up = {c: shocks[c] / 100.0 * (-e) for c, e in exposure.items() if e < 0.0}
    return max(_fx_aggregate(down), _fx_aggregate(up))   # price volatility: 0 (v1)


# ---------------------------------------------------------------------------
# Asset concentration risk -- the idiosyncratic risk of an undiversified book.
# K-ICS (tables 23 / 24): for each counterparty, the exposure ABOVE a limit (total
# assets x a rating-based percentage) is charged a factor; the per-counterparty
# charges combine at correlation 0 (root-sum-of-squares). Property is charged
# separately (individual and whole-book limits, the worse of the two). The asset
# concentration SCR is sqrt(counterparty^2 + property^2). It enters the market
# module as the independent (correlation-0) fifth sub-risk. Solvency II
# concentration is a separate calibration and is deferred.
# ---------------------------------------------------------------------------

_CONCENTRATION_BANDS = {       # K-ICS table 23: (limit % of total assets, factor)
    "1-2": (0.040, 0.15),
    "3-4": (0.030, 0.25),
    "5-7": (0.015, 0.50),
}
_PROPERTY_CONCENTRATION = {    # K-ICS table 24
    "individual_limit": 0.06, "total_limit": 0.25, "factor": 0.20,
}
_BAND_ORDER = {"1-2": 0, "3-4": 1, "5-7": 2}   # higher = more conservative

_RATING_TO_BAND = {
    "AAA": "1-2", "AA": "1-2", "A": "3-4", "BBB": "3-4",
    "BB": "5-7", "B": "5-7", "CCC": "5-7",
}


def _concentration_band(rating: str) -> str:
    """Map an external rating to its K-ICS concentration band; unrated and default
    map to the most conservative band."""
    r = (rating or "").strip().upper()
    if r in ("UNRATED", "NR", "", "D"):
        return "5-7"
    base = r.rstrip("+-0123456789")
    return _RATING_TO_BAND.get(base, "5-7")


def concentration_scr(portfolio: AssetPortfolio, regime, discount_annual, *,
                      total_assets: float | None = None) -> float:
    """The asset-concentration SCR -- ``sqrt(counterparty^2 + property^2)``.

    Counterparty: exposures are grouped by ``issuer`` (deposits, equity and bonds);
    the amount above the limit (``total_assets`` times the rating band's percentage,
    table 23) is charged the band's factor, and the per-issuer charges combine at
    correlation 0. Property: each holding above the individual limit (6% of total
    assets) and the whole book above the total limit (25%) are charged 20% (table
    24), taking the worse of the two. ``total_assets`` defaults to the portfolio
    value. Returns 0 for Solvency II (its concentration calibration is deferred)
    and when a book has no tagged issuers and no property."""
    if regime.name != "K-ICS":
        return 0.0
    ta = total_assets if total_assets is not None else portfolio_value(
        portfolio, discount_annual)
    if ta <= 0.0:
        return 0.0

    exposure, band = {}, {}
    for h in portfolio.holdings:
        issuer = getattr(h, "issuer", "").strip()
        if not issuer or isinstance(h, Property):
            continue
        exposure[issuer] = exposure.get(issuer, 0.0) + holding_value(h, discount_annual)
        b = _concentration_band(getattr(h, "credit_rating", "AA"))
        if issuer not in band or _BAND_ORDER[b] > _BAND_ORDER[band[issuer]]:
            band[issuer] = b
    cp_sq = 0.0
    for issuer, exp in exposure.items():
        limit_pct, factor = _CONCENTRATION_BANDS[band[issuer]]
        excess = max(0.0, exp - ta * limit_pct)
        cp_sq += (excess * factor) ** 2
    counterparty = float(np.sqrt(cp_sq))

    props = [holding_value(h, discount_annual)
             for h in portfolio.holdings if isinstance(h, Property)]
    f = _PROPERTY_CONCENTRATION["factor"]
    ind_limit = ta * _PROPERTY_CONCENTRATION["individual_limit"]
    tot_limit = ta * _PROPERTY_CONCENTRATION["total_limit"]
    individual = float(np.sqrt(sum((max(0.0, p - ind_limit) * f) ** 2 for p in props)))
    whole = max(0.0, sum(props) - tot_limit) * f
    property_conc = max(individual, whole)

    return float(np.sqrt(counterparty ** 2 + property_conc ** 2))


def market_module_scr(portfolio: AssetPortfolio, model_points: ModelPoints,
                      basis: Basis, *, regime) -> float:
    """The market-risk module SCR -- the interest (net of liabilities), equity,
    property, FX and asset-concentration sub-risks aggregated through the regime's
    market correlation matrix (``sqrt(c^T R c)``). Interest is the net
    :func:`net_interest_scr` (zero when the regime supplies no interest curves,
    e.g. K-ICS)."""
    cal = _market_cal(regime)
    interest = (net_interest_scr(portfolio, model_points, basis,
                                 interest_curves=regime.interest_curves)
                if regime.interest_curves is not None else 0.0)
    c = np.array([interest, equity_scr(portfolio, regime),
                  property_scr(portfolio, regime),
                  fx_scr(portfolio, regime, basis.discount_annual),
                  concentration_scr(portfolio, regime, basis.discount_annual)],
                 dtype=np.float64)
    R = np.asarray(cal["market_correlation"], dtype=np.float64)
    return float(np.sqrt(max(0.0, c @ R @ c)))


# ---------------------------------------------------------------------------
# Operational risk -- a liability-side factor charge, added on top of the BSCR.
# K-ICS (table 40): max(premium exposure x 3.5%, current-estimate liability x
# 0.4%) for general life / long-term. Solvency II (Art 204): min(0.3 x BSCR,
# max(0.04 x premiums, 0.0045 x technical provisions)) + 0.25 x unit-linked
# expenses. Computed from the liability (premiums and BEL) -- no asset model.
# ---------------------------------------------------------------------------

_OPERATIONAL_CALIBRATION = {
    "K-ICS": {"method": "kics", "premium_factor": 0.035, "bel_factor": 0.004},
    "Solvency II": {"method": "sii", "premium_factor": 0.04, "bel_factor": 0.0045,
                    "cap_bscr": 0.30, "expul_factor": 0.25},
}


def _operational_cal(regime):
    try:
        return _OPERATIONAL_CALIBRATION[regime.name]
    except KeyError:
        raise ValueError(
            f"no operational-risk calibration for regime {regime.name!r}")


def operational_scr(model_points: ModelPoints, basis: Basis, regime, *,
                    bscr: float | None = None) -> float:
    """The operational-risk SCR -- a factor on the liability (premiums and BEL).

    K-ICS: ``max(premium x 3.5%, BEL x 0.4%)``. Solvency II: ``min(0.3 x bscr,
    max(0.04 x premium, 0.0045 x BEL)) + 0.25 x unit-linked expenses`` -- pass
    ``bscr`` (the basic SCR) for the cap; unit-linked expenses are 0 in v1. The
    premium exposure is the first projection year's earned premium; the BEL
    exposure is floored at zero."""
    cal = _operational_cal(regime)
    m = measure(model_points, basis, full=True)
    bel = max(0.0, float(m.bel.sum()))
    premium = max(0.0, float(m.cashflows.premium_cf[:, :12].sum()))
    op = max(premium * cal["premium_factor"], bel * cal["bel_factor"])
    if cal["method"] == "sii":
        if bscr is not None:
            op = min(cal["cap_bscr"] * bscr, op)
        op += cal["expul_factor"] * 0.0          # unit-linked expenses (v1: 0)
    return op


# ---------------------------------------------------------------------------
# Credit risk -- an asset-side factor charge on credit exposures (bonds).
# K-ICS (chapter 5): the credit risk factor = default + downgrade charge, read
# off a (rating x effective-maturity) grid that differs by exposure class
# (public / corporate / securitisation -- handbook tables 29 / 30 / 31). The
# factor is a percent of market value (it already embeds the spread shock), so
# the charge is factor x market value -- no re-measure. Effective maturity is the
# cash-flow-weighted average maturity (:func:`fastcashflow.effective_maturity`).
# External (S&P) ratings map to the K-ICS grades AAA/AA -> 1-2, A -> 3, BBB -> 4,
# BB -> 5, B -> 6, CCC and below -> 7, D -> default. Solvency II credit (spread /
# counterparty default) is a separate framework not encoded here -- deferred.
# ---------------------------------------------------------------------------

_CREDIT_FACTORS = {            # K-ICS handbook tables 29 / 30 / 31, in PERCENT;
    "public": {                # rows: K-ICS grade, columns: maturity bucket 0-1 .. 14+
        "1-2": (0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.3),
        "3": (0.4, 1, 1.3, 1.5, 1.8, 2, 2.2, 2.4, 2.5, 2.7, 2.8, 2.9, 3, 3, 3.1),
        "4": (1, 2.2, 2.6, 3, 3.3, 3.6, 3.9, 4.1, 4.2, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9),
        "5": (2.5, 5.1, 6, 6.6, 7, 7.3, 7.5, 7.6, 7.6, 7.7, 7.8, 7.8, 7.9, 7.9, 7.9),
        "6": (6.3, 10.8, 11.8, 12.3, 12.5, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7),
        "7": (22, 24.7, 25.2, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3),
        "unrated": (2.5, 5.1, 6, 6.6, 7, 7.3, 7.5, 7.6, 7.6, 7.7, 7.8, 7.8, 7.9, 7.9, 7.9),
        "default": (35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35),
    },
    "corporate": {             # the non-covered-bond ("other") rows for grades 1-2 / 3
        "1-2": (0.2, 0.7, 0.9, 1.2, 1.4, 1.6, 1.7, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.4, 2.5),
        "3": (0.6, 1.3, 1.6, 1.8, 2.1, 2.3, 2.6, 2.8, 3, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7),
        "4": (1.4, 3, 3.6, 4.1, 4.5, 4.9, 5.1, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9, 6, 6),
        "5": (3.6, 7.1, 8.3, 9, 9.4, 9.7, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8),
        "6": (8.9, 14.4, 15.3, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6),
        "7": (35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35),
        "unrated": (6.3, 10.7, 11.8, 12.3, 12.5, 12.6, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7),
        "default": (35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35),
    },
    "securitisation": {
        "1-2": (0.2, 0.7, 0.9, 1.2, 1.4, 1.6, 1.7, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.4, 2.5),
        "3": (0.6, 1.3, 1.6, 1.8, 2.1, 2.3, 2.6, 2.8, 3, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7),
        "4": (1.4, 3, 3.6, 4.1, 4.5, 4.9, 5.1, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9, 6, 6),
        "5": (10.8, 21.3, 24.9, 27, 28.2, 29.1, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4),
        "6": (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
        "7": (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
        "unrated": (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
        "default": (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
    },
}

# Per regime: the K-ICS grid above; Solvency II spread/counterparty is deferred.
_CREDIT_CALIBRATION = {"K-ICS": _CREDIT_FACTORS, "Solvency II": None}

_RATING_TO_ROW = {             # external (S&P) base letter -> K-ICS factor-table row
    "AAA": "1-2", "AA": "1-2", "A": "3", "BBB": "4", "BB": "5", "B": "6",
    "CCC": "7", "CC": "7", "C": "7", "D": "default",
}


def _rating_row(rating: str) -> str:
    """Map an external rating (e.g. ``"AA+"``, ``"BBB-"``, ``"unrated"``) to its
    K-ICS factor-table row, stripping the +/- and any numeric modifier."""
    r = (rating or "").strip().upper()
    if r in ("UNRATED", "NR", ""):
        return "unrated"
    base = r.rstrip("+-0123456789")
    return _RATING_TO_ROW.get(base, "unrated")


def _credit_bucket(maturity: float) -> int:
    """The maturity-bucket index for the factor grid: bucket k is ``k < m <= k+1``
    (so ``0-1`` is index 0), capped at index 14 (the ``14+`` bucket)."""
    return min(14, max(0, math.ceil(maturity) - 1))


def credit_scr(portfolio: AssetPortfolio, regime, discount_annual) -> float:
    """The credit-risk SCR -- each bond's market value times its credit factor.

    The factor is read off the K-ICS (rating x effective-maturity) grid for the
    bond's ``exposure_class`` (handbook tables 29 / 30 / 31); it is a percent of
    market value and already embeds the default + downgrade charge, so the SCR is
    ``sum(market_value x factor)`` -- no re-measure. ``Cash`` is risk-free and
    equity / property carry market (not credit) risk, so only bonds contribute.
    Returns 0 for Solvency II (its spread / counterparty framework is deferred)."""
    factors = _CREDIT_CALIBRATION.get(regime.name)
    if factors is None:
        return 0.0
    total = 0.0
    for h in portfolio.holdings:
        if isinstance(h, Bond):
            table = factors.get(h.exposure_class)
            if table is None:
                raise ValueError(
                    f"unknown bond exposure_class {h.exposure_class!r}; "
                    f"known: {sorted(factors)}")
            row = table[_rating_row(h.credit_rating)]
            factor = row[_credit_bucket(effective_maturity(h))] / 100.0
            total += bond_value(h, discount_annual) * factor
    return max(0.0, total)


# K-ICS table 3: the top-level correlation among the insurance, market and credit
# risk modules -- all pairwise 0.25 (life-long-term / market / credit).
_TOPLEVEL_CORRELATION = np.array([
    [1.00, 0.25, 0.25],
    [0.25, 1.00, 0.25],
    [0.25, 0.25, 1.00],
])

# Table 3 with the general (P&C) insurance module added: order is
# (life-long-term, general insurance, market, credit). Life-vs-general is 0; all
# other pairs are 0.25. Used when a caller supplies a general-insurance SCR (for a
# life + P&C book); the life-only engine leaves it at zero.
_TOPLEVEL_CORRELATION_4 = np.array([
    [1.00, 0.00, 0.25, 0.25],
    [0.00, 1.00, 0.25, 0.25],
    [0.25, 0.25, 1.00, 0.25],
    [0.25, 0.25, 0.25, 1.00],
])


def aggregate_required_capital(insurance: float, market: float, credit: float, *,
                               regime, operational: float = 0.0,
                               general_insurance: float = 0.0) -> float:
    """The basic required capital from disclosed module amounts -- the top-level
    aggregate of the (life) insurance, market and credit modules plus the
    operational charge (added OUTSIDE the aggregate).

    K-ICS uses the table-3 correlation (life-vs-general 0, every other pair 0.25;
    ``general_insurance`` adds a fourth P&C module); Solvency II's top-level matrix
    is not extracted, so the modules sum. The disclosed ``diversification effect``
    is the simple module sum minus this aggregate. Use it to reproduce a disclosed
    basic required capital from the published module risk amounts, or for a what-if
    on the module mix without re-running a book."""
    if general_insurance > 0.0:
        c = np.array([insurance, general_insurance, market, credit], dtype=np.float64)
        R = _TOPLEVEL_CORRELATION_4
    else:
        c = np.array([insurance, market, credit], dtype=np.float64)
        R = _TOPLEVEL_CORRELATION
    if regime.name == "K-ICS":
        agg = float(np.sqrt(c @ R @ c))
    else:                                   # SII: top-level matrix not extracted
        agg = float(c.sum())
    return agg + operational


@dataclass(frozen=True, slots=True, eq=False)
class SolvencyAssessment:
    """The asset-inclusive solvency picture at t=0 -- the full ratio and its parts.

    ``available_capital`` is ``portfolio_value - (bel + risk_margin)``. The market
    module aggregates the ``net_interest_scr`` (assets and liabilities), the
    ``equity_scr``, the ``property_scr``, the ``fx_scr`` and the
    ``concentration_scr`` through the market correlation; the
    ``bscr`` (basic SCR) aggregates the ``insurance_scr``, the (optional)
    ``general_insurance_scr``, the ``market_module_scr`` and the ``credit_scr`` at
    the top level; ``basic_required_capital`` adds the
    ``operational_scr`` on top of the BSCR. ``total_scr`` (the ratio denominator)
    subtracts the ``tax_adjustment`` (loss-absorbing capacity of deferred taxes)
    from the basic required capital. ``solvency_ratio`` is
    ``available_capital / total_scr``."""

    portfolio_value: float
    bel: float
    risk_margin: float
    available_capital: float
    insurance_scr: float
    general_insurance_scr: float
    net_interest_scr: float
    equity_scr: float
    property_scr: float
    fx_scr: float
    concentration_scr: float
    market_module_scr: float
    credit_scr: float
    operational_scr: float
    bscr: float
    basic_required_capital: float
    tax_adjustment: float
    total_scr: float
    solvency_ratio: float


def assess_solvency(portfolio: AssetPortfolio, model_points: ModelPoints,
                    basis: Basis, *, regime: RegimeSpec, tax_rate: float = 0.0,
                    tax_recoverability_limit: float | None = None,
                    catastrophe: float = 0.0, property_codes=(),
                    general_insurance_scr: float = 0.0) -> SolvencyAssessment:
    """Assemble the t=0 solvency ratio from the assets and the liability SCR.

    Runs :func:`~fastcashflow.required_capital` for the liability (insurance) SCR,
    values the portfolio, forms available capital (assets less the technical
    provision), and builds the market-risk module (net interest, equity, property,
    FX, concentration) aggregated through the market correlation. The BSCR
    aggregates the insurance, market and credit modules at the top level: K-ICS uses
    the table-3 correlation (all pairwise 0.25); Solvency II's top-level inter-module
    matrix is not extracted here, so it falls back to a simple sum (no
    diversification credit -- conservative). The operational-risk SCR is added on
    top to form the basic required capital.

    ``tax_adjustment`` (K-ICS chapter 7 -- the loss-absorbing capacity of deferred
    taxes) is then subtracted to give the total required capital, the ratio
    denominator: ``min(basic_required_capital x tax_rate, tax_recoverability_limit)``.
    ``tax_rate`` is the company's average effective rate (over its recent pre-tax
    profits) and defaults to 0 (no tax relief -- conservative); supply
    ``tax_recoverability_limit`` for the regulatory recoverability cap (else the
    relief is uncapped at ``basic x tax_rate``).

    ``catastrophe`` (the K-ICS catastrophe amount from
    :func:`~fastcashflow.catastrophe_scr`) and ``property_codes`` (the long-term
    property / other coverages, a +16% rate shock) fold into the insurance module
    (table-6 correlation); both default to off. ``general_insurance_scr`` (a
    caller-supplied P&C amount for a life + general book) enters the BSCR as a
    fourth top-level module (table 3: life-vs-general 0, else 0.25); the life-only
    engine leaves it at 0.

    Notes: K-ICS supplies no interest curves (its scenarios are caller-supplied),
    so the net interest component is zero here -- equity and property still apply.
    Credit, FX and concentration risk are charged for K-ICS only (the Solvency II
    equivalents are deferred). A non-positive total required capital (a risk-free
    book) gives an unbounded ratio.
    """
    scr = required_capital(model_points, basis, regime=regime,
                           catastrophe=catastrophe, property_codes=property_codes)
    pv = portfolio_value(portfolio, basis.discount_annual)
    ac = available_capital(pv, scr.base_bel, scr.risk_margin)

    cal = _market_cal(regime)
    ni = (net_interest_scr(portfolio, model_points, basis,
                           interest_curves=regime.interest_curves)
          if regime.interest_curves is not None else 0.0)
    eq = equity_scr(portfolio, regime)
    pr = property_scr(portfolio, regime)
    fx = fx_scr(portfolio, regime, basis.discount_annual)
    conc = concentration_scr(portfolio, regime, basis.discount_annual, total_assets=pv)
    c = np.array([ni, eq, pr, fx, conc], dtype=np.float64)
    R = np.asarray(cal["market_correlation"], dtype=np.float64)
    market = float(np.sqrt(max(0.0, c @ R @ c)))

    ins = scr.insurance_scr
    gen = max(0.0, general_insurance_scr)   # general (P&C) insurance, caller-supplied
    cr = credit_scr(portfolio, regime, basis.discount_annual)
    bscr = aggregate_required_capital(ins, market, cr, regime=regime,
                                      general_insurance=gen)   # table 3 (no operational)

    op = operational_scr(model_points, basis, regime, bscr=bscr)
    basic = bscr + op                       # K-ICS basic required capital (incl. op)

    tax_adj = 0.0
    if tax_rate > 0.0:                       # K-ICS ch.7: tax loss-absorption
        tax_adj = basic * tax_rate
        if tax_recoverability_limit is not None:
            tax_adj = min(tax_adj, max(0.0, tax_recoverability_limit))
    total = basic - tax_adj                  # total required capital (ratio denominator)

    if total > 0.0:
        ratio = ac / total
    else:
        ratio = float("inf") if ac >= 0.0 else float("-inf")
    return SolvencyAssessment(
        portfolio_value=pv, bel=scr.base_bel, risk_margin=scr.risk_margin,
        available_capital=ac, insurance_scr=ins, general_insurance_scr=gen,
        net_interest_scr=ni,
        equity_scr=eq, property_scr=pr, fx_scr=fx, concentration_scr=conc,
        market_module_scr=market, credit_scr=cr, operational_scr=op, bscr=bscr,
        basic_required_capital=basic, tax_adjustment=tax_adj,
        total_scr=total, solvency_ratio=ratio)


__all__ = [
    "Equity", "Property", "Cash", "AssetPortfolio", "SolvencyAssessment",
    "holding_value", "portfolio_value", "available_capital", "net_interest_scr",
    "equity_scr", "property_scr", "fx_scr", "concentration_scr",
    "market_module_scr", "credit_scr", "operational_scr",
    "aggregate_required_capital", "assess_solvency",
]
