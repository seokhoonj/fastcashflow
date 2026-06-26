"""Solvency assessment -- asset-side market-risk SCR.

The market sub-risks (equity / property / FX / asset-concentration shocks) and
their correlation aggregation :func:`market_scr`, which also folds in the net
interest SCR from :mod:`._interaction`. Concentration risk reuses the credit
rating mapper from :mod:`._credit`.
"""
from __future__ import annotations

import numpy as np

from fastcashflow.basis import Basis
from fastcashflow.model_points import ModelPoints
from fastcashflow.solvency._engine import KICSInterest
from fastcashflow.assets import (
    Equity, Property, Portfolio, holding_value, portfolio_value,
)
from fastcashflow.solvency._assessment._interaction import _module_interest
from fastcashflow.solvency._assessment._credit import _rating_row, _SII_CQS


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

# Preferred equity (table 20): the price-fall shock differs by the K-ICS grade of
# the issue -- 1-2 grade 4%, 3 grade 6%, 4 grade 11%, 5 grade 21%, 6+ grade 35%;
# unrated defaults to 35% (the "other" issue-form row).
_PREFERRED_SHOCK_BY_GRADE = {
    "1-2": 0.04, "3": 0.06, "4": 0.11, "5": 0.21,
    "6": 0.35, "7": 0.35, "default": 0.35, "unrated": 0.35,
}


def _preferred_shock(rating: str) -> float:
    """The table-20 preferred-equity shock for a rating (via its K-ICS grade)."""
    return _PREFERRED_SHOCK_BY_GRADE[_rating_row(rating)]

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


def equity_scr(portfolio: Portfolio, regime) -> float:
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
            # preferred equity is charged by the issue's rating (table 20)
            shock = (_preferred_shock(h.credit_rating)
                     if h.risk_type == "preferred" else shocks[h.risk_type])
            by_type[h.risk_type] = (by_type.get(h.risk_type, 0.0)
                                    + h.market_value * shock)
    amounts = [a for a in by_type.values() if a > 0.0]    # losing types only
    n = len(amounts)
    if n == 0:
        return 0.0
    a = np.array(amounts, dtype=np.float64)
    R = np.full((n, n), _EQUITY_TYPE_CORR)
    np.fill_diagonal(R, 1.0)
    return float(np.sqrt(max(0.0, a @ R @ a)))


def property_scr(portfolio: Portfolio, regime) -> float:
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
# (Article 188) is a flat 25% shock per currency, summed. Holding values are in the
# reporting currency (won); the currency tag only selects the shock.
# ---------------------------------------------------------------------------

_FX_SHOCK_KRW = {              # K-ICS table 22, won-base currency shock (percent)
    "AUD": 30, "BRL": 50, "CAD": 25, "CHF": 40, "CLP": 30, "CNY": 25,
    "COP": 35, "CZK": 35, "DKK": 35, "EUR": 35, "GBP": 30, "HKD": 25,
    "HUF": 40, "IDR": 40, "ILS": 30, "INR": 25, "JPY": 40, "MXN": 30,
    "MYR": 25, "NOK": 35, "NZD": 35, "PEN": 25, "PHP": 25, "PLN": 35,
    "RON": 35, "RUB": 40, "SAR": 25, "SEK": 35, "SGD": 20, "THB": 25,
    "TRY": 55, "TWD": 20, "USD": 25, "ZAR": 45,
}

_SII_FX_SHOCK = 0.25           # Solvency II Article 188: 25% per foreign currency

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


def fx_scr(portfolio: Portfolio, regime, discount_annual) -> float:
    """The FX-risk SCR on the net foreign-currency exposure (vs the won, the local
    currency here).

    K-ICS: each currency's table-22 shock, summing the net-asset-value losses of
    the declining currencies under a won-up and a won-down scenario through a 0.5
    correlation, the worse of the two. Solvency II (Article 188): a flat 25% per
    currency, each currency's larger of the up / down loss, SUMMED (no
    diversification). Returns 0 for an unknown regime."""
    if regime.name not in ("K-ICS", "Solvency II"):
        return 0.0
    exposure = {}
    for h in portfolio.holdings:
        cur = getattr(h, "currency", "KRW")
        if cur == "KRW":
            continue
        exposure[cur] = exposure.get(cur, 0.0) + holding_value(h, discount_annual)

    if regime.name == "Solvency II":        # Article 188: 25% flat, per-currency, summed
        return float(sum(_SII_FX_SHOCK * abs(e) for e in exposure.values()))

    for cur in exposure:                    # K-ICS: table 22 must list the currency
        if cur not in _FX_SHOCK_KRW:
            raise ValueError(f"unknown currency {cur!r}; known: {sorted(_FX_SHOCK_KRW)}")
    down = {c: _FX_SHOCK_KRW[c] / 100.0 * e for c, e in exposure.items() if e > 0.0}
    up = {c: _FX_SHOCK_KRW[c] / 100.0 * (-e) for c, e in exposure.items() if e < 0.0}
    return max(_fx_aggregate(down), _fx_aggregate(up))   # price volatility: 0 (v1)


# ---------------------------------------------------------------------------
# Asset concentration risk -- the idiosyncratic risk of an undiversified book.
# K-ICS (tables 23 / 24): for each counterparty, the exposure ABOVE a limit (total
# assets x a rating-based percentage) is charged a factor; the per-counterparty
# charges combine at correlation 0 (root-sum-of-squares). Property is charged
# separately (individual and whole-book limits, the worse of the two). The asset
# concentration SCR is sqrt(counterparty^2 + property^2). It enters the market
# module as the independent (correlation-0) fifth sub-risk. Solvency II
# concentration (Articles 184-187) is a single-name excess charge, root-sum-of-squares.
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

# Solvency II concentration (Article 185 threshold CT, Article 186 risk factor g) by CQS.
_SII_CONC_THRESHOLD = {0: 0.03, 1: 0.03, 2: 0.03, 3: 0.015, 4: 0.015, 5: 0.015, 6: 0.015}
_SII_CONC_FACTOR = {0: 0.12, 1: 0.12, 2: 0.21, 3: 0.27, 4: 0.73, 5: 0.73, 6: 0.73}


def _sii_cqs(rating: str) -> int:
    """The Solvency II credit quality step for a rating; unrated -> CQS 3."""
    base = (rating or "").strip().upper().rstrip("+-0123456789")
    return _SII_CQS.get(base, 3)

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


def concentration_scr(portfolio: Portfolio, regime, discount_annual, *,
                      total_assets: float | None = None) -> float:
    """The asset-concentration SCR -- ``sqrt(counterparty^2 + property^2)``.

    Counterparty: exposures are grouped by ``issuer`` (deposits, equity and bonds);
    the amount above the limit (``total_assets`` times the rating band's percentage,
    table 23) is charged the band's factor, and the per-issuer charges combine at
    correlation 0. Property: each holding above the individual limit (6% of total
    assets) and the whole book above the total limit (25%) are charged 20% (table
    24), taking the worse of the two. ``total_assets`` defaults to the portfolio
    value. Solvency II (Articles 184-187) uses the single-name excess
    ``max(0, exposure - threshold(CQS) x assets) x g(CQS)`` aggregated as a
    root-sum-of-squares. Returns 0 when a book has no tagged issuers and no
    property, or for an unknown regime."""
    if regime.name not in ("K-ICS", "Solvency II"):
        return 0.0
    ta = total_assets if total_assets is not None else portfolio_value(
        portfolio, discount_annual)
    if ta <= 0.0:
        return 0.0

    if regime.name == "Solvency II":        # Articles 184-187: single-name excess, RSS
        exp_s, cqs_s = {}, {}
        for h in portfolio.holdings:
            issuer = getattr(h, "issuer", "").strip()
            if not issuer or isinstance(h, Property):
                continue
            exp_s[issuer] = exp_s.get(issuer, 0.0) + holding_value(h, discount_annual)
            q = _sii_cqs(getattr(h, "credit_rating", "AA"))
            cqs_s[issuer] = max(q, cqs_s.get(issuer, 0))   # most conservative (highest CQS)
        sq = 0.0
        for issuer, exp in exp_s.items():
            q = cqs_s[issuer]
            xs = max(0.0, exp - ta * _SII_CONC_THRESHOLD[q])
            sq += (xs * _SII_CONC_FACTOR[q]) ** 2
        return float(np.sqrt(sq))

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


def market_scr(portfolio: Portfolio, model_points: ModelPoints,
                      basis: Basis, *, regime,
                      interest_scenarios: KICSInterest | None = None) -> float:
    """The market-risk module SCR -- the interest (net of liabilities), equity,
    property, FX and asset-concentration sub-risks aggregated through the regime's
    market correlation matrix (``sqrt(c^T R c)``). Interest is the K-ICS five-
    scenario :func:`net_interest_kics_scr` when ``interest_scenarios`` is supplied
    (the supervisor-published shock set), else the worst-of-curves
    :func:`net_interest_scr` (Solvency II), else zero."""
    cal = _market_cal(regime)
    interest = _module_interest(portfolio, model_points, basis, regime,
                                interest_scenarios)
    c = np.array([interest, equity_scr(portfolio, regime),
                  property_scr(portfolio, regime),
                  fx_scr(portfolio, regime, basis.discount_annual),
                  concentration_scr(portfolio, regime, basis.discount_annual)],
                 dtype=np.float64)
    R = np.asarray(cal["market_correlation"], dtype=np.float64)
    return float(np.sqrt(max(0.0, c @ R @ c)))
