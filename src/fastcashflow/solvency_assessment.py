"""Solvency assessment -- the capital / coverage-ratio layer.

Assembles the asset-side market and credit SCR, the liability SCR
(:mod:`fastcashflow.solvency`), and available capital into the coverage ratio
(:func:`assess_solvency` / :func:`vfa_assess_solvency`), plus the asset-liability
interaction loss and the dynamic-scenario overlay (:func:`dynamic_solvency` /
:func:`dynamic_solvency_vfa`). Sits above :mod:`fastcashflow.assets` (the asset
model and the cash-flow gap), :mod:`fastcashflow.alm` and
:mod:`fastcashflow.solvency`; it adds the asset-side risk modules and the assembly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                       # annotation only -- avoids a runtime cycle
    from fastcashflow._mass_lapse_reinsurance import CedantSolvencyRelief

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.engine import measure
from fastcashflow.model_points import ModelPoints
from fastcashflow.alm import (
    Bond, bond_value, bond_duration, effective_maturity, _annual_df,
)
from fastcashflow.solvency import (
    RegimeSpec, required_capital, vfa_required_capital, vfa_equity_scr,
    vfa_interest_scr, KICSInterest, interest_with_dynamic_lapse,
)
from fastcashflow.assets import (
    Equity, Property, Cash, AssetPortfolio, holding_value, asset_portfolio_value,
    available_capital, asset_portfolio_cashflows, asset_value_path,
    asset_value_by_scenario,
    CashflowGap, cashflow_gap, vfa_cashflow_gap, liquidate, LiquidationResult,
)


@dataclass(frozen=True, slots=True)
class InteractionResult:
    """The asset-liability interaction loss under a coupled rate / dynamic-lapse
    stress -- the two distinct, additive bites of the same rate move.

    ``base_nav`` / ``stressed_nav`` are the net asset value (assets less BEL) before
    and after the coupled stress, both at market value. ``revaluation_loss`` is
    their difference -- the mark-to-market hit (bonds reprice down, the BEL moves
    with the dynamic lapse). ``forced_sale_loss`` is the liquidation FRICTION ON TOP
    -- the haircut cost of selling assets below fair value to meet the surge in
    surrender outflow (a cost the fair-value revaluation does not see). ``total_loss``
    is the two summed; they do not double count (one is mark-to-market, the other is
    the friction below market)."""

    base_nav: float
    stressed_nav: float
    forced_sale_loss: float

    @property
    def revaluation_loss(self) -> float:
        """The mark-to-market NAV loss under the coupled stress."""
        return self.base_nav - self.stressed_nav

    @property
    def total_loss(self) -> float:
        """The full interaction loss -- revaluation plus forced-sale friction."""
        return self.revaluation_loss + self.forced_sale_loss


def _portfolio_nav(portfolio: AssetPortfolio, model_points: ModelPoints,
                   basis: Basis) -> float:
    """Net asset value at market: ``asset_portfolio_value(curve) - BEL``."""
    return (asset_portfolio_value(portfolio, basis.discount_annual)
            - float(measure(model_points, basis, full=False).bel.sum()))


def interaction_loss(portfolio: AssetPortfolio, model_points: ModelPoints,
                     basis: Basis, *, shift: float, lapse_sensitivity: float,
                     haircut: float, reinvest_rate=0.0,
                     opening_balance: float = 0.0) -> InteractionResult:
    """The asset-liability interaction loss of a coupled rate / dynamic-lapse stress.

    Ties the pieces together: a parallel ``shift`` reprices the bonds down and,
    through :func:`~fastcashflow.solvency.interest_with_dynamic_lapse`
    (``lapse_sensitivity`` the injected elasticity), lifts the lapse rate -- the
    mark-to-market ``revaluation_loss``. The lifted lapse surges the surrender
    outflow, deepening the liquidity shortfall on the stressed cash-flow gap
    (:func:`cashflow_gap`); funding it as a forced seller (:func:`liquidate` at
    ``haircut``) crystallises the ``forced_sale_loss`` on top. ``total_loss`` is the
    full bite -- the friction the duration-matched, fair-value view alone misses.

    ``reinvest_rate`` / ``opening_balance`` parameterise the liquidation roll-forward
    (a surplus earns the rate before a later shortfall; the opening cash cushion
    defaults to zero -- shortfalls are met purely by selling)."""
    return _interaction(portfolio, model_points, basis, shift=shift,
                        lapse_sensitivity=lapse_sensitivity, haircut=haircut,
                        reinvest_rate=reinvest_rate, opening_balance=opening_balance)[0]


def _interaction(portfolio: AssetPortfolio, model_points: ModelPoints, basis: Basis,
                 *, shift, lapse_sensitivity, haircut, reinvest_rate, opening_balance):
    """The interaction loss AND the underlying forced-sale roll-forward, so a caller
    that needs the liquidity trajectory (e.g. :func:`dynamic_solvency`) does not
    re-run the stressed measurement."""
    base_nav = _portfolio_nav(portfolio, model_points, basis)
    mp_s, basis_s = interest_with_dynamic_lapse(shift, lapse_sensitivity).apply(
        model_points, basis)
    stressed_nav = _portfolio_nav(portfolio, mp_s, basis_s)
    liq = liquidate(cashflow_gap(portfolio, measure(mp_s, basis_s, full=True)),
                    haircut=haircut, reinvest_rate=reinvest_rate,
                    opening_balance=opening_balance)
    res = InteractionResult(base_nav=base_nav, stressed_nav=stressed_nav,
                            forced_sale_loss=liq.total_realized_loss)
    return res, liq


def _portfolio_nav_vfa(portfolio: AssetPortfolio, model_points: ModelPoints,
                       basis: Basis) -> float:
    """Net asset value for a variable (VFA) book: assets at market less the VFA net
    BEL (the guarantee-excess + expense - fee leg). The account-value portion of
    the benefit is unit-funded, so only that net liability sits on the entity's
    general account; the assets discount at the entity curve ``basis.discount_annual``
    while the VFA BEL discounts internally at the underlying-items return."""
    from fastcashflow._vfa import measure_vfa
    return (asset_portfolio_value(portfolio, basis.discount_annual)
            - float(measure_vfa(model_points, basis, full=False).bel.sum()))


def vfa_interaction_loss(portfolio: AssetPortfolio, model_points: ModelPoints,
                         basis: Basis, *, return_shock: float,
                         lapse_sensitivity: float, haircut: float,
                         reinvest_rate=0.0,
                         opening_balance: float = 0.0) -> InteractionResult:
    """The asset-liability interaction loss of a market shock on a variable book.

    The VFA counterpart of :func:`interaction_loss`. An immediate proportional
    account-value shock ``return_shock`` (a market drop today, ``-0.30`` = -30%)
    lowers the account value, so the GMDB / GMAB moves in-the-money and the
    guarantee cost (the VFA net BEL) rises -- the mark-to-market ``revaluation_loss``.
    The moneyness dynamic lapse (:func:`fastcashflow.vfa.moneyness_lapse_scale`,
    ``lapse_sensitivity`` the elasticity) holds MORE policies on the now-valuable
    guarantee, deepening that cost AND surging the guarantee-excess outflow; funding
    the liquidity shortfall on the stressed VFA cash-flow gap as a forced seller
    (:func:`liquidate` at ``haircut``) crystallises the ``forced_sale_loss`` on top.
    ``total_loss`` is the full bite of the coupled shock.

    Only the LIABILITY side moves here (assets held at the unchanged entity curve);
    a market shock to the entity's own equity / property holdings is the separable
    static market SCR. Both the closed-form variable-annuity and the account-backed
    universal-life paths are supported."""
    return _interaction_vfa(portfolio, model_points, basis, return_shock=return_shock,
                            lapse_sensitivity=lapse_sensitivity, haircut=haircut,
                            reinvest_rate=reinvest_rate,
                            opening_balance=opening_balance)[0]


def _interaction_vfa(portfolio: AssetPortfolio, model_points: ModelPoints,
                     basis: Basis, *, return_shock, lapse_sensitivity, haircut,
                     reinvest_rate, opening_balance):
    """The VFA interaction loss AND the forced-sale roll-forward, so
    :func:`dynamic_solvency` does not re-measure the stressed book."""
    from fastcashflow._vfa import measure_vfa
    base_nav = _portfolio_nav_vfa(portfolio, model_points, basis)
    av = np.asarray(model_points.account_value, dtype=np.float64)
    mp_s = replace(model_points, account_value=av * (1.0 + return_shock))
    asset_val = asset_portfolio_value(portfolio, basis.discount_annual)
    stressed_nav = asset_val - float(measure_vfa(
        mp_s, basis, full=False, lapse_sensitivity=lapse_sensitivity).bel.sum())
    m_s = measure_vfa(mp_s, basis, full=True, lapse_sensitivity=lapse_sensitivity)
    liq = liquidate(vfa_cashflow_gap(portfolio, m_s), haircut=haircut,
                    reinvest_rate=reinvest_rate, opening_balance=opening_balance)
    res = InteractionResult(base_nav=base_nav, stressed_nav=stressed_nav,
                            forced_sale_loss=liq.total_realized_loss)
    return res, liq


def _nav_delta(portfolio: AssetPortfolio, model_points: ModelPoints, basis: Basis):
    """A callable mapping a curve :class:`~fastcashflow.solvency.Stress` to the NET
    asset value DECREASE it causes -- ``NAV(base) - NAV(stress)`` with
    ``NAV(c) = asset_portfolio_value(c) - BEL(c)`` (see :func:`_portfolio_nav`). The
    asset and liability legs re-price on the SAME shocked curve (the stress rebuilds
    ``basis.discount_annual``, which prices the bonds and the liability alike), so a
    duration-matched book gives ~0."""
    base_nav = _portfolio_nav(portfolio, model_points, basis)

    def delta(stress) -> float:
        mp_s, basis_s = stress.apply(model_points, basis)
        return base_nav - _portfolio_nav(portfolio, mp_s, basis_s)
    return delta


def net_interest_scr(portfolio: AssetPortfolio, model_points: ModelPoints,
                     basis: Basis, *, interest_curves: tuple) -> float:
    """The net interest-rate SCR -- the worst loss in own funds (assets less
    liabilities) over the regime's up / down curve shocks.

    A rate rise lowers BOTH the asset value (bonds) and the BEL; the capital is
    the fall in net asset value (see :func:`_nav_delta`). The worst of the up /
    down shocks is taken, floored at zero. A duration-matched book gives ~ 0 -- the
    immunised gap. This is the Solvency II form; K-ICS uses the five-scenario
    :func:`net_interest_kics_scr`.

    ``interest_curves`` is the regime's tuple of interest-rate stresses
    (``RegimeSpec.interest_curves``); pass a non-empty tuple (the assembler
    handles a regime with no curves)."""
    delta = _nav_delta(portfolio, model_points, basis)
    return max(0.0, max((delta(s) for s in interest_curves), default=0.0))


def net_interest_kics_scr(portfolio: AssetPortfolio, model_points: ModelPoints,
                          basis: Basis, *, scenarios: KICSInterest) -> float:
    """The K-ICS net interest-rate SCR -- the five-scenario aggregation on NET asset
    value (handbook p.205):

        sqrt( max(up, down)^2 + max(flat, steep)^2 ) + mean_reversion

    The net-asset-value decrease (assets less liabilities, both re-priced on the
    same shocked curve; see :func:`_nav_delta`) is the per-scenario amount -- the
    proper K-ICS interest risk is measured on the whole balance sheet, not the
    liability alone. Each directional amount is floored at zero; the mean-reversion
    amount is signed and can raise OR lower the charge (handbook 4-2.(1)-5), so the
    result is returned without an outer floor (matching the formula). ``scenarios``
    is the supervisor-published shock set as a
    :class:`~fastcashflow.solvency.KICSInterest`."""
    delta = _nav_delta(portfolio, model_points, basis)
    cap, _ = scenarios.capital(delta)
    return cap


def _module_interest(portfolio, model_points, basis, regime,
                     interest_scenarios) -> float:
    """The market module's interest sub-risk: the K-ICS five-scenario net amount
    when ``interest_scenarios`` is supplied, else the worst-of-curves net amount
    when the regime carries interest curves (Solvency II), else zero."""
    if interest_scenarios is not None:
        return net_interest_kics_scr(portfolio, model_points, basis,
                                     scenarios=interest_scenarios)
    if regime.interest_curves is not None:
        return net_interest_scr(portfolio, model_points, basis,
                                interest_curves=regime.interest_curves)
    return 0.0


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
# (Art 188) is a flat 25% shock per currency, summed. Holding values are in the
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

_SII_FX_SHOCK = 0.25           # Solvency II Art 188: 25% per foreign currency

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
    """The FX-risk SCR on the net foreign-currency exposure (vs the won, the local
    currency here).

    K-ICS: each currency's table-22 shock, summing the net-asset-value losses of
    the declining currencies under a won-up and a won-down scenario through a 0.5
    correlation, the worse of the two. Solvency II (Art 188): a flat 25% per
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

    if regime.name == "Solvency II":        # Art 188: 25% flat, per-currency, summed
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
# concentration (Art 184-187) is a single-name excess charge, root-sum-of-squares.
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

# Solvency II concentration (Art 185 threshold CT, Art 186 risk factor g) by CQS.
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


def concentration_scr(portfolio: AssetPortfolio, regime, discount_annual, *,
                      total_assets: float | None = None) -> float:
    """The asset-concentration SCR -- ``sqrt(counterparty^2 + property^2)``.

    Counterparty: exposures are grouped by ``issuer`` (deposits, equity and bonds);
    the amount above the limit (``total_assets`` times the rating band's percentage,
    table 23) is charged the band's factor, and the per-issuer charges combine at
    correlation 0. Property: each holding above the individual limit (6% of total
    assets) and the whole book above the total limit (25%) are charged 20% (table
    24), taking the worse of the two. ``total_assets`` defaults to the portfolio
    value. Solvency II (Art 184-187) uses the single-name excess
    ``max(0, exposure - threshold(CQS) x assets) x g(CQS)`` aggregated as a
    root-sum-of-squares. Returns 0 when a book has no tagged issuers and no
    property, or for an unknown regime."""
    if regime.name not in ("K-ICS", "Solvency II"):
        return 0.0
    ta = total_assets if total_assets is not None else asset_portfolio_value(
        portfolio, discount_annual)
    if ta <= 0.0:
        return 0.0

    if regime.name == "Solvency II":        # Art 184-187: single-name excess, RSS
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


def market_module_scr(portfolio: AssetPortfolio, model_points: ModelPoints,
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
                    bscr: float | None = None, measure_fn=None) -> float:
    """The operational-risk SCR -- a factor on the liability (premiums and BEL).

    K-ICS: ``max(premium x 3.5%, BEL x 0.4%)``. Solvency II: ``min(0.3 x bscr,
    max(0.04 x premium, 0.0045 x BEL)) + 0.25 x unit-linked expenses`` -- pass
    ``bscr`` (the basic SCR) for the cap; unit-linked expenses are 0 in v1. The
    premium exposure is the first projection year's earned premium; the BEL
    exposure is floored at zero.

    ``measure_fn`` is the liability measurement (default the GMM
    :func:`~fastcashflow.engine.measure`); pass :func:`~fastcashflow.vfa.measure`
    for a variable book's BEL / premium exposure."""
    cal = _operational_cal(regime)
    mf = measure_fn if measure_fn is not None else measure
    m = mf(model_points, basis, full=True)
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
# BB -> 5, B -> 6, CCC and below -> 7, D -> default. Solvency II uses the Art-176
# spread stress (piecewise-linear in modified duration by credit quality step).
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

# Solvency II spread risk on bonds (Art 176): the stress factor is piecewise-linear
# in modified duration, a + b x (dur - lower), with (a, b) per credit quality step
# (CQS 0-6) and duration bucket [0-5, 5-10, 10-15, 15-20, 20+].
_SII_SPREAD = {                # CQS -> [(lower, a, b), ...]
    0: [(0, 0.000, 0.009), (5, 0.045, 0.005), (10, 0.070, 0.005), (15, 0.095, 0.005), (20, 0.120, 0.005)],
    1: [(0, 0.000, 0.011), (5, 0.055, 0.006), (10, 0.085, 0.005), (15, 0.110, 0.005), (20, 0.135, 0.005)],
    2: [(0, 0.000, 0.014), (5, 0.070, 0.007), (10, 0.105, 0.005), (15, 0.130, 0.005), (20, 0.155, 0.005)],
    3: [(0, 0.000, 0.025), (5, 0.125, 0.015), (10, 0.200, 0.010), (15, 0.250, 0.010), (20, 0.300, 0.005)],
    4: [(0, 0.000, 0.045), (5, 0.225, 0.025), (10, 0.350, 0.018), (15, 0.440, 0.005), (20, 0.466, 0.005)],
    5: [(0, 0.000, 0.075), (5, 0.375, 0.042), (10, 0.585, 0.005), (15, 0.610, 0.005), (20, 0.635, 0.005)],
}
_SII_CQS = {                   # S&P base letter -> credit quality step (5 = CQS 5 and 6)
    "AAA": 0, "AA": 1, "A": 2, "BBB": 3, "BB": 4, "B": 5, "CCC": 5, "CC": 5, "C": 5,
}


def _sii_spread_stress(rating: str, modified_duration: float) -> float:
    """The Solvency II Art-176 spread stress factor for a bond's rating and modified
    duration; unrated maps to CQS 3 (BBB-equivalent) as a v1 simplification."""
    r = (rating or "").strip().upper()
    base = r.rstrip("+-0123456789")
    cqs = _SII_CQS.get(base, 3)
    d = max(0.0, modified_duration)
    buckets = _SII_SPREAD[cqs]
    idx = min(4, int(d // 5)) if d > 0 else 0
    if d > 20.0:
        idx = 4
    lower, a, b = buckets[idx]
    return min(1.0, a + b * (d - lower))


# Per regime: the K-ICS grid above; Solvency II uses the Art-176 spread stress.
_CREDIT_CALIBRATION = {"K-ICS": _CREDIT_FACTORS, "Solvency II": "sii_spread"}

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
    Solvency II uses the Art-176 spread stress (piecewise-linear in modified
    duration by credit quality step)."""
    factors = _CREDIT_CALIBRATION.get(regime.name)
    if factors is None:
        return 0.0
    total = 0.0
    for h in portfolio.holdings:
        if not isinstance(h, Bond):
            continue
        if factors == "sii_spread":             # Solvency II Art 176
            mod = bond_duration(h, discount_annual).modified
            factor = _sii_spread_stress(h.credit_rating, mod)
        else:                                   # K-ICS rating x maturity grid
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

# Solvency II BSCR top-level correlation (Delegated Regulation (EU) 2015/35, Annex
# IV). Among (life, market, counterparty-default) every pair is 0.25 -- the same
# values as K-ICS table 3, so the 3-module matrix is shared. With the non-life
# (general insurance) module added the only differences from K-ICS are
# non-life <-> default = 0.5 and life <-> non-life = 0 (the rest 0.25); order is
# (life, general/non-life, market, default/credit).
_SII_TOPLEVEL_CORRELATION_4 = np.array([
    [1.00, 0.00, 0.25, 0.25],
    [0.00, 1.00, 0.25, 0.50],
    [0.25, 0.25, 1.00, 0.25],
    [0.25, 0.50, 0.25, 1.00],
])


def aggregate_required_capital(insurance: float, market: float, credit: float, *,
                               regime, operational: float = 0.0,
                               general_insurance: float = 0.0) -> float:
    """The basic required capital from disclosed module amounts -- the top-level
    aggregate of the (life) insurance, market and credit modules plus the
    operational charge (added OUTSIDE the aggregate).

    K-ICS uses the table-3 correlation; Solvency II uses the Annex IV BSCR matrix.
    For (life, market, credit) the two coincide (all pairs 0.25), so the 3-module
    aggregate is the same; with ``general_insurance`` (a fourth P&C module) they
    differ only in general-vs-credit (K-ICS 0.25, Solvency II 0.5) and share
    life-vs-general 0. The disclosed ``diversification effect`` is the simple module
    sum minus this aggregate. Use it to reproduce a disclosed basic required capital
    from the published module risk amounts, or for a what-if on the module mix
    without re-running a book."""
    if general_insurance > 0.0:
        c = np.array([insurance, general_insurance, market, credit], dtype=np.float64)
        R = (_SII_TOPLEVEL_CORRELATION_4 if regime.name == "Solvency II"
             else _TOPLEVEL_CORRELATION_4)
    else:
        c = np.array([insurance, market, credit], dtype=np.float64)
        R = _TOPLEVEL_CORRELATION       # 3-module values coincide for K-ICS and SII
    return float(np.sqrt(c @ R @ c)) + operational


@dataclass(frozen=True, slots=True, eq=False)
class SolvencyAssessment:
    """The asset-inclusive solvency picture at t=0 -- the full ratio and its parts.

    ``available_capital`` is ``asset_portfolio_value - (bel + risk_margin)``. The market
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

    asset_portfolio_value: float
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


_RELIEF_TOL = 1e-9   # float-noise band before a relief is deemed to exceed its module


def assess_solvency(portfolio: AssetPortfolio, model_points: ModelPoints,
                    basis: Basis, *, regime: RegimeSpec, tax_rate: float = 0.0,
                    tax_recoverability_limit: float | None = None,
                    catastrophe: float = 0.0, property_codes=(),
                    general_insurance_scr: float = 0.0,
                    interest_scenarios: KICSInterest | None = None,
                    relief: "CedantSolvencyRelief | None" = None) -> SolvencyAssessment:
    """Assemble the t=0 solvency ratio from the assets and the liability SCR.

    Runs :func:`~fastcashflow.required_capital` for the liability (insurance) SCR,
    values the portfolio, forms available capital (assets less the technical
    provision), and builds the market-risk module (net interest, equity, property,
    FX, concentration) aggregated through the market correlation. ``interest_scenarios``
    (a :class:`~fastcashflow.solvency.KICSInterest`) makes the net interest sub-risk
    the K-ICS five-scenario amount on net asset value; without it the net interest is
    the regime's worst-of curves (Solvency II) or zero (K-ICS supplies no curves). The
    interest risk sits in the market module (net of assets and liabilities), NOT in
    the insurance module -- ``required_capital`` is run without interest here. The BSCR
    aggregates the insurance, market and credit modules at the top level: K-ICS uses
    the table-3 correlation, Solvency II the Annex IV BSCR matrix; for the (life,
    market, credit) modules the two coincide (all pairwise 0.25). The operational-risk
    SCR is added on top to form the basic required capital.

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

    ``relief`` (a :class:`~fastcashflow.mass_lapse_reinsurance.CedantSolvencyRelief`
    from :func:`~fastcashflow.mass_lapse_reinsurance.cedant_solvency_relief`) folds
    a mass-lapse reinsurance treaty into the ratio: the insurance module drops by
    ``relief.insurance_relief`` (the diversified life-module lapse relief), the
    counterparty-default charge ``relief.counterparty_default`` is added into the
    credit module, and the risk margin falls by ``relief.risk_margin_relief``
    (raising available capital). Compute the relief under the SAME ``regime`` (its
    own ``required_capital`` run must match). Both default off (``None``). Because
    the counterparty-default charge here enters the BSCR through the top-level
    correlation (it diversifies with the life and market modules), the ratio
    benefit can differ from the undiversified ``relief.net_scr_benefit``; the
    insurance / credit modules and the risk margin reported are the post-treaty
    (net) figures. Pricing the relief on a ``catastrophe`` / ``property_codes``
    insurance module is a small approximation -- the relief delta is built on the
    base life module (those default off, the common case). A relief that exceeds
    the module it offsets (almost always a regime / basis mismatch) raises a
    ``ValueError`` rather than flooring to zero and understating the SCR.

    Notes: K-ICS supplies no interest curves (its scenarios are caller-supplied),
    so the net interest component is zero here -- equity and property still apply.
    Credit, FX and concentration risk are calibrated for both regimes (K-ICS
    handbook / Solvency II Delegated Regulation). A non-positive total required
    capital (a risk-free book) gives an unbounded ratio.
    """
    scr = required_capital(model_points, basis, regime=regime,
                           catastrophe=catastrophe, property_codes=property_codes)
    pv = asset_portfolio_value(portfolio, basis.discount_annual)

    cal = _market_cal(regime)
    ni = _module_interest(portfolio, model_points, basis, regime, interest_scenarios)
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
    rm = scr.risk_margin
    if relief is not None:                   # mass-lapse reinsurance into the ratio
        # A relief that exceeds the module it offsets is a caller error -- almost
        # always a relief computed under a different regime / basis than this
        # assessment. Raise loudly rather than silently flooring it to zero (which
        # would understate the SCR with no diagnostic).
        if (relief.insurance_relief > ins + _RELIEF_TOL
                or relief.risk_margin_relief > rm + _RELIEF_TOL):
            raise ValueError(
                "relief exceeds the module it offsets -- the relief was likely "
                "computed under a different regime / basis than this assessment "
                f"(insurance_relief={relief.insurance_relief:.6g} vs "
                f"insurance_scr={ins:.6g}; risk_margin_relief="
                f"{relief.risk_margin_relief:.6g} vs risk_margin={rm:.6g}).")
        ins = max(0.0, ins - relief.insurance_relief)        # diversified lapse relief
        cr = cr + relief.counterparty_default                # reinsurer credit charge
        rm = max(0.0, rm - relief.risk_margin_relief)        # lower technical provision
    ac = available_capital(pv, scr.base_bel, rm)

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
        asset_portfolio_value=pv, bel=scr.base_bel, risk_margin=rm,
        available_capital=ac, insurance_scr=ins, general_insurance_scr=gen,
        net_interest_scr=ni,
        equity_scr=eq, property_scr=pr, fx_scr=fx, concentration_scr=conc,
        market_module_scr=market, credit_scr=cr, operational_scr=op, bscr=bscr,
        basic_required_capital=basic, tax_adjustment=tax_adj,
        total_scr=total, solvency_ratio=ratio)


def vfa_assess_solvency(portfolio: AssetPortfolio, model_points: ModelPoints,
                        basis: Basis, *, regime: RegimeSpec, tax_rate: float = 0.0,
                        tax_recoverability_limit: float | None = None,
                        catastrophe: float = 0.0, general_insurance_scr: float = 0.0,
                        guarantee_equity_shock: float | None = None,
                        interest_shift: float = 0.01) -> SolvencyAssessment:
    """The static t=0 solvency assessment for a variable (VFA) book.

    The VFA counterpart of :func:`assess_solvency`. The liability (insurance) SCR is
    :func:`~fastcashflow.solvency.vfa_required_capital` -- the life sub-risks priced
    on the VFA NET BEL -- and available capital is the assets less the VFA technical
    provision (VFA BEL + risk margin). The market module adds the guarantee's equity
    sensitivity to the asset-side equity SCR: under one equity shock the entity's own
    equity holdings fall AND the guarantee cost rises (the same risk factor), so the
    equity component is ``equity_scr(portfolio) + vfa_equity_scr`` -- added, not
    diversified. ``guarantee_equity_shock`` defaults to the regime's developed
    listed-equity shock (the unit fund's assumed equity stress).

    The net interest module is the VFA :func:`~fastcashflow.solvency.vfa_interest_scr`
    -- a parallel ``+/- interest_shift`` to the underlying-items return (default
    100bp), the guarantee's rate sensitivity. Property / FX / concentration / credit
    / operational follow the asset-side modules (operational on the VFA BEL /
    premium). Both the closed-form variable-annuity and the universal-life paths are
    supported."""
    from fastcashflow._vfa import measure_vfa
    scr = vfa_required_capital(model_points, basis, regime=regime,
                               catastrophe=catastrophe)
    pv = asset_portfolio_value(portfolio, basis.discount_annual)
    ac = available_capital(pv, scr.base_bel, scr.risk_margin)

    cal = _market_cal(regime)
    eq_shock = (guarantee_equity_shock if guarantee_equity_shock is not None
                else cal["equity_shocks"]["developed"])
    ni = vfa_interest_scr(model_points, basis, shift=interest_shift)
    eq = (equity_scr(portfolio, regime)
          + vfa_equity_scr(model_points, basis, equity_shock=eq_shock))
    pr = property_scr(portfolio, regime)
    fx = fx_scr(portfolio, regime, basis.discount_annual)
    conc = concentration_scr(portfolio, regime, basis.discount_annual, total_assets=pv)
    c = np.array([ni, eq, pr, fx, conc], dtype=np.float64)
    R = np.asarray(cal["market_correlation"], dtype=np.float64)
    market = float(np.sqrt(max(0.0, c @ R @ c)))

    ins = scr.insurance_scr
    gen = max(0.0, general_insurance_scr)
    cr = credit_scr(portfolio, regime, basis.discount_annual)
    bscr = aggregate_required_capital(ins, market, cr, regime=regime,
                                      general_insurance=gen)
    op = operational_scr(model_points, basis, regime, bscr=bscr,
                         measure_fn=measure_vfa)
    basic = bscr + op

    tax_adj = 0.0
    if tax_rate > 0.0:
        tax_adj = basic * tax_rate
        if tax_recoverability_limit is not None:
            tax_adj = min(tax_adj, max(0.0, tax_recoverability_limit))
    total = basic - tax_adj

    if total > 0.0:
        ratio = ac / total
    else:
        ratio = float("inf") if ac >= 0.0 else float("-inf")
    return SolvencyAssessment(
        asset_portfolio_value=pv, bel=scr.base_bel, risk_margin=scr.risk_margin,
        available_capital=ac, insurance_scr=ins, general_insurance_scr=gen,
        net_interest_scr=ni,
        equity_scr=eq, property_scr=pr, fx_scr=fx, concentration_scr=conc,
        market_module_scr=market, credit_scr=cr, operational_scr=op, bscr=bscr,
        basic_required_capital=basic, tax_adjustment=tax_adj,
        total_scr=total, solvency_ratio=ratio)


@dataclass(frozen=True, slots=True, eq=False)
class DynamicSolvency:
    """The solvency picture after a coupled rate / dynamic-lapse scenario bites --
    the dynamic asset-liability view layered on the static t=0 assessment.

    ``static`` is the unchanged :func:`assess_solvency` picture (available capital,
    SCR modules, t=0 ratio). ``interaction`` is the coupled-stress
    :class:`InteractionResult` (mark-to-market revaluation plus the forced-sale
    friction) and ``liquidation`` its underlying :class:`LiquidationResult` -- the
    month-by-month surplus / forced-sale trajectory under the stressed run-off.
    ``stressed_available_capital`` is the surplus after the scenario --
    ``static.available_capital - interaction.total_loss`` -- and ``stressed_ratio``
    is that over the (unchanged) required capital ``static.total_scr``.

    This is a SCENARIO OVERLAY on the coverage ratio (how the ratio looks after this
    specific rate / lapse / liquidation scenario), NOT a re-derived regulatory SCR:
    the denominator is still the prescribed 1-in-200 capital, so the static path and
    its FSS-validated numbers are untouched. The scenario answers a reverse-stress
    question the static t=0 ratio cannot -- the asset-liability interaction and the
    liquidity friction it forces."""

    static: SolvencyAssessment
    interaction: InteractionResult
    liquidation: LiquidationResult
    stressed_available_capital: float
    stressed_ratio: float


def dynamic_solvency(portfolio: AssetPortfolio, model_points: ModelPoints,
                     basis: Basis, *, regime: RegimeSpec, shift: float,
                     lapse_sensitivity: float, haircut: float, reinvest_rate=0.0,
                     opening_balance: float = 0.0, **assess_kwargs) -> DynamicSolvency:
    """Layer a coupled rate / dynamic-lapse scenario onto the static solvency ratio.

    Runs :func:`assess_solvency` for the static t=0 picture, then
    :func:`interaction_loss` for the asset-liability interaction the static modules
    miss (the dynamic-lapse-amplified mark-to-market fall plus the forced-sale
    friction). The scenario loss is taken off available capital to give the
    ``stressed_available_capital`` and a ``stressed_ratio`` over the unchanged
    required capital -- the dynamic view feeding the coverage ratio.

    ``shift`` / ``lapse_sensitivity`` / ``haircut`` define the coupled scenario (see
    :func:`interaction_loss`); ``reinvest_rate`` / ``opening_balance`` parameterise
    the liquidation roll-forward. Extra keyword arguments
    (``interest_scenarios``, ``tax_rate``, ``catastrophe``, ...) pass through to
    :func:`assess_solvency`. A zero scenario (``shift = haircut = 0``) leaves the
    ratio at the static value."""
    static = assess_solvency(portfolio, model_points, basis, regime=regime,
                             **assess_kwargs)
    interaction, liq = _interaction(
        portfolio, model_points, basis, shift=shift,
        lapse_sensitivity=lapse_sensitivity, haircut=haircut,
        reinvest_rate=reinvest_rate, opening_balance=opening_balance)
    stressed_ac = static.available_capital - interaction.total_loss
    if static.total_scr > 0.0:
        stressed_ratio = stressed_ac / static.total_scr
    else:
        stressed_ratio = float("inf") if stressed_ac >= 0.0 else float("-inf")
    return DynamicSolvency(static=static, interaction=interaction, liquidation=liq,
                           stressed_available_capital=stressed_ac,
                           stressed_ratio=stressed_ratio)


@dataclass(frozen=True, slots=True)
class VFADynamicSolvency:
    """A market-shock / moneyness-lapse scenario overlaid on a variable book's
    coverage ratio.

    The VFA counterpart of :class:`DynamicSolvency`. ``interaction`` is the VFA
    asset-liability interaction loss (the guarantee-cost revaluation under the
    account-value shock, amplified by the moneyness dynamic lapse, plus the
    forced-sale friction); ``liquidation`` is the underlying forced-sale roll.
    ``stressed_available_capital`` is ``static_available_capital -
    interaction.total_loss`` and ``stressed_ratio`` is that over ``total_scr``.

    Like ``DynamicSolvency`` this is a SCENARIO OVERLAY, not a re-derived SCR. The
    static position (``static_available_capital`` / ``total_scr``) is computed by
    :func:`vfa_assess_solvency` when a ``regime`` is passed (``static`` then carries
    the full :class:`SolvencyAssessment`), or supplied directly by the caller
    (``static`` is then ``None``). This layer adds only the dynamic interaction the
    static modules miss."""

    interaction: InteractionResult
    liquidation: LiquidationResult
    static_available_capital: float
    total_scr: float
    stressed_available_capital: float
    stressed_ratio: float
    static: SolvencyAssessment | None = None


def dynamic_solvency_vfa(portfolio: AssetPortfolio, model_points: ModelPoints,
                         basis: Basis, *, return_shock: float,
                         lapse_sensitivity: float, haircut: float,
                         reinvest_rate=0.0, opening_balance: float = 0.0,
                         regime: RegimeSpec | None = None,
                         static_available_capital: float | None = None,
                         total_scr: float | None = None,
                         **assess_kwargs) -> VFADynamicSolvency:
    """Layer a market-shock / moneyness-lapse scenario onto a variable book's
    coverage ratio.

    Runs :func:`vfa_interaction_loss` for the asset-liability interaction a static
    solvency view misses -- an immediate account-value shock ``return_shock`` moves
    the GMDB / GMAB in-the-money, the moneyness dynamic lapse (``lapse_sensitivity``)
    holds more policies on the guarantee, and the lifted guarantee-excess outflow is
    funded as a forced seller (``haircut``). The total loss is taken off the static
    available capital to give ``stressed_available_capital`` and a ``stressed_ratio``
    over the (unchanged) ``total_scr``.

    The static position comes one of two ways: pass ``regime=`` (with optional
    :func:`vfa_assess_solvency` keyword arguments -- ``tax_rate``,
    ``guarantee_equity_shock``, ...) to COMPUTE it (the result's ``static`` then
    carries the full :class:`SolvencyAssessment`), or supply
    ``static_available_capital`` and ``total_scr`` directly (from your own capital
    model). A null scenario (``return_shock = haircut = 0``, ``lapse_sensitivity =
    0``) leaves the ratio at the static value. Both the closed-form variable-annuity
    and the universal-life paths are supported."""
    static = None
    if regime is not None:
        static = vfa_assess_solvency(portfolio, model_points, basis, regime=regime,
                                     **assess_kwargs)
        static_available_capital = static.available_capital
        total_scr = static.total_scr
    elif static_available_capital is None or total_scr is None:
        raise ValueError(
            "dynamic_solvency_vfa needs either regime= (to compute the static "
            "assessment) or both static_available_capital= and total_scr=.")

    interaction, liq = _interaction_vfa(
        portfolio, model_points, basis, return_shock=return_shock,
        lapse_sensitivity=lapse_sensitivity, haircut=haircut,
        reinvest_rate=reinvest_rate, opening_balance=opening_balance)
    stressed_ac = static_available_capital - interaction.total_loss
    if total_scr > 0.0:
        stressed_ratio = stressed_ac / total_scr
    else:
        stressed_ratio = float("inf") if stressed_ac >= 0.0 else float("-inf")
    return VFADynamicSolvency(
        interaction=interaction, liquidation=liq,
        static_available_capital=static_available_capital, total_scr=total_scr,
        stressed_available_capital=stressed_ac, stressed_ratio=stressed_ratio,
        static=static)


@dataclass(frozen=True, slots=True, eq=False)
class StochasticSolvency:
    """The coverage-ratio DISTRIBUTION of a variable book over fund-return scenarios.

    ``static`` is the t=0 :func:`vfa_assess_solvency` picture. ``available_capital``
    and ``ratio`` are ``(n_scenarios,)`` -- the assets less the per-scenario VFA net
    liability (the realised guarantee cost) and risk margin, over the prescribed
    (unchanged) required capital. Read the distribution with :meth:`mean`,
    :meth:`percentile` and :meth:`cte`. The mean ratio sits below the static ratio by
    the guarantee time-value drag (TVOG / SCR); the lower tail is the stochastic
    guarantee bite the t=0 ratio hides."""

    static: SolvencyAssessment
    available_capital: FloatArray
    ratio: FloatArray

    def mean(self) -> dict[str, float]:
        return {"available_capital": float(self.available_capital.mean()),
                "ratio": float(self.ratio.mean())}

    def percentile(self, q: float) -> dict[str, float]:
        return {"available_capital": float(np.percentile(self.available_capital, q)),
                "ratio": float(np.percentile(self.ratio, q))}

    def cte(self, q: float) -> float:
        """Conditional tail expectation of the coverage ratio at level ``q`` -- the
        mean ratio over the WORST ``q`` percent of scenarios (the lowest ratios).
        ``cte(5)`` is the mean of the lowest 5%, the tail solvency a percentile alone
        does not show."""
        if not 0.0 < q < 100.0:
            raise ValueError("q must be in (0, 100)")
        thresh = np.percentile(self.ratio, q)
        tail = self.ratio[self.ratio <= thresh]
        return float(tail.mean()) if tail.size else float(thresh)


def stochastic_solvency_vfa(portfolio: AssetPortfolio, model_points: ModelPoints,
                            basis: Basis, return_scenarios, *, regime: RegimeSpec,
                            co_moving_assets: bool = False,
                            **assess_kwargs) -> StochasticSolvency:
    """The coverage-ratio distribution of a variable book over fund-return scenarios.

    Runs the static :func:`vfa_assess_solvency` for the prescribed SCR and the asset
    value, then the VFA liability distribution (:func:`fastcashflow.vfa.stochastic`)
    over ``return_scenarios``: available capital per scenario is the assets less the
    realised VFA net liability and risk margin, and the ratio is that over the
    (unchanged) required capital -- the stochastic counterpart of the prescribed-SCR
    ratio. The mean ratio reconciles to the static ratio less the guarantee
    time-value drag.

    ``return_scenarios`` is an ``(n_scenarios, n_time)`` array of monthly
    underlying-items returns OR an :class:`~fastcashflow.EconomicScenarios` (its
    ``returns`` are used). Extra keywords pass through to :func:`vfa_assess_solvency`.
    Both the variable-annuity and universal-life paths are supported.

    ``co_moving_assets`` (default ``False``) makes the entity's general-account
    bonds MOVE WITH the scenario's RATE path. A variable book's account value is
    unit-funded (it tracks the fund, not the entity), so the entity holds bonds for
    the guarantee, and those co-move with INTEREST RATES, not the fund return -- a
    different axis from ``return_scenarios``. So ``co_moving_assets=True`` needs an
    :class:`~fastcashflow.EconomicScenarios` (the fund ``returns`` drive the
    guarantee liability, the joint ``rates`` revalue the bonds, keeping their
    correlation); a raw returns array carries no rate path and is rejected. Off (the
    default) holds the asset value at its t=0 base-curve level. The prescribed
    required capital stays at its t=0 value either way (a scenario overlay on the
    ratio)."""
    from fastcashflow._vfa import measure_vfa_stochastic
    rs = getattr(return_scenarios, "returns", return_scenarios)
    static = vfa_assess_solvency(portfolio, model_points, basis, regime=regime,
                                 **assess_kwargs)
    dist = measure_vfa_stochastic(model_points, basis, rs)
    if co_moving_assets:
        rates = getattr(return_scenarios, "rates", None)
        if rates is None:
            raise ValueError(
                "co_moving_assets=True needs an EconomicScenarios -- its .rates "
                "revalue the entity's bonds (which co-move with interest rates, not "
                "the fund return); a raw returns array carries no rate path.")
        asset_value = asset_value_by_scenario(portfolio, rates)
    else:
        asset_value = static.asset_portfolio_value
    ac = asset_value - (dist.bel + static.risk_margin)
    if static.total_scr > 0.0:
        ratio = ac / static.total_scr
    else:
        ratio = np.where(ac >= 0.0, np.inf, -np.inf)
    return StochasticSolvency(static=static, available_capital=ac, ratio=ratio)


def stochastic_solvency_gmm(portfolio: AssetPortfolio, model_points: ModelPoints,
                            basis: Basis, rate_scenarios, *, regime: RegimeSpec,
                            co_moving_assets: bool = False,
                            **assess_kwargs) -> StochasticSolvency:
    """The coverage-ratio distribution of a book over discount-rate scenarios.

    The GMM counterpart of :func:`stochastic_solvency_vfa`. Runs the static
    :func:`assess_solvency` for the prescribed SCR and the asset value, then the
    GMM liability distribution (:func:`fastcashflow.gmm.stochastic`) over
    ``rate_scenarios``: available capital per scenario is the assets less that
    scenario's BEL and the risk margin, and the ratio is that over the (unchanged)
    required capital. A scenario equal to the basis's own discount curve
    reproduces the static ratio exactly.

    ``rate_scenarios`` is a 1-D ``(n_scenarios,)`` array of flat annual rates, a
    2-D ``(n_scenarios, n_time)`` array of rate curves, OR an
    :class:`~fastcashflow.EconomicScenarios` (its ``rates`` are used). Extra
    keywords pass through to :func:`assess_solvency`.

    ``co_moving_assets`` (default ``False``) makes the asset value MOVE WITH each
    rate scenario: the bond portfolio is revalued on the same curve the liability is
    discounted at (:func:`~fastcashflow.asset_value_by_scenario`), so a rate fall
    lifts BOTH the BEL and the bonds and the ratio reflects the asset-liability
    DURATION GAP, not just the liability move. Off (the default) holds the asset
    value at its t=0 base-curve level -- the liability-only distribution. The
    prescribed required capital (the denominator) is held at its t=0 value either
    way (a scenario overlay on the ratio, not a re-derived SCR)."""
    from fastcashflow.stochastic import measure_stochastic
    rs = getattr(rate_scenarios, "rates", rate_scenarios)
    static = assess_solvency(portfolio, model_points, basis, regime=regime,
                             **assess_kwargs)
    dist = measure_stochastic(model_points, basis, rs)
    asset_value = (asset_value_by_scenario(portfolio, rs) if co_moving_assets
                   else static.asset_portfolio_value)
    ac = asset_value - (dist.bel + static.risk_margin)
    if static.total_scr > 0.0:
        ratio = ac / static.total_scr
    else:
        ratio = np.where(ac >= 0.0, np.inf, -np.inf)
    return StochasticSolvency(static=static, available_capital=ac, ratio=ratio)


__all__ = [
    "InteractionResult", "interaction_loss", "vfa_interaction_loss",
    "net_interest_scr", "net_interest_kics_scr",
    "equity_scr", "property_scr", "fx_scr", "concentration_scr",
    "market_module_scr", "credit_scr", "operational_scr",
    "aggregate_required_capital",
    "SolvencyAssessment", "assess_solvency", "vfa_assess_solvency",
    "DynamicSolvency", "dynamic_solvency",
    "VFADynamicSolvency", "dynamic_solvency_vfa",
    "StochasticSolvency", "stochastic_solvency_vfa", "stochastic_solvency_gmm",
]
