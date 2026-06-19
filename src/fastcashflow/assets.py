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

v1 limitation (documented on :func:`assess_solvency`): equity and property
holdings are carried at their market value -- they raise available capital but do
NOT yet contribute an asset-side market-risk SCR (the equity / property shock
sub-modules are a follow-up). A book with material equity / property therefore has
an understated SCR, so its ratio is an upper bound until that follow-up lands.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow.alm import Bond, bond_value
from fastcashflow.basis import Basis
from fastcashflow.engine import measure
from fastcashflow.model_points import ModelPoints
from fastcashflow.solvency import RegimeSpec, required_capital


@dataclass(frozen=True, slots=True)
class Equity:
    """An equity holding carried at a given market value (asset-positive).

    ``risk_type`` selects the market-risk shock (``"developed"`` or ``"emerging"``
    market listed equity); the shock magnitude is the regime's calibration."""

    market_value: float
    risk_type: str = "developed"


@dataclass(frozen=True, slots=True)
class Property:
    """A property holding carried at a given market value. As for :class:`Equity`,
    v1 carries the value without a property-shock SCR."""

    market_value: float


@dataclass(frozen=True, slots=True)
class Cash:
    """A cash holding, carried at face (curve-insensitive)."""

    market_value: float


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
_MARKET_CORRELATION = np.array([
    [1.00, 0.25, 0.25],
    [0.25, 1.00, 0.25],
    [0.25, 0.25, 1.00],
])

_MARKET_CALIBRATION = {
    "K-ICS": {
        "equity_shocks": {"developed": 0.35, "emerging": 0.48},
        "property_shock": 0.25,
        "market_correlation": _MARKET_CORRELATION,
        "insurance_market_corr": 0.25,     # table 3 (life-long-term <-> market)
    },
    "Solvency II": {
        "equity_shocks": {"developed": 0.35, "emerging": 0.48},
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
    """The equity market-risk SCR -- each equity holding's market value times the
    regime's price-fall shock for its ``risk_type``. Raises on an unknown type."""
    shocks = _market_cal(regime)["equity_shocks"]
    total = 0.0
    for h in portfolio.holdings:
        if isinstance(h, Equity):
            if h.risk_type not in shocks:
                raise ValueError(
                    f"unknown equity risk_type {h.risk_type!r} for regime "
                    f"{regime.name!r}; known: {sorted(shocks)}")
            total += h.market_value * shocks[h.risk_type]
    return max(0.0, total)         # a capital requirement is non-negative


def property_scr(portfolio: AssetPortfolio, regime) -> float:
    """The property market-risk SCR -- property market value times the regime's
    price-fall shock."""
    shock = _market_cal(regime)["property_shock"]
    total = sum(h.market_value * shock
                for h in portfolio.holdings if isinstance(h, Property))
    return max(0.0, float(total))


def market_module_scr(portfolio: AssetPortfolio, model_points: ModelPoints,
                      basis: Basis, *, regime) -> float:
    """The market-risk module SCR -- the interest (net of liabilities), equity and
    property sub-risks aggregated through the regime's market correlation matrix
    (``sqrt(c^T R c)``). Interest is the net :func:`net_interest_scr` (zero when
    the regime supplies no interest curves, e.g. K-ICS)."""
    cal = _market_cal(regime)
    interest = (net_interest_scr(portfolio, model_points, basis,
                                 interest_curves=regime.interest_curves)
                if regime.interest_curves is not None else 0.0)
    c = np.array([interest, equity_scr(portfolio, regime),
                  property_scr(portfolio, regime)], dtype=np.float64)
    R = np.asarray(cal["market_correlation"], dtype=np.float64)
    return float(np.sqrt(c @ R @ c))


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


@dataclass(frozen=True, slots=True, eq=False)
class SolvencyAssessment:
    """The asset-inclusive solvency picture at t=0 -- the full ratio and its parts.

    ``available_capital`` is ``portfolio_value - (bel + risk_margin)``. The market
    module aggregates the ``net_interest_scr`` (assets and liabilities), the
    ``equity_scr`` and the ``property_scr`` through the market correlation; the
    ``total_scr`` (BSCR) aggregates the ``insurance_scr`` and the
    ``market_module_scr`` at the top level. ``solvency_ratio`` is
    ``available_capital / total_scr``."""

    portfolio_value: float
    bel: float
    risk_margin: float
    available_capital: float
    insurance_scr: float
    net_interest_scr: float
    equity_scr: float
    property_scr: float
    market_module_scr: float
    total_scr: float
    solvency_ratio: float


def assess_solvency(portfolio: AssetPortfolio, model_points: ModelPoints,
                    basis: Basis, *, regime: RegimeSpec) -> SolvencyAssessment:
    """Assemble the t=0 solvency ratio from the assets and the liability SCR.

    Runs :func:`~fastcashflow.required_capital` for the liability (insurance) SCR,
    values the portfolio, forms available capital (assets less the technical
    provision), and builds the market-risk module (net interest, equity, property)
    aggregated through the market correlation. The total SCR (BSCR) aggregates the
    insurance and market modules at the top level: K-ICS uses the life-vs-market
    correlation (0.25); Solvency II's top-level inter-module matrix is not
    extracted here, so it falls back to a simple sum (no diversification credit --
    conservative). The ratio is available capital over the BSCR.

    Notes: K-ICS supplies no interest curves (its scenarios are caller-supplied),
    so the net interest component is zero here -- equity and property still apply.
    A non-positive BSCR (a risk-free book) gives an unbounded ratio.
    """
    scr = required_capital(model_points, basis, regime=regime)
    pv = portfolio_value(portfolio, basis.discount_annual)
    ac = available_capital(pv, scr.base_bel, scr.risk_margin)

    cal = _market_cal(regime)
    ni = (net_interest_scr(portfolio, model_points, basis,
                           interest_curves=regime.interest_curves)
          if regime.interest_curves is not None else 0.0)
    eq = equity_scr(portfolio, regime)
    pr = property_scr(portfolio, regime)
    c = np.array([ni, eq, pr], dtype=np.float64)
    R = np.asarray(cal["market_correlation"], dtype=np.float64)
    market = float(np.sqrt(c @ R @ c))

    ins = scr.insurance_scr
    imc = cal["insurance_market_corr"]
    if imc is None:                         # SII: top-level matrix not extracted
        bscr = ins + market
    else:                                   # K-ICS: life <-> market correlation
        bscr = float(np.sqrt(ins * ins + market * market + 2.0 * imc * ins * market))

    if bscr > 0.0:
        ratio = ac / bscr
    else:
        ratio = float("inf") if ac >= 0.0 else float("-inf")
    return SolvencyAssessment(
        portfolio_value=pv, bel=scr.base_bel, risk_margin=scr.risk_margin,
        available_capital=ac, insurance_scr=ins, net_interest_scr=ni,
        equity_scr=eq, property_scr=pr, market_module_scr=market,
        total_scr=bscr, solvency_ratio=ratio)


__all__ = [
    "Equity", "Property", "Cash", "AssetPortfolio", "SolvencyAssessment",
    "holding_value", "portfolio_value", "available_capital", "net_interest_scr",
    "equity_scr", "property_scr", "market_module_scr", "operational_scr",
    "assess_solvency",
]
