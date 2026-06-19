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
    """An equity holding carried at a given market value (asset-positive). v1
    applies no equity shock, so it adds to available capital but not to the SCR."""

    market_value: float


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


@dataclass(frozen=True, slots=True, eq=False)
class SolvencyAssessment:
    """The asset-inclusive solvency picture at t=0 -- the full ratio and its parts.

    ``available_capital`` is ``portfolio_value - (bel + risk_margin)``;
    ``total_scr`` is ``insurance_scr + net_interest_scr`` (the asset-aware net
    interest replacing the liability-only ``liability_interest_capital``);
    ``solvency_ratio`` is ``available_capital / total_scr``."""

    portfolio_value: float
    bel: float
    risk_margin: float
    available_capital: float
    insurance_scr: float
    net_interest_scr: float
    liability_interest_capital: float
    total_scr: float
    solvency_ratio: float


def assess_solvency(portfolio: AssetPortfolio, model_points: ModelPoints,
                    basis: Basis, *, regime: RegimeSpec) -> SolvencyAssessment:
    """Assemble the t=0 solvency ratio from the assets and the liability SCR.

    Runs :func:`~fastcashflow.required_capital` for the liability-side SCR, values
    the portfolio, forms available capital (assets less the technical provision),
    and replaces the liability-only interest capital with the NET interest SCR
    (assets and liabilities re-priced together) when the regime carries interest
    curves -- else it keeps the regime's liability interest figure (e.g. K-ICS,
    whose curve scenarios are caller-supplied).

    v1 LIMITATION: equity and property holdings raise available capital but carry
    NO asset-side market-risk SCR yet (the equity / property shock sub-modules are
    a follow-up). A book with material equity / property therefore has an
    understated ``total_scr`` and an OVERSTATED ``solvency_ratio`` -- read it as an
    upper bound until that follow-up lands.
    """
    scr = required_capital(model_points, basis, regime=regime)
    pv = portfolio_value(portfolio, basis.discount_annual)
    ac = available_capital(pv, scr.base_bel, scr.risk_margin)
    if regime.interest_curves is not None:
        ni = net_interest_scr(portfolio, model_points, basis,
                              interest_curves=regime.interest_curves)
    else:
        ni = scr.interest_capital
    total = scr.insurance_scr + ni
    # A non-positive required capital (a risk-free / fully-immunised book) makes
    # the ratio unbounded -- avoid the divide-by-zero and signal it as infinite.
    if total > 0.0:
        ratio = ac / total
    else:
        ratio = float("inf") if ac >= 0.0 else float("-inf")
    return SolvencyAssessment(
        portfolio_value=pv, bel=scr.base_bel, risk_margin=scr.risk_margin,
        available_capital=ac, insurance_scr=scr.insurance_scr, net_interest_scr=ni,
        liability_interest_capital=scr.interest_capital, total_scr=total,
        solvency_ratio=ratio)


__all__ = [
    "Equity", "Property", "Cash", "AssetPortfolio", "SolvencyAssessment",
    "holding_value", "portfolio_value", "available_capital", "net_interest_scr",
    "assess_solvency",
]
