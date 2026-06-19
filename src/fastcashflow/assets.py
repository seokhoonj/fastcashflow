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


__all__ = [
    "Equity", "Property", "Cash", "AssetPortfolio",
    "holding_value", "portfolio_value", "available_capital", "net_interest_scr",
]
