"""Solvency assessment -- asset-liability interaction and net interest SCR.

The base sub-risk module: the interaction loss between the asset and liability
sides under an interest shock (assets liquidated to meet the liability cash-flow
gap) and the net interest SCR. The market aggregation (:mod:`._market`) and the
assembly (:mod:`._assemble`) build on these; this module imports neither, so it
is the package's base layer.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastcashflow.basis import Basis
from fastcashflow._measurement.gmm import measure
from fastcashflow.model_points import ModelPoints
from fastcashflow.solvency._engine import (
    KICSInterest, interest_with_dynamic_lapse,
)
from fastcashflow.assets import (
    AssetPortfolio, asset_portfolio_value, cashflow_gap, liquidate,
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
    that needs the liquidity trajectory (e.g. :func:`assess_dynamic`) does not
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
