"""VFA solvency -- the variable-book required capital, SCR modules, and dynamic /
stochastic solvency, exposed prefix-free under ``fcf.vfa.*``.

Apart from the generic ``_engine`` because VFA risk is structurally
distinct: the GMDB / GMAB guarantee is an account-value market risk with
moneyness-driven dynamic lapse, not a ``measure_fn`` variant of the GMM path.
Only ``vfa_required_capital`` is a thin wrapper over the generic engine; the rest
are genuine VFA computations. Generic primitives import one-way from ``_engine``
/ ``_assessment`` / ``assets``; ``measure_vfa`` and the merged
``DynamicAssessment`` import lazily in-body to avoid a runtime cycle.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from fastcashflow.basis import Basis
from fastcashflow.model_points import ModelPoints
from fastcashflow.assets import (
    AssetPortfolio, available_capital, asset_portfolio_value,
    asset_value_by_scenario, LiquidationResult, liquidate, vfa_cashflow_gap,
)
from fastcashflow.solvency._engine import (
    required_capital, RegimeSpec, KICSInterest, SCRResult,
)
from fastcashflow.solvency._assessment import (
    Assessment, InteractionResult, StochasticAssessment, DynamicAssessment,
    equity_scr, property_scr, fx_scr, concentration_scr, credit_scr,
    operational_scr, basic_scr,
    _market_cal, _module_interest,
)

__all__ = [
    "vfa_required_capital", "vfa_equity_scr", "vfa_interest_scr",
    "vfa_interaction_loss", "assess_vfa",
    "assess_dynamic_vfa", "assess_stochastic_vfa",
]

def vfa_required_capital(
    model_points: ModelPoints, basis: Basis, *, regime: RegimeSpec,
    catastrophe: float = 0.0, interest_scenarios: KICSInterest | None = None,
) -> SCRResult:
    """Required capital for a variable (VFA) book's LIFE sub-risks.

    The VFA counterpart of :func:`required_capital`: re-runs the regime's life
    sub-risks (mortality / longevity / lapse / expense / catastrophe / ...) on the
    VFA NET BEL (guarantee-excess + expense - fee) through
    :func:`fastcashflow.vfa.measure`, so the stresses bite the guarantee cost the
    way they do for a variable annuity -- a lapse-DOWN holds more policies on a
    valuable guarantee (raising the cost), an expense shock lifts the maintenance
    leg. ``base_bel`` is therefore the VFA net BEL, not a GMM measure.

    This is the LIFE module only. The dominant variable-annuity risk -- the equity
    sensitivity of the guarantee (an account-value fall lifting the GMDB / GMAB
    cost) -- is MARKET risk, added by the VFA solvency assembly, not here.
    ``property_codes`` are not accepted (a variable book carries no
    long-term-property coverage). Closed-form variable-annuity path only.

    NOTE on mass lapse: unlike the GMM path (where a surrender_value_curve adds the
    t=0 surrender outflow), a variable book's surrender value IS the account value,
    which is UNIT-FUNDED -- the unit fund pays it, not the entity's general account.
    So no t=0 add-back is correct here: the mass-lapse capital is the re-measured
    change in the entity NET BEL (the guarantee-excess + expense - fee leg), driven
    by the lost variable-fee income and the changed guarantee cost, not by an
    account-value outflow."""
    from fastcashflow._vfa import measure_vfa
    return required_capital(
        model_points, basis, regime=regime, catastrophe=catastrophe,
        interest_scenarios=interest_scenarios, measure_fn=measure_vfa)

def vfa_equity_scr(model_points: ModelPoints, basis: Basis, *,
                   equity_shock: float) -> float:
    """The equity capital of a variable (VFA) book's net BEL.

    An immediate equity-type shock drops the account value by ``equity_shock`` (a
    fraction, ``0.39`` = -39%). The lower account value moves the VFA net BEL on two
    legs, BOTH raising the liability: the GMDB / GMAB goes in-the-money so the
    GUARANTEE cost rises (the dominant, headline driver), and the variable FEE
    income (skimmed off the account value) falls. Returns ``max(0, Delta BEL)``, the
    instantaneous 1-in-200 capital for the book's equity sensitivity.

    This is the DOMINANT variable-annuity market risk and the piece the asset-side
    :func:`fastcashflow.solvency.equity_scr` (which shocks only the
    entity's own equity holdings) does not see. Measured on the STATIC lapse: the
    instantaneous stress capital, distinct from the behavioural moneyness dynamic
    lapse, which is the dynamic scenario overlay
    (:func:`fastcashflow.vfa.assess_dynamic`).
    Closed-form variable-annuity path only."""
    from fastcashflow._vfa import measure_vfa
    base = float(measure_vfa(model_points, basis, full=False).bel.sum())
    av = np.asarray(model_points.account_value, dtype=np.float64)
    mp_s = replace(model_points, account_value=av * (1.0 - equity_shock))
    stressed = float(measure_vfa(mp_s, basis, full=False).bel.sum())
    return max(0.0, stressed - base)

def vfa_interest_scr(model_points: ModelPoints, basis: Basis, *,
                     shift: float = 0.01) -> float:
    """The interest capital of a variable (VFA) book's net BEL.

    Worst of re-measuring the VFA net BEL under a parallel ``+/- shift`` to the
    underlying-items return -- the VFA basis rate that BOTH discounts the liability
    and grows the account value. A return FALL lowers the account growth, pushing
    the GMDB / GMAB in-the-money (higher guarantee cost): the binding direction.
    Returns ``max(0, worst Delta BEL)``, the instantaneous interest-rate capital.

    v1: a parallel shift to ``investment_return`` (the VFA's single rate driving
    growth and discount together), the transparent proxy for the guarantee's rate
    sensitivity. Distinct from the spot equity LEVEL shock (:func:`vfa_equity_scr`):
    a rate-assumption move versus a one-time market fall, the two market sub-risks a
    variable annuity carries. A regime-curve-calibrated VFA interest stress (a
    maturity-relative curve, not a flat parallel move) is future work. Closed-form
    variable-annuity path only."""
    from fastcashflow._vfa import measure_vfa
    base = float(measure_vfa(model_points, basis, full=False).bel.sum())
    r = basis.investment_return
    up = float(measure_vfa(
        model_points, replace(basis, investment_return=r + shift), full=False).bel.sum())
    dn = float(measure_vfa(
        model_points, replace(basis, investment_return=r - shift), full=False).bel.sum())
    return max(0.0, up - base, dn - base)

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

def _interaction_vfa(portfolio: AssetPortfolio, model_points: ModelPoints,
                     basis: Basis, *, return_shock, lapse_sensitivity, haircut,
                     reinvest_rate, opening_balance):
    """The VFA interaction loss AND the forced-sale roll-forward, so
    :func:`assess_dynamic` does not re-measure the stressed book."""
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

def assess_vfa(portfolio: AssetPortfolio, model_points: ModelPoints,
                        basis: Basis, *, regime: RegimeSpec, tax_rate: float = 0.0,
                        tax_recoverability_limit: float | None = None,
                        catastrophe: float = 0.0, general_insurance_scr: float = 0.0,
                        guarantee_equity_shock: float | None = None,
                        interest_shift: float = 0.01) -> Assessment:
    """The static t=0 solvency assessment for a variable (VFA) book.

    The VFA counterpart of :func:`assess`. The liability (insurance) SCR is
    :func:`~fastcashflow.vfa.required_capital` -- the life sub-risks priced
    on the VFA NET BEL -- and available capital is the assets less the VFA technical
    provision (VFA BEL + risk margin). The market module adds the guarantee's equity
    sensitivity to the asset-side equity SCR: under one equity shock the entity's own
    equity holdings fall AND the guarantee cost rises (the same risk factor), so the
    equity component is ``equity_scr(portfolio) + vfa_equity_scr`` -- added, not
    diversified. ``guarantee_equity_shock`` defaults to the regime's developed
    listed-equity shock (the unit fund's assumed equity stress).

    The net interest module is the VFA :func:`~fastcashflow.vfa.interest_scr`
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
    bscr = basic_scr(ins, market, cr, regime=regime,
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
    return Assessment(
        asset_portfolio_value=pv, bel=scr.base_bel, risk_margin=scr.risk_margin,
        available_capital=ac, insurance_scr=ins, general_insurance_scr=gen,
        net_interest_scr=ni,
        equity_scr=eq, property_scr=pr, fx_scr=fx, concentration_scr=conc,
        market_scr=market, credit_scr=cr, operational_scr=op, basic_scr=bscr,
        basic_required_capital=basic, tax_adjustment=tax_adj,
        total_scr=total, ratio=ratio)

def assess_dynamic_vfa(portfolio: AssetPortfolio, model_points: ModelPoints,
                         basis: Basis, *, return_shock: float,
                         lapse_sensitivity: float, haircut: float,
                         reinvest_rate=0.0, opening_balance: float = 0.0,
                         regime: RegimeSpec | None = None,
                         static_available_capital: float | None = None,
                         total_scr: float | None = None,
                         **assess_kwargs) -> "DynamicAssessment":
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
    :func:`assess_vfa` keyword arguments -- ``tax_rate``,
    ``guarantee_equity_shock``, ...) to COMPUTE it (the result's ``static`` then
    carries the full :class:`Assessment`), or supply
    ``static_available_capital`` and ``total_scr`` directly (from your own capital
    model). A null scenario (``return_shock = haircut = 0``, ``lapse_sensitivity =
    0``) leaves the ratio at the static value. Both the closed-form variable-annuity
    and the universal-life paths are supported."""
    static = None
    if regime is not None:
        static = assess_vfa(portfolio, model_points, basis, regime=regime,
                                     **assess_kwargs)
        static_available_capital = static.available_capital
        total_scr = static.total_scr
    elif static_available_capital is None or total_scr is None:
        raise ValueError(
            "assess_dynamic_vfa needs either regime= (to compute the static "
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
    return DynamicAssessment(
        interaction=interaction, liquidation=liq,
        static_available_capital=static_available_capital, total_scr=total_scr,
        stressed_available_capital=stressed_ac, stressed_ratio=stressed_ratio,
        static=static)

def assess_stochastic_vfa(portfolio: AssetPortfolio, model_points: ModelPoints,
                            basis: Basis, return_scenarios, *, regime: RegimeSpec,
                            co_moving_assets: bool = False,
                            **assess_kwargs) -> StochasticAssessment:
    """The coverage-ratio distribution of a variable book over fund-return scenarios.

    Runs the static :func:`assess_vfa` for the prescribed SCR and the asset
    value, then the VFA liability distribution (:func:`fastcashflow.vfa.stochastic`)
    over ``return_scenarios``: available capital per scenario is the assets less the
    realised VFA net liability and risk margin, and the ratio is that over the
    (unchanged) required capital -- the stochastic counterpart of the prescribed-SCR
    ratio. The mean ratio reconciles to the static ratio less the guarantee
    time-value drag.

    ``return_scenarios`` is an ``(n_scenarios, n_time)`` array of monthly
    underlying-items returns OR an :class:`~fastcashflow.esg.EconomicScenarios` (its
    ``returns`` are used). Extra keywords pass through to :func:`assess_vfa`.
    Both the variable-annuity and universal-life paths are supported.

    ``co_moving_assets`` (default ``False``) makes the entity's general-account
    bonds MOVE WITH the scenario's RATE path. A variable book's account value is
    unit-funded (it tracks the fund, not the entity), so the entity holds bonds for
    the guarantee, and those co-move with INTEREST RATES, not the fund return -- a
    different axis from ``return_scenarios``. So ``co_moving_assets=True`` needs an
    :class:`~fastcashflow.esg.EconomicScenarios` (the fund ``returns`` drive the
    guarantee liability, the joint ``rates`` revalue the bonds, keeping their
    correlation); a raw returns array carries no rate path and is rejected. Off (the
    default) holds the asset value at its t=0 base-curve level. The prescribed
    required capital stays at its t=0 value either way (a scenario overlay on the
    ratio)."""
    from fastcashflow._vfa import measure_vfa_stochastic
    rs = getattr(return_scenarios, "returns", return_scenarios)
    static = assess_vfa(portfolio, model_points, basis, regime=regime,
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
    return StochasticAssessment(static=static, available_capital=ac, ratio=ratio)
