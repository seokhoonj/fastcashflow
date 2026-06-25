"""Solvency assessment -- the assembly (assess / dynamic / stochastic).

Ties the asset-side sub-risk SCRs, the liability SCR, the asset-liability
interaction and available capital into the coverage ratio: :func:`assess`
(t=0), :func:`assess_dynamic` (scenario overlay) and :func:`assess_stochastic`
(distribution), with their result types. The top layer of the package -- it
imports every sub-risk module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                       # annotation only -- avoids a runtime cycle
    from fastcashflow._mass_lapse_reinsurance import CedantSolvencyRelief

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.model_points import ModelPoints
from fastcashflow.solvency._engine import (
    RegimeSpec, required_capital, KICSInterest,
)
from fastcashflow.assets import (
    AssetPortfolio, asset_portfolio_value, available_capital,
    asset_value_by_scenario, LiquidationResult,
)
from fastcashflow.solvency._assessment._interaction import (
    InteractionResult, _interaction, net_interest_scr,
    _module_interest,
)
from fastcashflow.solvency._assessment._market import (
    _market_cal, market_scr, equity_scr, property_scr, fx_scr,
    concentration_scr,
)
from fastcashflow.solvency._assessment._credit import credit_scr
from fastcashflow.solvency._assessment._operational import operational_scr
from fastcashflow.solvency._assessment._toplevel import basic_scr, _pct


@dataclass(frozen=True, slots=True, eq=False)
class Assessment:
    """The asset-inclusive solvency picture at t=0 -- the full ratio and its parts.

    ``available_capital`` is ``asset_portfolio_value - (bel + risk_margin)``. The market
    module aggregates the ``net_interest_scr`` (assets and liabilities), the
    ``equity_scr``, the ``property_scr``, the ``fx_scr`` and the
    ``concentration_scr`` through the market correlation; the
    ``basic_scr`` (basic SCR) aggregates the ``insurance_scr``, the (optional)
    ``general_insurance_scr``, the ``market_scr`` and the ``credit_scr`` at
    the top level; ``basic_required_capital`` adds the
    ``operational_scr`` on top of the BSCR. ``total_scr`` (the ratio denominator)
    subtracts the ``tax_adjustment`` (loss-absorbing capacity of deferred taxes)
    from the basic required capital. ``ratio`` is
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
    market_scr: float
    credit_scr: float
    operational_scr: float
    basic_scr: float
    basic_required_capital: float
    tax_adjustment: float
    total_scr: float
    ratio: float

    def __repr__(self) -> str:
        return (f"<solvency.Assessment: AC={self.available_capital:,.0f}, "
                f"SCR={self.total_scr:,.0f}, ratio={_pct(self.ratio)}>")


_RELIEF_TOL = 1e-9   # float-noise band before a relief is deemed to exceed its module


def assess(portfolio: AssetPortfolio, model_points: ModelPoints,
                    basis: Basis, *, regime: RegimeSpec, tax_rate: float = 0.0,
                    tax_recoverability_limit: float | None = None,
                    catastrophe: float = 0.0, property_codes=(),
                    general_insurance_scr: float = 0.0,
                    interest_scenarios: KICSInterest | None = None,
                    relief: "CedantSolvencyRelief | None" = None) -> Assessment:
    """Assemble the t=0 solvency ratio from the assets and the liability SCR.

    Runs :func:`~fastcashflow.gmm.required_capital` for the liability (insurance) SCR,
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
    :func:`~fastcashflow.solvency.kics_catastrophe`) and ``property_codes`` (the long-term
    property / other coverages, a +16% rate shock) fold into the insurance module
    (table-6 correlation); both default to off. ``general_insurance_scr`` (a
    caller-supplied P&C amount for a life + general book) enters the BSCR as a
    fourth top-level module (table 3: life-vs-general 0, else 0.25); the life-only
    engine leaves it at 0.

    ``relief`` (a :class:`~fastcashflow.reinsurance.mass_lapse.CedantSolvencyRelief`
    from :func:`~fastcashflow.reinsurance.mass_lapse.cedant_solvency_relief`) folds
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

    bscr = basic_scr(ins, market, cr, regime=regime,
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
    return Assessment(
        asset_portfolio_value=pv, bel=scr.base_bel, risk_margin=rm,
        available_capital=ac, insurance_scr=ins, general_insurance_scr=gen,
        net_interest_scr=ni,
        equity_scr=eq, property_scr=pr, fx_scr=fx, concentration_scr=conc,
        market_scr=market, credit_scr=cr, operational_scr=op, basic_scr=bscr,
        basic_required_capital=basic, tax_adjustment=tax_adj,
        total_scr=total, ratio=ratio)


@dataclass(frozen=True, slots=True, eq=False)
class DynamicAssessment:
    """The solvency picture after a coupled rate / dynamic-lapse scenario bites --
    the dynamic asset-liability view layered on the static t=0 assessment.

    ``static`` is the unchanged :func:`assess` picture (available capital,
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

    interaction: InteractionResult
    liquidation: LiquidationResult
    static_available_capital: float
    total_scr: float
    stressed_available_capital: float
    stressed_ratio: float
    static: Assessment | None = None

    def __repr__(self) -> str:
        return (f"<solvency.DynamicAssessment: "
                f"stressed_AC={self.stressed_available_capital:,.0f}, "
                f"SCR={self.total_scr:,.0f}, "
                f"stressed_ratio={_pct(self.stressed_ratio)}>")


def assess_dynamic(portfolio: AssetPortfolio, model_points: ModelPoints,
                     basis: Basis, *, regime: RegimeSpec, shift: float,
                     lapse_sensitivity: float, haircut: float, reinvest_rate=0.0,
                     opening_balance: float = 0.0, **assess_kwargs) -> DynamicAssessment:
    """Layer a coupled rate / dynamic-lapse scenario onto the static solvency ratio.

    Runs :func:`assess` for the static t=0 picture, then
    :func:`interaction_loss` for the asset-liability interaction the static modules
    miss (the dynamic-lapse-amplified mark-to-market fall plus the forced-sale
    friction). The scenario loss is taken off available capital to give the
    ``stressed_available_capital`` and a ``stressed_ratio`` over the unchanged
    required capital -- the dynamic view feeding the coverage ratio.

    ``shift`` / ``lapse_sensitivity`` / ``haircut`` define the coupled scenario (see
    :func:`interaction_loss`); ``reinvest_rate`` / ``opening_balance`` parameterise
    the liquidation roll-forward. Extra keyword arguments
    (``interest_scenarios``, ``tax_rate``, ``catastrophe``, ...) pass through to
    :func:`assess`. A zero scenario (``shift = haircut = 0``) leaves the
    ratio at the static value."""
    static = assess(portfolio, model_points, basis, regime=regime,
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
    return DynamicAssessment(static=static, interaction=interaction, liquidation=liq,
                           static_available_capital=static.available_capital,
                           total_scr=static.total_scr,
                           stressed_available_capital=stressed_ac,
                           stressed_ratio=stressed_ratio)


@dataclass(frozen=True, slots=True, eq=False)
class StochasticAssessment:
    """The coverage-ratio DISTRIBUTION of a variable book over fund-return scenarios.

    ``static`` is the t=0 :func:`assess_vfa` picture. ``available_capital``
    and ``ratio`` are ``(n_scenarios,)`` -- the assets less the per-scenario VFA net
    liability (the realised guarantee cost) and risk margin, over the prescribed
    (unchanged) required capital. Read the distribution with :meth:`mean`,
    :meth:`percentile` and :meth:`cte`. The mean ratio sits below the static ratio by
    the guarantee time-value drag (TVOG / SCR); the lower tail is the stochastic
    guarantee bite the t=0 ratio hides."""

    static: Assessment
    available_capital: FloatArray
    ratio: FloatArray

    def __repr__(self) -> str:
        n = self.ratio.size
        finite = self.ratio[np.isfinite(self.ratio)]
        mr = _pct(float(finite.mean())) if finite.size else "n/a"
        return f"<solvency.StochasticAssessment: n_scen={n}, mean_ratio={mr}>"

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


def assess_stochastic(portfolio: AssetPortfolio, model_points: ModelPoints,
                            basis: Basis, rate_scenarios, *, regime: RegimeSpec,
                            co_moving_assets: bool = False,
                            **assess_kwargs) -> StochasticAssessment:
    """The coverage-ratio distribution of a book over discount-rate scenarios.

    The GMM counterpart of :func:`assess_stochastic_vfa`. Runs the static
    :func:`assess` for the prescribed SCR and the asset value, then the
    GMM liability distribution (:func:`fastcashflow.gmm.stochastic`) over
    ``rate_scenarios``: available capital per scenario is the assets less that
    scenario's BEL and the risk margin, and the ratio is that over the (unchanged)
    required capital. A scenario equal to the basis's own discount curve
    reproduces the static ratio exactly.

    ``rate_scenarios`` is a 1-D ``(n_scenarios,)`` array of flat annual rates, a
    2-D ``(n_scenarios, n_time)`` array of rate curves, OR an
    :class:`~fastcashflow.esg.EconomicScenarios` (its ``rates`` are used). Extra
    keywords pass through to :func:`assess`.

    ``co_moving_assets`` (default ``False``) makes the asset value MOVE WITH each
    rate scenario: the bond portfolio is revalued on the same curve the liability is
    discounted at (:func:`~fastcashflow.assets.asset_value_by_scenario`), so a rate fall
    lifts BOTH the BEL and the bonds and the ratio reflects the asset-liability
    DURATION GAP, not just the liability move. Off (the default) holds the asset
    value at its t=0 base-curve level -- the liability-only distribution. The
    prescribed required capital (the denominator) is held at its t=0 value either
    way (a scenario overlay on the ratio, not a re-derived SCR)."""
    from fastcashflow._measurement.stochastic import measure_stochastic
    rs = getattr(rate_scenarios, "rates", rate_scenarios)
    static = assess(portfolio, model_points, basis, regime=regime,
                             **assess_kwargs)
    dist = measure_stochastic(model_points, basis, rs)
    asset_value = (asset_value_by_scenario(portfolio, rs) if co_moving_assets
                   else static.asset_portfolio_value)
    ac = asset_value - (dist.bel + static.risk_margin)
    if static.total_scr > 0.0:
        ratio = ac / static.total_scr
    else:
        ratio = np.where(ac >= 0.0, np.inf, -np.inf)
    return StochasticAssessment(static=static, available_capital=ac, ratio=ratio)
