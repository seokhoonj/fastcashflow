"""Mass-lapse reinsurance -- the analysis report (Phase E).

One portfolio-level report tying the cedant relief, the reinsurer pricing and
the reinsurer IFRS 17 measurement together; ASCII / English, the trace family's
counterpart for the whole treaty.
"""
from __future__ import annotations

import sys
from typing import IO

from fastcashflow.basis import Basis
from fastcashflow.model_points import ModelPoints
from fastcashflow.solvency._engine import RegimeSpec
from fastcashflow._mass_lapse._cedant import (
    LapseXL, cedant_solvency_relief)
from fastcashflow._mass_lapse._reinsurer import (
    LapseDistribution, LapseTailDistribution, price_treaty,
    measure_assumed_treaty)


# ---------------------------------------------------------------------------
# Analysis package -- one report tying the cedant relief, the reinsurer pricing
# and the reinsurer IFRS 17 measurement together (Phase E). Portfolio-level
# (not per-model-point), ASCII / English -- the trace family's counterpart for
# the whole treaty.
# ---------------------------------------------------------------------------

def _money(x: float, width: int = 18) -> str:
    return f"{x:>{width},.0f}"


def report(
    model_points: ModelPoints, basis: Basis, treaty: LapseXL, *,
    regime: RegimeSpec, reinsurer_pd: float,
    distribution: LapseDistribution | None = None,
    diversification_factor: float = 1.0, shock: float | None = None,
    duration_years: int = 3, discount_annual: float = 0.0,
    cost_of_capital: float = 0.06, file: "IO | None" = None,
) -> None:
    """Print the mass-lapse reinsurance analysis package for a portfolio.

    Three sections: (1) the cedant's capital relief
    (:func:`cedant_solvency_relief`), (2) the reinsurer's pricing
    (:func:`price_treaty`) over ``distribution`` (the baseline F(L) when None),
    and (3) the reinsurer's IFRS 17 measurement of the assumed treaty
    (:func:`measure_assumed_treaty`). All amounts are in own-funds currency.

    ``distribution`` is the lapse tail F(L); pass a regime-appropriate one (the
    default baseline is calibrated to the Solvency II 40% / 15% anchors). The
    cedant relief reads the mass-lapse shock from the regime, so the same call
    serves Solvency II and K-ICS."""
    if file is None:
        file = sys.stdout
    F = distribution if distribution is not None else LapseTailDistribution.from_anchors()

    relief = cedant_solvency_relief(
        model_points, basis, treaty, regime=regime, reinsurer_pd=reinsurer_pd,
        shock=shock)
    pricing = price_treaty(
        relief.loss_density, treaty, F, cost_of_capital=cost_of_capital,
        diversification_factor=diversification_factor)
    meas = measure_assumed_treaty(
        pricing, duration_years=duration_years, discount_annual=discount_annual,
        risk_adjustment_cost_of_capital=cost_of_capital)

    bar = "=" * 78
    out: list[str] = [
        bar,
        " Mass-lapse reinsurance -- analysis package",
        (f" regime={regime.name}   treaty=LapseXL(attach={treaty.attachment:g}, "
         f"detach={treaty.detachment:g})   reinsurer PD={reinsurer_pd:.3%}"),
        bar,
        "",
        "[1] Cedant capital relief",
        f"  loss density S                 : {_money(relief.loss_density)}",
        (f"  mass-lapse SCR    gross / net  : {_money(relief.mass_gross_scr)} /"
         f"{_money(relief.mass_net_scr)}"),
        (f"  lapse SCR         gross / net  : {_money(relief.lapse_gross_scr)} /"
         f"{_money(relief.lapse_net_scr)}   (net floored by lapse up/down)"),
        (f"  insurance SCR     gross / net  : {_money(relief.insurance_gross_scr)} /"
         f"{_money(relief.insurance_net_scr)}   (diversified)"),
        f"  insurance relief (RM_re)       : {_money(relief.insurance_relief)}",
        f"  counterparty default add-back  : {_money(relief.counterparty_default)}",
        f"  risk margin relief             : {_money(relief.risk_margin_relief)}",
        "  " + "-" * 60,
        f"  total benefit (pre-premium)    : {_money(relief.total_benefit)}",
        "",
        (f"[2] Reinsurer pricing   (F(L)={type(F).__name__}, "
         f"diversification={diversification_factor:g})"),
        (f"  expected recovery              : {_money(pricing.expected_recovery)}"
         f"   (loss-on-line {pricing.loss_on_line:.2%})"),
        f"  assumed capital                : {_money(pricing.capital)}",
        (f"  premium                        : {_money(pricing.premium)}"
         f"   ({pricing.rate_on_line:.2%} of capacity)"),
        f"  expected profit                : {_money(pricing.expected_profit)}",
        "",
        (f"[3] Reinsurer IFRS 17 measurement   (duration={duration_years}y, "
         f"discount={discount_annual:g})"),
        f"  PV premium / PV recovery       : {_money(meas.pv_premium)} /"
        f"{_money(meas.pv_expected_recovery)}",
        f"  BEL                            : {_money(meas.bel)}   (neg = profitable)",
        f"  risk adjustment                : {_money(meas.risk_adjustment)}",
        f"  CSM / loss component           : {_money(meas.csm)} /"
        f"{_money(meas.loss_component)}",
        bar,
    ]
    file.write("\n".join(out) + "\n")
