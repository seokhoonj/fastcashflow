"""Mass-lapse reinsurance (lapse-XL) -- implementation.

A non-proportional reinsurance treaty that transfers the tail of mass-lapse
risk: the cedant retains lapses up to an attachment point, the reinsurer pays
the layer between attachment and detachment, and the cedant retains anything
beyond detachment. The detachment is usually set to the Solvency II standard-
formula mass-lapse stress (40% over best-estimate lapse rates, Delegated
Regulation Art. 142(6)(b)), so the layer caps exactly the regulatory shock.

Public surface is the ``fcf.mass_lapse_reinsurance`` namespace.

The treaty's loss base is the own-funds strain of a mass lapse -- the
:func:`lapse_loss_density`. When an extra fraction ``L`` (over best estimate) of
the in-force surrenders, the loss in basic own funds is ``L x S``, where ``S``
is the loss density: per policy, the surrender value paid less the liability
released, taken where that is a loss (the Art. 142(6) per-policy worst-
discontinuance selection). The recovery is the layer of that loss between the
attachment and detachment points, so it mirrors the loss linearly (EIOPA's
preferred form -- a flat band payout would create cliff-edge basis risk).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import IO, Protocol, runtime_checkable

import numpy as np

from fastcashflow.basis import Basis
from fastcashflow.engine import inforce_surrender_value, measure
from fastcashflow.model_points import ModelPoints
from fastcashflow.numerics import _norm_ppf
from fastcashflow._solvency import RegimeSpec, aggregate, required_capital

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the complementary error function."""
    return 0.5 * math.erfc(-x / _SQRT2)

# Solvency II standard-formula mass-lapse stress (Delegated Regulation
# Art. 142(6)(b)): an instantaneous 40% lapse of the in-force (70% for the
# management of group pension funds, Art. 2(3)(b)(iii)/(iv) of the Directive).
SF_MASS_LAPSE_SHOCK = 0.40
SF_MASS_LAPSE_SHOCK_GROUP_PENSION = 0.70


def lapse_loss_density(model_points: ModelPoints, basis: Basis) -> float:
    """Mass-lapse loss density ``S`` -- the loss in basic own funds per unit of
    excess (over best-estimate) lapse fraction.

    For each model point the loss from surrendering is the surrender value paid
    less the liability released, ``surrender_value - BEL``; a profitable
    contract (negative BEL) loses both the surrender value and the future
    profit, so its loss is largest. Following Delegated Regulation Art. 142(6)
    -- the discontinuance "which most negatively affects the basic own funds of
    the undertaking on a per policy basis" -- only model points where
    surrendering IS a loss contribute (``max(0, .)``); a model point that is
    onerous enough that surrender is a gain contributes zero, not a negative
    offset. (A model point is homogeneous, so the per-policy selection is a
    per-model-point ``max``.)

    Then a mass lapse of fraction ``L`` over best estimate loses ``L x S`` of
    basic own funds, and the standard-formula 40% shock loses
    ``SF_MASS_LAPSE_SHOCK x S``. This per-policy density is the treaty's loss
    base; it differs from :func:`fastcashflow.solvency.mass_lapse` (whose
    capital is ``fraction x max(0, sum(surrender_value - BEL))`` -- an aggregate
    that nets per-model-point gains against losses). The per-policy density is
    never smaller (``sum max(0, .) >= max(0, sum .)``) and is the form the
    standard-formula mass-lapse scenario prescribes.

    Surrender value is the valuation-date in-force surrender value
    (:func:`fastcashflow.engine.inforce_surrender_value`); zero where the basis
    prices none. Note ``S`` is NOT zero for a surrender-value-less book: a
    profitable model point (negative BEL) still loses its embedded value when it
    lapses (``surrender_value - BEL = -BEL > 0``), so ``S = sum max(0, -BEL)``
    there -- the lost-business value. The surrender value adds to that strain.
    ``S`` is zero only when every model point is onerous enough that surrender
    is a gain."""
    bel = measure(model_points, basis, full=False).bel
    surrender_value = inforce_surrender_value(model_points, basis)
    return float(np.sum(np.maximum(0.0, surrender_value - bel)))


@dataclass(frozen=True, slots=True)
class LapseXL:
    """A mass-lapse excess-of-loss treaty layer.

    ``attachment`` and ``detachment`` are excess lapse fractions OVER the
    best-estimate lapse rate (Delegated Regulation Art. 142(6) measures the
    shock as a lapse rate over assumed best estimate). The cedant retains
    losses below ``attachment``; the reinsurer pays the layer up to
    ``detachment``; the cedant retains losses beyond ``detachment``. A typical
    structure is ``LapseXL(0.15, 0.40)`` -- attach at 15% over best estimate,
    detach at the 40% standard-formula shock (EIOPA Annex 3.6).

    ``capacity`` is the layer width ``detachment - attachment`` (in excess-lapse
    terms); the recovery in loss terms is ``capacity x loss_density``.
    """

    attachment: float
    detachment: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.attachment < self.detachment <= 1.0):
            raise ValueError(
                "require 0 <= attachment < detachment <= 1, got "
                f"attachment={self.attachment}, detachment={self.detachment}")

    @property
    def capacity(self) -> float:
        """Layer width in excess-lapse terms (``detachment - attachment``)."""
        return self.detachment - self.attachment

    def covered_fraction(self, excess_lapse: float) -> float:
        """The excess-lapse fraction the treaty covers at observed
        ``excess_lapse`` -- ``clip(excess_lapse - attachment, 0, capacity)``."""
        return float(np.clip(excess_lapse - self.attachment, 0.0, self.capacity))

    def recovery(self, excess_lapse: float, loss_density: float) -> float:
        """Reinsurer recovery at observed ``excess_lapse`` over best estimate.

        ``loss_density x clip(excess_lapse - attachment, 0, capacity)`` -- the
        loss in the covered layer. Linear in the loss (no cliff-edge), so the
        cover mirrors the own-funds loss the cedant suffers in the band."""
        return loss_density * self.covered_fraction(excess_lapse)


# ---------------------------------------------------------------------------
# Measurement period -- the time axis (EIOPA Annex 3.8 / 3.9 / footnote 10).
# The treaty aggregates lapses over a measurement window (the "risk window"),
# NOT instantaneously: a claim exists where the window's accumulated excess
# lapse passes the attachment. This is distinct from the treaty's contractual
# duration. A 12-month window can miss a multi-year mass-lapse event (3.9); a
# longer window catches it.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MeasurementPeriod:
    """How the treaty aggregates lapses over time.

    ``months`` is the window length (the risk window / cover period -- EIOPA
    Annex 3.8, usually 12). ``mode`` is ``"reset"`` (non-overlapping windows; the
    accumulation resets each period -- 3.8) or ``"rolling"`` (a new window starts
    every ``step_months``, so windows overlap -- footnote 10). Under ``"rolling"``
    a single event falls in several windows; the treaty pays the high-water mark
    (the largest single-window claim), not the sum, to avoid paying several times
    for the same claim (footnote 10's "adjustment mechanisms").
    """

    months: int = 12
    mode: str = "reset"
    step_months: int = 3

    def __post_init__(self) -> None:
        if self.months <= 0:
            raise ValueError(f"months must be positive, got {self.months}")
        if self.mode not in ("reset", "rolling"):
            raise ValueError(f"mode must be 'reset' or 'rolling', got {self.mode!r}")
        if self.mode == "rolling" and self.step_months <= 0:
            raise ValueError(
                f"step_months must be positive for rolling, got {self.step_months}")

    def _window_starts(self, duration_months: int) -> list[int]:
        """The month each measurement window opens within the treaty duration."""
        if self.mode == "reset":
            return list(range(0, duration_months, self.months))
        return list(range(0, max(1, duration_months - self.months + 1), self.step_months))


def windowed_claim(
    excess_lapse_monthly, treaty: LapseXL, loss_density: float,
    measurement: MeasurementPeriod, *, duration_months: int | None = None,
) -> float:
    """Total treaty claim over its duration from a monthly excess-lapse path.

    ``excess_lapse_monthly`` is the per-month lapse fraction in EXCESS of best
    estimate (of the original in-force). Each measurement window accumulates its
    months' excess lapse; the window claim is the treaty recovery on that
    accumulated excess (``loss_density x clip(window_sum - attachment, 0,
    capacity)``). ``reset`` windows sum their claims (each period is a fresh
    layer); ``rolling`` windows take the high-water mark -- the largest single
    window claim -- so a single event spanning several overlapping windows is
    paid once (EIOPA footnote 10).

    A 12-month window may miss a multi-year event (e.g. 20% + 20% over two years
    never reaches a 20% attachment in any one window), which a longer window
    catches -- EIOPA Annex 3.9."""
    incr = np.asarray(excess_lapse_monthly, dtype=np.float64)
    n = duration_months if duration_months is not None else incr.shape[0]
    claims = []
    for start in measurement._window_starts(n):
        end = min(start + measurement.months, incr.shape[0])
        window_excess = float(incr[start:end].sum())
        claims.append(treaty.recovery(window_excess, loss_density))
    if not claims:
        return 0.0
    if measurement.mode == "reset":
        return float(sum(claims))
    return float(max(claims))                # rolling: high-water mark


@dataclass(frozen=True, slots=True)
class LapseReliefResult:
    """The cedant's mass-lapse capital relief from a :class:`LapseXL` treaty.

    All amounts are in own-funds currency. ``gross_scr`` is the standard-formula
    mass-lapse capital before the treaty (``shock x loss_density``);
    ``recovery`` is what the treaty pays in the shock scenario; ``net_scr`` is
    the capital the cedant still holds after the treaty; ``relief`` is the
    reduction (``gross_scr - net_scr == recovery``). This is the headline number
    a reinsurer quotes on the cedant's book -- the mass-lapse module only; the
    lapse-risk SCR is ``max(net mass, lapse up, lapse down)``, so a smaller mass
    capital may let another lapse scenario bite (handled by the solvency
    integration, not here).
    """

    loss_density: float
    shock: float
    gross_scr: float
    recovery: float
    net_scr: float

    @property
    def relief(self) -> float:
        """Capital relief == gross_scr - net_scr == recovery."""
        return self.gross_scr - self.net_scr


def capital_relief(model_points: ModelPoints, basis: Basis, treaty: LapseXL,
                   *, shock: float = SF_MASS_LAPSE_SHOCK) -> LapseReliefResult:
    """Cedant mass-lapse capital relief from ``treaty`` on the portfolio.

    The standard-formula mass-lapse scenario lapses ``shock`` of the in-force
    (40% retail, 70% group pension); the loss is ``shock x loss_density``. The
    treaty pays its layer recovery at that lapse level, so the net mass-lapse
    capital is the loss the cedant retains -- ``shock x loss_density`` less the
    recovery. With a detachment at the shock (the usual structure), the retained
    capital collapses to the attachment layer ``attachment x loss_density``.

    The detachment and attachment are excess-over-best-estimate fractions and
    the shock is aligned to that scale (EIOPA Annex 3.6: the detachment is
    usually set to the 40% standard-formula stress over best estimate)."""
    S = lapse_loss_density(model_points, basis)
    gross = shock * S
    recovery = treaty.recovery(shock, S)
    return LapseReliefResult(
        loss_density=S, shock=shock, gross_scr=gross,
        recovery=recovery, net_scr=gross - recovery)


# ---------------------------------------------------------------------------
# Counterparty default risk on the reinsurer exposure
# (Delegated Regulation Art. 189-201). Buying the treaty adds a credit charge
# on the reinsurer, which partly offsets the lapse-SCR relief.
# ---------------------------------------------------------------------------

# Probability of default by credit quality step (Delegated Regulation Art. 199),
# steps 0..6. A reinsurer is typically AAA/AA/A -> step 0/1/2.
CREDIT_QUALITY_STEP_PD = (0.00002, 0.0001, 0.0005, 0.0024, 0.012, 0.042, 0.042)


def counterparty_default_scr(
    recoverables: float, risk_mitigating_effect: float,
    probability_of_default: float, *,
    collateral: float = 0.0, collateral_factor: float = 0.0,
) -> float:
    """SCR for counterparty default on a single type-1 reinsurance exposure
    (Delegated Regulation Art. 192, 200, 201).

    The reinsurance recoverable plus the loss of the treaty's risk-mitigating
    effect on default is the loss-given-default (Art. 192(2)):

        LGD = max(0, 0.50 x (recoverables + 0.50 x risk_mitigating_effect)
                       - collateral_factor x collateral)

    ``risk_mitigating_effect`` (``RM_re``) is the SCR reduction the treaty
    provides -- here the lapse-SCR relief; on the reinsurer's default the cedant
    loses both the recoverable and that mitigation. With a SINGLE counterparty
    the Art. 201 variance collapses to ``V = PD (1 - PD) LGD^2`` (the Vinter +
    Vintra cross terms cancel: ``(1 - PD) + 1.5 = 2.5 - PD``), so
    ``sigma = LGD sqrt(PD (1 - PD))`` and Art. 200 gives, with
    ``sigma / sum(LGD) = sqrt(PD (1 - PD))``:

        sqrt(PD(1-PD)) <= 7%   ->  SCR_def = 3 sigma
        7% < .         <= 20%  ->  SCR_def = 5 sigma
        > 20%                  ->  SCR_def = LGD

    Type 2 is zero for a pure reinsurance counterparty, so ``SCR_def`` is the
    type-1 amount. A typical reinsurer (PD 0.01-0.24%) lands in the first case,
    so the charge is small (~1-7% of LGD) -- the add-back that makes the net
    mass-lapse benefit less than the gross relief."""
    lgd = max(0.0, 0.50 * (recoverables + 0.50 * risk_mitigating_effect)
                   - collateral_factor * collateral)
    if lgd == 0.0:
        return 0.0
    pd = probability_of_default
    sigma = lgd * math.sqrt(pd * (1.0 - pd))
    ratio = math.sqrt(pd * (1.0 - pd))          # sigma / sum(LGD) for one exposure
    if ratio <= 0.07:
        return 3.0 * sigma
    if ratio <= 0.20:
        return 5.0 * sigma
    return lgd


# ---------------------------------------------------------------------------
# Cedant solvency relief -- the full picture into the life underwriting module
# (diversified) plus the counterparty-default add-back and the risk margin.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CedantSolvencyRelief:
    """The cedant's full Solvency II benefit from a mass-lapse treaty.

    The lapse sub-risk is the worst of lapse up / lapse down / mass; the treaty
    cuts only the mass leg, so ``lapse_net`` can fall to the next-biting leg
    (the relief is bounded by how far mass exceeds lapse up/down). The lapse
    capital re-aggregates into the life underwriting module with the other
    sub-risks, so the diversified ``insurance`` relief is smaller than the
    standalone lapse relief. Buying the treaty adds ``counterparty_default``
    (the reinsurer credit charge) and lowers the risk margin.

    All mass-lapse figures use the per-policy loss density
    (:func:`lapse_loss_density`, Art. 142(6)), so ``mass_gross`` may exceed the
    aggregate ``solvency.mass_lapse`` used by a plain
    :func:`fastcashflow.gmm.required_capital` run.
    """

    loss_density: float
    mass_gross_scr: float
    mass_net_scr: float
    lapse_gross_scr: float
    lapse_net_scr: float
    insurance_gross_scr: float
    insurance_net_scr: float
    counterparty_default: float
    risk_margin_gross: float
    risk_margin_net: float

    @property
    def lapse_relief(self) -> float:
        """Standalone lapse-module relief (before life diversification)."""
        return self.lapse_gross_scr - self.lapse_net_scr

    @property
    def insurance_relief(self) -> float:
        """Diversified life-underwriting-module SCR relief (RM_re for the
        counterparty-default charge)."""
        return self.insurance_gross_scr - self.insurance_net_scr

    @property
    def risk_margin_relief(self) -> float:
        """Risk-margin reduction (an own-funds gain)."""
        return self.risk_margin_gross - self.risk_margin_net

    @property
    def net_scr_benefit(self) -> float:
        """SCR-side benefit: insurance relief less the counterparty-default
        add-back (no inter-module diversification credit -- fcf v1)."""
        return self.insurance_relief - self.counterparty_default

    @property
    def total_benefit(self) -> float:
        """Total own-funds + SCR benefit before the reinsurance premium:
        net SCR benefit plus the risk-margin relief."""
        return self.net_scr_benefit + self.risk_margin_relief


def _is_mass_lapse_variant(stress) -> bool:
    return stress.name.startswith("mass lapse")


def _regime_mass_lapse_shock(lapse_sub_risk) -> float:
    """The regime's own mass-lapse fraction, read from its lapse sub-risk's
    mass-lapse variant (``mass_lapse`` names itself ``"mass lapse {fraction}"``)
    -- 40% under Solvency II, 30% under K-ICS. Falls back to the Solvency II
    shock if the regime has no mass-lapse variant."""
    for v in lapse_sub_risk.variants:
        if _is_mass_lapse_variant(v):
            try:
                return float(v.name.rsplit(None, 1)[1])
            except (IndexError, ValueError):
                break
    return SF_MASS_LAPSE_SHOCK


def cedant_solvency_relief(
    model_points: ModelPoints, basis: Basis, treaty: LapseXL, *,
    regime: RegimeSpec, reinsurer_pd: float, shock: float | None = None,
    recoverables: float = 0.0, collateral: float = 0.0,
    collateral_factor: float = 0.0,
) -> CedantSolvencyRelief:
    """The cedant's full Solvency II relief from ``treaty`` under ``regime``.

    Re-aggregates the life underwriting module with the treaty-reduced lapse
    capital, charges counterparty-default risk on the reinsurer, and scales the
    risk margin. The lapse sub-risk is ``max(lapse_up, lapse_down, mass)``; the
    treaty cuts the mass leg from ``shock x loss_density`` to its net (post-
    recovery) value, so the lapse capital drops only to the next-biting leg.

    ``reinsurer_pd`` is the reinsurer's probability of default
    (:data:`CREDIT_QUALITY_STEP_PD` by credit quality step). The
    counterparty-default charge uses the diversified insurance relief as the
    risk-mitigating effect ``RM_re`` (Art. 192). The risk margin is scaled by
    the regime's risk-margin-to-insurance ratio applied to the gross / net
    insurance SCR.

    Lapse up / down come from the regime's own lapse variants (everything in the
    lapse sub-risk that is not the mass-lapse variant); the other sub-risk
    capitals come from one gross :func:`fastcashflow.gmm.required_capital`
    run and re-aggregate unchanged.

    ``shock`` defaults to the regime's own mass-lapse fraction (40% Solvency II,
    30% K-ICS), read from its lapse sub-risk, so the relief is consistent with
    the regime; pass it explicitly only to override."""
    lapse_sr = next(sr for sr in regime.sub_risks if sr.name == "lapse")
    if shock is None:
        shock = _regime_mass_lapse_shock(lapse_sr)
    gross = required_capital(model_points, basis, regime=regime)

    base_bel = float(measure(model_points, basis, full=False).bel.sum())

    def delta(stress) -> float:
        mp2, basis2 = stress.apply(model_points, basis)
        d = float(measure(mp2, basis2, full=False).bel.sum()) - base_bel
        if stress.bel_addon is not None:
            d += stress.bel_addon(model_points, basis)
        return d

    updown = [max(0.0, delta(v)) for v in lapse_sr.variants
              if not _is_mass_lapse_variant(v)]
    floor = max(updown) if updown else 0.0           # the next-biting lapse leg

    relief = capital_relief(model_points, basis, treaty, shock=shock)
    mass_gross = relief.gross_scr
    mass_net = relief.net_scr
    lapse_gross = max(floor, mass_gross)
    lapse_net = max(floor, mass_net)

    caps = dict(gross.sub_risk_capital)
    insurance_gross = aggregate({**caps, "lapse": lapse_gross}, regime)
    insurance_net = aggregate({**caps, "lapse": lapse_net}, regime)
    insurance_relief = insurance_gross - insurance_net

    cpd = counterparty_default_scr(
        recoverables, risk_mitigating_effect=insurance_relief,
        probability_of_default=reinsurer_pd,
        collateral=collateral, collateral_factor=collateral_factor)

    rm_per_unit = (gross.risk_margin / gross.insurance_scr
                   if gross.insurance_scr > 0.0 else 0.0)

    return CedantSolvencyRelief(
        loss_density=relief.loss_density,
        mass_gross_scr=mass_gross, mass_net_scr=mass_net,
        lapse_gross_scr=lapse_gross, lapse_net_scr=lapse_net,
        insurance_gross_scr=insurance_gross, insurance_net_scr=insurance_net,
        counterparty_default=cpd,
        risk_margin_gross=rm_per_unit * insurance_gross,
        risk_margin_net=rm_per_unit * insurance_net)


# ---------------------------------------------------------------------------
# Reinsurer side: the lapse tail distribution and pricing (Phase D).
#
# The cedant side is deterministic because the standard formula fixes the lapse
# at a single point (40%). The reinsurer prices over the WHOLE tail of the
# cumulative excess-over-best-estimate lapse L, so it needs a distribution F(L).
# Pricing depends on F only through ``survival(x) = P(L > x)``: the expected
# layer is the integral of the survival over the layer (a standard identity),
#
#     E[clip(L - a, 0, b - a)] = integral_a^b P(L > x) dx,
#
# so any object exposing ``survival`` (and the ``expected_layer`` it implies) is
# a drop-in F(L). :class:`LapseTailDistribution` is the public baseline,
# calibrated to public Solvency II anchors; a reinsurer's proprietary F(L)
# (dynamic lapse, dependence structure, cross-portfolio tail) replaces it
# without touching the pricing.
# ---------------------------------------------------------------------------

# Public tail anchors for the excess-over-best-estimate lapse L (exceedance
# probabilities): the 40% standard-formula stress is the 1-in-200 (99.5%) point
# (Art. 142(6)(b)); EIOPA notes attachment points are typically set around a
# 1-in-30 event (e.g. 15%). The second anchor is a calibration choice, not a
# regulatory law -- override it with the book's own lapse volatility.
SF_LAPSE_TAIL_ANCHOR = (0.40, 1.0 / 200.0)
ATTACHMENT_TAIL_ANCHOR = (0.15, 1.0 / 30.0)


class LapseDistribution:
    """The Engine/Model seam: the distribution F(L) of the cumulative excess-
    over-best-estimate lapse fraction ``L`` that reinsurer pricing integrates
    against.

    This is fastcashflow's plug-in point. The valuable, proprietary part of
    mass-lapse reinsurance is the MODEL that produces F(L) -- calibrated to a
    reinsurer's cross-portfolio lapse experience (its deepest moat), the
    economic-to-lapse link, and channel-level clustering. fastcashflow is the
    ENGINE: it takes ANY F(L) and returns the capital relief, pricing, capital,
    risk adjustment and CSM. To plug in a model, subclass this and implement
    ``survival``; ``expected_layer`` and ``value_at_risk`` then work for free
    (pricing needs nothing else). A subclass MAY override them with closed forms
    (as :class:`LapseTailDistribution` does) for speed and precision."""

    __slots__ = ()

    def survival(self, x: float) -> float:
        """``P(L > x)`` -- the one method a plug-in F(L) must provide."""
        raise NotImplementedError

    def expected_layer(self, attachment: float, detachment: float) -> float:
        """``E[clip(L - attachment, 0, detachment - attachment)]`` -- the expected
        covered fraction, equal to the survival integral over the layer (the
        identity any survival-only F(L) prices through). Numerical default;
        override for a closed form."""
        if not (0.0 <= attachment < detachment):
            raise ValueError("require 0 <= attachment < detachment")
        grid = np.linspace(attachment, detachment, 4001)
        surv = np.array([self.survival(float(x)) for x in grid])
        return float(np.trapezoid(surv, grid))

    def value_at_risk(self, q: float) -> float:
        """The ``q``-quantile of ``L`` -- the lapse level ``x`` with
        ``survival(x) = 1 - q`` -- by bisection on the (decreasing) survival.
        Numerical default; override for a closed form."""
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q}")
        target = 1.0 - q
        lo, hi = 0.0, 1.0
        while self.survival(hi) > target and hi < 1e6:    # expand to bracket
            hi *= 2.0
        for _ in range(100):                              # bisection
            mid = 0.5 * (lo + hi)
            if self.survival(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


@runtime_checkable
class LapseModel(Protocol):
    """A lapse MODEL: produces a :class:`LapseDistribution` for a given context
    (e.g. a book's channel mix or an economic scenario).

    This is the proprietary layer fastcashflow deliberately leaves to the user.
    A reinsurer's model -- calibrated to cross-portfolio tail data and the
    channel-level clustering that actually drives mass lapse -- is its real IP,
    and is never open-sourced. The baseline
    :meth:`LapseTailDistribution.from_anchors` is a trivial context-free model;
    the engine consumes only the returned distribution, so any model satisfying
    this protocol drops in."""

    def distribution(self, context=None) -> LapseDistribution:
        ...


@dataclass(frozen=True, slots=True)
class LapseTailDistribution(LapseDistribution):
    """Lognormal distribution of the cumulative excess-over-best-estimate lapse
    fraction ``L``, the public baseline F(L) for reinsurer pricing.

    Calibrated by :meth:`from_anchors` to two tail exceedance probabilities
    (default: 15% at 1-in-30, 40% at 1-in-200). Pricing depends on a
    distribution ONLY through :meth:`survival`; a reinsurer's proprietary F(L)
    is a drop-in replacement that provides the same method. The closed-form
    :meth:`expected_layer` equals ``integral_attach^detach survival(x) dx`` -- the
    identity any survival-only F(L) can use numerically."""

    mu: float
    sigma: float

    @classmethod
    def from_anchors(cls, lower=ATTACHMENT_TAIL_ANCHOR,
                     upper=SF_LAPSE_TAIL_ANCHOR) -> "LapseTailDistribution":
        """Calibrate the lognormal to two ``(lapse_level, exceedance_prob)``
        anchors. ``P(L > a) = p`` gives ``(ln a - mu)/sigma = z`` with
        ``z = norm_ppf(1 - p)``; two anchors solve ``mu`` and ``sigma``."""
        (a1, p1), (a2, p2) = lower, upper
        if not (0.0 < a1 < a2 and 0.0 < p2 < p1 < 1.0):
            raise ValueError(
                "anchors must satisfy 0 < a1 < a2 and 0 < p2 < p1 < 1 "
                f"(a higher lapse is rarer), got {lower}, {upper}")
        z1, z2 = _norm_ppf(1.0 - p1), _norm_ppf(1.0 - p2)
        sigma = (math.log(a2) - math.log(a1)) / (z2 - z1)
        mu = math.log(a1) - sigma * z1
        return cls(mu=mu, sigma=sigma)

    def survival(self, x: float) -> float:
        """``P(L > x)`` -- the only method pricing requires of an F(L)."""
        if x <= 0.0:
            return 1.0
        return 1.0 - _norm_cdf((math.log(x) - self.mu) / self.sigma)

    @property
    def mean(self) -> float:
        """``E[L]`` -- the expected excess lapse."""
        return math.exp(self.mu + 0.5 * self.sigma * self.sigma)

    def expected_excess(self, k: float) -> float:
        """``E[(L - k)+]`` -- the lognormal stop-loss above ``k``."""
        if k <= 0.0:
            return self.mean
        d1 = (self.mu + self.sigma * self.sigma - math.log(k)) / self.sigma
        d2 = d1 - self.sigma
        return self.mean * _norm_cdf(d1) - k * _norm_cdf(d2)

    def expected_layer(self, attachment: float, detachment: float) -> float:
        """``E[clip(L - attachment, 0, detachment - attachment)]`` -- the expected
        covered excess-lapse fraction (equals the survival integral over the
        layer)."""
        return self.expected_excess(attachment) - self.expected_excess(detachment)

    def value_at_risk(self, q: float) -> float:
        """The ``q``-quantile of ``L`` -- ``exp(mu + sigma x norm_ppf(q))``. At
        ``q = 0.995`` this is the 1-in-200 lapse (the standard-formula stress; by
        calibration it returns the upper anchor)."""
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q}")
        return math.exp(self.mu + self.sigma * _norm_ppf(q))


@dataclass(frozen=True, slots=True)
class ReinsurancePricing:
    """The reinsurer's price and assumed capital for a :class:`LapseXL` treaty.

    ``expected_recovery`` is the pure premium (expected loss) ``S x E[layer]``;
    ``capital`` is the assumed risk capital -- the unexpected loss
    ``VaR(recovery) - expected_recovery`` at ``var_level`` -- after the
    reinsurer's diversification factor; ``premium`` loads the cost of capital on
    top of the pure premium. ``expected_profit`` is the load (premium less
    expected loss)."""

    loss_density: float
    capacity_at_risk: float          # S x capacity -- the most the layer can pay
    expected_recovery: float
    capital: float
    premium: float

    @property
    def expected_profit(self) -> float:
        """Premium less expected loss -- the cost-of-capital load."""
        return self.premium - self.expected_recovery

    @property
    def rate_on_line(self) -> float:
        """Premium as a fraction of the capacity at risk (the market quote
        convention -- typically a low single-digit percent)."""
        return self.premium / self.capacity_at_risk if self.capacity_at_risk else 0.0

    @property
    def loss_on_line(self) -> float:
        """Expected loss as a fraction of the capacity at risk."""
        return (self.expected_recovery / self.capacity_at_risk
                if self.capacity_at_risk else 0.0)


def price_treaty(
    loss_density: float, treaty: LapseXL, distribution: LapseDistribution, *,
    cost_of_capital: float = 0.06, var_level: float = 0.995,
    diversification_factor: float = 1.0,
) -> ReinsurancePricing:
    """Price the treaty from the reinsurer's side over the lapse tail
    ``distribution`` (any object exposing ``expected_layer`` and
    ``value_at_risk``; :class:`LapseTailDistribution` is the public baseline).

    ``expected_recovery = S x E[layer]`` is the pure premium. The assumed capital
    is the unexpected loss ``S x covered_fraction(VaR_level(L)) -
    expected_recovery`` scaled by ``diversification_factor`` (1.0 = standalone;
    a reinsurer diversifies the assumed lapse risk against its own book, so its
    marginal capital is a fraction of standalone -- pass e.g. 0.25). The premium
    loads ``cost_of_capital`` on that capital:

        premium = expected_recovery + cost_of_capital x capital.

    Everything tail-dependent flows through ``distribution`` -- the proprietary
    F(L) the reinsurer plugs in sets the price."""
    S = loss_density
    capacity_at_risk = S * treaty.capacity
    expected_recovery = S * distribution.expected_layer(
        treaty.attachment, treaty.detachment)
    var_lapse = distribution.value_at_risk(var_level)
    unexpected = S * treaty.covered_fraction(var_lapse) - expected_recovery
    capital = max(0.0, unexpected) * diversification_factor
    premium = expected_recovery + cost_of_capital * capital
    return ReinsurancePricing(
        loss_density=S, capacity_at_risk=capacity_at_risk,
        expected_recovery=expected_recovery, capital=capital, premium=premium)


# ---------------------------------------------------------------------------
# Reinsurer-side IFRS 17 measurement of the assumed treaty (Phase D3).
# The treaty is a stream of premium inflows and contingent recovery outflows --
# a portfolio-level structure, not a per-policy projection, so it is measured by
# direct discounting. Sign convention (the engine's): outflow-positive, so the
# recovery the reinsurer pays is positive and the premium it receives is
# negative; BEL = PV(recovery) - PV(premium). For an out-of-the-money treaty the
# premium exceeds the expected recovery, so the BEL is negative (profitable) and
# the unearned profit sits in the CSM.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AssumedTreatyMeasurement:
    """IFRS 17 measurement of the treaty from the reinsurer's (assuming) side.

    ``bel = PV(expected recovery) - PV(premium)`` (outflow-positive);
    ``risk_adjustment`` is the cost-of-capital margin on the assumed capital over
    the treaty. ``fulfilment_cash_flows = bel + risk_adjustment``; the CSM is the
    unearned profit ``max(0, -fcf)`` and the loss component ``max(0, fcf)``
    (General Measurement Model -- IFRS 17 Sec. 38, 47)."""

    pv_premium: float
    pv_expected_recovery: float
    bel: float
    risk_adjustment: float

    @property
    def fulfilment_cash_flows(self) -> float:
        """``BEL + RA`` -- negative for a profitable assumed treaty."""
        return self.bel + self.risk_adjustment

    @property
    def csm(self) -> float:
        """Contractual service margin -- the unearned profit ``max(0, -FCF)``."""
        return max(0.0, -self.fulfilment_cash_flows)

    @property
    def loss_component(self) -> float:
        """Onerous loss component ``max(0, FCF)`` (zero unless the premium is
        below the risk-adjusted expected recovery)."""
        return max(0.0, self.fulfilment_cash_flows)


def measure_assumed_treaty(
    pricing: ReinsurancePricing, *, duration_years: int,
    discount_annual: float = 0.0, risk_adjustment_cost_of_capital: float = 0.06,
) -> AssumedTreatyMeasurement:
    """Measure the assumed treaty over ``duration_years`` annual periods.

    Premium is received in advance (start of each period); the expected recovery
    is paid in arrears (end of each period, when the measurement window closes).
    The risk adjustment is the cost of capital on the assumed capital held each
    period. Discounting is a flat ``discount_annual``.

    ``BEL = PV(recovery) - PV(premium)``, ``RA = ra_coc x capital x annuity``,
    and the CSM / loss component follow the standard sign convention. A treaty
    priced at exactly the cost of capital has BEL + RA ~ 0 (no unearned profit);
    a premium loaded above the cost of capital leaves a positive CSM."""
    if duration_years <= 0:
        raise ValueError(f"duration_years must be positive, got {duration_years}")
    v = 1.0 / (1.0 + discount_annual)
    annuity_advance = sum(v ** t for t in range(duration_years))        # t = 0..n-1
    annuity_arrear = sum(v ** t for t in range(1, duration_years + 1))  # t = 1..n
    pv_premium = pricing.premium * annuity_advance
    pv_recovery = pricing.expected_recovery * annuity_arrear
    bel = pv_recovery - pv_premium
    ra = risk_adjustment_cost_of_capital * pricing.capital * annuity_arrear
    return AssumedTreatyMeasurement(
        pv_premium=pv_premium, pv_expected_recovery=pv_recovery,
        bel=bel, risk_adjustment=ra)


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
