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
from dataclasses import dataclass

import numpy as np

from fastcashflow.basis import Basis
from fastcashflow.engine import inforce_surrender_value, measure
from fastcashflow.model_points import ModelPoints

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
