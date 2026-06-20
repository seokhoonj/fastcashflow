"""Mass-lapse reinsurance namespace -- ``fcf.mass_lapse_reinsurance.*``.

A non-proportional (lapse excess-of-loss) treaty that transfers the tail of
mass-lapse risk and reduces the cedant's standard-formula lapse capital. The
treaty's loss base is the mass-lapse own-funds strain (:func:`loss_density`);
the layer between attachment and detachment is ceded.
"""
from fastcashflow._mass_lapse_reinsurance import (
    CREDIT_QUALITY_STEP_PD,
    SF_MASS_LAPSE_SHOCK,
    SF_MASS_LAPSE_SHOCK_GROUP_PENSION,
    ATTACHMENT_TAIL_ANCHOR,
    CedantSolvencyRelief,
    LapseReliefResult,
    LapseTailDistribution,
    LapseXL,
    MeasurementPeriod,
    ReinsurancePricing,
    SF_LAPSE_TAIL_ANCHOR,
    capital_relief,
    cedant_solvency_relief,
    counterparty_default_scr,
    lapse_loss_density as loss_density,
    price_treaty,
    windowed_claim,
)

__all__ = [
    "LapseXL",
    "MeasurementPeriod",
    "LapseTailDistribution",
    "ReinsurancePricing",
    "LapseReliefResult",
    "CedantSolvencyRelief",
    "loss_density",
    "capital_relief",
    "windowed_claim",
    "counterparty_default_scr",
    "cedant_solvency_relief",
    "price_treaty",
    "SF_LAPSE_TAIL_ANCHOR",
    "ATTACHMENT_TAIL_ANCHOR",
    "CREDIT_QUALITY_STEP_PD",
    "SF_MASS_LAPSE_SHOCK",
    "SF_MASS_LAPSE_SHOCK_GROUP_PENSION",
]
