"""Mass-lapse reinsurance namespace -- ``fcf.lapse_reinsurance.*``.

A non-proportional (lapse excess-of-loss) treaty that transfers the tail of
mass-lapse risk and reduces the cedant's standard-formula lapse capital. The
treaty's loss base is the mass-lapse own-funds strain (:func:`loss_density`);
the layer between attachment and detachment is ceded.
"""
from fastcashflow._lapse_reinsurance import (
    SF_MASS_LAPSE_SHOCK,
    SF_MASS_LAPSE_SHOCK_GROUP_PENSION,
    LapseReliefResult,
    LapseXL,
    capital_relief,
    lapse_loss_density as loss_density,
)

__all__ = [
    "LapseXL",
    "LapseReliefResult",
    "loss_density",
    "capital_relief",
    "SF_MASS_LAPSE_SHOCK",
    "SF_MASS_LAPSE_SHOCK_GROUP_PENSION",
]
