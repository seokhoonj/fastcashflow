"""Mass-lapse reinsurance implementation package.

Private -- the public surface is the facade module
:mod:`fastcashflow.reinsurance.mass_lapse` (``fcf.reinsurance.mass_lapse``).
Layered cedant -> reinsurer -> report (a one-directional DAG, no cycle). This
__init__ re-exports the names the facade and cross-module callers import from
``fastcashflow._mass_lapse_reinsurance``.
"""
from fastcashflow._mass_lapse_reinsurance._cedant import (
    SF_MASS_LAPSE_SHOCK, SF_MASS_LAPSE_SHOCK_GROUP_PENSION,
    CREDIT_QUALITY_STEP_PD,
    lapse_loss_density, LapseXL, MeasurementPeriod, windowed_claim,
    LapseReliefResult, capital_relief, counterparty_default_scr,
    CedantSolvencyRelief, cedant_solvency_relief)
from fastcashflow._mass_lapse_reinsurance._reinsurer import (
    SF_LAPSE_TAIL_ANCHOR, ATTACHMENT_TAIL_ANCHOR,
    LapseDistribution, LapseModel, LapseTailDistribution, ReinsurancePricing,
    price_treaty, AssumedTreatyMeasurement, measure_assumed_treaty)
from fastcashflow._mass_lapse_reinsurance._report import report
