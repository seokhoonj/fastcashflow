"""fastcashflow -- fast IFRS 17 GMM cash flow projection engine.

Two entry points:

* :func:`value`   -- fast, fused valuation (BEL, RA, CSM per model point).
* :func:`measure` -- detailed: full cash flow and CSM trajectories.

Conventional import alias::

    import fastcashflow as fcf

`fcf` also reads as Fulfilment Cash Flows (IFRS 17: BEL + RA) -- the very
quantity this engine computes.
"""
from fastcashflow.assumptions import Assumptions, RiderRate
from fastcashflow.coverage import DEATH, RISK_MORBIDITY, RISK_MORTALITY
from fastcashflow.engine import (
    Measurement, Valuation, measure, value, value_segmented,
)
from fastcashflow.grouping import group
from fastcashflow.io import (
    load_sample_assumptions,
    load_sample_model_points,
    read_assumptions,
    read_model_points,
    read_scenarios,
    value_file,
    write_valuation,
)
from fastcashflow.modelpoints import (
    STATE_ACTIVE,
    STATE_PAIDUP,
    STATE_WAIVER,
    ModelPoints,
)
from fastcashflow.movement import (
    PAAPeriodMovement,
    PAAReconciliation,
    PeriodMovement,
    Reconciliation,
    VFAPeriodMovement,
    VFAReconciliation,
    reconcile,
    roll_forward,
)
from fastcashflow.paa import PAAMeasurement, measure_paa
from fastcashflow.plots import (
    plot_analysis_of_change,
    plot_cashflows,
    plot_csm_runoff,
    plot_liability,
    plot_risk_adjustment,
    plot_stochastic,
)
from fastcashflow.pricing import solve_premium
from fastcashflow.reinsurance import ReinsuranceMeasurement, measure_reinsurance
from fastcashflow.report import Report, report
from fastcashflow.statemodel import (
    STATE_MODELS,
    WAIVER_MODEL,
    State,
    StateModel,
    Transition,
)
from fastcashflow.stochastic import StochasticResult, value_stochastic
from fastcashflow.transition import transition
from fastcashflow.tvog import TVOGResult, measure_tvog
from fastcashflow.vfa import VFAMeasurement, measure_vfa

__version__ = "0.0.1.dev1"
__all__ = [
    "Assumptions", "ModelPoints", "measure", "value", "value_segmented",
    "value_stochastic",
    "measure_paa", "measure_vfa", "measure_reinsurance", "measure_tvog",
    "report", "roll_forward", "reconcile", "group", "transition",
    "Measurement", "Valuation", "PAAMeasurement", "VFAMeasurement",
    "ReinsuranceMeasurement", "Report", "StochasticResult", "TVOGResult",
    "PeriodMovement", "Reconciliation", "PAAPeriodMovement", "PAAReconciliation",
    "VFAPeriodMovement", "VFAReconciliation",
    "read_model_points", "read_assumptions", "read_scenarios",
    "write_valuation", "value_file",
    "load_sample_model_points", "load_sample_assumptions",
    "solve_premium",
    "plot_liability", "plot_cashflows", "plot_csm_runoff",
    "plot_risk_adjustment", "plot_analysis_of_change", "plot_stochastic",
    "DEATH", "RiderRate", "RISK_MORTALITY", "RISK_MORBIDITY",
    "STATE_ACTIVE", "STATE_WAIVER", "STATE_PAIDUP",
    "StateModel", "State", "Transition", "WAIVER_MODEL", "STATE_MODELS",
]
