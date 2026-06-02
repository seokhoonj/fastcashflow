"""fastcashflow -- open-source IFRS 17 GMM cash flow projection engine.

Two entry points:

* :func:`value`   -- fast, fused valuation (BEL, RA, CSM per model point).
* :func:`measure` -- detailed: full cash flow and CSM trajectories.

Conventional import alias::

    import fastcashflow as fcf

`fcf` also reads as Fulfilment Cash Flows (IFRS 17: BEL + RA) -- the very
quantity this engine computes.
"""
from fastcashflow.basis import (
    Basis, CoverageRate, EXPENSE_BASES, ExpenseItem,
    derive_expense_components, describe_basis,
)
from fastcashflow.coverage import CalculationMethod, RISK_MORBIDITY, RISK_MORTALITY
from fastcashflow.engine import GMMMeasurement, clear_codegen_cache
from fastcashflow.grouping import group
from fastcashflow.io import (
    read_basis,
    read_inforce_policies,
    read_inforce_state,
    read_model_points,
    read_scenarios,
    read_vfa_model_points,
    sample_data_dir,
    write_measurement,
)
from fastcashflow.modelpoints import (
    STATE_ACTIVE,
    STATE_PAIDUP,
    STATE_WAIVER,
    InforceState,
    ModelPoints,
    apply_inforce_state,
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
from fastcashflow._paa import PAAMeasurement
from fastcashflow.plots import (
    plot_analysis_of_change,
    plot_cashflows,
    plot_csm_runoff,
    plot_liability,
    plot_risk_adjustment,
    plot_stochastic,
)
from fastcashflow.pricing import solve_premium
from fastcashflow._reinsurance import ReinsuranceMeasurement
from fastcashflow.report import Report, report
from fastcashflow.statemodel import (
    STATE_MODELS,
    State,
    StateModel,
    Transition,
)
from fastcashflow.stochastic import StochasticResult
from fastcashflow.transition import transition
from fastcashflow.tvog import TVOGResult
from fastcashflow._vfa import VFAMeasurement
from fastcashflow import gmm, paa, reinsurance, samples, vfa  # namespaces

__version__ = "0.1.0.dev1"
__all__ = [
    "Basis", "ModelPoints", "clear_codegen_cache",
    "report", "roll_forward", "reconcile", "group", "transition",
    "GMMMeasurement", "PAAMeasurement", "VFAMeasurement",
    "ReinsuranceMeasurement", "Report", "StochasticResult", "TVOGResult",
    "PeriodMovement", "Reconciliation", "PAAPeriodMovement", "PAAReconciliation",
    "VFAPeriodMovement", "VFAReconciliation",
    "read_model_points", "read_vfa_model_points", "read_basis", "read_scenarios",
    "read_inforce_state", "read_inforce_policies",
    "apply_inforce_state", "InforceState",
    "write_measurement",
    "sample_data_dir",
    "describe_basis",
    "solve_premium",
    "plot_liability", "plot_cashflows", "plot_csm_runoff",
    "plot_risk_adjustment", "plot_analysis_of_change", "plot_stochastic",
    "CalculationMethod", "CoverageRate", "ExpenseItem", "EXPENSE_BASES",
    "derive_expense_components",
    "RISK_MORTALITY", "RISK_MORBIDITY",
    "STATE_ACTIVE", "STATE_WAIVER", "STATE_PAIDUP",
    "StateModel", "State", "Transition", "STATE_MODELS",
]
