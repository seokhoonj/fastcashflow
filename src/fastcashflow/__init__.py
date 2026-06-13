"""fastcashflow -- open-source IFRS 17 GMM cash flow projection engine.

The GMM entry point is :func:`fastcashflow.gmm.measure`, selected by ``full``:

* ``measure(..., full=False)`` -- fast, fused valuation (headline BEL, RA, CSM per model point).
* ``measure(..., full=True)``  -- detailed: full cash flow and CSM trajectories.

Conventional import alias::

    import fastcashflow as fcf

`fcf` also reads as Fulfilment Cash Flows (IFRS 17: BEL + RA) -- the very
quantity this engine computes.
"""
from fastcashflow.basis import (
    Basis, BasisRouter, CoverageRate, EXPENSE_BASES, RA_METHODS,
    SURRENDER_VALUE_BASES, ExpenseItem, derive_expense_components, describe_basis,
)
from fastcashflow.coverage import CalculationMethod, RISK_MORBIDITY, RISK_MORTALITY
from fastcashflow.engine import (
    GMMAggregate, GMMMeasurement, clear_codegen_cache,
)
from fastcashflow.grouping import group, group_of_contracts
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
    NO_GUARANTEE_RATE,
    STATE_ACTIVE,
    STATE_PAIDUP,
    STATE_WAIVER,
    InforceState,
    ModelPoints,
    align_inforce_state,
    apply_inforce_state,
)
from fastcashflow.movement import (
    GMMSettlementAggregate,
    GMMSettlementMovement,
    GMMSettlementReconciliation,
    PAAPeriodMovement,
    PAAReconciliation,
    PAASettlementMovement,
    PAASettlementReconciliation,
    PeriodMovement,
    Reconciliation,
    ReinsurancePeriodMovement,
    ReinsuranceReconciliation,
    VFAPeriodMovement,
    VFAReconciliation,
    VFASettlementAggregate,
    VFASettlementMovement,
    VFASettlementReconciliation,
    reconcile,
    roll_forward,
)
from fastcashflow._paa import PAAMeasurement, PAAAggregate
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
from fastcashflow.report import ReinsuranceReport, Report, report
from fastcashflow.statemodel import (
    STATE_MODELS,
    State,
    StateModel,
    Transition,
)
from fastcashflow.stochastic import StochasticResult
from fastcashflow.transition import transition
from fastcashflow.tvog import TVOGResult
from fastcashflow._vfa import VFAMeasurement, VFAAggregate
from fastcashflow.portfolio import GoCSettlement, settle_group_of_contracts
from fastcashflow import gmm, paa, portfolio, reinsurance, samples, vfa  # namespaces

__version__ = "0.1.0.dev1"
__all__ = [
    # measurement-model namespaces -- the headline entry points live here
    # (e.g. ``fastcashflow.gmm.measure``, ``fastcashflow.samples.basis``).
    "gmm", "paa", "vfa", "reinsurance", "samples",
    "Basis", "BasisRouter", "ModelPoints", "clear_codegen_cache",
    "report", "roll_forward", "reconcile", "group", "group_of_contracts",
    "transition",
    "GMMMeasurement", "GMMAggregate", "PAAMeasurement", "PAAAggregate",
    "VFAMeasurement", "VFAAggregate",
    "ReinsuranceMeasurement", "Report", "ReinsuranceReport",
    "StochasticResult", "TVOGResult",
    "PeriodMovement", "Reconciliation", "PAAPeriodMovement", "PAAReconciliation",
    "VFAPeriodMovement", "VFAReconciliation",
    "GMMSettlementMovement", "GMMSettlementReconciliation",
    "GMMSettlementAggregate",
    "GoCSettlement", "settle_group_of_contracts",
    "PAASettlementMovement", "PAASettlementReconciliation",
    "VFASettlementMovement", "VFASettlementReconciliation",
    "VFASettlementAggregate",
    "ReinsurancePeriodMovement", "ReinsuranceReconciliation",
    "read_model_points", "read_vfa_model_points", "read_basis", "read_scenarios",
    "read_inforce_state", "read_inforce_policies",
    "apply_inforce_state", "align_inforce_state", "InforceState",
    "write_measurement",
    "sample_data_dir",
    "describe_basis",
    "solve_premium",
    "plot_liability", "plot_cashflows", "plot_csm_runoff",
    "plot_risk_adjustment", "plot_analysis_of_change", "plot_stochastic",
    "CalculationMethod", "CoverageRate", "ExpenseItem", "EXPENSE_BASES",
    "RA_METHODS", "SURRENDER_VALUE_BASES",
    "derive_expense_components",
    "RISK_MORTALITY", "RISK_MORBIDITY",
    "STATE_ACTIVE", "STATE_WAIVER", "STATE_PAIDUP", "NO_GUARANTEE_RATE",
    "StateModel", "State", "Transition", "STATE_MODELS",
]
