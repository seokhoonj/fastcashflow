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
from fastcashflow.closing import (
    ClosePackage, assemble_finance, assemble_service_result, assemble_sofp, close,
)
from fastcashflow.coverage import CalculationMethod, RISK_MORBIDITY, RISK_MORTALITY
from fastcashflow.disclosure import (
    line_metadata, reconciliation_to_frame, write_close_pack, write_reconciliation,
)
from fastcashflow._measurement.gmm import clear_codegen_cache
from fastcashflow.grouping import group, group_of_contracts
from fastcashflow.compression import compress, CompressionResult
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
from fastcashflow.model_points import (
    NO_GUARANTEE_RATE,
    STATE_ACTIVE,
    STATE_PAIDUP,
    STATE_WAIVER,
    InforceState,
    ModelPoints,
    align_inforce_state,
    apply_inforce_state,
)
from fastcashflow.movement import reconcile, roll_forward
from fastcashflow.plots import (
    plot_analysis_of_change,
    plot_cashflows,
    plot_csm_runoff,
    plot_liability,
    plot_risk_adjustment,
    plot_stochastic,
)
from fastcashflow.pricing import solve_premium, interest_guarantee_tvog
from fastcashflow.embedded_value import EmbeddedValue, embedded_value
# Asset / solvency surface is namespace-only (fcf.solvency / fcf.alm / fcf.assets);
# fcf.solvency is the merged facade over the _solvency (engine) + _solvency_assessment
# (assembly) impl modules. No flat re-exports -- reach members via their namespace.
from fastcashflow import solvency
from fastcashflow import alm
from fastcashflow import assets
from fastcashflow.report import (
    DynamicAssessmentReport, Report, report,
)
from fastcashflow.projection import Cashflows, AccountTrajectory
from fastcashflow.profit import ProfitSignature
from fastcashflow.state_model import (
    STATE_MODELS,
    State,
    StateModel,
    Transition,
)
from fastcashflow._measurement.stochastic import StochasticResult
from fastcashflow.transition import transition
from fastcashflow.tvog import TVOGResult
from fastcashflow.portfolio import settle_group_of_contracts
from fastcashflow import (  # namespaces
    curves, gmm, paa, portfolio, projection, reinsurance, samples, vfa, esg,
)

__version__ = "0.1.0.dev1"
__all__ = [
    # measurement-model namespaces -- the headline entry points live here
    # (e.g. ``fastcashflow.gmm.measure``, ``fastcashflow.samples.basis``).
    "curves", "gmm", "paa", "vfa", "reinsurance",
    "samples", "esg", "projection",
    "Basis", "BasisRouter", "ModelPoints", "Cashflows", "AccountTrajectory",
    "clear_codegen_cache",
    "report", "roll_forward", "reconcile", "group", "group_of_contracts",
    "compress", "CompressionResult",
    "transition",
    "close", "ClosePackage", "assemble_sofp", "assemble_finance",
    "assemble_service_result", "reconciliation_to_frame", "line_metadata",
    "write_reconciliation", "write_close_pack",
    # result types now live in their producing namespace (fcf.gmm.Measurement,
    # fcf.vfa.Measurement, fcf.reinsurance.Report, fcf.esg.*, ...);
    # only genuinely shared / universal result types stay flat below.
    "Report", "DynamicAssessmentReport",
    "StochasticResult", "TVOGResult",
    "settle_group_of_contracts",
    "ProfitSignature",
    "read_model_points", "read_vfa_model_points", "read_basis", "read_scenarios",
    "read_inforce_state", "read_inforce_policies",
    "apply_inforce_state", "align_inforce_state", "InforceState",
    "write_measurement",
    "sample_data_dir",
    "describe_basis",
    "solve_premium", "interest_guarantee_tvog",
    "embedded_value", "EmbeddedValue",
    # asset / solvency: namespace-only (members under fcf.solvency / fcf.alm / fcf.assets)
    "solvency", "alm", "assets",
    "plot_liability", "plot_cashflows", "plot_csm_runoff",
    "plot_risk_adjustment", "plot_analysis_of_change", "plot_stochastic",
    "CalculationMethod", "CoverageRate", "ExpenseItem", "EXPENSE_BASES",
    "RA_METHODS", "SURRENDER_VALUE_BASES",
    "derive_expense_components",
    "RISK_MORTALITY", "RISK_MORBIDITY",
    "STATE_ACTIVE", "STATE_WAIVER", "STATE_PAIDUP", "NO_GUARANTEE_RATE",
    "StateModel", "State", "Transition", "STATE_MODELS",
]
