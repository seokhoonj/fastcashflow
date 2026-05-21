"""fastcashflow -- fast IFRS 17 GMM cash flow projection engine.

Two entry points:

* :func:`value`   -- fast, fused valuation (BEL, RA, CSM per model point).
* :func:`measure` -- detailed: full cash flow and CSM trajectories.

Conventional import alias::

    import fastcashflow as fcf

`fcf` also reads as Fulfilment Cash Flows (IFRS 17: BEL + RA) -- the very
quantity this engine computes.
"""
from fastcashflow.assumptions import Assumptions
from fastcashflow.coverage import DEATH, DIAGNOSIS, INPATIENT, OUTPATIENT, SURGERY
from fastcashflow.engine import Measurement, Valuation, measure, value
from fastcashflow.io import read_model_points, value_file, write_valuation
from fastcashflow.modelpoint import ModelPointSet
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
from fastcashflow.pricing import solve_premium
from fastcashflow.reinsurance import ReinsuranceMeasurement, measure_reinsurance
from fastcashflow.report import Report, report
from fastcashflow.stochastic import StochasticResult, value_stochastic
from fastcashflow.tvog import TVOGResult, measure_tvog
from fastcashflow.vfa import VFAMeasurement, measure_vfa

__version__ = "0.0.1"
__all__ = [
    "Assumptions", "ModelPointSet", "measure", "value", "value_stochastic",
    "measure_paa", "measure_vfa", "measure_reinsurance", "measure_tvog",
    "report", "roll_forward", "reconcile",
    "Measurement", "Valuation", "PAAMeasurement", "VFAMeasurement",
    "ReinsuranceMeasurement", "Report", "StochasticResult", "TVOGResult",
    "PeriodMovement", "Reconciliation", "PAAPeriodMovement", "PAAReconciliation",
    "VFAPeriodMovement", "VFAReconciliation",
    "read_model_points", "write_valuation", "value_file", "solve_premium",
    "DEATH", "DIAGNOSIS", "INPATIENT", "OUTPATIENT", "SURGERY",
]
