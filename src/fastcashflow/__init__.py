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
from fastcashflow.pricing import solve_premium

__version__ = "0.0.1"
__all__ = [
    "Assumptions", "ModelPointSet", "measure", "value", "Measurement", "Valuation",
    "read_model_points", "write_valuation", "value_file", "solve_premium",
    "DEATH", "DIAGNOSIS", "INPATIENT", "OUTPATIENT", "SURGERY",
]
