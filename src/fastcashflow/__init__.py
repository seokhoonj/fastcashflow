"""fastcashflow -- fast IFRS 17 GMM cash flow projection engine.

Two entry points:

* :func:`value` -- fast, fused valuation (BEL, RA, CSM per model point).
* :func:`run`   -- detailed projection with full cash flow / CSM trajectories.

Conventional import alias::

    import fastcashflow as fcf

`fcf` also reads as Fulfilment Cash Flows (IFRS 17: BEL + RA) -- the very
quantity this engine computes.
"""
from fastcashflow.assumptions import Assumptions
from fastcashflow.engine import GMMResult, Valuation, run, value
from fastcashflow.io import read_model_points, value_file, write_valuation
from fastcashflow.modelpoint import ModelPointSet

__version__ = "0.0.1"
__all__ = [
    "Assumptions", "ModelPointSet", "run", "value", "GMMResult", "Valuation",
    "read_model_points", "write_valuation", "value_file",
]
