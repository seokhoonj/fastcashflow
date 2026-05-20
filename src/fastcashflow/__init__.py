"""fastcashflow -- fast IFRS 17 GMM cash flow projection engine.

Phase 0: single fixed-benefit protection product, deterministic projection,
BEL / RA / CSM measurement.

Conventional import alias::

    import fastcashflow as fcf

`fcf` also reads as Fulfilment Cash Flows (IFRS 17: BEL + RA) -- the very
quantity this engine computes.
"""
from fastcashflow.assumptions import Assumptions
from fastcashflow.modelpoint import ModelPointSet
from fastcashflow.engine import run, GMMResult

__version__ = "0.0.1"
__all__ = ["Assumptions", "ModelPointSet", "run", "GMMResult"]
