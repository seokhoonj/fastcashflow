"""Private GMM implementation namespace.

Not public -- the public API is the facade module :mod:`fastcashflow.gmm`
(``fcf.gmm``). The facade and the cross-model modules (trace / grouping /
movement / ...) import the GMM engine and result types from here. Nothing in
this package imports the facade or the cross-model diagnostic modules, so it
sits at the base of the import graph: importing ``_measurement.gmm`` can never
fire the ``fcf.gmm`` facade, and an import cycle is therefore impossible.
"""
from fastcashflow._measurement.gmm.results import (
    Measurement, CurrentEstimate, Aggregate, PeriodMovement, Reconciliation,
    SettlementMovement, SettlementReconciliation, SettlementAggregate,
    _GMM_RECON_BLOCKS, _GMM_SETTLEMENT_LINES, _measure_full)
from fastcashflow._measurement.gmm.engine import (
    measure, measure_aggregate, measure_inforce, settle, settle_aggregate,
    recognition_schedule, requires_full, _require_full,
    _measure_inforce_fast, _measure_inforce_full, _factorise_segments)
from fastcashflow._measurement.gmm.codegen import clear_codegen_cache
