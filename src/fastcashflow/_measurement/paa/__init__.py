"""Private PAA implementation namespace.

Not public -- the public API is the facade module :mod:`fastcashflow.paa`
(``fcf.paa``). The facade and the cross-model modules (trace / grouping /
movement / ...) import the PAA engine and result types from here. Nothing in
this package imports the facade or the cross-model diagnostic modules, so it
sits at the base of the import graph: importing ``_measurement.paa`` can never
fire the ``fcf.paa`` facade, and an import cycle is therefore impossible.
"""
from fastcashflow._measurement.paa.results import (
    Measurement, Aggregate, PeriodMovement, Reconciliation, SettlementMovement,
    SettlementReconciliation, SettlementAggregate,
    _PAA_RECON_BLOCKS, _PAA_SETTLEMENT_LINES)
from fastcashflow._measurement.paa.engine import (
    measure, measure_aggregate, measure_inforce, measure_stream,
    settle, settle_aggregate, settle_stream,
    _require_full, _stitch_measurements, _scatter_headline)
