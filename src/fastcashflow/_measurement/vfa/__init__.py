"""Private VFA implementation namespace.

Not public -- the public API is the facade module :mod:`fastcashflow.vfa`
(``fcf.vfa``). The facade and the cross-model modules (trace / grouping /
movement / ...) import the VFA engine and result types from here. Nothing in
this package imports the facade or the cross-model diagnostic modules, so it
sits at the base of the import graph: importing ``_measurement.vfa`` can never
fire the ``fcf.vfa`` facade, and an import cycle is therefore impossible.
"""
from fastcashflow._measurement.vfa.results import (
    Measurement, Aggregate, PeriodMovement, Reconciliation, SettlementMovement,
    SettlementReconciliation, SettlementAggregate, GoCSettlement, GuaranteeTVOG,
    CSM_BASES, CSM_BASIS_INITIAL, CSM_BASIS_PROJECTED_RUNOFF,
    CSM_BASIS_CARRY_ONLY, CSM_BASIS_PARAGRAPH_45, _CSM_TO_MEASUREMENT_BASIS,
    _VFA_RECON_BLOCKS, _VFA_SETTLEMENT_LINES,
    _VFA_GOC_SETTLEMENT_LINEAR, _VFA_GOC_SETTLEMENT_NONLINEAR)
from fastcashflow._measurement.vfa.engine import (
    measure, measure_stochastic, measure_aggregate, measure_inforce,
    measure_stream, settle, settle_aggregate, settle_stream,
    recognition_schedule, guarantee_tvog,
    moneyness_lapse_multiplier, moneyness_lapse_scale,
    _require_settlement_csm, _csm_loss_component_step, _project,
    _stitch_measurements, _scatter_headline)
