"""Reinsurance namespace -- ``fcf.reinsurance.*``.

Measurement of a reinsurance contract HELD over a direct portfolio.
``measure`` takes the direct portfolio, its basis and a cession and returns
the reinsurance asset/liability (BEL/RA/CSM), measured with general-model
mechanics.

The ASSUMED / mass-lapse side lives under the ``mass_lapse`` sub-namespace
(``fcf.reinsurance.mass_lapse.*``): the cedant standard-formula capital
relief and the reinsurer's lapse-XL pricing / assumed-treaty toolkit. It
shares no machinery with held measurement (a different perspective) but is
the same reinsurance domain drawer, so it nests here rather than standing as
a separate top-level namespace.
"""
from fastcashflow._measurement.reinsurance import (
    QuotaShare,
    Aggregate,
    InforceAggregate,
    Measurement,
    PeriodMovement,
    Reconciliation,
    SettlementMovement,
    SettlementReconciliation,
    SettlementAggregate,
    Report,
    measure,
    measure_aggregate,
    measure_inforce,
    measure_inforce_aggregate,
    measure_stream,
    settle,
    settle_aggregate,
    settle_stream,
)
from fastcashflow._trace.reinsurance import (
    trace,
    trace_diff,
)
# Assumed / mass-lapse reinsurance toolkit, exposed as fcf.reinsurance.mass_lapse.
# The mass_lapse submodule imports only the private _mass_lapse
# package (never this held-measurement facade), so this binding is cycle-free.
from fastcashflow.reinsurance import mass_lapse

__all__ = ["measure", "measure_aggregate", "measure_inforce",
           "measure_inforce_aggregate", "measure_stream",
           "settle", "settle_aggregate", "settle_stream",
           "trace", "trace_diff", "QuotaShare", "Measurement",
           "Aggregate", "InforceAggregate",
           "SettlementMovement",
           "SettlementReconciliation",
           "SettlementAggregate",
           # result types (produced by reinsurance.measure / roll_forward / report)
           "PeriodMovement", "Reconciliation",
           "Report",
           # assumed / mass-lapse sub-namespace
           "mass_lapse"]
