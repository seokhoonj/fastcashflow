"""Reinsurance-held namespace -- ``fcf.reinsurance.*``.

Measurement of a reinsurance contract held over a direct portfolio.
``measure`` takes the direct portfolio, its basis and a cession and returns
the reinsurance asset/liability (BEL/RA/CSM), measured with general-model
mechanics.
"""
from fastcashflow._reinsurance import (
    QuotaShare,
    Aggregate,
    InforceAggregate,
    Measurement,
    measure_reinsurance as measure,
    measure_reinsurance_aggregate as measure_aggregate,
    measure_reinsurance_inforce as measure_inforce,
    measure_reinsurance_inforce_aggregate as measure_inforce_aggregate,
    measure_reinsurance_stream as measure_stream,
    settle_reinsurance as settle,
    settle_reinsurance_aggregate as settle_aggregate,
    settle_reinsurance_stream as settle_stream,
)
from fastcashflow.movement import (
    ReinsuranceSettlementMovement,
    ReinsuranceSettlementReconciliation,
    ReinsuranceSettlementAggregate,
    ReinsurancePeriodMovement,
    ReinsuranceReconciliation,
)
from fastcashflow.report import ReinsuranceReport
from fastcashflow.trace import (
    show_trace_reinsurance as trace,
    show_trace_diff_reinsurance as trace_diff,
)

__all__ = ["measure", "measure_aggregate", "measure_inforce",
           "measure_inforce_aggregate", "measure_stream",
           "settle", "settle_aggregate", "settle_stream",
           "trace", "trace_diff", "QuotaShare", "Measurement",
           "Aggregate", "InforceAggregate",
           "ReinsuranceSettlementMovement",
           "ReinsuranceSettlementReconciliation",
           "ReinsuranceSettlementAggregate",
           # result types (produced by reinsurance.measure / roll_forward / report)
           "ReinsurancePeriodMovement", "ReinsuranceReconciliation",
           "ReinsuranceReport"]
