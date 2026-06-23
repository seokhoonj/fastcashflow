"""PAA (Premium Allocation Approach) namespace -- ``fcf.paa.*``.

The short-duration simplified model. ``measure`` returns the LRC/revenue
roll-forward; ``settle`` is the paragraph-55(b) period-close settlement of an
in-force book (LRC roll, Sec. 57-58 loss-component re-test, LIC movement);
``measure_aggregate`` is the bounded-memory portfolio-aggregate view for
books too large to hold every trajectory and ``measure_stream`` its
out-of-core variant; ``measure_inforce`` is the in-force diagnostic / runoff
view, valuing an in-force book's remaining LRC at its valuation date (the
PAA has no CSM, so there is no carry); ``trace`` walks one model point's PAA
measurement.
"""
from fastcashflow._paa import (
    Measurement, Aggregate, PeriodMovement, measure_paa as measure, measure_aggregate,
    measure_inforce, measure_stream, settle, settle_aggregate, settle_stream)
from fastcashflow.movement import (
    PAASettlementAggregate, PAAReconciliation,
    PAASettlementMovement, PAASettlementReconciliation)
from fastcashflow.trace import (
    show_trace_paa as trace, show_trace_diff_paa as trace_diff)

__all__ = ["measure", "measure_aggregate", "measure_inforce", "measure_stream",
           "settle", "settle_aggregate", "settle_stream",
           "trace", "trace_diff", "Measurement", "PAASettlementAggregate",
           # result types (produced by paa.measure / settle / roll_forward)
           "Aggregate", "PeriodMovement", "PAAReconciliation",
           "PAASettlementMovement", "PAASettlementReconciliation"]
