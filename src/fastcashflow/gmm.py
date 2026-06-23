"""GMM (General Measurement Model) namespace -- ``fcf.gmm.*``.

The default IFRS 17 measurement model. ``measure`` is the single entry point
for new-business inception (``full=True`` for trajectories, ``full=False``
for the fused headline-only fast path); ``settle`` is the paragraph-44
settlement (period-close) entry point -- the opening -> closing movement of
an in-force book; ``settle_aggregate`` is its bounded-memory portfolio-total
variant; ``measure_inforce`` is the in-force diagnostic / runoff
projector (valuation-date BEL / RA with a prior-CSM carry, no unlocking);
``stochastic`` runs measure across economic scenarios for the liability
distribution; ``measure_aggregate`` is the bounded-memory portfolio-aggregate
``full=True`` view for books too large to hold every trajectory;
``measure_stream`` is the out-of-core variant; ``trace`` walks one model
point's measurement as a tree.
"""
from fastcashflow.engine import (
    measure, measure_aggregate, measure_inforce, settle, settle_aggregate,
    recognition_schedule, CSMRecognitionSchedule,
    Measurement, Aggregate, CurrentEstimate, PeriodMovement, Reconciliation,
    SettlementMovement)
from fastcashflow.movement import (
    GMMSettlementReconciliation, GMMSettlementAggregate)
from fastcashflow.io import measure_stream, settle_stream
from fastcashflow.pricing import interest_guarantee_tvog
from fastcashflow.stochastic import measure_stochastic as stochastic
from fastcashflow.trace import (
    show_trace_bel_step as trace_bel_step,
    show_trace_csm_step as trace_csm_step,
    show_trace as trace,
    show_trace_diff as trace_diff,
)
from fastcashflow.alm import (
    liability_duration, liability_dv01, key_rate_dv01s,
    net_liability_cashflows)

__all__ = ["measure", "measure_aggregate", "measure_inforce",
           "measure_stream", "settle", "settle_aggregate", "settle_stream",
           "recognition_schedule", "CSMRecognitionSchedule",
           "stochastic", "interest_guarantee_tvog",
           "trace", "trace_diff", "trace_bel_step", "trace_csm_step",
           "liability_duration", "liability_dv01", "key_rate_dv01s",
           "net_liability_cashflows",
           # result types (produced by gmm.measure / settle / roll_forward)
           "Measurement", "Aggregate", "CurrentEstimate",
           "SettlementMovement", "GMMSettlementReconciliation",
           "GMMSettlementAggregate", "PeriodMovement", "Reconciliation"]
