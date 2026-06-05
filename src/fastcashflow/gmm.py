"""GMM (General Measurement Model) namespace -- ``fcf.gmm.*``.

The default IFRS 17 measurement model. ``measure`` is the single entry point
for new-business inception (``full=True`` for trajectories, ``full=False``
for the fused headline-only fast path); ``measure_inforce`` is the
subsequent-measurement (settlement / period-close) entry point, valuing an
in-force book at its valuation date with prior-CSM carry-forward;
``stochastic`` runs measure across economic scenarios for the liability
distribution; ``measure_aggregate`` is the bounded-memory portfolio-aggregate
``full=True`` view for books too large to hold every trajectory;
``measure_stream`` is the out-of-core variant; ``trace`` walks one model
point's measurement as a tree.
"""
from fastcashflow.engine import measure, measure_aggregate, measure_inforce
from fastcashflow.io import measure_stream
from fastcashflow.stochastic import measure_stochastic as stochastic
from fastcashflow.trace import (
    show_trace_bel_step as trace_bel_step,
    show_trace_csm_step as trace_csm_step,
    show_trace as trace,
    show_trace_diff as trace_diff,
)

__all__ = ["measure", "measure_aggregate", "measure_inforce", "measure_stream", "stochastic", "trace", "trace_diff", "trace_bel_step", "trace_csm_step"]
