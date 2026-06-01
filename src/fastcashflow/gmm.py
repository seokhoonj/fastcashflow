"""GMM (General Measurement Model) namespace -- ``fcf.gmm.*``.

The default IFRS 17 measurement model. ``measure`` is the single entry point
(``full=True`` for trajectories, ``full=False`` for the fused headline-only
fast path); ``stochastic`` runs it across economic scenarios for the
liability distribution; ``measure_stream`` is the out-of-core variant;
``trace`` walks one model point's measurement as a tree.
"""
from fastcashflow.engine import measure
from fastcashflow.io import measure_stream
from fastcashflow.stochastic import value_stochastic as stochastic
from fastcashflow.trace import (
    show_bel_step as bel_step,
    show_csm_step as csm_step,
    show_trace as trace,
    show_trace_diff as trace_diff,
)

__all__ = ["measure", "measure_stream", "stochastic", "trace", "trace_diff", "bel_step", "csm_step"]
