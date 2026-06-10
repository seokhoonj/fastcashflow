"""VFA (Variable Fee Approach) namespace -- ``fcf.vfa.*``.

The direct-participation (account-value) model. ``measure`` returns the
account-value/BEL/RA/CSM measurement (the guarantee time value folded in);
``measure_aggregate`` is the bounded-memory portfolio-aggregate view for books
too large to hold every trajectory; ``measure_inforce`` is the
subsequent-measurement (settlement / period-close) entry point, valuing an
in-force book at its valuation date from the observed fund value; ``tvog`` is
the standalone time-value analysis of the credited-rate guarantee over return
scenarios (the GMDB / GMAB floor time value lives in
``measure(..., return_scenarios).time_value``); ``trace`` walks one model
point's VFA measurement.
"""
from fastcashflow._vfa import (
    VFAMeasurement, measure_vfa as measure, measure_aggregate,
    measure_inforce, measure_stream, CSM_BASES)
from fastcashflow.tvog import measure_tvog as tvog
from fastcashflow.trace import (
    show_trace_vfa as trace, show_trace_diff_vfa as trace_diff)

__all__ = ["measure", "measure_aggregate", "measure_inforce",
           "measure_stream", "tvog", "trace", "trace_diff", "CSM_BASES",
           "VFAMeasurement"]
