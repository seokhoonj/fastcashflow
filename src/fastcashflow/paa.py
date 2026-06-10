"""PAA (Premium Allocation Approach) namespace -- ``fcf.paa.*``.

The short-duration simplified model. ``measure`` returns the LRC/revenue
roll-forward; ``measure_aggregate`` is the bounded-memory portfolio-aggregate
view for books too large to hold every trajectory; ``measure_inforce`` is the
subsequent-measurement (settlement / period-close) entry point, valuing an
in-force book's remaining LRC at its valuation date; ``trace`` walks one model
point's PAA measurement.
"""
from fastcashflow._paa import (
    PAAMeasurement, measure_paa as measure, measure_aggregate, measure_inforce,
    measure_stream)
from fastcashflow.trace import (
    show_trace_paa as trace, show_trace_diff_paa as trace_diff)

__all__ = ["measure", "measure_aggregate", "measure_inforce", "measure_stream",
           "trace", "trace_diff", "PAAMeasurement"]
