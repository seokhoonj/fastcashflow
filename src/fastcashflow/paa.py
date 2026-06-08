"""PAA (Premium Allocation Approach) namespace -- ``fcf.paa.*``.

The short-duration simplified model. ``measure`` returns the LRC/revenue
roll-forward; ``measure_aggregate`` is the bounded-memory portfolio-aggregate
view for books too large to hold every trajectory; ``trace`` walks one model
point's PAA measurement.
"""
from fastcashflow._paa import (
    PAAMeasurement, measure_paa as measure, measure_aggregate)
from fastcashflow.trace import show_trace_paa as trace

__all__ = ["measure", "measure_aggregate", "trace", "PAAMeasurement"]
