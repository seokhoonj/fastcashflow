"""PAA (Premium Allocation Approach) namespace -- ``fcf.paa.*``.

The short-duration simplified model. ``measure`` returns the LRC/revenue
roll-forward; ``trace`` walks one model point's PAA measurement.
"""
from fastcashflow._paa import PAAMeasurement, measure_paa as measure
from fastcashflow.trace import show_trace_paa as trace

__all__ = ["measure", "trace", "PAAMeasurement"]
