"""VFA (Variable Fee Approach) namespace -- ``fcf.vfa.*``.

The direct-participation (account-value) model. ``measure`` returns the
account-value/BEL/RA/CSM measurement (the guarantee time value folded in);
``tvog`` is the standalone time-value-of-guarantee analysis over return
scenarios; ``trace`` walks one model point's VFA measurement.
"""
from fastcashflow._vfa import VFAMeasurement, measure_vfa as measure
from fastcashflow.tvog import measure_tvog as tvog
from fastcashflow.trace import show_trace_vfa as trace

__all__ = ["measure", "tvog", "trace", "VFAMeasurement"]
