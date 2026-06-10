"""Reinsurance-held namespace -- ``fcf.reinsurance.*``.

Measurement of a reinsurance contract held (出再) over a direct portfolio.
``measure`` takes the direct portfolio, its basis and a cession and returns
the reinsurance asset/liability (BEL/RA/CSM), measured with general-model
mechanics.
"""
from fastcashflow._reinsurance import (
    QuotaShare,
    ReinsuranceAggregate,
    ReinsuranceMeasurement,
    measure_reinsurance as measure,
    measure_reinsurance_aggregate as measure_aggregate,
    measure_reinsurance_inforce as measure_inforce,
    measure_reinsurance_stream as measure_stream,
)
from fastcashflow.trace import (
    show_trace_reinsurance as trace,
    show_trace_diff_reinsurance as trace_diff,
)

__all__ = ["measure", "measure_aggregate", "measure_inforce", "measure_stream",
           "trace", "trace_diff", "QuotaShare", "ReinsuranceMeasurement",
           "ReinsuranceAggregate"]
