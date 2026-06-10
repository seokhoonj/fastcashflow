"""Reinsurance-held namespace -- ``fcf.reinsurance.*``.

Measurement of a reinsurance contract held (出再) over a direct portfolio.
``measure`` takes the direct portfolio, its basis and a cession and returns
the reinsurance asset/liability (BEL/RA/CSM), measured with general-model
mechanics.
"""
from fastcashflow._reinsurance import (
    QuotaShare,
    ReinsuranceMeasurement,
    measure_reinsurance as measure,
    measure_reinsurance_inforce as measure_inforce,
)
from fastcashflow.trace import show_trace_reinsurance as trace

__all__ = ["measure", "measure_inforce", "trace", "QuotaShare",
           "ReinsuranceMeasurement"]
