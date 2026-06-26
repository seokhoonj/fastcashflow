"""VFA (Variable Fee Approach) namespace -- ``fcf.vfa.*``.

The direct-participation (account-value) model. ``measure`` returns the
account-value/BEL/RA/CSM measurement (the guarantee time value folded in);
``settle`` is the paragraph-45 settlement (period-close) entry point -- the
opening -> closing movement of an in-force book, the CSM remeasured for the
entity's share of the underlying items and the future-service changes;
``settle_aggregate`` is its bounded-memory portfolio-total variant and
``settle_stream`` the out-of-core variant; ``measure_inforce`` is the
in-force diagnostic / runoff projector, valuing an in-force book at its
valuation date from the observed fund value (with a carry-only CSM, no
paragraph-45 remeasurement); ``measure_aggregate`` is the bounded-memory
portfolio-aggregate view for books too large to hold every trajectory and
``measure_stream`` its out-of-core variant; ``tvog`` is the standalone
time-value analysis of the crediting-rate guarantee over return scenarios
(the GMDB / GMAB floor time value lives in
``measure(..., return_scenarios).time_value``); ``trace`` walks one model
point's VFA measurement.
"""
from fastcashflow._measurement.vfa import (
    Measurement, Aggregate, PeriodMovement, Reconciliation, SettlementMovement,
    SettlementReconciliation, SettlementAggregate, GoCSettlement, CSM_BASES,
    GuaranteeTVOG,
    measure, measure_aggregate,
    measure_inforce, measure_stream, settle, settle_aggregate,
    settle_stream, recognition_schedule, guarantee_tvog,
    moneyness_lapse_multiplier, moneyness_lapse_scale,
    measure_stochastic as stochastic)
from fastcashflow._measurement.tvog import measure_tvog as tvog
from fastcashflow._trace.vfa import (
    trace, trace_diff)
from fastcashflow.alm._vfa import (
    liability_duration, liability_dv01, net_liability_cashflows)
# VFA-specific solvency / asset-liability tools -- the sole home is fcf.vfa.*
# (the symmetric counterpart to fcf.vfa.measure). Impl lives in solvency._vfa
# (the VFA bodies) / assets (cashflow gap); the merged DynamicAssessment result
# type is owned by the solvency assembly.
from fastcashflow.solvency._vfa import (
    required_capital, equity_scr, interest_scr, assess,
    interaction_loss, assess_dynamic, assess_stochastic)
from fastcashflow.assets import _vfa_cashflow_gap as cashflow_gap
from fastcashflow.solvency._assessment import DynamicAssessment

__all__ = ["measure", "measure_aggregate", "measure_inforce",
           "measure_stream", "settle", "settle_aggregate", "settle_stream",
           "recognition_schedule", "tvog", "guarantee_tvog", "trace",
           "trace_diff", "CSM_BASES", "Measurement", "GuaranteeTVOG",
           "SettlementMovement", "moneyness_lapse_multiplier",
           "moneyness_lapse_scale", "stochastic",
           "liability_duration", "liability_dv01", "net_liability_cashflows",
           "required_capital", "equity_scr", "interest_scr", "cashflow_gap",
           "assess", "interaction_loss", "assess_stochastic",
           # result types (produced by vfa.measure / settle / roll_forward)
           "Aggregate", "PeriodMovement", "Reconciliation",
           "SettlementReconciliation", "SettlementAggregate", "GoCSettlement",
           "DynamicAssessment", "assess_dynamic"]
