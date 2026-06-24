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
time-value analysis of the credited-rate guarantee over return scenarios
(the GMDB / GMAB floor time value lives in
``measure(..., return_scenarios).time_value``); ``trace`` walks one model
point's VFA measurement.
"""
from fastcashflow._vfa import (
    Measurement, Aggregate, PeriodMovement, Reconciliation, SettlementMovement,
    SettlementReconciliation, SettlementAggregate, GoCSettlement,
    measure_vfa as measure, measure_aggregate,
    measure_inforce, measure_stream, settle, settle_aggregate,
    settle_stream, recognition_schedule, CSM_BASES,
    GuaranteeTVOG, guarantee_tvog, moneyness_lapse_multiplier,
    moneyness_lapse_scale, measure_vfa_stochastic as stochastic)
from fastcashflow.tvog import measure_tvog as tvog
from fastcashflow.trace import (
    show_trace_vfa as trace, show_trace_diff_vfa as trace_diff)
from fastcashflow.alm import (
    vfa_liability_duration as liability_duration,
    vfa_liability_dv01 as liability_dv01,
    vfa_net_liability_cashflows as net_liability_cashflows)
# VFA-specific solvency / asset-liability tools -- the sole home is fcf.vfa.*
# (the symmetric counterpart to fcf.vfa.measure). Impl lives in _vfa_solvency
# (the VFA bodies) / assets (cashflow gap); the merged DynamicSolvency result
# type is owned by the solvency assembly.
from fastcashflow._vfa_solvency import (
    vfa_required_capital as required_capital,
    vfa_equity_scr as equity_scr,
    vfa_interest_scr as interest_scr,
    vfa_assess_solvency as assess_solvency,
    vfa_interaction_loss as interaction_loss,
    dynamic_solvency_vfa as dynamic_solvency,
    stochastic_solvency_vfa as stochastic_solvency)
from fastcashflow.assets import vfa_cashflow_gap as cashflow_gap
from fastcashflow._solvency_assessment import DynamicSolvency

__all__ = ["measure", "measure_aggregate", "measure_inforce",
           "measure_stream", "settle", "settle_aggregate", "settle_stream",
           "recognition_schedule", "tvog", "guarantee_tvog", "trace",
           "trace_diff", "CSM_BASES", "Measurement", "GuaranteeTVOG",
           "SettlementMovement", "moneyness_lapse_multiplier",
           "moneyness_lapse_scale", "stochastic",
           "liability_duration", "liability_dv01", "net_liability_cashflows",
           "required_capital", "equity_scr", "interest_scr", "cashflow_gap",
           "assess_solvency", "interaction_loss", "stochastic_solvency",
           # result types (produced by vfa.measure / settle / roll_forward)
           "Aggregate", "PeriodMovement", "Reconciliation",
           "SettlementReconciliation", "SettlementAggregate", "GoCSettlement",
           "DynamicSolvency", "dynamic_solvency"]
