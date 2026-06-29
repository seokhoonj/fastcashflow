"""Reporting layer: the period close (statement of financial position).

close() assembles the close pack from a reporting period's settlement
reconciliations. The keystone is the SoFP carrying-amount split (LRC excluding
loss component / loss component / liability for incurred claims), which is
model-specific: the loss component is a sub-ledger within the LRC for GMM/VFA but
additive on top of the unearned-premium LRC for PAA, and reinsurance held is an
asset that nets against the issued liability.

The reconciliations are built directly with chosen round numbers so the SoFP
arithmetic is hand-checked in isolation from the settle pipeline.
"""
import dataclasses

import numpy as np
import polars as pl
import pytest

from fastcashflow.reporting.closing import (
    ClosePackage, assemble_sofp, assemble_finance, assemble_service_result, close,
    _COMP_LRC, _COMP_LC, _COMP_LIC, _COMP_TOTAL, _FINANCE_TOTAL,
    _KIND_ISSUED, _KIND_REINSURANCE, _KIND_NET)
import fastcashflow as fcf
from fastcashflow.reporting.report import Report


def _build(cls, **over):
    """Construct a reconciliation with every float field zero unless overridden
    -- so a test states only the balances it reasons about."""
    kw = {}
    for f in dataclasses.fields(cls):
        if f.name in over:
            kw[f.name] = over[f.name]
        elif f.name == "period_months":
            kw[f.name] = 12
        elif f.name == "revenue_basis":
            kw[f.name] = "time"
        else:
            kw[f.name] = 0.0
    return cls(**kw)


def _cell(df, kind, component, col="closing"):
    row = df.filter((pl.col("kind") == kind) & (pl.col("component") == component))
    assert row.height == 1, f"{kind}/{component} not emitted once"
    return row[col][0]


def test_gmm_sofp_loss_component_is_within_the_lrc():
    """GMM: LRC = BEL + RA + CSM and LRC-excl-LC = LRC - loss component."""
    recon = _build(
        fcf.gmm.SettlementReconciliation,
        bel_closing=700.0, ra_closing=200.0, csm_closing=100.0,
        loss_component_closing=150.0, lic_closing=300.0,
        bel_opening=650.0, ra_opening=180.0, csm_opening=120.0,
        loss_component_opening=140.0, lic_opening=250.0)
    df = assemble_sofp([recon])
    # LRC total = 700 + 200 + 100 = 1000; LRC-excl-LC = 1000 - 150 = 850
    assert _cell(df, _KIND_ISSUED, _COMP_LRC) == pytest.approx(850.0)
    assert _cell(df, _KIND_ISSUED, _COMP_LC) == pytest.approx(150.0)
    assert _cell(df, _KIND_ISSUED, _COMP_LIC) == pytest.approx(300.0)
    # carrying amount = 850 + 150 + 300 = 1300 (== LRC 1000 + LIC 300)
    assert _cell(df, _KIND_ISSUED, _COMP_TOTAL) == pytest.approx(1300.0)
    # opening LRC = 650 + 180 + 120 = 950; excl-LC = 950 - 140 = 810
    assert _cell(df, _KIND_ISSUED, _COMP_LRC, "opening") == pytest.approx(810.0)
    assert _cell(df, _KIND_ISSUED, _COMP_TOTAL, "opening") == pytest.approx(
        950.0 + 250.0)


def test_paa_sofp_loss_component_is_additive():
    """PAA: LRC-excl-LC = lrc balance; the onerous loss is additive on top."""
    recon = _build(
        fcf.paa.SettlementReconciliation,
        lrc_closing=400.0, loss_component_closing=50.0, lic_closing=120.0,
        lrc_opening=500.0, loss_component_opening=0.0, lic_opening=100.0)
    df = assemble_sofp([recon])
    assert _cell(df, _KIND_ISSUED, _COMP_LRC) == pytest.approx(400.0)
    assert _cell(df, _KIND_ISSUED, _COMP_LC) == pytest.approx(50.0)
    assert _cell(df, _KIND_ISSUED, _COMP_LIC) == pytest.approx(120.0)
    # carrying = 400 + 50 + 120 = 570 (LRC including LC is 450, plus LIC 120)
    assert _cell(df, _KIND_ISSUED, _COMP_TOTAL) == pytest.approx(570.0)


def test_reinsurance_is_an_asset_that_nets_against_issued():
    """Reinsurance held: an asset (BEL+RA+CSM), no LC / no LIC, netted off."""
    issued = _build(
        fcf.gmm.SettlementReconciliation,
        bel_closing=700.0, ra_closing=200.0, csm_closing=100.0,
        loss_component_closing=0.0, lic_closing=300.0)
    held = _build(
        fcf.reinsurance.SettlementReconciliation,
        bel_closing=-250.0, ra_closing=40.0, csm_closing=60.0)
    df = assemble_sofp([issued, held])
    # reinsurance asset for remaining coverage = -250 + 40 + 60 = -150
    assert _cell(df, _KIND_REINSURANCE, _COMP_LRC) == pytest.approx(-150.0)
    assert _cell(df, _KIND_REINSURANCE, _COMP_LC) == pytest.approx(0.0)
    assert _cell(df, _KIND_REINSURANCE, _COMP_LIC) == pytest.approx(0.0)
    assert _cell(df, _KIND_REINSURANCE, _COMP_TOTAL) == pytest.approx(-150.0)
    # net = issued + reinsurance held, component by component (one signed frame:
    # the -150 reinsurance recoverable is added in, reducing the net).
    # issued: LRC 1000, LC 0, LIC 300, total 1300
    assert _cell(df, _KIND_NET, _COMP_LRC) == pytest.approx(1000.0 + (-150.0))
    assert _cell(df, _KIND_NET, _COMP_LIC) == pytest.approx(300.0 + 0.0)
    assert _cell(df, _KIND_NET, _COMP_TOTAL) == pytest.approx(1300.0 + (-150.0))


def test_issued_aggregates_across_models():
    """GMM + VFA + PAA all land in the issued kind and sum component-wise."""
    gmm = _build(fcf.gmm.SettlementReconciliation,
                 bel_closing=100.0, ra_closing=0.0, csm_closing=0.0,
                 lic_closing=10.0)
    vfa = _build(fcf.vfa.SettlementReconciliation,
                 bel_closing=200.0, ra_closing=0.0, csm_closing=0.0,
                 lic_closing=20.0)
    paa = _build(fcf.paa.SettlementReconciliation,
                 lrc_closing=300.0, lic_closing=30.0)
    df = assemble_sofp([gmm, vfa, paa])
    assert _cell(df, _KIND_ISSUED, _COMP_LRC) == pytest.approx(600.0)
    assert _cell(df, _KIND_ISSUED, _COMP_LIC) == pytest.approx(60.0)
    assert _cell(df, _KIND_ISSUED, _COMP_TOTAL) == pytest.approx(660.0)


def test_every_row_foots_opening_plus_change_equals_closing():
    recon = _build(
        fcf.gmm.SettlementReconciliation,
        bel_closing=700.0, ra_closing=200.0, csm_closing=100.0,
        loss_component_closing=150.0, lic_closing=300.0,
        bel_opening=650.0, ra_opening=180.0, csm_opening=120.0,
        loss_component_opening=140.0, lic_opening=250.0)
    df = assemble_sofp([recon])
    foots = df.with_columns(
        (pl.col("opening") + pl.col("change") - pl.col("closing")).abs().alias("err"))
    assert foots["err"].max() == pytest.approx(0.0, abs=1e-9)
    # the Total row of each kind is the sum of its three components
    for kind in (_KIND_ISSUED, _KIND_REINSURANCE, _KIND_NET):
        parts = sum(_cell(df, kind, c) for c in (_COMP_LRC, _COMP_LC, _COMP_LIC))
        assert _cell(df, kind, _COMP_TOTAL) == pytest.approx(parts)


def _fcell(df, kind, line):
    row = df.filter((pl.col("kind") == kind) & (pl.col("line") == line))
    assert row.height == 1, f"{kind}/{line} not emitted once"
    return row["amount"][0]


def test_finance_sums_sources_and_keeps_loss_finance_a_memo():
    """The five finance sources sum to the total; loss_component_finance is a
    memo (a share of BEL finance), excluded from the total."""
    recon = _build(
        fcf.gmm.SettlementReconciliation,
        bel_interest=10.0, ra_interest=2.0, csm_accretion=5.0,
        lic_finance=1.0, finance_wedge=3.0, loss_component_finance=4.0)
    df = assemble_finance([recon])
    # total = 10 + 2 + 5 + 1 + 3 = 21 (finance_wedge included, B97(a))
    assert _fcell(df, _KIND_ISSUED, _FINANCE_TOTAL) == pytest.approx(21.0)
    assert _fcell(df, _KIND_ISSUED, "Locked-in rate adjustment") == pytest.approx(3.0)
    # the memo is reported but NOT part of the total
    memo = df.filter((pl.col("kind") == _KIND_ISSUED)
                     & (pl.col("line") == "Loss component finance"))
    assert bool(memo["is_memo"][0]) is True
    assert memo["amount"][0] == pytest.approx(4.0)


def test_finance_paa_has_only_lic_finance():
    """PAA holds the LRC undiscounted: its only finance line is the LIC unwind."""
    recon = _build(fcf.paa.SettlementReconciliation, lic_finance=7.0)
    df = assemble_finance([recon])
    assert _fcell(df, _KIND_ISSUED, "LIC finance") == pytest.approx(7.0)
    assert _fcell(df, _KIND_ISSUED, "BEL finance") == pytest.approx(0.0)
    assert _fcell(df, _KIND_ISSUED, _FINANCE_TOTAL) == pytest.approx(7.0)


def test_finance_reinsurance_nets_against_issued():
    issued = _build(fcf.gmm.SettlementReconciliation,
                    bel_interest=10.0, ra_interest=2.0, csm_accretion=5.0,
                    lic_finance=1.0, finance_wedge=3.0)
    held = _build(fcf.reinsurance.SettlementReconciliation,
                  bel_interest=-1.0, ra_interest=0.5, csm_accretion=1.0,
                  finance_wedge=0.5)
    df = assemble_finance([issued, held])
    reins_total = -1.0 + 0.5 + 1.0 + 0.0 + 0.5
    assert _fcell(df, _KIND_REINSURANCE, _FINANCE_TOTAL) == pytest.approx(reins_total)
    # reinsurance held has no LIC block -> zero finance there
    assert _fcell(df, _KIND_REINSURANCE, "LIC finance") == pytest.approx(0.0)
    # net finance = issued + reinsurance held in the one signed frame
    assert _fcell(df, _KIND_NET, _FINANCE_TOTAL) == pytest.approx(21.0 + reins_total)


def test_close_packages_sofp_finance_and_reconciliation_detail():
    gmm = _build(fcf.gmm.SettlementReconciliation, bel_closing=100.0, lic_closing=10.0,
                 bel_interest=4.0)
    held = _build(fcf.reinsurance.SettlementReconciliation, bel_closing=-30.0,
                  bel_interest=-1.0)
    pack = close([gmm, held], group_ids=["GoC-1", "RE-1"])
    assert isinstance(pack, ClosePackage)
    assert pack.period_months == 12
    assert set(pack.to_frames()) == {"sofp", "finance", "reconciliation"}
    assert _fcell(pack.finance, _KIND_NET, _FINANCE_TOTAL) == pytest.approx(4.0 + (-1.0))
    # the reconciliation detail is stamped with the group ids and both models
    recon = pack.reconciliation
    assert set(recon["group_id"].unique().to_list()) == {"GoC-1", "RE-1"}
    assert set(recon["model"].unique().to_list()) == {"gmm", "reinsurance"}
    assert str(pack).startswith("IFRS 17 close pack")


def _report(revenue, expense, **over):
    revenue = np.asarray(revenue, dtype=float)
    expense = np.asarray(expense, dtype=float)
    z = np.zeros_like(revenue)
    fields = dict(
        insurance_revenue=revenue, insurance_service_expense=expense,
        insurance_service_result=revenue - expense,
        insurance_finance_expense=z, bel_finance_expense=z, ra_finance_expense=z,
        csm_finance_expense=z, loss_component=np.zeros(revenue.shape[0]),
        csm_opening=z, csm_accretion=z, csm_release=z, csm_closing=z)
    fields.update(over)
    return Report(**fields)


def _reins_report(premium, recovered, **over):
    premium = np.asarray(premium, dtype=float)
    recovered = np.asarray(recovered, dtype=float)
    z = np.zeros_like(premium)
    fields = dict(
        reinsurance_premium_allocated=premium, amounts_recovered=recovered,
        reinsurance_service_result=z, ra_release=z, reinsurance_finance_expense=z,
        bel_finance_expense=z, ra_finance_expense=z, csm_finance_expense=z,
        csm_opening=z, csm_accretion=z, csm_release=z, csm_closing=z)
    fields.update(over)
    return fcf.reinsurance.Report(**fields)


def _scell(df, kind, line, period):
    row = df.filter((pl.col("kind") == kind) & (pl.col("line") == line)
                    & (pl.col("period_index") == period))
    assert row.height == 1, f"{kind}/{line}/p{period} not emitted once"
    return row["amount"][0]


def test_service_result_buckets_issued_revenue_expense_result():
    # 1 MP, 24 months, period_months=12 -> 2 periods
    revenue = np.array([[10.0] * 12 + [20.0] * 12])
    expense = np.array([[4.0] * 12 + [5.0] * 12])
    df = assemble_service_result([_report(revenue, expense)], period_months=12)
    assert _scell(df, _KIND_ISSUED, "Insurance revenue", 0) == pytest.approx(120.0)
    assert _scell(df, _KIND_ISSUED, "Insurance revenue", 1) == pytest.approx(240.0)
    assert _scell(df, _KIND_ISSUED, "Insurance service expense", 1) == pytest.approx(60.0)
    assert _scell(df, _KIND_ISSUED, "Insurance service result", 1) == pytest.approx(180.0)


def test_service_result_sums_across_reports_and_pads_periods():
    short = _report(np.array([[1.0] * 12]), np.zeros((1, 12)))            # 1 period
    longr = _report(np.array([[2.0] * 24]), np.zeros((1, 24)))           # 2 periods
    df = assemble_service_result([short, longr], period_months=12)
    # period 0: 12 + 24 = 36; period 1: only the long report contributes 24
    assert _scell(df, _KIND_ISSUED, "Insurance revenue", 0) == pytest.approx(36.0)
    assert _scell(df, _KIND_ISSUED, "Insurance revenue", 1) == pytest.approx(24.0)


def test_service_result_reinsurance_net_is_recovered_less_premium():
    premium = np.array([[3.0] * 12])
    recovered = np.array([[10.0] * 12])
    df = assemble_service_result([_reins_report(premium, recovered)], period_months=12)
    assert _scell(df, _KIND_REINSURANCE, "Reinsurance premium", 0) == pytest.approx(36.0)
    assert _scell(df, _KIND_REINSURANCE, "Amounts recovered", 0) == pytest.approx(120.0)
    assert _scell(df, _KIND_REINSURANCE, "Net reinsurance result", 0) == pytest.approx(84.0)


def test_service_result_rejects_non_report():
    recon = _build(fcf.gmm.SettlementReconciliation, bel_closing=1.0)
    with pytest.raises(TypeError, match="Report"):
        assemble_service_result([recon])


def test_close_with_reports_adds_service_result():
    gmm = _build(fcf.gmm.SettlementReconciliation, bel_closing=100.0)
    rep = _report(np.array([[10.0] * 12]), np.array([[4.0] * 12]))
    pack = close([gmm], reports=[rep])
    assert set(pack.to_frames()) == {"sofp", "finance", "service_result", "reconciliation"}
    assert pack.service_result is not None
    assert _scell(pack.service_result, _KIND_ISSUED, "Insurance revenue", 0) == pytest.approx(120.0)


def test_close_without_reports_omits_service_result():
    gmm = _build(fcf.gmm.SettlementReconciliation, bel_closing=100.0)
    pack = close([gmm])
    assert pack.service_result is None
    assert "service_result" not in pack.to_frames()


def test_close_rejects_mixed_periods():
    a = _build(fcf.gmm.SettlementReconciliation, period_months=12, bel_closing=100.0)
    b = _build(fcf.gmm.SettlementReconciliation, period_months=6, bel_closing=50.0)
    with pytest.raises(ValueError, match="period_months"):
        close([a, b])


def test_close_rejects_mismatched_group_ids():
    a = _build(fcf.gmm.SettlementReconciliation, bel_closing=100.0)
    with pytest.raises(ValueError, match="group_ids"):
        close([a], group_ids=["x", "y"])


def test_close_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        close([])
