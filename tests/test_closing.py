"""Phase-3 reporting layer: the period close (statement of financial position).

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

import polars as pl
import pytest

from fastcashflow.closing import (
    ClosePackage, assemble_sofp, close,
    _COMP_LRC, _COMP_LC, _COMP_LIC, _COMP_TOTAL,
    _KIND_ISSUED, _KIND_REINSURANCE, _KIND_NET)
from fastcashflow.movement import (
    GMMSettlementReconciliation, PAASettlementReconciliation,
    ReinsuranceSettlementReconciliation, VFASettlementReconciliation)


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
        GMMSettlementReconciliation,
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
        PAASettlementReconciliation,
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
        GMMSettlementReconciliation,
        bel_closing=700.0, ra_closing=200.0, csm_closing=100.0,
        loss_component_closing=0.0, lic_closing=300.0)
    held = _build(
        ReinsuranceSettlementReconciliation,
        bel_closing=-250.0, ra_closing=40.0, csm_closing=60.0)
    df = assemble_sofp([issued, held])
    # reinsurance asset for remaining coverage = -250 + 40 + 60 = -150
    assert _cell(df, _KIND_REINSURANCE, _COMP_LRC) == pytest.approx(-150.0)
    assert _cell(df, _KIND_REINSURANCE, _COMP_LC) == pytest.approx(0.0)
    assert _cell(df, _KIND_REINSURANCE, _COMP_LIC) == pytest.approx(0.0)
    assert _cell(df, _KIND_REINSURANCE, _COMP_TOTAL) == pytest.approx(-150.0)
    # net = issued less reinsurance, component by component
    # issued: LRC 1000, LC 0, LIC 300, total 1300
    assert _cell(df, _KIND_NET, _COMP_LRC) == pytest.approx(1000.0 - (-150.0))
    assert _cell(df, _KIND_NET, _COMP_LIC) == pytest.approx(300.0 - 0.0)
    assert _cell(df, _KIND_NET, _COMP_TOTAL) == pytest.approx(1300.0 - (-150.0))


def test_issued_aggregates_across_models():
    """GMM + VFA + PAA all land in the issued kind and sum component-wise."""
    gmm = _build(GMMSettlementReconciliation,
                 bel_closing=100.0, ra_closing=0.0, csm_closing=0.0,
                 lic_closing=10.0)
    vfa = _build(VFASettlementReconciliation,
                 bel_closing=200.0, ra_closing=0.0, csm_closing=0.0,
                 lic_closing=20.0)
    paa = _build(PAASettlementReconciliation,
                 lrc_closing=300.0, lic_closing=30.0)
    df = assemble_sofp([gmm, vfa, paa])
    assert _cell(df, _KIND_ISSUED, _COMP_LRC) == pytest.approx(600.0)
    assert _cell(df, _KIND_ISSUED, _COMP_LIC) == pytest.approx(60.0)
    assert _cell(df, _KIND_ISSUED, _COMP_TOTAL) == pytest.approx(660.0)


def test_every_row_foots_opening_plus_change_equals_closing():
    recon = _build(
        GMMSettlementReconciliation,
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


def test_close_packages_sofp_and_reconciliation_detail():
    gmm = _build(GMMSettlementReconciliation, bel_closing=100.0, lic_closing=10.0)
    held = _build(ReinsuranceSettlementReconciliation, bel_closing=-30.0)
    pack = close([gmm, held], group_ids=["GoC-1", "RE-1"])
    assert isinstance(pack, ClosePackage)
    assert pack.period_months == 12
    assert set(pack.to_frames()) == {"sofp", "reconciliation"}
    # the reconciliation detail is stamped with the group ids and both models
    recon = pack.reconciliation
    assert set(recon["group_id"].unique().to_list()) == {"GoC-1", "RE-1"}
    assert set(recon["model"].unique().to_list()) == {"gmm", "reinsurance"}
    assert str(pack).startswith("IFRS 17 close pack")


def test_close_rejects_mixed_periods():
    a = _build(GMMSettlementReconciliation, period_months=12, bel_closing=100.0)
    b = _build(GMMSettlementReconciliation, period_months=6, bel_closing=50.0)
    with pytest.raises(ValueError, match="period_months"):
        close([a, b])


def test_close_rejects_mismatched_group_ids():
    a = _build(GMMSettlementReconciliation, bel_closing=100.0)
    with pytest.raises(ValueError, match="group_ids"):
        close([a], group_ids=["x", "y"])


def test_close_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        close([])
