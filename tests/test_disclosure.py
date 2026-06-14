"""Phase-3 reporting layer: the canonical tidy frame (to_frame foundation).

reconciliation_to_frame turns a settlement reconciliation into the lean canonical
disclosure frame. Two guarantees pinned here: the block spec covers EXACTLY the
reconciliation's fields (no line dropped or invented -- the single-source spine),
and the frame faithfully carries each line's amount.
"""
import numpy as np
import polars as pl

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from fastcashflow.disclosure import (
    reconciliation_to_frame, _GMM_RECON_BLOCKS, _LEAN_COLUMNS)
from fastcashflow.movement import GMMSettlementReconciliation
from conftest import PATTERNS, make_death_basis


def _gmm_recon():
    """An onerous GMM settle (LC + claims + releases nonzero), reconciled."""
    basis = make_death_basis(
        mortality_q=0.02, lapse_q=0.03, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        settlement_pattern=np.array([0.6, 0.4]))
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([36]), benefits={0: np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=PATTERNS),
        basis, full=True).cashflows.inforce[0]
    eo, ec, scale = 12, 24, 1000.0
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([36]), benefits={0: np.array([1e6])},
        count=np.array([scale * surv[ec]]), elapsed_months=np.array([ec]),
        mp_id=np.array(["P0"]), product=np.array(["A"]),
        calculation_methods=PATTERNS)
    st = InforceState(
        mp_id=np.array(["P0"]), elapsed_months=np.array([ec]),
        count=np.array([scale * surv[ec]]), prior_csm=np.array([0.0]),
        lock_in_rate=0.03, prior_count=np.array([scale * surv[eo]]),
        prior_loss_component=np.array([200_000.0]))
    mv = fcf.gmm.settle(mp, st, basis, period_months=12)
    return fcf.reconcile([mv])[0]


def _spec_fields(blocks):
    return {field for _b, lines in blocks for _l, field, _p, _m in lines}


def test_gmm_block_spec_covers_exactly_the_reconciliation_fields():
    """The disclosure spine is single-source: the GMM block spec names exactly
    the reconciliation's float fields -- no disclosure line is dropped or
    invented. (loss_component_reversed / recognised appear in two blocks; the
    field set still matches exactly.)"""
    float_fields = {n for n, f in GMMSettlementReconciliation.__dataclass_fields__.items()
                    if str(f.type) == "float"}
    spec = _spec_fields(_GMM_RECON_BLOCKS)
    assert spec == float_fields, (
        f"spec-only: {sorted(spec - float_fields)}; "
        f"fields-only: {sorted(float_fields - spec)}")


def test_gmm_to_frame_is_lean_and_faithful():
    recon = _gmm_recon()
    df = reconciliation_to_frame(recon)
    # lean canonical schema, nothing more
    assert tuple(df.columns) == _LEAN_COLUMNS
    assert df["model"].unique().to_list() == ["gmm"]
    assert df["statement"].unique().to_list() == ["settlement"]
    # every (block, line) amount equals the reconciliation field it reads
    for block, lines in _GMM_RECON_BLOCKS:
        for line, field, _p, _m in lines:
            row = df.filter((pl.col("block") == block) & (pl.col("line") == line))
            assert row.height == 1, f"{block}/{line} not emitted once"
            np.testing.assert_allclose(
                row["amount"][0], float(getattr(recon, field)), rtol=1e-12,
                err_msg=f"{block}/{line}")
    # the period is carried
    assert df["period_end"].unique().to_list() == [12]
