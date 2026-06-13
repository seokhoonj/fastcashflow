"""reinsurance.settle -- IFRS 17 para 65/66 subsequent measurement (skeleton).

Authoritative skeleton: written before the implementation and activated
unchanged once it lands. Anchor facts from the G-gate
(dev/reinsurance-settle-gate.md) and its handcalc
(dev/scratch_reinsurance_settle_gate.py, re-run here through the public entry).

The reinsurance CSM is adjusted in subsequent measurement "in the same manner
as a group of insurance contracts issued" (para 66: interest accreted,
future-service change, coverage-unit release) BUT -- pocket guide verbatim --
"reinsurance contracts held cannot be onerous. Accordingly, the requirements
on onerous contracts do not apply": NO zero floor, NO loss component. So
reinsurance.settle is the gmm.settle algorithm with the CSM step replaced by
``csm_after = csm_opening + csm_accretion + csm_experience_unlocking`` (no
floor, no paragraph-48/50(b) algebra), then the single B119 release. The
loss-recovery component (para 66A-66B) needs the underlying group's loss
component (cross-contract) and is a documented v1 cut.

Pinned numbers (the handcalc's net-cost book: QuotaShare 0.4, age 40, premium
400k, benefit 1e8, term 240, elapsed 36, period 12, scale 1000):
* ON-TRACK settle csm_closing == carry bridge measure_inforce csm ==
  -5,636,496,191.63 (telescoping).
* OFF-TRACK (+50% survivors) the CSM stays NEGATIVE (a GMM floor would clamp a
  multi-billion net cost to zero).
"""
import numpy as np
import pytest
from dataclasses import replace

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from conftest import make_death_basis, PATTERNS

settle = getattr(fcf.reinsurance, "settle", None)
pytestmark = pytest.mark.skipif(
    settle is None,
    reason="reinsurance.settle not implemented yet (skeleton activates "
           "unchanged once it lands)")

QS = fcf.reinsurance.QuotaShare
ON_TRACK_CSM_CLOSING = -5_636_496_191.63


def _basis():
    return make_death_basis(mortality_q=0.002, lapse_q=0.005,
                            discount_annual=0.03, ra_confidence=0.75,
                            mortality_cv=0.10)


def _book(*, n=1, elapsed=36, period=12, cession=0.4, count_factor=1.0,
          premium=400_000.0):
    """A net-cost ceded book (premium high enough that the reinsurance CSM is
    negative -- the typical 'cost of cover' case). ``count_factor`` scales the
    closing count off the expected survival."""
    basis = _basis()
    treaty = QS(cession)
    unit = ModelPoints.single(40, premium, 240, benefits={0: 1e8},
                              calculation_methods=PATTERNS)
    m = fcf.reinsurance.measure(unit, basis, treaty=treaty)
    surv = m.cashflows.inforce[0]
    csm_seed = float(m.csm_path[0, elapsed - period])
    scale = np.array([1000.0, 500.0, 2000.0])[:n]
    prior_count = scale * surv[elapsed - period]
    count = scale * surv[elapsed] * count_factor
    ids = np.array([f"R{i}" for i in range(n)])
    rep = lambda v: np.full(n, v)
    mp = ModelPoints(
        issue_age=rep(40).astype(np.int64), premium=rep(premium),
        term_months=rep(240).astype(np.int64), benefits={0: rep(1e8)},
        count=count, elapsed_months=rep(elapsed).astype(np.int64), mp_id=ids,
        calculation_methods=PATTERNS)
    state = InforceState(
        mp_id=ids, elapsed_months=rep(elapsed).astype(np.int64), count=count,
        prior_csm=csm_seed * scale, lock_in_rate=basis.discount_annual,
        prior_count=prior_count)
    return mp, state, basis, treaty


# ---------------------------------------------------------------------------
# block reconciliation -- and NO loss component anywhere
# ---------------------------------------------------------------------------
def test_blocks_reconcile_and_there_is_no_loss_component():
    mp, state, basis, treaty = _book()
    mv = settle(mp, state, basis, treaty=treaty, period_months=12)
    np.testing.assert_allclose(
        mv.bel_opening + mv.bel_interest - mv.bel_release + mv.bel_experience,
        mv.bel_closing, rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(
        mv.ra_opening + mv.ra_interest - mv.ra_release + mv.ra_experience,
        mv.ra_closing, rtol=1e-9, atol=1e-3)
    # CSM recursion is FOUR-term: no loss-component reversal/recognition
    np.testing.assert_allclose(
        mv.csm_opening + mv.csm_accretion + mv.csm_experience_unlocking
        - mv.csm_release, mv.csm_closing, rtol=1e-9, atol=1e-3)
    # the three-term cross tie still holds (same B72(c) locked-in measure)
    np.testing.assert_allclose(
        mv.csm_experience_unlocking + mv.finance_wedge,
        -(mv.bel_experience + mv.ra_experience), rtol=1e-9, atol=1e-3)
    # there is genuinely no loss-component surface on a reinsurance movement
    for attr in ("loss_component_opening", "loss_component_closing",
                 "loss_component_reversed", "loss_component_recognised"):
        assert not hasattr(mv, attr), attr


# ---------------------------------------------------------------------------
# on-track telescoping: settle closing CSM == the carry bridge
# ---------------------------------------------------------------------------
def test_on_track_telescopes_to_carry_bridge():
    mp, state, basis, treaty = _book(count_factor=1.0)
    mv = settle(mp, state, basis, treaty=treaty, period_months=12)
    with pytest.warns(DeprecationWarning):       # the carry bridge is deprecated
        carry = fcf.reinsurance.measure_inforce(mp, state, basis, treaty=treaty,
                                                period_months=12)
    np.testing.assert_allclose(mv.csm_closing, carry.csm, rtol=1e-9, atol=1e-3)
    # experience lines vanish on-track
    assert abs(float(mv.bel_experience[0])) < 1e-2
    assert abs(float(mv.ra_experience[0])) < 1e-2
    np.testing.assert_allclose(float(mv.csm_closing[0]), ON_TRACK_CSM_CLOSING,
                               rtol=1e-6)


# ---------------------------------------------------------------------------
# the decisive reinsurance fact: the CSM is NOT floored (para 65)
# ---------------------------------------------------------------------------
def test_csm_is_not_floored_net_cost_stays_negative():
    mp, state, basis, treaty = _book(count_factor=1.5)   # off-track, more cost
    mv = settle(mp, state, basis, treaty=treaty, period_months=12)
    # a net-cost reinsurance CSM stays negative -- a GMM zero floor would clamp
    assert float(mv.csm_opening[0]) < 0.0
    csm_after = float(mv.csm_closing[0] + mv.csm_release[0])
    assert csm_after < 0.0
    assert float(mv.csm_closing[0]) < 0.0


def test_net_gain_book_has_positive_csm():
    # a low-premium cession is a net gain -> positive CSM, still no floor logic
    mp, state, basis, treaty = _book(premium=80_000.0)
    mv = settle(mp, state, basis, treaty=treaty, period_months=12)
    assert float(mv.csm_opening[0]) > 0.0


# ---------------------------------------------------------------------------
# scale variant + carry-bridge deprecation
# ---------------------------------------------------------------------------
def test_settle_aggregate_equals_per_mp_settle_sum():
    agg = getattr(fcf.reinsurance, "settle_aggregate", None)
    if agg is None:
        pytest.skip("reinsurance.settle_aggregate not implemented")
    mp, state, basis, treaty = _book(n=3, count_factor=1.2)
    a = agg(mp, state, basis, treaty=treaty, period_months=12)
    per = settle(mp, state, basis, treaty=treaty, period_months=12)
    np.testing.assert_allclose(a.csm_closing, float(per.csm_closing.sum()),
                               rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(a.bel_closing, float(per.bel_closing.sum()),
                               rtol=1e-9, atol=1e-3)


def test_carry_bridge_warns_now_that_settle_exists():
    mp, state, basis, treaty = _book(n=3)
    with pytest.warns(DeprecationWarning, match="settle"):
        fcf.reinsurance.measure_inforce_aggregate(
            mp, state, basis, treaty=treaty, period_months=12)


# ---------------------------------------------------------------------------
# settle_stream: out-of-core, matches the in-memory settle
# ---------------------------------------------------------------------------
def test_settle_stream_matches_in_memory(tmp_path):
    import polars as pl
    settle_stream = getattr(fcf.reinsurance, "settle_stream", None)
    if settle_stream is None:
        pytest.skip("reinsurance.settle_stream not implemented")
    mp, state, basis, treaty = _book(n=3, count_factor=1.2)
    n = mp.n_mp
    spec = {
        "mp_id": np.asarray(mp.mp_id).astype(str),
        "issue_age": np.asarray(mp.issue_age),
        "premium": np.asarray(mp.premium),
        "term_months": np.asarray(mp.term_months),
    }
    st = {
        "mp_id": np.asarray(state.mp_id).astype(str),
        "elapsed_months": np.asarray(state.elapsed_months),
        "count": np.asarray(state.count),
        "prior_csm": np.asarray(state.prior_csm),
        "lock_in_rate": np.full(n, float(state.lock_in_rate)),
        "prior_count": np.asarray(state.prior_count),
    }
    cov = pl.DataFrame({"mp_id": spec["mp_id"], "coverage": ["DEATH"] * n,
                        "amount": np.asarray(mp.benefits[0], dtype=np.float64)})
    cp = tmp_path / "rcov.parquet"; cov.write_parquet(cp)
    ip = tmp_path / "rinforce.parquet"
    pl.DataFrame({**spec, **{k: v for k, v in st.items()
                             if k != "mp_id"}}).write_parquet(ip)
    out = tmp_path / "rout"
    count = settle_stream(ip, out, basis, treaty=treaty, coverages=cp,
                          calculation_methods=PATTERNS, period_months=12,
                          chunk_size=2)
    assert count == n
    parts = pl.read_parquet(str(out / "part-*.parquet")).sort("id")
    mv = settle(mp, state, basis, treaty=treaty, period_months=12)
    order = {str(i): k for k, i in enumerate(np.asarray(mp.mp_id).astype(str))}
    idx = [order[i] for i in parts["id"].to_list()]
    np.testing.assert_allclose(parts["csm_closing"].to_numpy(),
                               np.asarray(mv.csm_closing)[idx], rtol=1e-9,
                               atol=1e-3)
