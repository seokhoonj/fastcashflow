"""fcf.portfolio.measure -- the mixed-model orchestrator (P-3 / P-4).

The orchestrator partitions rows by measurement model, runs each block through
its own kernel, and keeps each model's native result separate. P-3 added the
partition + GMM execution; P-4 adds the PAA and VFA executors (each model's
segments stitched into one native measurement; the VFA stitch carries a per-MP
2-D discount curve, which roll_forward / group consume). The master invariant:
routing is numerically a no-op -- the portfolio's slice for model m is
byte-identical to the standalone specialist on m's rows.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, ModelPoints, CoverageRate
from fastcashflow import group
from fastcashflow._paa import measure_paa
from fastcashflow._vfa import measure_vfa
from fastcashflow.basis import BasisRouter
from fastcashflow.portfolio import measure, PortfolioMeasurement, ModelMeasurement
from conftest import PATTERNS


def _flat_basis(discount=0.05, investment_return=0.0):
    return Basis(
        mortality_annual=lambda s, ia, d: np.full(s.shape, 0.001),
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.0,
        investment_return=investment_return, fund_fee=0.015,
        coverages=(CoverageRate("DEATH", lambda s, ia, d: np.full(s.shape, 0.001)),))


def _mp(products, channels):
    n = len(products)
    return ModelPoints(
        issue_age=np.full(n, 40), premium=np.zeros(n),
        term_months=np.full(n, 60), benefits={"DEATH": np.full(n, 1e4)},
        product=np.array(products), channel=np.array(channels),
        calculation_methods=PATTERNS)


# ---------------------------------------------------------------------------
# all-GMM: matches gmm.measure, partition is the full range
# ---------------------------------------------------------------------------
def test_portfolio_all_gmm_matches_gmm_measure():
    router = BasisRouter({("A", "GA"): _flat_basis(0.03),
                          ("B", "GA"): _flat_basis(0.10)})
    mp = _mp(["A", "B", "A"], ["GA", "GA", "GA"])
    pm = measure(mp, router)
    ref = fcf.gmm.measure(mp, router)
    assert isinstance(pm, PortfolioMeasurement)
    assert np.array_equal(pm.gmm.index, np.arange(3))
    assert np.allclose(pm.gmm.measurement.bel, ref.bel)        # incl. per-segment discount
    assert pm.paa is None and pm.vfa is None
    assert pm.model_points.n_mp == 3


def test_portfolio_full_trajectory_matches():
    router = BasisRouter({("A", "GA"): _flat_basis()})
    mp = _mp(["A", "A"], ["GA", "GA"])
    pm = measure(mp, router, full=True)
    ref = fcf.gmm.measure(mp, router, full=True)
    assert np.allclose(pm.gmm.measurement.bel, ref.bel)


# ---------------------------------------------------------------------------
# router-only; mixed rows raise after partition; unused non-GMM segment ignored
# ---------------------------------------------------------------------------
def test_portfolio_rejects_single_basis():
    with pytest.raises(TypeError, match="requires a BasisRouter"):
        measure(_mp(["A"], ["GA"]), _flat_basis())


def test_portfolio_measures_paa_rows_matching_measure_paa():
    """A PAA segment is now executed (P-4), not raised: the portfolio's PAA
    slice is identical to measure_paa on that subset -- routing is a no-op."""
    router = BasisRouter({("A", "GA"): _flat_basis(), ("B", "GA"): _flat_basis()},
                         measurement_models={("B", "GA"): "PAA"})
    mp = ModelPoints(                               # row 0 GMM, rows 1-2 PAA
        issue_age=np.full(3, 40), premium=np.array([0.0, 1200.0, 1200.0]),
        term_months=np.full(3, 60), benefits={"DEATH": np.full(3, 1e4)},
        product=np.array(["A", "B", "B"]), channel=np.array(["GA", "GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    assert pm.gmm.index.tolist() == [0]
    assert pm.paa.index.tolist() == [1, 2]
    assert pm.vfa is None
    ref = measure_paa(mp.subset([1, 2]), _flat_basis())
    assert np.allclose(pm.paa.measurement.lrc, ref.lrc)
    assert np.allclose(pm.paa.measurement.lrc_path, ref.lrc_path)
    assert np.allclose(pm.paa.measurement.revenue, ref.revenue)
    assert np.allclose(pm.paa.measurement.loss_component, ref.loss_component)
    assert sorted(np.concatenate([pm.gmm.index, pm.paa.index])) == [0, 1, 2]


def test_portfolio_paa_stitches_ragged_segments():
    """Two PAA segments with different coverage terms stitch into one ragged
    PAAMeasurement -- each row matches its standalone measure_paa, the shorter
    segment zero-padded on the right (LRC is fully earned past coverage)."""
    router = BasisRouter(
        {("P", "GA"): _flat_basis(), ("Q", "GA"): _flat_basis()},
        measurement_models={("P", "GA"): "PAA", ("Q", "GA"): "PAA"})
    mp = ModelPoints(
        issue_age=np.full(3, 40), premium=np.full(3, 1200.0),
        term_months=np.array([60, 24, 60]),        # Q (row 1) shorter -> ragged
        benefits={"DEATH": np.full(3, 1e4)},
        product=np.array(["P", "Q", "P"]), channel=np.array(["GA", "GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    assert pm.paa.index.tolist() == [0, 1, 2]
    refP = measure_paa(mp.subset([0, 2]), _flat_basis())
    refQ = measure_paa(mp.subset([1]), _flat_basis())
    wP, wQ = refP.lrc_path.shape[1], refQ.lrc_path.shape[1]
    assert np.allclose(pm.paa.measurement.lrc_path[[0, 2], :wP], refP.lrc_path)
    assert np.allclose(pm.paa.measurement.lrc_path[1, :wQ], refQ.lrc_path[0])
    assert np.allclose(pm.paa.measurement.lrc_path[1, wQ:], 0.0)  # earned-out tail


def test_portfolio_vfa_stitches_segments_with_distinct_curves():
    """VFA segments are executed (P-4) and stitched. Each segment discounts at
    its own underlying-items return, so discount_bom is per-MP 2-D -- each row
    carries its segment's curve, and every figure matches measure_vfa alone."""
    router = BasisRouter(
        {("V1", "GA"): _flat_basis(investment_return=0.03),
         ("V2", "GA"): _flat_basis(investment_return=0.06)},
        measurement_models={("V1", "GA"): "VFA", ("V2", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(3, 40), premium=np.zeros(3), term_months=np.full(3, 60),
        account_value=np.full(3, 1e6),
        product=np.array(["V1", "V2", "V1"]), channel=np.array(["GA", "GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    assert pm.vfa.index.tolist() == [0, 1, 2]
    assert pm.gmm is None and pm.paa is None
    refV1 = measure_vfa(mp.subset([0, 2]), _flat_basis(investment_return=0.03))
    refV2 = measure_vfa(mp.subset([1]), _flat_basis(investment_return=0.06))
    assert np.allclose(pm.vfa.measurement.bel[[0, 2]], refV1.bel)
    assert np.allclose(pm.vfa.measurement.bel[1], refV2.bel)
    assert np.allclose(pm.vfa.measurement.csm[[0, 2]], refV1.csm)
    assert np.allclose(pm.vfa.measurement.csm[1], refV2.csm)
    assert np.allclose(pm.vfa.measurement.variable_fee[[0, 2]], refV1.variable_fee)
    # discount_bom is per-MP 2-D, each row on its segment's own curve
    assert pm.vfa.measurement.discount_bom.shape == (3, 61)
    assert np.allclose(pm.vfa.measurement.discount_bom[[0, 2]], refV1.discount_bom)
    assert np.allclose(pm.vfa.measurement.discount_bom[1], refV2.discount_bom)
    assert not np.allclose(refV1.discount_bom, refV2.discount_bom)   # curves differ


def test_portfolio_vfa_ragged_stitch_pads_and_flat_fills():
    """VFA segments with different terms stitch ragged: the shorter segment's
    trajectories zero-pad on the right and its discount_bom tail repeats the last
    factor (a forward rate read off the tail is zero, not a 0/0)."""
    router = BasisRouter(
        {("L", "GA"): _flat_basis(investment_return=0.06),
         ("S", "GA"): _flat_basis(investment_return=0.06)},
        measurement_models={("L", "GA"): "VFA", ("S", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(3, 40), premium=np.zeros(3),
        term_months=np.array([60, 24, 60]),       # S (row 1) shorter -> ragged
        account_value=np.full(3, 1e6),
        product=np.array(["L", "S", "L"]), channel=np.array(["GA", "GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    refL = measure_vfa(mp.subset([0, 2]), _flat_basis(investment_return=0.06))
    refS = measure_vfa(mp.subset([1]), _flat_basis(investment_return=0.06))
    nL, nS = refL.bel_path.shape[1] - 1, refS.bel_path.shape[1] - 1
    assert nS < nL
    db = pm.vfa.measurement.discount_bom
    assert db.shape == (3, nL + 1)                         # padded to longer horizon
    assert np.allclose(db[1, :nS + 1], refS.discount_bom)        # S's own curve
    assert np.allclose(db[1, nS + 1:], refS.discount_bom[-1])    # flat-filled tail
    bp = pm.vfa.measurement.bel_path
    assert np.allclose(bp[1, :nS + 1], refS.bel_path[0])
    assert np.allclose(bp[1, nS + 1:], 0.0)                      # zero-padded tail
    assert np.isclose(pm.vfa.measurement.bel[1], refS.bel[0])


def test_portfolio_vfa_roll_forward_matches_per_segment():
    """roll_forward on a multi-curve VFA portfolio result (2-D discount_bom)
    matches per-segment roll_forward -- the movement layer slices the time axis,
    not the model-point axis, on the 2-D curve."""
    router = BasisRouter(
        {("V1", "GA"): _flat_basis(investment_return=0.03),
         ("V2", "GA"): _flat_basis(investment_return=0.06)},
        measurement_models={("V1", "GA"): "VFA", ("V2", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(3, 40), premium=np.zeros(3), term_months=np.full(3, 60),
        account_value=np.full(3, 1e6),
        product=np.array(["V1", "V2", "V1"]), channel=np.array(["GA", "GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    mv = fcf.roll_forward(pm.vfa.measurement)
    rv1 = fcf.roll_forward(measure_vfa(mp.subset([0, 2]),
                                       _flat_basis(investment_return=0.03)))
    rv2 = fcf.roll_forward(measure_vfa(mp.subset([1]),
                                       _flat_basis(investment_return=0.06)))
    assert np.allclose(mv[0].bel_interest[[0, 2]], rv1[0].bel_interest)
    assert np.allclose(mv[0].bel_interest[1], rv2[0].bel_interest)
    assert np.allclose(mv[0].csm_release[[0, 2]], rv1[0].csm_release)


def test_portfolio_vfa_group_by_curve_succeeds():
    """group() on a multi-curve VFA result, each group sitting in one curve
    (one product per curve), reconciles per-group and returns two groups."""
    router = BasisRouter(
        {("V1", "GA"): _flat_basis(investment_return=0.03),
         ("V2", "GA"): _flat_basis(investment_return=0.06)},
        measurement_models={("V1", "GA"): "VFA", ("V2", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(3, 40), premium=np.zeros(3), term_months=np.full(3, 60),
        account_value=np.full(3, 1e6),
        product=np.array(["V1", "V2", "V1"]), channel=np.array(["GA", "GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    g = group(pm.vfa.measurement, "product")
    assert g.csm.shape[0] == 2 and g.discount_bom.shape[0] == 2


def test_portfolio_vfa_group_rejects_mixed_curves():
    """A group spanning two underlying-items returns is incoherent (the CSM
    accretes at one curve) -- group() raises, the same guard the GMM grouping
    applies to a segmented result."""
    router = BasisRouter(
        {("V1", "GA"): _flat_basis(investment_return=0.03),
         ("V2", "GA"): _flat_basis(investment_return=0.06)},
        measurement_models={("V1", "GA"): "VFA", ("V2", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(2, 40), premium=np.zeros(2), term_months=np.full(2, 60),
        account_value=np.full(2, 1e6),
        product=np.array(["V1", "V2"]), channel=np.array(["GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    with pytest.raises(ValueError, match="different .*discount curves"):
        group(pm.vfa.measurement, "channel")     # GA spans both curves -> one group


def test_portfolio_measures_all_three_models_in_one_call():
    """One mixed table, one call: GMM + PAA + VFA each routed to its own kernel
    and slot, every slice matching the standalone specialist -- routing is a
    numeric no-op, and the partition is a clean 0..n_mp-1 cover."""
    router = BasisRouter(
        {("G", "GA"): _flat_basis(),
         ("P", "GA"): _flat_basis(),
         ("V", "GA"): _flat_basis(investment_return=0.04)},
        measurement_models={("P", "GA"): "PAA", ("V", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(4, 40), premium=np.array([0.0, 1200.0, 0.0, 0.0]),
        term_months=np.full(4, 60), benefits={"DEATH": np.full(4, 1e4)},
        account_value=np.array([0.0, 0.0, 1e6, 1e6]),
        product=np.array(["G", "P", "V", "V"]),
        channel=np.array(["GA", "GA", "GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    assert pm.gmm.index.tolist() == [0]
    assert pm.paa.index.tolist() == [1]
    assert pm.vfa.index.tolist() == [2, 3]
    assert np.allclose(pm.gmm.measurement.bel,
                       fcf.gmm.measure(mp.subset([0]), _flat_basis()).bel)
    assert np.allclose(pm.paa.measurement.lrc_path,
                       measure_paa(mp.subset([1]), _flat_basis()).lrc_path)
    assert np.allclose(pm.vfa.measurement.csm,
                       measure_vfa(mp.subset([2, 3]),
                                   _flat_basis(investment_return=0.04)).csm)
    assert sorted(np.concatenate(
        [pm.gmm.index, pm.paa.index, pm.vfa.index])) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# P-5a aggregation contract: loss_component_total + summary (no cross-model
# pooling of figures that mean different things)
# ---------------------------------------------------------------------------
def _three_model_inputs():
    router = BasisRouter(
        {("G", "GA"): _flat_basis(),
         ("P", "GA"): _flat_basis(),
         ("V", "GA"): _flat_basis(investment_return=0.04)},
        measurement_models={("P", "GA"): "PAA", ("V", "GA"): "VFA"})
    mp = ModelPoints(                              # premium 0 + claims -> GMM onerous
        issue_age=np.full(4, 40), premium=np.array([0.0, 1200.0, 0.0, 0.0]),
        term_months=np.full(4, 60), benefits={"DEATH": np.full(4, 1e4)},
        account_value=np.array([0.0, 0.0, 1e6, 1e6]),
        product=np.array(["G", "P", "V", "V"]),
        channel=np.array(["GA", "GA", "GA", "GA"]),
        calculation_methods=PATTERNS)
    return mp, router


def _three_model_portfolio():
    return measure(*_three_model_inputs())


def test_loss_component_total_sums_only_the_loss_across_models():
    """loss_component_total is the one cross-model additive figure -- the sum of
    each present model's loss_component, sign-identical max(0, FCF) at inception."""
    pm = _three_model_portfolio()
    expected = (pm.gmm.measurement.loss_component.sum()
                + pm.paa.measurement.loss_component.sum()
                + pm.vfa.measurement.loss_component.sum())
    assert isinstance(pm.loss_component_total(), float)
    assert np.isclose(pm.loss_component_total(), expected)
    assert pm.loss_component_total() > 0.0          # the GMM block is onerous


def test_summary_keeps_each_model_in_its_own_block():
    """summary() never pools a BEL with an LRC: each model has its own block with
    its own headline fields, and loss_component_total is the only sum."""
    pm = _three_model_portfolio()
    s = pm.summary()
    assert set(s) == {"loss_component_total", "gmm", "paa", "vfa"}
    assert set(s["gmm"]) == {"bel", "ra", "csm", "loss_component"}
    assert set(s["paa"]) == {"lrc", "loss_component"}      # LRC, not BEL/CSM
    assert set(s["vfa"]) == {"bel", "ra", "csm", "loss_component"}
    assert np.isclose(s["gmm"]["bel"], pm.gmm.measurement.bel.sum())
    assert np.isclose(s["paa"]["lrc"], pm.paa.measurement.lrc.sum())
    assert np.isclose(s["vfa"]["csm"], pm.vfa.measurement.csm.sum())
    assert np.isclose(s["loss_component_total"], pm.loss_component_total())


# ---------------------------------------------------------------------------
# P-5b Gate B (1): full=False chunks PAA/VFA to headline-only, bounded memory.
# ---------------------------------------------------------------------------
def test_portfolio_full_false_headline_matches_full_per_mp():
    """full=False yields the per-MP headline identical to full=True, with the
    PAA/VFA trajectories dropped (summary works; group/roll/report would not)."""
    mp, router = _three_model_inputs()
    full = measure(mp, router, full=True)
    head = measure(mp, router, full=False)
    assert np.allclose(head.gmm.measurement.bel, full.gmm.measurement.bel)
    assert np.allclose(head.paa.measurement.lrc, full.paa.measurement.lrc)
    assert np.allclose(head.paa.measurement.loss_component,
                       full.paa.measurement.loss_component)
    assert np.allclose(head.vfa.measurement.csm, full.vfa.measurement.csm)
    assert np.allclose(head.vfa.measurement.bel, full.vfa.measurement.bel)
    # trajectories dropped on the headline path
    assert head.paa.measurement.lrc_path is None
    assert head.vfa.measurement.bel_path is None and head.vfa.measurement.cashflows is None
    # the aggregation contract still holds on the headline result
    assert np.isclose(head.loss_component_total(), full.loss_component_total())


def test_portfolio_full_false_chunking_is_numeric_noop():
    """chunk_size changes only peak memory, never the numbers -- chunk_size=1
    (a block per row, across segments) matches one big block and full=True."""
    prod = np.array(["PA", "VB", "PB", "VA", "PA", "VB", "PB", "VA"])
    n = len(prod)
    is_v = np.array([p[0] == "V" for p in prod])
    router = BasisRouter(
        {("PA", "GA"): _flat_basis(), ("PB", "GA"): _flat_basis(),
         ("VA", "GA"): _flat_basis(investment_return=0.03),
         ("VB", "GA"): _flat_basis(investment_return=0.06)},
        measurement_models={("PA", "GA"): "PAA", ("PB", "GA"): "PAA",
                            ("VA", "GA"): "VFA", ("VB", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(n, 40),
        premium=np.where(is_v, 0.0, 1200.0), term_months=np.full(n, 60),
        benefits={"DEATH": np.full(n, 1e4)},
        account_value=np.where(is_v, 1e6, 0.0),
        product=prod, channel=np.full(n, "GA"),
        calculation_methods=PATTERNS)
    a = measure(mp, router, full=False, chunk_size=1)        # a block per row
    b = measure(mp, router, full=False, chunk_size=1000)     # one block
    full = measure(mp, router, full=True)
    for attr in ("lrc", "loss_component", "fcf"):
        assert np.allclose(getattr(a.paa.measurement, attr),
                           getattr(b.paa.measurement, attr))
        assert np.allclose(getattr(a.paa.measurement, attr),
                           getattr(full.paa.measurement, attr))
    for attr in ("bel", "ra", "csm", "variable_fee", "loss_component"):
        assert np.allclose(getattr(a.vfa.measurement, attr),
                           getattr(b.vfa.measurement, attr))
        assert np.allclose(getattr(a.vfa.measurement, attr),
                           getattr(full.vfa.measurement, attr))
    assert a.vfa.measurement.bel_path is None        # headline only, regardless of chunk
    assert a.paa.measurement.lrc_path is None


def test_portfolio_rejects_non_positive_chunk_size():
    """chunk_size <= 0 must raise, not silently skip every block and scatter
    uninitialised headline arrays."""
    mp, router = _three_model_inputs()
    with pytest.raises(ValueError, match="chunk_size"):
        measure(mp, router, full=False, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        measure(mp, router, full=False, chunk_size=-5)


def test_summary_omits_absent_models():
    """A block appears only for a model the portfolio carries -- an all-GMM book
    has no paa / vfa block (the slot is None, not a fabricated zero)."""
    router = BasisRouter({("A", "GA"): _flat_basis()})
    pm = measure(_mp(["A", "A"], ["GA", "GA"]), router)
    s = pm.summary()
    assert set(s) == {"loss_component_total", "gmm"}
    assert "paa" not in s and "vfa" not in s


# ---------------------------------------------------------------------------
# Gate A: single declared model short-circuits the partition (structural -- no
# wall-clock assert; the <5% throughput target is a documented gate, measured in
# dev/p5b_bottleneck.py, not pinned here where CI noise would flake it).
# ---------------------------------------------------------------------------
def _spy_partition(monkeypatch):
    """Patch _partition_by_model to record calls, delegating to the original."""
    import fastcashflow.portfolio as P
    calls = []
    orig = P._partition_by_model
    monkeypatch.setattr(
        P, "_partition_by_model",
        lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    return calls


def test_single_model_gmm_router_skips_partition(monkeypatch):
    calls = _spy_partition(monkeypatch)
    router = BasisRouter({("A", "GA"): _flat_basis(0.03),
                          ("B", "GA"): _flat_basis(0.10)})   # both GMM
    mp = _mp(["A", "B", "A"], ["GA", "GA", "GA"])
    pm = measure(mp, router)
    assert calls == []                                       # short-circuited
    assert pm.gmm.index.tolist() == [0, 1, 2] and pm.paa is None and pm.vfa is None
    assert np.allclose(pm.gmm.measurement.bel, fcf.gmm.measure(mp, router).bel)


def test_single_model_paa_router_skips_partition(monkeypatch):
    calls = _spy_partition(monkeypatch)
    router = BasisRouter({("P", "GA"): _flat_basis(), ("Q", "GA"): _flat_basis()},
                         measurement_models={("P", "GA"): "PAA", ("Q", "GA"): "PAA"})
    mp = ModelPoints(
        issue_age=np.full(2, 40), premium=np.full(2, 1200.0),
        term_months=np.full(2, 60), benefits={"DEATH": np.full(2, 1e4)},
        product=np.array(["P", "Q"]), channel=np.array(["GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    assert calls == []
    assert pm.paa.index.tolist() == [0, 1] and pm.gmm is None
    assert np.allclose(pm.paa.measurement.lrc_path[0],
                       measure_paa(mp.subset([0]), _flat_basis()).lrc_path[0])


def test_single_model_vfa_router_skips_partition(monkeypatch):
    calls = _spy_partition(monkeypatch)
    router = BasisRouter({("V", "GA"): _flat_basis(investment_return=0.05)},
                         measurement_models={("V", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(2, 40), premium=np.zeros(2), term_months=np.full(2, 60),
        account_value=np.full(2, 1e6),
        product=np.array(["V", "V"]), channel=np.array(["GA", "GA"]),
        calculation_methods=PATTERNS)
    pm = measure(mp, router)
    assert calls == []
    assert pm.vfa.index.tolist() == [0, 1] and pm.gmm is None


def test_multi_declared_but_rows_all_gmm_still_partitions(monkeypatch):
    """A router that declares a PAA segment the rows never use is NOT a single
    declared model, so it keeps the row partition -- the unused-segment handling
    must stay on that path (else a stray declaration would change routing)."""
    calls = _spy_partition(monkeypatch)
    router = BasisRouter({("A", "GA"): _flat_basis(), ("Z", "GA"): _flat_basis()},
                         measurement_models={("Z", "GA"): "PAA"})   # mixed declared
    mp = _mp(["A", "A"], ["GA", "GA"])                      # rows all GMM, Z unused
    pm = measure(mp, router)
    assert calls == [1]                                     # took the partition path
    assert pm.gmm.index.size == 2 and pm.paa is None


def test_portfolio_unused_non_gmm_segment_is_ignored():
    """A PAA segment the model points never use must not block an all-GMM book --
    the orchestrator partitions the rows present, not the router's declarations."""
    router = BasisRouter({("A", "GA"): _flat_basis(), ("Z", "GA"): _flat_basis()},
                         measurement_models={("Z", "GA"): "PAA"})
    mp = _mp(["A", "A"], ["GA", "GA"])             # no Z (PAA) rows
    pm = measure(mp, router)
    assert pm.gmm.index.size == 2 and pm.paa is None


# ---------------------------------------------------------------------------
# container invariants (construction-time)
# ---------------------------------------------------------------------------
def test_model_measurement_validates_index():
    router = BasisRouter({("A", "GA"): _flat_basis()})
    mp = _mp(["A", "A"], ["GA", "GA"])
    meas = fcf.gmm.measure(mp, router)
    with pytest.raises(ValueError, match="sorted and unique"):
        ModelMeasurement(np.array([1, 0]), meas)
    with pytest.raises(ValueError, match="rows"):
        ModelMeasurement(np.array([0]), meas)         # size 1 != 2 measurement rows


def test_portfolio_partition_must_be_complete():
    router = BasisRouter({("A", "GA"): _flat_basis()})
    mp = _mp(["A", "A", "A"], ["GA", "GA", "GA"])
    meas = fcf.gmm.measure(mp.subset([0, 1]), router)
    with pytest.raises(ValueError, match="partition covers"):
        PortfolioMeasurement(model_points=mp,
                             gmm=ModelMeasurement(np.array([0, 1]), meas))   # 2 of 3


def test_portfolio_rejects_wrong_measurement_type_in_slot():
    """A slot must hold its own model's native measurement -- a GMMMeasurement in
    the paa slot defeats the per-model separation invariant."""
    router = BasisRouter({("A", "GA"): _flat_basis()})
    mp = _mp(["A", "A"], ["GA", "GA"])
    gmm_meas = fcf.gmm.measure(mp, router)
    with pytest.raises(TypeError, match="paa must hold a PAAMeasurement"):
        PortfolioMeasurement(model_points=mp,
                             paa=ModelMeasurement(np.arange(2), gmm_meas))


# ---------------------------------------------------------------------------
# write_measurement: one file per model present
# ---------------------------------------------------------------------------
def test_write_measurement_portfolio_one_file_per_model(tmp_path):
    """A portfolio writes results-{gmm,paa,vfa}.parquet, each slot's native
    columns plus an id column carrying the portfolio row positions."""
    import polars as pl

    pm = _three_model_portfolio()
    fcf.write_measurement(pm, tmp_path / "results.parquet")
    assert not (tmp_path / "results.parquet").exists()

    gmm = pl.read_parquet(tmp_path / "results-gmm.parquet")
    paa = pl.read_parquet(tmp_path / "results-paa.parquet")
    vfa = pl.read_parquet(tmp_path / "results-vfa.parquet")
    assert gmm["id"].to_list() == pm.gmm.index.tolist()
    assert paa["id"].to_list() == pm.paa.index.tolist()
    assert vfa["id"].to_list() == pm.vfa.index.tolist()
    assert "bel" in gmm.columns and "loss_component" in gmm.columns
    assert "lrc" in paa.columns and "bel" not in paa.columns
    assert "variable_fee" in vfa.columns and "time_value" in vfa.columns
    assert np.allclose(gmm["bel"].to_numpy(), pm.gmm.measurement.bel)
    assert np.allclose(paa["lrc"].to_numpy(), pm.paa.measurement.lrc)
    assert np.allclose(vfa["csm"].to_numpy(), pm.vfa.measurement.csm)


def test_write_measurement_portfolio_maps_ids_through_the_partition(tmp_path):
    """Caller ids land on each model's rows via the partition index, and a
    wrong-length ids array is rejected. An absent model writes no file."""
    import polars as pl

    pm = _three_model_portfolio()
    ids = np.array(["p0", "p1", "p2", "p3"])
    fcf.write_measurement(pm, tmp_path / "out.csv", ids=ids)
    vfa = pl.read_csv(tmp_path / "out-vfa.csv")
    assert vfa["id"].to_list() == ids[pm.vfa.index].tolist()

    with pytest.raises(ValueError, match="ids has 2 rows"):
        fcf.write_measurement(pm, tmp_path / "bad.csv", ids=ids[:2])

    router = BasisRouter({("A", "GA"): _flat_basis()})
    all_gmm = measure(_mp(["A", "A"], ["GA", "GA"]), router)
    fcf.write_measurement(all_gmm, tmp_path / "g.parquet")
    assert (tmp_path / "g-gmm.parquet").exists()
    assert not (tmp_path / "g-paa.parquet").exists()
    assert not (tmp_path / "g-vfa.parquet").exists()


def test_write_measurement_portfolio_preflights_carry_only_vfa(tmp_path):
    """A carry-only VFA CSM is refused before any file is written -- no
    partial gmm/paa output left on disk."""
    from dataclasses import replace

    from fastcashflow._vfa import CSM_BASIS_CARRY_ONLY

    pm = _three_model_portfolio()
    carry = replace(pm.vfa.measurement, csm_basis=CSM_BASIS_CARRY_ONLY)
    pm2 = PortfolioMeasurement(
        model_points=pm.model_points, gmm=pm.gmm, paa=pm.paa,
        vfa=ModelMeasurement(pm.vfa.index, carry))
    with pytest.raises(ValueError, match="carry-only"):
        fcf.write_measurement(pm2, tmp_path / "results.parquet")
    assert not list(tmp_path.iterdir())
