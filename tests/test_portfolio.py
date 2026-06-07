"""fcf.portfolio.measure -- the mixed-model orchestrator (P-3 / P-4).

The orchestrator partitions rows by measurement model, runs each block through
its own kernel, and keeps each model's native result separate. P-3 added the
partition + GMM execution; P-4 adds the PAA executor (segments stitched into one
PAAMeasurement). A portfolio that also carries VFA rows still raises
NotImplementedError after the partition is validated. The master invariant:
routing is numerically a no-op -- the portfolio's slice for model m is
byte-identical to the standalone specialist on m's rows.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, ModelPoints, CoverageRate
from fastcashflow._paa import measure_paa
from fastcashflow.basis import BasisRouter
from fastcashflow.portfolio import measure, PortfolioMeasurement, ModelMeasurement


def _flat_basis(discount=0.05):
    return Basis(
        mortality_annual=lambda s, ia, d: np.full(s.shape, 0.001),
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", lambda s, ia, d: np.full(s.shape, 0.001)),))


def _mp(products, channels):
    n = len(products)
    return ModelPoints(
        issue_age=np.full(n, 40), premium=np.zeros(n),
        term_months=np.full(n, 60), benefits={0: np.full(n, 1e4)},
        product=np.array(products), channel=np.array(channels))


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
        term_months=np.full(3, 60), benefits={0: np.full(3, 1e4)},
        product=np.array(["A", "B", "B"]), channel=np.array(["GA", "GA", "GA"]))
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
        benefits={0: np.full(3, 1e4)},
        product=np.array(["P", "Q", "P"]), channel=np.array(["GA", "GA", "GA"]))
    pm = measure(mp, router)
    assert pm.paa.index.tolist() == [0, 1, 2]
    refP = measure_paa(mp.subset([0, 2]), _flat_basis())
    refQ = measure_paa(mp.subset([1]), _flat_basis())
    wP, wQ = refP.lrc_path.shape[1], refQ.lrc_path.shape[1]
    assert np.allclose(pm.paa.measurement.lrc_path[[0, 2], :wP], refP.lrc_path)
    assert np.allclose(pm.paa.measurement.lrc_path[1, :wQ], refQ.lrc_path[0])
    assert np.allclose(pm.paa.measurement.lrc_path[1, wQ:], 0.0)  # earned-out tail


def test_portfolio_raises_on_vfa_rows_after_partition():
    router = BasisRouter({("A", "GA"): _flat_basis(), ("B", "GA"): _flat_basis()},
                         measurement_models={("B", "GA"): "VFA"})
    mp = _mp(["A", "B"], ["GA", "GA"])              # row 1 is a VFA segment
    with pytest.raises(NotImplementedError, match="VFA"):
        measure(mp, router)


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
