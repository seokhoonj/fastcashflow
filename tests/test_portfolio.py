"""fcf.portfolio.measure -- the mixed-model orchestrator (P-3).

P-3 implements the row partition by measurement model + the GMM execution; a
portfolio that also carries PAA / VFA rows raises NotImplementedError after the
partition is validated. The container (PortfolioMeasurement / ModelMeasurement)
validates the 0..n_mp-1 partition as a construction invariant.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, ModelPoints, CoverageRate
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


def test_portfolio_raises_on_paa_rows_after_partition():
    router = BasisRouter({("A", "GA"): _flat_basis(), ("B", "GA"): _flat_basis()},
                         measurement_models={("B", "GA"): "PAA"})
    mp = _mp(["A", "B"], ["GA", "GA"])              # row 1 is a PAA segment
    with pytest.raises(NotImplementedError, match="PAA"):
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
