"""Model-point compression -- fcf.compress.

Hand-calc anchor: a book made of K identical-within-type groups compresses to K
representatives with ZERO valuation error (each cluster is identical policies, so
its representative reproduces the group exactly once rescaled to the group count).
Plus the identity (n_clusters == n_mp) and accuracy / determinism invariants.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from conftest import make_death_basis, PATTERNS


def _basis():
    return make_death_basis(mortality_q=0.002, lapse_q=0.02, discount_annual=0.03,
                            mortality_cv=0.0)


def _typed_book(types, copies):
    """A book of identical-within-type policies. ``types`` is a list of
    (issue_age, term_months, benefit) tuples; each repeated ``copies`` times with
    count 1, so a K-type book has K distinct calibration vectors."""
    ages, terms, bens = [], [], []
    for (age, term, ben) in types:
        ages += [age] * copies
        terms += [term] * copies
        bens += [ben] * copies
    n = len(ages)
    return fcf.ModelPoints(
        issue_age=np.array(ages), premium=np.full(n, 1000.0),
        term_months=np.array(terms), benefits={"DEATH": np.array(bens, float)},
        count=np.ones(n), calculation_methods=PATTERNS)


def test_compress_identical_groups_is_exact():
    """K identical-within-type groups -> compress to K -> zero PV error and the
    representative counts are the group totals."""
    types = [(35, 240, 2.0e8), (50, 120, 5.0e7), (45, 180, 1.0e8)]
    mp = _typed_book(types, copies=20)            # 60 policies, 3 types
    res = fcf.compress(mp, _basis(), n_clusters=3, seed=0)
    assert res.n_clusters == 3
    assert res.max_abs_rel_error < 1e-9           # identical clusters -> exact
    assert np.isclose(res.model_points.count.sum(), 60.0)
    assert sorted(res.model_points.count.tolist()) == [20.0, 20.0, 20.0]


def test_compress_identity_when_clusters_equal_rows():
    """n_clusters == n_mp -> each policy its own cluster, exact, counts unchanged."""
    mp = _typed_book([(40, 120, 1e8), (55, 240, 2e8)], copies=5)
    res = fcf.compress(mp, _basis(), n_clusters=mp.n_mp, seed=0)
    assert res.n_clusters == mp.n_mp
    assert res.max_abs_rel_error == 0.0
    assert np.isclose(res.model_points.count.sum(), mp.count.sum())


def test_compress_preserves_total_count():
    rng = np.random.default_rng(1)
    n = 500
    mp = fcf.ModelPoints(
        issue_age=rng.integers(30, 60, n), premium=rng.uniform(500, 2000, n),
        term_months=rng.choice([120, 180, 240], n),
        benefits={"DEATH": rng.uniform(5e7, 2e8, n)},
        count=rng.uniform(1.0, 50.0, n), calculation_methods=PATTERNS)
    res = fcf.compress(mp, _basis(), n_clusters=40, seed=0)
    assert np.isclose(res.model_points.count.sum(), mp.count.sum())


def test_compress_accuracy_on_varied_book():
    """A varied 1000-policy book compresses 25x with a small aggregate-PV error
    across the base + stress scenarios."""
    rng = np.random.default_rng(2)
    n = 1000
    mp = fcf.ModelPoints(
        issue_age=rng.integers(25, 65, n), premium=rng.uniform(300, 3000, n),
        term_months=rng.choice([120, 180, 240, 300], n),
        benefits={"DEATH": rng.uniform(2e7, 3e8, n)},
        count=rng.uniform(1.0, 100.0, n), calculation_methods=PATTERNS)
    res = fcf.compress(mp, _basis(), n_clusters=40, seed=0)
    assert res.model_points.n_mp <= 40
    assert res.max_abs_rel_error < 0.02           # < 2% on every scenario
    assert res.scenario_names == ("base", "mortality x1.15", "lapse x1.15")


def test_compress_is_deterministic():
    mp = _typed_book([(40, 120, 1e8), (50, 240, 2e8), (45, 180, 1.5e8)], copies=15)
    a = fcf.compress(mp, _basis(), n_clusters=10, seed=7)
    b = fcf.compress(mp, _basis(), n_clusters=10, seed=7)
    assert np.array_equal(a.cluster_id, b.cluster_id)
    assert np.array_equal(a.representative, b.representative)
    assert np.allclose(a.model_points.count, b.model_points.count)


def test_compress_cluster_id_indexes_representatives():
    """Every original row maps to a valid representative slot (0..n_clusters-1)."""
    mp = _typed_book([(40, 120, 1e8), (55, 240, 2e8)], copies=10)
    res = fcf.compress(mp, _basis(), n_clusters=2, seed=0)
    assert res.cluster_id.min() >= 0
    assert res.cluster_id.max() < res.n_clusters
    assert res.cluster_id.shape[0] == mp.n_mp


def test_compress_custom_stresses():
    mp = _typed_book([(40, 120, 1e8), (50, 240, 2e8)], copies=10)
    res = fcf.compress(mp, _basis(), n_clusters=2,
                       stresses=(fcf.solvency.scale_mortality(1.25),), seed=0)
    assert res.scenario_names == ("base", "mortality x1.25")
    assert res.pv_full.shape == (2,)


@pytest.mark.parametrize("k", [0, -1, 999])
def test_compress_rejects_bad_n_clusters(k):
    mp = _typed_book([(40, 120, 1e8)], copies=5)   # n_mp = 5
    with pytest.raises(ValueError, match="n_clusters"):
        fcf.compress(mp, _basis(), n_clusters=k)
