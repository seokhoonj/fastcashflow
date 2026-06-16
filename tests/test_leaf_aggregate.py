"""Leaf-model bounded-memory aggregates -- paa.measure_aggregate /
vfa.measure_aggregate.

Each is the per-model-point ``measure(full=True)`` trajectories summed over the
model-point axis, computed in ``chunk_size`` row-blocks. The result must not
depend on chunk_size and must reproduce the full measure's per-MP sums -- the
PAA / VFA analogue of ``gmm.measure_aggregate`` (which test_segmented_full
covers). A scalable sum, not a group remeasurement.
"""
import numpy as np

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, ModelPoints, CoverageRate


def _flat_basis(discount=0.05, investment_return=0.0):
    return Basis(
        mortality_annual=lambda s, ia, d: np.full(s.shape, 0.001),
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.0,
        investment_return=investment_return, fund_fee=0.015,
        coverages=(CoverageRate("DEATH", lambda s, ia, d: np.full(s.shape, 0.001)),))


def _book(n=7, *, account_value=False):
    """A small ragged book -- varying terms, some onerous (zero premium)."""
    rng_terms = np.array([36, 60, 24, 60, 48, 12, 60])[:n]
    premium = np.array([5000.0, 0.0, 3000.0, 0.0, 4000.0, 1000.0, 0.0])[:n]
    kw = dict(
        issue_age=np.full(n, 40),
        premium=premium,
        term_months=rng_terms,
        benefits={"DEATH": np.full(n, 1e4)},
        calculation_methods={"DEATH": CalculationMethod.DEATH})
    if account_value:
        kw["account_value"] = np.full(n, 1e6)
    return ModelPoints(**kw)


def test_paa_aggregate_matches_full_summed_and_chunk_independent():
    mp = _book()
    basis = _flat_basis()
    full = fcf.paa.measure(mp, basis, full=True)
    n_time = int(np.asarray(mp.contract_boundary_months).max())

    multi = fcf.paa.measure_aggregate(mp, basis, chunk_size=2)    # several chunks
    one = fcf.paa.measure_aggregate(mp, basis, chunk_size=10_000)  # single chunk

    # chunk-independent
    assert np.allclose(multi.lrc_path, one.lrc_path)
    assert np.allclose(multi.revenue, one.revenue)
    assert np.allclose(multi.service_expense, one.service_expense)
    assert np.allclose(multi.lic_path, one.lic_path)
    # reproduces the full measure summed over the model-point axis
    ref_lrc = np.zeros(n_time + 1)
    ref_lrc[:full.lrc_path.shape[1]] = full.lrc_path.sum(axis=0)
    assert np.allclose(multi.lrc_path, ref_lrc)
    assert np.isclose(multi.lrc, float(full.lrc.sum()))
    assert np.isclose(multi.loss_component, float(full.loss_component.sum()))
    assert np.isclose(multi.lrc, multi.lrc_path[0])          # col 0 = inception total


def test_paa_aggregate_rejects_non_positive_chunk_size():
    """chunk_size <= 0 would skip every block and return zero aggregates -- reject
    it instead of silently measuring nothing."""
    import pytest
    with pytest.raises(ValueError, match="chunk_size"):
        fcf.paa.measure_aggregate(_book(), _flat_basis(), chunk_size=0)


def test_gmm_aggregate_rejects_non_positive_chunk_size():
    import pytest
    with pytest.raises(ValueError, match="chunk_size"):
        fcf.gmm.measure_aggregate(_book(), _flat_basis(), chunk_size=0)


def test_vfa_aggregate_rejects_non_positive_chunk_size():
    import pytest
    with pytest.raises(ValueError, match="chunk_size"):
        fcf.vfa.measure_aggregate(_book(account_value=True),
                                  _flat_basis(investment_return=0.04), chunk_size=0)


def test_vfa_aggregate_matches_full_summed_and_chunk_independent():
    mp = _book(account_value=True)
    basis = _flat_basis(investment_return=0.04)
    full = fcf.vfa.measure(mp, basis, full=True)
    n_time = int(np.asarray(mp.contract_boundary_months).max())

    multi = fcf.vfa.measure_aggregate(mp, basis, chunk_size=2)
    one = fcf.vfa.measure_aggregate(mp, basis, chunk_size=10_000)

    assert np.allclose(multi.bel_path, one.bel_path)
    assert np.allclose(multi.csm_path, one.csm_path)
    assert np.allclose(multi.lic_path, one.lic_path)
    ref_bel = np.zeros(n_time + 1)
    ref_bel[:full.bel_path.shape[1]] = full.bel_path.sum(axis=0)
    assert np.allclose(multi.bel_path, ref_bel)
    assert np.isclose(multi.bel, float(full.bel.sum()))
    assert np.isclose(multi.csm, float(full.csm.sum()))
    assert np.isclose(multi.variable_fee, float(full.variable_fee.sum()))
    assert np.isclose(multi.time_value, float(full.time_value.sum()))
    assert np.isclose(multi.loss_component, float(full.loss_component.sum()))
    assert np.isclose(multi.bel, multi.bel_path[0])
