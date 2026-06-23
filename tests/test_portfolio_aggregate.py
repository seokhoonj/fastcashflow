"""Contract skeleton for fcf.portfolio.measure_aggregate -- P-5b Gate B(2).

The chunked *aggregate* view of a mixed portfolio: a **scalable sum of measured
model-point results**, computed in bounded memory so it works where a per-model
-point measure(full=True) would OOM. It holds no per-model-point row -- only each
model's inception totals and the (n_time+1,) aggregate run-off trajectories,
summed over the model-point axis.

Deliberately scoped. The aggregate is **not an IFRS group remeasurement** and
**not a group re-floor engine**: every figure is the sum of the per-model-point
results (so CSM is the sum of each contract's floored CSM, matching the measure()
headline -- NOT group()'s CSM(sum FCF)). A genuine per-group re-floor (floor on the
group's fulfilment cash flows, keyed by portfolio / annual cohort / profitability)
is a separate concern -- see [[assumption-unit-vs-csm-unit]].

This file was written as the contract before the implementation (Codex's
skeleton-first order); the implementation then activated it unchanged.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, ModelPoints, CoverageRate
from fastcashflow.basis import BasisRouter
# Native aggregate types are public on each model namespace (fcf.gmm.Aggregate,
# fcf.paa.Aggregate, fcf.vfa.Aggregate); the portfolio orchestrator entry +
# container live in the fcf.portfolio namespace (like PortfolioMeasurement).
from fastcashflow.portfolio import measure, measure_aggregate, PortfolioAggregate


def _flat_basis(discount=0.05, investment_return=0.0):
    return Basis(
        mortality_annual=lambda s, ia, d: np.full(s.shape, 0.001),
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.0,
        investment_return=investment_return, fund_fee=0.015,
        coverages=(CoverageRate("DEATH", lambda s, ia, d: np.full(s.shape, 0.001)),))


def _three_model_inputs():
    """One mixed book: GMM (onerous, premium 0 + claims), PAA, two VFA rows."""
    router = BasisRouter(
        {("G", "GA"): _flat_basis(),
         ("P", "GA"): _flat_basis(),
         ("V", "GA"): _flat_basis(investment_return=0.04)},
        measurement_models={("P", "GA"): "PAA", ("V", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(4, 40), premium=np.array([0.0, 1200.0, 0.0, 0.0]),
        term_months=np.full(4, 60), benefits={"DEATH": np.full(4, 1e4)},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        account_value=np.array([0.0, 0.0, 1e6, 1e6]),
        product=np.array(["G", "P", "V", "V"]),
        channel=np.array(["GA", "GA", "GA", "GA"]))
    return mp, router


# ---------------------------------------------------------------------------
# 1) master invariant: the aggregate IS the sum of the measured per-MP results
# ---------------------------------------------------------------------------
def test_aggregate_is_the_sum_of_full_per_mp_measurement():
    mp, router = _three_model_inputs()
    agg = measure_aggregate(mp, router)
    full = measure(mp, router, full=True)
    assert isinstance(agg, PortfolioAggregate)

    # GMM block -- reuses engine.Aggregate
    assert isinstance(agg.gmm, fcf.gmm.Aggregate)
    assert np.isclose(agg.gmm.bel, full.gmm.measurement.bel.sum())
    assert np.isclose(agg.gmm.csm, full.gmm.measurement.csm.sum())
    assert np.allclose(agg.gmm.bel_path,
                       full.gmm.measurement.bel_path.sum(axis=0))

    # PAA block
    assert isinstance(agg.paa, fcf.paa.Aggregate)
    assert np.isclose(agg.paa.lrc, full.paa.measurement.lrc.sum())
    assert np.allclose(agg.paa.lrc_path,
                       full.paa.measurement.lrc_path.sum(axis=0))
    assert np.allclose(agg.paa.lic_path, full.paa.measurement.lic_path.sum(axis=0))

    # VFA block (note: lic_path is carried here too -- VFA full measurement has it)
    assert isinstance(agg.vfa, fcf.vfa.Aggregate)
    assert np.isclose(agg.vfa.csm, full.vfa.measurement.csm.sum())
    assert np.allclose(agg.vfa.csm_path,
                       full.vfa.measurement.csm_path.sum(axis=0))
    assert np.allclose(agg.vfa.lic_path, full.vfa.measurement.lic_path.sum(axis=0))
    # No account_value_path on the aggregate -- a per-policy level, not a clean
    # group/aggregate quantity (group() drops it for the same reason).
    assert not hasattr(agg.vfa, "account_value_path")


def test_aggregate_ragged_terms_pad_into_leading_slice():
    """Mixed coverage terms within a model: each (shorter) block's path adds into
    the leading slice of the global run-off. chunk_size=1 puts every distinct
    horizon in its own block, and the aggregate still equals the full per-MP
    measurement summed -- pinning the ragged padding directly."""
    router = BasisRouter(
        {("P", "GA"): _flat_basis(),
         ("V", "GA"): _flat_basis(investment_return=0.04)},
        measurement_models={("P", "GA"): "PAA", ("V", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(4, 40),
        premium=np.array([1200.0, 1200.0, 0.0, 0.0]),
        term_months=np.array([24, 60, 36, 60]),      # ragged within each model
        benefits={"DEATH": np.full(4, 1e4)},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        account_value=np.array([0.0, 0.0, 1e6, 1e6]),
        product=np.array(["P", "P", "V", "V"]),
        channel=np.array(["GA", "GA", "GA", "GA"]))
    agg = measure_aggregate(mp, router, chunk_size=1)   # a block per row
    full = measure(mp, router, full=True)
    assert np.allclose(agg.paa.lrc_path, full.paa.measurement.lrc_path.sum(axis=0))
    assert np.allclose(agg.paa.lic_path, full.paa.measurement.lic_path.sum(axis=0))
    assert np.allclose(agg.vfa.bel_path, full.vfa.measurement.bel_path.sum(axis=0))
    assert np.allclose(agg.vfa.csm_path, full.vfa.measurement.csm_path.sum(axis=0))
    # the aggregate is also chunk-invariant under ragged terms
    big = measure_aggregate(mp, router, chunk_size=1000)
    assert np.allclose(agg.vfa.csm_path, big.vfa.csm_path)
    assert np.allclose(agg.paa.lrc_path, big.paa.lrc_path)


# ---------------------------------------------------------------------------
# 2) chunking is a numeric no-op (only peak memory changes)
# ---------------------------------------------------------------------------
def test_aggregate_chunking_is_numeric_noop():
    mp, router = _three_model_inputs()
    a = measure_aggregate(mp, router, chunk_size=1)
    b = measure_aggregate(mp, router, chunk_size=10_000)
    assert np.isclose(a.gmm.bel, b.gmm.bel) and np.allclose(a.gmm.csm_path, b.gmm.csm_path)
    assert np.isclose(a.paa.lrc, b.paa.lrc) and np.allclose(a.paa.lrc_path, b.paa.lrc_path)
    assert np.isclose(a.vfa.csm, b.vfa.csm) and np.allclose(a.vfa.bel_path, b.vfa.bel_path)


def test_aggregate_rejects_non_positive_chunk_size():
    mp, router = _three_model_inputs()
    with pytest.raises(ValueError, match="chunk_size"):
        measure_aggregate(mp, router, chunk_size=0)


# ---------------------------------------------------------------------------
# 3) the one cross-model additive figure, and no cross-model pooling otherwise
# ---------------------------------------------------------------------------
def test_loss_component_total_is_the_only_cross_model_sum():
    mp, router = _three_model_inputs()
    agg = measure_aggregate(mp, router)
    expected = agg.gmm.loss_component + agg.paa.loss_component + agg.vfa.loss_component
    assert np.isclose(agg.loss_component_total(), expected)
    # equals the per-MP measurement's portfolio loss total (aggregate == sum)
    assert np.isclose(agg.loss_component_total(),
                      measure(mp, router, full=True).loss_component_total())


def test_summary_keeps_each_model_in_its_own_block():
    mp, router = _three_model_inputs()
    s = measure_aggregate(mp, router).summary()
    assert set(s) == {"loss_component_total", "gmm", "paa", "vfa"}
    assert set(s["paa"]) == {"lrc", "loss_component"}        # LRC, never BEL/CSM
    assert "bel" in s["gmm"] and "bel" in s["vfa"]
    # there is no flat field where a BEL and an LRC could be added
    assert not hasattr(measure_aggregate(mp, router), "bel")


# ---------------------------------------------------------------------------
# 4) NOT a group re-floor: CSM is the sum of per-MP floored CSM (measure headline),
#    not group()'s CSM(sum FCF)
# ---------------------------------------------------------------------------
def test_aggregate_csm_is_per_mp_floor_sum_not_group_refloor():
    mp, router = _three_model_inputs()
    agg = measure_aggregate(mp, router)
    full = measure(mp, router, full=True)
    # identical to summing the per-MP (per-contract floored) CSM -- the measure()
    # headline aggregated, deliberately NOT a group-level CSM(sum FCF).
    assert np.isclose(agg.gmm.csm, full.gmm.measurement.csm.sum())
    assert np.isclose(agg.vfa.csm, full.vfa.measurement.csm.sum())


# ---------------------------------------------------------------------------
# 5) absent models -> absent slots (no fabricated zero aggregate)
# ---------------------------------------------------------------------------
def test_aggregate_omits_absent_models():
    router = BasisRouter({("A", "GA"): _flat_basis()})
    mp = ModelPoints(
        issue_age=np.full(2, 40), premium=np.zeros(2), term_months=np.full(2, 60),
        benefits={"DEATH": np.full(2, 1e4)},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        product=np.array(["A", "A"]), channel=np.array(["GA", "GA"]))
    agg = measure_aggregate(mp, router)
    assert agg.gmm is not None and agg.paa is None and agg.vfa is None
    assert set(agg.summary()) == {"loss_component_total", "gmm"}
