"""Phase 3 -- the fused fast path (`value`) agrees with the detailed `measure`.

`measure` is anchored by hand calculation (test_phase0 / test_phase1). `value`
is then validated transitively: it must reproduce `measure`'s headline numbers,
and the GPU backend must reproduce the CPU backend.
"""
import numpy as np
import pytest
from numba import cuda

from conftest import annual_from_monthly as _annual
from fastcashflow import (
    STATE_MODELS,
    Basis,
    CalculationMethod,
    ExpenseItem,
    ModelPoints,
    CoverageRate,
    measure,
    value,
)


def test_value_matches_measure():
    """The fast fused path reproduces the detailed path's headline numbers."""
    def mortality_annual(sex, issue_age, duration):
        attained = issue_age + duration
        annual_q = 0.0008 * (1.0 + 0.05 * (attained - 30.0))
        return annual_q

    asmp = Basis(
        mortality_annual=mortality_annual,
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.012)),
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    250_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  48_000.0),
        ),
        ra_confidence=0.85,
        mortality_cv=0.12,
        coverages=(CoverageRate("DEATH", mortality_annual),),
    )
    # distinct and repeated issue ages -- exercises the unique-age grid
    mps = ModelPoints(
        issue_age=np.array([30, 45, 45, 55, 38]),
        benefits={0: np.array([1e8, 5e7, 8e7, 3e7, 6e7])},
        level_premium=np.array([70_000, 90_000, 110_000, 130_000, 80_000]),
        term_months=np.array([120, 120, 120, 120, 120]),
    )

    fast = value(mps, asmp)
    detailed = measure(mps, asmp)

    assert np.allclose(fast.bel, detailed.bel[:, 0])
    assert np.allclose(fast.ra, detailed.ra[:, 0])
    assert np.allclose(fast.csm, detailed.csm[:, 0])
    assert np.allclose(fast.loss_component, detailed.loss_component)


def test_value_onerous():
    """The fast path also flags onerous contracts -- CSM floored at 0."""
    asmp = Basis(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.05)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, 0.0),
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.05,
        coverages=(CoverageRate("DEATH", lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.05))),),
    )
    mps = ModelPoints.single(
        issue_age=40, benefits={0: 1_000_000.0},
        level_premium=100.0, term_months=12,
    )
    v = value(mps, asmp)
    assert v.csm[0] == 0.0
    assert v.loss_component[0] > 0.0


@pytest.mark.skipif(not cuda.is_available(), reason="no CUDA device available")
@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
def test_value_gpu_matches_cpu():
    """The GPU backend reproduces the CPU backend exactly."""
    def mortality_annual(sex, issue_age, duration):
        attained = issue_age + duration
        annual_q = 0.0008 * (1.0 + 0.05 * (attained - 30.0))
        return annual_q

    asmp = Basis(
        mortality_annual=mortality_annual,
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.012)),
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    250_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  48_000.0),
        ),
        ra_confidence=0.85,
        mortality_cv=0.12,
        coverages=(CoverageRate("DEATH", mortality_annual),),
    )
    rng = np.random.default_rng(7)
    n = 5_000
    mps = ModelPoints(
        issue_age=rng.integers(25, 60, n),
        benefits={0: rng.integers(10, 100, n) * 1_000_000},
        level_premium=rng.integers(3, 15, n) * 10_000,
        term_months=np.full(n, 120),
    )

    cpu = value(mps, asmp, backend="cpu")
    gpu = value(mps, asmp, backend="gpu")

    assert np.allclose(gpu.bel, cpu.bel)
    assert np.allclose(gpu.ra, cpu.ra)
    assert np.allclose(gpu.csm, cpu.csm)
    assert np.allclose(gpu.loss_component, cpu.loss_component)


@pytest.mark.skipif(not cuda.is_available(), reason="no CUDA device available")
@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
def test_value_gpu_matches_cpu_with_transition():
    """GPU and CPU agree under a waiver transition with a diagnosis coverage --
    the GPU two-track in-force and diagnosis pool reproduce the CPU kernel."""
    def flat(rate):
        return lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(rate))

    asmp = Basis(
        mortality_annual=flat(0.001),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.012)),
        waiver_incidence_annual=flat(0.02),
        state_model=STATE_MODELS["WAIVER"],
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    200_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  48_000.0),
        ),
        ra_confidence=0.85,
        mortality_cv=0.12,
        morbidity_cv=0.10,
        coverages=(
            CoverageRate("DEATH", flat(0.001)),
            CoverageRate("dx", flat(0.003)),
        ),
    )
    rng = np.random.default_rng(13)
    n = 4_000
    mps = ModelPoints(
        issue_age=rng.integers(25, 60, n).astype(float),
        benefits={0: rng.integers(10, 100, n) * 1_000_000.0, 1: rng.integers(5, 30, n) * 1_000_000.0},
        level_premium=rng.integers(3, 15, n) * 10_000.0,
        term_months=np.full(n, 120),
        state=rng.integers(0, 3, n),
        calculation_methods={"DEATH": CalculationMethod.DEATH, "dx": CalculationMethod.DIAGNOSIS},
    )
    cpu = value(mps, asmp, backend="cpu")
    gpu = value(mps, asmp, backend="gpu")

    assert np.allclose(gpu.bel, cpu.bel)
    assert np.allclose(gpu.ra, cpu.ra)
    assert np.allclose(gpu.csm, cpu.csm)
    assert np.allclose(gpu.loss_component, cpu.loss_component)
