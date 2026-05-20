"""Phase 3 -- the fused fast path (`value`) agrees with the detailed `run`.

`run` is anchored by hand calculation (test_phase0 / test_phase1). `value`
is then validated transitively: it must reproduce `run`'s headline numbers,
and the GPU backend must reproduce the CPU backend.
"""
import numpy as np
import pytest
from numba import cuda

from fastcashflow import Assumptions, ModelPointSet, run, value


def test_value_matches_run():
    """The fast fused path reproduces the detailed path's headline numbers."""
    def mortality_monthly(issue_age, duration):
        attained = issue_age + duration
        annual_q = 0.0008 * (1.0 + 0.05 * (attained - 30.0))
        return 1.0 - (1.0 - annual_q) ** (1.0 / 12.0)

    asmp = Assumptions(
        mortality_monthly=mortality_monthly,
        lapse_monthly=lambda duration: np.full(duration.shape, 0.012),
        discount_annual=0.03,
        expense_acquisition=250_000.0,
        expense_maintenance_annual=48_000.0,
        expense_inflation=0.02,
        ra_confidence=0.85,
        claims_cv=0.12,
    )
    # distinct and repeated issue ages -- exercises the unique-age grid
    mps = ModelPointSet(
        issue_age=np.array([30, 45, 45, 55, 38]),
        sum_assured=np.array([1e8, 5e7, 8e7, 3e7, 6e7]),
        monthly_premium=np.array([70_000, 90_000, 110_000, 130_000, 80_000]),
        term_months=np.array([120, 120, 120, 120, 120]),
    )

    fast = value(mps, asmp)
    detailed = run(mps, asmp)

    assert np.allclose(fast.bel, detailed.bel)
    assert np.allclose(fast.ra, detailed.ra)
    assert np.allclose(fast.csm, detailed.csm0)
    assert np.allclose(fast.loss_component, detailed.loss_component)


def test_value_onerous():
    """The fast path also flags onerous contracts -- CSM floored at 0."""
    asmp = Assumptions(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.05),
        lapse_monthly=lambda duration: np.full(duration.shape, 0.0),
        discount_annual=0.0,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        claims_cv=0.05,
    )
    mps = ModelPointSet.single(
        issue_age=40, sum_assured=1_000_000.0,
        monthly_premium=100.0, term_months=12,
    )
    v = value(mps, asmp)
    assert v.csm[0] == 0.0
    assert v.loss_component[0] > 0.0


@pytest.mark.skipif(not cuda.is_available(), reason="no CUDA device available")
def test_value_gpu_matches_cpu():
    """The GPU backend reproduces the CPU backend exactly."""
    def mortality_monthly(issue_age, duration):
        attained = issue_age + duration
        annual_q = 0.0008 * (1.0 + 0.05 * (attained - 30.0))
        return 1.0 - (1.0 - annual_q) ** (1.0 / 12.0)

    asmp = Assumptions(
        mortality_monthly=mortality_monthly,
        lapse_monthly=lambda duration: np.full(duration.shape, 0.012),
        discount_annual=0.03,
        expense_acquisition=250_000.0,
        expense_maintenance_annual=48_000.0,
        expense_inflation=0.02,
        ra_confidence=0.85,
        claims_cv=0.12,
    )
    rng = np.random.default_rng(7)
    n = 5_000
    mps = ModelPointSet(
        issue_age=rng.integers(25, 60, n),
        sum_assured=rng.integers(10, 100, n) * 1_000_000,
        monthly_premium=rng.integers(3, 15, n) * 10_000,
        term_months=np.full(n, 120),
    )

    cpu = value(mps, asmp, backend="cpu")
    gpu = value(mps, asmp, backend="gpu")

    assert np.allclose(gpu.bel, cpu.bel)
    assert np.allclose(gpu.ra, cpu.ra)
    assert np.allclose(gpu.csm, cpu.csm)
    assert np.allclose(gpu.loss_component, cpu.loss_component)
