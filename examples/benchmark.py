"""Benchmark -- fast valuation of large synthetic portfolios.

Times the CPU backend, and the GPU backend too when a CUDA device is found.

Run from the project root::

    python examples/benchmark.py
"""
import time

import numpy as np
from numba import cuda

from fastcashflow import Assumptions, ModelPointSet, value


def mortality_monthly(sex: np.ndarray, issue_age: np.ndarray, duration: np.ndarray) -> np.ndarray:
    attained = issue_age + duration
    annual_q = 0.0005 * (1.0 + 0.04 * (attained - 30.0))
    return 1.0 - (1.0 - annual_q) ** (1.0 / 12.0)


def make_portfolio(n_mp: int, seed: int = 42) -> ModelPointSet:
    rng = np.random.default_rng(seed)
    return ModelPointSet(
        issue_age=rng.integers(25, 60, n_mp),
        death_benefit=rng.integers(10, 100, n_mp) * 1_000_000,
        monthly_premium=rng.integers(3, 15, n_mp) * 10_000,
        term_months=np.full(n_mp, 120),
    )


def _time(mps: ModelPointSet, asmp: Assumptions, backend: str) -> float:
    value(mps, asmp, backend=backend)        # warm-up (triggers compilation)
    start = time.perf_counter()
    value(mps, asmp, backend=backend)
    return time.perf_counter() - start


def main() -> None:
    asmp = Assumptions(
        mortality_monthly=mortality_monthly,
        lapse_monthly=lambda duration: np.full(duration.shape, 0.01),
        discount_annual=0.03,
        expense_acquisition=300_000.0,
        expense_maintenance_annual=60_000.0,
        expense_inflation=0.02,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )
    gpu = cuda.is_available()
    print("fastcashflow benchmark -- value(), term = 120 months"
          + ("  (CPU + GPU)" if gpu else "  (CPU only)"))

    for n_mp in (10_000, 100_000, 1_000_000, 5_000_000):
        mps = make_portfolio(n_mp)
        cpu_t = _time(mps, asmp, "cpu")
        line = f"  {n_mp:>10,} MP  ({n_mp * 120:>14,} cells) : CPU {cpu_t:8.3f} s"
        if gpu:
            gpu_t = _time(mps, asmp, "gpu")
            line += f"  |  GPU {gpu_t:8.3f} s  ({cpu_t / gpu_t:5.1f}x)"
        print(line)


if __name__ == "__main__":
    main()
