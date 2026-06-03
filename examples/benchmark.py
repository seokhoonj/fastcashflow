"""Benchmark -- fast valuation of large synthetic portfolios.

Times the CPU backend, and the GPU backend too when a CUDA device is found.

Run from the project root::

    python examples/benchmark.py
"""
import time

import numpy as np
from numba import cuda

from fastcashflow import Basis, CalculationMethod, CoverageRate, ExpenseItem, ModelPoints
from fastcashflow.gmm import measure


def _annual(m):
    return 1.0 - (1.0 - m) ** 12


def mortality_annual(sex, issue_age, duration, issue_class, elapsed) -> np.ndarray:
    # The engine calls every rate with the full (sex, issue_age, duration,
    # issue_class, elapsed) signature; this table depends only on attained age,
    # so it ignores issue_class / elapsed.
    attained = issue_age + duration
    annual_q = 0.0005 * (1.0 + 0.04 * (attained - 30.0))
    return annual_q


def make_portfolio(n_mp: int, seed: int = 42) -> ModelPoints:
    rng = np.random.default_rng(seed)
    return ModelPoints(
        issue_age=rng.integers(25, 60, n_mp),
        benefits={0: rng.integers(10, 100, n_mp) * 1_000_000},
        level_premium=rng.integers(3, 15, n_mp) * 10_000,
        term_months=np.full(n_mp, 120),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )


def _time(model_points: ModelPoints, basis: Basis, backend: str) -> float:
    measure(model_points, basis, backend=backend, full=False)        # warm-up (triggers compilation)
    start = time.perf_counter()
    measure(model_points, basis, backend=backend, full=False)
    return time.perf_counter() - start


def main() -> None:
    basis = Basis(
        mortality_annual=mortality_annual,
        lapse_annual=lambda sex, issue_age, duration, issue_class, elapsed: np.full(duration.shape, _annual(0.01)),
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    300_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  60_000.0),
        ),
        ra_confidence=0.75,
        mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", mortality_annual),),
    )
    gpu = cuda.is_available()
    print("fastcashflow benchmark -- measure(), term = 120 months"
          + ("  (CPU + GPU)" if gpu else "  (CPU only)"))

    for n_mp in (10_000, 100_000, 1_000_000, 5_000_000):
        model_points = make_portfolio(n_mp)
        cpu_t = _time(model_points, basis, "cpu")
        line = f"  {n_mp:>10,} MP  ({n_mp * 120:>14,} cells) : CPU {cpu_t:8.3f} s"
        if gpu:
            gpu_t = _time(model_points, basis, "gpu")
            line += f"  |  GPU {gpu_t:8.3f} s  ({cpu_t / gpu_t:5.1f}x)"
        print(line)


if __name__ == "__main__":
    main()
