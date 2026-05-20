"""Benchmark -- project large synthetic portfolios and report run times.

Run from the project root::

    python examples/benchmark.py
"""
import time

import numpy as np

from fastcashflow import Assumptions, ModelPointSet, run


def mortality_monthly(ages: np.ndarray) -> np.ndarray:
    annual_q = 0.0005 * (1.0 + 0.04 * (ages - 30.0))
    return 1.0 - (1.0 - annual_q) ** (1.0 / 12.0)


def make_portfolio(n_mp: int, seed: int = 42) -> ModelPointSet:
    rng = np.random.default_rng(seed)
    return ModelPointSet(
        issue_age=rng.integers(25, 60, n_mp),
        sum_assured=rng.integers(10, 100, n_mp) * 1_000_000,
        monthly_premium=rng.integers(3, 15, n_mp) * 10_000,
        term_months=np.full(n_mp, 120),
    )


def main() -> None:
    asmp = Assumptions(
        mortality_monthly=mortality_monthly,
        lapse_monthly=0.01,
        discount_annual=0.03,
        expense_acquisition=300_000.0,
        expense_maintenance_annual=60_000.0,
        expense_inflation=0.02,
        ra_confidence=0.75,
        claims_cv=0.10,
    )

    print("fastcashflow benchmark -- single core, term = 120 months")
    for n_mp in (10_000, 50_000, 200_000, 500_000):
        mps = make_portfolio(n_mp)
        run(mps, asmp)                       # warm-up (triggers JIT compilation)
        start = time.perf_counter()
        run(mps, asmp)
        elapsed = time.perf_counter() - start
        rows = n_mp * 120
        print(f"  {n_mp:>9,} MP  ({rows:>12,} cells) : {elapsed:8.3f} s")


if __name__ == "__main__":
    main()
