"""Stochastic valuation -- the liability across economic scenarios.

Inputs are in examples/data/ (Excel files).

    python examples/stochastic.py
"""
from pathlib import Path

import numpy as np

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_assumptions(DATA / "assumptions.xlsx")
    assumptions = basis[("TERM_LIFE", "FC")]
    book = fcf.read_model_points(DATA / "model_points_wide.xlsx", assumptions)

    # Value the book under a range of discount-rate scenarios.
    rates = np.array([0.02, 0.03, 0.04, 0.05])
    dist = fcf.value_stochastic(book, assumptions, rates)

    print("stochastic valuation -- BEL across discount-rate scenarios")
    for rate, bel in zip(rates, dist.bel):
        print(f"  discount {rate:>5.0%}   BEL {bel:>16,.0f}")


if __name__ == "__main__":
    main()
