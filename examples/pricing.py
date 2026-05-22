"""Pricing -- solve the level premium for three objectives.

Inputs are in examples/data/ (Excel files).

    python examples/pricing.py
"""
from pathlib import Path

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    assumptions = fcf.read_assumptions(DATA / "assumptions.xlsx")
    book = fcf.read_model_points(DATA / "model_points_wide.xlsx", assumptions)
    print(f"solving the level monthly premium for {book.n_mp} model points")
    print("(first model point shown)\n")

    break_even = fcf.solve_premium(book, assumptions, break_even=True)
    print(f"  break-even          {break_even[0]:>12,.0f}")

    margin = fcf.solve_premium(book, assumptions, margin=0.10)
    print(f"  10% profit margin   {margin[0]:>12,.0f}")

    target = fcf.solve_premium(book, assumptions, csm=2_000_000.0)
    print(f"  CSM of 2,000,000    {target[0]:>12,.0f}")


if __name__ == "__main__":
    main()
