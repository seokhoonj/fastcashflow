"""Reinsurance -- a quota-share treaty held over a direct portfolio.

Inputs are in examples/data/ (Excel files).

    python examples/reinsurance.py
"""
from pathlib import Path

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_assumptions(DATA / "assumptions.xlsx")
    assumptions = basis[("TERM_LIFE_A", "FC")]
    book = fcf.read_model_points(DATA / "model_points_wide.xlsx", assumptions, benefit_patterns=DATA / "benefit_patterns.csv")

    # A 30% quota-share cession of the direct book.
    reins = fcf.measure_reinsurance(book, assumptions, cession_rate=0.30)

    print("reinsurance held -- 30% quota share")
    print(f"  BEL (PV premiums - recoveries)  {reins.bel.sum():>16,.0f}")
    print(f"  RA  (risk transferred)          {reins.ra.sum():>16,.0f}")
    print(f"  CSM (net cost/gain of cover)    {reins.csm[:, 0].sum():>16,.0f}")


if __name__ == "__main__":
    main()
