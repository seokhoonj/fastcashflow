"""Transition -- re-set the CSM on the fair value approach at first adoption.

Inputs are in examples/data/ (Excel files).

    python examples/transition.py
"""
from pathlib import Path

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_assumptions(DATA / "assumptions.xlsx")
    assumptions, = basis.values()
    book = fcf.read_model_points(DATA / "model_points_wide.xlsx", assumptions)

    # Measure the in-force book at the transition date.
    m = fcf.measure(book, assumptions)

    # The fair value of each contract. In practice it comes from a
    # fair-value exercise; here it is the fulfilment cash flows plus a margin.
    fcf0 = m.bel[:, 0] + m.ra[:, 0]
    fair_value = fcf0 + 1_000_000.0

    transitioned = fcf.transition(m, fair_value)
    print("transition -- CSM re-set on the fair value approach")
    print(f"  CSM at transition  {transitioned.csm[:, 0].sum():>16,.0f}")
    print(f"  loss component     {transitioned.loss_component.sum():>16,.0f}")


if __name__ == "__main__":
    main()
