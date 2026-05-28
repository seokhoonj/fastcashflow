"""Quickstart -- read the inputs and measure.

The inputs live in examples/data/. Open assumptions.xlsx and
model_points_wide.xlsx in Excel, replace them with your own figures, and
run this again -- there is no Python to edit.

    python examples/quickstart.py
"""
from pathlib import Path

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_assumptions(DATA / "assumptions.xlsx")
    # The sample workbook now carries several (product, channel) segments;
    # pick TERM_LIFE / FC for this single-segment quickstart. A real run
    # over a multi-segment portfolio would use fcf.value_segmented.
    assumptions = basis[("TERM_LIFE_A", "FC")]
    model_points = fcf.read_model_points(DATA / "model_points_wide.xlsx", calculation_methods=DATA / "calculation_methods.csv")

    m = fcf.measure(model_points, assumptions)
    print(f"measured {model_points.n_mp} model points -- portfolio totals at issue")
    print(f"  BEL  {m.bel[:, 0].sum():>16,.0f}")
    print(f"  RA   {m.ra[:, 0].sum():>16,.0f}")
    print(f"  CSM  {m.csm[:, 0].sum():>16,.0f}")


if __name__ == "__main__":
    main()
