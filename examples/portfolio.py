"""Portfolio at scale -- the fast measure() path and writing results out.

Inputs are in examples/data/ (Excel files).

    python examples/portfolio.py
"""
import tempfile
from pathlib import Path

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_basis(DATA / "basis.xlsx")
    basis = basis[("TERM_LIFE_A", "FC")]
    book = fcf.read_model_points(DATA / "model_points_wide.xlsx", calculation_methods=DATA / "calculation_methods.csv")

    # measure() is the fast path -- BEL/RA/CSM/loss component per model point,
    # with no per-month trajectories materialised.
    val = fcf.gmm.measure(book, basis, full=False)
    print(f"measure() -- {book.n_mp} model points,  total CSM {val.csm.sum():,.0f}")

    # Write the per-model-point results to a file.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results.csv"
        fcf.write_measurement(val, out)
        print(f"  per-model-point results written to {out.name}")

    # For portfolios past what memory holds, gmm.measure_stream streams a parquet
    # file chunk by chunk straight to a result dataset.


if __name__ == "__main__":
    main()
