"""Portfolio at scale -- the fast measure() path and writing results out.

The inputs are the bundled sample portfolio (``fcf.samples``).

    python examples/portfolio.py
"""
import tempfile
from pathlib import Path

import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis()
    book = fcf.samples.model_points()

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
