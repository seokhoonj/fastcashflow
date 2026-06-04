"""Quickstart -- load the bundled sample and measure.

The inputs are the bundled sample portfolio (``fcf.samples``). To work from
your own figures, write the sample out with ``fcf.samples.export("my_dir")``,
edit the files, and read them back with ``fcf.read_basis`` /
``fcf.read_model_points`` -- there is no Python to edit.

    python examples/quickstart.py
"""
import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis()                 # per-segment {(product, channel): Basis}
    model_points = fcf.samples.model_points()   # the multi-segment protection book

    m = fcf.gmm.measure(model_points, basis)
    print(f"measured {model_points.n_mp} model points -- portfolio totals at issue")
    print(f"  BEL  {m.bel_path[:, 0].sum():>16,.0f}")
    print(f"  RA   {m.ra_path[:, 0].sum():>16,.0f}")
    print(f"  CSM  {m.csm_path[:, 0].sum():>16,.0f}")


if __name__ == "__main__":
    main()
