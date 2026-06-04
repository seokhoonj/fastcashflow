"""Transition -- re-set the CSM on the fair value approach at first adoption.

The inputs are the bundled sample portfolio (``fcf.samples``).

    python examples/transition.py
"""
import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis()
    book = fcf.samples.model_points()

    # Measure the in-force book at the transition date.
    m = fcf.gmm.measure(book, basis)

    # The fair value of each contract. In practice it comes from a
    # fair-value exercise; here it is the fulfilment cash flows plus a margin.
    fcf0 = m.bel_path[:, 0] + m.ra_path[:, 0]
    fair_value = fcf0 + 1_000_000.0

    transitioned = fcf.transition(m, fair_value)
    print("transition -- CSM re-set on the fair value approach")
    print(f"  CSM at transition  {transitioned.csm_path[:, 0].sum():>16,.0f}")
    print(f"  loss component     {transitioned.loss_component.sum():>16,.0f}")


if __name__ == "__main__":
    main()
