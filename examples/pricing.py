"""Pricing -- solve the level premium for three objectives.

The inputs are the bundled sample portfolio (``fcf.samples``).

    python examples/pricing.py
"""
import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis()
    book = fcf.samples.model_points()
    print(f"solving the level monthly premium for {book.n_mp} model points")
    print("(first model point shown)\n")

    break_even = fcf.solve_premium(book, basis, break_even=True)
    print(f"  break-even          {break_even[0]:>12,.0f}")

    margin = fcf.solve_premium(book, basis, margin=0.10)
    print(f"  10% profit margin   {margin[0]:>12,.0f}")

    target = fcf.solve_premium(book, basis, csm=2_000_000.0)
    print(f"  CSM of 2,000,000    {target[0]:>12,.0f}")


if __name__ == "__main__":
    main()
