"""Stochastic valuation -- the liability across economic scenarios.

The inputs are the bundled sample portfolio (``fcf.samples``). ``gmm.stochastic``
takes a single :class:`Basis`, so this values one segment of the book.

    python examples/stochastic.py
"""
import numpy as np

import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "FC"))
    book = fcf.samples.model_points()
    seg = np.where((book.product == "TERM_LIFE_A") & (book.channel == "FC"))[0]
    book = book.subset(seg)

    # Value the book under a range of discount-rate scenarios.
    rates = np.array([0.02, 0.03, 0.04, 0.05])
    dist = fcf.gmm.stochastic(book, basis, rates)

    print("stochastic valuation -- BEL across discount-rate scenarios")
    for rate, bel in zip(rates, dist.bel):
        print(f"  discount {rate:>5.0%}   BEL {bel:>16,.0f}")


if __name__ == "__main__":
    main()
