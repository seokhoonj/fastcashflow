"""Reinsurance -- a quota-share treaty held over a direct portfolio.

The inputs are the bundled sample portfolio (``fcf.samples``).
``reinsurance.measure`` takes a single :class:`Basis`, so this cedes one
segment of the book.

    python examples/reinsurance.py
"""
import numpy as np

import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "FC"))
    book = fcf.samples.model_points()
    seg = np.where((book.product == "TERM_LIFE_A") & (book.channel == "FC"))[0]
    book = book.subset(seg)

    # A 30% quota-share cession of the direct book.
    reins = fcf.reinsurance.measure(book, basis, treaty=fcf.reinsurance.QuotaShare(cession=0.30))

    print("reinsurance held -- 30% quota share")
    print(f"  BEL (PV premiums - recoveries)  {reins.bel.sum():>16,.0f}")
    print(f"  RA  (risk transferred)          {reins.ra.sum():>16,.0f}")
    print(f"  CSM (net cost/gain of cover)    {reins.csm_path[:, 0].sum():>16,.0f}")


if __name__ == "__main__":
    main()
