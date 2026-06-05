"""The three IFRS 17 measurement models -- GMM, PAA and VFA.

GMM and PAA measure the protection book; VFA the account-value book. All come
from the bundled sample (``fcf.samples``). GMM and PAA here take a single
:class:`Basis`, so they value one segment of the protection book.

    python examples/models.py
"""
import numpy as np

import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis()[("TERM_LIFE_A", "FC")]
    book = fcf.samples.model_points()
    seg = np.where((book.product == "TERM_LIFE_A") & (book.channel == "FC"))[0]
    book = book.subset(seg)

    # GMM -- the general measurement model.
    gmm = fcf.gmm.measure(book, basis)
    print(f"GMM  -- CSM                       {gmm.csm_path[:, 0].sum():>14,.0f}")

    # PAA -- the simplified model for short-coverage business.
    paa = fcf.paa.measure(book, basis)
    print(f"PAA  -- insurance service result  {paa.service_result.sum():>14,.0f}")

    # VFA -- account-value (direct-participation) contracts: their own book and basis.
    account = fcf.samples.model_points(template="vfa")
    vfa = fcf.vfa.measure(account, fcf.samples.basis(template="vfa"))
    print(f"VFA  -- CSM (the variable fee)    {vfa.csm_path[:, 0].sum():>14,.0f}")


if __name__ == "__main__":
    main()
