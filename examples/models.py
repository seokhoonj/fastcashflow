"""The three IFRS 17 measurement models -- GMM, PAA and VFA.

GMM and PAA read the protection book from examples/data/ (policies +
coverages); VFA uses the bundled account-value sample.

    python examples/models.py
"""
from pathlib import Path

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_basis(DATA / "basis.xlsx")
    basis = basis[("TERM_LIFE_A", "FC")]
    book = fcf.read_model_points(DATA / "policies.csv", coverages=DATA / "coverages.csv", calculation_methods=DATA / "calculation_methods.csv")

    # GMM -- the general measurement model.
    gmm = fcf.gmm.measure(book, basis)
    print(f"GMM  -- CSM                       {gmm.csm_path[:, 0].sum():>14,.0f}")

    # PAA -- the simplified model for short-coverage business.
    paa = fcf.paa.measure(book, basis)
    print(f"PAA  -- insurance service result  {paa.service_result.sum():>14,.0f}")

    # VFA -- account-value (direct-participation) contracts.
    account = fcf.samples.model_points("vfa")
    vfa = fcf.vfa.measure(account, basis)
    print(f"VFA  -- CSM (the variable fee)    {vfa.csm_path[:, 0].sum():>14,.0f}")


if __name__ == "__main__":
    main()
