"""The three IFRS 17 measurement models -- GMM, PAA and VFA.

Inputs are in examples/data/ (Excel files). GMM and PAA read the
protection book; VFA reads the account-value book.

    python examples/models.py
"""
from pathlib import Path

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_assumptions(DATA / "assumptions.xlsx")
    assumptions = basis[("TERM_LIFE", "FC")]
    book = fcf.read_model_points(DATA / "model_points_wide.xlsx", assumptions)

    # GMM -- the general measurement model.
    gmm = fcf.measure(book, assumptions)
    print(f"GMM  -- CSM                       {gmm.csm[:, 0].sum():>14,.0f}")

    # PAA -- the simplified model for short-coverage business.
    paa = fcf.measure_paa(book, assumptions)
    print(f"PAA  -- insurance service result  {paa.service_result.sum():>14,.0f}")

    # VFA -- account-value (direct-participation) contracts.
    account = fcf.read_model_points(DATA / "account_values.xlsx", assumptions)
    vfa = fcf.measure_vfa(account, assumptions)
    print(f"VFA  -- CSM (the variable fee)    {vfa.csm[:, 0].sum():>14,.0f}")


if __name__ == "__main__":
    main()
