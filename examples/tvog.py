"""Time value of a guarantee (TVOG) -- a VFA minimum-rate guarantee.

The VFA book is account_values.csv -- account-value contracts carry no
coverage-code coverages, so a single policies file read by
read_vfa_model_points. The guaranteed rate is a per-policy contract term --
the ``minimum_crediting_rate`` field -- locked at issue and varying by issue
cohort. The return scenarios are generated here -- a 2,000 x term grid is not
something to keep in a spreadsheet.

    python examples/tvog.py
"""
from pathlib import Path

import numpy as np

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_basis(DATA / "basis.xlsx")
    basis = basis[("TERM_LIFE_A", "FC")]
    account = fcf.read_vfa_model_points(DATA / "account_values.csv",
                                        calculation_methods=DATA / "calculation_methods.csv")

    # 2,000 monthly underlying-items return paths around the central return.
    rng = np.random.default_rng(7)
    monthly_return = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    n_time = int(account.term_months.max())
    scenarios = monthly_return + rng.normal(0.0, 0.012, size=(2_000, n_time))

    res = fcf.vfa.tvog(account, basis, scenarios)
    print("TVOG -- the minimum-rate guarantee from basis.xlsx")
    print(f"  intrinsic value  {res.intrinsic_value:>16,.0f}")
    print(f"  time value       {res.time_value:>16,.0f}")
    print(f"  total value      {res.total_value:>16,.0f}")


if __name__ == "__main__":
    main()
