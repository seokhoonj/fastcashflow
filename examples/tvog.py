"""Time value of a guarantee (TVOG) -- a VFA minimum-rate guarantee.

Inputs are in examples/data/. The guaranteed rate is a per-policy contract
term -- the ``guaranteed_credit_rate`` column on account_values.xlsx --
locked at issue and varying by issue cohort. The return scenarios are
generated here -- a 2,000 x term grid is not something to keep in a
spreadsheet.

    python examples/tvog.py
"""
from pathlib import Path

import numpy as np

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_assumptions(DATA / "assumptions.xlsx")
    assumptions = basis[("TERM_LIFE_A", "FC")]
    account = fcf.read_model_points(DATA / "account_values.xlsx", assumptions, benefit_patterns=DATA / "benefit_patterns.csv")

    # 2,000 monthly underlying-items return paths around the central return.
    rng = np.random.default_rng(7)
    monthly_return = (1.0 + assumptions.investment_return) ** (1.0 / 12.0) - 1.0
    n_time = int(account.term_months.max())
    scenarios = monthly_return + rng.normal(0.0, 0.012, size=(2_000, n_time))

    res = fcf.measure_tvog(account, assumptions, scenarios)
    print("TVOG -- the minimum-rate guarantee from assumptions.xlsx")
    print(f"  intrinsic value  {res.intrinsic_value:>16,.0f}")
    print(f"  time value       {res.time_value:>16,.0f}")
    print(f"  total value      {res.total_value:>16,.0f}")


if __name__ == "__main__":
    main()
