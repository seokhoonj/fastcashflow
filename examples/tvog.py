"""Time value of a guarantee (TVOG) -- a VFA minimum-rate guarantee.

The VFA book and basis come from the bundled sample (``fcf.samples``).
Account-value contracts carry no coverage-code coverages; the guaranteed rate
is a per-policy contract term -- the ``minimum_crediting_rate`` field -- locked
at issue. The return scenarios are generated here -- a 2,000 x term grid is not
something to keep in a spreadsheet.

    python examples/tvog.py
"""
import numpy as np

import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis(template="vfa")
    account = fcf.samples.model_points(template="vfa")

    # 2,000 monthly underlying-items return paths around the central return.
    rng = np.random.default_rng(7)
    monthly_return = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    n_time = int(account.term_months.max())
    scenarios = monthly_return + rng.normal(0.0, 0.012, size=(2_000, n_time))

    res = fcf.vfa.tvog(account, basis, scenarios)
    print("TVOG -- the minimum-rate guarantee")
    print(f"  intrinsic value  {res.intrinsic_value:>16,.0f}")
    print(f"  time value       {res.time_value:>16,.0f}")
    print(f"  total value      {res.total_value:>16,.0f}")


if __name__ == "__main__":
    main()
