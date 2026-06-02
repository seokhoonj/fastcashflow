"""VFA (Variable Fee Approach) -- account-value contracts with guarantees.

The bundled VFA sample is a small book of single-premium variable annuities:
the account value grows at the underlying-items return less the fund fee, and
every exit pays the account value -- floored by a minimum credited rate, a
death (GMDB) and a maturity (GMAB) guarantee. The variable fee the entity
keeps is the source of the CSM; the time value of the guarantees (TVOG) is
read off a set of underlying-return scenarios.

    python examples/vfa.py
"""
import numpy as np

import fastcashflow as fcf


def main() -> None:
    mp = fcf.samples.model_points(template="vfa")
    basis = fcf.samples.basis(template="vfa")

    # Deterministic VFA measurement -- the headline liability and CSM.
    m = fcf.vfa.measure(mp, basis)
    print("VFA measurement -- variable annuities with GMDB / GMAB")
    print(f"  account value   {mp.account_value.sum():>16,.0f}")
    print(f"  BEL             {m.bel_path[:, 0].sum():>16,.0f}")
    print(f"  RA              {m.ra_path[:, 0].sum():>16,.0f}")
    print(f"  CSM             {m.csm_path[:, 0].sum():>16,.0f}")
    print(f"  loss component  {m.loss_component.sum():>16,.0f}")

    # Time value of the guarantees -- the put cost over return scenarios.
    rng = np.random.default_rng(7)
    monthly_return = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    n_time = int(mp.term_months.max())
    scenarios = monthly_return + rng.normal(0.0, 0.012, size=(2_000, n_time))
    tvog = fcf.vfa.tvog(mp, basis, scenarios)
    print("\nTVOG -- time value of the minimum-rate / GMDB / GMAB guarantees")
    print(f"  intrinsic value {tvog.intrinsic_value:>16,.0f}")
    print(f"  time value      {tvog.time_value:>16,.0f}")
    print(f"  total value     {tvog.total_value:>16,.0f}")

    # Walk one contract's measurement as a tree. The return scenarios must be
    # as wide as the contract being traced (its own projection horizon).
    term0 = int(mp.term_months[0])
    scen0 = monthly_return + rng.normal(0.0, 0.012, size=(2_000, term0))
    print()
    fcf.vfa.trace(0, mp, basis, return_scenarios=scen0)


if __name__ == "__main__":
    main()
