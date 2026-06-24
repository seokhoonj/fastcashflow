"""Stochastic solvency -- the ESG -> VFA -> coverage-ratio pipeline.

Generate risk-neutral economic scenarios, value a variable (VFA) book's
guarantee under each, and read the coverage-ratio DISTRIBUTION (mean, a lower
percentile, and the conditional tail expectation) -- the stochastic view the
prescribed-SCR t=0 ratio cannot show.

    python examples/stochastic_solvency.py
"""
import numpy as np

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, ModelPoints, esg,
)
from fastcashflow.alm import Bond
from fastcashflow.assets import AssetPortfolio
from fastcashflow import solvency as sv


def main() -> None:
    # A single-premium variable annuity with a 2% crediting guarantee and a GMAB
    # above the account value -- a guarantee that bites when fund returns are poor.
    basis = Basis(mortality_annual=0.004, lapse_annual=0.02, discount_annual=0.03,
                  ra_confidence=0.75, mortality_cv=0.1, investment_return=0.05,
                  fund_fee=0.015, coverages=(CoverageRate("DEATH", 0.004),))
    mp = ModelPoints.single(45, 0.0, 60, account_value=1e7,
                            minimum_accumulation_benefit=1.2e7,
                            minimum_crediting_rate=0.02,
                            benefits={"DEATH": 0.0},
                            calculation_methods={"DEATH": CalculationMethod.DEATH})
    portfolio = AssetPortfolio(holdings=(
        Bond(face=1.5e7, coupon_rate=0.04, maturity_years=5, frequency=1),))

    # Risk-neutral scenarios (Hull-White rates + lognormal fund returns).
    scenarios = esg.simulate(
        np.array([1., 2., 3., 5., 10., 20.]),
        np.array([0.031, 0.0355, 0.0368, 0.039, 0.0408, 0.041]),
        ufr=0.0405, alpha=0.10, mean_reversion=0.10, rate_vol=0.01,
        equity_vol=0.15, correlation=-0.2, n_scenarios=1000, n_time=60, seed=7)
    bond_err, equity_err = scenarios.martingale_error()
    print(f"ESG no-arbitrage check  bond {bond_err:.2e}  equity {equity_err:.2e}")

    # The VFA liability distribution over the fund-return scenarios.
    dist = fcf.vfa.stochastic(mp, basis, scenarios.returns)
    print("VFA liability over scenarios")
    print(f"  BEL  mean {dist.mean()['bel']:>16,.0f}")
    print(f"  loss p95  {dist.percentile(95)['loss_component']:>16,.0f}")

    # The coverage-ratio distribution: the static t=0 ratio vs the stochastic tail.
    ss = fcf.vfa.stochastic_solvency(portfolio, mp, basis, scenarios, regime=sv.KICS)
    print("Coverage ratio")
    print(f"  static (t=0)   {ss.static.solvency_ratio:>8.2%}")
    print(f"  stochastic mean{ss.mean()['ratio']:>8.2%}")
    print(f"  5th percentile {ss.percentile(5)['ratio']:>8.2%}")
    print(f"  CTE 5%         {ss.cte(5):>8.2%}")


if __name__ == "__main__":
    main()
