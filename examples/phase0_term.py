"""Worked example -- a single level-premium protection policy.

Run from the project root::

    python examples/phase0_term.py
"""
import numpy as np

from fastcashflow import Assumptions, ModelPoints, measure


def main() -> None:
    # Illustrative age-based monthly mortality.
    def mortality_monthly(sex: np.ndarray, issue_age: np.ndarray, duration: np.ndarray) -> np.ndarray:
        attained = issue_age + duration
        annual_q = 0.0005 * (1.0 + 0.04 * (attained - 30.0))
        return 1.0 - (1.0 - annual_q) ** (1.0 / 12.0)

    assumptions = Assumptions(
        mortality_monthly=mortality_monthly,
        lapse_monthly=lambda duration: np.full(duration.shape, 0.01),
        discount_annual=0.03,
        expense_acquisition=300_000.0,
        expense_maintenance_annual=60_000.0,
        expense_inflation=0.02,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )

    model_points = ModelPoints.single(
        issue_age=40,
        death_benefit=100_000_000,
        monthly_premium=70_000,
        term_months=120,
    )

    res = measure(model_points, assumptions)

    print("fastcashflow -- single protection policy")
    print(f"  BEL            : {res.bel[0, 0]:>16,.0f}")
    print(f"  RA             : {res.ra[0, 0]:>16,.0f}")
    print(f"  FCF (BEL + RA) : {res.bel[0, 0] + res.ra[0, 0]:>16,.0f}")
    print(f"  CSM (t=0)      : {res.csm[0, 0]:>16,.0f}")
    print(f"  loss component : {res.loss_component[0]:>16,.0f}")
    print(f"  CSM[0..5]      : {np.round(res.csm[0, :6], 0)}")
    print(f"  CSM[t=term]    : {res.csm[0, -1]:>16,.2f}  (should be ~0)")


if __name__ == "__main__":
    main()
