"""Phase 4 validation -- BEL/RA roll-forward and the liability runoff.

`measure()` now returns BEL, RA and CSM as month-by-month trajectories.
The BEL trajectory is anchored to an independent backward recursion, and
the total liability must run off to zero by the end of the term.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPointSet, measure, value


def _assumptions(**overrides) -> Assumptions:
    base = dict(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.002),
        lapse_monthly=lambda duration: np.full(duration.shape, 0.01),
        discount_annual=0.04,
        expense_acquisition=100_000.0,
        expense_maintenance_annual=24_000.0,
        expense_inflation=0.02,
        ra_confidence=0.80,
        claims_cv=0.10,
    )
    base.update(overrides)
    return Assumptions(**base)


def test_bel_rollforward():
    """The BEL trajectory matches an independent backward recursion."""
    asmp = _assumptions()
    one = ModelPointSet.single(
        issue_age=45, death_benefit=80_000_000,
        monthly_premium=150_000, term_months=36,
    )
    res = measure(one, asmp)

    i = asmp.discount_monthly
    half = (1.0 + i) ** -0.5
    full = 1.0 / (1.0 + i)
    cf = res.cashflows
    n_time = cf.n_time

    bel = np.zeros(n_time + 1)
    for t in range(n_time - 1, -1, -1):
        bel[t] = (
            (cf.claim_cf[0, t] + cf.expense_cf[0, t]) * half
            - cf.premium_cf[0, t]
            + bel[t + 1] * full
        )
    assert np.allclose(res.bel[0], bel)
    assert res.bel[0, -1] == 0.0

    # column 0 of the detailed trajectory equals the fast headline BEL
    assert np.isclose(res.bel[0, 0], value(one, asmp).bel[0])


def test_liability_runs_off():
    """BEL + RA + CSM fully runs off to zero by the end of the term."""
    asmp = _assumptions()
    rng = np.random.default_rng(4)
    n = 150
    mps = ModelPointSet(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(20, 100, n) * 1_000_000,
        monthly_premium=rng.integers(10, 25, n) * 10_000,
        term_months=rng.integers(48, 120, n),
    )
    res = measure(mps, asmp)

    liability = res.bel + res.ra + res.csm
    assert np.allclose(liability[:, -1], 0.0)
