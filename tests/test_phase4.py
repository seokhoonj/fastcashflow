"""Phase 4 validation -- BEL/RA roll-forward and the liability runoff.

`measure()` now returns BEL, RA and CSM as month-by-month trajectories.
The BEL trajectory is anchored to an independent backward recursion, and
the total liability must run off to zero by the end of the term.
"""
import numpy as np

from fastcashflow import ExpenseItem, ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


def _basis(**overrides):
    kw = dict(
        mortality_q       = 0.002,
        lapse_q           = 0.01,
        discount_annual   = 0.04,
        expense_inflation = 0.02,
        expense_items     = (
            ExpenseItem("acquisition", "per_policy",    100_000.0),
            ExpenseItem("maintenance", "per_policy",  24_000.0),
        ),
        ra_confidence     = 0.80,
        mortality_cv      = 0.10,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


def test_bel_rollforward():
    """The BEL trajectory matches an independent backward recursion."""
    basis = _basis()
    one = ModelPoints.single(
        issue_age=45, benefits={"DEATH": 80_000_000},
        premium=150_000, term_months=36,
        calculation_methods=PATTERNS,
    )
    res = measure(one, basis)

    i = basis.discount_monthly
    half = (1.0 + i) ** -0.5
    full = 1.0 / (1.0 + i)
    cf = res.cashflows
    n_time = cf.n_time

    bel = np.zeros(n_time + 1)
    for t in range(n_time - 1, -1, -1):
        bel[t] = (
            (cf.mortality_cf[0, t] + cf.expense_cf[0, t]) * half
            - cf.premium_cf[0, t]
            + bel[t + 1] * full
        )
    assert np.allclose(res.bel_path[0], bel)
    assert res.bel_path[0, -1] == 0.0

    # column 0 of the detailed trajectory equals the fast headline BEL
    assert np.isclose(res.bel_path[0, 0], measure(one, basis, full=False).bel[0])


def test_liability_runs_off():
    """BEL + RA + CSM fully runs off to zero by the end of the term."""
    basis = _basis()
    rng = np.random.default_rng(4)
    n = 150
    mps = ModelPoints(
        issue_age=rng.integers(30, 55, n),
        benefits={"DEATH": rng.integers(20, 100, n) * 1_000_000},
        premium=rng.integers(10, 25, n) * 10_000,
        term_months=rng.integers(48, 120, n),
        calculation_methods=PATTERNS,
    )
    res = measure(mps, basis)

    liability = res.bel_path + res.ra_path + res.csm_path
    assert np.allclose(liability[:, -1], 0.0)
