"""Product validation -- the maturity benefit (endowment / pure endowment).

A maturity benefit is paid to the policies still in force at the end of the
term. It must raise the BEL by exactly its present value; a contract with no
death benefit carries no death claims and so no Risk Adjustment.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPointSet, measure, value

Q = 0.002          # flat monthly mortality
LAPSE = 0.005      # flat monthly lapse


def _assumptions(**overrides) -> Assumptions:
    base = dict(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, Q),
        lapse_monthly=lambda duration: np.full(duration.shape, LAPSE),
        discount_annual=0.04,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.80,
        claims_cv=0.10,
    )
    base.update(overrides)
    return Assumptions(**base)


def test_maturity_benefit_adds_its_present_value():
    """Adding a maturity benefit raises BEL by exactly its present value."""
    asmp = _assumptions()
    death_benefit, maturity, premium, term = 1e8, 5e7, 50_000.0, 24

    term_life = measure(
        ModelPointSet.single(40, death_benefit, premium, term), asmp
    )
    endowment = measure(
        ModelPointSet.single(
            40, death_benefit, premium, term, maturity_benefit=maturity
        ),
        asmp,
    )

    i = asmp.discount_monthly
    survivors = ((1.0 - Q) * (1.0 - LAPSE)) ** term
    pv_maturity = survivors * maturity * (1.0 + i) ** (-term)
    assert np.isclose(endowment.bel[0, 0] - term_life.bel[0, 0], pv_maturity)


def test_pure_endowment():
    """A pure endowment (no death benefit) carries zero RA -- hand-checked BEL."""
    asmp = _assumptions()
    maturity, premium, term = 5e7, 50_000.0, 24
    res = measure(
        ModelPointSet.single(40, 0.0, premium, term, maturity_benefit=maturity),
        asmp,
    )

    # no death benefit -> no death claims -> zero Risk Adjustment
    assert np.allclose(res.ra, 0.0)

    # inception BEL = PV(maturity benefit) - PV(premiums), zero expenses
    i = asmp.discount_monthly
    surv = (1.0 - Q) * (1.0 - LAPSE)
    t = np.arange(term)
    pv_premiums = float(np.sum(surv ** t * premium * (1.0 + i) ** (-t)))
    pv_maturity = surv ** term * maturity * (1.0 + i) ** (-term)
    assert np.isclose(res.bel[0, 0], pv_maturity - pv_premiums)


def test_value_matches_measure_endowment():
    """value() and measure() agree on endowment contracts."""
    rng = np.random.default_rng(12)
    n = 400
    mps = ModelPointSet(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(10, 80, n) * 1_000_000,
        monthly_premium=rng.integers(5, 20, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        maturity_benefit=rng.integers(5, 40, n) * 1_000_000,
    )
    asmp = _assumptions()
    fast = value(mps, asmp)
    detailed = measure(mps, asmp)

    assert np.allclose(fast.bel, detailed.bel[:, 0])
    assert np.allclose(fast.ra, detailed.ra[:, 0])
    assert np.allclose(fast.csm, detailed.csm[:, 0])
    assert np.allclose(fast.loss_component, detailed.loss_component)
