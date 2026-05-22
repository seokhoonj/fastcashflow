"""Product validation -- survival benefits (endowment and annuity).

The maturity benefit (endowment) and the annuity payment (immediate annuity)
are both paid on survival. The maturity benefit must raise the BEL by exactly
its present value; survival benefits carry longevity risk, priced through the
``longevity_cv`` component of the Risk Adjustment.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPoints, measure, value
from fastcashflow.gmm import _norm_ppf

Q = 0.002          # flat monthly mortality
LAPSE = 0.005      # flat monthly lapse


def _annual(m):
    """Convert a flat monthly rate to its annual equivalent."""
    return 1.0 - (1.0 - m) ** 12


def _assumptions(**overrides) -> Assumptions:
    base = dict(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(Q)),
        lapse_annual=lambda duration: np.full(duration.shape, _annual(LAPSE)),
        discount_annual=0.04,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.80,
        mortality_cv=0.10,
    )
    base.update(overrides)
    return Assumptions(**base)


def test_maturity_benefit_adds_its_present_value():
    """Adding a maturity benefit raises BEL by exactly its present value."""
    asmp = _assumptions()
    death_benefit, maturity, premium, term = 1e8, 5e7, 50_000.0, 24

    term_life = measure(
        ModelPoints.single(40, death_benefit, premium, term), asmp
    )
    endowment = measure(
        ModelPoints.single(
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
        ModelPoints.single(40, 0.0, premium, term, maturity_benefit=maturity),
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
    mps = ModelPoints(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(10, 80, n) * 1_000_000,
        level_premium=rng.integers(5, 20, n) * 10_000,
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


def test_immediate_annuity_hand_calc():
    """A pure immediate annuity -- hand-checked inception BEL and RA."""
    asmp = _assumptions(longevity_cv=0.08)
    single, annuity, term = 1.2e8, 600_000.0, 24
    res = measure(
        ModelPoints.single(
            40, 0.0, 0.0, term, annuity_payment=annuity, single_premium=single
        ),
        asmp,
    )

    i = asmp.discount_monthly
    surv = (1.0 - Q) * (1.0 - LAPSE)
    full = 1.0 / (1.0 + i)
    t = np.arange(term)
    pv_annuity = float(np.sum((surv * full) ** t)) * annuity

    # BEL = PV(annuity outgo) - the single premium (paid at t=0, discount 1)
    assert np.isclose(res.bel[0, 0], pv_annuity - single)
    # longevity RA = z(confidence) * longevity_cv * PV(survival benefits)
    z = _norm_ppf(asmp.ra_confidence)
    assert np.isclose(res.ra[0, 0], z * asmp.longevity_cv * pv_annuity)


def test_value_matches_measure_annuity():
    """value() and measure() agree on immediate-annuity contracts."""
    rng = np.random.default_rng(7)
    n = 300
    mps = ModelPoints(
        issue_age=rng.integers(55, 75, n),
        death_benefit=np.zeros(n),
        level_premium=np.zeros(n),
        term_months=rng.integers(120, 300, n),
        annuity_payment=rng.integers(30, 100, n) * 10_000,
        single_premium=rng.integers(80, 200, n) * 1_000_000,
    )
    asmp = _assumptions(longevity_cv=0.08)
    fast = value(mps, asmp)
    detailed = measure(mps, asmp)

    assert np.allclose(fast.bel, detailed.bel[:, 0])
    assert np.allclose(fast.ra, detailed.ra[:, 0])
    assert np.allclose(fast.csm, detailed.csm[:, 0])
    assert np.allclose(fast.loss_component, detailed.loss_component)


def test_longevity_ra_responds_to_its_cv():
    """The longevity RA is zero without longevity_cv and linear in it."""
    annuity = ModelPoints.single(
        60, 0.0, 0.0, 180, annuity_payment=500_000.0, single_premium=8e7
    )
    no_cv = measure(annuity, _assumptions(longevity_cv=0.0))
    full_cv = measure(annuity, _assumptions(longevity_cv=0.10))
    half_cv = measure(annuity, _assumptions(longevity_cv=0.05))

    assert np.allclose(no_cv.ra, 0.0)            # no longevity_cv -> no RA
    assert full_cv.ra[0, 0] > 0.0                # longevity risk is now priced
    assert np.isclose(half_cv.ra[0, 0], 0.5 * full_cv.ra[0, 0])
