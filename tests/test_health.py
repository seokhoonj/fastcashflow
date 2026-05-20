"""Health products -- inpatient, surgery and outpatient morbidity coverages.

Health claims are multiple-occurrence: a claim leaves the policy in force, so
unlike a death benefit a health coverage does not decrement. Its risk is
morbidity, priced by its own RA component (``morbidity_cv``).
"""
import numpy as np

from fastcashflow import (
    DIAGNOSIS,
    INPATIENT,
    OUTPATIENT,
    SURGERY,
    Assumptions,
    ModelPointSet,
    measure,
    value,
)
from fastcashflow.gmm import _norm_ppf

Q = 0.002            # flat monthly mortality
LAPSE = 0.005        # flat monthly lapse
MORB_RATE = 0.03     # flat monthly morbidity rate (events per in-force month)


def _assumptions(**overrides) -> Assumptions:
    flat_morb = lambda issue_age, duration: np.full(issue_age.shape, MORB_RATE)
    base = dict(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, Q),
        lapse_monthly=lambda duration: np.full(duration.shape, LAPSE),
        discount_annual=0.04,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.80,
        mortality_cv=0.10,
        morbidity_rates={INPATIENT: flat_morb, SURGERY: flat_morb,
                         OUTPATIENT: flat_morb, DIAGNOSIS: flat_morb},
    )
    base.update(overrides)
    return Assumptions(**base)


def test_inpatient_benefit_adds_its_present_value():
    """An inpatient coverage adds its present value to BEL; RA via morbidity_cv."""
    asmp = _assumptions(morbidity_cv=0.15)
    benefit, term = 30_000.0, 24
    res = measure(
        ModelPointSet.single(40, 0.0, 0.0, term, benefits={INPATIENT: benefit}),
        asmp,
    )

    i = asmp.discount_monthly
    surv = (1.0 - Q) * (1.0 - LAPSE)
    half = (1.0 + i) ** (-0.5)
    full = 1.0 / (1.0 + i)
    t = np.arange(term)
    # health claims are mid-month: PV = rate * benefit * sum(inforce * (1+i)^-(t+.5))
    pv = MORB_RATE * benefit * half * float(np.sum((surv * full) ** t))

    assert np.isclose(res.bel[0, 0], pv)
    z = _norm_ppf(asmp.ra_confidence)
    assert np.isclose(res.ra[0, 0], z * asmp.morbidity_cv * pv)


def test_health_claim_is_non_decrementing():
    """A health claim leaves the policy in force -- it does not decrement."""
    asmp = _assumptions()
    term = 36
    plain = measure(ModelPointSet.single(40, 1e8, 50_000.0, term), asmp)
    with_health = measure(
        ModelPointSet.single(
            40, 1e8, 50_000.0, term, benefits={INPATIENT: 30_000.0, SURGERY: 2e6}
        ),
        asmp,
    )

    # death claims and the in-force run-off are untouched by health coverages
    assert np.allclose(plain.cashflows.claim_cf, with_health.cashflows.claim_cf)
    assert np.allclose(plain.cashflows.inforce, with_health.cashflows.inforce)
    # health only adds its own outflow, which raises the BEL
    assert with_health.cashflows.morbidity_cf.sum() > 0.0
    assert with_health.bel[0, 0] > plain.bel[0, 0]


def test_morbidity_ra_responds_to_its_cv():
    """The morbidity RA is zero without morbidity_cv and linear in it."""
    health = ModelPointSet.single(40, 0.0, 0.0, 60, benefits={INPATIENT: 30_000.0})
    no_cv = measure(health, _assumptions(morbidity_cv=0.0))
    full_cv = measure(health, _assumptions(morbidity_cv=0.20))
    half_cv = measure(health, _assumptions(morbidity_cv=0.10))

    assert np.allclose(no_cv.ra, 0.0)
    assert full_cv.ra[0, 0] > 0.0
    assert np.isclose(half_cv.ra[0, 0], 0.5 * full_cv.ra[0, 0])


def test_value_matches_measure_health():
    """value() and measure() agree on contracts with health coverages."""
    rng = np.random.default_rng(11)
    n = 300
    mps = ModelPointSet(
        issue_age=rng.integers(30, 55, n),
        monthly_premium=rng.integers(5, 20, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        death_benefit=rng.integers(10, 80, n) * 1_000_000,
        benefits={
            INPATIENT: rng.integers(0, 5, n) * 10_000,
            SURGERY: rng.integers(0, 3, n) * 1_000_000,
            OUTPATIENT: rng.integers(0, 4, n) * 5_000,
        },
    )
    asmp = _assumptions(morbidity_cv=0.15)
    fast = value(mps, asmp)
    detailed = measure(mps, asmp)

    assert np.allclose(fast.bel, detailed.bel[:, 0])
    assert np.allclose(fast.ra, detailed.ra[:, 0])
    assert np.allclose(fast.csm, detailed.csm[:, 0])
    assert np.allclose(fast.loss_component, detailed.loss_component)


def test_diagnosis_benefit_hand_calc():
    """A diagnosis benefit -- hand-checked inception BEL and morbidity RA."""
    asmp = _assumptions(morbidity_cv=0.12)
    benefit, term = 5e7, 24
    res = measure(
        ModelPointSet.single(40, 0.0, 0.0, term, benefits={DIAGNOSIS: benefit}),
        asmp,
    )

    i = asmp.discount_monthly
    d = MORB_RATE
    half = (1.0 + i) ** (-0.5)
    full = 1.0 / (1.0 + i)
    g = (1.0 - Q) * (1.0 - LAPSE) * (1.0 - d)   # not-yet-diagnosed survival
    t = np.arange(term)
    # claim each month is the not-yet-diagnosed pool g^t times the rate
    pv = d * benefit * half * float(np.sum((g * full) ** t))

    assert np.isclose(res.bel[0, 0], pv)
    z = _norm_ppf(asmp.ra_confidence)
    assert np.isclose(res.ra[0, 0], z * asmp.morbidity_cv * pv)


def test_diagnosis_pool_depletes():
    """A diagnosis benefit pays once on a shrinking pool -- at the same rate
    it is worth less than a multiple-occurrence inpatient benefit."""
    asmp = _assumptions()
    term, amount = 120, 1e7
    diagnosis = measure(
        ModelPointSet.single(40, 0.0, 0.0, term, benefits={DIAGNOSIS: amount}),
        asmp,
    )
    inpatient = measure(
        ModelPointSet.single(40, 0.0, 0.0, term, benefits={INPATIENT: amount}),
        asmp,
    )
    # inpatient claims on the full in-force each month; diagnosis on the
    # depleting not-yet-diagnosed pool -- so diagnosis is worth strictly less
    assert 0.0 < diagnosis.bel[0, 0] < inpatient.bel[0, 0]


def test_value_matches_measure_diagnosis():
    """value() and measure() agree on contracts with diagnosis coverages."""
    rng = np.random.default_rng(19)
    n = 250
    mps = ModelPointSet(
        issue_age=rng.integers(30, 55, n),
        monthly_premium=rng.integers(5, 20, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        death_benefit=rng.integers(10, 80, n) * 1_000_000,
        benefits={
            DIAGNOSIS: rng.integers(0, 6, n) * 10_000_000,
            INPATIENT: rng.integers(0, 4, n) * 10_000,
        },
    )
    asmp = _assumptions(morbidity_cv=0.15)
    fast = value(mps, asmp)
    detailed = measure(mps, asmp)

    assert np.allclose(fast.bel, detailed.bel[:, 0])
    assert np.allclose(fast.ra, detailed.ra[:, 0])
    assert np.allclose(fast.csm, detailed.csm[:, 0])
    assert np.allclose(fast.loss_component, detailed.loss_component)
