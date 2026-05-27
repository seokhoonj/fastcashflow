"""Per-coverage benefit rules -- waiting and reduced-benefit periods.

A coverage may pay nothing for an initial waiting period (Korean: myeonchaek)
and a reduced multiple of the benefit until a later cut-off (gamaek). The rule
scales the benefit paid; it never touches the decrement -- a policyholder
still dies, lapses or is diagnosed at the assumed rate, the contract just
pays less. For a single-payment diagnosis benefit that means the waiting
period suppresses the payment while the not-yet-diagnosed pool depletes all
the same.
"""
import numpy as np

from fastcashflow import (
    Assumptions,
    BenefitPattern,
    ModelPoints,
    CoverageRate,
    measure,
    value,
)
from fastcashflow.numerics import _norm_ppf
from conftest import annual_from_monthly as _annual

Q = 0.002            # flat monthly mortality
LAPSE = 0.005        # flat monthly lapse
MORB_RATE = 0.03     # flat monthly diagnosis rate
# Local coverage codes -- the order of CoverageRate entries in _assumptions().
DEATH = 0            # the death coverage -> coverages[0]
DIAGNOSIS = 1        # the diagnosis rider -> coverages[1]

PATTERNS = {
    "death":     BenefitPattern.DEATH,
    "diagnosis": BenefitPattern.DIAGNOSIS,
}


def _mortality(sex, issue_age, duration):
    return np.full(issue_age.shape, _annual(Q))


def _assumptions(**overrides) -> Assumptions:
    flat_morb = lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(MORB_RATE))
    base = dict(
        mortality_annual=_mortality,
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(LAPSE)),
        discount_annual=0.04,
        ra_confidence=0.80,
        mortality_cv=0.10,
        coverages=(
            CoverageRate("death", _mortality),
            CoverageRate("diagnosis", flat_morb),
        ),
    )
    base.update(overrides)
    return Assumptions(**base)


def _one_coverage(kind, benefit, term, *, waiting=0,
                  reduction_end=0, reduction_factor=1.0) -> ModelPoints:
    """A single-policy, single-coverage model point carrying a benefit rule."""
    return ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([0.0]),
        term_months=np.array([term]),
        coverage_kind=np.array([kind]),
        coverage_amount=np.array([float(benefit)]),
        coverage_offset=np.array([0, 1]),
        coverage_waiting=np.array([waiting]),
        coverage_reduction_end=np.array([reduction_end]),
        coverage_reduction_factor=np.array([reduction_factor]),
        benefit_patterns=PATTERNS,
    )


def test_waiting_period_hand_calc():
    """A diagnosis benefit with a waiting period -- hand-checked BEL and RA.

    The waiting months pay nothing; the not-yet-diagnosed pool depletes
    through them all the same, so months from ``wait`` on are unchanged.
    """
    asmp = _assumptions(morbidity_cv=0.12)
    benefit, term, wait = 5e7, 24, 3
    res = measure(_one_coverage(DIAGNOSIS, benefit, term, waiting=wait), asmp)

    i = asmp.discount_monthly
    d = MORB_RATE
    half = (1.0 + i) ** (-0.5)
    full = 1.0 / (1.0 + i)
    g = (1.0 - Q) * (1.0 - LAPSE) * (1.0 - d)   # not-yet-diagnosed survival
    t = np.arange(wait, term)                   # months t < wait pay nothing
    pv = d * benefit * half * float(np.sum((g * full) ** t))

    assert np.isclose(res.bel[0, 0], pv)
    z = _norm_ppf(asmp.ra_confidence)
    assert np.isclose(res.ra[0, 0], z * asmp.morbidity_cv * pv)


def test_waiting_suppresses_payment_not_the_pool():
    """Waiting zeroes the waiting-month claims and leaves the rest exactly as
    the no-waiting projection -- the not-yet-diagnosed pool is unchanged."""
    asmp = _assumptions()
    benefit, term, wait = 3e7, 36, 6
    plain = measure(_one_coverage(DIAGNOSIS, benefit, term), asmp)
    waited = measure(_one_coverage(DIAGNOSIS, benefit, term, waiting=wait), asmp)

    mcf_plain = plain.cashflows.morbidity_cf[0]
    mcf_waited = waited.cashflows.morbidity_cf[0]
    assert np.allclose(mcf_waited[:wait], 0.0)
    assert np.allclose(mcf_waited[wait:], mcf_plain[wait:])
    # the suppressed months make the contract cheaper
    assert waited.bel[0, 0] < plain.bel[0, 0]


def test_reduction_period_hand_calc():
    """A diagnosis benefit reduced to a fraction until a cut-off month --
    hand-checked BEL and RA."""
    asmp = _assumptions(morbidity_cv=0.12)
    benefit, term, red_end, rf = 5e7, 24, 12, 0.5
    res = measure(
        _one_coverage(DIAGNOSIS, benefit, term,
                      reduction_end=red_end, reduction_factor=rf),
        asmp,
    )

    i = asmp.discount_monthly
    d = MORB_RATE
    half = (1.0 + i) ** (-0.5)
    gf = (1.0 - Q) * (1.0 - LAPSE) * (1.0 - d) / (1.0 + i)
    t_red = np.arange(0, red_end)               # reduced months
    t_full = np.arange(red_end, term)           # full-benefit months
    pv = d * benefit * half * (
        rf * float(np.sum(gf ** t_red)) + float(np.sum(gf ** t_full))
    )

    assert np.isclose(res.bel[0, 0], pv)
    z = _norm_ppf(asmp.ra_confidence)
    assert np.isclose(res.ra[0, 0], z * asmp.morbidity_cv * pv)


def test_reduction_on_death_benefit():
    """A reduced-benefit period on a death coverage scales the death claim,
    not the mortality decrement."""
    asmp = _assumptions()
    benefit, term, red_end, rf = 1e8, 48, 24, 0.5
    plain = measure(_one_coverage(DEATH, benefit, term), asmp)
    reduced = measure(
        _one_coverage(DEATH, benefit, term,
                      reduction_end=red_end, reduction_factor=rf),
        asmp,
    )

    ccf_plain = plain.cashflows.claim_cf[0]
    ccf_reduced = reduced.cashflows.claim_cf[0]
    # claims are scaled by rf during the reduced period, full afterwards
    assert np.allclose(ccf_reduced[:red_end], rf * ccf_plain[:red_end])
    assert np.allclose(ccf_reduced[red_end:], ccf_plain[red_end:])
    # the decrement -- in-force and deaths -- is untouched by the rule
    assert np.allclose(reduced.cashflows.inforce, plain.cashflows.inforce)
    assert np.allclose(reduced.cashflows.deaths, plain.cashflows.deaths)


def test_default_rule_is_inert():
    """Explicit off-rule fields equal omitting them entirely."""
    asmp = _assumptions(morbidity_cv=0.10)
    explicit = _one_coverage(DIAGNOSIS, 4e7, 36,
                             waiting=0, reduction_end=0, reduction_factor=1.0)
    omitted = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([0.0]),
        term_months=np.array([36]),
        coverage_kind=np.array([DIAGNOSIS]),
        coverage_amount=np.array([4e7]),
        coverage_offset=np.array([0, 1]),
        benefit_patterns=PATTERNS,
    )
    a, b = value(explicit, asmp), value(omitted, asmp)
    assert np.isclose(a.bel[0], b.bel[0])
    assert np.isclose(a.ra[0], b.ra[0])


def test_value_matches_measure_with_rules():
    """value() and measure() agree on a portfolio carrying waiting and
    reduced-benefit rules across death and diagnosis coverages."""
    rng = np.random.default_rng(23)
    n = 200
    n_cov = 2 * n                               # one death + one diagnosis each
    coverage_kind = np.empty(n_cov, np.int64)
    coverage_kind[0::2] = DEATH
    coverage_kind[1::2] = DIAGNOSIS
    coverage_amount = np.empty(n_cov)
    coverage_amount[0::2] = rng.integers(10, 80, n) * 1_000_000
    coverage_amount[1::2] = rng.integers(10, 50, n) * 1_000_000

    mps = ModelPoints(
        issue_age=rng.integers(30, 55, n).astype(float),
        level_premium=rng.integers(5, 20, n) * 10_000.0,
        term_months=rng.integers(60, 180, n),
        coverage_kind=coverage_kind,
        coverage_amount=coverage_amount,
        coverage_offset=np.arange(0, n_cov + 1, 2),
        coverage_waiting=rng.integers(0, 8, n_cov),
        coverage_reduction_end=rng.integers(0, 30, n_cov),
        coverage_reduction_factor=rng.choice([0.3, 0.5, 0.7], n_cov),
        benefit_patterns=PATTERNS,
    )
    asmp = _assumptions(morbidity_cv=0.15)
    fast = value(mps, asmp)
    detailed = measure(mps, asmp)

    assert np.allclose(fast.bel, detailed.bel[:, 0])
    assert np.allclose(fast.ra, detailed.ra[:, 0])
    assert np.allclose(fast.csm, detailed.csm[:, 0])
    assert np.allclose(fast.loss_component, detailed.loss_component)
