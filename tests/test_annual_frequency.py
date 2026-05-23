"""Annual rate input and payment / annuity frequency.

Two mechanics introduced together:

* Rates are supplied annual; the engine converts each to a monthly rate on
  the constant-force basis -- twelve monthly applications reproduce the
  annual rate exactly. ``assumptions.annual_to_monthly`` does the conversion.
* A level premium is collected every ``premium_frequency_months`` and a
  survival annuity paid every ``annuity_frequency_months``.

Every figure is derived by hand on a flat, zero-discount basis.
"""
import numpy as np
import pytest

from fastcashflow import Assumptions, ModelPoints, measure, value
from fastcashflow.assumptions import annual_to_monthly


def _asmp(*, q_annual=0.0, lapse_annual=0.0, **overrides) -> Assumptions:
    """Flat-rate, zero-discount, zero-expense basis."""
    base = dict(
        mortality_annual=lambda s, a, d: np.full(a.shape, q_annual),
        lapse_annual=lambda sex, issue_age, d: np.full(d.shape, lapse_annual),
        discount_annual=0.0,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )
    base.update(overrides)
    return Assumptions(**base)


# ---------------------------------------------------------------------------
# Annual -> monthly rate conversion
# ---------------------------------------------------------------------------

def test_annual_to_monthly_round_trips():
    """The constant-force conversion: twelve monthly applications of the
    converted rate reproduce the annual rate exactly, and the monthly rate
    sits below the annual one."""
    for q_a in (0.001, 0.05, 0.12, 0.30):
        q_m = annual_to_monthly(q_a)
        assert np.isclose(1.0 - (1.0 - q_m) ** 12, q_a)
        assert q_m < q_a


def test_annual_mortality_reproduced_over_a_year():
    """A flat annual mortality drives the monthly decrement so the in-force
    after twelve months is exactly 1 - q_annual -- the constant-force basis
    preserves the annual rate."""
    q_annual = 0.12
    mp = ModelPoints.single(issue_age=40, death_benefit=1_000_000.0,
                            level_premium=0.0, term_months=13)
    res = measure(mp, _asmp(q_annual=q_annual))
    assert np.isclose(res.cashflows.inforce[0][12], 1.0 - q_annual)


# ---------------------------------------------------------------------------
# Premium payment frequency
# ---------------------------------------------------------------------------

def test_premium_frequency_payment_months():
    """A quarterly premium is collected at months 0, 3, 6, 9; the single
    premium is added at month 0 regardless of the frequency."""
    mp = ModelPoints.single(issue_age=40, death_benefit=1_000_000.0,
                            level_premium=10_000.0, term_months=12,
                            single_premium=5_000.0,
                            premium_frequency_months=3)
    res = measure(mp, _asmp())            # no decrements -- in-force stays 1
    pcf = res.cashflows.premium_cf[0]
    assert np.isclose(pcf[0], 10_000.0 + 5_000.0)
    for t in (3, 6, 9):
        assert np.isclose(pcf[t], 10_000.0)
    for t in (1, 2, 4, 5, 7, 8, 10, 11):
        assert pcf[t] == 0.0


def test_quarterly_premium_hand_calculation():
    """A 6-month contract with a quarterly premium and a flat 1% monthly
    mortality -- BEL derived by hand from the two payment months."""
    q_m = 0.01
    death_benefit, premium = 1_000_000.0, 12_000.0
    mp = ModelPoints.single(issue_age=40, death_benefit=death_benefit,
                            level_premium=premium, term_months=6,
                            premium_frequency_months=3)
    asmp = _asmp(q_annual=1.0 - (1.0 - q_m) ** 12)   # monthly q_m at the engine

    inforce = [(1.0 - q_m) ** t for t in range(6)]
    pv_claims = sum(i * q_m * death_benefit for i in inforce)
    pv_prem = premium * (inforce[0] + inforce[3])    # premium at months 0, 3
    bel = pv_claims - pv_prem

    assert np.isclose(value(mp, asmp).bel[0], bel)
    assert np.isclose(measure(mp, asmp).bel[0, 0], bel)


def test_premium_frequency_respects_premium_term():
    """Frequency and premium term compose: a quarterly premium on an
    8-month premium term pays at months 0, 3, 6 only."""
    mp = ModelPoints.single(issue_age=40, death_benefit=1_000_000.0,
                            level_premium=10_000.0, term_months=24,
                            premium_term_months=8,
                            premium_frequency_months=3)
    pcf = measure(mp, _asmp()).cashflows.premium_cf[0]
    assert list(np.flatnonzero(pcf)) == [0, 3, 6]


# ---------------------------------------------------------------------------
# Annuity payout frequency
# ---------------------------------------------------------------------------

def test_annuity_frequency_payout_months():
    """An annual annuity on a 24-month contract is paid at months 0 and 12;
    with zero discount the BEL is the two payments."""
    annuity = 2_000_000.0
    mp = ModelPoints.single(issue_age=60, death_benefit=0.0,
                            level_premium=0.0, term_months=24,
                            annuity_payment=annuity,
                            annuity_frequency_months=12)
    res = measure(mp, _asmp())            # no decrements -- in-force stays 1
    acf = res.cashflows.annuity_cf[0]
    assert np.isclose(acf[0], annuity)
    assert np.isclose(acf[12], annuity)
    assert np.count_nonzero(acf) == 2
    assert np.isclose(value(mp, _asmp()).bel[0], 2 * annuity)


# ---------------------------------------------------------------------------
# Cross-checks and validation
# ---------------------------------------------------------------------------

def test_measure_value_agree_under_frequency():
    """The fused and detailed paths agree across a portfolio with mixed
    premium and annuity frequencies."""
    rng = np.random.default_rng(3)
    n = 40
    freqs = np.array([1, 3, 6, 12])
    mps = ModelPoints(
        issue_age=rng.integers(35, 55, n).astype(float),
        death_benefit=rng.integers(10, 60, n) * 1_000_000.0,
        level_premium=rng.integers(2, 8, n) * 10_000.0,
        term_months=np.full(n, 120),
        annuity_payment=rng.integers(0, 3, n) * 1_000_000.0,
        premium_frequency_months=freqs[rng.integers(0, 4, n)],
        annuity_frequency_months=freqs[rng.integers(0, 4, n)],
    )
    asmp = _asmp(q_annual=0.08, lapse_annual=0.05)
    m, v = measure(mps, asmp), value(mps, asmp)
    assert np.allclose(m.bel[:, 0], v.bel)
    assert np.allclose(m.ra[:, 0], v.ra)


def test_default_frequency_is_monthly():
    """An unset frequency means monthly -- the ordinary every-month premium
    and annuity, identical to passing 1 explicitly."""
    kw = dict(issue_age=45, death_benefit=20_000_000.0, level_premium=30_000.0,
              term_months=60, annuity_payment=100_000.0)
    asmp = _asmp(q_annual=0.05)
    default = value(ModelPoints.single(**kw), asmp)
    explicit = value(ModelPoints.single(**kw, premium_frequency_months=1,
                                        annuity_frequency_months=1), asmp)
    assert np.isclose(default.bel[0], explicit.bel[0])


def test_frequency_must_be_positive():
    """A frequency below one month is rejected at build time."""
    with pytest.raises(ValueError, match="premium_frequency_months"):
        ModelPoints.single(issue_age=40, death_benefit=1_000_000.0,
                           level_premium=1_000.0, term_months=12,
                           premium_frequency_months=0)
    with pytest.raises(ValueError, match="annuity_frequency_months"):
        ModelPoints.single(issue_age=40, death_benefit=1_000_000.0,
                           level_premium=1_000.0, term_months=12,
                           annuity_frequency_months=0)
