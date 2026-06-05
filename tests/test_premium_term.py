"""Premium-paying term -- a level premium collected over fewer months than
the coverage term (a "20-year pay, 30-year coverage" contract).

``premium_term_months`` gates the level premium: it is charged while
``t < premium_term_months`` and stops thereafter, the coverage continuing.
It defaults to the full coverage term -- premium every in-force month, the
ordinary case. The hand case is a 3-month contract paying premium for 2.
"""
import numpy as np

from fastcashflow import ModelPoints, read_model_points
from fastcashflow.gmm import measure
from conftest import (PATTERNS, annual_from_monthly as _annual,
                      make_death_basis, mp_to_frames)


def _assumptions(**overrides):
    """Flat-rate, zero-discount, zero-expense basis -- every figure by hand."""
    kw = dict(
        mortality_q     = 0.01,
        lapse_q         = 0.02,
        discount_annual = 0.0,
        ra_confidence   = 0.75,
        mortality_cv    = 0.10,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


def test_premium_term_hand_calculation():
    """A 3-month contract paying premium for 2 months: the third month
    projects coverage with no premium."""
    death_benefit = 1_000_000.0
    premium = 12_000.0
    mp = ModelPoints.single(
        issue_age=40, benefits={0: death_benefit}, premium=premium,
        term_months=3, premium_term_months=2,
        calculation_methods=PATTERNS,
    )
    basis = _assumptions()
    res = measure(mp, basis)

    # in force [1.0, 0.99*0.98, (0.99*0.98)**2]; zero discount.
    s = 0.99 * 0.98
    inforce = [1.0, s, s * s]

    # premium is charged in months 0 and 1, then stops.
    premium_cf = [inforce[0] * premium, inforce[1] * premium, 0.0]
    assert np.allclose(res.cashflows.premium_cf[0], premium_cf)
    assert res.cashflows.premium_cf[0, 2] == 0.0

    # claims still run every month of the coverage term.
    pv_claims = sum(i * 0.01 * death_benefit for i in inforce)
    pv_premiums = sum(premium_cf)

    bel = pv_claims - pv_premiums
    assert np.isclose(res.bel_path[0, 0], bel)
    # measure() reproduces the same headline number.
    assert np.isclose(measure(mp, basis, full=False).bel[0], bel)


def test_premium_term_defaults_to_full_term():
    """With no `premium_term_months`, premium is collected the whole term --
    the same result as setting it equal to `term_months`."""
    kw = dict(issue_age=40, benefits={0: 1_000_000.0},
              premium=12_000.0, term_months=120)
    basis = _assumptions()

    default = ModelPoints.single(**kw, calculation_methods=PATTERNS)
    assert np.all(default.premium_term_months == 120)

    explicit = ModelPoints.single(**kw, premium_term_months=120, calculation_methods=PATTERNS)
    assert np.isclose(measure(default, basis, full=False).bel[0], measure(explicit, basis, full=False).bel[0])


def test_shorter_premium_term_raises_the_liability():
    """Collecting premium for fewer months drops a premium inflow, so the
    liability is larger than the same contract paid for the full term."""
    kw = dict(issue_age=45, benefits={0: 50_000_000.0},
              premium=30_000.0, term_months=240)
    basis = _assumptions()

    full_pay = measure(ModelPoints.single(**kw, premium_term_months=240, calculation_methods=PATTERNS), basis, full=False)
    short_pay = measure(ModelPoints.single(**kw, premium_term_months=120, calculation_methods=PATTERNS), basis, full=False)

    assert short_pay.bel[0] > full_pay.bel[0]


def test_premium_term_round_trips(tmp_path):
    """The `premium_term_months` column reads back unchanged."""
    basis = _assumptions()
    mp = ModelPoints(
        issue_age=np.array([40, 40]),
        premium=np.array([12_000.0, 12_000.0]),
        term_months=np.array([120, 120]),
        benefits={0: np.array([1_000_000.0, 1_000_000.0])},
        premium_term_months=np.array([120, 60]),
        calculation_methods=PATTERNS,
    )
    pol, cov = mp_to_frames(mp, basis)
    pol.write_csv(tmp_path / "policies.csv")
    cov.write_csv(tmp_path / "coverages.csv")

    back = read_model_points(tmp_path / "policies.csv",
                             coverages=tmp_path / "coverages.csv",
                             calculation_methods=PATTERNS)
    assert list(back.premium_term_months) == [120, 60]
    # the 60-month-pay policy collects less premium -> larger liability.
    val = measure(back, basis, full=False)
    assert val.bel[1] > val.bel[0]
