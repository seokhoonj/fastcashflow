"""Premium-paying term -- a level premium collected over fewer months than
the coverage term (a "20-year pay, 30-year coverage" contract).

``premium_term_months`` gates the level premium: it is charged while
``t < premium_term_months`` and stops thereafter, the coverage continuing.
It defaults to the full coverage term -- premium every in-force month, the
ordinary case. The hand case is a 3-month contract paying premium for 2.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPoints, measure, read_model_points, value


def _annual(m):
    """Convert a monthly rate to the equivalent annual rate the engine expects."""
    return 1.0 - (1.0 - m) ** 12


def _assumptions(**overrides) -> Assumptions:
    """Flat-rate, zero-discount, zero-expense basis -- every figure by hand."""
    base = dict(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.01)),
        lapse_annual=lambda duration: np.full(duration.shape, _annual(0.02)),
        discount_annual=0.0,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )
    base.update(overrides)
    return Assumptions(**base)


def test_premium_term_hand_calculation():
    """A 3-month contract paying premium for 2 months: the third month
    projects coverage with no premium."""
    death_benefit = 1_000_000.0
    premium = 12_000.0
    mp = ModelPoints.single(
        issue_age=40, death_benefit=death_benefit, level_premium=premium,
        term_months=3, premium_term_months=2,
    )
    asmp = _assumptions()
    res = measure(mp, asmp)

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
    assert np.isclose(res.bel[0, 0], bel)
    # value() reproduces the same headline number.
    assert np.isclose(value(mp, asmp).bel[0], bel)


def test_premium_term_defaults_to_full_term():
    """With no `premium_term_months`, premium is collected the whole term --
    the same result as setting it equal to `term_months`."""
    kw = dict(issue_age=40, death_benefit=1_000_000.0,
              level_premium=12_000.0, term_months=120)
    asmp = _assumptions()

    default = ModelPoints.single(**kw)
    assert np.all(default.premium_term_months == 120)

    explicit = ModelPoints.single(**kw, premium_term_months=120)
    assert np.isclose(value(default, asmp).bel[0], value(explicit, asmp).bel[0])


def test_shorter_premium_term_raises_the_liability():
    """Collecting premium for fewer months drops a premium inflow, so the
    liability is larger than the same contract paid for the full term."""
    kw = dict(issue_age=45, death_benefit=50_000_000.0,
              level_premium=30_000.0, term_months=240)
    asmp = _assumptions()

    full_pay = value(ModelPoints.single(**kw, premium_term_months=240), asmp)
    short_pay = value(ModelPoints.single(**kw, premium_term_months=120), asmp)

    assert short_pay.bel[0] > full_pay.bel[0]


def test_premium_term_round_trips(tmp_path):
    """A wide file's `premium_term_months` column reads back unchanged."""
    asmp = _assumptions()
    mp = ModelPoints(
        issue_age=np.array([40, 40]),
        level_premium=np.array([12_000.0, 12_000.0]),
        term_months=np.array([120, 120]),
        death_benefit=np.array([1_000_000.0, 1_000_000.0]),
        premium_term_months=np.array([120, 60]),
    )
    path = tmp_path / "model_points.csv"
    mp.to_wide(asmp).write_csv(path)

    back = read_model_points(path, asmp)
    assert list(back.premium_term_months) == [120, 60]
    # the 60-month-pay policy collects less premium -> larger liability.
    val = value(back, asmp)
    assert val.bel[1] > val.bel[0]
