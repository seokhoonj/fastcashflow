"""Phase 2 validation -- mid-month discounting and CSM movement detail.

The discounting precision is hand-checked: claims arise during the month
(discounted mid-month) and premiums at the start of the month. The CSM
roll-forward identity is checked exactly.
"""
import numpy as np

from fastcashflow import BenefitPattern, Assumptions, ModelPoints, measure, CoverageRate



PATTERNS = {"DEATH": BenefitPattern.DEATH}

def _annual(m):
    """Convert a monthly rate to its annual equivalent (engine converts back)."""
    return 1.0 - (1.0 - m) ** 12


def _flat_assumptions(**overrides) -> Assumptions:
    base = dict(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.01)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.0)),
        discount_annual=0.06,
        ra_confidence=0.75,
        mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.01))),),
    )
    base.update(overrides)
    return Assumptions(**base)


def test_mid_month_discounting():
    """Claims discounted mid-month, premiums start-of-month -- hand-checked."""
    death_benefit = 1_000_000.0
    premium = 12_000.0
    term = 2
    q = 0.01

    res = measure(
        ModelPoints.single(
            issue_age=40, benefits={0: death_benefit},
            level_premium=premium, term_months=term,
            benefit_patterns=PATTERNS,
        ),
        _flat_assumptions(),
    )

    i = (1.0 + 0.06) ** (1.0 / 12.0) - 1.0
    inforce = np.array([1.0, 1.0 - q])           # zero lapse
    deaths = inforce * q
    t = np.arange(term)
    d_start = (1.0 + i) ** (-t)
    d_mid = (1.0 + i) ** (-(t + 0.5))

    pv_claims = float(np.sum(deaths * death_benefit * d_mid))
    pv_premiums = float(np.sum(inforce * premium * d_start))
    assert np.isclose(res.bel[0, 0], pv_claims - pv_premiums)

    # the timing genuinely differs from the old all-start-of-month basis
    bel_all_start = float(np.sum(deaths * death_benefit * d_start)
                          - np.sum(inforce * premium * d_start))
    assert not np.isclose(res.bel[0, 0], bel_all_start)


def test_csm_movement_identity():
    """The CSM roll-forward decomposes exactly into accretion and release."""
    asmp = _flat_assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.001)),
        mortality_cv=0.05,
    )
    rng = np.random.default_rng(2)
    n = 200
    mps = ModelPoints(
        issue_age=rng.integers(25, 55, n),
        benefits={0: rng.integers(10, 100, n) * 1_000_000},
        level_premium=rng.integers(8, 20, n) * 10_000,
        term_months=rng.integers(60, 120, n),
        benefit_patterns=PATTERNS,
    )
    res = measure(mps, asmp)

    # csm[t+1] = csm[t] + accretion[t] - release[t], exactly
    opening = res.csm[:, :-1]
    closing = res.csm[:, 1:]
    assert np.array_equal(closing, opening + res.csm_accretion - res.csm_release)

    # accretion is interest on the opening balance
    assert np.array_equal(res.csm_accretion, opening * asmp.discount_monthly)
