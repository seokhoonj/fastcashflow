"""Coverage architecture -- the variable-length benefit list.

A policy's claim benefits are a compressed-sparse-row coverage list, not
fixed fields, and the kernels loop it generically. Two properties must hold:
N coverages summing to an amount value identically to one coverage of that
amount, and an empty list equals a zero death benefit.
"""
import numpy as np

from fastcashflow import (
    Assumptions, BenefitPattern, CoverageRate, ExpenseItem, ModelPoints,
    measure, value,
)

Q = 0.002
LAPSE = 0.005
DEATH = 0   # the death coverage's index in _assumptions().coverages
PATTERNS = {"DEATH": BenefitPattern.DEATH}


def _annual(m):
    """Convert a monthly rate to its annual equivalent (engine converts back)."""
    return 1.0 - (1.0 - m) ** 12


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(Q)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(LAPSE)),
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    200_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  60_000.0),
        ),
        ra_confidence=0.75,
        mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(Q))),),
    )


def test_multiple_death_coverages_sum_to_one():
    """Two death coverages of A and B value as one coverage of A + B."""
    asmp = _assumptions()
    a, b, term = 6e7, 4e7, 36

    split = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([80_000.0]),
        term_months=np.array([term]),
        coverage_kind=np.array([DEATH, DEATH]),
        coverage_amount=np.array([a, b]),
        coverage_offset=np.array([0, 2]),
        benefit_patterns=PATTERNS,
    )
    combined = ModelPoints.single(
        40, 80_000.0, term, benefits={0: a + b}, benefit_patterns=PATTERNS,
    )

    m_split, m_comb = measure(split, asmp), measure(combined, asmp)
    assert np.allclose(m_split.bel, m_comb.bel)
    assert np.allclose(m_split.ra, m_comb.ra)
    assert np.allclose(m_split.csm, m_comb.csm)

    v_split, v_comb = value(split, asmp), value(combined, asmp)
    assert np.allclose(v_split.bel, v_comb.bel)
    assert np.allclose(v_split.ra, v_comb.ra)
    assert np.allclose(v_split.csm, v_comb.csm)


def test_no_coverages_matches_zero_death_benefit():
    """An empty coverage list equals a death benefit of zero."""
    asmp = _assumptions()
    explicit_zero = ModelPoints.single(
        45, 50_000.0, 60, benefits={0: 0.0}, benefit_patterns=PATTERNS,
    )
    no_coverages = ModelPoints(
        issue_age=np.array([45.0]),
        level_premium=np.array([50_000.0]),
        term_months=np.array([60]),
        benefit_patterns=PATTERNS,
    )
    a, b = value(explicit_zero, asmp), value(no_coverages, asmp)
    assert np.allclose(a.bel, b.bel)
    assert np.allclose(a.ra, b.ra)
    # no death coverage -> no claims -> zero Risk Adjustment
    assert np.isclose(b.ra[0], 0.0)
