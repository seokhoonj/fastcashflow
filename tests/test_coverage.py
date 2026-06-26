"""Coverage architecture -- the variable-length benefit list.

A policy's claim benefits are a compressed-sparse-row coverage list, not
fixed fields, and the kernels loop it generically. Two properties must hold:
N coverages summing to an amount value identically to one coverage of that
amount, and an empty list equals a zero death benefit.
"""
import numpy as np

from fastcashflow import ExpenseItem, ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


Q = 0.002
LAPSE = 0.005
DEATH = 0   # the death coverage's index in _basis().coverages


def _basis():
    return make_death_basis(
        mortality_q       = Q,
        lapse_q           = LAPSE,
        discount_annual   = 0.03,
        expense_inflation = 0.02,
        expense_items     = (
            ExpenseItem("acquisition", "per_policy",    200_000.0),
            ExpenseItem("maintenance", "per_policy",  60_000.0),
        ),
        ra_confidence     = 0.75,
        mortality_cv      = 0.10,
    )


def test_multiple_death_coverages_sum_to_one():
    """Two death coverages of A and B value as one coverage of A + B."""
    basis = _basis()
    a, b, term = 6e7, 4e7, 36

    split = ModelPoints(
        issue_age=np.array([40.0]),
        premium=np.array([80_000.0]),
        term_months=np.array([term]),
        coverage_index=np.array([DEATH, DEATH]),
        coverage_amount=np.array([a, b]),
        coverage_offset=np.array([0, 2]),
        calculation_methods=PATTERNS,
    )
    combined = ModelPoints.single(
        40, 80_000.0, term, benefits={"DEATH": a + b}, calculation_methods=PATTERNS,
    )

    m_split, m_comb = measure(split, basis), measure(combined, basis)
    assert np.allclose(m_split.bel, m_comb.bel)
    assert np.allclose(m_split.ra, m_comb.ra)
    assert np.allclose(m_split.csm, m_comb.csm)

    v_split, v_comb = measure(split, basis, full=False), measure(combined, basis, full=False)
    assert np.allclose(v_split.bel, v_comb.bel)
    assert np.allclose(v_split.ra, v_comb.ra)
    assert np.allclose(v_split.csm, v_comb.csm)


def test_no_coverages_matches_zero_death_benefit():
    """An empty coverage list equals a death benefit of zero."""
    basis = _basis()
    explicit_zero = ModelPoints.single(
        45, 50_000.0, 60, benefits={"DEATH": 0.0}, calculation_methods=PATTERNS,
    )
    no_coverages = ModelPoints(
        issue_age=np.array([45.0]),
        premium=np.array([50_000.0]),
        term_months=np.array([60]),
        calculation_methods=PATTERNS,
    )
    a, b = measure(explicit_zero, basis, full=False), measure(no_coverages, basis, full=False)
    assert np.allclose(a.bel, b.bel)
    assert np.allclose(a.ra, b.ra)
    # no death coverage -> no claims -> zero Risk Adjustment
    assert np.isclose(b.ra[0], 0.0)


def test_basis_rejects_duplicate_coverage_code():
    """Coverage code is the key the engine resolves a model point's coverage
    against; a duplicate would silently keep only the last rate, so Basis
    rejects it at construction (covers the file path via read_basis too)."""
    import pytest
    from fastcashflow import Basis, CoverageRate
    r = lambda s, a, d: np.full(np.shape(a), 0.01)
    with pytest.raises(ValueError, match="duplicate coverage code"):
        Basis(mortality_annual=r, lapse_annual=r, discount_annual=0.0,
              ra_confidence=0.75, mortality_cv=0.10,
              coverages=(CoverageRate("CANCER", r), CoverageRate("CANCER", r)))


# ---------------------------------------------------------------------------
# coverage_term -- per-coverage maturity (a coverage ends before the contract
# boundary, e.g. a whole-life main + an 80-age term rider).
# ---------------------------------------------------------------------------

def test_coverage_term_cuts_a_death_rider_at_its_own_maturity():
    """A DEATH rider with coverage_term=T pays with the main before T and
    nothing from month T on; the main (no term) is unaffected. Hand-checked
    against the no-term run (the rider amount is simply removed past T)."""
    import dataclasses
    basis = _basis()
    a, b, term, T = 6e7, 4e7, 120, 60     # main 6e7 to 120m, rider 4e7 to 60m

    base = ModelPoints(
        issue_age=np.array([40.0]),
        premium=np.array([80_000.0]),
        term_months=np.array([term]),
        coverage_index=np.array([DEATH, DEATH]),
        coverage_amount=np.array([a, b]),
        coverage_offset=np.array([0, 2]),
        calculation_methods=PATTERNS,
    )
    termed = dataclasses.replace(base, coverage_term=np.array([0, T]))

    m0 = measure(base, basis)
    m1 = measure(termed, basis)
    mort0 = m0.cashflows.mortality_cf[0]
    mort1 = m1.cashflows.mortality_cf[0]
    # Before T: identical (both pay on a + b).
    assert np.allclose(mort1[:T], mort0[:T])
    # From T on: the rider is gone, so only the a-share remains.
    assert np.allclose(mort1[T:], mort0[T:] * (a / (a + b)))
    assert np.all(mort1[T:] < mort0[T:] - 1e-9)
    # Dropping the rider past T lowers the liability.
    assert m1.bel[0] < m0.bel[0]


def test_coverage_term_at_or_past_boundary_is_a_noop():
    """coverage_term >= the contract boundary cuts nothing -- identical to the
    no-term run (the default 0 and any T >= boundary both run full)."""
    import dataclasses
    basis = _basis()
    term = 120
    base = ModelPoints(
        issue_age=np.array([40.0]), premium=np.array([80_000.0]),
        term_months=np.array([term]),
        coverage_index=np.array([DEATH, DEATH]),
        coverage_amount=np.array([6e7, 4e7]),
        coverage_offset=np.array([0, 2]),
        calculation_methods=PATTERNS,
    )
    past = dataclasses.replace(base, coverage_term=np.array([0, term]))
    m0, m1 = measure(base, basis), measure(past, basis)
    assert np.allclose(m0.cashflows.mortality_cf, m1.cashflows.mortality_cf)
    assert np.isclose(m0.bel[0], m1.bel[0])


def test_coverage_term_routes_fast_to_full():
    """The fused fast path does not carry coverage_term, so a book with it
    routes to the full path; the fast call returns the full-path BEL."""
    import dataclasses
    basis = _basis()
    base = ModelPoints(
        issue_age=np.array([40.0]), premium=np.array([80_000.0]),
        term_months=np.array([120]),
        coverage_index=np.array([DEATH, DEATH]),
        coverage_amount=np.array([6e7, 4e7]),
        coverage_offset=np.array([0, 2]),
        calculation_methods=PATTERNS,
    )
    termed = dataclasses.replace(base, coverage_term=np.array([0, 60]))
    m_full = measure(termed, basis, full=True)
    m_fast = measure(termed, basis, full=False)
    assert np.isclose(m_full.bel_path[0, 0], m_fast.bel[0])
