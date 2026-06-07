"""measure(full=True)/measure(full=False) parity tests -- the regression net for the (B) refactor.

Three gaps the 2nd review surfaced:

1. ``make_death_basis`` wires the in-force decrement and the DEATH
   coverage's payout from a single callable, so every existing test would
   stay green even if the engine silently reverted to the pre-(B) slot-0
   hardwire (using ``mortality_annual`` as the death claim rate). One
   explicit decoupled-rate test plugs that hole.

2. ``measure()`` and ``measure()`` must agree on the same basis even when
   ``settlement_pattern`` is set -- both code paths apply the factor, but
   neither was tested together.

3. ``measure()`` builds the rate-evaluation grid at ``issue_class = 0``; a
   portfolio with non-zero classes would silently land at class 0. Until
   measure() grows per-class grid support it must raise.
"""
import numpy as np
import pytest

from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


def _flat(annual_q):
    return lambda sex, issue_age, duration: np.full(issue_age.shape, annual_q)


# ---------------------------------------------------------------------------
# 1. Decoupled-rate regression net
# ---------------------------------------------------------------------------

def test_value_uses_coverage_rate_not_mortality_annual():
    """A re-introduction of the pre-(B) slot-0 hardwire would make measure()
    use ``mortality_annual`` instead of the DEATH coverage's own rate.
    Two contracts that differ ONLY in the DEATH coverage rate (same
    in-force decrement) must produce different BELs."""
    mort = _flat(_annual(0.005))
    basis_low = Basis(
        mortality_annual=mort,
        lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", _flat(_annual(0.005))),),  # death = mort
    )
    basis_high = Basis(
        mortality_annual=mort,
        lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", _flat(_annual(0.020))),),  # death > mort
    )
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 1_000_000.0},
        premium=12_000.0, term_months=60,
        calculation_methods=PATTERNS,
    )
    v_low  = measure(mp, basis_low, full=False)
    v_high = measure(mp, basis_high, full=False)
    # If slot 0 were hardwired to mortality_annual, both BELs would match
    # (the coverage's own rate would be ignored).
    assert not np.isclose(v_low.bel[0], v_high.bel[0], rtol=1e-6), (
        f"measure() ignored the DEATH coverage rate: BEL is {v_low.bel[0]} "
        f"under both 0.5%% and 2%% death-claim incidence")
    # The higher death-claim rate produces a larger claim PV (more onerous).
    assert v_high.bel[0] > v_low.bel[0]


def test_measure_uses_coverage_rate_not_mortality_annual():
    """Same regression check on measure()."""
    mort = _flat(_annual(0.005))
    basis_low = Basis(
        mortality_annual=mort,
        lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", _flat(_annual(0.005))),),
    )
    basis_high = Basis(
        mortality_annual=mort,
        lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", _flat(_annual(0.020))),),
    )
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 1_000_000.0},
        premium=12_000.0, term_months=60,
        calculation_methods=PATTERNS,
    )
    m_low  = measure(mp, basis_low)
    m_high = measure(mp, basis_high)
    assert not np.isclose(m_low.bel_path[0, 0], m_high.bel_path[0, 0], rtol=1e-6)
    assert m_high.bel_path[0, 0] > m_low.bel_path[0, 0]


# ---------------------------------------------------------------------------
# 2. measure() + settlement_pattern parity
# ---------------------------------------------------------------------------

def test_value_and_measure_agree_with_settlement_pattern():
    """``settlement_pattern`` discounts claim outflows to their payment
    dates. measure()'s fused path applies the factor inline; measure()'s
    detailed path multiplies the cash flow arrays. The two must agree."""
    basis = make_death_basis(
        mortality_q     = 0.005,
        lapse_q         = 0.01,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = 0.10,
        settlement_pattern = np.array([0.5, 0.3, 0.2]),
    )
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 1e8},
        premium=80_000.0, term_months=120,
        calculation_methods=PATTERNS,
    )
    v = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(v.bel[0], m.bel_path[0, 0])
    assert np.isclose(v.ra[0],  m.ra_path[0, 0])
    assert np.isclose(v.csm[0], m.csm_path[0, 0])


# ---------------------------------------------------------------------------
# 3. measure(full=False) auto-routes non-zero issue_class to the full kernel
# ---------------------------------------------------------------------------

def test_value_auto_routes_nonzero_issue_class():
    """The fast grid is built at class 0, so a non-zero issue_class book is
    auto-routed to the full kernel (no longer raises) -- byte-identical."""
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        premium=np.array([12_000.0]),
        term_months=np.array([60]),
        issue_class=np.array([1]),               # non-default class
        benefits={0: np.array([1e8])},
        calculation_methods=PATTERNS,
    )
    basis = make_death_basis(mortality_q=0.005, lapse_q=0.01)
    fast = measure(mp, basis, full=False)
    full = measure(mp, basis)
    assert fast.bel.shape[0] == 1
    assert np.allclose(fast.bel, full.bel)


def test_value_accepts_default_issue_class():
    """The default (zero everywhere) issue_class must not trigger the guard."""
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 1e8},
        premium=12_000.0, term_months=60,
        calculation_methods=PATTERNS,
    )
    basis = make_death_basis(mortality_q=0.005, lapse_q=0.01)
    v = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(v.bel[0], m.bel_path[0, 0])
