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
from numba import cuda

from fastcashflow import Basis, CalculationMethod, CoverageRate, ExpenseItem, ModelPoints
from fastcashflow.multistate import Model
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
        issue_age=40, benefits={"DEATH": 1_000_000.0},
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
        issue_age=40, benefits={"DEATH": 1_000_000.0},
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
        issue_age=40, benefits={"DEATH": 1e8},
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
        benefits={"DEATH": np.array([1e8])},
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
        issue_age=40, benefits={"DEATH": 1e8},
        premium=12_000.0, term_months=60,
        calculation_methods=PATTERNS,
    )
    basis = make_death_basis(mortality_q=0.005, lapse_q=0.01)
    v = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(v.bel[0], m.bel_path[0, 0])


# ---------------------------------------------------------------------------
# 4. Fused fast path (`measure(full=False)`) vs the detailed path, and the
#    GPU backend vs the CPU backend.
# ---------------------------------------------------------------------------
def test_value_matches_measure():
    """The fast fused path reproduces the detailed path's headline numbers."""
    def mortality_annual(sex, issue_age, duration):
        attained = issue_age + duration
        annual_q = 0.0008 * (1.0 + 0.05 * (attained - 30.0))
        return annual_q

    basis = Basis(
        mortality_annual=mortality_annual,
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.012)),
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition", "per_policy",    250_000.0),
            ExpenseItem("maintenance", "per_policy",  48_000.0),
        ),
        ra_confidence=0.85,
        mortality_cv=0.12,
        coverages=(CoverageRate("DEATH", mortality_annual),),
    )
    # distinct and repeated issue ages -- exercises the unique-age grid
    mps = ModelPoints(
        issue_age=np.array([30, 45, 45, 55, 38]),
        benefits={"DEATH": np.array([1e8, 5e7, 8e7, 3e7, 6e7])},
        premium=np.array([70_000, 90_000, 110_000, 130_000, 80_000]),
        term_months=np.array([120, 120, 120, 120, 120]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )

    fast = measure(mps, basis, full=False)
    detailed = measure(mps, basis)

    assert np.allclose(fast.bel, detailed.bel_path[:, 0])
    assert np.allclose(fast.ra, detailed.ra_path[:, 0])
    assert np.allclose(fast.csm, detailed.csm_path[:, 0])
    assert np.allclose(fast.loss_component, detailed.loss_component)


def test_value_onerous():
    """The fast path also flags onerous contracts -- CSM floored at 0."""
    basis = Basis(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.05)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, 0.0),
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.05,
        coverages=(CoverageRate("DEATH", lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.05))),),
    )
    mps = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 1_000_000.0},
        premium=100.0, term_months=12,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    v = measure(mps, basis, full=False)
    assert v.csm[0] == 0.0
    assert v.loss_component[0] > 0.0


@pytest.mark.skipif(not cuda.is_available(), reason="no CUDA device available")
@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
def test_fast_gpu_matches_cpu():
    """The GPU backend reproduces the CPU backend exactly."""
    def mortality_annual(sex, issue_age, duration):
        attained = issue_age + duration
        annual_q = 0.0008 * (1.0 + 0.05 * (attained - 30.0))
        return annual_q

    basis = Basis(
        mortality_annual=mortality_annual,
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.012)),
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition", "per_policy",    250_000.0),
            ExpenseItem("maintenance", "per_policy",  48_000.0),
        ),
        ra_confidence=0.85,
        mortality_cv=0.12,
        coverages=(CoverageRate("DEATH", mortality_annual),),
    )
    rng = np.random.default_rng(7)
    n = 5_000
    mps = ModelPoints(
        issue_age=rng.integers(25, 60, n),
        benefits={"DEATH": rng.integers(10, 100, n) * 1_000_000},
        premium=rng.integers(3, 15, n) * 10_000,
        term_months=np.full(n, 120),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )

    cpu = measure(mps, basis, backend="cpu", full=False)
    gpu = measure(mps, basis, backend="gpu", full=False)

    assert np.allclose(gpu.bel, cpu.bel)
    assert np.allclose(gpu.ra, cpu.ra)
    assert np.allclose(gpu.csm, cpu.csm)
    assert np.allclose(gpu.loss_component, cpu.loss_component)


@pytest.mark.skipif(not cuda.is_available(), reason="no CUDA device available")
@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
def test_fast_gpu_matches_cpu_with_transition():
    """GPU and CPU agree under a waiver transition with a diagnosis coverage --
    the GPU two-track in-force and diagnosis pool reproduce the CPU kernel."""
    def flat(rate):
        return lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(rate))

    basis = Basis(
        mortality_annual=flat(0.001),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.012)),
        waiver_incidence_annual=flat(0.02),
        state_machine=Model.from_preset("ACTIVE_WAIVER"),
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition", "per_policy",    200_000.0),
            ExpenseItem("maintenance", "per_policy",  48_000.0),
        ),
        ra_confidence=0.85,
        mortality_cv=0.12,
        morbidity_cv=0.10,
        coverages=(
            CoverageRate("DEATH", flat(0.001)),
            CoverageRate("dx", flat(0.003)),
        ),
    )
    rng = np.random.default_rng(13)
    n = 4_000
    mps = ModelPoints(
        issue_age=rng.integers(25, 60, n).astype(float),
        benefits={"DEATH": rng.integers(10, 100, n) * 1_000_000.0, "dx": rng.integers(5, 30, n) * 1_000_000.0},
        premium=rng.integers(3, 15, n) * 10_000.0,
        term_months=np.full(n, 120),
        state=rng.integers(0, 3, n),
        calculation_methods={"DEATH": CalculationMethod.DEATH, "dx": CalculationMethod.DIAGNOSIS},
    )
    cpu = measure(mps, basis, backend="cpu", full=False)
    gpu = measure(mps, basis, backend="gpu", full=False)

    assert np.allclose(gpu.bel, cpu.bel)
    assert np.allclose(gpu.ra, cpu.ra)
    assert np.allclose(gpu.csm, cpu.csm)
    assert np.allclose(gpu.loss_component, cpu.loss_component)
