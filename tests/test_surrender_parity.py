"""Fast path surrender_cf -- measure(full=False) must match measure(full=True) BEL.

``measure(full=True)`` has carried surrender value since the surrender
mechanism landed; the fused fast path ``measure(full=False)`` must match. This file
pins the invariant that the two paths return the same BEL when a
``surrender_value_curve`` is set, across all CPU kernels:

* scalar fast path (no StateModel, no waiver),
* Markov codegen (WAIVER_MODEL),
* semi-Markov codegen (sojourn-aware cohort tracking).
"""
import numpy as np

from fastcashflow import (Basis, CalculationMethod, ExpenseItem, ModelPoints,
                          STATE_MODELS, CoverageRate)
from fastcashflow.gmm import measure


def _flat_rate(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis(**overrides):
    base = dict(
        mortality_annual=_flat_rate(0.005),
        lapse_annual=_flat_rate(0.05),
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.10,
        morbidity_cv=0.10,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    100_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  30_000.0),
        ),
        coverages=(CoverageRate("DEATH", _flat_rate(0.005)),),
    )
    base.update(overrides)
    return Basis(**base)


def _mp():
    return ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=50_000.0, term_months=240,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )


def test_value_scalar_matches_measure_with_surrender():
    """Scalar fast path -- no StateModel, no waiver. measure() must agree
    with measure() to floating-point tolerance once surrender is on."""
    n_time = 240
    # A non-trivial monotone surrender curve: 0 in years 1-2 (typical
    # surrender penalty), ramping up to 1.0 by the end of the term.
    curve = np.clip((np.arange(n_time) - 24) / (n_time - 24.0), 0.0, 1.0)
    basis = _basis(surrender_value_curve=curve)
    mp = _mp()
    v = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(v.bel[0], m.bel_path[0, 0])


def test_value_scalar_zero_curve_matches_no_surrender():
    """``surrender_value_curve = 0`` everywhere collapses to the legacy
    no-surrender BEL."""
    n_time = 240
    basis_off = _basis()
    basis_on = _basis(surrender_value_curve=np.zeros(n_time))
    mp = _mp()
    assert np.isclose(measure(mp, basis_off, full=False).bel[0], measure(mp, basis_on, full=False).bel[0])


def test_value_surrender_increases_bel():
    """Adding a positive surrender curve adds an insurer outflow on lapse;
    BEL (claims - premiums + surrender) goes up."""
    n_time = 240
    basis_off = _basis()
    basis_on = _basis(surrender_value_curve=np.full(n_time, 0.5))
    mp = _mp()
    assert measure(mp, basis_on, full=False).bel[0] > measure(mp, basis_off, full=False).bel[0]


def test_surrender_scales_linearly_in_count():
    """A 10-policy grouped MP must produce 10x the surrender flow of an
    otherwise-identical 1-policy MP -- not 100x. Earlier the projection
    multiplied lapse_flow (already inforce-weighted) by cum_premium (also
    inforce-weighted), giving a cnt^2 scaling."""
    n_time = 120
    basis = _basis(surrender_value_curve=np.full(n_time, 0.5))
    mp_single = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=50_000.0, term_months=n_time, count=1.0,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    mp_grouped = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=50_000.0, term_months=n_time, count=10.0,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    m_single = measure(mp_single, basis)
    m_grouped = measure(mp_grouped, basis)
    ratio = m_grouped.cashflows.surrender_cf.sum() / m_single.cashflows.surrender_cf.sum()
    # Linear in count -- ratio must be ~10, not ~100.
    assert np.isclose(ratio, 10.0)


def test_value_state_model_matches_measure_with_surrender():
    """Markov codegen path (WAIVER_MODEL) -- measure() must agree with
    measure() once surrender is on."""
    n_time = 240
    curve = np.clip((np.arange(n_time) - 24) / (n_time - 24.0), 0.0, 1.0)
    basis = _basis(
        surrender_value_curve=curve,
        state_model=STATE_MODELS["WAIVER"],
        waiver_incidence_annual=_flat_rate(0.001),
    )
    mp = _mp()
    v = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(v.bel[0], m.bel_path[0, 0])


# --- amount_per_policy parity (the surrender curve is the per-policy amount
#     at each duration, applied to the in-force scalar rather than to
#     cumulative premium) ---------------------------------------------------

def _amount_curve(n_time):
    """A monotone per-policy surrender amount: 0 in years 1-2, ramping to
    1,000,000 by the end of the term."""
    return np.clip((np.arange(n_time) - 24) / (n_time - 24.0), 0.0, 1.0) * 1e6


def test_value_scalar_matches_measure_amount_per_policy():
    """Scalar fast path -- amount_per_policy must match the full projection."""
    n_time = 240
    basis = _basis(surrender_value_curve=_amount_curve(n_time),
                   surrender_value_basis="amount_per_policy")
    mp = _mp()
    v = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(v.bel[0], m.bel_path[0, 0])


def test_value_state_model_matches_measure_amount_per_policy():
    """Markov codegen path (WAIVER) -- amount_per_policy must match the full
    projection. The codegen kernel branches on the baked surrender mode, so
    this pins that the amount form is generated correctly."""
    n_time = 240
    basis = _basis(
        surrender_value_curve=_amount_curve(n_time),
        surrender_value_basis="amount_per_policy",
        state_model=STATE_MODELS["WAIVER"],
        waiver_incidence_annual=_flat_rate(0.001),
    )
    mp = _mp()
    v = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(v.bel[0], m.bel_path[0, 0])


# --- amount_per_unit parity (the surrender amount additionally scales by the
#     per-MP surrender_base_amount; the kernels multiply by a per-MP base
#     array, 1.0 for the other modes) -----------------------------------------

def _mp_with_base(n_time, base):
    """One MP carrying a surrender_base_amount."""
    return ModelPoints(
        issue_age=np.array([40]), premium=np.array([50_000.0]),
        term_months=np.array([n_time]), benefits={"DEATH": np.array([1e8])},
        count=np.array([1.0]), surrender_base_amount=np.array([float(base)]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )


def test_value_scalar_matches_measure_amount_per_unit():
    """Scalar fast path -- amount_per_unit must match the full projection."""
    n_time = 240
    # per-unit curve: surrender per unit of base, x base 100,000.
    curve = np.clip((np.arange(n_time) - 24) / (n_time - 24.0), 0.0, 1.0)
    basis = _basis(surrender_value_curve=curve,
                   surrender_value_basis="amount_per_unit")
    mp = _mp_with_base(n_time, 100_000.0)
    v = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(v.bel[0], m.bel_path[0, 0])


def test_value_state_model_matches_measure_amount_per_unit():
    """Markov codegen path (WAIVER) -- amount_per_unit must match the full
    projection. Pins that the per-MP base array reaches the codegen kernel."""
    n_time = 240
    curve = np.clip((np.arange(n_time) - 24) / (n_time - 24.0), 0.0, 1.0)
    basis = _basis(
        surrender_value_curve=curve,
        surrender_value_basis="amount_per_unit",
        state_model=STATE_MODELS["WAIVER"],
        waiver_incidence_annual=_flat_rate(0.001),
    )
    mp = _mp_with_base(n_time, 100_000.0)
    v = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(v.bel[0], m.bel_path[0, 0])


# ---------------------------------------------------------------------------
# P1-1: surrender follows the state-machine lapse, not a global rate on the
# total in-force. A paid-up state lapses at lapse_paidup; a non-lapsing state
# (WAIVER) is never surrendered.
# ---------------------------------------------------------------------------
from fastcashflow import State, Transition, StateModel
from fastcashflow.basis import annual_to_monthly


def test_paidup_surrender_uses_lapse_paidup_not_global_lapse():
    """A paid-up cohort surrenders at lapse_paidup. The pre-fix kernel applied
    the global lapse_annual to the total in-force -- a 10x overstatement here."""
    zero = lambda s, a, d: np.full(np.shape(a), 0.0)
    paidup = StateModel(states=(
        State("paidup", pays_premium=False, transitions=(
            Transition("mortality"), Transition("lapse_paidup"))),
    ), seating=(0,))
    n, V = 12, 1000.0
    basis = Basis(
        mortality_annual    = zero,
        lapse_annual        = lambda s, a, d: np.full(np.shape(d), 0.10),  # global 10%
        lapse_paidup_annual = lambda s, a, d: np.full(np.shape(d), 0.01),  # paid-up 1%
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        surrender_value_curve=np.full(n, V), surrender_value_basis="amount_per_policy",
        state_model=paidup, coverages=(CoverageRate("DEATH", zero),))
    mp = ModelPoints(
        issue_age=np.array([40], dtype=np.int64), benefits={"DEATH": np.array([0.0])},
        premium=np.array([0.0]), term_months=np.array([n], dtype=np.int64),
        state=np.array([0], dtype=np.int64),
        calculation_methods={"DEATH": CalculationMethod.DEATH})
    m = measure(mp, basis, full=True)
    inforce = m.cashflows.inforce[0]
    expected = inforce * annual_to_monthly(0.01) * V        # the paid-up rate
    buggy = inforce * annual_to_monthly(0.10) * V           # the old global rate
    assert np.allclose(m.cashflows.surrender_cf[0], expected)
    assert not np.allclose(m.cashflows.surrender_cf[0], buggy)
    # full == fast (the fused kernel applies the same state-machine lapse)
    assert np.isclose(m.bel[0], measure(mp, basis, full=False).bel[0])


def test_waiver_state_is_not_surrendered():
    """The WAIVER state does not lapse, so its in-force is never surrendered.
    The surrender is exactly the active state's lapse exits."""
    zero = lambda s, a, d: np.full(np.shape(a), 0.0)
    n, V = 24, 500.0
    basis = Basis(
        mortality_annual        = zero,
        lapse_annual            = lambda s, a, d: np.full(np.shape(d), 0.08),
        waiver_incidence_annual = lambda s, a, d: np.full(np.shape(a), 0.30),  # heavy -> waiver
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        surrender_value_curve=np.full(n, V), surrender_value_basis="amount_per_policy",
        state_model=STATE_MODELS["WAIVER"], coverages=(CoverageRate("DEATH", zero),))
    mp = ModelPoints(
        issue_age=np.array([40], dtype=np.int64), benefits={"DEATH": np.array([0.0])},
        premium=np.array([1000.0]), term_months=np.array([n], dtype=np.int64),
        calculation_methods={"DEATH": CalculationMethod.DEATH})
    m = measure(mp, basis, full=True)
    # The total in-force exceeds the active in-force (a waiver fraction builds
    # up), so a global-rate surrender (lapse x total) would strictly exceed the
    # true surrender (lapse x active). Confirm the surrender is below that bound.
    inforce = m.cashflows.inforce[0]
    global_bound = inforce * annual_to_monthly(0.08) * V
    surr = m.cashflows.surrender_cf[0]
    assert np.all(surr <= global_bound + 1e-9)
    assert np.any(surr < global_bound - 1e-6)        # strictly less once waiver builds up
    assert np.isclose(m.bel[0], measure(mp, basis, full=False).bel[0])
