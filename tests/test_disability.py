"""Phase (b) Stage 3 -- disability cash flows.

Two cash-flow kinds attached to the state machine as data:

* a state benefit -- disability income paid each month a benefit state is
  occupied (``State.benefit`` + ``ModelPoints.disability_income``);
* a transition lump sum -- a one-off benefit when a flagged transition fires
  (``Transition.lump_sum`` + ``ModelPoints.disability_benefit``).

The disability model below is the active / waiver model with the waiver
state turned into a benefit-paying ``disabled`` state and the inception
transition flagged to pay a lump sum. Every figure is derived by hand on a
flat, zero-discount basis.
"""
import numpy as np
import pytest

from fastcashflow import (
    STATE_WAIVER,
    Assumptions,
    ModelPoints,
    State,
    StateModel,
    Transition,
    measure,
    value,
)
from fastcashflow.statemodel import compile_state_model

# Standard-normal 75th percentile -- so the RA check does not lean on the
# engine's own quantile code.
Z_75 = 0.6744897501960817


def _annual(monthly):
    """The annual rate whose constant-force monthly equivalent is ``monthly``."""
    return 1.0 - (1.0 - monthly) ** 12


def _disability_model(*, lump_sum=True) -> StateModel:
    """Active / disabled model -- disability income on the disabled state,
    an optional lump sum on the inception transition."""
    return StateModel(
        states=(
            State("active", premium=True, transitions=(
                Transition("mortality"),
                Transition("waiver_incidence", to="disabled", lump_sum=lump_sum),
                Transition("lapse"),
            )),
            State("disabled", benefit=True, transitions=(
                Transition("mortality"),
            )),
        ),
        seating=(0, 1, 1),
    )


def _asmp(*, q=0.01, lapse=0.0, inception=0.05, disability_cv=0.0,
          lump_sum=True) -> Assumptions:
    """Flat-rate, zero-discount basis. ``q`` / ``lapse`` / ``inception`` are
    the monthly rates the hand calculations use."""
    return Assumptions(
        mortality_annual=lambda s, a, d: np.full(a.shape, _annual(q)),
        lapse_annual=lambda sex, issue_age, d: np.full(d.shape, _annual(lapse)),
        waiver_incidence_annual=lambda s, a, d: np.full(a.shape, _annual(inception)),
        discount_annual=0.0,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
        disability_cv=disability_cv,
        state_model=_disability_model(lump_sum=lump_sum),
    )


# ---------------------------------------------------------------------------
# compile_state_model -- the new flags
# ---------------------------------------------------------------------------

def test_compile_marks_benefit_state_and_lump_sum():
    """compile_state_model surfaces the benefit state and the lump-sum edge."""
    rates = {"mortality": np.array([[0.01]]),
             "waiver_incidence": np.array([[0.05]]),
             "lapse": np.array([[0.0]])}
    edge_from, edge_to, _, edge_lump_sum, n_states, premium, benefit = (
        compile_state_model(_disability_model(), rates))
    assert n_states == 2
    assert list(premium) == [True, False]    # active pays premium
    assert list(benefit) == [False, True]    # disabled pays a benefit
    lump = {(int(f), int(t)): bool(s)
            for f, t, s in zip(edge_from, edge_to, edge_lump_sum)}
    assert lump[(0, 1)] is True              # active -> disabled carries it
    assert lump[(0, 0)] is False             # the stay edges do not
    assert lump[(1, 1)] is False


def test_lump_sum_on_exit_transition_rejected():
    """A lump sum must attach to a transition with a destination."""
    with pytest.raises(ValueError, match="lump-sum"):
        StateModel(states=(
            State("active", premium=True, transitions=(
                Transition("mortality", lump_sum=True),    # to=None -- an exit
            )),
        ))


# ---------------------------------------------------------------------------
# Disability income -- a state cash flow on a benefit state
# ---------------------------------------------------------------------------

def test_disability_income_hand_calculation():
    """A contract already disabled: income is paid on the disabled occupancy,
    which decays by mortality alone. Every figure by hand."""
    income = 500_000.0
    mp = ModelPoints.single(issue_age=45, death_benefit=0.0, level_premium=0.0,
                            term_months=3, disability_income=income,
                            state=STATE_WAIVER)        # seated on 'disabled'
    asmp = _asmp(disability_cv=0.20)

    occ = [1.0, 0.99, 0.99 ** 2]                       # mortality only
    disability_cf = [o * income for o in occ]
    bel = sum(disability_cf)                  # zero discount, no premium

    res = measure(mp, asmp)
    assert np.allclose(res.cashflows.disability_cf[0], disability_cf)
    assert np.isclose(res.bel[0, 0], bel)
    assert np.isclose(value(mp, asmp).bel[0], bel)
    # RA: the disability-risk component, z * cv * PV(disability).
    assert np.isclose(value(mp, asmp).ra[0], Z_75 * 0.20 * bel)


def test_disability_income_needs_a_benefit_state():
    """With no benefit state the income is never paid -- the default waiver
    model has no benefit state, so disability_income is inert there."""
    kw = dict(issue_age=45, death_benefit=0.0, level_premium=0.0,
              term_months=12, disability_income=500_000.0, state=STATE_WAIVER)
    # default model (no state_model) -- waiver state is not a benefit state
    plain = Assumptions(
        mortality_annual=lambda s, a, d: np.full(a.shape, _annual(0.01)),
        lapse_annual=lambda sex, issue_age, d: np.full(d.shape, 0.0),
        discount_annual=0.0, expense_acquisition=0.0,
        expense_maintenance_annual=0.0, expense_inflation=0.0,
        ra_confidence=0.75, mortality_cv=0.10)
    res = measure(ModelPoints.single(**kw), plain)
    assert np.all(res.cashflows.disability_cf[0] == 0.0)


# ---------------------------------------------------------------------------
# Disability lump sum -- a transition cash flow
# ---------------------------------------------------------------------------

def test_disability_lump_sum_hand_calculation():
    """An active contract: a lump sum is paid on each cohort that becomes
    disabled. Two-month term, derived by hand from the transition flow."""
    lump = 10_000_000.0
    mp = ModelPoints.single(issue_age=40, death_benefit=0.0, level_premium=0.0,
                            term_months=2, disability_benefit=lump)
    asmp = _asmp(q=0.01, lapse=0.0, inception=0.05)

    # active -> disabled transition prob = (survive death) * inception
    incep = 0.99 * 0.05
    active = [1.0, 0.99 * 0.95]                # active occupancy at t = 0, 1
    disability_cf = [active[0] * incep * lump, active[1] * incep * lump]
    bel = sum(disability_cf)                   # zero discount, no premium

    res = measure(mp, asmp)
    assert np.allclose(res.cashflows.disability_cf[0], disability_cf)
    assert np.isclose(res.bel[0, 0], bel)
    assert np.isclose(value(mp, asmp).bel[0], bel)


def test_lump_sum_off_when_unflagged():
    """With the inception transition not flagged, no lump sum is paid even
    when disability_benefit is set."""
    mp = ModelPoints.single(issue_age=40, death_benefit=0.0, level_premium=0.0,
                            term_months=24, disability_benefit=5_000_000.0)
    res = measure(mp, _asmp(lump_sum=False))
    assert np.all(res.cashflows.disability_cf[0] == 0.0)


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------

def test_measure_value_agree_disability_portfolio():
    """The detailed and the fused path agree on a mixed disability portfolio
    -- income, lump sum and mixed starting states."""
    rng = np.random.default_rng(5)
    n = 40
    mps = ModelPoints(
        issue_age=rng.integers(35, 55, n).astype(float),
        death_benefit=rng.integers(0, 50, n) * 1_000_000.0,
        level_premium=rng.integers(2, 8, n) * 10_000.0,
        term_months=np.full(n, 120),
        disability_income=rng.integers(0, 5, n) * 100_000.0,
        disability_benefit=rng.integers(0, 30, n) * 1_000_000.0,
        state=rng.integers(0, 2, n),           # active or disabled start
    )
    asmp = _asmp(q=0.008, lapse=0.04, inception=0.02, disability_cv=0.25)
    m, v = measure(mps, asmp), value(mps, asmp)
    assert np.allclose(m.bel[:, 0], v.bel)
    assert np.allclose(m.ra[:, 0], v.ra)


def test_disability_cv_drives_the_risk_adjustment():
    """The disability-risk RA component is governed by disability_cv -- with
    cv = 0 a disability-only contract carries no RA."""
    mp = ModelPoints.single(issue_age=45, death_benefit=0.0, level_premium=0.0,
                            term_months=12, disability_income=300_000.0,
                            state=STATE_WAIVER)
    assert value(mp, _asmp(disability_cv=0.0)).ra[0] == 0.0
    assert value(mp, _asmp(disability_cv=0.30)).ra[0] > 0.0
