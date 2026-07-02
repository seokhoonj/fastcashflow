"""Per-state policy value V^i(t) -- measure(..., state_reserve=True).

The opt-in per-state reserve decomposes the aggregate BEL into one value per
transient state, via the backward (Thiele) recursion over the compiled edge
list. The defining identity is

    sum_i occ_i(t) V^i(t) == bel_path[t]

for every month t (the aggregate roll-forward is the occupancy-weighted sum of
the per-state recursions). The reference occupancy here is recomputed
independently from the compiled edges, so the check does not lean on the
engine's own internal parity guard.

The test model is a Markov disability-income contract (active pays premium ->
disabled pays a monthly income -> dead; active also lapses), which exercises a
genuinely non-trivial V^i: at an interior month multiple states are occupied
and their values differ (a disabled life is worth far more to the liability
than an active one). A one-state contract collapses V^0 back to bel_path.
"""
import numpy as np
import pytest

from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow._numerics import _forward_occupancy_kernel
from fastcashflow.gmm import measure
from fastcashflow.multistate import State, Model, Transition

# active --ci_incidence--> disabled (pays income) --mortality--> dead;
# active also dies / lapses. No recovery -> plain Markov, V^i(t) well defined.
DISABILITY = Model(states=(
    State("active", pays_premium=True, transitions=(
        Transition("ci_incidence", to="disabled"),
        Transition("mortality"),
        Transition("lapse"),
    )),
    State("disabled", pays_periodic_benefit=True, transitions=(
        Transition("mortality"),
    )),
))


def _disability_case(onset=0.01, income=1000.0, lapse=0.05):
    q = 0.005
    mp = ModelPoints(
        sex=np.array([0, 1]),
        issue_age=np.array([40.0, 45.0]),
        term_months=np.array([240, 240]),
        premium_term_months=np.array([240, 240]),
        premium=np.array([100.0, 100.0]),
        count=np.array([1.0, 3.0]),
        disability_income=np.full(2, income),
        benefits={"DEATH": np.array([100_000.0, 100_000.0])},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        product=np.array(["DI", "DI"]),
        channel=np.array(["FC", "FC"]),
    )
    basis = Basis(
        mortality_annual=q,
        lapse_annual=lapse,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", q),),
        state_machine=DISABILITY,
        ci_incidence_annual=onset,
    )
    return mp, basis


def _parity(m, mp):
    """max relative error of sum_i occ_i V^i vs bel_path, plus the occupancy."""
    rbs = np.asarray(m.state_reserve)          # (n_mp, n_states, n_time+1)
    bel_path = np.asarray(m.bel_path)             # (n_mp, n_time+1)
    st = m.cashflows.state_trace
    n_time = rbs.shape[2] - 1
    occ = _forward_occupancy_kernel(
        st.edge_from, st.edge_to, st.edge_prob, st.n_states,
        st.start_state, st.count, mp.contract_boundary_months, n_time)
    parity = (occ * rbs).sum(axis=1)
    err = np.max(np.abs(parity - bel_path) / np.maximum(np.abs(bel_path), 1.0))
    return err, occ, rbs, bel_path


def test_state_reserve_reconciles_to_bel_path():
    mp, basis = _disability_case()
    m = measure(mp, basis, full=True, state_reserve=True)
    err, occ, rbs, bel_path = _parity(m, mp)
    assert err < 1e-9, f"parity sum_i occ_i V^i != bel_path: {err:.2e}"
    # shape: (n_mp, n_states=2, n_time+1)
    assert rbs.shape == (2, 2, int(mp.contract_boundary_months.max()) + 1)


def test_state_reserve_interior_multistate_nontrivial():
    """At an interior month with both states occupied, V^active != V^disabled."""
    mp, basis = _disability_case()
    m = measure(mp, basis, full=True, state_reserve=True)
    _, occ, rbs, _ = _parity(m, mp)
    # A month where both states carry occupancy for model point 0.
    occ0 = occ[0]
    both = [t for t in range(occ0.shape[1]) if np.all(occ0[:, t] > 1e-9)]
    assert both, "expected an interior month with both states occupied"
    t = both[len(both) // 2]
    v_active, v_disabled = rbs[0, 0, t], rbs[0, 1, t]
    # The disabled state owes a stream of income + the death benefit; the active
    # state still collects premium -- the two values are materially different.
    assert abs(v_disabled - v_active) > 1.0
    assert v_disabled > v_active


def test_state_reserve_column0_matches_bel():
    """Column 0 of the occupancy-weighted per-state value is the inception BEL."""
    mp, basis = _disability_case()
    m = measure(mp, basis, full=True, state_reserve=True)
    _, occ, rbs, bel_path = _parity(m, mp)
    inception = (occ[:, :, 0] * rbs[:, :, 0]).sum(axis=1)
    assert np.allclose(inception, np.asarray(m.bel), rtol=1e-10, atol=1e-8)


def test_state_reserve_none_by_default():
    """The per-state value is absent unless explicitly requested."""
    mp, basis = _disability_case()
    m = measure(mp, basis, full=True)
    assert m.state_reserve is None
    assert m.cashflows.state_trace is None


def test_state_reserve_requires_full():
    mp, basis = _disability_case()
    with pytest.raises(ValueError, match="requires full=True"):
        measure(mp, basis, full=False, state_reserve=True)


def test_state_reserve_rejects_account_book():
    """v1 gate: a universal-life account book is not yet supported."""
    face = 100_000_000.0
    mp = ModelPoints(
        sex=np.array([0]),
        issue_age=np.array([40.0]),
        term_months=np.array([12]),
        premium=np.array([500_000.0]),
        count=np.array([1.0]),
        account_value=np.array([0.0]),
        minimum_death_benefit=np.array([face]),
        benefits={"DEATH": np.array([face])},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    basis = Basis(
        mortality_annual=0.0, lapse_annual=0.0, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1, investment_return=0.0,
        coi_annual=0.001,
        coverages=(CoverageRate("DEATH", 0.001, funds_from_account=True,
                                pays_account_balance=True),),
    )
    with pytest.raises(NotImplementedError, match="account"):
        measure(mp, basis, full=True, state_reserve=True)


def test_state_reserve_plain_term_contract():
    """A plain term contract (default state model) still reconciles to bel_path.

    With no state_machine the engine uses its default active / paid-up tracks; the
    paid-up track stays empty here, so the whole liability sits on the active
    state and the occupancy-weighted per-state value still equals bel_path.
    """
    mp = ModelPoints(
        sex=np.array([0]),
        issue_age=np.array([40.0]),
        term_months=np.array([120]),
        premium_term_months=np.array([120]),
        premium=np.array([100.0]),
        count=np.array([1.0]),
        benefits={"DEATH": np.array([100_000.0])},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        product=np.array(["TERM"]),
        channel=np.array(["FC"]),
    )
    q = 0.005
    basis = Basis(
        mortality_annual=q, lapse_annual=0.05, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", q),),
    )
    m = measure(mp, basis, full=True, state_reserve=True)
    err, occ, rbs, bel_path = _parity(m, mp)
    assert err < 1e-9
    # the liability sits entirely on the active track; the paid-up track is empty
    active_only = (occ[:, 0, :] * rbs[:, 0, :])
    assert np.allclose(active_only, bel_path, rtol=1e-9, atol=1e-8)
