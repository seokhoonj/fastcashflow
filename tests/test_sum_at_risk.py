"""Per-transition sum at risk -- measure(..., sum_at_risk=True).

sum_at_risk = S^ij + V^j - V^i is the net exposure if a transition fires: the
benefit paid on the edge, plus the reserve of the destination state, minus the
reserve released by the source state (all per unit in the source state). It is
built on the S1 per-state reserve V^i, so it is a Markov, full=True opt-in.

Three transition kinds are enumerated, one row each on the sum_at_risk axis:
  * death  from every state with a death decrement -- the net amount at risk
    death_benefit - V^i (destination reserve is 0, death leaves the in-force set);
  * lapse  from every state with a lapse decrement -- csv - V^i;
  * transfer for every inter-state edge -- lump + V^j - V^i.

The disability model (active pays premium -> disabled pays income -> dead)
exercises the economically interesting signs: onset is costly, but death while
disabled RELEASES the large disabled-income reserve (a negative NAR).
"""
import dataclasses

import numpy as np
import pytest

from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow.gmm import measure
from fastcashflow.state_model import State, StateModel, Transition

DISABILITY = StateModel(states=(
    State("active", pays_premium=True, transitions=(
        Transition("ci_incidence", to="disabled"),
        Transition("mortality"),
        Transition("lapse"),
    )),
    State("disabled", pays_periodic_benefit=True, transitions=(
        Transition("mortality"),
    )),
))

FACE = 100_000.0


def _case(onset=0.01, income=1000.0):
    mp = ModelPoints(
        sex=np.array([0, 1]),
        issue_age=np.array([40.0, 45.0]),
        term_months=np.array([240, 240]),
        premium_term_months=np.array([240, 240]),
        premium=np.array([100.0, 100.0]),
        count=np.array([1.0, 2.0]),
        disability_income=np.full(2, income),
        benefits={"DEATH": np.array([FACE, FACE])},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        product=np.array(["DI", "DI"]),
        channel=np.array(["FC", "FC"]),
    )
    basis = Basis(
        mortality_annual=0.005, lapse_annual=0.05, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", 0.005),),
        state_model=DISABILITY, ci_incidence_annual=onset,
    )
    return mp, basis


def _by_kind(m):
    """Map (kind, from_name, to_name) -> its sum_at_risk row, model point 0."""
    sar = np.asarray(m.sum_at_risk)
    return {(t.kind, t.from_name, t.to_name): sar[0, k]
            for k, t in enumerate(m.transitions)}


def test_sum_at_risk_shape_and_transition_set():
    mp, basis = _case()
    m = measure(mp, basis, full=True, sum_at_risk=True)
    sar = np.asarray(m.sum_at_risk)
    # n_transition = death(active), death(disabled), lapse(active), transfer(->disabled)
    # (disabled has NO lapse transition, so no lapse row for it)
    assert sar.shape == (2, 4, int(mp.contract_boundary_months.max()) + 1)
    kinds = {(t.kind, t.from_name, t.to_name) for t in m.transitions}
    assert ("death", "active", "death") in kinds
    assert ("death", "disabled", "death") in kinds
    assert ("lapse", "active", "lapse") in kinds
    assert ("transfer", "active", "disabled") in kinds
    assert ("lapse", "disabled", "lapse") not in kinds  # disabled cannot lapse


def test_sum_at_risk_death_is_nar():
    """death sum_at_risk == death_benefit - V^i (the net amount at risk)."""
    mp, basis = _case()
    m = measure(mp, basis, full=True, sum_at_risk=True)
    V = np.asarray(m.state_reserve)          # (n_mp, n_states, n_time+1)
    rows = _by_kind(m)
    nar_active = rows[("death", "active", "death")]
    assert np.allclose(nar_active, FACE - V[0, 0, :], rtol=1e-10, atol=1e-8)


def test_sum_at_risk_death_while_disabled_releases_reserve():
    """Dying while disabled releases the income reserve -> its NAR < the active NAR
    and turns negative once the disabled reserve exceeds the death benefit."""
    mp, basis = _case()
    m = measure(mp, basis, full=True, sum_at_risk=True)
    rows = _by_kind(m)
    t = 120
    nar_active = rows[("death", "active", "death")][t]
    nar_disabled = rows[("death", "disabled", "death")][t]
    assert nar_disabled < nar_active
    assert nar_disabled < 0.0            # disabled reserve > death benefit here


def test_sum_at_risk_transfer_is_reserve_jump():
    """active->disabled transfer sum_at_risk == V^disabled - V^i (no lump here)."""
    mp, basis = _case()
    m = measure(mp, basis, full=True, sum_at_risk=True)
    V = np.asarray(m.state_reserve)
    rows = _by_kind(m)
    onset = rows[("transfer", "active", "disabled")]
    assert np.allclose(onset, V[0, 1, :] - V[0, 0, :], rtol=1e-10, atol=1e-8)
    assert onset[120] > 0.0              # onset is costly


def test_sum_at_risk_implies_state_reserve():
    """sum_at_risk=True fills state_reserve too (it is built from V^i)."""
    mp, basis = _case()
    m = measure(mp, basis, full=True, sum_at_risk=True)
    assert m.state_reserve is not None
    assert m.sum_at_risk is not None and m.transitions is not None


def test_sum_at_risk_none_by_default():
    mp, basis = _case()
    m = measure(mp, basis, full=True)
    assert m.sum_at_risk is None and m.transitions is None


def test_sum_at_risk_requires_full():
    mp, basis = _case()
    with pytest.raises(ValueError, match="requires full=True"):
        measure(mp, basis, full=False, sum_at_risk=True)


def test_sum_at_risk_lapse_uses_surrender_value():
    """With a surrender-value curve, the lapse sum_at_risk = csv - V^i."""
    mp, basis = _case()
    # a flat cash surrender value of 5,000 per policy
    csv = 5_000.0
    basis = dataclasses.replace(basis, surrender_value_curve=np.full(240, csv),
                                surrender_value_basis="amount_per_policy")
    m = measure(mp, basis, full=True, sum_at_risk=True)
    V = np.asarray(m.state_reserve)
    rows = _by_kind(m)
    lapse_active = rows[("lapse", "active", "lapse")]
    # interior months where active still lapses: csv - V^active
    t = 60
    assert np.isclose(lapse_active[t], csv - V[0, 0, t], rtol=1e-7, atol=1e-4)
