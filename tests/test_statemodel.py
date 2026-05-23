"""The in-force state machine -- StateModel as product input.

Phase (b) Stage 2: a product declares its states, transitions and
premium-paying states as data. The default active / waiver model is one
StateModel among many; these tests drive custom ones through both the fused
``value`` and the detailed ``measure`` path, with the figures derived by hand
from the multiple-decrement recursion.
"""
import numpy as np
import pytest

from fastcashflow import (
    STATE_ACTIVE,
    STATE_PAIDUP,
    STATE_WAIVER,
    WAIVER_MODEL,
    Assumptions,
    ModelPoints,
    State,
    StateModel,
    Transition,
    measure,
    value,
)
from fastcashflow.statemodel import compile_state_model


def _annual(monthly):
    """The annual rate whose constant-force monthly equivalent is ``monthly``."""
    return 1.0 - (1.0 - monthly) ** 12


def _asmp(*, waiver_rate=0.0, lapse=0.02, q=0.01, state_model=None) -> Assumptions:
    """Flat-rate, zero-discount, zero-expense basis -- every figure by hand.

    ``q``, ``lapse`` and ``waiver_rate`` are the monthly rates the hand
    calculations use; each is supplied to the engine as the annual rate the
    engine converts straight back to that monthly rate.
    """
    waiver = None
    if waiver_rate != 0.0:
        waiver_a = _annual(waiver_rate)
        def waiver(sex, issue_age, duration):
            return np.full(issue_age.shape, waiver_a)
    q_a = _annual(q)
    lapse_a = _annual(lapse)
    return Assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, q_a),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, lapse_a),
        waiver_inception_annual=waiver,
        discount_annual=0.0,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
        state_model=state_model,
    )


# ---------------------------------------------------------------------------
# compile_state_model -- the StateModel -> kernel edge arrays compiler
# ---------------------------------------------------------------------------

def test_compile_waiver_edges():
    """The waiver model compiles to the expected competing-decrement edges."""
    rates = {
        "mortality": np.array([[0.01]]),
        "waiver_incidence": np.array([[0.05]]),
        "lapse": np.array([[0.02]]),
    }
    (edge_from, edge_to, edge_prob, edge_lump_sum, n_states, premium_state,
     benefit_state) = compile_state_model(WAIVER_MODEL, rates)
    assert n_states == 2
    assert list(premium_state) == [True, False]
    assert list(benefit_state) == [False, False]   # waiver model: no benefit
    assert not edge_lump_sum.any()                 # ... and no lump sums

    prob = {(int(f), int(t)): float(edge_prob[i, 0, 0])
            for i, (f, t) in enumerate(zip(edge_from, edge_to))}
    # active: survive death, then a fraction takes waiver, the rest survive
    # lapse too -- the standard ordered multiple-decrement composition.
    assert np.isclose(prob[(0, 1)], 0.99 * 0.05)
    assert np.isclose(prob[(0, 0)], 0.99 * 0.95 * 0.98)
    assert np.isclose(prob[(1, 1)], 0.99)
    # Occupancy is conserved: what leaves state 0 plus what stays, plus the
    # death and lapse exits, sums to 1.
    death = 0.01
    lapse_exit = 0.99 * 0.95 * 0.02
    assert np.isclose(prob[(0, 1)] + prob[(0, 0)] + death + lapse_exit, 1.0)


def test_compile_missing_rate_raises():
    """A decrement naming a rate that was not supplied is a clear error."""
    with pytest.raises(ValueError, match="lapse"):
        compile_state_model(WAIVER_MODEL, {"mortality": np.array([[0.01]]),
                                           "waiver_incidence": np.array([[0.0]])})


# ---------------------------------------------------------------------------
# StateModel validation
# ---------------------------------------------------------------------------

def test_unknown_destination_state_rejected():
    """A decrement to a state that does not exist is rejected at build time."""
    with pytest.raises(ValueError, match="unknown state"):
        StateModel(states=(
            State("active", premium=True,
                  transitions=(Transition("waiver_inception", to="ghost"),)),
        ))


def test_duplicate_state_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        StateModel(states=(State("a"), State("a")))


def test_seating_out_of_range_rejected():
    with pytest.raises(ValueError, match="seating"):
        StateModel(states=(State("a"),), seating=(0, 1))


# ---------------------------------------------------------------------------
# Custom state machines through the engine
# ---------------------------------------------------------------------------

def test_explicit_waiver_model_matches_default():
    """A StateModel rebuilt to the same shape as the built-in default
    reproduces it exactly -- the default path is just one StateModel."""
    rebuilt = StateModel(
        states=(
            State("active", premium=True, transitions=(
                Transition("mortality"),
                Transition("waiver_inception", to="waiver"),
                Transition("lapse"),
            )),
            State("waiver", transitions=(Transition("mortality"),)),
        ),
        seating=(0, 1, 1),
    )
    kw = dict(issue_age=45, death_benefit=50_000_000.0,
              level_premium=30_000.0, term_months=120)
    for state in (STATE_ACTIVE, STATE_WAIVER, STATE_PAIDUP):
        mp = ModelPoints.single(**kw, state=state)
        default = value(mp, _asmp(waiver_rate=0.03))
        custom = value(mp, _asmp(waiver_rate=0.03, state_model=rebuilt))
        assert np.isclose(default.bel[0], custom.bel[0])


def test_single_state_no_lapse_hand_calculation():
    """A one-state model -- mortality only, no lapse, no waiver. With a flat
    1% mortality the in-force is [1, 0.99, 0.99^2]; every figure by hand."""
    no_lapse = StateModel(states=(
        State("active", premium=True, transitions=(Transition("mortality"),)),
    ))
    death_benefit = 1_000_000.0
    premium = 12_000.0
    mp = ModelPoints.single(issue_age=40, death_benefit=death_benefit,
                            level_premium=premium, term_months=3)
    asmp = _asmp(state_model=no_lapse)

    inforce = [1.0, 0.99, 0.99 ** 2]
    pv_claims = sum(i * 0.01 * death_benefit for i in inforce)
    pv_premiums = sum(i * premium for i in inforce)
    bel = pv_claims - pv_premiums

    val = value(mp, asmp)
    assert np.isclose(val.bel[0], bel)
    assert np.isclose(measure(mp, asmp).bel[0, 0], bel)
    assert np.allclose(measure(mp, asmp).cashflows.inforce[0], inforce)


def test_decrement_order_matters():
    """The decrement order is data: applying lapse before waiver inception
    feeds the waiver state a fraction (1 - lapse) smaller than the default
    waiver-before-lapse order, a different BEL -- derived by hand."""
    lapse_first = StateModel(
        states=(
            State("active", premium=True, transitions=(
                Transition("mortality"),
                Transition("lapse"),
                Transition("waiver_inception", to="waiver"),
            )),
            State("waiver", transitions=(Transition("mortality"),)),
        ),
        seating=(0, 1, 1),
    )
    death_benefit = 1_000_000.0
    premium = 12_000.0
    mp = ModelPoints.single(issue_age=40, death_benefit=death_benefit,
                            level_premium=premium, term_months=2)
    asmp = _asmp(waiver_rate=0.05, lapse=0.02, state_model=lapse_first)

    # t=0: act=1, wav=0.
    #   act[1] = 1 * 0.99 * 0.98 * 0.95 = 0.92169  (death, lapse, then waiver)
    #   wav[1] = 1 * 0.99 * 0.98 * 0.05 = 0.04851
    act1 = 0.99 * 0.98 * 0.95
    wav1 = 0.99 * 0.98 * 0.05
    inforce = [1.0, act1 + wav1]
    pv_claims = sum(i * 0.01 * death_benefit for i in inforce)
    pv_premiums = (1.0 + act1) * premium       # premium on the active track
    bel = pv_claims - pv_premiums

    assert np.isclose(value(mp, asmp).bel[0], bel)
    assert np.allclose(measure(mp, asmp).cashflows.inforce[0], inforce)
    # The default waiver-before-lapse order gives a distinct figure.
    default = value(mp, _asmp(waiver_rate=0.05, lapse=0.02)).bel[0]
    assert not np.isclose(default, bel)


def test_three_state_model_runs():
    """A three-state model (active, waiver, paid-up kept as a distinct state)
    runs through both kernels: n_states = 3 flows through the occupancy
    recursion, and the extra paid-up state -- mortality only, no premium --
    values a paid-up contract exactly as the two-state default does."""
    three = StateModel(
        states=(
            State("active", premium=True, transitions=(
                Transition("mortality"),
                Transition("waiver_inception", to="waiver"),
                Transition("lapse"),
            )),
            State("waiver", transitions=(Transition("mortality"),)),
            State("paidup", transitions=(Transition("mortality"),)),
        ),
        seating=(0, 1, 2),       # active / waiver / paid-up each own a state
    )
    assert three.n_states == 3
    kw = dict(issue_age=42, death_benefit=80_000_000.0,
              level_premium=40_000.0, term_months=180)

    # A paid-up contract: identical to the default, which seats paid-up on
    # the waiver state -- both are mortality-only, premium-free.
    paidup = ModelPoints.single(**kw, state=STATE_PAIDUP)
    base = value(paidup, _asmp(waiver_rate=0.03))
    custom = value(paidup, _asmp(waiver_rate=0.03, state_model=three))
    for field in ("bel", "ra", "csm", "loss_component"):
        assert np.isclose(getattr(base, field)[0], getattr(custom, field)[0])

    # An active contract is unaffected by the unreachable paid-up state.
    active = ModelPoints.single(**kw, state=STATE_ACTIVE)
    assert np.isclose(value(active, _asmp(waiver_rate=0.03)).bel[0],
                      value(active, _asmp(waiver_rate=0.03, state_model=three)).bel[0])


def test_measure_and_value_agree_under_custom_model():
    """The detailed and the fused path agree across a mixed-state portfolio
    valued on a custom three-state model."""
    three = StateModel(
        states=(
            State("active", premium=True, transitions=(
                Transition("mortality"),
                Transition("waiver_inception", to="waiver"),
                Transition("lapse"),
            )),
            State("waiver", transitions=(Transition("mortality"),)),
            State("paidup", transitions=(Transition("mortality"),)),
        ),
        seating=(0, 1, 2),
    )
    rng = np.random.default_rng(7)
    n = 50
    mps = ModelPoints(
        issue_age=rng.integers(30, 55, n).astype(float),
        death_benefit=rng.integers(10, 80, n) * 1_000_000.0,
        level_premium=rng.integers(2, 10, n) * 10_000.0,
        term_months=np.full(n, 120),
        state=rng.integers(0, 3, n),
    )
    asmp = _asmp(waiver_rate=0.03, state_model=three)
    assert np.allclose(measure(mps, asmp).bel[:, 0], value(mps, asmp).bel)


# ---------------------------------------------------------------------------
# Deprecation of "waiver_inception" -> "waiver_incidence" (T)
# ---------------------------------------------------------------------------
#
# The project standardised rate names on the actuarial term ``incidence``
# (a per-unit-time event rate). The legacy ``inception`` spelling is still
# accepted in two places -- the Assumptions field and the Transition rate
# name -- with a ``DeprecationWarning`` so existing user code keeps working
# until a future major version drops the alias.


def test_assumptions_waiver_inception_alias_deprecated():
    """Setting the deprecated ``waiver_inception_annual`` warns and routes
    to ``waiver_incidence_annual``."""
    rate = lambda s, a, d: np.full(d.shape, 0.05)
    with pytest.warns(DeprecationWarning,
                      match="waiver_inception_annual is deprecated"):
        asmp = Assumptions(
            mortality_annual=lambda s, a, d: np.full(d.shape, 0.01),
            lapse_annual=lambda s, a, d: np.full(d.shape, 0.02),
            discount_annual=0.03,
            expense_acquisition=0.0,
            expense_maintenance_annual=0.0,
            expense_inflation=0.0,
            ra_confidence=0.5,
            mortality_cv=0.0,
            waiver_inception_annual=rate,
        )
    # The alias is routed to the canonical field and the legacy field
    # is cleared so downstream code never has to look at both.
    assert asmp.waiver_incidence_annual is rate
    assert asmp.waiver_inception_annual is None


def test_assumptions_both_waiver_spellings_raise():
    """Setting both ``waiver_incidence_annual`` and the deprecated
    ``waiver_inception_annual`` is an error -- the caller has to pick one.
    """
    rate = lambda s, a, d: np.full(d.shape, 0.05)
    with pytest.raises(ValueError, match="not both"):
        Assumptions(
            mortality_annual=lambda s, a, d: np.full(d.shape, 0.01),
            lapse_annual=lambda s, a, d: np.full(d.shape, 0.02),
            discount_annual=0.03,
            expense_acquisition=0.0,
            expense_maintenance_annual=0.0,
            expense_inflation=0.0,
            ra_confidence=0.5,
            mortality_cv=0.0,
            waiver_incidence_annual=rate,
            waiver_inception_annual=rate,
        )


def test_transition_waiver_inception_rate_name_deprecated():
    """A StateModel that names its transition with the legacy
    ``"waiver_inception"`` still compiles, but compile_state_model emits
    a DeprecationWarning at lookup time.
    """
    model = StateModel(states=(
        State("active", premium=True, transitions=(
            Transition("mortality"),
            Transition("waiver_inception", to="waiver"),
            Transition("lapse"),
        )),
        State("waiver", premium=False, transitions=(
            Transition("mortality"),
        )),
    ), seating=(0, 1, 1))
    rates = {
        "mortality": np.array([[0.01]]),
        "waiver_incidence": np.array([[0.05]]),
        "lapse": np.array([[0.02]]),
    }
    with pytest.warns(DeprecationWarning,
                      match="rate name 'waiver_inception' is deprecated"):
        out = compile_state_model(model, rates)
    # The compile output for the deprecated spelling matches the canonical
    # spelling exactly.
    canonical = StateModel(states=(
        State("active", premium=True, transitions=(
            Transition("mortality"),
            Transition("waiver_incidence", to="waiver"),
            Transition("lapse"),
        )),
        State("waiver", premium=False, transitions=(
            Transition("mortality"),
        )),
    ), seating=(0, 1, 1))
    expected = compile_state_model(canonical, rates)
    for a, b in zip(out, expected):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b


# ---------------------------------------------------------------------------
# WAIVER_MODEL implicit fallback deprecation (U+W)
# ---------------------------------------------------------------------------
#
# When Assumptions.state_model is None but the contract still implies a
# multi-state path (waiver_incidence_annual set, or model_points.state has
# non-zero codes), the engine falls back to WAIVER_MODEL. The fallback is
# kept for backward compatibility but emits a DeprecationWarning -- a
# future major version will require state_model to be set explicitly.


def test_implicit_waiver_model_fallback_deprecated():
    """value() warns when it has to default to WAIVER_MODEL because
    waiver_incidence_annual is set but state_model is None.
    """
    asmp = Assumptions(
        mortality_annual=lambda s, a, d: np.full(d.shape, 0.001),
        lapse_annual=lambda s, a, d: np.full(d.shape, 0.005),
        waiver_incidence_annual=lambda s, a, d: np.full(d.shape, 0.003),
        discount_annual=0.03,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )
    mp = ModelPoints(
        issue_age=np.array([45], dtype=np.int64),
        death_benefit=np.array([10_000_000.0]),
        level_premium=np.array([50_000.0]),
        term_months=np.array([24], dtype=np.int64),
    )
    with pytest.warns(DeprecationWarning,
                      match="WAIVER_MODEL"):
        value(mp, asmp)


def test_explicit_state_model_silences_fallback_warning():
    """Setting ``state_model=STATE_MODELS['WAIVER']`` removes the implicit
    fallback warning while preserving the same result.
    """
    import warnings
    from fastcashflow import STATE_MODELS
    asmp_no = Assumptions(
        mortality_annual=lambda s, a, d: np.full(d.shape, 0.001),
        lapse_annual=lambda s, a, d: np.full(d.shape, 0.005),
        waiver_incidence_annual=lambda s, a, d: np.full(d.shape, 0.003),
        discount_annual=0.03,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
        state_model=STATE_MODELS["WAIVER"],
    )
    mp = ModelPoints(
        issue_age=np.array([45], dtype=np.int64),
        death_benefit=np.array([10_000_000.0]),
        level_premium=np.array([50_000.0]),
        term_months=np.array([24], dtype=np.int64),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # No DeprecationWarning raised here -- state_model is explicit.
        v_explicit = value(mp, asmp_no)
    # And the explicit-WAIVER result matches the implicit-fallback result.
    asmp_implicit = Assumptions(
        mortality_annual=lambda s, a, d: np.full(d.shape, 0.001),
        lapse_annual=lambda s, a, d: np.full(d.shape, 0.005),
        waiver_incidence_annual=lambda s, a, d: np.full(d.shape, 0.003),
        discount_annual=0.03,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        v_implicit = value(mp, asmp_implicit)
    assert np.allclose(v_explicit.bel, v_implicit.bel)
