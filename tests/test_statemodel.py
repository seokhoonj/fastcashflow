"""The in-force state machine -- StateModel as product input.

Phase (b) Stage 2: a product declares its states, transitions and
premium-paying states as data. The default active / waiver model is one
StateModel among many; these tests drive custom ones through both the fused
``value`` and the detailed ``measure`` path, with the figures derived by hand
from the multiple-decrement recursion.
"""
import numpy as np
import pytest

from fastcashflow import STATE_ACTIVE, STATE_PAIDUP, STATE_WAIVER, STATE_MODELS, Basis, ModelPoints, State, StateModel, Transition, CoverageRate, CalculationMethod
from fastcashflow.gmm import measure
from fastcashflow.statemodel import compile_state_model

from conftest import annual_from_monthly as _annual


def _basis(*, waiver_rate=0.0, lapse=0.02, q=0.01, state_model=None) -> Basis:
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
    return Basis(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, q_a),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, lapse_a),
        waiver_incidence_annual=waiver,
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
        state_model=state_model,
        coverages=(CoverageRate("DEATH", lambda sex, issue_age, duration: np.full(issue_age.shape, q_a)),),
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
    compiled = compile_state_model(STATE_MODELS["WAIVER"], rates)
    assert compiled.n_states == 2
    assert list(compiled.premium_state) == [True, False]
    assert list(compiled.benefit_state) == [False, False]   # waiver: no benefit
    assert not compiled.edge_lump_sum.any()                 # ... no lump sums

    prob = {(int(f), int(t)): float(compiled.edge_prob[i, 0, 0])
            for i, (f, t) in enumerate(
                zip(compiled.edge_from, compiled.edge_to))}
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
        compile_state_model(STATE_MODELS["WAIVER"], {"mortality": np.array([[0.01]]),
                                           "waiver_incidence": np.array([[0.0]])})


# ---------------------------------------------------------------------------
# StateModel validation
# ---------------------------------------------------------------------------

def test_unknown_destination_state_rejected():
    """A decrement to a state that does not exist is rejected at build time."""
    with pytest.raises(ValueError, match="unknown state"):
        StateModel(states=(
            State("active", pays_premium=True,
                  transitions=(Transition("waiver_incidence", to="ghost"),)),
        ))


def test_duplicate_state_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        StateModel(states=(State("a"), State("a")))


def test_seating_out_of_range_rejected():
    with pytest.raises(ValueError, match="seating"):
        StateModel(states=(State("a"),), seating=(0, 1))


def test_state_models_registry_is_read_only():
    """STATE_MODELS is exposed as a read-only mapping -- a stray assignment
    from user / plugin code cannot silently swap the bundled topology
    process-wide. Lookup still works the same."""
    assert STATE_MODELS["WAIVER"] is not None
    with pytest.raises(TypeError):
        STATE_MODELS["WAIVER"] = StateModel(states=(State("x"),))    # type: ignore[index]


def test_markov_can_reference_ci_incidence_annual():
    """A custom Markov topology that wires a transition to ci_incidence
    works through both measure() and measure(). The Markov rate dict now
    threads ci_incidence_annual when the assumption is set -- before, the
    same topology would fail at compile_state_model with a "rate not
    supplied" ValueError, surprising anyone porting a Markov dx model
    from the semi-Markov branch."""
    healthy_to_diag = StateModel(
        states=(
            State("healthy", pays_premium=True, transitions=(
                Transition("mortality"),
                Transition("ci_incidence", to="diagnosed", pays_lump_sum=True),
                Transition("lapse"),
            )),
            State("diagnosed", pays_premium=False, transitions=(
                Transition("mortality"),
            )),
        ),
        seating=(0, 1, 1),
    )
    q_a = _annual(0.001)
    basis = Basis(
        mortality_annual=lambda s, a, d: np.full(d.shape, q_a),
        lapse_annual=lambda s, a, d: np.full(d.shape, _annual(0.005)),
        ci_incidence_annual=lambda s, a, d: np.full(d.shape, _annual(0.003)),
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
        state_model=healthy_to_diag,
        coverages=(CoverageRate("DEATH", lambda s, a, d: np.full(d.shape, q_a)),),
    )
    mp = ModelPoints.single(issue_age=40, benefits={0: 1_000_000.0},
                            premium=0.0, term_months=12)
    val = measure(mp, basis, full=False)
    m = measure(mp, basis)
    assert np.isclose(m.bel_path[0, 0], val.bel[0])


# ---------------------------------------------------------------------------
# Custom state machines through the engine
# ---------------------------------------------------------------------------

def test_explicit_waiver_model_matches_default():
    """A StateModel rebuilt to the same shape as the built-in default
    reproduces it exactly -- the default path is just one StateModel."""
    rebuilt = StateModel(
        states=(
            State("active", pays_premium=True, transitions=(
                Transition("mortality"),
                Transition("waiver_incidence", to="waiver"),
                Transition("lapse"),
            )),
            State("waiver", transitions=(Transition("mortality"),)),
        ),
        seating=(0, 1, 1),
    )
    kw = dict(issue_age=45, benefits={0: 50_000_000.0},
              premium=30_000.0, term_months=120)
    for state in (STATE_ACTIVE, STATE_WAIVER, STATE_PAIDUP):
        mp = ModelPoints.single(**kw, state=state)
        default = measure(mp, _basis(waiver_rate=0.03), full=False)
        custom = measure(mp, _basis(waiver_rate=0.03, state_model=rebuilt), full=False)
        assert np.isclose(default.bel[0], custom.bel[0])


def test_single_state_no_lapse_hand_calculation():
    """A one-state model -- mortality only, no lapse, no waiver. With a flat
    1% mortality the in-force is [1, 0.99, 0.99^2]; every figure by hand."""
    no_lapse = StateModel(states=(
        State("active", pays_premium=True, transitions=(Transition("mortality"),)),
    ))
    death_benefit = 1_000_000.0
    premium = 12_000.0
    mp = ModelPoints.single(issue_age=40, benefits={0: death_benefit},
                            premium=premium, term_months=3)
    basis = _basis(state_model=no_lapse)

    inforce = [1.0, 0.99, 0.99 ** 2]
    pv_claims = sum(i * 0.01 * death_benefit for i in inforce)
    pv_premiums = sum(i * premium for i in inforce)
    bel = pv_claims - pv_premiums

    val = measure(mp, basis, full=False)
    assert np.isclose(val.bel[0], bel)
    assert np.isclose(measure(mp, basis).bel_path[0, 0], bel)
    assert np.allclose(measure(mp, basis).cashflows.inforce[0], inforce)


def test_decrement_order_matters():
    """The decrement order is data: applying lapse before waiver inception
    feeds the waiver state a fraction (1 - lapse) smaller than the default
    waiver-before-lapse order, a different BEL -- derived by hand."""
    lapse_first = StateModel(
        states=(
            State("active", pays_premium=True, transitions=(
                Transition("mortality"),
                Transition("lapse"),
                Transition("waiver_incidence", to="waiver"),
            )),
            State("waiver", transitions=(Transition("mortality"),)),
        ),
        seating=(0, 1, 1),
    )
    death_benefit = 1_000_000.0
    premium = 12_000.0
    mp = ModelPoints.single(issue_age=40, benefits={0: death_benefit},
                            premium=premium, term_months=2)
    basis = _basis(waiver_rate=0.05, lapse=0.02, state_model=lapse_first)

    # t=0: act=1, wav=0.
    #   act[1] = 1 * 0.99 * 0.98 * 0.95 = 0.92169  (death, lapse, then waiver)
    #   wav[1] = 1 * 0.99 * 0.98 * 0.05 = 0.04851
    act1 = 0.99 * 0.98 * 0.95
    wav1 = 0.99 * 0.98 * 0.05
    inforce = [1.0, act1 + wav1]
    pv_claims = sum(i * 0.01 * death_benefit for i in inforce)
    pv_premiums = (1.0 + act1) * premium       # premium on the active track
    bel = pv_claims - pv_premiums

    assert np.isclose(measure(mp, basis, full=False).bel[0], bel)
    assert np.allclose(measure(mp, basis).cashflows.inforce[0], inforce)
    # The default waiver-before-lapse order gives a distinct figure.
    default = measure(mp, _basis(waiver_rate=0.05, lapse=0.02), full=False).bel[0]
    assert not np.isclose(default, bel)


def test_three_state_model_runs():
    """A three-state model (active, waiver, paid-up kept as a distinct state)
    runs through both kernels: n_states = 3 flows through the occupancy
    recursion, and the extra paid-up state -- mortality only, no premium --
    values a paid-up contract exactly as the two-state default does."""
    three = StateModel(
        states=(
            State("active", pays_premium=True, transitions=(
                Transition("mortality"),
                Transition("waiver_incidence", to="waiver"),
                Transition("lapse"),
            )),
            State("waiver", transitions=(Transition("mortality"),)),
            State("paidup", transitions=(Transition("mortality"),)),
        ),
        seating=(0, 1, 2),       # active / waiver / paid-up each own a state
    )
    assert three.n_states == 3
    kw = dict(issue_age=42, benefits={0: 80_000_000.0},
              premium=40_000.0, term_months=180)

    # A paid-up contract: identical to the default, which seats paid-up on
    # the waiver state -- both are mortality-only, premium-free.
    paidup = ModelPoints.single(**kw, state=STATE_PAIDUP)
    base = measure(paidup, _basis(waiver_rate=0.03), full=False)
    custom = measure(paidup, _basis(waiver_rate=0.03, state_model=three), full=False)
    for field in ("bel", "ra", "csm", "loss_component"):
        assert np.isclose(getattr(base, field)[0], getattr(custom, field)[0])

    # An active contract is unaffected by the unreachable paid-up state.
    active = ModelPoints.single(**kw, state=STATE_ACTIVE)
    assert np.isclose(measure(active, _basis(waiver_rate=0.03), full=False).bel[0],
                      measure(active, _basis(waiver_rate=0.03, state_model=three), full=False).bel[0])


def test_paidup_state_uses_its_own_lapse():
    """STATE_MODELS["WAIVER_PAIDUP"] keeps paid-up a distinct state so it can
    carry its own lapse (Basis.lapse_paidup_annual). A paid-up-seated
    contract decrements by mortality + the paid-up lapse; with the paid-up
    lapse above the active lapse its in-force falls faster than the active
    track -- the Korean post-payment (납입후) lapse jump."""
    q = _annual(0.01)
    basis = Basis(
        mortality_annual=lambda s, a, d: np.full(a.shape, q),
        lapse_annual=lambda s, a, d: np.full(d.shape, _annual(0.02)),
        lapse_paidup_annual=lambda s, a, d: np.full(d.shape, _annual(0.10)),
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH",
                                lambda s, a, d: np.full(a.shape, q)),),
        state_model=STATE_MODELS["WAIVER_PAIDUP"],
    )
    kw = dict(issue_age=40, benefits={0: 100_000.0}, premium=0.0,
              term_months=3)
    paid = measure(ModelPoints.single(**kw, state=STATE_PAIDUP), basis)
    step = 0.99 * 0.90        # (1 - mortality)(1 - paid-up lapse)
    assert np.allclose(paid.cashflows.inforce[0, :3], [1.0, step, step ** 2])
    # falls faster than the active 2%-lapse track
    act = measure(ModelPoints.single(**kw, state=STATE_ACTIVE), basis)
    assert paid.cashflows.inforce[0, 1] < act.cashflows.inforce[0, 1]


def test_paidup_lapse_falls_back_to_lapse_annual():
    """With lapse_paidup_annual unset, the paid-up state's lapse_paidup rate
    falls back to lapse_annual -- the WAIVER_PAIDUP model still runs, the
    paid-up state just lapses at the ordinary rate."""
    basis = _basis(q=0.01, lapse=0.05,
                 state_model=STATE_MODELS["WAIVER_PAIDUP"])
    kw = dict(issue_age=40, benefits={0: 100_000.0}, premium=0.0,
              term_months=3)
    paid = measure(ModelPoints.single(**kw, state=STATE_PAIDUP), basis)
    step = 0.99 * 0.95        # falls back to the 5% active lapse
    assert np.allclose(paid.cashflows.inforce[0, :3], [1.0, step, step ** 2])


def test_measure_and_value_agree_under_custom_model():
    """The detailed and the fused path agree across a mixed-state portfolio
    valued on a custom three-state model."""
    three = StateModel(
        states=(
            State("active", pays_premium=True, transitions=(
                Transition("mortality"),
                Transition("waiver_incidence", to="waiver"),
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
        benefits={0: rng.integers(10, 80, n) * 1_000_000.0},
        premium=rng.integers(2, 10, n) * 10_000.0,
        term_months=np.full(n, 120),
        state=rng.integers(0, 3, n),
    )
    basis = _basis(waiver_rate=0.03, state_model=three)
    assert np.allclose(measure(mps, basis).bel_path[:, 0], measure(mps, basis, full=False).bel)



# ---------------------------------------------------------------------------
# Death count respects the within-month competing-risk order (P2 fix). The
# deaths reporter fires on the survivors of the transitions listed before
# mortality -- exact, not occ x raw rate. VFA reads `deaths` into its benefit
# split (deaths get the GMDB floor), so this is a latent BEL input there, not
# only a display figure.
# ---------------------------------------------------------------------------
def test_deaths_respect_within_month_competing_risk_order():
    from fastcashflow.projection import project_cashflows
    from fastcashflow.basis import annual_to_monthly
    death = lambda s, a, d: np.full(np.shape(a), 0.01)
    lapse = lambda s, a, d: np.full(np.shape(d), 0.05)

    def _project(model):
        basis = Basis(mortality_annual=death, lapse_annual=lapse,
                      discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
                      state_model=model, coverages=(CoverageRate("DEATH", death),))
        mp = ModelPoints(issue_age=np.array([40], dtype=np.int64),
                         benefits={0: np.array([0.0])}, premium=np.array([0.0]),
                         term_months=np.array([3], dtype=np.int64),
                         calculation_methods={"DEATH": CalculationMethod.DEATH})
        return project_cashflows(mp, basis)

    mq, lq = annual_to_monthly(0.01), annual_to_monthly(0.05)
    # mortality first (every bundled model) -> occ x raw rate
    first = StateModel(states=(State("active", pays_premium=True, transitions=(
        Transition("mortality"), Transition("lapse"))),), seating=(0,))
    # mortality after lapse -> occ x (1 - lapse) x rate (fires on lapse survivors)
    second = StateModel(states=(State("active", pays_premium=True, transitions=(
        Transition("lapse"), Transition("mortality"))),), seating=(0,))

    assert np.isclose(_project(first).deaths[0, 0], mq)                  # unchanged
    assert np.isclose(_project(second).deaths[0, 0], (1.0 - lq) * mq)   # exact, lower
    assert _project(second).deaths[0, 0] < mq                            # was raw mq before
