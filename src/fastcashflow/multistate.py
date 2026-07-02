"""The in-force state machine -- a product's states and transitions as data.

Phase (b) generalises the in-force projection from a single survival track to
an N-state Markov occupancy model. In-force is an occupancy vector ``occ`` over
a small set of transient states; each month a transition matrix advances it,
``occ[t+1] = occ[t] @ P[t]``. The kernels -- ``projection._project_kernel``,
the engine's codegen fast kernel and the CUDA kernel -- run that recursion on a flat
edge list and are state-machine-agnostic: they carry no hardcoded state set.

This module is the product-facing layer. A :class:`Model` declares the
states, their transitions and which states pay premium or a benefit, all as
data. States can *be* data -- rather than a per-product DSL -- because the
occupancy recursion treats every state identically; there is no per-state
engine logic. (Coverage ``type``, by contrast, needs per-type logic and so
stays a fixed vocabulary.)

:func:`compile_model` turns a :class:`Model` plus the evaluated
assumption rates into the flat edge arrays the kernels consume.

The transition probabilities follow the standard ordered multiple-decrement
model. A state's transitions are applied IN ORDER as competing decrements:
transition ``i`` fires, among the entrants to the state, with the dependent
probability ``rate_i * prod_{j<i}(1 - rate_j)`` -- it acts on the survivors of
every earlier transition. The residual ``prod_j(1 - rate_j)`` stays in the
state. A transition either moves occupancy to another transient state (waiver
inception: active -> waiver; recovery: disabled -> active) or removes it from
the in-force set entirely (death, lapse). The fulfilment cash flows reflect
the contract's actual terms at the measurement date (IFRS 17 paragraphs 33-34).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from fastcashflow._typing import BoolArray, FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class CompiledModel:
    """Flat edge-tensor view of a compiled :class:`Model`.

    Returned by both :func:`compile_model` (Markov) and
    :func:`compile_model_with_duration` (semi-Markov). The two
    paths share this shape; ``state_duration_max`` distinguishes them --
    ``None`` from the Markov compile, an ``(n_states,)`` int array from
    the semi-Markov compile (per-state cohort count, 1 for untracked
    states).

    Fields
    ------
    edge_from, edge_to
        ``(n_edges,)`` int arrays of source and destination state indices.
    edge_prob
        Transition probabilities. Markov: ``(n_edges, *grid)``.
        Semi-Markov: ``(n_edges, *grid, max_D)`` with cohort axis last.
    edge_lump_sum
        ``(n_edges,)`` bool, the lump-sum transitions.
    n_states
        The number of transient states.
    state_pays_premium, state_pays_benefit
        ``(n_states,)`` bool flags.
    state_duration_max
        ``None`` for Markov; ``(n_states,)`` int with the effective cohort
        count per state for semi-Markov.
    periodic_benefit_term_months
        ``None`` for Markov; ``(n_states,)`` int with the per-state monthly
        benefit cap (``0`` = unbounded). Sojourn cohorts ``tau >= cap`` stop
        being paid while the lives stay in force.
    """
    edge_from: IntArray
    edge_to: IntArray
    edge_prob: FloatArray
    edge_lump_sum: BoolArray
    n_states: int
    state_pays_premium: BoolArray
    state_pays_benefit: BoolArray
    state_duration_max: IntArray | None = None
    periodic_benefit_term_months: IntArray | None = None
    # Per-state exact in-force death-exit probability -- ``survive x mortality``,
    # where ``survive`` is the product of ``(1 - rate)`` over the transitions
    # listed before mortality in the state. The deaths reporter multiplies it by
    # the occupancy, so the death count respects the within-month competing-risk
    # order (it equals the raw rate exactly when mortality is the first
    # transition, which every bundled model declares). ``(n_states, *grid)``.
    state_death_exit: FloatArray | None = None
    # Per-state death-benefit multiplier (``State.death_benefit_factor``),
    # ``(n_states,)`` float, all-ones default. Occupancy-weighted into the
    # aggregate death claim: ``claim = (sum_s occ[s]*factor[s]) * claim_rate``.
    state_death_benefit_factor: FloatArray | None = None
    # Per-state deterministic transition (the <=1 Transition with
    # after_sojourn_months > 0), all ``(n_states,)``, semi-Markov only / ``None``
    # for Markov. ``state_det_at`` = the sojourn month K at which it fires (0 =
    # none); ``state_det_to`` = destination state index, or -1 for ``to=None``
    # (leave the in-force set); ``state_det_lump`` = its ``pays_lump_sum`` flag.
    # A cohort advancing into sojourn >= K is routed prob-1 to the destination
    # (or dropped), instead of advancing along the residual stay edge.
    state_det_at: IntArray | None = None
    state_det_to: IntArray | None = None
    state_det_lump: BoolArray | None = None
    # Per-state calendar-keyed deterministic transition (the <=1 Transition with
    # ``at_premium_term=True``), ``(n_states,)`` int. ``state_premium_term_to[s]``
    # = destination state index for state ``s``'s at-premium_term transition,
    # or -1 if it has none. Unlike state_det_at (sojourn-keyed, semi-Markov),
    # this fires at the model point's own ``premium_term_months`` -- a per-MP
    # calendar trigger the Markov kernel applies directly (active -> paid-up when
    # the premium-paying period ends). No probability, no sojourn cohorts.
    state_premium_term_to: IntArray | None = None


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One transition in a model's transition structure (its transMat analogue).

    Enumerated by :attr:`Model.transitions` -- the model-level list of every
    transition the topology declares, in a stable order, independent of any
    particular book's rates (a transition is listed if declared, even where a
    book sets its rate to zero). It is the descriptor axis the per-transition
    sum at risk reads for its ``n_transition`` order and labels.

    ``from_state`` / ``to_state`` are transient-state indices; ``to_state`` is
    ``None`` for an absorbing exit (death / lapse leave the in-force set).
    ``kind`` is ``"death"``, ``"lapse"`` or ``"transfer"`` (an inter-state edge).
    ``from_name`` / ``to_name`` label them for display (``to_name`` is the
    destination state name, or ``"death"`` / ``"lapse"``).
    """
    from_state: int
    to_state: "int | None"
    kind: str
    from_name: str
    to_name: str


@dataclass(frozen=True, slots=True)
class Transition:
    """One transition out of a state.

    ``rate`` names an assumption rate -- ``"mortality"``, ``"lapse"``,
    ``"waiver_incidence"`` and so on -- evaluated by the engine and supplied
    to :func:`compile_model`. ``to`` is the destination state's name
    when the transition moves occupancy to another transient state (waiver
    inception, recovery, reincidence), or ``None`` when it removes occupancy
    from the in-force set entirely (death, lapse).

    ``pays_lump_sum`` flags a transition that pays a one-off benefit when it
    fires -- the ``ModelPoints.disability_benefit`` amount times the
    transitioning occupancy. It applies only to a transition with a
    destination; death and diagnosis lump sums stay on the coverage list.

    ``sojourn_dependent`` flags a semi-Markov transition: the rate depends
    on the **sojourn time** in the source state (time since entering it),
    not just on the policy duration. The source state must have
    ``sojourn_tracking_months > 0`` -- the engine tracks per-cohort occupancy there.
    The rate function for a duration-dependent transition takes a fourth
    argument ``state_duration`` (months in source state).

    ``after_sojourn_months`` makes the transition **deterministic** (probability
    one) at a fixed sojourn: when a cohort's sojourn in the source state reaches
    this many months, all of it moves to ``to`` (or leaves the in-force set when
    ``to is None``). It carries no ``rate`` (the move is certain). This expresses
    a cover that ends after a fixed term (``to=None``), or a guaranteed
    conversion to another state (``to="active"``). At most one deterministic
    transition per state.
    """

    rate: str | None = None
    to: str | None = None
    pays_lump_sum: bool = False
    sojourn_dependent: bool = False
    after_sojourn_months: int = 0
    at_premium_term: bool = False

    def __post_init__(self) -> None:
        det_sojourn = int(self.after_sojourn_months) > 0
        det_cal = bool(self.at_premium_term)
        det = det_sojourn or det_cal
        object.__setattr__(self, "after_sojourn_months", int(self.after_sojourn_months))
        if self.after_sojourn_months < 0:
            raise ValueError(
                "Transition.after_sojourn_months must be non-negative, got "
                f"{self.after_sojourn_months}"
            )
        if det_sojourn and det_cal:
            raise ValueError(
                "a Transition has at most one deterministic trigger; set either "
                "after_sojourn_months (sojourn-keyed) or at_premium_term "
                "(calendar-keyed at the model point's premium_term), not both"
            )
        if det and self.rate is not None:
            raise ValueError(
                "a deterministic transition (after_sojourn_months > 0 or "
                "at_premium_term) carries no rate; it fires with probability one"
            )
        if not det and self.rate is None:
            raise ValueError(
                "a Transition needs a rate, after_sojourn_months > 0, or "
                "at_premium_term"
            )
        if det and self.sojourn_dependent:
            raise ValueError(
                "a deterministic transition is already keyed (sojourn or "
                "premium_term); do not also set sojourn_dependent"
            )


@dataclass(frozen=True, slots=True)
class State:
    """One transient state of the in-force model.

    ``pays_premium`` flags a premium-paying state -- the level and single premium
    accrue on the occupancy of the states so flagged. ``pays_periodic_benefit`` flags a
    benefit-paying state -- the ``ModelPoints.disability_income`` amount is
    paid each month its occupancy is held (disability income on a disabled
    state). ``transitions`` are the transitions out of the state, held in
    application order: the competing-decrement convention (see the module
    docstring) applies each in turn to the survivors of the previous.

    ``sojourn_tracking_months`` switches the state to a **semi-Markov** model. When set
    to ``D > 0``, the engine tracks ``D`` monthly cohorts of in-force in
    this state (cohort 0 entered this month, cohort 1 entered last month,
    and cohort ``D - 1`` absorbs everyone who has been here ``D - 1`` months
    or longer). Transitions with ``sojourn_dependent=True`` then receive a
    cohort index and may carry different rates per cohort -- the natural
    way to express recovery, reincidence, exclusion periods, and
    other duration-since-entry effects. The default ``0`` keeps the state
    Markov (a single cohort, identical to the pre-Phase-(c) behaviour).

    ``mortality_rate`` routes this state's in-force death decrement to a
    named rate (default ``"mortality"``, the global decrement). A
    post-diagnosis state can carry an elevated death rate by naming a
    different rate, supplied via ``Basis.state_mortality_annual``.

    ``periodic_benefit_term_months`` caps how many months a ``benefit`` state pays
    (``0`` = unbounded); see the field comment in ``__post_init__``.

    ``death_benefit_factor`` scales the death-coverage benefit paid for the
    lives residing in this state (default ``1.0`` = no change). The aggregate
    death claim is occupancy-weighted: ``claim = (sum_s occ[s]*factor[s]) *
    claim_rate``. It multiplies the benefit AMOUNT, not the decrement, so the
    death count is unchanged. A post-diagnosis state paying a richer death
    benefit (e.g. 2x after a cancer diagnosis) sets ``death_benefit_factor=2.0``.
    Supported on the full path only (``measure(full=True)``); the fast path and
    the VFA path reject a non-default factor.

    A cover that ends after a fixed sojourn (or a guaranteed conversion to
    another state at a fixed sojourn) is a deterministic
    ``Transition(after_sojourn_months=K, to=...)`` -- ``to=None`` ends the
    cover, ``to="active"`` converts. It is distinct from
    ``periodic_benefit_term_months`` (which stops the payment but keeps the
    lives in force): a guaranteed-payout state that pays a fixed term then lapses
    sets ``periodic_benefit_term_months`` (pay window) and a
    ``Transition(after_sojourn_months=K, to=None)`` (cover end) together.
    """

    name: str
    pays_premium: bool = False
    pays_periodic_benefit: bool = False
    transitions: tuple[Transition, ...] = ()
    sojourn_tracking_months: int = 0
    periodic_benefit_term_months: int = 0
    mortality_rate: str = "mortality"
    death_benefit_factor: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "sojourn_tracking_months", int(self.sojourn_tracking_months))
        if self.sojourn_tracking_months < 0:
            raise ValueError(
                f"state {self.name!r}: sojourn_tracking_months must be non-negative, "
                f"got {self.sojourn_tracking_months}"
            )
        # ``periodic_benefit_term_months`` caps how many months a benefit state pays --
        # the monthly ``disability_income`` stops once a cohort's sojourn
        # reaches the cap, while the lives stay in force (a guaranteed-payout
        # LTC / dementia annuity: pay 36 months, then keep cover with no
        # payment). ``0`` means unbounded (the historical behaviour).
        cap = int(self.periodic_benefit_term_months)
        object.__setattr__(self, "periodic_benefit_term_months", cap)
        if cap < 0:
            raise ValueError(
                f"state {self.name!r}: periodic_benefit_term_months must be "
                f"non-negative, got {cap}"
            )
        if cap > 0 and not self.pays_periodic_benefit:
            raise ValueError(
                f"state {self.name!r}: periodic_benefit_term_months > 0 requires "
                f"pays_periodic_benefit=True (a cap on a non-paying state has no effect)"
            )
        # ``death_benefit_factor`` multiplies the death-coverage benefit amount
        # for lives in this state. Non-negative; defaults 1.0 (no change).
        factor = float(self.death_benefit_factor)
        object.__setattr__(self, "death_benefit_factor", factor)
        if factor < 0.0:
            raise ValueError(
                f"state {self.name!r}: death_benefit_factor must be "
                f"non-negative, got {factor}"
            )
        # Deterministic transitions: at most one per state (two prob-1 edges
        # from one cohort are ill-defined), counting BOTH the sojourn-keyed
        # (after_sojourn_months > 0) and the calendar-keyed (at_premium_term)
        # forms; and it must clear the pay cap (pay the cap, then move / exit).
        det_trs = [tr for tr in self.transitions
                   if tr.after_sojourn_months > 0 or tr.at_premium_term]
        if len(det_trs) > 1:
            raise ValueError(
                f"state {self.name!r}: at most one deterministic transition "
                f"(after_sojourn_months > 0 or at_premium_term) per state, "
                f"got {len(det_trs)}"
            )
        det_months = [tr.after_sojourn_months for tr in self.transitions
                      if tr.after_sojourn_months > 0]
        det = max(det_months) if det_months else 0
        if det > 0 and cap > 0 and det < cap:
            raise ValueError(
                f"state {self.name!r}: a deterministic transition's "
                f"after_sojourn_months ({det}) must be >= "
                f"periodic_benefit_term_months ({cap}); pay the cap, then move."
            )
        # Sojourn tracking: the engine tracks one monthly cohort per sojourn
        # month, the last cohort absorbing everyone at that sojourn or beyond.
        # A deterministic boundary (periodic_benefit_term_months / a deterministic
        # transition's after_sojourn_months) needs one guard cohort past it --
        # otherwise the absorbing cohort sits at the boundary and capped / exiting
        # lives re-accumulate and never stop. Auto-derive that ``+1`` so the caller
        # never meets the off-by-one; an explicit sojourn_tracking_months only sets
        # a *longer* tail (for a sojourn-dependent rate whose tail outruns the
        # boundary) and must still clear the boundary.
        boundary = max(cap, det)
        if boundary > 0:
            if self.sojourn_tracking_months == 0:
                object.__setattr__(self, "sojourn_tracking_months", boundary + 1)
            elif self.sojourn_tracking_months <= boundary:
                raise ValueError(
                    f"state {self.name!r}: sojourn_tracking_months "
                    f"({self.sojourn_tracking_months}) must exceed the deterministic "
                    f"sojourn boundary ({boundary} = max(periodic_benefit_term_months, "
                    f"after_sojourn_months)); the absorbing cohort must sit strictly "
                    f"past it. Omit sojourn_tracking_months to auto-derive {boundary + 1}."
                )


@dataclass(frozen=True, slots=True)
class Model:
    """A product's in-force state machine, declared as data.

    ``states`` are the transient states; position fixes the kernel state
    index, and state 0 is the issue state. ``seating`` maps a model point's
    input contract state -- the ``ModelPoints.state`` code (``STATE_ACTIVE``,
    ``STATE_WAIVER``, ``STATE_PAIDUP``) -- to the index of the state its
    in-force is seated on at the valuation date: ``seating[code]`` is that
    index. It defaults to seating every model point on state 0.

    The occupancy recursion treats every state identically, so an arbitrary
    Model runs on the existing kernels with no per-product code -- see
    the module docstring and :func:`compile_model`.
    """

    states: tuple[State, ...]
    seating: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        states = tuple(self.states)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "seating", tuple(int(s) for s in self.seating))
        if not states:
            raise ValueError("a Model needs at least one state")
        names = {s.name for s in states}
        if len(names) != len(states):
            raise ValueError("state names must be unique")
        for s in states:
            for tr in s.transitions:
                if tr.to is not None and tr.to not in names:
                    raise ValueError(
                        f"state {s.name!r} has a transition to an unknown "
                        f"state {tr.to!r}"
                    )
                if (tr.pays_lump_sum and tr.to is None
                        and tr.after_sojourn_months == 0):
                    raise ValueError(
                        f"state {s.name!r} has a rate-driven lump-sum transition "
                        f"with no destination; a rate-driven lump attaches to a "
                        f"transition with a destination (a deterministic "
                        f"after_sojourn_months exit may pay a lump on its way out)"
                    )
                if tr.sojourn_dependent and s.sojourn_tracking_months <= 0:
                    raise ValueError(
                        f"state {s.name!r} has a sojourn_dependent "
                        f"transition {tr.rate!r} but its sojourn_tracking_months is 0; "
                        f"set sojourn_tracking_months > 0 to track cohorts"
                    )
        if any(not 0 <= i < len(states) for i in self.seating):
            raise ValueError(
                f"seating index out of range for a {len(states)}-state model"
            )

    @property
    def n_states(self) -> int:
        """Number of transient states."""
        return len(self.states)

    @property
    def transitions(self) -> tuple[TransitionRecord, ...]:
        """The model's transition structure as an ordered tuple of records.

        The transMat analogue: every transition the topology declares, in a
        stable order, derived from the state / transition declaration alone
        (a transition is listed even if a particular book sets its rate to
        zero). The order groups all death exits, then all lapse exits, then
        all inter-state transfers -- each group in state-index order, and
        within a state in declaration order:

        * a **death** record for each state that declares a mortality decrement
          (a transition with ``rate == "mortality"``);
        * a **lapse** record for each state that declares a lapse decrement (a
          rate-driven ``to=None`` exit that is not the mortality one);
        * a **transfer** record for each declared inter-state transition
          (``to`` set to a different state), the ``ModelPoints`` occupancy
          moving from ``from_state`` to ``to_state``.

        The per-transition sum at risk reads this for its axis order and its
        descriptors; it emits a row for the transitions a given book actually
        exercises (so a book with a zero decrement carries no row for it),
        keeping this list a structural superset of that book's axis.
        """
        index = {s.name: i for i, s in enumerate(self.states)}
        records: list[TransitionRecord] = []
        for i, s in enumerate(self.states):
            if any(tr.rate == "mortality" for tr in s.transitions):
                records.append(TransitionRecord(i, None, "death", s.name, "death"))
        for i, s in enumerate(self.states):
            if any(tr.to is None and tr.rate is not None and tr.rate != "mortality"
                   for tr in s.transitions):
                records.append(TransitionRecord(i, None, "lapse", s.name, "lapse"))
        for i, s in enumerate(self.states):
            for tr in s.transitions:
                if tr.to is not None and index[tr.to] != i:
                    records.append(
                        TransitionRecord(i, index[tr.to], "transfer", s.name, tr.to))
        return tuple(records)

    @classmethod
    def from_preset(cls, name: str) -> "Model":
        """Return the bundled model registered under ``name``.

        A non-programmer actuary can pick a topology by name -- in the
        ``segments`` sheet's ``state_model`` column, or in Python via
        ``Model.from_preset("ACTIVE_WAIVER")``. The preset key lists the
        transient states in state-index order (``ACTIVE_WAIVER`` = active +
        waiver; ``ACTIVE_WAIVER_PAIDUP`` = active + waiver + paid-up). Users
        with a topology outside the registry build their own :class:`Model`.
        """
        try:
            return _PRESETS[name]
        except KeyError:
            raise ValueError(
                f"unknown state model preset {name!r} "
                f"(known: {', '.join(cls.presets())})"
            ) from None

    @classmethod
    def presets(cls) -> tuple[str, ...]:
        """The available :meth:`from_preset` names, in sorted order."""
        return tuple(sorted(_PRESETS))


# The default in-force model -- two transient states. ``active`` pays premium
# and is subject to mortality, waiver inception and lapse; ``waiver`` (premium
# waived on a triggering event) keeps the coverage in force, pays no premium
# and is subject to mortality and its OWN lapse ``lapse_waiver``. The
# waiver-inception transition moves active in-force onto the waiver state.
# ``lapse_waiver`` (Basis.lapse_waiver_annual) defaults to a 0 rate, so the
# waiver state does NOT lapse unless a rate is set -- the pure-waiver default
# (a waived contract holds free-of-premium cover, so anti-selection keeps it
# in force). Set a (typically low) ``lapse_waiver_annual`` to model the
# residual waived-state surrender. ``seating`` seats STATE_ACTIVE (code 0) on
# the active state and both STATE_WAIVER (1) and STATE_PAIDUP (2) on the waiver
# state: a paid-up contract and a waiver contract have identical cash flows,
# differing only in the cause premiums ceased.
ACTIVE_WAIVER_MODEL = Model(
    states=(
        State("active", pays_premium=True, transitions=(
            Transition("mortality"),
            Transition("waiver_incidence", to="waiver"),
            Transition("lapse"),
        )),
        State("waiver", pays_premium=False, transitions=(
            Transition("mortality"),
            Transition("lapse_waiver"),
        )),
    ),
    seating=(0, 1, 1),
)


# Three-state variant -- active / waiver / paid-up as *separate* states.
# Unlike ACTIVE_WAIVER_MODEL (which seats paid-up onto the waiver state, giving the
# two identical cash flows), this model keeps paid-up distinct so it can carry
# its own lapse: the paid-up state references the ``lapse_paidup`` rate
# (Basis.lapse_paidup_annual, falling back to lapse_annual). The Korean
# post-payment lapse jump is the motivating case -- a contract that
# has finished paying premium typically surrenders at a different rate than a
# premium-paying active. Paid-up still has no premium and is exposed to
# mortality + its own lapse; there is no waiver-inception out of paid-up (you
# cannot waive a premium you no longer pay). ``seating`` seats STATE_ACTIVE on
# active (0), STATE_WAIVER on waiver (1) and STATE_PAIDUP on paid-up (2).
# The active state carries a calendar-keyed ``at_premium_term`` transition to
# paid-up: when a model point's premium-paying period ends (month
# ``premium_term_months``), its active occupancy moves prob-1 to paid-up -- a
# deterministic per-MP relabel (not a rate, not a sojourn cohort). So a
# new-business projection seasons active -> paid-up at premium_term, and an
# in-force valuation of an already-paid-up cohort can still seat directly on
# paid-up (STATE_PAIDUP). Waiver stays a separate state through premium_term
# (a premium-waived contract is not the same population as a normal paid-up
# one), so it keeps ``lapse_waiver``, not ``lapse_paidup``.
ACTIVE_WAIVER_PAIDUP_MODEL = Model(
    states=(
        State("active", pays_premium=True, transitions=(
            Transition("mortality"),
            Transition("waiver_incidence", to="waiver"),
            Transition("lapse"),
            Transition(at_premium_term=True, to="paidup"),
        )),
        State("waiver", pays_premium=False, transitions=(
            Transition("mortality"),
            Transition("lapse_waiver"),
        )),
        State("paidup", pays_premium=False, transitions=(
            Transition("mortality"),
            Transition("lapse_paidup"),
        )),
    ),
    seating=(0, 1, 2),
)


# Named registry of bundled models, reached through :meth:`Model.from_preset`.
# A non-programmer actuary can pick a topology by name -- in the ``segments``
# sheet's ``state_model`` column, or in Python via
# ``Model.from_preset("ACTIVE_WAIVER")``. Additions land here as
# fixed-vocabulary entries -- the same convention as the coverage
# CalculationMethods; users with a topology outside the registry still build
# their own ``Model`` in code. The key lists the transient states in
# state-index order. It is module-private and reached only through the
# ``from_preset`` / ``presets`` factory, so user / plugin code cannot swap a
# bundled topology process-wide (which would change every later segment that
# resolves the name).
_PRESETS: Mapping[str, Model] = MappingProxyType({
    "ACTIVE_WAIVER": ACTIVE_WAIVER_MODEL,
    "ACTIVE_WAIVER_PAIDUP": ACTIVE_WAIVER_PAIDUP_MODEL,
})


def model_references_rate(model: Model, rate_name: str) -> bool:
    """Return True if any transition in the model references ``rate_name``.

    The engine builds the rate dict it hands to :func:`compile_model`
    from the rates the resolved model actually references, so an optional
    rate (e.g. ``lapse_paidup``) is built only when the topology in play
    uses it.
    """
    return any(tr.rate == rate_name
               for s in model.states for tr in s.transitions)


def is_semi_markov(model: Model) -> bool:
    """Return True if any state in the model tracks duration cohorts.

    A semi-Markov state has ``sojourn_tracking_months > 0`` and tracks per-cohort
    occupancy; its outgoing transitions may then be ``sojourn_dependent``.
    A model with no such state is pure Markov and runs through the original
    :func:`compile_model` path.
    """
    return any(s.sojourn_tracking_months > 0 for s in model.states)


def resolve_model(basis) -> "Model":
    """Return the Model driving the projection for these basis.

    Uses the caller-supplied ``basis.state_model`` when set, and falls
    back to the bundled :data:`ACTIVE_WAIVER_MODEL` -- the most common Korean
    protection topology, active / waiver. Centralising the fallback keeps the
    engine and the projection layer from drifting.
    """
    return basis.state_model or ACTIVE_WAIVER_MODEL


def needs_state_machine(model_points, basis) -> bool:
    """True when the N-state occupancy kernel is needed, not the scalar fast path.

    The scalar fused path carries in-force as a single number; it cannot
    represent a state machine. The N-state path is required when the basis
    declares a state model, carries a waiver decrement, or any model point is
    seated outside the active state.

    Extracted (no behaviour change) from the fast-path branch in
    ``engine._measure_fast`` so the routing decision is one named, testable
    predicate -- the seed of the planned portfolio-orchestrator classifier.
    """
    return (basis.state_model is not None
            or basis.waiver_incidence_annual is not None
            or bool(np.any(model_points.state)))


def compile_model(
    model: Model, rates: dict[str, FloatArray]
) -> CompiledModel:
    """Compile a Model and its rates into the kernel edge arrays.

    ``rates`` maps each rate name a transition references to its evaluated
    array; the arrays broadcast to a common grid shape -- the kernels index
    its trailing axes (per model point, or per sex / age / duration).

    Returns a :class:`CompiledModel` with ``state_duration_max=None``.

    Each state contributes one edge per transition with a transient
    destination -- carrying that transition's dependent probability -- plus
    one stay-in-state edge carrying the residual (see the module docstring). A
    transition that exits the in-force set contributes no edge: its occupancy
    simply leaves the recursion.

    This function is **Markov-only**: it raises ``ValueError`` if the model
    has any state with ``sojourn_tracking_months > 0``. Use
    :func:`compile_model_with_duration` for semi-Markov models.
    """
    if is_semi_markov(model):
        raise ValueError(
            "compile_model is Markov-only; use "
            "compile_model_with_duration for a model with "
            "duration-tracked states"
        )
    for s in model.states:
        if any(tr.after_sojourn_months > 0 for tr in s.transitions):
            raise ValueError(
                f"state {s.name!r}: a deterministic transition "
                f"(after_sojourn_months > 0) needs sojourn tracking and is "
                f"semi-Markov only"
            )
    arrays = {name: np.asarray(arr, dtype=np.float64)
              for name, arr in rates.items()}
    if not arrays:
        raise ValueError("compile_model needs at least one rate array")
    grid = np.broadcast_shapes(*(a.shape for a in arrays.values()))
    index = {s.name: i for i, s in enumerate(model.states)}

    edge_from: list[int] = []
    edge_to: list[int] = []
    edge_prob: list[FloatArray] = []
    edge_lump_sum: list[bool] = []
    death_exit_rows: list[FloatArray] = []
    premium_term_to: list[int] = []   # per-state calendar (at_premium_term) dest
    for i, state in enumerate(model.states):
        # ``survive`` accumulates prod_{j}(1 - rate_j) across the transitions
        # applied so far; a leaving transition fires on those survivors.
        survive = np.ones(grid)
        death_exit = np.zeros(grid)   # exact death exit for the deaths reporter
        pt_to = -1                    # this state's at-premium_term dest (-1 none)
        for tr in state.transitions:
            # The calendar-keyed deterministic transition carries no rate and
            # does not reduce ``survive`` (it is applied separately by the
            # kernel at the model point's premium_term); record its destination
            # (-2 = exit the in-force set, to=None) and skip the rate edge.
            if tr.at_premium_term:
                pt_to = index[tr.to] if tr.to is not None else -2
                continue
            # A state's mortality decrement is routed to its own rate name
            # (State.mortality_rate, default "mortality") so a post-diagnosis
            # state can carry an elevated mortality without re-declaring the
            # transition. Any other rate name passes through unchanged.
            rname = (state.mortality_rate
                     if tr.rate == "mortality" else tr.rate)
            try:
                rate = arrays[rname]
            except KeyError:
                raise ValueError(
                    f"state {state.name!r} references rate {rname!r}, "
                    f"which was not supplied to compile_model"
                ) from None
            if tr.rate == "mortality":
                # The death exit fires on whoever survived the earlier
                # transitions this month -- ``survive`` here is that product.
                death_exit = survive * rate
            if tr.to is not None:
                edge_from.append(i)
                edge_to.append(index[tr.to])
                edge_prob.append(survive * rate)
                edge_lump_sum.append(tr.pays_lump_sum)
            survive = survive * (1.0 - rate)
        edge_from.append(i)        # the residual stays in the state
        edge_to.append(i)
        edge_prob.append(survive)
        edge_lump_sum.append(False)
        death_exit_rows.append(death_exit)
        premium_term_to.append(pt_to)

    return CompiledModel(
        edge_from=np.array(edge_from, dtype=np.int64),
        edge_to=np.array(edge_to, dtype=np.int64),
        edge_prob=np.ascontiguousarray(np.stack(edge_prob)),
        edge_lump_sum=np.array(edge_lump_sum, dtype=np.bool_),
        n_states=len(model.states),
        state_pays_premium=np.array([s.pays_premium for s in model.states], dtype=np.bool_),
        state_pays_benefit=np.array([s.pays_periodic_benefit for s in model.states], dtype=np.bool_),
        state_duration_max=None,
        state_death_exit=np.ascontiguousarray(np.stack(death_exit_rows)),
        state_death_benefit_factor=np.array(
            [s.death_benefit_factor for s in model.states], dtype=np.float64),
        state_det_at=None, state_det_to=None, state_det_lump=None,
        state_premium_term_to=np.array(premium_term_to, dtype=np.int64),
    )


def compile_model_with_duration(
    model: Model, rates: dict[str, FloatArray]
) -> CompiledModel:
    """Compile a semi-Markov Model into duration-aware kernel arrays.

    The cohort-aware counterpart of :func:`compile_model`. States
    declared with ``sojourn_tracking_months > 0`` are tracked by monthly cohort: the
    occupancy is a length-``sojourn_tracking_months`` vector indexed by sojourn time
    (months since entering the state, with the last cohort absorbing
    everyone who has been there at least ``sojourn_tracking_months - 1`` months).
    Transitions marked ``sojourn_dependent=True`` may then carry different
    rates per cohort.

    ``rates`` carries one array per rate name referenced by the model's
    transitions. Static (non-duration-dependent) rates broadcast to the
    ``grid`` shape -- the same convention as the Markov path. A duration-
    dependent rate has an extra trailing axis of length ``sojourn_tracking_months``
    for the source state (cohort axis).

    Returns a :class:`CompiledModel`:

    * ``edge_from`` / ``edge_to`` -- ``(n_edges,)`` state indices.
      ``edge_to == edge_from`` marks the residual stay edge (cohort
      advances by one).
    * ``edge_prob`` -- ``(n_edges, *grid, max_D)`` where ``max_D`` is the
      max ``sojourn_tracking_months`` across states (1 if no state is tracked). The
      tau axis carries the cohort index for the source state; for an edge
      out of an untracked state only ``tau = 0`` is meaningful.
    * ``edge_lump_sum`` -- ``(n_edges,)`` bool, the lump-sum transitions.
    * ``n_states`` -- the number of transient states.
    * ``state_pays_premium`` / ``state_pays_benefit`` -- ``(n_states,)`` bool.
    * ``state_duration_max`` -- ``(n_states,)`` int. The effective cohort
      count per state (``max(s.sojourn_tracking_months, 1)``). Untracked states have
      value 1; tracked states have the declared ``sojourn_tracking_months``.
    """
    if any(tr.at_premium_term for s in model.states for tr in s.transitions):
        raise NotImplementedError(
            "at_premium_term (calendar-keyed) transitions are supported on the "
            "Markov projection path only; this model also has sojourn-tracked "
            "states, which route to the semi-Markov path. Combining a "
            "premium_term calendar transition with sojourn tracking is a later "
            "step."
        )
    arrays = {name: np.asarray(arr, dtype=np.float64)
              for name, arr in rates.items()}
    if not arrays:
        raise ValueError(
            "compile_model_with_duration needs at least one rate array"
        )
    index = {s.name: i for i, s in enumerate(model.states)}
    # Effective cohort count per state: untracked -> 1, tracked -> sojourn_tracking_months.
    state_duration_max = np.array(
        [max(s.sojourn_tracking_months, 1) for s in model.states], dtype=np.int64,
    )
    max_D = int(state_duration_max.max())

    # The grid (sex, age, year, ...) is the broadcast of the static-rate
    # shapes -- the duration-dependent rates share that grid with an extra
    # trailing cohort axis. Inferring it from the *static* rates avoids
    # baking the cohort axis into the grid.
    static_shapes = []
    for name, arr in arrays.items():
        any_dyn = any(tr.rate == name and tr.sojourn_dependent
                       for s in model.states for tr in s.transitions)
        if any_dyn:
            static_shapes.append(arr.shape[:-1])
        else:
            static_shapes.append(arr.shape)
    grid = np.broadcast_shapes(*static_shapes)
    grid_ndim = len(grid)

    def rate_at(rate_name: str, src_state: State, tau: int) -> FloatArray:
        """Evaluate a rate at cohort ``tau`` of the source state.

        For a non-duration-dependent transition the array is returned as
        is. For a duration-dependent one the tau-th slice of the trailing
        cohort axis is taken.
        """
        arr = arrays[rate_name]
        # Determine whether *this* transition reads the rate as dynamic;
        # the same rate name may be referenced by both kinds across states.
        # We pass the source state to look up the transition's flag.
        for tr in src_state.transitions:
            if tr.rate == rate_name:
                if tr.sojourn_dependent:
                    if arr.ndim != grid_ndim + 1:
                        raise ValueError(
                            f"rate {rate_name!r} is sojourn_dependent in "
                            f"state {src_state.name!r} but its array shape "
                            f"{arr.shape} has no cohort axis"
                        )
                    if arr.shape[-1] < src_state.sojourn_tracking_months:
                        raise ValueError(
                            f"rate {rate_name!r} cohort axis "
                            f"{arr.shape[-1]} shorter than state "
                            f"{src_state.name!r} sojourn_tracking_months "
                            f"{src_state.sojourn_tracking_months}"
                        )
                    return arr[..., tau]
                return arr
        # Reached when ``rate_name`` is supplied for this state but is not the
        # rate of any of its transitions (e.g. a state's own mortality routed
        # by name without a matching transition row); the un-indexed array is
        # the correct static (non-cohort) fallback. NOT an invariant break --
        # a review flagged this as "unreachable", but several models reach it.
        return arr

    edge_from: list[int] = []
    edge_to: list[int] = []
    edge_prob_blocks: list[FloatArray] = []   # one (max_D, *grid) per edge
    edge_lump_sum: list[bool] = []
    death_exit_rows: list[FloatArray] = []    # per-state exact death exit (cohort 0)
    det_at_list: list[int] = []               # per-state deterministic transition (<=1)
    det_to_list: list[int] = []
    det_lump_list: list[bool] = []

    for i, state in enumerate(model.states):
        # Split the deterministic transition (prob-1 at a fixed sojourn) out of
        # the rate-driven ones: it carries no rate and is NOT a competing-
        # decrement edge -- it rides the residual gate in the kernel via the
        # state_det_* scalars, exactly as the old exit attribute did.
        rate_trs = [tr for tr in state.transitions if tr.after_sojourn_months == 0]
        det = next((tr for tr in state.transitions
                    if tr.after_sojourn_months > 0), None)
        det_at_list.append(det.after_sojourn_months if det else 0)
        det_to_list.append(index[det.to] if (det and det.to is not None) else -1)
        det_lump_list.append(bool(det.pays_lump_sum) if det else False)
        # Validate this state's rate transitions reference rates we have. A
        # "mortality" transition routes to the state's own mortality rate
        # name (State.mortality_rate), so validate the effective name.
        for tr in rate_trs:
            rname = (state.mortality_rate
                     if tr.rate == "mortality" else tr.rate)
            if rname not in arrays:
                raise ValueError(
                    f"state {state.name!r} references rate {rname!r}, "
                    f"which was not supplied to "
                    f"compile_model_with_duration"
                )

        D = max(state.sojourn_tracking_months, 1)
        # Edges produced by this state are emitted in declaration order:
        # first the transient transitions (one per Transition with `to`),
        # then the residual stay edge. We collect their per-edge cohort
        # blocks here and pad to max_D below.
        out_edges_to: list[int] = []
        out_edges_lump: list[bool] = []
        out_edges_blocks: list[np.ndarray] = []   # each shape (D, *grid)

        # Compose one cohort at a time. For each cohort tau, run the
        # ordered competing-decrement composition using rate values that
        # depend on tau when the transition is sojourn_dependent.
        # We accumulate per-edge probabilities along the tau axis.
        per_edge_per_tau: list[list[np.ndarray]] = [
            [] for _ in range(len([tr for tr in rate_trs
                                   if tr.to is not None]))
        ]
        res_per_tau: list[np.ndarray] = []

        death_exit = np.zeros(grid)   # exact death exit (cohort 0) for the reporter
        for tau in range(D):
            survive = np.ones(grid)
            transient_idx = 0
            for tr in rate_trs:
                rname = (state.mortality_rate
                         if tr.rate == "mortality" else tr.rate)
                r = rate_at(rname, state, tau)
                if tr.rate == "mortality" and tau == 0:
                    # The deaths reporter sums occupancy over cohorts and reads a
                    # single per-state rate, so the death exit is taken at cohort
                    # 0; exact whenever the pre-mortality transitions are
                    # cohort-independent (every bundled model) or mortality is
                    # the state's first transition.
                    death_exit = survive * r
                if tr.to is not None:
                    prob = survive * r
                    per_edge_per_tau[transient_idx].append(prob)
                    transient_idx += 1
                survive = survive * (1.0 - r)
            res_per_tau.append(survive)
        death_exit_rows.append(death_exit)

        # Stack tau slices per transient edge to (D, *grid), pad to
        # max_D (extra cohorts hold zeros; codegen won't touch them).
        for tr_idx, tr in enumerate([t for t in rate_trs
                                      if t.to is not None]):
            stacked = np.stack(per_edge_per_tau[tr_idx])  # (D, *grid)
            if D < max_D:
                pad = np.zeros((max_D - D,) + grid)
                stacked = np.concatenate([stacked, pad], axis=0)
            out_edges_to.append(index[tr.to])
            out_edges_lump.append(tr.pays_lump_sum)
            out_edges_blocks.append(stacked)

        residual = np.stack(res_per_tau)
        if D < max_D:
            pad = np.zeros((max_D - D,) + grid)
            residual = np.concatenate([residual, pad], axis=0)
        out_edges_to.append(i)
        out_edges_lump.append(False)
        out_edges_blocks.append(residual)

        # All edges out of state i carry source index i.
        for block, dst, lump in zip(out_edges_blocks, out_edges_to,
                                     out_edges_lump):
            edge_from.append(i)
            edge_to.append(dst)
            edge_prob_blocks.append(block)
            edge_lump_sum.append(lump)

    # Stack edges to (n_edges, max_D, *grid), then move max_D axis to the
    # *end* so the layout matches the Markov path's (..., edges) extension:
    # final shape is (n_edges, *grid, max_D), cohort innermost. Codegen
    # then transposes once more in engine.py to put edge index and cohort
    # last for cache-friendly inner-loop access.
    stacked = np.stack(edge_prob_blocks)  # (n_edges, max_D, *grid)
    # Move axis 1 (max_D) to the end.
    perm = (0,) + tuple(range(2, stacked.ndim)) + (1,)
    edge_prob = np.ascontiguousarray(np.transpose(stacked, perm))

    return CompiledModel(
        edge_from=np.array(edge_from, dtype=np.int64),
        edge_to=np.array(edge_to, dtype=np.int64),
        edge_prob=edge_prob,
        edge_lump_sum=np.array(edge_lump_sum, dtype=np.bool_),
        n_states=len(model.states),
        state_pays_premium=np.array([s.pays_premium for s in model.states], dtype=np.bool_),
        state_pays_benefit=np.array([s.pays_periodic_benefit for s in model.states], dtype=np.bool_),
        state_duration_max=state_duration_max,
        periodic_benefit_term_months=np.array(
            [s.periodic_benefit_term_months for s in model.states], dtype=np.int64),
        state_death_exit=np.ascontiguousarray(np.stack(death_exit_rows)),
        state_death_benefit_factor=np.array(
            [s.death_benefit_factor for s in model.states], dtype=np.float64),
        state_det_at=np.array(det_at_list, dtype=np.int64),
        state_det_to=np.array(det_to_list, dtype=np.int64),
        state_det_lump=np.array(det_lump_list, dtype=np.bool_),
    )
