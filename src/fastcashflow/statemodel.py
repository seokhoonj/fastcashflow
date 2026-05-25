"""The in-force state machine -- a product's states and transitions as data.

Phase (b) generalises the in-force projection from a single survival track to
an N-state Markov occupancy model. In-force is an occupancy vector ``occ`` over
a small set of transient states; each month a transition matrix advances it,
``occ[t+1] = occ[t] @ P[t]``. The kernels -- ``projection._project_kernel``,
``engine._value_kernel`` and the CUDA kernel -- run that recursion on a flat
edge list and are state-machine-agnostic: they carry no hardcoded state set.

This module is the product-facing layer. A :class:`StateModel` declares the
states, their transitions and which states pay premium or a benefit, all as
data. States can *be* data -- rather than a per-product DSL -- because the
occupancy recursion treats every state identically; there is no per-state
engine logic. (Coverage ``type``, by contrast, needs per-type logic and so
stays a fixed vocabulary.)

:func:`compile_state_model` turns a :class:`StateModel` plus the evaluated
assumption rates into the flat edge arrays the kernels consume.

The transition probabilities follow the standard ordered multiple-decrement
model. A state's transitions are applied IN ORDER as competing decrements:
transition ``i`` fires, among the entrants to the state, with the dependent
probability ``rate_i * prod_{j<i}(1 - rate_j)`` -- it acts on the survivors of
every earlier transition. The residual ``prod_j(1 - rate_j)`` stays in the
state. A transition either moves occupancy to another transient state (waiver
inception: active -> waiver; recovery: disabled -> active) or removes it from
the in-force set entirely (death, lapse). The fulfilment cash flows reflect
the contract's actual terms at the measurement date (IFRS 17 Sec. 33-34).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class CompiledStateModel:
    """Flat edge-tensor view of a compiled :class:`StateModel`.

    Returned by both :func:`compile_state_model` (Markov) and
    :func:`compile_state_model_with_duration` (semi-Markov). The two
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
    premium_state, benefit_state
        ``(n_states,)`` bool flags.
    state_duration_max
        ``None`` for Markov; ``(n_states,)`` int with the effective cohort
        count per state for semi-Markov.
    """
    edge_from: IntArray
    edge_to: IntArray
    edge_prob: FloatArray
    edge_lump_sum: np.ndarray
    n_states: int
    premium_state: np.ndarray
    benefit_state: np.ndarray
    state_duration_max: IntArray | None = None


@dataclass(frozen=True, slots=True)
class Transition:
    """One transition out of a state.

    ``rate`` names an assumption rate -- ``"mortality"``, ``"lapse"``,
    ``"waiver_incidence"`` and so on -- evaluated by the engine and supplied
    to :func:`compile_state_model`. ``to`` is the destination state's name
    when the transition moves occupancy to another transient state (waiver
    inception, recovery, reincidence), or ``None`` when it removes occupancy
    from the in-force set entirely (death, lapse).

    ``lump_sum`` flags a transition that pays a one-off benefit when it
    fires -- the ``ModelPoints.disability_benefit`` amount times the
    transitioning occupancy. It applies only to a transition with a
    destination; death and diagnosis lump sums stay on the coverage list.

    ``duration_dependent`` flags a semi-Markov transition: the rate depends
    on the **sojourn time** in the source state (time since entering it),
    not just on the policy duration. The source state must have
    ``duration_max > 0`` -- the engine tracks per-cohort occupancy there.
    The rate function for a duration-dependent transition takes a fourth
    argument ``state_duration`` (months in source state).
    """

    rate: str
    to: str | None = None
    lump_sum: bool = False
    duration_dependent: bool = False


@dataclass(frozen=True, slots=True)
class State:
    """One transient state of the in-force model.

    ``premium`` flags a premium-paying state -- the level and single premium
    accrue on the occupancy of the states so flagged. ``benefit`` flags a
    benefit-paying state -- the ``ModelPoints.disability_income`` amount is
    paid each month its occupancy is held (disability income on a disabled
    state). ``transitions`` are the transitions out of the state, held in
    application order: the competing-decrement convention (see the module
    docstring) applies each in turn to the survivors of the previous.

    ``duration_max`` switches the state to a **semi-Markov** model. When set
    to ``D > 0``, the engine tracks ``D`` monthly cohorts of in-force in
    this state (cohort 0 entered this month, cohort 1 entered last month,
    and cohort ``D - 1`` absorbs everyone who has been here ``D - 1`` months
    or longer). Transitions with ``duration_dependent=True`` then receive a
    cohort index and may carry different rates per cohort -- the natural
    way to express recovery, reincidence, exclusion (면책) periods, and
    other duration-since-entry effects. The default ``0`` keeps the state
    Markov (a single cohort, identical to the pre-Phase-(c) behaviour).
    """

    name: str
    premium: bool = False
    benefit: bool = False
    transitions: tuple[Transition, ...] = ()
    duration_max: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "duration_max", int(self.duration_max))
        if self.duration_max < 0:
            raise ValueError(
                f"state {self.name!r}: duration_max must be non-negative, "
                f"got {self.duration_max}"
            )


@dataclass(frozen=True, slots=True)
class StateModel:
    """A product's in-force state machine, declared as data.

    ``states`` are the transient states; position fixes the kernel state
    index, and state 0 is the issue state. ``seating`` maps a model point's
    input contract state -- the ``ModelPoints.state`` code (``STATE_ACTIVE``,
    ``STATE_WAIVER``, ``STATE_PAID_UP``) -- to the index of the state its
    in-force is seated on at the valuation date: ``seating[code]`` is that
    index. It defaults to seating every model point on state 0.

    The occupancy recursion treats every state identically, so an arbitrary
    StateModel runs on the existing kernels with no per-product code -- see
    the module docstring and :func:`compile_state_model`.
    """

    states: tuple[State, ...]
    seating: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        states = tuple(self.states)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "seating", tuple(int(s) for s in self.seating))
        if not states:
            raise ValueError("a StateModel needs at least one state")
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
                if tr.lump_sum and tr.to is None:
                    raise ValueError(
                        f"state {s.name!r} has a lump-sum transition with no "
                        f"destination; a lump sum attaches to a transition"
                    )
                if tr.duration_dependent and s.duration_max <= 0:
                    raise ValueError(
                        f"state {s.name!r} has a duration_dependent "
                        f"transition {tr.rate!r} but its duration_max is 0; "
                        f"set duration_max > 0 to track cohorts"
                    )
        if any(not 0 <= i < len(states) for i in self.seating):
            raise ValueError(
                f"seating index out of range for a {len(states)}-state model"
            )

    @property
    def n_states(self) -> int:
        """Number of transient states."""
        return len(self.states)


# The default in-force model -- two transient states. ``active`` pays premium
# and is subject to mortality, waiver inception and lapse; ``waiver`` (premium
# waived on a triggering event) keeps the coverage in force, pays no premium
# and is subject to mortality alone -- it does not lapse. The waiver-inception
# transition moves active in-force onto the waiver state. ``seating`` seats
# STATE_ACTIVE (code 0) on the active state and both STATE_WAIVER (1) and
# STATE_PAID_UP (2) on the waiver state: a paid-up contract and a waiver
# contract have identical cash flows, differing only in the cause premiums
# ceased.
WAIVER_MODEL = StateModel(
    states=(
        State("active", premium=True, transitions=(
            Transition("mortality"),
            Transition("waiver_incidence", to="waiver"),
            Transition("lapse"),
        )),
        State("waiver", premium=False, transitions=(
            Transition("mortality"),
        )),
    ),
    seating=(0, 1, 1),
)


# Named registry of bundled StateModels. A non-programmer actuary can pick a
# topology by name -- in the ``segments`` sheet's ``state_model`` column, in
# Python via ``STATE_MODELS["WAIVER"]``, or anywhere else a string label is a
# natural input. Additions land here as fixed-vocabulary entries (the same
# pattern as the coverage types -- see [[phase5-coverage-design]] in the
# project memory); users with a topology outside the registry still build
# their own ``StateModel`` in code.
STATE_MODELS: dict[str, StateModel] = {
    "WAIVER": WAIVER_MODEL,
}


def is_semi_markov(model: StateModel) -> bool:
    """Return True if any state in the model tracks duration cohorts.

    A semi-Markov state has ``duration_max > 0`` and tracks per-cohort
    occupancy; its outgoing transitions may then be ``duration_dependent``.
    A model with no such state is pure Markov and runs through the original
    :func:`compile_state_model` path.
    """
    return any(s.duration_max > 0 for s in model.states)


def resolve_state_model(assumptions) -> "StateModel":
    """Return the StateModel driving the projection for these assumptions.

    Uses the caller-supplied ``assumptions.state_model`` when set, and falls
    back to the bundled :data:`WAIVER_MODEL` -- the most common Korean
    protection topology, active / waiver / paid-up. Centralising the
    fallback keeps the engine and the projection layer from drifting.
    """
    return assumptions.state_model or WAIVER_MODEL


def compile_state_model(
    model: StateModel, rates: dict[str, FloatArray]
) -> CompiledStateModel:
    """Compile a StateModel and its rates into the kernel edge arrays.

    ``rates`` maps each rate name a transition references to its evaluated
    array; the arrays broadcast to a common grid shape -- the kernels index
    its trailing axes (per model point, or per sex / age / duration).

    Returns a :class:`CompiledStateModel` with ``state_duration_max=None``.

    Each state contributes one edge per transition with a transient
    destination -- carrying that transition's dependent probability -- plus
    one stay-in-state edge carrying the residual (see the module docstring). A
    transition that exits the in-force set contributes no edge: its occupancy
    simply leaves the recursion.

    This function is **Markov-only**: it raises ``ValueError`` if the model
    has any state with ``duration_max > 0``. Use
    :func:`compile_state_model_with_duration` for semi-Markov models.
    """
    if is_semi_markov(model):
        raise ValueError(
            "compile_state_model is Markov-only; use "
            "compile_state_model_with_duration for a model with "
            "duration-tracked states"
        )
    arrays = {name: np.asarray(arr, dtype=np.float64)
              for name, arr in rates.items()}
    if not arrays:
        raise ValueError("compile_state_model needs at least one rate array")
    grid = np.broadcast_shapes(*(a.shape for a in arrays.values()))
    index = {s.name: i for i, s in enumerate(model.states)}

    edge_from: list[int] = []
    edge_to: list[int] = []
    edge_prob: list[FloatArray] = []
    edge_lump: list[bool] = []
    for i, state in enumerate(model.states):
        # ``survive`` accumulates prod_{j}(1 - rate_j) across the transitions
        # applied so far; a leaving transition fires on those survivors.
        survive = np.ones(grid)
        for tr in state.transitions:
            try:
                rate = arrays[tr.rate]
            except KeyError:
                raise ValueError(
                    f"state {state.name!r} references rate {tr.rate!r}, "
                    f"which was not supplied to compile_state_model"
                ) from None
            if tr.to is not None:
                edge_from.append(i)
                edge_to.append(index[tr.to])
                edge_prob.append(survive * rate)
                edge_lump.append(tr.lump_sum)
            survive = survive * (1.0 - rate)
        edge_from.append(i)        # the residual stays in the state
        edge_to.append(i)
        edge_prob.append(survive)
        edge_lump.append(False)

    return CompiledStateModel(
        edge_from=np.array(edge_from, dtype=np.int64),
        edge_to=np.array(edge_to, dtype=np.int64),
        edge_prob=np.ascontiguousarray(np.stack(edge_prob)),
        edge_lump_sum=np.array(edge_lump, dtype=np.bool_),
        n_states=len(model.states),
        premium_state=np.array([s.premium for s in model.states], dtype=np.bool_),
        benefit_state=np.array([s.benefit for s in model.states], dtype=np.bool_),
        state_duration_max=None,
    )


def compile_state_model_with_duration(
    model: StateModel, rates: dict[str, FloatArray]
) -> CompiledStateModel:
    """Compile a semi-Markov StateModel into duration-aware kernel arrays.

    The cohort-aware counterpart of :func:`compile_state_model`. States
    declared with ``duration_max > 0`` are tracked by monthly cohort: the
    occupancy is a length-``duration_max`` vector indexed by sojourn time
    (months since entering the state, with the last cohort absorbing
    everyone who has been there at least ``duration_max - 1`` months).
    Transitions marked ``duration_dependent=True`` may then carry different
    rates per cohort.

    ``rates`` carries one array per rate name referenced by the model's
    transitions. Static (non-duration-dependent) rates broadcast to the
    ``grid`` shape -- the same convention as the Markov path. A duration-
    dependent rate has an extra trailing axis of length ``duration_max``
    for the source state (cohort axis).

    Returns a :class:`CompiledStateModel`:

    * ``edge_from`` / ``edge_to`` -- ``(n_edges,)`` state indices.
      ``edge_to == edge_from`` marks the residual stay edge (cohort
      advances by one).
    * ``edge_prob`` -- ``(n_edges, *grid, max_D)`` where ``max_D`` is the
      max ``duration_max`` across states (1 if no state is tracked). The
      tau axis carries the cohort index for the source state; for an edge
      out of an untracked state only ``tau = 0`` is meaningful.
    * ``edge_lump_sum`` -- ``(n_edges,)`` bool, the lump-sum transitions.
    * ``n_states`` -- the number of transient states.
    * ``premium_state`` / ``benefit_state`` -- ``(n_states,)`` bool.
    * ``state_duration_max`` -- ``(n_states,)`` int. The effective cohort
      count per state (``max(s.duration_max, 1)``). Untracked states have
      value 1; tracked states have the declared ``duration_max``.
    """
    arrays = {name: np.asarray(arr, dtype=np.float64)
              for name, arr in rates.items()}
    if not arrays:
        raise ValueError(
            "compile_state_model_with_duration needs at least one rate array"
        )
    index = {s.name: i for i, s in enumerate(model.states)}
    # Effective cohort count per state: untracked -> 1, tracked -> duration_max.
    state_duration_max = np.array(
        [max(s.duration_max, 1) for s in model.states], dtype=np.int64,
    )
    max_D = int(state_duration_max.max())

    # The grid (sex, age, year, ...) is the broadcast of the static-rate
    # shapes -- the duration-dependent rates share that grid with an extra
    # trailing cohort axis. Inferring it from the *static* rates avoids
    # baking the cohort axis into the grid.
    static_shapes = []
    for name, arr in arrays.items():
        any_dyn = any(tr.rate == name and tr.duration_dependent
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
                if tr.duration_dependent:
                    if arr.ndim != grid_ndim + 1:
                        raise ValueError(
                            f"rate {rate_name!r} is duration_dependent in "
                            f"state {src_state.name!r} but its array shape "
                            f"{arr.shape} has no cohort axis"
                        )
                    if arr.shape[-1] < src_state.duration_max:
                        raise ValueError(
                            f"rate {rate_name!r} cohort axis "
                            f"{arr.shape[-1]} shorter than state "
                            f"{src_state.name!r} duration_max "
                            f"{src_state.duration_max}"
                        )
                    return arr[..., tau]
                return arr
        # Should be unreachable (caller is iterating this state's transitions).
        return arr

    edge_from: list[int] = []
    edge_to: list[int] = []
    edge_prob_blocks: list[FloatArray] = []   # one (max_D, *grid) per edge
    edge_lump: list[bool] = []

    for i, state in enumerate(model.states):
        # Validate this state's transitions reference rates we have.
        for tr in state.transitions:
            if tr.rate not in arrays:
                raise ValueError(
                    f"state {state.name!r} references rate {tr.rate!r}, "
                    f"which was not supplied to "
                    f"compile_state_model_with_duration"
                )

        D = max(state.duration_max, 1)
        # Edges produced by this state are emitted in declaration order:
        # first the transient transitions (one per Transition with `to`),
        # then the residual stay edge. We collect their per-edge cohort
        # blocks here and pad to max_D below.
        out_edges_to: list[int] = []
        out_edges_lump: list[bool] = []
        out_edges_blocks: list[np.ndarray] = []   # each shape (D, *grid)

        # Compose one cohort at a time. For each cohort tau, run the
        # ordered competing-decrement composition using rate values that
        # depend on tau when the transition is duration_dependent.
        # We accumulate per-edge probabilities along the tau axis.
        per_edge_per_tau: list[list[np.ndarray]] = [
            [] for _ in range(len([tr for tr in state.transitions
                                   if tr.to is not None]))
        ]
        res_per_tau: list[np.ndarray] = []

        for tau in range(D):
            survive = np.ones(grid)
            transient_idx = 0
            for tr in state.transitions:
                r = rate_at(tr.rate, state, tau)
                if tr.to is not None:
                    prob = survive * r
                    per_edge_per_tau[transient_idx].append(prob)
                    transient_idx += 1
                survive = survive * (1.0 - r)
            res_per_tau.append(survive)

        # Stack tau slices per transient edge to (D, *grid), pad to
        # max_D (extra cohorts hold zeros; codegen won't touch them).
        for tr_idx, tr in enumerate([t for t in state.transitions
                                      if t.to is not None]):
            stacked = np.stack(per_edge_per_tau[tr_idx])  # (D, *grid)
            if D < max_D:
                pad = np.zeros((max_D - D,) + grid)
                stacked = np.concatenate([stacked, pad], axis=0)
            out_edges_to.append(index[tr.to])
            out_edges_lump.append(tr.lump_sum)
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
            edge_lump.append(lump)

    # Stack edges to (n_edges, max_D, *grid), then move max_D axis to the
    # *end* so the layout matches the Markov path's (..., edges) extension:
    # final shape is (n_edges, *grid, max_D), cohort innermost. Codegen
    # then transposes once more in engine.py to put edge index and cohort
    # last for cache-friendly inner-loop access.
    stacked = np.stack(edge_prob_blocks)  # (n_edges, max_D, *grid)
    # Move axis 1 (max_D) to the end.
    perm = (0,) + tuple(range(2, stacked.ndim)) + (1,)
    edge_prob = np.ascontiguousarray(np.transpose(stacked, perm))

    return CompiledStateModel(
        edge_from=np.array(edge_from, dtype=np.int64),
        edge_to=np.array(edge_to, dtype=np.int64),
        edge_prob=edge_prob,
        edge_lump_sum=np.array(edge_lump, dtype=np.bool_),
        n_states=len(model.states),
        premium_state=np.array([s.premium for s in model.states], dtype=np.bool_),
        benefit_state=np.array([s.benefit for s in model.states], dtype=np.bool_),
        state_duration_max=state_duration_max,
    )
