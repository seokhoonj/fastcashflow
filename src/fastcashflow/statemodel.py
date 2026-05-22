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
class Transition:
    """One transition out of a state.

    ``rate`` names an assumption rate -- ``"mortality"``, ``"lapse"`` or
    ``"waiver_inception"`` -- evaluated by the engine and supplied to
    :func:`compile_state_model`. ``to`` is the destination state's name when
    the transition moves occupancy to another transient state (waiver
    inception, recovery), or ``None`` when it removes occupancy from the
    in-force set entirely (death, lapse).

    ``lump_sum`` flags a transition that pays a one-off benefit when it
    fires -- the ``ModelPoints.disability_benefit`` amount times the
    transitioning occupancy. It applies only to a transition with a
    destination; death and diagnosis lump sums stay on the coverage list.
    """

    rate: str
    to: str | None = None
    lump_sum: bool = False


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
    """

    name: str
    premium: bool = False
    benefit: bool = False
    transitions: tuple[Transition, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", tuple(self.transitions))


@dataclass(frozen=True, slots=True)
class StateModel:
    """A product's in-force state machine, declared as data.

    ``states`` are the transient states; position fixes the kernel state
    index, and state 0 is the issue state. ``seating`` maps a model point's
    input contract state -- the ``ModelPoints.state`` code (``STATE_ACTIVE``,
    ``STATE_WAIVER``, ``STATE_PAIDUP``) -- to the index of the state its
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
# STATE_PAIDUP (2) on the waiver state: a paid-up contract and a waiver
# contract have identical cash flows, differing only in the cause premiums
# ceased.
WAIVER_MODEL = StateModel(
    states=(
        State("active", premium=True, transitions=(
            Transition("mortality"),
            Transition("waiver_inception", to="waiver"),
            Transition("lapse"),
        )),
        State("waiver", premium=False, transitions=(
            Transition("mortality"),
        )),
    ),
    seating=(0, 1, 1),
)


def compile_state_model(
    model: StateModel, rates: dict[str, FloatArray]
) -> tuple[IntArray, IntArray, FloatArray, np.ndarray, int,
           np.ndarray, np.ndarray]:
    """Compile a StateModel and its rates into the kernel edge arrays.

    ``rates`` maps each rate name a transition references to its evaluated
    array; the arrays broadcast to a common grid shape -- the kernels index
    its trailing axes (per model point, or per sex / age / duration).

    Returns ``(edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
    premium_state, benefit_state)``:

    * ``edge_from`` / ``edge_to`` -- ``(n_edges,)`` state indices.
    * ``edge_prob`` -- ``(n_edges, *grid)`` transition probabilities.
    * ``edge_lump_sum`` -- ``(n_edges,)`` bool, the lump-sum transitions.
    * ``n_states`` -- the number of transient states.
    * ``premium_state`` -- ``(n_states,)`` bool, the premium-paying states.
    * ``benefit_state`` -- ``(n_states,)`` bool, the benefit-paying states.

    Each state contributes one edge per transition with a transient
    destination -- carrying that transition's dependent probability -- plus
    one stay-in-state edge carrying the residual (see the module docstring). A
    transition that exits the in-force set contributes no edge: its occupancy
    simply leaves the recursion.
    """
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

    return (
        np.array(edge_from, dtype=np.int64),
        np.array(edge_to, dtype=np.int64),
        np.ascontiguousarray(np.stack(edge_prob)),
        np.array(edge_lump, dtype=np.bool_),
        len(model.states),
        np.array([s.premium for s in model.states], dtype=np.bool_),
        np.array([s.benefit for s in model.states], dtype=np.bool_),
    )
