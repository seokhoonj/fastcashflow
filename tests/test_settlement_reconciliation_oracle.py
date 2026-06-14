"""Phase-0 refactor test net: the settlement reconciliation surface oracle.

The four settlement Movement / Reconciliation / Aggregate families and their
reconcile / write dispatch are exercised only incidentally inside the per-feature
settle test modules (test_gmm_settle*, test_vfa_settle*, ...). Nothing pins the
SHARED invariants the reporting-layer single-source (the `_LINE_META` line spine
+ to_frame + write_reconciliation) will rest on. This module is that pin -- it
must be green on today's code BEFORE any spine / serializer refactor.

Two oracles here:

* the line spine is one source -- ``set(_X_SETTLEMENT_LINES)`` equals the
  per-MP array (``FloatArray``) fields of the matching settlement Movement
  dataclass, in EVERY family. This is what lets the writers (and to_frame) be
  driven from the tuples; if a future edit adds a movement line but forgets the
  tuple (or vice versa) this fails at collection time.
* the display-negation SIGN convention is absolute, not merely round-trip
  consistent. ``reconcile`` stores release / reversed / paid / revenue lines
  NEGATED so opening + every row foots to closing; an independent sign oracle
  catches a uniform-flip that ``reconcile(aggregate)==reconcile(per-MP)`` would
  mask. (The sign + identity oracles that need a constructed movement are added
  alongside the per-model settle fixtures; this file pins the structural spine,
  which needs no projection.)
"""
import numpy as np

from fastcashflow.movement import (
    GMMSettlementMovement, PAASettlementMovement,
    ReinsuranceSettlementMovement, VFASettlementMovement,
    _GMM_SETTLEMENT_LINES, _PAA_SETTLEMENT_LINES,
    _REINSURANCE_SETTLEMENT_LINES, _VFA_SETTLEMENT_LINES,
)


# (lines tuple, movement class) for each of the four settlement families.
_FAMILIES = (
    ("gmm", _GMM_SETTLEMENT_LINES, GMMSettlementMovement),
    ("vfa", _VFA_SETTLEMENT_LINES, VFASettlementMovement),
    ("reinsurance", _REINSURANCE_SETTLEMENT_LINES, ReinsuranceSettlementMovement),
    ("paa", _PAA_SETTLEMENT_LINES, PAASettlementMovement),
)


def _float_array_fields(cls):
    """The per-MP array lines of a settlement Movement dataclass -- the fields
    annotated FloatArray (movement.py uses ``from __future__ import annotations``
    so the annotation is the string 'FloatArray'). The scalar / reference fields
    (period_months, lock_in_rate, model_points, measurement_basis, revenue_basis)
    carry their own rules and are NOT line-spine entries."""
    return {name for name, f in cls.__dataclass_fields__.items()
            if str(f.type) == "FloatArray"}


def test_settlement_lines_tuple_equals_movement_float_array_fields():
    """The line spine is one source: every _X_SETTLEMENT_LINES tuple equals the
    set of FloatArray fields on its Movement dataclass. This is the invariant
    that lets the writers and to_frame be driven from the tuples (refactor
    Delta 1); a drift between tuple and dataclass fails here, at collection."""
    for key, lines, cls in _FAMILIES:
        tuple_set = set(lines)
        field_set = _float_array_fields(cls)
        assert len(lines) == len(tuple_set), f"{key}: _SETTLEMENT_LINES has duplicates"
        missing = field_set - tuple_set
        extra = tuple_set - field_set
        assert not missing, f"{key}: FloatArray fields not in the lines tuple: {sorted(missing)}"
        assert not extra, f"{key}: lines-tuple entries that are not FloatArray fields: {sorted(extra)}"


def test_settlement_lines_are_ordered_and_nonempty():
    """Each family declares a non-empty, hashable, string line spine (the
    ordering is the canonical display / serialization order the writers and the
    _LINE_META registry will key on)."""
    for key, lines, _cls in _FAMILIES:
        assert lines, f"{key}: empty _SETTLEMENT_LINES"
        assert all(isinstance(n, str) for n in lines), f"{key}: non-str line name"
