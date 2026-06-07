"""``fcf.portfolio.measure`` -- the mixed-model portfolio orchestrator.

One heterogeneous portfolio (GMM / PAA / VFA contracts in a single file, routed
by a :class:`~fastcashflow.basis.BasisRouter`) measured in one call, returning a
:class:`PortfolioMeasurement` that keeps each model's native result separate -- a
BEL and an LRC are never summed into one array. Each contract is routed to its
segment's measurement model; that row partition is the orchestrator's core and
is validated as a construction invariant.

P-3 implemented the row partition and the GMM execution; P-4 adds the PAA and
VFA executors (each model's segments stitched into one native measurement). A
non-GMM row is never silently measured as GMM. The VFA stitch carries a per-MP
2-D ``discount_bom`` because segments discount at their own underlying-items
return, and the movement / grouping consumers handle that 2-D curve (grouping
keeps a group inside one curve).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace

import numpy as np

from fastcashflow._typing import IntArray
from fastcashflow._paa import (
    PAAMeasurement, measure_paa, _stitch_paa_measurements)
from fastcashflow._vfa import (
    VFAMeasurement, measure_vfa, _stitch_vfa_measurements)
from fastcashflow.basis import BasisRouter
from fastcashflow.engine import (
    GMMMeasurement, _factorise_segments, measure as _measure_gmm)
from fastcashflow.modelpoints import ModelPoints

#: The native measurement type each model slot must hold (the per-model
#: separation invariant: a paa slot can never carry a GMMMeasurement).
_SLOT_MEASUREMENT_TYPE = {
    "gmm": GMMMeasurement, "paa": PAAMeasurement, "vfa": VFAMeasurement}


def _measurement_rows(measurement) -> int:
    """Row count of a native measurement (GMM / VFA expose ``bel``, PAA ``lrc``)."""
    for attr in ("bel", "lrc"):
        value = getattr(measurement, attr, None)
        if value is not None:
            return len(value)
    raise TypeError(
        f"cannot determine the row count of a {type(measurement).__name__}")


@dataclass(frozen=True, slots=True)
class ModelMeasurement:
    """One measurement model's slice of a portfolio: the original row positions
    (``index``, sorted and unique) and the native result over exactly those rows.
    """

    index: IntArray
    measurement: "GMMMeasurement | PAAMeasurement | VFAMeasurement"

    def __post_init__(self):
        index = np.asarray(self.index, dtype=np.int64)
        if index.ndim != 1:
            raise ValueError("ModelMeasurement.index must be 1-D")
        if index.size and np.any(index[1:] <= index[:-1]):
            raise ValueError("ModelMeasurement.index must be sorted and unique")
        rows = _measurement_rows(self.measurement)
        if index.size != rows:
            raise ValueError(
                f"ModelMeasurement.index has {index.size} rows but the "
                f"measurement covers {rows}")
        object.__setattr__(self, "index", index)


@dataclass(frozen=True, slots=True)
class PortfolioMeasurement:
    """Result of :func:`measure`: one :class:`ModelMeasurement` per model present
    (``None`` when absent), keyed by model so a BEL and an LRC are never
    conflated. ``model_points`` is the full portfolio -- the grouping axes for
    downstream per-segment / GIC analysis (e.g. ``group(pm.gmm.measurement,
    by=["product"])``). The per-model indices must partition ``0..n_mp-1``
    exactly; this is checked at construction.
    """

    model_points: ModelPoints
    gmm: ModelMeasurement | None = None
    paa: ModelMeasurement | None = None
    vfa: ModelMeasurement | None = None

    def __post_init__(self):
        # Each slot must hold its own model's native measurement -- a paa slot
        # carrying a GMMMeasurement would defeat the per-model separation.
        for slot, expected in _SLOT_MEASUREMENT_TYPE.items():
            mm = getattr(self, slot)
            if mm is not None and not isinstance(mm.measurement, expected):
                raise TypeError(
                    f"PortfolioMeasurement.{slot} must hold a "
                    f"{expected.__name__}, got "
                    f"{type(mm.measurement).__name__}")
        present = [mm for mm in (self.gmm, self.paa, self.vfa) if mm is not None]
        n = self.model_points.n_mp
        covered = sum(mm.index.size for mm in present)
        if covered != n:
            raise ValueError(
                f"portfolio partition covers {covered} rows but the portfolio "
                f"has {n}")
        if present:
            allidx = np.sort(np.concatenate([mm.index for mm in present]))
            if not np.array_equal(allidx, np.arange(n, dtype=np.int64)):
                raise ValueError(
                    "portfolio partition is not a clean 0..n_mp-1 cover "
                    "(rows missing, duplicated, or out of range)")
        elif n != 0:
            raise ValueError(
                f"portfolio has {n} rows but carries no model measurement")


def _nfc_key(key) -> tuple:
    """NFC-normalise a segment key tuple -- mirrors ``_factorise_segments`` so a
    composed/decomposed Unicode mismatch never splits a key from its model."""
    parts = key if isinstance(key, tuple) else (key,)
    return tuple(unicodedata.normalize("NFC", str(p)) for p in parts)


def _partition_by_model(model_points: ModelPoints, router: BasisRouter):
    """Map each row to its segment's measurement model.

    Returns ``{"GMM": idx, "PAA": idx, "VFA": idx}`` (sorted int64; empty when a
    model is absent). Driven by the *rows actually present*, not the router's
    declared segments -- an unused PAA segment in the router does not make an
    all-GMM portfolio fail.
    """
    model_by_key = {_nfc_key(k): router.measurement_model_of(k)
                    for k in router.segments}
    try:
        _, segments = _factorise_segments(
            router, model_points, router.segment_axes, model_points.n_mp)
    except KeyError:
        if len(router.segments) == 1:
            only = next(iter(router.segments))
            segments = [(_nfc_key(only),
                         np.arange(model_points.n_mp, dtype=np.int64))]
        else:
            raise ValueError(
                "model_points has no segment axes set but the router has "
                f"{len(router.segments)} segments; set the routing columns "
                f"{router.segment_axes}")
    buckets: dict[str, list] = {"GMM": [], "PAA": [], "VFA": []}
    for key, idx in segments:
        buckets[model_by_key[key]].append(np.asarray(idx, dtype=np.int64))
    return {model: (np.sort(np.concatenate(idxs)) if idxs
                    else np.empty(0, dtype=np.int64))
            for model, idxs in buckets.items()}


def _submodel_router(router: BasisRouter, model: str) -> BasisRouter:
    """A router carrying only ``model``'s segments, so the per-model entry accepts it."""
    keys = [k for k in router.segments
            if router.measurement_model_of(k) == model]
    return BasisRouter({k: router.resolve(k) for k in keys},
                       segment_axes=router.segment_axes,
                       measurement_models={k: model for k in keys})


def _measure_model_segmented(sub_mp, sub_router, measure_one, stitch):
    """Measure one model's partition that spans several routing segments.

    ``measure_paa`` / ``measure_vfa`` take a single :class:`Basis`, so the
    orchestrator splits the partition by segment, measures each on its own
    basis, and scatters the per-segment native results back into one stitched
    measurement -- the PAA / VFA analogue of the GMM ``_measure_segmented``.
    ``measure_one(sub, basis)`` runs one segment; ``stitch(n_rows, sub_results)``
    scatters ``[(idx, measurement)]`` into the combined result. The stitched
    result is stamped with ``sub_mp`` so ``group(...)`` resolves the axes, just
    as ``fcf.gmm.measure`` stamps its result.
    """
    try:
        basis_norm, segments = _factorise_segments(
            sub_router, sub_mp, sub_router.segment_axes, sub_mp.n_mp)
    except KeyError:
        if len(sub_router.segments) == 1:
            (basis,) = sub_router.segments.values()
            return measure_one(sub_mp, basis)
        raise ValueError(
            "model_points has no segment axes set but the sub-router has "
            f"{len(sub_router.segments)} segments; set the routing columns "
            f"{sub_router.segment_axes}")
    sub_results = [(idx, measure_one(sub_mp.subset(idx), basis_norm[key]))
                   for key, idx in segments]
    return replace(stitch(sub_mp.n_mp, sub_results), model_points=sub_mp)


def measure(model_points: ModelPoints, basis, *, full: bool = True,
            backend: str = "cpu") -> PortfolioMeasurement:
    """Measure a mixed-model portfolio in one call.

    ``basis`` must be a :class:`~fastcashflow.basis.BasisRouter` whose segments
    carry their IFRS 17 measurement model (``read_basis`` reads it from the
    ``measurement_model`` column of the segments sheet). Each row is routed to
    its segment's model; the result keeps each model's native measurement
    separate. ``full`` matches :func:`fcf.gmm.measure` (the full trajectory vs
    the fused headline).

    Each model's rows are routed to its own kernel and kept in a separate slot
    of the result. A non-GMM row is never silently measured as GMM. Per-segment
    VFA return scenarios (the guarantee time value) are a future extension --
    the mixed path measures the VFA intrinsic value (deterministic) only.
    """
    if not isinstance(basis, BasisRouter):
        raise TypeError(
            "fcf.portfolio.measure requires a BasisRouter (a routed, possibly "
            "mixed-model portfolio); for a single Basis use fcf.gmm.measure / "
            "fcf.paa.measure / fcf.vfa.measure")
    parts = _partition_by_model(model_points, basis)
    covered = sum(part.size for part in parts.values())
    if covered != model_points.n_mp:                  # factorisation must be total
        raise ValueError(
            f"model partition covers {covered} of {model_points.n_mp} rows")
    slots = {}
    gmm_idx = parts["GMM"]
    if gmm_idx.size:
        gmm_meas = _measure_gmm(
            model_points.subset(gmm_idx), _submodel_router(basis, "GMM"),
            full=full, backend=backend)
        slots["gmm"] = ModelMeasurement(index=gmm_idx, measurement=gmm_meas)
    paa_idx = parts["PAA"]
    if paa_idx.size:
        paa_meas = _measure_model_segmented(
            model_points.subset(paa_idx), _submodel_router(basis, "PAA"),
            measure_paa, _stitch_paa_measurements)
        slots["paa"] = ModelMeasurement(index=paa_idx, measurement=paa_meas)
    vfa_idx = parts["VFA"]
    if vfa_idx.size:
        vfa_meas = _measure_model_segmented(
            model_points.subset(vfa_idx), _submodel_router(basis, "VFA"),
            measure_vfa, _stitch_vfa_measurements)
        slots["vfa"] = ModelMeasurement(index=vfa_idx, measurement=vfa_meas)
    return PortfolioMeasurement(model_points=model_points, **slots)
