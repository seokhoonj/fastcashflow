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
    PAAMeasurement, PAAAggregate, measure_paa, _stitch_paa_measurements,
    _scatter_paa_headline)
from fastcashflow._vfa import (
    VFAMeasurement, VFAAggregate, measure_vfa, _stitch_vfa_measurements,
    _scatter_vfa_headline)
from fastcashflow.basis import BasisRouter
from fastcashflow.engine import (
    GMMMeasurement, GMMAggregate, _factorise_segments, measure as _measure_gmm)
from fastcashflow.modelpoints import ModelPoints

#: The native measurement type each model slot must hold (the per-model
#: separation invariant: a paa slot can never carry a GMMMeasurement).
_SLOT_MEASUREMENT_TYPE = {
    "gmm": GMMMeasurement, "paa": PAAMeasurement, "vfa": VFAMeasurement}

#: Per non-GMM model: the single-Basis specialist, the full-trajectory stitch,
#: and the headline-only scatter -- so a model's partition that spans several
#: routing segments is measured then combined into one native measurement
#: (trajectories when ``full=True``, headline-only when ``full=False``). GMM is
#: not here: it routes through ``fcf.gmm.measure`` (its own ``full`` flag and
#: segment stitch), not the PAA/VFA scatter.
_MODEL_EXEC = {
    "PAA": (measure_paa, _stitch_paa_measurements, _scatter_paa_headline),
    "VFA": (measure_vfa, _stitch_vfa_measurements, _scatter_vfa_headline)}

#: Default chunk for the ``full=False`` PAA/VFA path -- matches
#: ``engine.measure_aggregate``. Bounds peak memory to ``O(chunk x n_time)``
#: since PAA/VFA have no fused kernel (each block still builds dense transients).
_CHUNK_SIZE = 200_000

#: Per model, the chunked-aggregate spec for ``measure_aggregate``: the
#: full-trajectory single-Basis measure, the scalar headline fields to total, the
#: ``(field, extra)`` trajectory fields to sum over the model-point axis (``extra``
#: = +1 for an ``(n_time+1,)`` path, +0 for an ``(n_time,)`` series), and the
#: aggregate type to build. Field names match each aggregate's constructor.
_AGG_SPEC = {
    "GMM": (lambda mp, b: _measure_gmm(mp, b, full=True),
            ("bel", "ra", "csm", "loss_component"),
            (("bel_path", 1), ("ra_path", 1), ("csm_path", 1)),
            GMMAggregate),
    "PAA": (lambda mp, b: measure_paa(mp, b, full=True),
            ("lrc", "loss_component"),
            (("lrc_path", 1), ("revenue", 0), ("service_expense", 0), ("lic", 1)),
            PAAAggregate),
    "VFA": (lambda mp, b: measure_vfa(mp, b, full=True),
            ("bel", "ra", "csm", "variable_fee", "time_value", "loss_component"),
            (("bel_path", 1), ("ra_path", 1), ("csm_path", 1), ("lic", 1)),
            VFAAggregate)}


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

    def loss_component_total(self) -> float:
        """The portfolio's total onerous-contract loss at inception.

        This is the **only** quantity summed across measurement models: the loss
        component is ``max(0, fulfilment cash flows)`` at inception, defined and
        signed identically under GMM, PAA and VFA, so adding it across models is
        meaningful. A BEL, an LRC and a VFA BEL are *not* added -- they measure
        different things; reach those through the per-model blocks of
        :meth:`summary` (e.g. ``pm.gmm.measurement.bel``).
        """
        return float(sum(
            mm.measurement.loss_component.sum()
            for mm in (self.gmm, self.paa, self.vfa) if mm is not None))

    def summary(self) -> dict:
        """Per-model headline totals, each model in its own block.

        Returns ``{"loss_component_total": float, <model>: {...}}`` with a block
        only for the models the portfolio carries. Each block sums that model's
        own headline figures over its rows -- ``gmm`` / ``vfa`` give
        ``bel`` / ``ra`` / ``csm`` / ``loss_component``, ``paa`` gives ``lrc`` /
        ``loss_component``. Figures of different meaning are never pooled into
        one number; ``loss_component_total`` is the single cross-model sum (see
        :meth:`loss_component_total`).
        """
        out: dict = {"loss_component_total": self.loss_component_total()}
        if self.gmm is not None:
            m = self.gmm.measurement
            out["gmm"] = {"bel": float(m.bel.sum()), "ra": float(m.ra.sum()),
                          "csm": float(m.csm.sum()),
                          "loss_component": float(m.loss_component.sum())}
        if self.paa is not None:
            m = self.paa.measurement
            out["paa"] = {"lrc": float(m.lrc.sum()),
                          "loss_component": float(m.loss_component.sum())}
        if self.vfa is not None:
            m = self.vfa.measurement
            out["vfa"] = {"bel": float(m.bel.sum()), "ra": float(m.ra.sum()),
                          "csm": float(m.csm.sum()),
                          "loss_component": float(m.loss_component.sum())}
        return out


@dataclass(frozen=True, slots=True)
class PortfolioAggregate:
    """Result of :func:`measure_aggregate`: one aggregate per model present
    (``None`` when absent), each holding that model's inception totals and
    run-off trajectories summed over the model-point axis. A **scalable sum of
    measured model-point results** -- no per-model-point row, so it works at a
    scale where the per-row :class:`PortfolioMeasurement` would not fit in
    memory. **Not an IFRS group remeasurement** and **not a GIC re-floor
    engine**: every figure is the sum of the per-model-point results (CSM is the
    sum of each contract's floored CSM, the headline aggregated -- not
    ``group()``'s ``CSM(sum FCF)``). A BEL and an LRC are never pooled;
    ``loss_component_total`` is the one cross-model sum.
    """

    gmm: GMMAggregate | None = None
    paa: PAAAggregate | None = None
    vfa: VFAAggregate | None = None

    def loss_component_total(self) -> float:
        """Portfolio total onerous-contract loss -- the only cross-model sum
        (``max(0, FCF)`` at inception, sign-identical under GMM / PAA / VFA)."""
        return float(sum(
            a.loss_component for a in (self.gmm, self.paa, self.vfa)
            if a is not None))

    def summary(self) -> dict:
        """Per-model headline totals, each model in its own block (a BEL and an
        LRC are never pooled); ``loss_component_total`` is the lone cross-model
        sum. A block appears only for a model the portfolio carries."""
        out: dict = {"loss_component_total": self.loss_component_total()}
        if self.gmm is not None:
            out["gmm"] = {"bel": self.gmm.bel, "ra": self.gmm.ra,
                          "csm": self.gmm.csm,
                          "loss_component": self.gmm.loss_component}
        if self.paa is not None:
            out["paa"] = {"lrc": self.paa.lrc,
                          "loss_component": self.paa.loss_component}
        if self.vfa is not None:
            out["vfa"] = {"bel": self.vfa.bel, "ra": self.vfa.ra,
                          "csm": self.vfa.csm,
                          "loss_component": self.vfa.loss_component}
        return out


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


def _model_segments(sub_router, sub_mp):
    """``[(basis, idx)]`` for one model's partition, split by routing segment.

    Falls back to the whole partition under a single basis when the model points
    carry no segment axes (mirrors ``_measure_segmented``'s convenience)."""
    try:
        basis_norm, segments = _factorise_segments(
            sub_router, sub_mp, sub_router.segment_axes, sub_mp.n_mp)
    except KeyError:
        if len(sub_router.segments) == 1:
            (basis,) = sub_router.segments.values()
            return [(basis, np.arange(sub_mp.n_mp, dtype=np.int64))]
        raise ValueError(
            "model_points has no segment axes set but the sub-router has "
            f"{len(sub_router.segments)} segments; set the routing columns "
            f"{sub_router.segment_axes}")
    return [(basis_norm[key], idx) for key, idx in segments]


def _measure_model_segmented(sub_mp, sub_router, model, *, full=True,
                             chunk_size=_CHUNK_SIZE):
    """Measure one model's partition that spans several routing segments.

    ``measure_paa`` / ``measure_vfa`` take a single :class:`Basis`, so the
    orchestrator splits the partition by segment and combines the per-segment
    results -- the PAA / VFA analogue of the GMM ``_measure_segmented``.

    ``full=True`` measures each segment's full trajectories and stitches them.
    ``full=False`` chunks each segment into ``chunk_size`` row-blocks, measures
    each block headline-only (``measure_one(..., full=False)``), and scatters the
    headline back -- so peak memory is ``O(chunk_size x n_time)`` (PAA/VFA have no
    fused kernel, so a block still builds dense transients) rather than
    ``O(n_mp x n_time)``. The result is stamped with ``sub_mp`` so ``group(...)``
    resolves the axes, as ``fcf.gmm.measure`` does.
    """
    measure_one, stitch_full, scatter_headline = _MODEL_EXEC[model]
    segs = _model_segments(sub_router, sub_mp)
    if full:
        if len(segs) == 1 and segs[0][1].size == sub_mp.n_mp:
            # One segment over the whole partition -- measure directly, no
            # re-scatter (measure_one already stamps model_points).
            return measure_one(sub_mp, segs[0][0], full=True)
        sub_results = [(idx, measure_one(sub_mp.subset(idx), basis, full=True))
                       for basis, idx in segs]
        return replace(stitch_full(sub_mp.n_mp, sub_results), model_points=sub_mp)
    # Headline path -- chunk within each segment to bound peak memory.
    results = []
    for basis, idx in segs:
        for start in range(0, idx.size, chunk_size):
            block = idx[start:start + chunk_size]
            results.append(
                (block, measure_one(sub_mp.subset(block), basis, full=False)))
    return replace(scatter_headline(sub_mp.n_mp, results), model_points=sub_mp)


def measure(model_points: ModelPoints, basis, *, full: bool = True,
            backend: str = "cpu",
            chunk_size: int = _CHUNK_SIZE) -> PortfolioMeasurement:
    """Measure a mixed-model portfolio in one call.

    ``basis`` must be a :class:`~fastcashflow.basis.BasisRouter` whose segments
    carry their IFRS 17 measurement model (``read_basis`` reads it from the
    ``measurement_model`` column of the segments sheet). Each row is routed to
    its segment's model; the result keeps each model's native measurement
    separate. ``full`` matches :func:`fcf.gmm.measure` (the full trajectory vs
    the fused headline).

    With ``full=False`` the PAA / VFA partitions are measured headline-only in
    ``chunk_size`` row-blocks, so peak memory stays ``O(chunk_size x n_time)``
    instead of materialising every contract's trajectory at once (GMM's
    ``full=False`` is already a fused, no-trajectory kernel; ``chunk_size`` does
    not affect it). The headline-only result still serves ``summary()`` /
    ``loss_component_total()``; ``group`` / ``roll_forward`` / ``report`` need
    ``full=True``. ``full=True`` keeps every trajectory (and is memory-bound).

    Each model's rows are routed to its own kernel and kept in a separate slot
    of the result. A non-GMM row is never silently measured as GMM. Per-segment
    VFA return scenarios (the guarantee time value) are a future extension --
    the mixed path measures the VFA intrinsic value (deterministic) only.

    When the router declares a single measurement model the model partition is a
    no-op, so the whole book routes straight to that model -- no per-row
    partition factorise, no subset copy. (A router that *declares* PAA / VFA
    segments but whose rows happen to be all GMM is not this case: it keeps the
    row partition, which is what makes the unused declared segment harmless.)
    """
    if not isinstance(basis, BasisRouter):
        raise TypeError(
            "fcf.portfolio.measure requires a BasisRouter (a routed, possibly "
            "mixed-model portfolio); for a single Basis use fcf.gmm.measure / "
            "fcf.paa.measure / fcf.vfa.measure")
    if chunk_size < 1:
        # Guard before the chunk loop: chunk_size <= 0 would skip every block and
        # scatter uninitialised np.empty() headline arrays (silently wrong).
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    declared = {basis.measurement_model_of(k) for k in basis.segments}
    if len(declared) == 1:
        # Single declared model -> every row is that model; skip the partition's
        # per-row factorise (engine.py:_factorise_segments) and the subset copy
        # and route the whole book directly. Conservative by design: keyed on the
        # router's *declarations*, not the rows present, so a multi-model router
        # whose rows are currently one model still takes the partition path below.
        (model,) = declared
        index = np.arange(model_points.n_mp, dtype=np.int64)
        if model == "GMM":
            meas = _measure_gmm(model_points, basis, full=full, backend=backend)
        else:
            meas = _measure_model_segmented(
                model_points, basis, model, full=full, chunk_size=chunk_size)
        return PortfolioMeasurement(
            model_points=model_points,
            **{model.lower(): ModelMeasurement(index=index, measurement=meas)})
    parts = _partition_by_model(model_points, basis)
    covered = sum(part.size for part in parts.values())
    if covered != model_points.n_mp:                  # factorisation must be total
        raise ValueError(
            f"model partition covers {covered} of {model_points.n_mp} rows")
    slots = {}
    gmm_idx = parts["GMM"]
    if gmm_idx.size:
        slots["gmm"] = ModelMeasurement(index=gmm_idx, measurement=_measure_gmm(
            model_points.subset(gmm_idx), _submodel_router(basis, "GMM"),
            full=full, backend=backend))
    for model in ("PAA", "VFA"):
        idx = parts[model]
        if idx.size:
            meas = _measure_model_segmented(
                model_points.subset(idx), _submodel_router(basis, model),
                model, full=full, chunk_size=chunk_size)
            slots[model.lower()] = ModelMeasurement(index=idx, measurement=meas)
    return PortfolioMeasurement(model_points=model_points, **slots)


def _aggregate_model(sub_mp, segs, model, chunk_size):
    """Chunked aggregate of one model's partition: measure each (segment, block)
    full=True and sum its headline + trajectories over the model-point axis.

    ``segs`` is ``[(basis, idx)]`` from :func:`_model_segments`. The global
    horizon is the partition's longest contract boundary; each block's (shorter)
    path adds into the leading slice -- a contract's trajectory is zero past its
    term. Peak memory is ``O(chunk_size x n_time)``: no per-model-point row is
    retained, only the running sums.
    """
    measure_full_one, headline_fields, path_fields, cls = _AGG_SPEC[model]
    n_time = int(np.asarray(sub_mp.contract_boundary_months).max())
    totals = {f: 0.0 for f in headline_fields}
    paths = {name: np.zeros(n_time + extra) for name, extra in path_fields}
    for basis, idx in segs:
        for start in range(0, idx.size, chunk_size):
            block = idx[start:start + chunk_size]
            m = measure_full_one(sub_mp.subset(block), basis)
            for f in headline_fields:
                totals[f] += float(getattr(m, f).sum())
            for name, _extra in path_fields:
                summed = getattr(m, name).sum(axis=0)
                paths[name][:summed.shape[0]] += summed
    return cls(**totals, **paths)


def measure_aggregate(model_points: ModelPoints, basis, *,
                      chunk_size: int = _CHUNK_SIZE) -> PortfolioAggregate:
    """Chunked aggregate measurement of a mixed-model portfolio.

    A **scalable sum of measured model-point results**: each model's inception
    totals and run-off trajectories summed over the model-point axis, computed in
    ``chunk_size`` row-blocks so peak memory is ``O(chunk_size x n_time)`` -- it
    works where the per-model-point :func:`measure` ``full=True`` would OOM.
    Returns a :class:`PortfolioAggregate` keeping each model's native figures
    separate (a BEL and an LRC are never pooled).

    **Not an IFRS group remeasurement** and **not a GIC re-floor engine**: every
    figure is the sum of the per-model-point results (CSM is the sum of each
    contract's floored CSM -- the :func:`measure` headline aggregated, not
    ``group()``'s ``CSM(sum FCF)``). A per-GIC re-floor is a separate concern.
    """
    if not isinstance(basis, BasisRouter):
        raise TypeError(
            "fcf.portfolio.measure_aggregate requires a BasisRouter (a routed, "
            "possibly mixed-model portfolio); for a single Basis use "
            "fcf.gmm.measure_aggregate")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    declared = {basis.measurement_model_of(k) for k in basis.segments}
    slots = {}
    if len(declared) == 1:
        # Single declared model -> the whole book is that model; skip the model
        # partition (mirrors measure()'s short-circuit).
        (model,) = declared
        segs = _model_segments(basis, model_points)
        slots[model.lower()] = _aggregate_model(
            model_points, segs, model, chunk_size)
        return PortfolioAggregate(**slots)
    parts = _partition_by_model(model_points, basis)
    covered = sum(part.size for part in parts.values())
    if covered != model_points.n_mp:
        raise ValueError(
            f"model partition covers {covered} of {model_points.n_mp} rows")
    for model in ("GMM", "PAA", "VFA"):
        idx = parts[model]
        if idx.size:
            sub_mp = model_points.subset(idx)
            segs = _model_segments(_submodel_router(basis, model), sub_mp)
            slots[model.lower()] = _aggregate_model(
                sub_mp, segs, model, chunk_size)
    return PortfolioAggregate(**slots)
