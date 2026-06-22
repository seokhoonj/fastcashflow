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
2-D ``discount_factor_bom`` because segments discount at their own underlying-items
return, and the movement / grouping consumers handle that 2-D curve (grouping
keeps a group inside one curve).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from fastcashflow._typing import IntArray
from fastcashflow._paa import (
    PAAMeasurement, PAAAggregate, measure_paa, measure_aggregate as _paa_aggregate,
    measure_inforce as _paa_inforce,
    _stitch_paa_measurements, _scatter_paa_headline)
from fastcashflow._vfa import (
    VFAMeasurement, VFAAggregate, measure_vfa, measure_aggregate as _vfa_aggregate,
    measure_inforce as _vfa_inforce, _require_settlement_csm,
    settle as _settle_vfa,
    _stitch_vfa_measurements, _scatter_vfa_headline)
from fastcashflow.basis import BasisRouter
from fastcashflow.engine import (
    GMMMeasurement, GMMAggregate, _factorise_segments, measure as _measure_gmm,
    measure_aggregate as _gmm_aggregate, measure_inforce as _gmm_inforce,
    settle as _settle_gmm, _reconcile_state)
from fastcashflow.io import (
    _stream_validate, write_measurement, _model_points_from_frames,
    _parse_calculation_methods, _write_measurement_columns)
from fastcashflow.grouping import (
    group, group_of_contracts, _GroupReducer, _join_keys, _finalise_gmm_group,
    _finalise_vfa_group, _finalise_paa_group, _INFORCE_EPS)
from fastcashflow.model_points import ModelPoints, InforceState, align_inforce_state
from fastcashflow.movement import (
    roll_forward, reconcile, GMMSettlementReconciliation,
    VFASettlementReconciliation)
from fastcashflow.numerics import _paragraph45_csm_algebra
from fastcashflow.projection import Cashflows
from fastcashflow.report import report, Report
from fastcashflow.trace import (
    show_trace, show_trace_vfa, show_trace_paa,
    show_trace_diff, show_trace_diff_vfa, show_trace_diff_paa)

#: The orchestrator's public surface -- the measurement entry points and their
#: result containers. Set explicitly so ``from fastcashflow.portfolio import *``
#: does not leak the imported helpers (np, dataclass, BasisRouter, ModelPoints,
#: the leaf measure functions, ...).
__all__ = [
    "measure", "measure_aggregate", "measure_inforce", "measure_stream",
    "measure_group", "measure_group_of_contracts",
    "settle_group_of_contracts", "trace", "trace_diff",
    "PortfolioMeasurement", "PortfolioAggregate", "PortfolioGroups",
    "GoCSettlement",
    "PortfolioReport", "PortfolioMovements", "PortfolioReconciliation",
    "ModelMeasurement",
]

#: Route one model point to its model's tracer (the mixed-portfolio trace).
_MODEL_TRACE = {"GMM": show_trace, "VFA": show_trace_vfa, "PAA": show_trace_paa}
_MODEL_TRACE_DIFF = {"GMM": show_trace_diff, "VFA": show_trace_diff_vfa,
                     "PAA": show_trace_diff_paa}


def _model_of_row(mp_index: int, model_points: ModelPoints,
                  router: BasisRouter) -> str:
    """The measurement model of one row, via the same partition the measures
    use. Shared by the routed tracers."""
    if not isinstance(router, BasisRouter):
        raise TypeError(
            "a routed (mixed-model) portfolio is required here -- pass a "
            "BasisRouter; for a single Basis use the per-model fcf.gmm / "
            "fcf.paa / fcf.vfa entry points")
    if not 0 <= mp_index < model_points.n_mp:
        raise IndexError(
            f"mp_index {mp_index} out of range for n_mp={model_points.n_mp}")
    partition = _partition_by_model(model_points, router)
    model = next((m for m, idx in partition.items()
                  if idx.size and mp_index in idx), None)
    if model is None:   # every row is partitioned, so this is defensive only
        raise ValueError(f"mp_index {mp_index} was not routed to any model")
    return model


def trace(mp_index: int, model_points: ModelPoints, basis, *, file=None) -> None:
    """Trace one model point in a mixed portfolio, routed to its model's tracer.

    The portfolio counterpart of the per-model ``trace``: ``basis`` is a
    :class:`~fastcashflow.basis.BasisRouter` (as for :func:`measure`), the row's
    segment selects its measurement model, and the trace is rendered by that
    model's tracer (``fcf.gmm.trace`` / ``fcf.vfa.trace`` / ``fcf.paa.trace``) --
    so a VFA row is never traced as GMM and the reader need not know a contract's
    model up front. ``file`` defaults to stdout.
    """
    model = _model_of_row(mp_index, model_points, basis)
    _MODEL_TRACE[model](mp_index, model_points, basis, file=file)


def trace_diff(mp_index: int, model_points: ModelPoints, basis_a, basis_b, *,
               label_a: str = "before", label_b: str = "after",
               file=None) -> None:
    """Diff one model point across two bases in a mixed portfolio, routed to its
    model's diff tracer.

    The portfolio counterpart of the per-model ``trace_diff``: ``basis_a`` /
    ``basis_b`` are two :class:`~fastcashflow.basis.BasisRouter` s, the row's
    segment selects its measurement model, and the shock diff is rendered by
    that model's diff tracer (``fcf.gmm.trace_diff`` / ``fcf.vfa.trace_diff`` /
    ``fcf.paa.trace_diff``). The model is taken from ``basis_a``. ``file``
    defaults to stdout.
    """
    model = _model_of_row(mp_index, model_points, basis_a)
    if not isinstance(basis_b, BasisRouter):
        raise TypeError("fcf.portfolio.trace_diff requires both bases to be a "
                        "BasisRouter")
    _MODEL_TRACE_DIFF[model](mp_index, model_points, basis_a, basis_b,
                             label_a=label_a, label_b=label_b, file=file)


def measure_inforce(model_points: ModelPoints, state: InforceState, basis, *,
                    period_months: int | None = None, full: bool = True,
                    backend: str = "cpu") -> PortfolioMeasurement:
    """In-force subsequent measurement of a mixed-model portfolio (IFRS 17 Sec. 44).

    The portfolio counterpart of :func:`fcf.gmm.measure_inforce`: each row is
    routed to its segment's model and valued at its ``elapsed_months`` valuation
    date by that model's ``measure_inforce``, the native measurements kept in
    separate slots (a non-GMM row is never measured as GMM). ``basis`` is a
    :class:`~fastcashflow.basis.BasisRouter`; ``state`` an :class:`InforceState`
    aligned to ``model_points`` by ``mp_id`` (each model's leaf re-aligns it to
    its own partition). ``period_months`` drives the GMM / VFA prior-CSM carry
    (PAA has no CSM, so it is immaterial to the PAA slot).

    Like :func:`measure`, the mixed path uses each model's default sub-options
    (the PAA partition uses the ``"time"`` revenue basis; per-segment VFA return
    scenarios are deferred) -- set a model-specific option through that model's
    own ``measure_inforce``. GMM rows may span multiple segments; the VFA / PAA
    partitions must resolve to a single :class:`Basis` per model (their leaf
    in-force is single-Basis), as for :func:`fcf.vfa.measure` / :func:`fcf.paa.measure`.
    """
    if not isinstance(basis, BasisRouter):
        raise TypeError(
            "fcf.portfolio.measure_inforce requires a BasisRouter (a routed, "
            "possibly mixed-model portfolio); for a single Basis use "
            "fcf.gmm.measure_inforce / fcf.paa.measure_inforce / "
            "fcf.vfa.measure_inforce")
    # Row-align the state to the full book once, so each model partition's
    # state.subset(idx) lines up with model_points.subset(idx) -- the leaf
    # measure_inforce requires the state to cover exactly its valued contracts.
    state = align_inforce_state(model_points, state)
    parts = _partition_by_model(model_points, basis)
    covered = sum(part.size for part in parts.values())
    if covered != model_points.n_mp:                  # factorisation must be total
        raise ValueError(
            f"model partition covers {covered} of {model_points.n_mp} rows")
    slots = {}
    gmm_idx = parts["GMM"]
    if gmm_idx.size:
        slots["gmm"] = ModelMeasurement(index=gmm_idx, measurement=_gmm_inforce(
            model_points.subset(gmm_idx), state.subset(gmm_idx),
            _submodel_router(basis, "GMM"), period_months=period_months, full=full))
    paa_idx = parts["PAA"]
    if paa_idx.size:
        slots["paa"] = ModelMeasurement(index=paa_idx, measurement=_paa_inforce(
            model_points.subset(paa_idx), state.subset(paa_idx),
            _submodel_router(basis, "PAA"), full=full))
    vfa_idx = parts["VFA"]
    if vfa_idx.size:
        slots["vfa"] = ModelMeasurement(index=vfa_idx, measurement=_vfa_inforce(
            model_points.subset(vfa_idx), state.subset(vfa_idx),
            _submodel_router(basis, "VFA"), period_months=period_months))
    return PortfolioMeasurement(model_points=model_points, **slots)


def measure_stream(input_path, output_dir, basis, *, coverages=None,
                   calculation_methods=None, chunk_size: int = 20_000_000,
                   full: bool = False, backend: str = "cpu",
                   id_column: str | None = None,
                   validate_unique_mp_id: bool = True) -> int:
    """Stream a mixed-model portfolio valuation through a parquet file, chunk by
    chunk.

    The portfolio counterpart of the per-model ``measure_stream``: read the
    policies (+ ``coverages``) parquet in ``chunk_size`` blocks, route each
    chunk by model with :func:`measure`, and write each model's results to its
    OWN subdirectory -- ``output_dir/{gmm,paa,vfa}/part-NNNNN.parquet`` -- since
    the models' result columns differ (GMM / VFA write bel / ra / csm, PAA
    lrc / loss_component). Returns the number of model points processed.
    ``basis`` is a :class:`~fastcashflow.basis.BasisRouter`, as for
    :func:`measure`.

    Marginal benefit note: :func:`measure` already bounds memory via
    ``chunk_size``; this is the file-based out-of-core form for a book too large
    to read at once. The split-by-model output is what lets one streamed book
    carry heterogeneous per-model result schemas.
    """
    if not isinstance(basis, BasisRouter):
        raise TypeError(
            "fcf.portfolio.measure_stream requires a BasisRouter (a routed, "
            "possibly mixed-model portfolio); for a single Basis use "
            "fcf.gmm.measure_stream / fcf.paa.measure_stream / "
            "fcf.vfa.measure_stream")
    input_path, output_dir = Path(input_path), Path(output_dir)
    scan, n_total, id_col = _stream_validate(
        input_path, output_dir, id_column, validate_unique_mp_id)
    if any(output_dir.glob("*/part-*.parquet")):
        raise ValueError(
            f"output directory {str(output_dir)!r} already contains model part "
            "files; use a fresh directory")
    methods_dict = (_parse_calculation_methods(calculation_methods)
                    if isinstance(calculation_methods, (str, Path))
                    else calculation_methods)
    cov_scan = pl.scan_parquet(Path(coverages)) if coverages is not None else None
    processed = 0
    for part, offset in enumerate(range(0, n_total, chunk_size)):
        pol = scan.slice(offset, chunk_size).collect()
        ids = pol[id_col].to_numpy()
        cov = (cov_scan.join(pol.lazy().select("mp_id"), on="mp_id", how="semi")
               .collect() if cov_scan is not None else None)
        chunk_mp = _model_points_from_frames(pol, cov, methods_dict)
        pm = measure(chunk_mp, basis, full=full, backend=backend,
                     chunk_size=chunk_size)
        for model in ("gmm", "paa", "vfa"):
            slot = getattr(pm, model)
            if slot is not None:
                sub = output_dir / model
                sub.mkdir(parents=True, exist_ok=True)
                write_measurement(slot.measurement,
                                  sub / f"part-{part:05d}.parquet",
                                  ids=ids[slot.index])
        processed += chunk_mp.n_mp
    return processed

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

#: Per model: measure one block with full trajectories (single Basis). Used by
#: the per-group aggregate (``measure_group`` / ``measure_group_of_contracts``),
#: which needs each chunk's full per-MP result to group-sum and to read the
#: discount curve.
_MEASURE_FULL = {
    "GMM": lambda mp, b: _measure_gmm(mp, b, full=True),
    "PAA": lambda mp, b: measure_paa(mp, b, full=True),
    "VFA": lambda mp, b: measure_vfa(mp, b, full=True)}

#: Per model: the leaf bounded-memory aggregate (single Basis). ``measure_aggregate``
#: reuses these so the chunked-sum logic lives once in each model's namespace, not
#: duplicated in the orchestrator.
_LEAF_AGGREGATE = {
    "GMM": _gmm_aggregate, "PAA": _paa_aggregate, "VFA": _vfa_aggregate}


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
    downstream per-segment / group of contracts analysis (e.g. ``group(pm.gmm.measurement,
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


@write_measurement.register
def _(measurement: PortfolioMeasurement, path, *, ids=None):
    """Write each model slot's headline results to its own file.

    The models' column sets differ (GMM / VFA write bel / ra / csm, PAA
    lrc / loss_component), so one portfolio writes one file per model
    present, the model name appended to the stem -- ``results.parquet``
    becomes ``results-gmm.parquet`` / ``results-paa.parquet`` /
    ``results-vfa.parquet``. Unlike the single-model writers, every file
    carries an ``id`` column even when ``ids`` is not passed: a model's
    rows are a partition slice of the portfolio, so without an id they
    could not be joined back. The column is ``ids[index]`` when ``ids`` is
    given, the portfolio row position otherwise.
    """
    path = Path(path)
    if ids is not None:
        ids = np.asarray(ids)
        if ids.shape[0] != measurement.model_points.n_mp:
            raise ValueError(
                f"ids has {ids.shape[0]} rows but the portfolio has "
                f"{measurement.model_points.n_mp}")
    # Preflight the one known per-slot rejection (a carry-only VFA CSM, e.g.
    # from measure_inforce) before the first file is written, so a refusal
    # cannot leave a partial gmm/paa output on disk.
    if measurement.vfa is not None:
        _require_settlement_csm(measurement.vfa.measurement, "write_measurement")
    for model in ("gmm", "paa", "vfa"):
        mm = getattr(measurement, model)
        if mm is None:
            continue
        row_ids = mm.index if ids is None else ids[mm.index]
        write_measurement(
            mm.measurement,
            path.with_name(f"{path.stem}-{model}{path.suffix}"),
            ids=row_ids)


@dataclass(frozen=True, slots=True)
class PortfolioAggregate:
    """Result of :func:`measure_aggregate`: one aggregate per model present
    (``None`` when absent), each holding that model's inception totals and
    run-off trajectories summed over the model-point axis. A **scalable sum of
    measured model-point results** -- no per-model-point row, so it works at a
    scale where the per-row :class:`PortfolioMeasurement` would not fit in
    memory. **Not an IFRS group remeasurement** and **not a group re-floor
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
                             chunk_size=_CHUNK_SIZE, return_scenarios=None):
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

    ``return_scenarios`` (VFA only -- the guarantee time value) is forwarded to
    ``measure_vfa``. When given, the headline path is NOT chunked: the scenario
    matrix is ``(n_scenarios, horizon)`` and ``measure_vfa`` validates its width
    against each segment's horizon, so a short row-block would mismatch. The
    per-model-point time value is additive, and the headline result still drops
    trajectories, so each segment is measured in a single ``full=False`` call --
    peak memory ``O(n_seg_mp x n_scenarios)`` for the scenario pass rather than
    ``O(n_mp x n_time)`` trajectories.
    """
    measure_one, stitch_full, scatter_headline = _MODEL_EXEC[model]
    segs = _model_segments(sub_router, sub_mp)
    extra = {} if return_scenarios is None else {"return_scenarios": return_scenarios}
    if full:
        if len(segs) == 1 and segs[0][1].size == sub_mp.n_mp:
            # One segment over the whole partition -- measure directly, no
            # re-scatter (measure_one already stamps model_points).
            return measure_one(sub_mp, segs[0][0], full=True, **extra)
        sub_results = [(idx, measure_one(sub_mp.subset(idx), basis, full=True, **extra))
                       for basis, idx in segs]
        return replace(stitch_full(sub_mp.n_mp, sub_results), model_points=sub_mp)
    if return_scenarios is not None:
        # Scenario pass: measure each segment in one block (the scenario horizon
        # must match the segment horizon, and the per-MP time value is additive).
        results = [(idx, measure_one(sub_mp.subset(idx), basis, full=False, **extra))
                   for basis, idx in segs]
        return replace(scatter_headline(sub_mp.n_mp, results), model_points=sub_mp)
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
            chunk_size: int = _CHUNK_SIZE,
            return_scenarios=None) -> PortfolioMeasurement:
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
    of the result. A non-GMM row is never silently measured as GMM.

    ``return_scenarios`` -- an ``(n_scenarios, horizon)`` array of monthly
    underlying-items returns -- prices the guarantee time value (TVOG) of the
    VFA partition, exactly as ``fcf.vfa.measure(..., return_scenarios=...)``
    does for a single book; it is forwarded only to the VFA slot (GMM / PAA
    carry no return guarantee). It is an error to pass it for a portfolio with
    no VFA rows. When supplied, the VFA partition is measured without chunking
    (the scenario horizon must match the segment horizon; the per-model-point
    time value is additive), so ``chunk_size`` does not bound the VFA scenario
    pass. Omitted (the default), the mixed path measures the VFA intrinsic value
    (deterministic) only.

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
        if return_scenarios is not None and model != "VFA":
            raise ValueError(
                "return_scenarios (the guarantee time value) applies to the VFA "
                f"partition; this portfolio declares only {model}")
        index = np.arange(model_points.n_mp, dtype=np.int64)
        if model == "GMM":
            meas = _measure_gmm(model_points, basis, full=full, backend=backend)
        else:
            meas = _measure_model_segmented(
                model_points, basis, model, full=full, chunk_size=chunk_size,
                return_scenarios=return_scenarios if model == "VFA" else None)
        return PortfolioMeasurement(
            model_points=model_points,
            **{model.lower(): ModelMeasurement(index=index, measurement=meas)})
    parts = _partition_by_model(model_points, basis)
    covered = sum(part.size for part in parts.values())
    if covered != model_points.n_mp:                  # factorisation must be total
        raise ValueError(
            f"model partition covers {covered} of {model_points.n_mp} rows")
    if return_scenarios is not None and parts["VFA"].size == 0:
        raise ValueError(
            "return_scenarios (the guarantee time value) applies to the VFA "
            "partition; this portfolio has no VFA rows")
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
                model, full=full, chunk_size=chunk_size,
                return_scenarios=return_scenarios if model == "VFA" else None)
            slots[model.lower()] = ModelMeasurement(index=idx, measurement=meas)
    return PortfolioMeasurement(model_points=model_points, **slots)


def _sum_aggregates(aggs):
    """Sum same-type ``<Model>Aggregate`` objects field by field.

    Scalars add; trajectory arrays add into the leading slice of the longest (a
    shorter routing segment carries nothing past its horizon, and the
    longest-horizon segment reaches the partition horizon). Generic over the
    aggregate dataclass, so no per-model field list is repeated in the
    orchestrator -- the field set lives with each aggregate type.
    """
    cls = type(aggs[0])
    out = {}
    for f in fields(cls):
        vals = [getattr(a, f.name) for a in aggs]
        if isinstance(vals[0], np.ndarray):
            acc = np.zeros(max(v.shape[0] for v in vals))
            for v in vals:
                acc[:v.shape[0]] += v
            out[f.name] = acc
        else:
            out[f.name] = float(sum(float(v) for v in vals))
    return cls(**out)


def _aggregate_model(sub_mp, segs, model, chunk_size):
    """Chunked aggregate of one model's partition -- the leaf bounded-memory
    aggregate per routing segment, summed across segments.

    Each routing segment is one Basis, so it runs the model's own
    ``measure_aggregate`` (``gmm`` / ``paa`` / ``vfa``) -- the chunked-sum logic
    lives there, single-sourced -- and the per-segment aggregates are summed field
    by field (trajectories into the leading slice of the partition horizon).
    ``segs`` is ``[(basis, idx)]`` from :func:`_model_segments`. Peak memory stays
    ``O(chunk_size x n_time)``: the leaf retains only running sums per block.
    """
    leaf = _LEAF_AGGREGATE[model]
    aggs = [leaf(sub_mp.subset(idx), basis, chunk_size=chunk_size)
            for basis, idx in segs]
    return _sum_aggregates(aggs)


def measure_aggregate(model_points: ModelPoints, basis, *,
                      chunk_size: int = _CHUNK_SIZE) -> PortfolioAggregate:
    """Chunked aggregate measurement of a mixed-model portfolio.

    A **scalable sum of measured model-point results**: each model's inception
    totals and run-off trajectories summed over the model-point axis, computed in
    ``chunk_size`` row-blocks so peak memory is ``O(chunk_size x n_time)`` -- it
    works where the per-model-point :func:`measure` ``full=True`` would OOM.
    Returns a :class:`PortfolioAggregate` keeping each model's native figures
    separate (a BEL and an LRC are never pooled).

    **Not an IFRS group remeasurement** and **not a group re-floor engine**: every
    figure is the sum of the per-model-point results (CSM is the sum of each
    contract's floored CSM -- the :func:`measure` headline aggregated, not
    ``group()``'s ``CSM(sum FCF)``). A per-group re-floor is a separate concern.
    """
    if not isinstance(basis, BasisRouter):
        raise TypeError(
            "fcf.portfolio.measure_aggregate requires a BasisRouter (a routed, "
            "possibly mixed-model portfolio); for a single Basis use "
            "fcf.gmm.measure_aggregate / fcf.paa.measure_aggregate / "
            "fcf.vfa.measure_aggregate")
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


# ---------------------------------------------------------------------------
# Per-group aggregate -- the scalable form of group_of_contracts (P-5c).
# ---------------------------------------------------------------------------

#: Cash-flow streams scattered as ``(n_mp, n_time)`` (the rest -- ``maturity_cf``
#: / ``maturity_survivors`` -- are ``(n_mp,)`` scalars). Mirrors the stitch /
#: ``_sum_cashflows`` field set so the chunked group sum matches it exactly.
_CF_STREAMS_2D = (
    "inforce", "deaths", "premium_cf", "mortality_cf", "morbidity_cf",
    "expense_cf", "annuity_cf", "disability_cf", "surrender_cf")

#: The native grouped measurement type each slot of PortfolioGroups holds.
_SLOT_GROUP_TYPE = {
    "gmm": GMMMeasurement, "paa": PAAMeasurement, "vfa": VFAMeasurement}


@dataclass(frozen=True, slots=True)
class PortfolioGroups:
    """Result of :func:`measure_group` / :func:`measure_group_of_contracts`: one native grouped
    measurement per model present (``None`` when absent), its rows the groups (an
    IFRS 17 group of contracts for :func:`measure_group_of_contracts`). The scalable form of
    :func:`fcf.group_of_contracts` -- it **re-floors on each group's fulfilment
    cash flows** (``CSM(sum FCF)``), computed in bounded memory so it works where
    holding the per-model-point ``measure(full=True)`` would OOM. A BEL and an LRC
    are never pooled; ``loss_component_total`` is the one cross-model sum.

    Each slot is the same native type :func:`fcf.group` returns
    (``GMMMeasurement`` / ``PAAMeasurement`` / ``VFAMeasurement`` with
    ``group_labels`` / ``group_sizes`` set), so the group rows flow straight into
    :func:`fcf.roll_forward` / :func:`fcf.reconcile` / :func:`fcf.report`. There
    is no cross-model group of contracts: a portfolio (product) carries one measurement model, so
    a group always sits inside one model's slot.
    """

    gmm: GMMMeasurement | None = None
    paa: PAAMeasurement | None = None
    vfa: VFAMeasurement | None = None

    def __post_init__(self):
        for slot, expected in _SLOT_GROUP_TYPE.items():
            m = getattr(self, slot)
            if m is not None and not isinstance(m, expected):
                raise TypeError(
                    f"PortfolioGroups.{slot} must hold a {expected.__name__}, "
                    f"got {type(m).__name__}")

    def loss_component_total(self) -> float:
        """The portfolio's total onerous-group loss -- the **only** quantity
        summed across measurement models (``max(0, group FCF)``, defined and
        signed identically under GMM / PAA / VFA). A BEL, an LRC and a VFA BEL are
        not added; reach those through :meth:`summary`."""
        return float(sum(
            m.loss_component.sum()
            for m in (self.gmm, self.paa, self.vfa) if m is not None))

    def summary(self) -> dict:
        """Per-model headline totals, each model in its own block (a BEL and an
        LRC are never pooled); ``loss_component_total`` is the lone cross-model
        sum. Each block sums that model's figures over its group rows -- ``gmm`` /
        ``vfa`` give ``bel`` / ``ra`` / ``csm`` / ``loss_component``, ``paa``
        gives ``lrc`` / ``loss_component``. A block appears only for a model the
        portfolio carries."""
        out: dict = {"loss_component_total": self.loss_component_total()}
        if self.gmm is not None:
            m = self.gmm
            out["gmm"] = {"bel": float(m.bel.sum()), "ra": float(m.ra.sum()),
                          "csm": float(m.csm.sum()),
                          "loss_component": float(m.loss_component.sum())}
        if self.paa is not None:
            m = self.paa
            out["paa"] = {"lrc": float(m.lrc.sum()),
                          "loss_component": float(m.loss_component.sum())}
        if self.vfa is not None:
            m = self.vfa
            out["vfa"] = {"bel": float(m.bel.sum()), "ra": float(m.ra.sum()),
                          "csm": float(m.csm.sum()),
                          "loss_component": float(m.loss_component.sum())}
        return out


_GOC_SETTLEMENT_LINEAR = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "finance_wedge", "premium_experience_revenue", "csm_opening",
    "csm_accretion", "csm_experience_unlocking", "csm_premium_experience",
    "csm_investment_experience", "claims_experience", "expense_experience",
    "loss_component_opening", "loss_component_finance",
    "loss_component_amortised",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
)
_GOC_SETTLEMENT_NONLINEAR = (
    "csm_release", "csm_closing", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
)
_GOC_SETTLEMENT_UNIT_LINES = ("coverage_units_provided", "coverage_units_future")


@dataclass(frozen=True, slots=True, eq=False)
class GoCSettlement:
    """Group-of-contracts paragraph-44 settlement movement.

    Rows are IFRS 17 groups. Linear GMM settlement lines are group-summed; the
    paragraph-48/50(b) CSM/loss-component algebra and the B119 release are
    applied once at group grain. ``closing_inputs()`` allocates group closing
    balances back to model points by closing-count pro-rata, or by an explicit
    per-row allocation weight.
    """

    group_labels: np.ndarray
    group_sizes: IntArray
    period_months: int
    bel_opening: np.ndarray
    bel_interest: np.ndarray
    bel_release: np.ndarray
    bel_experience: np.ndarray
    bel_closing: np.ndarray
    ra_opening: np.ndarray
    ra_interest: np.ndarray
    ra_release: np.ndarray
    ra_experience: np.ndarray
    ra_closing: np.ndarray
    finance_wedge: np.ndarray
    premium_experience_revenue: np.ndarray
    csm_opening: np.ndarray
    csm_accretion: np.ndarray
    csm_experience_unlocking: np.ndarray
    csm_premium_experience: np.ndarray
    csm_investment_experience: np.ndarray
    claims_experience: np.ndarray
    expense_experience: np.ndarray
    loss_component_opening: np.ndarray
    loss_component_finance: np.ndarray
    loss_component_amortised: np.ndarray
    lic_opening: np.ndarray
    claims_incurred: np.ndarray
    lic_finance: np.ndarray
    claims_paid: np.ndarray
    lic_closing: np.ndarray
    coverage_units_provided: np.ndarray
    coverage_units_future: np.ndarray
    csm_release: np.ndarray
    csm_closing: np.ndarray
    loss_component_reversed: np.ndarray
    loss_component_recognised: np.ndarray
    loss_component_closing: np.ndarray
    lock_in_rate: np.ndarray
    model_points: ModelPoints | None = None
    group_inverse: IntArray | None = None
    lock_in_rate_by_mp: np.ndarray | float = 0.0
    profitability_by_mp: np.ndarray | None = None
    measurement_basis: str = "settlement"

    _LINEAR: ClassVar[tuple[str, ...]] = _GOC_SETTLEMENT_LINEAR
    _NONLINEAR: ClassVar[tuple[str, ...]] = _GOC_SETTLEMENT_NONLINEAR

    def closing_inputs(self, *, allocation=None):
        from fastcashflow.model_points import InforceState
        mp = self.model_points
        inv = self.group_inverse
        if mp is None or inv is None or mp.mp_id is None:
            raise ValueError(
                "closing_inputs() needs the source model points with mp_id and "
                "group membership; use settle_group_of_contracts to create it")
        n_mp = mp.n_mp
        if allocation is None:
            weights = np.asarray(mp.count, dtype=np.float64)
        else:
            weights = np.asarray(allocation, dtype=np.float64)
            if weights.shape != (n_mp,):
                raise ValueError(
                    f"allocation must have one entry per model point ({n_mp}), "
                    f"got shape {weights.shape}")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("allocation must be finite and >= 0")
        denom = np.bincount(inv, weights=weights, minlength=self.group_labels.shape[0])
        share = np.zeros(n_mp, dtype=np.float64)
        for g in range(self.group_labels.shape[0]):
            rows = inv == g
            if denom[g] > 0.0:
                share[rows] = weights[rows] / denom[g]
            else:
                share[rows] = 1.0 / max(1, int(rows.sum()))
        prior_csm = self.csm_closing[inv] * share
        prior_lc = self.loss_component_closing[inv] * share
        state = InforceState(
            mp_id=mp.mp_id,
            elapsed_months=np.asarray(mp.elapsed_months, dtype=np.int64),
            count=np.asarray(mp.count, dtype=np.float64),
            prior_csm=prior_csm,
            lock_in_rate=self.lock_in_rate_by_mp,
            prior_count=np.asarray(mp.count, dtype=np.float64),
            prior_loss_component=prior_lc,
            profitability=self.profitability_by_mp,
        )
        return mp, state


def _finalise_goc_settlement(pre: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    accreted = pre["csm_opening"] + pre["csm_accretion"]
    # The paragraph-50(a) incurred channel is LINEAR (group-summed per-MP); the
    # future-service algebra runs once at group grain on the POST-amortisation
    # loss component, mirroring the per-MP settle.
    lc_after_incurred = (pre["loss_component_opening"]
                         + pre["loss_component_finance"]
                         - pre["loss_component_amortised"])
    csm_after, lc_rev, lc_rec, lc_close = _paragraph45_csm_algebra(
        accreted,
        pre["csm_experience_unlocking"] + pre["csm_premium_experience"]
        + pre["csm_investment_experience"],
        lc_after_incurred)
    denom = pre["coverage_units_provided"] + pre["coverage_units_future"]
    frac = np.where(denom > 0.0, pre["coverage_units_provided"] / denom, 1.0)
    release = csm_after * frac
    out = {name: pre[name] for name in _GOC_SETTLEMENT_LINEAR}
    out["coverage_units_provided"] = pre["coverage_units_provided"]
    out["coverage_units_future"] = pre["coverage_units_future"]
    out["loss_component_reversed"] = lc_rev
    out["loss_component_recognised"] = lc_rec
    out["loss_component_closing"] = lc_close
    out["csm_release"] = release
    out["csm_closing"] = csm_after - release
    return out


_VFA_GOC_SETTLEMENT_LINEAR = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_fv_share", "csm_future_service", "csm_premium_experience",
    "premium_experience_revenue", "csm_investment_experience",
    "claims_experience", "expense_experience",
    "csm_opening", "csm_accretion",
    "variable_fee_closing", "account_value_closing", "loss_component_opening",
    "loss_component_finance", "loss_component_amortised",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
)
_VFA_GOC_SETTLEMENT_NONLINEAR = (
    "csm_release", "csm_closing", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
)


@dataclass(frozen=True, slots=True, eq=False)
class VFAGoCSettlement:
    """Group-of-contracts paragraph-45 settlement movement (VFA).

    The VFA mirror of :class:`GoCSettlement`. Rows are IFRS 17 groups. The
    LINEAR VFA settlement lines are group-summed -- including ``csm_fv_share``
    (45(b)) and ``csm_future_service`` (45(c)), each carrying its own
    ``v_half`` / ``k_obs``, so the group fv_share is the SUM of the per-MP
    fv_shares (not a re-derivation from a re-summed group account value). The
    paragraph-48/50(b) algebra and the single B119 release are applied once at
    group grain on the group-summed inputs (the future-service change is
    ``sum(csm_fv_share + csm_future_service)``). ``closing_inputs()`` allocates
    the group closing CSM / loss component back to model points by closing-
    count pro-rata (or an explicit weight) and carries each contract's observed
    account value forward.
    """

    group_labels: np.ndarray
    group_sizes: IntArray
    period_months: int
    bel_opening: np.ndarray
    bel_interest: np.ndarray
    bel_release: np.ndarray
    bel_experience: np.ndarray
    bel_closing: np.ndarray
    ra_opening: np.ndarray
    ra_interest: np.ndarray
    ra_release: np.ndarray
    ra_experience: np.ndarray
    ra_closing: np.ndarray
    csm_fv_share: np.ndarray
    csm_future_service: np.ndarray
    csm_premium_experience: np.ndarray
    premium_experience_revenue: np.ndarray
    csm_investment_experience: np.ndarray
    claims_experience: np.ndarray
    expense_experience: np.ndarray
    csm_opening: np.ndarray
    csm_accretion: np.ndarray
    variable_fee_closing: np.ndarray
    account_value_closing: np.ndarray
    loss_component_opening: np.ndarray
    loss_component_finance: np.ndarray
    loss_component_amortised: np.ndarray
    lic_opening: np.ndarray
    claims_incurred: np.ndarray
    lic_finance: np.ndarray
    claims_paid: np.ndarray
    lic_closing: np.ndarray
    coverage_units_provided: np.ndarray
    coverage_units_future: np.ndarray
    csm_release: np.ndarray
    csm_closing: np.ndarray
    loss_component_reversed: np.ndarray
    loss_component_recognised: np.ndarray
    loss_component_closing: np.ndarray
    lock_in_rate: np.ndarray
    model_points: ModelPoints | None = None
    group_inverse: IntArray | None = None
    lock_in_rate_by_mp: np.ndarray | float = 0.0
    profitability_by_mp: np.ndarray | None = None
    account_value_by_mp: np.ndarray | None = None
    measurement_basis: str = "settlement"

    _LINEAR: ClassVar[tuple[str, ...]] = _VFA_GOC_SETTLEMENT_LINEAR
    _NONLINEAR: ClassVar[tuple[str, ...]] = _VFA_GOC_SETTLEMENT_NONLINEAR

    def closing_inputs(self, *, allocation=None):
        from fastcashflow.model_points import InforceState
        mp = self.model_points
        inv = self.group_inverse
        if mp is None or inv is None or mp.mp_id is None:
            raise ValueError(
                "closing_inputs() needs the source model points with mp_id and "
                "group membership; use settle_group_of_contracts to create it")
        if self.account_value_by_mp is None:
            raise ValueError(
                "closing_inputs() needs the observed per-MP account value to "
                "carry forward (it is stamped by settle_group_of_contracts)")
        n_mp = mp.n_mp
        if allocation is None:
            weights = np.asarray(mp.count, dtype=np.float64)
        else:
            weights = np.asarray(allocation, dtype=np.float64)
            if weights.shape != (n_mp,):
                raise ValueError(
                    f"allocation must have one entry per model point ({n_mp}), "
                    f"got shape {weights.shape}")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("allocation must be finite and >= 0")
        n_groups = self.group_labels.shape[0]
        denom = np.bincount(inv, weights=weights, minlength=n_groups)
        share = np.zeros(n_mp, dtype=np.float64)
        for g in range(n_groups):
            rows = inv == g
            if denom[g] > 0.0:
                share[rows] = weights[rows] / denom[g]
            else:
                share[rows] = 1.0 / max(1, int(rows.sum()))
        prior_csm = self.csm_closing[inv] * share
        prior_lc = self.loss_component_closing[inv] * share
        av = np.asarray(self.account_value_by_mp, dtype=np.float64)
        state = InforceState(
            mp_id=mp.mp_id,
            elapsed_months=np.asarray(mp.elapsed_months, dtype=np.int64),
            count=np.asarray(mp.count, dtype=np.float64),
            prior_csm=prior_csm,
            lock_in_rate=self.lock_in_rate_by_mp,
            account_value=av,
            prior_count=np.asarray(mp.count, dtype=np.float64),
            prior_account_value=av,
            prior_loss_component=prior_lc,
            profitability=self.profitability_by_mp,
        )
        return mp, state


def _finalise_vfa_goc_settlement(
        pre: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    accreted = pre["csm_opening"] + pre["csm_accretion"]
    # The premium- and investment-experience future legs (B96(a)/(c)) are new
    # future-service changes with no BEL/RA counterpart, so they enter the
    # algebra on top of x (which stays the csm_fv_share / csm_future_service
    # cross-tie quantity). The paragraph-50(a) incurred channel is LINEAR
    # (group-summed); the algebra runs once on the POST-amortisation LC.
    lc_after_incurred = (pre["loss_component_opening"]
                         + pre["loss_component_finance"]
                         - pre["loss_component_amortised"])
    x = pre["csm_fv_share"] + pre["csm_future_service"]
    csm_after, lc_rev, lc_rec, lc_close = _paragraph45_csm_algebra(
        accreted,
        x + pre["csm_premium_experience"] + pre["csm_investment_experience"],
        lc_after_incurred)
    denom = pre["coverage_units_provided"] + pre["coverage_units_future"]
    # Full group derecognition (no coverage units) releases the whole remaining
    # CSM (B119 / paragraph 76), frac=1, matching the GMM GoC and per-MP settle;
    # a 0.0 fallback would strand a fully-derecognised group's CSM.
    frac = np.where(denom > 0.0, pre["coverage_units_provided"] / denom, 1.0)
    release = csm_after * frac
    out = {name: pre[name] for name in _VFA_GOC_SETTLEMENT_LINEAR}
    out["coverage_units_provided"] = pre["coverage_units_provided"]
    out["coverage_units_future"] = pre["coverage_units_future"]
    out["loss_component_reversed"] = lc_rev
    out["loss_component_recognised"] = lc_rec
    out["loss_component_closing"] = lc_close
    out["csm_release"] = release
    out["csm_closing"] = csm_after - release
    return out


def _settlement_axis(mp: ModelPoints, state: InforceState, spec, name: str):
    if isinstance(spec, str):
        if hasattr(state, spec):
            val = getattr(state, spec)
            if val is not None:
                return np.asarray(val)
        return np.asarray(mp.axis(spec))
    arr = np.asarray(spec)
    if arr.shape != (mp.n_mp,):
        raise ValueError(
            f"{name} must have one entry per model point ({mp.n_mp}), "
            f"got shape {arr.shape}")
    return arr


def settle_group_of_contracts(
    model_points: ModelPoints,
    inforce_state: InforceState,
    basis,
    period_months: int | None = None,
    *,
    portfolio="product",
    cohort="issue_year",
    coverage_units=None,
    profitability=None,
    premium_experience_future_fraction=0.0,
    chunk_size: int = _CHUNK_SIZE,
) -> "GoCSettlement | VFAGoCSettlement":
    """Group-of-contracts settlement for a routed CSM portfolio.

    A pure GMM book returns a :class:`GoCSettlement` (paragraph 44); a pure VFA
    book returns a :class:`VFAGoCSettlement` (paragraph 45). ``coverage_units``
    and ``profitability`` are required and explicit. PAA is rejected (no CSM /
    floor, so a per-GoC algebra is meaningless -- use ``paa.settle`` and sum by
    your own groupby), and a book mixing GMM and VFA is rejected whole: a group
    of contracts sits in one product, hence one measurement model. The
    ``premium_experience_future_fraction`` argument applies to both the GMM and
    VFA paths (Sec. B96(a); each routes the future leg into its own
    paragraph-45 algebra).
    """
    if not isinstance(basis, BasisRouter):
        raise TypeError(
            "settle_group_of_contracts requires a BasisRouter; for a single "
            "Basis use the per-model settlement or group a native measurement")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period}")
    if coverage_units is None:
        raise ValueError("coverage_units is required; pass 'count' or an array")
    if profitability is None:
        raise ValueError("profitability is required and must be explicit")
    if cohort == "issue_year":
        try:
            model_points.axis("issue_year")
        except KeyError:
            raise ValueError(
                "settle_group_of_contracts needs issue_date to derive the "
                "annual cohort; pass an explicit cohort array/column or set "
                "issue_date")

    state = _reconcile_state(model_points, inforce_state)
    parts = _partition_by_model(model_points, basis)
    if parts["PAA"].size:
        raise ValueError(
            "settle_group_of_contracts does not settle PAA: PAA has no "
            "CSM/floor, so a per-GoC algebra is meaningless; use paa.settle "
            "and sum by your own groupby for PAA books.")
    csm_models = [m for m in ("GMM", "VFA") if parts[m].size]
    if len(csm_models) > 1:
        raise ValueError(
            "settle_group_of_contracts settles one measurement model per "
            "book; this book mixes GMM and VFA. A group of contracts sits in "
            "one product, hence one model -- split the portfolio by model and "
            "settle each, then combine the group tables.")
    model = csm_models[0] if csm_models else "GMM"

    n_mp = model_points.n_mp
    prof = _settlement_axis(model_points, state, profitability, "profitability")
    if prof.shape != (n_mp,):
        raise ValueError("profitability must have one entry per model point")
    gids = _join_keys([
        _settlement_axis(model_points, state, portfolio, "portfolio"),
        _settlement_axis(model_points, state, cohort, "cohort"),
        prof,
    ], ["portfolio", "cohort", "profitability"])
    labels, inverse = np.unique(gids, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_groups = labels.shape[0]
    sizes = np.bincount(inverse, minlength=n_groups).astype(np.int64)

    lc_open = (np.asarray(state.prior_loss_component, dtype=np.float64)
               if state.prior_loss_component is not None
               else np.zeros(n_mp))
    csm_sum = np.bincount(inverse, weights=state.prior_csm, minlength=n_groups)
    lc_sum = np.bincount(inverse, weights=lc_open, minlength=n_groups)
    bad_xor = (csm_sum > 0.0) & (lc_sum > 0.0)
    if np.any(bad_xor):
        g = int(np.argmax(bad_xor))
        raise ValueError(
            f"group {labels[g]!r} has positive prior_csm and prior_loss_component; "
            "check GoC grouping/re-grouping before settlement")

    lock = np.asarray(state.lock_in_rate, dtype=np.float64)
    lock_mp = np.full(n_mp, float(lock)) if lock.ndim == 0 else lock
    group_lock = np.empty(n_groups, dtype=np.float64)
    for g in range(n_groups):
        vals = lock_mp[inverse == g]
        if not np.allclose(vals, vals[0], rtol=0.0, atol=1e-14):
            raise ValueError(
                f"group {labels[g]!r} mixes non-uniform lock_in_rate values; "
                "IFRS 17 B73 requires one cohort locked-in rate. Supply the "
                "cohort weighted-average rate per GoC.")
        group_lock[g] = vals[0]

    if isinstance(coverage_units, str):
        if coverage_units != "count":
            raise ValueError("coverage_units must be 'count' or an array")
        weights = np.ones(n_mp, dtype=np.float64)
    else:
        weights = np.asarray(coverage_units, dtype=np.float64)
        if weights.shape != (n_mp,):
            raise ValueError(
                f"coverage_units must have one entry per model point ({n_mp}), "
                f"got shape {weights.shape}")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("coverage_units must be finite and >= 0")

    # The B96(a) premium-experience split applies to both the GMM and VFA
    # paths (each routes the future leg into its own paragraph-45 algebra).
    pe_frac = np.asarray(premium_experience_future_fraction, dtype=np.float64)
    if pe_frac.ndim > 0 and pe_frac.shape != (n_mp,):
        raise ValueError(
            "premium_experience_future_fraction must be a scalar or one entry "
            f"per model point ({n_mp}), got shape {pe_frac.shape}")

    if model == "VFA":
        # The VFA mirror: group-sum the per-MP VFA settlement lines (each
        # carrying its own v_half / k_obs, so the group 45(b) fv_share is the
        # sum of the per-MP fv_shares) and apply the algebra + B119 release
        # once at group grain. lock_in_rate is a VFA state echo (group_lock,
        # uniform within the group like the GMM rate).
        vfa_pre = {name: np.zeros(n_groups, dtype=np.float64)
                   for name in (_VFA_GOC_SETTLEMENT_LINEAR
                                + _GOC_SETTLEMENT_UNIT_LINES)}
        av_by_mp = (np.asarray(state.account_value, dtype=np.float64)
                    if state.account_value is not None else None)
        vfa_router = _submodel_router(basis, "VFA")
        for seg_basis, seg_idx in _model_segments(vfa_router, model_points):
            for start in range(0, seg_idx.size, chunk_size):
                block = seg_idx[start:start + chunk_size]
                for g in np.unique(inverse[block]):
                    rows = block[inverse[block] == g]
                    sub_state = replace(state.subset(rows),
                                        lock_in_rate=group_lock[g])
                    frac_arg = (float(pe_frac) if pe_frac.ndim == 0
                                else pe_frac[rows])
                    mv = _settle_vfa(model_points.subset(rows), sub_state,
                                     seg_basis, period_months=period,
                                     premium_experience_future_fraction=frac_arg)
                    for name in _VFA_GOC_SETTLEMENT_LINEAR:
                        vfa_pre[name][g] += float(getattr(mv, name).sum())
                    for name in _GOC_SETTLEMENT_UNIT_LINES:
                        vfa_pre[name][g] += float(
                            (weights[rows] * getattr(mv, name)).sum())
        vfa_lines = _finalise_vfa_goc_settlement(vfa_pre)
        return VFAGoCSettlement(
            group_labels=labels, group_sizes=sizes, period_months=period,
            lock_in_rate=group_lock, model_points=model_points,
            group_inverse=inverse, lock_in_rate_by_mp=state.lock_in_rate,
            profitability_by_mp=prof, account_value_by_mp=av_by_mp,
            **vfa_lines)

    pre = {name: np.zeros(n_groups, dtype=np.float64)
           for name in _GOC_SETTLEMENT_LINEAR + _GOC_SETTLEMENT_UNIT_LINES}
    gmm_router = _submodel_router(basis, "GMM")
    for seg_basis, seg_idx in _model_segments(gmm_router, model_points):
        for start in range(0, seg_idx.size, chunk_size):
            block = seg_idx[start:start + chunk_size]
            for g in np.unique(inverse[block]):
                rows = block[inverse[block] == g]
                sub_state = replace(state.subset(rows), lock_in_rate=group_lock[g])
                frac_arg = (float(pe_frac) if pe_frac.ndim == 0
                            else pe_frac[rows])
                mv = _settle_gmm(
                    model_points.subset(rows), sub_state, seg_basis,
                    period_months=period,
                    premium_experience_future_fraction=frac_arg)
                for name in _GOC_SETTLEMENT_LINEAR:
                    pre[name][g] += float(getattr(mv, name).sum())
                for name in _GOC_SETTLEMENT_UNIT_LINES:
                    pre[name][g] += float((weights[rows] * getattr(mv, name)).sum())

    lines = _finalise_goc_settlement(pre)
    return GoCSettlement(
        group_labels=labels,
        group_sizes=sizes,
        period_months=period,
        lock_in_rate=group_lock,
        model_points=model_points,
        group_inverse=inverse,
        lock_in_rate_by_mp=state.lock_in_rate,
        profitability_by_mp=prof,
        **lines,
    )


@reconcile.register
def _(settlement: GoCSettlement) -> GMMSettlementReconciliation:
    a = settlement
    return GMMSettlementReconciliation(
        period_months=a.period_months,
        bel_opening=float(a.bel_opening.sum()),
        bel_interest=float(a.bel_interest.sum()),
        bel_release=float(-a.bel_release.sum()),
        bel_experience=float(a.bel_experience.sum()),
        bel_closing=float(a.bel_closing.sum()),
        ra_opening=float(a.ra_opening.sum()),
        ra_interest=float(a.ra_interest.sum()),
        ra_release=float(-a.ra_release.sum()),
        ra_experience=float(a.ra_experience.sum()),
        ra_closing=float(a.ra_closing.sum()),
        csm_opening=float(a.csm_opening.sum()),
        csm_accretion=float(a.csm_accretion.sum()),
        csm_experience_unlocking=float(a.csm_experience_unlocking.sum()),
        csm_premium_experience=float(a.csm_premium_experience.sum()),
        csm_investment_experience=float(a.csm_investment_experience.sum()),
        finance_wedge=float(a.finance_wedge.sum()),
        premium_experience_revenue=float(a.premium_experience_revenue.sum()),
        claims_experience=float(a.claims_experience.sum()),
        expense_experience=float(a.expense_experience.sum()),
        loss_component_finance=float(a.loss_component_finance.sum()),
        loss_component_amortised=float(-a.loss_component_amortised.sum()),
        loss_component_reversed=float(-a.loss_component_reversed.sum()),
        loss_component_recognised=float(a.loss_component_recognised.sum()),
        csm_release=float(-a.csm_release.sum()),
        csm_closing=float(a.csm_closing.sum()),
        loss_component_opening=float(a.loss_component_opening.sum()),
        loss_component_closing=float(a.loss_component_closing.sum()),
        lic_opening=float(a.lic_opening.sum()),
        claims_incurred=float(a.claims_incurred.sum()),
        lic_finance=float(a.lic_finance.sum()),
        claims_paid=float(-a.claims_paid.sum()),
        lic_closing=float(a.lic_closing.sum()),
    )


@write_measurement.register
def _(settlement: GoCSettlement, path, *, ids=None):
    cols = {
        "group_label": settlement.group_labels,
        "group_size": settlement.group_sizes,
        **{name: getattr(settlement, name)
           for name in (_GOC_SETTLEMENT_LINEAR + _GOC_SETTLEMENT_UNIT_LINES
                        + _GOC_SETTLEMENT_NONLINEAR)},
        "lock_in_rate": settlement.lock_in_rate,
        "measurement_basis": [settlement.measurement_basis]
                             * settlement.group_labels.shape[0],
    }
    _write_measurement_columns(cols, path, ids)


@reconcile.register
def _(settlement: VFAGoCSettlement) -> VFASettlementReconciliation:
    """The paragraph-45 settlement table of a VFA group-of-contracts movement
    -- run-off rows display-negated, exactly like the per-MP reconciliation."""
    a = settlement
    return VFASettlementReconciliation(
        period_months=a.period_months,
        bel_opening=float(a.bel_opening.sum()),
        bel_interest=float(a.bel_interest.sum()),
        bel_release=float(-a.bel_release.sum()),
        bel_experience=float(a.bel_experience.sum()),
        bel_closing=float(a.bel_closing.sum()),
        ra_opening=float(a.ra_opening.sum()),
        ra_interest=float(a.ra_interest.sum()),
        ra_release=float(-a.ra_release.sum()),
        ra_experience=float(a.ra_experience.sum()),
        ra_closing=float(a.ra_closing.sum()),
        csm_opening=float(a.csm_opening.sum()),
        csm_accretion=float(a.csm_accretion.sum()),
        csm_fv_share=float(a.csm_fv_share.sum()),
        csm_future_service=float(a.csm_future_service.sum()),
        csm_premium_experience=float(a.csm_premium_experience.sum()),
        premium_experience_revenue=float(a.premium_experience_revenue.sum()),
        csm_investment_experience=float(a.csm_investment_experience.sum()),
        claims_experience=float(a.claims_experience.sum()),
        expense_experience=float(a.expense_experience.sum()),
        loss_component_finance=float(a.loss_component_finance.sum()),
        loss_component_amortised=float(-a.loss_component_amortised.sum()),
        loss_component_reversed=float(-a.loss_component_reversed.sum()),
        loss_component_recognised=float(a.loss_component_recognised.sum()),
        csm_release=float(-a.csm_release.sum()),
        csm_closing=float(a.csm_closing.sum()),
        loss_component_opening=float(a.loss_component_opening.sum()),
        loss_component_closing=float(a.loss_component_closing.sum()),
        lic_opening=float(a.lic_opening.sum()),
        claims_incurred=float(a.claims_incurred.sum()),
        lic_finance=float(a.lic_finance.sum()),
        claims_paid=float(-a.claims_paid.sum()),
        lic_closing=float(a.lic_closing.sum()),
    )


@write_measurement.register
def _(settlement: VFAGoCSettlement, path, *, ids=None):
    cols = {
        "group_label": settlement.group_labels,
        "group_size": settlement.group_sizes,
        **{name: getattr(settlement, name)
           for name in (_VFA_GOC_SETTLEMENT_LINEAR + _GOC_SETTLEMENT_UNIT_LINES
                        + _VFA_GOC_SETTLEMENT_NONLINEAR)},
        "lock_in_rate": settlement.lock_in_rate,
        "measurement_basis": [settlement.measurement_basis]
                             * settlement.group_labels.shape[0],
    }
    _write_measurement_columns(cols, path, ids)


def _pad_curve(curve, n_time):
    """Flat-fill a segment's 1-D discount curve to the partition horizon.

    The representative curve of a group is its longest-horizon contract's curve;
    a contract shorter than the partition horizon repeats its last factor past
    maturity (a flat curve -> zero forward rate), matching the stitch's padding so
    the chunked representative equals the in-memory ``_per_group_bom`` choice.
    Returns ``(n_time + 1,)``.
    """
    curve = np.asarray(curve)
    out = np.empty(n_time + 1)
    out[:curve.shape[0]] = curve
    out[curve.shape[0]:] = curve[-1]
    return out


def _pad_mid(mid, n_time):
    """Flat-fill a segment's 1-D mid-of-month curve to ``(n_time,)`` (GMM only).

    Matches the stitch's ``discount_factor_mid`` padding: repeat the last factor past
    maturity, or ``1.0`` for a degenerate empty curve.
    """
    mid = np.asarray(mid)
    out = np.empty(n_time)
    lm = mid.shape[0]
    out[:lm] = mid
    if lm < n_time:
        out[lm:] = mid[-1] if lm > 0 else 1.0
    return out


def _resolve_rep_curves(segs, seg_bom, seg_mid, model, inverse, live,
                        n_groups, n_time, labels):
    """Resolve the per-group representative discount curve after the accumulate
    pass, from the globally-collected live horizons -- the contract's 2-pass
    correctness intent without a second measurement.

    A group must sit in one discount curve, spanning its longest **live** horizon
    (the last in-force month, ``cashflows.inforce > _INFORCE_EPS`` -- identical to
    :func:`grouping._per_group_bom`, *not* the contract boundary, so a count=0 or
    early-terminal contract's curve never falsely drives the choice). Each routing
    segment carries one curve (``seg_bom[s]``, the longest collected); for each
    group the representative is the curve of its longest-live contributing
    segment, and every other live contributor is reconciled to it over the
    overlapping live horizon (raising on a genuine mismatch). A segment with no
    live rows in a group contributes nothing -- as in ``_per_group_bom``, a dead
    row is compared only at column 0 (trivially equal).

    A group with no live row at all (every contract count=0) keeps the curve of
    its **lowest-index** contributing contract, matching ``_per_group_bom``'s
    ``argmax`` tie-break (it returns the first row when all live horizons are -1),
    so the public ``discount_factor_bom`` is identical too -- not a flat placeholder.

    Returns ``(rep_bom, rep_mid)`` -- ``(n_groups, n_time+1)`` and, for GMM,
    ``(n_groups, n_time)`` (``None`` for VFA, which has no ``discount_factor_mid``).
    """
    rep_bom = np.empty((n_groups, n_time + 1))
    rep_mid = np.empty((n_groups, n_time)) if model == "GMM" else None
    rep_h = np.full(n_groups, -1, dtype=np.int64)
    # Lowest contributing row index per group + its segment -- the all-dead
    # fallback representative (matches _per_group_bom's first-row tie-break).
    first_idx = np.full(n_groups, np.iinfo(np.int64).max, dtype=np.int64)
    first_seg = np.full(n_groups, -1, dtype=np.int64)
    pads = []
    for s, (_basis, idx) in enumerate(segs):
        pad_bom = _pad_curve(seg_bom[s], n_time)
        pad_mid = _pad_mid(seg_mid[s], n_time) if model == "GMM" else None
        pads.append((pad_bom, pad_mid))
        seg_groups = inverse[idx]
        seg_live = live[idx]
        for g in np.unique(seg_groups):
            mask = seg_groups == g
            gmin = int(idx[mask].min())
            if gmin < first_idx[g]:
                first_idx[g] = gmin
                first_seg[g] = s
            h_sg = int(seg_live[mask].max())              # group's live horizon in s
            if h_sg < 0:
                continue                                  # all dead here -- ignore
            if rep_h[g] < 0:
                rep_bom[g] = pad_bom
                if pad_mid is not None:
                    rep_mid[g] = pad_mid
                rep_h[g] = h_sg
            else:
                lcmp = min(h_sg, int(rep_h[g])) + 2        # +2 mirrors live+2 mask
                if not np.allclose(pad_bom[:lcmp], rep_bom[g][:lcmp]):
                    raise ValueError(
                        f"group {labels[g]!r} mixes model points with different "
                        "discount curves -- a group must sit in one portfolio "
                        "(basis). Split it by basis before grouping.")
                if h_sg > rep_h[g]:
                    rep_bom[g] = pad_bom
                    if pad_mid is not None:
                        rep_mid[g] = pad_mid
                    rep_h[g] = h_sg
    for g in np.nonzero(rep_h < 0)[0]:                    # all-dead groups
        pad_bom, pad_mid = pads[first_seg[g]]
        rep_bom[g] = pad_bom
        if pad_mid is not None:
            rep_mid[g] = pad_mid
    return rep_bom, rep_mid


def _aggregate_groups_model(sub_mp, sub_router, model, group_ids, chunk_size):
    """Chunked per-group aggregate of one model's partition.

    ``group_ids`` is the ``(n_sub,)`` composite group label per model point. The
    label space is global (``np.unique``), so a group spans chunks; the additive
    fields (BEL / RA / cash flows / LIC, plus VFA fee + time value, PAA revenue +
    service expense + FCF) are summed within each block and accumulated across
    blocks, and the floor / CSM roll is applied **once** at the end on the
    fully-accumulated group -- never per chunk (which would be silently wrong).
    Returns the native grouped measurement (rows = groups), identical to
    ``group_of_contracts`` run on the in-memory full measurement.
    """
    labels, inverse = np.unique(group_ids, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_groups = labels.shape[0]
    sizes = np.bincount(inverse, minlength=n_groups)
    n_time = int(np.asarray(sub_mp.contract_boundary_months).max())
    measure_full_one = _MEASURE_FULL[model]
    segs = _model_segments(sub_router, sub_mp)
    needs_curve = model in ("GMM", "VFA")
    # B119 coverage-unit discounting is an entity-level accounting policy, so
    # the grouped re-roll inherits it from the segments (uniform across them).
    discount_units = bool(segs[0][0].coverage_unit_discount) if segs else False

    # Accumulators -- bounded O(n_groups x n_time), no per-model-point row.
    bel = np.zeros((n_groups, n_time + 1))      # GMM/VFA BEL, PAA LRC
    ra = np.zeros((n_groups, n_time + 1))       # GMM/VFA only
    lic_path = np.zeros((n_groups, n_time + 1))
    revenue = np.zeros((n_groups, n_time))      # PAA only
    service_expense = np.zeros((n_groups, n_time))
    time_value = np.zeros(n_groups)             # VFA only
    variable_fee = np.zeros(n_groups)
    fcf = np.zeros(n_groups)                    # PAA only
    cf_acc = {name: np.zeros((n_groups, n_time)) for name in _CF_STREAMS_2D}
    maturity_cf = np.zeros(n_groups)
    maturity_survivors = np.zeros(n_groups)

    # Curve metadata collected during the pass (GMM/VFA): each row's live horizon
    # (last in-force month) and, per segment, its longest discount curve. The
    # representative per group is resolved after the pass, from global knowledge.
    live = np.full(sub_mp.n_mp, -1, dtype=np.int64) if needs_curve else None
    seg_bom = [None] * len(segs)
    seg_mid = [None] * len(segs)

    for s, (basis, idx) in enumerate(segs):
        for start in range(0, idx.size, chunk_size):
            block = idx[start:start + chunk_size]
            m = measure_full_one(sub_mp.subset(block), basis)
            red = _GroupReducer(inverse[block], n_groups)
            if model == "PAA":
                _add_leading(bel, red.sum(m.lrc_path))
                _add_leading(revenue, red.sum(m.revenue))
                _add_leading(service_expense, red.sum(m.service_expense))
                fcf += red.sum(m.fcf)
            else:
                _add_leading(bel, red.sum(m.bel_path))
                _add_leading(ra, red.sum(m.ra_path))
                if model == "VFA":
                    time_value += red.sum(m.time_value)
                    variable_fee += red.sum(m.variable_fee)
            _add_leading(lic_path, red.sum(m.lic_path))
            cf = m.cashflows
            for name in _CF_STREAMS_2D:
                _add_leading(cf_acc[name], red.sum(getattr(cf, name)))
            maturity_cf += red.sum(cf.maturity_cf)
            maturity_survivors += red.sum(cf.maturity_survivors)
            if needs_curve:
                inforce = np.asarray(cf.inforce)
                cols = np.arange(inforce.shape[1])
                live[block] = np.where(
                    inforce > _INFORCE_EPS, cols[None, :], -1).max(axis=1)
                bom = np.asarray(m.discount_factor_bom)
                if bom.ndim == 2:                  # single-basis blocks are 1-D
                    bom = bom[0]
                if seg_bom[s] is None or bom.shape[0] > seg_bom[s].shape[0]:
                    seg_bom[s] = bom
                    if model == "GMM":
                        seg_mid[s] = np.asarray(m.discount_factor_mid)

    rep_bom = rep_mid = None
    if needs_curve:
        rep_bom, rep_mid = _resolve_rep_curves(
            segs, seg_bom, seg_mid, model, inverse, live, n_groups, n_time,
            labels)

    grouped_cf = Cashflows(
        maturity_cf=maturity_cf, maturity_survivors=maturity_survivors,
        **cf_acc)
    if model == "GMM":
        return _finalise_gmm_group(
            bel, ra, grouped_cf, lic_path, rep_bom, rep_mid, labels, sizes,
            discount_units)
    if model == "VFA":
        return _finalise_vfa_group(
            bel, ra, grouped_cf, lic_path, time_value, variable_fee, rep_bom,
            labels, sizes, discount_units)
    return _finalise_paa_group(
        bel, revenue, service_expense, lic_path, fcf, grouped_cf, labels, sizes)


def _add_leading(acc, block_sum):
    """Add a block's group-sum into the leading slice of the accumulator.

    A block's trajectory is only as long as its own contracts' horizon (a
    contract carries nothing past its term); add it into ``acc[:, :width]`` so
    shorter blocks land in the leading columns of the partition-wide buffer.
    """
    acc[:, :block_sum.shape[1]] += block_sum


def _measure_groups(model_points, basis, label_fn, chunk_size):
    """Shared driver for :func:`measure_group` / :func:`measure_group_of_contracts`.

    ``label_fn(model, sub_mp, idx)`` returns the ``(n_sub,)`` group label per
    model point for one model's partition (``idx`` the partition's rows in the
    full portfolio, for a precomputed per-MP override). Routes a single declared
    model straight through (no partition), otherwise partitions by measurement
    model and aggregates each present model.
    """
    if not isinstance(basis, BasisRouter):
        raise TypeError(
            "this per-group aggregate requires a BasisRouter (a routed, possibly "
            "mixed-model portfolio); for a single Basis use fcf.group_of_"
            "contracts on a single-model measurement")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    declared = {basis.measurement_model_of(k) for k in basis.segments}
    slots = {}
    if len(declared) == 1:
        (model,) = declared
        idx = np.arange(model_points.n_mp, dtype=np.int64)
        gids = label_fn(model, model_points, idx)
        slots[model.lower()] = _aggregate_groups_model(
            model_points, basis, model, gids, chunk_size)
        return PortfolioGroups(**slots)
    parts = _partition_by_model(model_points, basis)
    covered = sum(part.size for part in parts.values())
    if covered != model_points.n_mp:
        raise ValueError(
            f"model partition covers {covered} of {model_points.n_mp} rows")
    for model in ("GMM", "PAA", "VFA"):
        idx = parts[model]
        if idx.size:
            sub_mp = model_points.subset(idx)
            gids = label_fn(model, sub_mp, idx)
            slots[model.lower()] = _aggregate_groups_model(
                sub_mp, _submodel_router(basis, model), model, gids, chunk_size)
    return PortfolioGroups(**slots)


def measure_group(model_points: ModelPoints, basis, by, *,
                   chunk_size: int = _CHUNK_SIZE) -> PortfolioGroups:
    """Scalable group aggregation of a mixed-model portfolio on any axis.

    The chunked, memory-bounded form of :func:`fcf.group`: each model's rows are
    aggregated to the ``by`` axis (re-flooring the CSM / loss component on each
    group's fulfilment cash flows, ``CSM(sum FCF)``), computed in ``chunk_size``
    row-blocks so peak memory is ``O(chunk_size x n_time)`` plus the
    ``O(n_groups x n_time)`` accumulator -- it works where holding the
    per-model-point ``measure(full=True)`` would OOM. ``by`` is one of a single
    axis name, a list of axis names and/or precomputed ``(n_mp,)`` label arrays,
    or a single ``(n_mp,)`` label array (as :func:`fcf.group`). For the IFRS 17
    unit of account use :func:`measure_group_of_contracts`, the preset on
    ``portfolio x annual cohort x profitability``.
    """
    full_ids = _resolve_full_group_ids(model_points, by)
    if full_ids.shape != (model_points.n_mp,):
        raise ValueError(
            f"group ids must have one entry per model point "
            f"({model_points.n_mp}), got shape {full_ids.shape}")
    return _measure_groups(
        model_points, basis, lambda model, sub_mp, idx: full_ids[idx],
        chunk_size)


def measure_group_of_contracts(model_points: ModelPoints, basis, *, portfolio: str = "product",
                cohort: str = "issue_year", profitability=None,
                chunk_size: int = _CHUNK_SIZE) -> PortfolioGroups:
    """Scalable group-of-contracts aggregation -- the IFRS 17 unit of account.

    The chunked, memory-bounded form of :func:`fcf.group_of_contracts`: the
    portfolio (paragraph 14) x annual cohort (22) x profitability (16) grouping,
    re-flooring the CSM / loss component on each group's fulfilment cash flows
    (``CSM(sum FCF)``, unlike :func:`measure_aggregate` which sums each contract's
    already-floored CSM). Computed in ``chunk_size`` row-blocks so it works where
    holding the per-model-point ``measure(full=True)`` would OOM.

    At initial recognition, under any paragraph-16-compliant grouping a group
    never mixes inception-FCF signs, so the re-floor equals the per-model-point
    floor sum -- ``measure_group_of_contracts`` and :func:`measure_aggregate` report the same
    totals; ``measure_group_of_contracts``'s value is the **per-group rows** (disclosure /
    roll-forward / the paragraph-44 foundation).

    Arguments mirror :func:`fcf.group_of_contracts`: ``portfolio`` / ``cohort``
    name the axis columns; ``profitability`` is ``None`` (derive the onerous /
    remaining split per model point from the inception loss component, one rule
    across GMM / PAA / VFA), a column name (a locked classification, paragraph
    24), or a precomputed ``(n_mp,)`` array (e.g. the three-way split).

    Unlike :func:`fcf.group_of_contracts`, a missing ``issue_date`` with the
    default cohort is **rejected**, not silently collapsed to one cohort: a silent
    collapse would mutualise across annual cohorts (paragraph 22) invisibly at
    settlement scale. ``issue_age`` / ``term_months`` are never a cohort
    substitute.
    """
    if not isinstance(basis, BasisRouter):
        raise TypeError(
            "measure_group_of_contracts requires a BasisRouter (a routed, possibly mixed-model "
            "portfolio); for a single Basis use fcf.group_of_contracts on a "
            "single-model measurement")
    if cohort == "issue_year":
        try:
            model_points.axis("issue_year")
        except KeyError:
            raise ValueError(
                "measure_group_of_contracts needs issue_date to derive the annual cohort "
                "(paragraph 22); it is not set. Unlike group_of_contracts, the "
                "scalable aggregate does not silently fall back to a single "
                "cohort -- set issue_date, or pass an explicit cohort column. "
                "issue_age / term_months are not a cohort substitute.")
    prof_override = None
    if profitability is not None and not isinstance(profitability, str):
        prof_override = np.asarray(profitability)
        if prof_override.shape != (model_points.n_mp,):
            raise ValueError(
                f"profitability must have one entry per model point "
                f"({model_points.n_mp}), got shape {prof_override.shape}")

    def label_fn(model, sub_mp, idx):
        portfolio_arr = np.asarray(sub_mp.axis(portfolio))
        cohort_arr = np.asarray(sub_mp.axis(cohort))
        if profitability is None:
            loss = _per_mp_loss_component(
                sub_mp, _model_router(basis, model), model, chunk_size)
            prof = np.where(loss > 0.0, "onerous", "remaining")
        elif isinstance(profitability, str):
            prof = np.asarray(sub_mp.axis(profitability))
        else:
            prof = prof_override[idx]
        return _join_keys([portfolio_arr, cohort_arr, prof], [None, None, None])

    return _measure_groups(model_points, basis, label_fn, chunk_size)


def _model_router(basis, model):
    """The single-model sub-router for ``model`` -- the whole router when it is
    already single-model (the short-circuit), else its restriction."""
    declared = {basis.measurement_model_of(k) for k in basis.segments}
    return basis if len(declared) == 1 else _submodel_router(basis, model)


def _per_mp_loss_component(sub_mp, sub_router, model, chunk_size):
    """Per-model-point inception loss component, chunked headline-only.

    The default profitability axis (paragraph 16's onerous test) is each
    contract's standalone ``loss_component > 0`` at inception -- a headline field,
    so this measures full=False in ``chunk_size`` blocks (``O(n_mp)`` retained,
    not ``O(n_mp x n_time)``). One rule, one field, across GMM / PAA / VFA.
    """
    measure_one = _HEADLINE_MEASURE[model]
    out = np.empty(sub_mp.n_mp)
    for basis, idx in _model_segments(sub_router, sub_mp):
        for start in range(0, idx.size, chunk_size):
            block = idx[start:start + chunk_size]
            m = measure_one(sub_mp.subset(block), basis, full=False)
            out[block] = m.loss_component
    return out


def _resolve_full_group_ids(model_points, by):
    """Resolve ``by`` to a ``(n_mp,)`` group-label array over the full portfolio.

    Mirrors :func:`fcf.group`'s ``by`` handling (a name, a list of names and/or
    arrays, or a single array), resolved against the full model points so the
    per-model partitions subset it consistently.
    """
    n_mp = model_points.n_mp

    def axis(name):
        return np.asarray(model_points.axis(name))

    if isinstance(by, str):
        return axis(by)
    if isinstance(by, (list, tuple)):
        cols = []
        for b in by:
            col = axis(b) if isinstance(b, str) else np.asarray(b)
            # Validate each precomputed array here, before _join_keys: a length-1
            # (or otherwise short) array would broadcast in np.char.add and pass
            # the final (n_mp,) check while silently tagging every row the same.
            if col.shape != (n_mp,):
                raise ValueError(
                    f"group axis array must have one entry per model point "
                    f"({n_mp}), got shape {col.shape}")
            cols.append(col)
        names = [b if isinstance(b, str) else None for b in by]
        if len(cols) == 1:
            return cols[0]
        return _join_keys(cols, names)
    return np.asarray(by)


#: Per model: the single-Basis headline-only measure, for the default
#: profitability axis (loss component at inception).
_HEADLINE_MEASURE = {
    "GMM": _measure_gmm, "PAA": measure_paa, "VFA": measure_vfa}


# ---------------------------------------------------------------------------
# In-memory grouping of a PortfolioMeasurement -- compose like the leaf models.
#
# fcf.group / fcf.group_of_contracts already dispatch on the native leaf
# measurements; register the container here (not in grouping.py) so the import
# stays one-way: portfolio depends on grouping, never the reverse. With these
# arms a caller groups a measured mixed portfolio the same way as a single
# model -- group_of_contracts(portfolio.measure(..., full=True)) -- instead of
# reaching into pm.gmm.measurement slot by slot. (At scale, where the full
# per-model-point measurement would not fit in memory, use the chunked
# measure_group / measure_group_of_contracts instead.)
# ---------------------------------------------------------------------------

def _subset_by(by, index):
    """Subset a ``by`` spec to one model slot's rows.

    Names pass through (they resolve against the slot measurement's stamped model
    points); precomputed ``(n_mp,)`` arrays are indexed by the slot's original row
    positions so each slot sees its own rows in order.
    """
    if isinstance(by, str):
        return by
    if isinstance(by, (list, tuple)):
        return [b if isinstance(b, str) else np.asarray(b)[index] for b in by]
    return np.asarray(by)[index]


@group.register
def _(measurement: PortfolioMeasurement, by) -> PortfolioGroups:
    """Group each model's slice and keep them in a :class:`PortfolioGroups`.

    The container analogue of :func:`fcf.group`: a BEL and an LRC are never
    pooled, so each model slot is grouped on its own native measurement and the
    per-model grouped results are returned side by side.
    """
    slots = {}
    for name in ("gmm", "paa", "vfa"):
        mm = getattr(measurement, name)
        if mm is not None:
            slots[name] = group(mm.measurement, _subset_by(by, mm.index))
    return PortfolioGroups(**slots)


@group_of_contracts.register
def _(measurement: PortfolioMeasurement, *, portfolio: str = "product",
      cohort: str = "issue_year", profitability=None) -> PortfolioGroups:
    """IFRS 17 group-of-contracts grouping of a measured mixed portfolio.

    The container analogue of :func:`fcf.group_of_contracts`: each model slot is
    grouped on ``portfolio x annual cohort x profitability`` and kept separate.
    For a book that fits in memory this equals the chunked
    :func:`measure_group_of_contracts` run on the same model points and router.
    """
    prof_is_array = profitability is not None and not isinstance(profitability, str)
    prof_arr = np.asarray(profitability) if prof_is_array else None
    slots = {}
    for name in ("gmm", "paa", "vfa"):
        mm = getattr(measurement, name)
        if mm is not None:
            prof = prof_arr[mm.index] if prof_is_array else profitability
            slots[name] = group_of_contracts(
                mm.measurement, portfolio=portfolio, cohort=cohort,
                profitability=prof)
    return PortfolioGroups(**slots)


# ---------------------------------------------------------------------------
# Reporting a measured portfolio -- per-model IFRS 17 reports in one container.
#
# fcf.report dispatches on the measurement type; register the portfolio
# containers here (one-way import: portfolio depends on report). A GMM, PAA and
# VFA report are never merged -- their revenue / finance-expense lines measure
# different liabilities -- so each model's Report is kept in its own slot, the
# same per-model separation as the measurement containers.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PortfolioReport:
    """Result of :func:`fcf.report` on a portfolio container: one
    :class:`~fastcashflow.report.Report` per model present (``None`` when absent),
    keyed by model. Accepts a :class:`PortfolioMeasurement` (per-model-point) or a
    :class:`PortfolioGroups` (grouped); a BEL and an LRC report are never pooled.
    """

    gmm: "Report | None" = None
    paa: "Report | None" = None
    vfa: "Report | None" = None


def _portfolio_report(measurement) -> PortfolioReport:
    """Report each present model slot on its own native measurement.

    A :class:`PortfolioMeasurement` slot is a :class:`ModelMeasurement` (carrying
    the per-MP index); a :class:`PortfolioGroups` slot is the grouped measurement
    directly. Unwrap the former, pass the latter through.
    """
    slots = {}
    for name in ("gmm", "paa", "vfa"):
        m = getattr(measurement, name)
        if m is not None:
            slots[name] = report(m.measurement
                                 if isinstance(m, ModelMeasurement) else m)
    return PortfolioReport(**slots)


report.register(PortfolioMeasurement, _portfolio_report)
report.register(PortfolioGroups, _portfolio_report)


# ---------------------------------------------------------------------------
# Period-close roll-forward / reconciliation of a measured portfolio.
#
# fcf.roll_forward (singledispatch) and fcf.reconcile (singledispatch) get a
# container arm here (one-way import: portfolio depends on movement). Each model
# rolls / reconciles on its own native measurement; the per-model results stay
# in their own slots -- a GMM CSM movement and a PAA LRC movement are never
# merged.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PortfolioMovements:
    """Result of :func:`fcf.roll_forward` on a portfolio container: one list of
    period movements per model present (``None`` when absent), keyed by model.
    Each slot is the model's own movement list (``list[PeriodMovement]`` /
    ``list[PAAPeriodMovement]`` / ``list[VFAPeriodMovement]``) -- a GMM CSM
    movement and a PAA LRC movement are never merged. Feed it to
    :func:`fcf.reconcile` for a :class:`PortfolioReconciliation`.
    """

    gmm: "list | None" = None
    paa: "list | None" = None
    vfa: "list | None" = None


@dataclass(frozen=True, slots=True)
class PortfolioReconciliation:
    """Result of :func:`fcf.reconcile` on a :class:`PortfolioMovements`: one list
    of reconciliation tables per model present (``None`` when absent), keyed by
    model (``list[Reconciliation]`` / ``list[PAAReconciliation]`` /
    ``list[VFAReconciliation]``).
    """

    gmm: "list | None" = None
    paa: "list | None" = None
    vfa: "list | None" = None


def _portfolio_roll_forward(measurement, period_months: int = 12, *,
                            revised=None, revised_at=None, actual_inforce=None,
                            experience_at=None) -> PortfolioMovements:
    """Roll each present model slot forward on its own native measurement.

    The revision / experience options are a single-GMM-measurement feature (they
    need a matching revised book), so they are rejected on the container -- roll
    ``pm.gmm.measurement`` directly for those. A :class:`PortfolioMeasurement`
    slot is a :class:`ModelMeasurement` (unwrap ``.measurement``); a
    :class:`PortfolioGroups` slot is the grouped measurement directly.
    """
    if any(opt is not None for opt in
           (revised, revised_at, actual_inforce, experience_at)):
        raise ValueError(
            "the revision / experience options apply to a single GMM "
            "measurement; roll pm.gmm.measurement forward directly for those")
    slots = {}
    for name in ("gmm", "paa", "vfa"):
        m = getattr(measurement, name)
        if m is not None:
            native = m.measurement if isinstance(m, ModelMeasurement) else m
            slots[name] = roll_forward(native, period_months)
    return PortfolioMovements(**slots)


roll_forward.register(PortfolioMeasurement, _portfolio_roll_forward)
roll_forward.register(PortfolioGroups, _portfolio_roll_forward)


@reconcile.register
def _(movements: PortfolioMovements) -> PortfolioReconciliation:
    """Reconcile each model's movement list, keeping the per-model split."""
    slots = {}
    for name in ("gmm", "paa", "vfa"):
        mv = getattr(movements, name)
        if mv is not None:
            slots[name] = reconcile(mv)
    return PortfolioReconciliation(**slots)
