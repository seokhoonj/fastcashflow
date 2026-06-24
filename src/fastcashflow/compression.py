"""Model-point compression -- shrink a large seriatim book to a small set of
representative policies that reproduce its valuation.

A portfolio of ``n`` individual contracts is expensive to project, especially
under many scenarios. Compression groups contracts whose *valuation behaviour*
is similar and replaces each group with one real representative policy, scaled up
to the group's total size. The compressed book runs through the same engine and
reproduces the full book's present values to a small error.

The method (a standard actuarial cluster compression, original implementation):

1. **Calibrate on outputs, not inputs.** Each contract is described by its
   per-policy BEL under the base run plus a few stresses (mortality up, lapse up,
   ...). Clustering on these calibration vectors preserves exactly the figures
   the model is built to produce -- the level and the sensitivities -- rather
   than merely grouping look-alike policies.
2. **Weight by importance.** Each contract's weight is its base liability
   ``|BEL|`` (count-weighted), so a few large contracts are never merged away.
3. **k-means (Lloyd + k-means++ seeding), importance-weighted**, on the
   standardised per-policy calibration vectors.
4. **One real representative per cluster** -- the contract nearest the cluster
   centroid -- rescaled so its ``count`` is the cluster's total count. A real
   policy (not an average) so the engine can re-project it unchanged.
5. **Validate.** Project the compressed book under the same base + stresses and
   compare aggregate PVs; the relative error per scenario is reported.

``compress(model_points, basis, n_clusters=...)`` returns a
:class:`CompressionResult` carrying the compressed ``model_points`` (``count``
rescaled), the per-contract cluster assignment, and the validation PVs. v1
calibrates through the GMM fast path (the large-book case); account-value (VFA)
and PAA calibration are a follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.basis import Basis
from fastcashflow.engine import measure
from fastcashflow.model_points import ModelPoints
from fastcashflow.solvency._engine import Stress, scale_lapse, scale_mortality

#: Default calibration stresses (besides the base run). A compact set spanning the
#: dominant biometric / behaviour sensitivities; the caller may pass its own.
def _default_stresses() -> tuple[Stress, ...]:
    return (scale_mortality(1.15), scale_lapse(1.15))

#: Assignment row-block: bound the (block x n_clusters) distance matrix.
_ASSIGN_BLOCK = 20_000


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """The outcome of :func:`compress`.

    ``model_points`` is the compressed book -- one representative contract per
    cluster, its ``count`` set to the cluster's total count -- ready to re-project
    through any ``fcf`` measure. ``cluster_id`` maps every original row to its
    cluster (``-1`` never occurs; empty clusters are dropped before labelling is
    renumbered). ``representative`` is the original-row index chosen for each
    kept cluster. ``scenario_names`` labels the calibration columns (``"base"``
    first). ``pv_full`` / ``pv_compressed`` are the aggregate BEL per scenario for
    the full and compressed books -- the compression's accuracy evidence.
    """

    model_points: ModelPoints
    cluster_id: IntArray
    representative: IntArray
    scenario_names: tuple[str, ...]
    pv_full: FloatArray
    pv_compressed: FloatArray

    @property
    def n_clusters(self) -> int:
        return self.representative.shape[0]

    @property
    def rel_error(self) -> FloatArray:
        """Relative aggregate-PV error per scenario, ``(compressed - full)/full``
        (zero-PV scenarios use a unit denominator)."""
        denom = np.where(np.abs(self.pv_full) > 0.0, self.pv_full, 1.0)
        return (self.pv_compressed - self.pv_full) / denom

    @property
    def max_abs_rel_error(self) -> float:
        """The worst absolute relative PV error across scenarios -- the headline
        accuracy number."""
        return float(np.max(np.abs(self.rel_error)))


def _counts(model_points: ModelPoints) -> FloatArray:
    """Per-policy count, defaulting ``None`` to 1 (a single policy per row)."""
    if model_points.count is None:
        return np.ones(model_points.n_mp, dtype=np.float64)
    return np.asarray(model_points.count, dtype=np.float64)


def _per_policy_bel(model_points: ModelPoints, basis: Basis,
                    count: FloatArray) -> tuple[FloatArray, float]:
    """The per-policy BEL ``(n_mp,)`` and the aggregate BEL for one (mp, basis):
    ``measure().bel`` is count-weighted, so per-policy is ``bel / count``."""
    bel = np.asarray(measure(model_points, basis, full=False).bel, dtype=np.float64)
    safe = np.where(count > 0.0, count, 1.0)
    return bel / safe, float(bel.sum())


def _calibration(model_points, basis, stresses, count):
    """Per-policy calibration matrix ``(n_mp, 1 + n_stress)`` (base + each stress),
    the importance weights ``|base BEL|`` (count-weighted), the aggregate base +
    stress PVs, and the scenario names."""
    names = ["base"]
    cols = []
    agg = []
    pp_base, agg_base = _per_policy_bel(model_points, basis, count)
    cols.append(pp_base)
    agg.append(agg_base)
    importance = np.abs(pp_base) * count                  # = |count-weighted base BEL|
    for s in stresses:
        mp_s, basis_s = s.apply(model_points, basis)
        pp_s, agg_s = _per_policy_bel(mp_s, basis_s, count)
        cols.append(pp_s)
        agg.append(agg_s)
        names.append(s.name)
    return (np.column_stack(cols), importance, np.asarray(agg, dtype=np.float64),
            tuple(names))


def _standardise(calibration: FloatArray) -> FloatArray:
    """z-score each calibration column so no single scenario dominates the
    Euclidean distance (a constant column collapses to zero, contributing nothing)."""
    mean = calibration.mean(axis=0)
    std = calibration.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return (calibration - mean) / std


def _assign(points: FloatArray, centres: FloatArray) -> IntArray:
    """Nearest-centre label per point, in row-blocks so the distance matrix stays
    ``O(block x k)``. Uses ``|x|^2 - 2 x.c + |c|^2`` (the cross term is the only
    per-pair work)."""
    n = points.shape[0]
    cn = (centres * centres).sum(axis=1)                  # (k,)
    labels = np.empty(n, dtype=np.int64)
    for start in range(0, n, _ASSIGN_BLOCK):
        block = points[start:start + _ASSIGN_BLOCK]
        d2 = (block * block).sum(axis=1)[:, None] - 2.0 * block @ centres.T + cn[None, :]
        labels[start:start + block.shape[0]] = d2.argmin(axis=1)
    return labels


def _kmeans_pp_init(points, weight, k, rng):
    """Importance-weighted k-means++ seeding: the first centre is sampled with
    probability proportional to the weight, each next with probability
    proportional to ``weight x distance^2`` to the nearest chosen centre -- spread
    seeds that still favour the heavy contracts."""
    n = points.shape[0]
    centres = np.empty((k, points.shape[1]), dtype=np.float64)
    p = weight / weight.sum() if weight.sum() > 0 else np.full(n, 1.0 / n)
    centres[0] = points[rng.choice(n, p=p)]
    d2 = ((points - centres[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        prob = weight * d2
        total = prob.sum()
        prob = prob / total if total > 0 else np.full(n, 1.0 / n)
        centres[c] = points[rng.choice(n, p=prob)]
        d2 = np.minimum(d2, ((points - centres[c]) ** 2).sum(axis=1))
    return centres


def _weighted_kmeans(points, weight, k, *, n_iter, seed):
    """Importance-weighted Lloyd's k-means. Returns ``(labels, centres)``. An empty
    cluster keeps its centre (it simply attracts no points and is dropped later)."""
    rng = np.random.default_rng(seed)
    centres = _kmeans_pp_init(points, weight, k, rng)
    labels = np.full(points.shape[0], -1, dtype=np.int64)
    for _ in range(n_iter):
        new_labels = _assign(points, centres)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            wc = weight[mask]
            total = wc.sum()
            if total > 0.0:
                centres[c] = (points[mask] * wc[:, None]).sum(axis=0) / total
    return labels, centres


def compress(model_points: ModelPoints, basis: Basis, *, n_clusters: int,
             stresses: tuple[Stress, ...] | None = None,
             n_iter: int = 50, seed: int = 0) -> CompressionResult:
    """Compress ``model_points`` to ``n_clusters`` representative policies.

    Clusters contracts by their per-policy BEL under the base run and each stress
    in ``stresses`` (default: mortality +15%, lapse +15%), weighting by base
    liability, and replaces each cluster with the contract nearest its centroid,
    its ``count`` set to the cluster's total. The compressed book reproduces the
    full book's aggregate PV under those scenarios to the error reported on the
    result. ``seed`` makes the k-means++ seeding (and thus the whole result)
    deterministic.

    ``n_clusters`` must be between 1 and ``model_points.n_mp``. Fewer occupied
    clusters than requested (identical contracts collapse) is allowed -- empty
    clusters are dropped, so the result may carry fewer than ``n_clusters`` rows.
    """
    n_mp = model_points.n_mp
    if not 1 <= n_clusters <= n_mp:
        raise ValueError(
            f"n_clusters must be in [1, n_mp={n_mp}], got {n_clusters}")
    stresses = _default_stresses() if stresses is None else tuple(stresses)
    count = _counts(model_points)
    calibration, importance, pv_full, names = _calibration(
        model_points, basis, stresses, count)
    points = _standardise(calibration)

    if n_clusters == n_mp:
        # No compression: each contract is its own cluster (an exact, useful
        # identity / base case), skipping k-means entirely.
        labels = np.arange(n_mp, dtype=np.int64)
        rep = np.arange(n_mp, dtype=np.int64)
        rep_count = count.copy()
    else:
        labels, centres = _weighted_kmeans(
            points, importance, n_clusters, n_iter=n_iter, seed=seed)
        rep, rep_count, labels = _representatives(points, labels, centres, count)

    compressed = replace(model_points.subset(rep), count=rep_count)
    pv_compressed = _aggregate_pvs(compressed, basis, stresses)
    return CompressionResult(
        model_points=compressed, cluster_id=labels, representative=rep,
        scenario_names=names, pv_full=pv_full, pv_compressed=pv_compressed)


def _representatives(points, labels, centres, count):
    """For each occupied cluster, the row nearest the centroid (the
    representative) and the cluster's total count. Empty clusters are dropped and
    ``labels`` is renumbered to the kept clusters (so ``cluster_id`` indexes the
    representatives array)."""
    k = centres.shape[0]
    reps = []
    rep_count = []
    remap = np.full(k, -1, dtype=np.int64)
    for c in range(k):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        d2 = ((points[members] - centres[c]) ** 2).sum(axis=1)
        remap[c] = len(reps)
        reps.append(int(members[d2.argmin()]))
        rep_count.append(float(count[members].sum()))
    return (np.asarray(reps, dtype=np.int64), np.asarray(rep_count, dtype=np.float64),
            remap[labels])


def _aggregate_pvs(model_points, basis, stresses) -> FloatArray:
    """Aggregate BEL per scenario (base + each stress) for validation -- the
    compressed-book counterpart of the calibration aggregates."""
    count = _counts(model_points)
    _, agg_base = _per_policy_bel(model_points, basis, count)
    agg = [agg_base]
    for s in stresses:
        mp_s, basis_s = s.apply(model_points, basis)
        _, agg_s = _per_policy_bel(mp_s, basis_s, count)
        agg.append(agg_s)
    return np.asarray(agg, dtype=np.float64)
