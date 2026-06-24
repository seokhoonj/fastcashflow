"""CSM recognition -- IFRS 17 paragraph-109 maturity-band disclosure.

The closing CSM allocated to maturity bands by expected coverage-unit
recognition. Shared by the GMM (paragraph 44) and VFA (paragraph 45) settlement
schedules -- the para-109 allocation is identical, only each model's public
``recognition_schedule`` (its own source settle) differs. CSM-domain code (PAA
has no CSM), kept as a distinct submodule of the shared measurement layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class CSMRecognitionSchedule:
    """IFRS 17 paragraph-109 disclosure: the closing CSM allocated to maturity
    bands by expected coverage-unit recognition.

    ``band_edges_months`` are the band boundaries in months from the valuation
    date (default 12 / 36 / 60, the four-band disclosure axis); the bands are
    ``[0, e0), [e0, e1), ..., [e_last, end)``. ``csm[b]`` is the closing CSM
    expected to be recognised in band ``b`` -- allocated by each contract's
    forward coverage-unit fraction, so the bands SUM TO ``closing_csm``. It is
    an allocation of the remaining balance, not the accreted nominal release;
    the coverage-unit proxy is the in-force count, undiscounted, matching the
    B119 amortisation kernel, so the schedule tracks the actual release pattern.
    """

    band_edges_months: tuple
    csm: FloatArray              # (n_bands,) -- sums to closing_csm
    closing_csm: float

    @property
    def labels(self) -> tuple:
        """Band labels, in years when the edges are whole years."""
        def fmt(m):
            return f"{m // 12}y" if m % 12 == 0 else f"{m}m"
        edges = self.band_edges_months
        out = [f"<= {fmt(edges[0])}"]
        out += [f"{fmt(lo)} - {fmt(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]
        out.append(f"{fmt(edges[-1])} +")
        return tuple(out)


def _validate_band_edges(band_edges_months) -> tuple:
    """Coerce / validate paragraph-109 band edges -- strictly ascending
    positive integer months from the valuation date. Shared by the GMM and VFA
    recognition schedules so the edge contract cannot drift between them."""
    edges = tuple(int(e) for e in band_edges_months)
    if (not edges or any(e <= 0 for e in edges)
            or list(edges) != sorted(set(edges))):
        raise ValueError(
            "band_edges_months must be strictly ascending positive integers "
            f"(months from the valuation date), got {band_edges_months!r}")
    return edges


def _build_recognition_schedule(csm_closing, inforce, em, boundary, edges):
    """Allocate the per-MP closing CSM to maturity bands by each contract's
    forward coverage-unit (in-force) fraction; the bands sum to the closing
    CSM. Onerous contracts (CSM <= 0) contribute nothing. Shared by the GMM
    (paragraph 44) and VFA (paragraph 45) settlement schedules -- the
    paragraph-109 allocation is identical, only the source settle differs."""
    bounds = (0,) + edges
    n_bands = len(bounds)
    band = np.zeros(n_bands)
    for i in range(csm_closing.shape[0]):
        csm_i = float(csm_closing[i])
        if csm_i <= 0.0:                  # onerous / no CSM -> nothing to recognise
            continue
        cu = inforce[i, em[i]:boundary[i]]    # forward coverage units (in-force)
        total = cu.sum()
        if total <= 0.0:                 # guarded by settle (em < boundary); belt-and-braces
            continue
        for b in range(n_bands):
            lo = bounds[b]
            hi = bounds[b + 1] if b + 1 < n_bands else cu.shape[0]
            band[b] += csm_i * cu[lo:hi].sum() / total
    return CSMRecognitionSchedule(
        band_edges_months=edges, csm=band,
        closing_csm=float(csm_closing[csm_closing > 0.0].sum()))
