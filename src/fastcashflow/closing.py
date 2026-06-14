"""IFRS 17 period close -- assemble the statements from the settlement reconciliations.

The close ASSEMBLES the close pack from numbers the settlement layer already
produced and foots; it does not measure or recompute. :func:`close` takes the
per-model settlement reconciliations of a reporting period (the GMM / VFA / PAA
issued books and any reinsurance contracts held) and returns a
:class:`ClosePackage` -- the assembled statements plus the stacked reconciliation
detail, ready for the disclosure emitter to serialise.

The keystone is the statement of financial position (IFRS 17 paragraphs 78,
99-101): the closing carrying amount of insurance contracts, split into the
liability for remaining coverage excluding the loss component, the loss
component, and the liability for incurred claims, shown for contracts issued,
reinsurance contracts held, and the net. The split is model-specific because the
loss component sits in a different place in each measurement model:

* GMM / VFA -- the loss component is a notional sub-ledger WITHIN the liability
  for remaining coverage (paragraphs 49-52), so ``LRC = BEL + RA + CSM`` and
  ``LRC excluding LC = LRC - loss_component``.
* PAA -- the onerous loss is an ADDITIONAL liability recognised on top of the
  unearned-premium balance (paragraphs 57-58), so ``LRC excluding LC = lrc`` and
  ``LRC = lrc + loss_component``.
* Reinsurance held -- an ASSET for remaining coverage (``BEL + RA + CSM``,
  paragraph 82), no loss component (a reinsurance contract held cannot be
  onerous, paragraph 65) and no liability for incurred claims block; it enters
  the net position as a deduction from the issued liability (paragraph 78).
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from fastcashflow.disclosure import reconciliation_to_frame
from fastcashflow.movement import (
    GMMSettlementReconciliation, PAASettlementReconciliation,
    ReinsuranceSettlementReconciliation, VFASettlementReconciliation)

# The SoFP statement frame -- a presentation table (one row per kind x
# component), not the tidy disclosure spine. opening + change == closing per row.
_SOFP_SCHEMA = {
    "kind": pl.Utf8, "component": pl.Utf8,
    "opening": pl.Float64, "change": pl.Float64, "closing": pl.Float64,
}

_KIND_ISSUED = "Insurance contracts issued"
_KIND_REINSURANCE = "Reinsurance contracts held"
_KIND_NET = "Net"

_COMP_LRC = "LRC excluding loss component"
_COMP_LC = "Loss component"
_COMP_LIC = "Liability for incurred claims"
_COMP_TOTAL = "Total"


@dataclass(frozen=True, slots=True)
class _Components:
    """One reconciliation's SoFP position: the opening and closing balance of
    each carrying-amount component, plus which side of the net it falls on."""

    kind: str               # "issued" or "reinsurance"
    lrc_excl_lc_opening: float
    lc_opening: float
    lic_opening: float
    lrc_excl_lc_closing: float
    lc_closing: float
    lic_closing: float


def _components(recon) -> _Components:
    """The SoFP carrying-amount components of one settlement reconciliation,
    mapped per measurement model (see the module docstring)."""
    if isinstance(recon, (GMMSettlementReconciliation, VFASettlementReconciliation)):
        # Loss component is a sub-ledger within the LRC: LRC = BEL + RA + CSM,
        # LRC-excl-LC = LRC - loss component.
        lrc_open = recon.bel_opening + recon.ra_opening + recon.csm_opening
        lrc_close = recon.bel_closing + recon.ra_closing + recon.csm_closing
        return _Components(
            kind="issued",
            lrc_excl_lc_opening=lrc_open - recon.loss_component_opening,
            lc_opening=recon.loss_component_opening,
            lic_opening=recon.lic_opening,
            lrc_excl_lc_closing=lrc_close - recon.loss_component_closing,
            lc_closing=recon.loss_component_closing,
            lic_closing=recon.lic_closing,
        )
    if isinstance(recon, PAASettlementReconciliation):
        # Onerous loss is additive on top of the unearned-premium LRC.
        return _Components(
            kind="issued",
            lrc_excl_lc_opening=recon.lrc_opening,
            lc_opening=recon.loss_component_opening,
            lic_opening=recon.lic_opening,
            lrc_excl_lc_closing=recon.lrc_closing,
            lc_closing=recon.loss_component_closing,
            lic_closing=recon.lic_closing,
        )
    if isinstance(recon, ReinsuranceSettlementReconciliation):
        # Asset for remaining coverage; no loss component, no LIC block. The
        # loss-recovery component stays within the remaining-coverage asset.
        return _Components(
            kind="reinsurance",
            lrc_excl_lc_opening=recon.bel_opening + recon.ra_opening + recon.csm_opening,
            lc_opening=0.0,
            lic_opening=0.0,
            lrc_excl_lc_closing=recon.bel_closing + recon.ra_closing + recon.csm_closing,
            lc_closing=0.0,
            lic_closing=0.0,
        )
    raise TypeError(
        f"close: no SoFP mapping for {type(recon).__name__}")


def _zero_position() -> dict[str, float]:
    return {
        "lrc_excl_lc_opening": 0.0, "lc_opening": 0.0, "lic_opening": 0.0,
        "lrc_excl_lc_closing": 0.0, "lc_closing": 0.0, "lic_closing": 0.0,
    }


def _accumulate(position: dict[str, float], comp: _Components) -> None:
    position["lrc_excl_lc_opening"] += comp.lrc_excl_lc_opening
    position["lc_opening"] += comp.lc_opening
    position["lic_opening"] += comp.lic_opening
    position["lrc_excl_lc_closing"] += comp.lrc_excl_lc_closing
    position["lc_closing"] += comp.lc_closing
    position["lic_closing"] += comp.lic_closing


def _net(issued: dict[str, float], reins: dict[str, float]) -> dict[str, float]:
    # Net carrying amount = issued liability less the reinsurance asset
    # (paragraph 78). The reinsurance asset reduces the net liability.
    return {key: issued[key] - reins[key] for key in issued}


def _kind_rows(kind: str, position: dict[str, float]) -> list[dict]:
    """The four component rows (LRC-excl-LC, LC, LIC, Total) for one kind, each
    carrying opening, change (= closing - opening) and closing."""
    comps = (
        (_COMP_LRC, "lrc_excl_lc_opening", "lrc_excl_lc_closing"),
        (_COMP_LC, "lc_opening", "lc_closing"),
        (_COMP_LIC, "lic_opening", "lic_closing"),
    )
    rows = []
    total_open = total_close = 0.0
    for component, open_key, close_key in comps:
        opening = position[open_key]
        closing = position[close_key]
        total_open += opening
        total_close += closing
        rows.append({"kind": kind, "component": component,
                     "opening": opening, "change": closing - opening,
                     "closing": closing})
    rows.append({"kind": kind, "component": _COMP_TOTAL,
                 "opening": total_open, "change": total_close - total_open,
                 "closing": total_close})
    return rows


def assemble_sofp(reconciliations) -> pl.DataFrame:
    """The statement of financial position (IFRS 17 paragraphs 78, 99-101).

    The closing carrying amount of insurance contracts, split into LRC excluding
    the loss component / loss component / liability for incurred claims, for
    contracts issued, reinsurance contracts held, and the net -- each with the
    opening balance, the period change and the closing balance. Per row,
    ``opening + change == closing``; per kind, the Total row is the sum of the
    three components (the carrying amount); the Net kind is issued less
    reinsurance.
    """
    issued = _zero_position()
    reins = _zero_position()
    for recon in reconciliations:
        comp = _components(recon)
        _accumulate(issued if comp.kind == "issued" else reins, comp)
    rows = (
        _kind_rows(_KIND_ISSUED, issued)
        + _kind_rows(_KIND_REINSURANCE, reins)
        + _kind_rows(_KIND_NET, _net(issued, reins))
    )
    return pl.DataFrame(rows, schema=_SOFP_SCHEMA)


@dataclass(frozen=True, slots=True)
class ClosePackage:
    """The assembled IFRS 17 close pack for one reporting period.

    ``sofp`` is the statement of financial position (:func:`assemble_sofp`);
    ``reconciliation`` is the stacked per-model settlement detail (the lean tidy
    frame of :func:`~fastcashflow.disclosure.reconciliation_to_frame`, one block
    of rows per reconciliation, stamped with ``group_id``). The disclosure
    emitter materialises these into the multi-sheet close-pack artifact.
    """

    period_months: int
    sofp: pl.DataFrame
    reconciliation: pl.DataFrame

    def to_frames(self) -> dict[str, pl.DataFrame]:
        """The close pack as named frames -- the sheet-shaped views the
        disclosure emitter writes."""
        return {"sofp": self.sofp, "reconciliation": self.reconciliation}

    def __str__(self) -> str:
        net = self.sofp.filter(pl.col("kind") == _KIND_NET)
        lines = [
            f"IFRS 17 close pack -- {self.period_months}-month period",
            "  Net statement of financial position",
        ]
        for row in net.iter_rows(named=True):
            lines.append(
                f"    {row['component']:32}"
                f"{row['opening']:>16,.0f}{row['change']:>16,.0f}{row['closing']:>16,.0f}")
        return "\n".join(lines)


def close(reconciliations, *, group_ids=None) -> ClosePackage:
    """Assemble the close pack from a reporting period's settlement reconciliations.

    ``reconciliations`` is the GMM / VFA / PAA / reinsurance settlement
    reconciliations of one reporting period (what :func:`fastcashflow.reconcile`
    returns, one per model / group). ``group_ids``, if given, names the group of
    contracts each reconciliation belongs to (parallel to ``reconciliations``);
    it stamps the reconciliation detail so per-group lines stay identifiable.

    All reconciliations must share the same ``period_months`` -- a close pack is
    one reporting period.
    """
    recons = list(reconciliations)
    if not recons:
        raise ValueError("close: needs at least one settlement reconciliation")
    periods = {r.period_months for r in recons}
    if len(periods) > 1:
        raise ValueError(
            f"close: all reconciliations must share period_months; got {sorted(periods)}")
    if group_ids is not None and len(group_ids) != len(recons):
        raise ValueError(
            f"close: group_ids has {len(group_ids)} entries for "
            f"{len(recons)} reconciliations")
    sofp = assemble_sofp(recons)
    frames = []
    for i, recon in enumerate(recons):
        frame = reconciliation_to_frame(recon)
        gid = None if group_ids is None else str(group_ids[i])
        frames.append(frame.with_columns(
            pl.lit(gid, dtype=pl.Utf8).alias("group_id")))
    reconciliation = pl.concat(frames)
    return ClosePackage(period_months=periods.pop(), sofp=sofp,
                        reconciliation=reconciliation)
