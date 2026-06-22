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
  onerous, paragraph 65) and no liability for incurred claims block. A
  recoverable is a negative carrying amount in the one signed liability frame,
  so it is ADDED into the net position and thereby reduces it (paragraph 78).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from fastcashflow.disclosure import reconciliation_to_frame
from fastcashflow.movement import (
    GMMSettlementReconciliation, PAASettlementReconciliation,
    ReinsuranceSettlementReconciliation, VFASettlementReconciliation)
from fastcashflow.report import ReinsuranceReport, Report

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
    # Net carrying amount = issued liability net of reinsurance held (paragraph
    # 78). Both kinds carry their amount in the one signed liability frame (BEL +
    # RA + CSM, _components), where a reinsurance recoverable is a NEGATIVE
    # carrying amount (a negative liability). So the net is the algebraic SUM:
    # adding the negative reinsurance asset reduces the net liability.
    return {key: issued[key] + reins[key] for key in issued}


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
    three components (the carrying amount); the Net kind sums issued and
    reinsurance held in the one signed liability frame (a reinsurance recoverable
    is a negative carrying amount, so it reduces the net).
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


# The insurance finance statement (IFRS 17 paragraphs 87-89, B130-B136). One
# row per kind x line. The five sources sum to the insurance finance expense;
# loss_component_finance is a MEMO -- the loss component's share (r x pool
# interest unwind, 51(c)) of the BEL finance, not an amount on top of it.
_FINANCE_SCHEMA = {
    "kind": pl.Utf8, "line": pl.Utf8, "is_memo": pl.Boolean, "amount": pl.Float64,
}
_FINANCE_LINES = (
    ("BEL finance", "bel_interest"),        # B130-B136: finance on the FCF estimates
    ("RA finance", "ra_interest"),          # finance on the risk adjustment
    ("CSM finance", "csm_accretion"),       # CSM interest at the locked-in rate (B72)
    ("LIC finance", "lic_finance"),         # 42(c): incurred-claims discount unwind
    ("Locked-in rate adjustment", "finance_wedge"),  # B97(a): current vs locked-in rate gap
)
_FINANCE_MEMO_LINES = (
    ("Loss component finance", "loss_component_finance"),  # 51(c): sub-component of BEL finance
)
_FINANCE_TOTAL = "Insurance finance income or expenses"


def _finance_position(recons, kind: str) -> dict[str, float]:
    """Sum each finance line of the reconciliations of one kind. Reads fields
    with getattr defaults so a model lacking a line (PAA has no CSM accretion,
    VFA no locked-in adjustment, reinsurance no LIC) contributes zero to it."""
    fields = [field for _l, field in _FINANCE_LINES + _FINANCE_MEMO_LINES]
    acc = {field: 0.0 for field in fields}
    for recon in recons:
        if _components(recon).kind != kind:
            continue
        for field in fields:
            acc[field] += float(getattr(recon, field, 0.0))
    return acc


def _kind_finance_rows(kind: str, acc: dict[str, float]) -> list[dict]:
    rows = []
    total = 0.0
    for line, field in _FINANCE_LINES:
        total += acc[field]
        rows.append({"kind": kind, "line": line, "is_memo": False,
                     "amount": acc[field]})
    rows.append({"kind": kind, "line": _FINANCE_TOTAL, "is_memo": False,
                 "amount": total})
    for line, field in _FINANCE_MEMO_LINES:
        rows.append({"kind": kind, "line": line, "is_memo": True,
                     "amount": acc[field]})
    return rows


def assemble_finance(reconciliations) -> pl.DataFrame:
    """The insurance finance statement (IFRS 17 paragraphs 87-89, B130-B136).

    The period's insurance finance income or expenses disaggregated by source --
    finance on the BEL, the RA, the CSM (accretion at the locked-in rate, B72),
    the liability for incurred claims (42(c)), and the B97(a) locked-in rate
    adjustment (the current-vs-locked-in rate gap on the experience adjustment) --
    for contracts issued, reinsurance contracts held, and the net. The five
    sources sum to the ``Insurance finance income or expenses`` total line.
    ``Loss component finance`` is a
    memo: the loss component's share of the BEL finance (51(c)), already inside
    the BEL finance line, not an additional amount.
    """
    recons = list(reconciliations)
    issued = _finance_position(recons, "issued")
    reins = _finance_position(recons, "reinsurance")
    # One signed frame (cf. _net): the net finance expense is the algebraic sum
    # of issued and reinsurance-held finance, not a subtraction.
    net = {field: issued[field] + reins[field] for field in issued}
    rows = (
        _kind_finance_rows(_KIND_ISSUED, issued)
        + _kind_finance_rows(_KIND_REINSURANCE, reins)
        + _kind_finance_rows(_KIND_NET, net)
    )
    return pl.DataFrame(rows, schema=_FINANCE_SCHEMA)


# The insurance service result statement (IFRS 17 paragraphs 83, B120-B124).
# One row per kind x line x period. Issued and reinsurance held are presented
# separately (paragraph 82): the reinsurance result is NOT netted into the
# insurance service result.
_SERVICE_SCHEMA = {
    "kind": pl.Utf8, "line": pl.Utf8, "period_index": pl.Int64,
    "amount": pl.Float64,
}
# (display line, the report.by_period field it reads) for an issued Report.
_SERVICE_ISSUED_LINES = (
    ("Insurance revenue", "insurance_revenue"),
    ("Insurance service expense", "insurance_service_expense"),
    ("Insurance service result", "insurance_service_result"),
)
# (display line, the ReinsuranceReport.by_period field) -- "Net reinsurance
# result" (amounts recovered less premiums allocated, paragraph 86) is computed,
# inserted after "Amounts recovered".
_SERVICE_REINS_LINES = (
    ("Reinsurance premium", "reinsurance_premium_allocated"),
    ("Amounts recovered", "amounts_recovered"),
    ("Reinsurance service result", "reinsurance_service_result"),
)


def _aggregate_service(reports, line_fields, period_months):
    """Sum each line's by_period schedule across the reports of one kind, padded
    to the longest schedule. Returns ``(dict[line -> FloatArray], n_periods)``."""
    schedules = [r.by_period(period_months) for r in reports]
    probe = line_fields[0][1]
    n_periods = max((sched[probe].shape[0] for sched in schedules), default=0)
    agg = {line: np.zeros(n_periods) for line, _f in line_fields}
    for sched in schedules:
        for line, field in line_fields:
            series = sched[field]
            agg[line][:series.shape[0]] += series
    return agg, n_periods


def _service_rows(kind, ordered_lines, n_periods):
    """One row per (line, period) from an ordered list of (line, FloatArray)."""
    return [
        {"kind": kind, "line": line, "period_index": t, "amount": float(series[t])}
        for line, series in ordered_lines
        for t in range(n_periods)
    ]


def assemble_service_result(reports, *, period_months: int = 12) -> pl.DataFrame:
    """The insurance service result statement (IFRS 17 paragraphs 83, B120-B124).

    The period-by-period insurance revenue, service expense and service result
    for contracts issued, and the premiums / recoveries / net / service result
    for reinsurance contracts held (presented separately, paragraph 82), summed
    across the reports of each kind. ``reports`` is a list of
    :class:`~fastcashflow.Report` (issued) and / or
    :class:`~fastcashflow.ReinsuranceReport` (held).

    The service result is sourced from :meth:`Report.by_period`, NOT the
    settlement reconciliation: insurance revenue (B120-B124) needs the gross
    expected claims and expenses, which the settlement table does not carry (its
    BEL release is net of premiums). It is therefore the EARNED / projected P&L
    of the measurement -- for a new-business group's first reporting period it
    ties to that period's settlement; for a later in-force period it is the
    projection, with experience variances carried in the reconciliation and
    finance memos. v1 buckets on the elapsed basis (the calendar basis needs a
    per-report inception offset -- use :meth:`Report.by_period` directly for it).
    """
    reports = list(reports)
    for r in reports:
        if not isinstance(r, (Report, ReinsuranceReport)):
            raise TypeError(
                "assemble_service_result: expects Report / ReinsuranceReport, "
                f"got {type(r).__name__}")
    rows = []
    issued = [r for r in reports if isinstance(r, Report)]
    if issued:
        agg, n_periods = _aggregate_service(issued, _SERVICE_ISSUED_LINES,
                                            period_months)
        ordered = [(line, agg[line]) for line, _f in _SERVICE_ISSUED_LINES]
        rows += _service_rows(_KIND_ISSUED, ordered, n_periods)
    held = [r for r in reports if isinstance(r, ReinsuranceReport)]
    if held:
        agg, n_periods = _aggregate_service(held, _SERVICE_REINS_LINES,
                                            period_months)
        net = agg["Amounts recovered"] - agg["Reinsurance premium"]
        ordered = [
            ("Reinsurance premium", agg["Reinsurance premium"]),
            ("Amounts recovered", agg["Amounts recovered"]),
            ("Net reinsurance result", net),
            ("Reinsurance service result", agg["Reinsurance service result"]),
        ]
        rows += _service_rows(_KIND_REINSURANCE, ordered, n_periods)
    return pl.DataFrame(rows, schema=_SERVICE_SCHEMA)


@dataclass(frozen=True, slots=True)
class ClosePackage:
    """The assembled IFRS 17 close pack for one reporting period.

    ``sofp`` is the statement of financial position (:func:`assemble_sofp`);
    ``finance`` is the insurance finance statement (:func:`assemble_finance`);
    ``service_result`` is the insurance service result statement
    (:func:`assemble_service_result`), present only when ``close`` is given the
    reports; ``reconciliation`` is the stacked per-model settlement detail (the
    lean tidy frame of :func:`~fastcashflow.disclosure.reconciliation_to_frame`,
    one block of rows per reconciliation, stamped with ``group_id``). The
    disclosure emitter materialises these into the multi-sheet close-pack artifact.
    """

    period_months: int
    sofp: pl.DataFrame
    finance: pl.DataFrame
    reconciliation: pl.DataFrame
    service_result: pl.DataFrame | None = None

    def to_frames(self) -> dict[str, pl.DataFrame]:
        """The close pack as named frames -- the sheet-shaped views the
        disclosure emitter writes (the service result only if it was assembled)."""
        frames = {"sofp": self.sofp, "finance": self.finance,
                  "reconciliation": self.reconciliation}
        if self.service_result is not None:
            frames["service_result"] = self.service_result
        return frames

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


def close(reconciliations, *, reports=None, group_ids=None) -> ClosePackage:
    """Assemble the close pack from a reporting period's settlement reconciliations.

    ``reconciliations`` is the GMM / VFA / PAA / reinsurance settlement
    reconciliations of one reporting period (what :func:`fastcashflow.reconcile`
    returns, one per model / group) -- the source of the SoFP, the finance
    statement and the reconciliation detail. ``reports``, if given, is the list
    of :class:`~fastcashflow.Report` / :class:`~fastcashflow.ReinsuranceReport`
    that adds the insurance service result statement (sourced from the report,
    not the settlement -- see :func:`assemble_service_result`). ``group_ids``, if
    given, names the group of contracts each reconciliation belongs to (parallel
    to ``reconciliations``); it stamps the reconciliation detail so per-group
    lines stay identifiable.

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
    period_months = periods.pop()
    sofp = assemble_sofp(recons)
    finance = assemble_finance(recons)
    service_result = (None if reports is None
                      else assemble_service_result(reports, period_months=period_months))
    frames = []
    for i, recon in enumerate(recons):
        frame = reconciliation_to_frame(recon)
        gid = None if group_ids is None else str(group_ids[i])
        frames.append(frame.with_columns(
            pl.lit(gid, dtype=pl.Utf8).alias("group_id")))
    reconciliation = pl.concat(frames)
    return ClosePackage(period_months=period_months, sofp=sofp, finance=finance,
                        reconciliation=reconciliation, service_result=service_result)
