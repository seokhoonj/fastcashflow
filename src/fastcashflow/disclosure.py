"""IFRS 17 disclosure assembly -- tidy frames over the settlement reconciliations.

The reporting layer that turns the footing settlement reconciliations into a
presentable close pack. This module is pure assembly + serialization over numbers
the settlement layer already produces and foots -- no measurement, no kernels.

Two ideas anchor it:

* a CANONICAL tidy frame -- one row per disclosure line -- so a reconciliation can
  leave Python as data (a sheet / PDF template / audit join consumes a tidy frame,
  not a __str__). The frame is LEAN: ``(model, group_id, statement, period_start,
  period_end, block, line, amount)``. The richer audit columns (line_code, the
  IFRS 17 paragraph anchor, the memo flag, the sort order) are reference data keyed
  on the line, materialised by the emitter at the contract boundary -- not
  denormalised onto every in-process row.
* the block SPEC -- ``_*_RECON_BLOCKS`` -- is the single source for the line spine:
  the ordered (block -> lines) structure, each line carrying its display name,
  the reconciliation field it reads, its paragraph anchor and whether it is a P&L
  memo (outside the balance recursion). The Phase-0 oracle pins that the spec
  covers exactly the reconciliation's fields, so a serialized line cannot drift
  from the reconciliation it explains.
"""
from __future__ import annotations

from functools import singledispatch

import polars as pl

from fastcashflow.io import _write_frame, write_measurement
from fastcashflow.movement import (
    GMMSettlementReconciliation, PAASettlementReconciliation,
    ReinsuranceSettlementReconciliation, VFASettlementReconciliation)

# The lean canonical schema returned by reconciliation_to_frame / to_frame.
_LEAN_COLUMNS = (
    "model", "group_id", "statement", "period_start", "period_end",
    "block", "line", "amount",
)
_LEAN_SCHEMA = {
    "model": pl.Utf8, "group_id": pl.Utf8, "statement": pl.Utf8,
    "period_start": pl.Int64, "period_end": pl.Int64,
    "block": pl.Utf8, "line": pl.Utf8, "amount": pl.Float64,
}
# The emitted artifact (file / sheet / audit join) is self-contained: the rich
# reference columns -- the machine line_code, the IFRS 17 paragraph anchor, the
# P&L-memo flag and the deterministic order -- are MATERIALISED at the write
# boundary, not denormalised onto the lean in-process frame.
_RICH_SCHEMA = {
    "model": pl.Utf8, "group_id": pl.Utf8, "statement": pl.Utf8,
    "period_start": pl.Int64, "period_end": pl.Int64,
    "block": pl.Utf8, "line": pl.Utf8, "line_code": pl.Utf8,
    "ifrs17_paragraph": pl.Utf8, "is_memo": pl.Boolean, "sort_order": pl.Int64,
    "amount": pl.Float64,
}

# Block spec -- the single source for the GMM settlement reconciliation line spine.
# Each line: (display name, reconciliation field, IFRS 17 paragraph, is P&L memo).
# loss_component_reversed / recognised legitimately appear in BOTH the CSM block
# (where they enter the CSM) and the Loss component block (where they run it off).
_GMM_RECON_BLOCKS = (
    ("BEL", (
        ("Opening", "bel_opening", "100(a)", False),
        ("Interest accreted", "bel_interest", "B72(a)", False),
        ("Release for service", "bel_release", "B123", False),
        ("Experience", "bel_experience", "B96", False),
        ("Closing", "bel_closing", "100(a)", False),
    )),
    ("RA", (
        ("Opening", "ra_opening", "101(b)", False),
        ("Interest accreted", "ra_interest", "B72(a)", False),
        ("Release for service", "ra_release", "B124", False),
        ("Experience", "ra_experience", "B96(d)", False),
        ("Closing", "ra_closing", "101(b)", False),
    )),
    ("CSM", (
        ("Opening", "csm_opening", "101(c)", False),
        ("Accretion", "csm_accretion", "44(b)/B72(b)", False),
        ("Experience unlocking", "csm_experience_unlocking", "44(c)/B96", False),
        ("Premium experience", "csm_premium_experience", "B96(a)", False),
        ("Investment experience", "csm_investment_experience", "B96(c)", False),
        ("Loss component reversed", "loss_component_reversed", "50(b)", False),
        ("Loss component recognised", "loss_component_recognised", "48", False),
        ("Release for service", "csm_release", "44(e)/B119", False),
        ("Closing", "csm_closing", "101(c)", False),
    )),
    ("Loss component", (
        ("Opening", "loss_component_opening", "49", False),
        ("Finance", "loss_component_finance", "51(c)", False),
        ("Amortised", "loss_component_amortised", "50(a)", False),
        ("Reversed", "loss_component_reversed", "50(b)", False),
        ("Recognised", "loss_component_recognised", "48", False),
        ("Closing", "loss_component_closing", "49", False),
    )),
    ("LIC", (
        ("Opening", "lic_opening", "100(c)", False),
        ("Claims incurred", "claims_incurred", "42(a)", False),
        ("Finance", "lic_finance", "42(c)", False),
        ("Claims paid", "claims_paid", "100(c)", False),
        ("Closing", "lic_closing", "100(c)", False),
    )),
    ("Memo (P&L)", (
        ("Finance wedge", "finance_wedge", "B97(a)", True),
        ("Premium experience (revenue)", "premium_experience_revenue", "B97(c)", True),
        ("Claims experience", "claims_experience", "B97(b)", True),
        ("Expense experience", "expense_experience", "B97(b)", True),
    )),
)


# VFA settlement reconciliation -- the paragraph-45 CSM (fair-value share +
# future service, no finance wedge) and an account-value-linked LIC.
_VFA_RECON_BLOCKS = (
    ("BEL", (
        ("Opening", "bel_opening", "100(a)", False),
        ("Interest accreted", "bel_interest", "B72(a)", False),
        ("Release for service", "bel_release", "B123", False),
        ("Experience", "bel_experience", "B96", False),
        ("Closing", "bel_closing", "100(a)", False),
    )),
    ("RA", (
        ("Opening", "ra_opening", "101(b)", False),
        ("Interest accreted", "ra_interest", "B72(a)", False),
        ("Release for service", "ra_release", "B124", False),
        ("Experience", "ra_experience", "B96(d)", False),
        ("Closing", "ra_closing", "101(b)", False),
    )),
    ("CSM", (
        ("Opening", "csm_opening", "101(c)", False),
        ("Accretion", "csm_accretion", "45(b)/B72(b)", False),
        ("Fair value share", "csm_fv_share", "45(b)", False),
        ("Future service", "csm_future_service", "45(c)", False),
        ("Premium experience", "csm_premium_experience", "B96(a)", False),
        ("Investment experience", "csm_investment_experience", "B96(c)", False),
        ("Loss component reversed", "loss_component_reversed", "50(b)", False),
        ("Loss component recognised", "loss_component_recognised", "48", False),
        ("Release for service", "csm_release", "45(e)/B119", False),
        ("Closing", "csm_closing", "101(c)", False),
    )),
    ("Loss component", (
        ("Opening", "loss_component_opening", "49", False),
        ("Finance", "loss_component_finance", "51(c)", False),
        ("Amortised", "loss_component_amortised", "50(a)", False),
        ("Reversed", "loss_component_reversed", "50(b)", False),
        ("Recognised", "loss_component_recognised", "48", False),
        ("Closing", "loss_component_closing", "49", False),
    )),
    ("LIC", (
        ("Opening", "lic_opening", "100(c)", False),
        ("Claims incurred", "claims_incurred", "42(a)", False),
        ("Finance", "lic_finance", "42(c)", False),
        ("Claims paid", "claims_paid", "100(c)", False),
        ("Closing", "lic_closing", "100(c)", False),
    )),
    ("Memo (P&L)", (
        ("Premium experience (revenue)", "premium_experience_revenue", "B97(c)", True),
        ("Claims experience", "claims_experience", "B97(b)", True),
        ("Expense experience", "expense_experience", "B97(b)", True),
    )),
)

# Reinsurance-held settlement reconciliation -- no loss component (paragraph 65,
# a reinsurance contract held cannot be onerous); a loss-RECOVERY component
# (66A-66B) instead, and no LIC block.
_REINSURANCE_RECON_BLOCKS = (
    ("BEL", (
        ("Opening", "bel_opening", "100(a)", False),
        ("Interest accreted", "bel_interest", "B72(a)", False),
        ("Release for service", "bel_release", "B123", False),
        ("Experience", "bel_experience", "B96", False),
        ("Closing", "bel_closing", "100(a)", False),
    )),
    ("RA", (
        ("Opening", "ra_opening", "101(b)", False),
        ("Interest accreted", "ra_interest", "B72(a)", False),
        ("Release for service", "ra_release", "B124", False),
        ("Experience", "ra_experience", "B96(d)", False),
        ("Closing", "ra_closing", "101(b)", False),
    )),
    ("CSM", (
        ("Opening", "csm_opening", "101(c)", False),
        ("Accretion", "csm_accretion", "66(b)/B72(b)", False),
        ("Experience unlocking", "csm_experience_unlocking", "66(c)/B96", False),
        ("Release for service", "csm_release", "66(e)/B119", False),
        ("Closing", "csm_closing", "101(c)", False),
    )),
    ("Loss-recovery component", (
        ("Opening", "loss_recovery_opening", "66B", False),
        ("Recognised", "loss_recovery_recognised", "66A", False),
        ("Reversed", "loss_recovery_reversed", "66B", False),
        ("Closing", "loss_recovery_closing", "66B", False),
    )),
    ("Memo (P&L)", (
        ("Finance wedge", "finance_wedge", "B97(a)", True),
    )),
)

# PAA settlement reconciliation -- an LRC (unearned premium) roll, no BEL/RA/CSM.
_PAA_RECON_BLOCKS = (
    ("LRC", (
        ("Opening", "lrc_opening", "100(a)", False),
        ("Premiums received", "premiums", "55(a)", False),
        ("Revenue recognised", "revenue", "B126", False),
        ("Experience", "lrc_experience", "55(b)", False),
        ("Closing", "lrc_closing", "100(a)", False),
    )),
    ("Loss component", (
        ("Opening", "loss_component_opening", "57", False),
        ("Recognised", "loss_component_recognised", "58", False),
        ("Reversed", "loss_component_reversed", "58", False),
        ("Closing", "loss_component_closing", "57", False),
    )),
    ("LIC", (
        ("Opening", "lic_opening", "100(c)", False),
        ("Claims incurred", "claims_incurred", "42(a)", False),
        ("Finance", "lic_finance", "42(c)", False),
        ("Claims paid", "claims_paid", "100(c)", False),
        ("Closing", "lic_closing", "100(c)", False),
    )),
    ("Memo (P&L)", (
        ("Claims experience", "claims_experience", "B97(b)", True),
        ("Expense experience", "expense_experience", "B97(b)", True),
    )),
)


def _recon_frame(recon, model: str, blocks, *, rich: bool = False) -> pl.DataFrame:
    """The canonical tidy frame for one reconciliation: one row per disclosure
    line, read from the block spec so the spine has a single source. period_start
    / period_end are the relative period (the close assembler re-stamps absolute
    reporting positions when stacking a schedule). ``rich`` adds the audit
    reference columns for an emitted, self-contained artifact."""
    rows = []
    order = 0
    for block, lines in blocks:
        for line, field, para, memo in lines:
            row = {
                "model": model, "group_id": None, "statement": "settlement",
                "period_start": 0, "period_end": int(recon.period_months),
                "block": block, "line": line, "amount": float(getattr(recon, field)),
            }
            if rich:
                row.update({"line_code": field, "ifrs17_paragraph": para,
                            "is_memo": memo, "sort_order": order})
            rows.append(row)
            order += 1
    return pl.DataFrame(rows, schema=_RICH_SCHEMA if rich else _LEAN_SCHEMA)


@singledispatch
def reconciliation_to_frame(recon) -> pl.DataFrame:
    """Return the lean canonical tidy frame for a settlement reconciliation."""
    raise TypeError(
        f"reconciliation_to_frame: no disclosure spec for {type(recon).__name__}")


@reconciliation_to_frame.register
def _(recon: GMMSettlementReconciliation) -> pl.DataFrame:
    return _recon_frame(recon, "gmm", _GMM_RECON_BLOCKS)


@reconciliation_to_frame.register
def _(recon: VFASettlementReconciliation) -> pl.DataFrame:
    return _recon_frame(recon, "vfa", _VFA_RECON_BLOCKS)


@reconciliation_to_frame.register
def _(recon: ReinsuranceSettlementReconciliation) -> pl.DataFrame:
    return _recon_frame(recon, "reinsurance", _REINSURANCE_RECON_BLOCKS)


@reconciliation_to_frame.register
def _(recon: PAASettlementReconciliation) -> pl.DataFrame:
    return _recon_frame(recon, "paa", _PAA_RECON_BLOCKS)


# (model, block spec, reconciliation class) for the four settlement families --
# the single registry the spec-covers-fields oracle iterates.
_RECON_SPECS = (
    ("gmm", _GMM_RECON_BLOCKS, GMMSettlementReconciliation),
    ("vfa", _VFA_RECON_BLOCKS, VFASettlementReconciliation),
    ("reinsurance", _REINSURANCE_RECON_BLOCKS, ReinsuranceSettlementReconciliation),
    ("paa", _PAA_RECON_BLOCKS, PAASettlementReconciliation),
)
_SPEC_BY_TYPE = {cls: (model, blocks) for model, blocks, cls in _RECON_SPECS}


def write_reconciliation(reconciliation, path) -> None:
    """Serialize a settlement reconciliation (or a list of them, one per
    reporting period) to a tidy file -- parquet / csv / xlsx -- with the rich
    audit columns materialised. A list is stacked with a 0-based ``period_index``
    so a multi-period close schedule round-trips as one long frame.

    Mirrors :func:`write_measurement` (which serializes the per-MP movements);
    this is the disclosure-shaped (reconciliation / close) serialization path.
    """
    recons = (list(reconciliation)
              if isinstance(reconciliation, (list, tuple)) else [reconciliation])
    frames = []
    for i, recon in enumerate(recons):
        model, blocks = _SPEC_BY_TYPE[type(recon)]
        frame = _recon_frame(recon, model, blocks, rich=True)
        frames.append(frame.with_columns(pl.lit(i, dtype=pl.Int64).alias("period_index")))
    _write_frame(pl.concat(frames), str(path))


def line_metadata() -> pl.DataFrame:
    """The disclosure line registry as a frame -- (model, block, line, line_code,
    ifrs17_paragraph, is_memo, sort_order) -- the single source the lean
    :func:`reconciliation_to_frame` and the emitter both read. Exposed so a user
    can join the reference columns onto a lean frame themselves."""
    rows = []
    for model, blocks, _cls in _RECON_SPECS:
        order = 0
        for block, lines in blocks:
            for line, field, para, memo in lines:
                rows.append({
                    "model": model, "block": block, "line": line,
                    "line_code": field, "ifrs17_paragraph": para,
                    "is_memo": memo, "sort_order": order})
                order += 1
    return pl.DataFrame(rows)


# The close pack's sheet order. The aggregate statements an entity reads; the
# per-model-point movement detail goes to a parquet sidecar (Excel's row limit).
_CLOSE_PACK_SHEETS = (
    ("00_Index", None),                      # cover metadata, built below
    ("01_SoFP", "sofp"),
    ("02_Service_Result", "service_result"),  # only if assembled
    ("03_Finance", "finance"),
    ("04_Reconciliation", "reconciliation"),  # the rich tidy detail
)


def _append_sheet(workbook, title: str, frame: pl.DataFrame) -> None:
    """Append a polars frame to a new sheet -- header row then data rows."""
    sheet = workbook.create_sheet(title=title)
    sheet.append(list(frame.columns))
    for row in frame.iter_rows():
        sheet.append(list(row))


def _index_frame(period_months: int, reconciliation: pl.DataFrame,
                 sheets, sidecar) -> pl.DataFrame:
    """The 00_Index cover sheet -- the reporting period, the models and groups
    in the pack, the sheet list, and the per-MP sidecar reference."""
    models = ", ".join(reconciliation["model"].unique().sort().to_list())
    groups = [g for g in reconciliation["group_id"].unique().sort().to_list()
              if g is not None]
    rows = [
        {"item": "Reporting period (months)", "value": str(period_months)},
        {"item": "Models", "value": models},
        {"item": "Groups", "value": ", ".join(groups) if groups else "(unlabelled)"},
        {"item": "Sheets", "value": ", ".join(sheets)},
        {"item": "Per-MP detail",
         "value": sidecar if sidecar else "(not included)"},
    ]
    return pl.DataFrame(rows, schema={"item": pl.Utf8, "value": pl.Utf8})


def write_close_pack(package, path, *, movements=None) -> None:
    """Write a close pack (a :class:`~fastcashflow.closing.ClosePackage`) to a
    multi-sheet ``.xlsx`` -- the aggregate IFRS 17 statements an entity reads --
    and, when ``movements`` is given, a per-model-point parquet sidecar.

    The workbook carries an index cover, the statement of financial position, the
    service result (if it was assembled), the finance statement, and the
    reconciliation detail with the rich audit columns (line_code, the IFRS 17
    paragraph anchor, the memo flag, the deterministic order) materialised by
    joining :func:`line_metadata` -- so the artifact is self-contained.

    The per-model-point settlement movement does NOT go in the workbook (a sheet
    caps at ~1,048,576 rows); ``movements`` -- one settlement movement or a list
    -- is written to ``<path>_permp[_i].parquet`` beside the workbook via
    :func:`write_measurement`, and the index sheet names the file(s).
    """
    p = str(path)
    if not p.endswith(".xlsx"):
        raise ValueError(
            f"write_close_pack: path must be a .xlsx workbook, got {path!r}")
    import openpyxl

    frames = package.to_frames()
    stem = p[:-len(".xlsx")]

    # Per-MP sidecar(s) first, so the index can name them. The naming keys off
    # the CALL SHAPE, not the count: a single movement -> one bare file; a list
    # (or tuple) -> one indexed file per entry, even if it holds just one.
    sidecar_label = None
    movement_list = None
    if movements is not None:
        is_single = not isinstance(movements, (list, tuple))
        movement_list = [movements] if is_single else list(movements)
        sidecar_paths = ([f"{stem}_permp.parquet"] if is_single
                         else [f"{stem}_permp_{i}.parquet"
                               for i in range(len(movement_list))])
        sidecar_label = ", ".join(sp.rsplit("/", 1)[-1] for sp in sidecar_paths)

    # The sheets actually present (service result only if assembled).
    present = [(title, key) for title, key in _CLOSE_PACK_SHEETS
               if key is None or key in frames]
    sheet_titles = [title for title, _key in present]

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)        # drop the default empty sheet
    for title, key in present:
        if key is None:
            frame = _index_frame(package.period_months, frames["reconciliation"],
                                 sheet_titles, sidecar_label)
        elif key == "reconciliation":
            # Materialise the rich audit columns at the emit boundary (DELTA 2):
            # the lean in-process frame stays lean; the file is self-contained.
            frame = frames[key].join(
                line_metadata(), on=["model", "block", "line"], how="left")
        else:
            frame = frames[key]
        _append_sheet(workbook, title, frame)
    workbook.save(p)

    if movement_list is not None:
        for movement, sidecar_path in zip(movement_list, sidecar_paths):
            write_measurement(movement, sidecar_path)
