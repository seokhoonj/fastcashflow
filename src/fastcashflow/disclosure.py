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

from fastcashflow.movement import GMMSettlementReconciliation

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


def _recon_frame(recon, model: str, blocks) -> pl.DataFrame:
    """The lean canonical tidy frame for one reconciliation: one row per
    disclosure line, read from the block spec so the spine has a single source.
    period_start / period_end are the relative period (the close assembler
    re-stamps absolute reporting positions when stacking a schedule)."""
    rows = [
        {
            "model": model, "group_id": None, "statement": "settlement",
            "period_start": 0, "period_end": int(recon.period_months),
            "block": block, "line": line, "amount": float(getattr(recon, field)),
        }
        for block, lines in blocks
        for line, field, _para, _memo in lines
    ]
    return pl.DataFrame(rows, schema=_LEAN_SCHEMA)


@singledispatch
def reconciliation_to_frame(recon) -> pl.DataFrame:
    """Return the lean canonical tidy frame for a settlement reconciliation."""
    raise TypeError(
        f"reconciliation_to_frame: no disclosure spec for {type(recon).__name__}")


@reconciliation_to_frame.register
def _(recon: GMMSettlementReconciliation) -> pl.DataFrame:
    return _recon_frame(recon, "gmm", _GMM_RECON_BLOCKS)
