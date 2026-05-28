"""File I/O for model points, the actuarial basis and valuation results.

Model points and results go through polars; the actuarial basis -- read by
:func:`read_assumptions` -- comes from an Excel workbook via openpyxl.

Model points come in two shapes, both producing the same ``ModelPoints``:

* **wide** -- one row per policy, every benefit a column:
  ``<coverage_code>_benefit`` per rate-driven coverage, plus the survival
  benefits ``maturity_benefit`` and ``annuity_payment``. The convenient
  form for a single, homogeneous product.
* **long-form** -- a policies frame (contract attributes) plus a coverages
  frame, one row per policy x coverage carrying ``amount`` and ``premium``.
  The form for a heterogeneous, multi-product portfolio.

:func:`read_model_points` reads either; ``ModelPoints.to_wide`` /
``ModelPoints.to_long`` convert between them.

The core engine stays identifier-free: the kernel never needs a policy id, so
none is carried through ``ModelPoints`` or ``Valuation``. Identifiers are a
file-boundary concern -- pass them to :func:`write_valuation` (or via
``value_file``'s ``id_column``) to join results back to policies.
"""
from __future__ import annotations

import importlib.resources as resources
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import openpyxl
import polars as pl

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import (
    Assumptions, CoverageRate, ExpenseItem,
)
from fastcashflow.statemodel import STATE_MODELS
from fastcashflow.coverage import (
    CalculationMethod, RATE_DRIVEN_PATTERNS,
)
from fastcashflow.modelpoints import STATE_ACTIVE, STATE_NAMES, ModelPoints

# ``engine`` is the largest module in the package (codegen + the numba CPU
# kernels) and importing it at module load pulls all of that into any
# downstream that needs the I/O layer. The two engine names used here --
# ``Valuation`` for write_valuation's type hint and ``value`` for the
# ``value_file`` stream -- are imported under TYPE_CHECKING (for the hint)
# and lazily inside ``value_file`` (for the call), so a script that only
# reads model points or writes a results frame never imports engine.py.
if TYPE_CHECKING:  # pragma: no cover -- import only for type hints
    from fastcashflow.engine import Valuation

# Wide model-point columns with a fixed meaning. Any other ``*_benefit``
# column names a coverage by its coverage code.
_NAMED_WIDE = frozenset((
    "mp_id", "product_code", "channel_code", "issue_age", "term_months",
    "sex", "count", "state", "level_premium", "single_premium",
    "premium_term_months", "premium_frequency_months",
    "annuity_frequency_months", "maturity_benefit",
    "annuity_payment", "disability_income", "disability_benefit",
))


def _read_frame(path) -> pl.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        return pl.read_parquet(p)
    if p.endswith(".csv"):
        return pl.read_csv(p)
    if p.endswith(".xlsx"):
        return pl.read_excel(p, engine="openpyxl")
    if p.endswith((".feather", ".arrow")):
        return pl.read_ipc(p)
    raise ValueError(
        f"unsupported file type: {path!r} "
        "(expected .parquet, .csv, .xlsx or .feather)"
    )


def _write_frame(df: pl.DataFrame, path) -> None:
    p = str(path)
    if p.endswith(".parquet"):
        df.write_parquet(p)
    elif p.endswith(".csv"):
        df.write_csv(p)
    elif p.endswith((".feather", ".arrow")):
        df.write_ipc(p)
    elif p.endswith(".xlsx"):
        # Single-sheet xlsx via openpyxl (polars's own write_excel needs an
        # extra ``xlsxwriter`` dependency we do not require). The reader
        # side already accepts .xlsx via ``_read_frame``.
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(list(df.columns))
        for row in df.iter_rows():
            ws.append(list(row))
        wb.save(p)
    else:
        raise ValueError(
            f"unsupported file type: {path!r} "
            "(expected .parquet, .csv, .xlsx or .feather)"
        )


# ---------------------------------------------------------------------------
# Actuarial basis -- the assumptions workbook
# ---------------------------------------------------------------------------
#
# A single workbook (``assumptions.xlsx``) carries every assumption the engine
# needs. Nine sheets:
#
#   * ``segments``       -- (product, channel) -> which tables + scalar params
#                           (a ``defaults`` row that blank cells inherit).
#   * ``coverages``      -- (product) -> coverage_code, type, optional rate_table.
#   * ``mortality_tables``, ``incidence_rate_tables``, ``waiver_tables``,
#     ``lapse_tables``, ``discount_tables``, ``inflation_tables`` -- the
#     named rate tables the segments reference.
#
# See docs/assumptions-format.md for the column-level schema and
# docs/naming-conventions.md for the value-case rules.
#
# v1 limitation (refined in a later round): the discount, inflation and
# maintenance tables are read but used flat (their first entry). The reader
# returns ``{(product, channel): Assumptions}`` -- splitting model points by
# segment and valuing each is left to the caller.


def _sheet_dicts(ws):
    """Yield each data row of a worksheet as a dict keyed by the header row."""
    rows = ws.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration:
        return
    names = [str(h).strip() if h is not None else "" for h in header]
    for row in rows:
        if all(c is None for c in row):
            continue
        yield {n: v for n, v in zip(names, row) if n}


def _require_sheet(wb, sheet_name):
    """Return ``wb[sheet_name]`` or raise a friendly error.

    Without this wrap a missing required sheet surfaces as a raw openpyxl
    ``KeyError(sheet_name)`` -- non-obvious to a non-programmer actuary
    reading the traceback.
    """
    if sheet_name not in wb.sheetnames:
        raise ValueError(
            f"assumptions workbook is missing required sheet "
            f"{sheet_name!r}; known sheets: {sorted(wb.sheetnames)}"
        )
    return wb[sheet_name]


def _require_row_cols(row, required, *, sheet, table_id=None, what="row"):
    """Raise a friendly error if any required column is missing from ``row``.

    ``row`` is a dict yielded by :func:`_sheet_dicts`. Without this wrap a
    missing column surfaces as a bare ``KeyError(col)`` deep in the
    reader -- the user gets a column name but no context.
    """
    missing = [c for c in required if c not in row]
    if not missing:
        return
    ctx = f" (table_id={table_id!r})" if table_id is not None else ""
    raise ValueError(
        f"sheet {sheet!r}{ctx}: {what} is missing required column(s) "
        f"{missing}; row has columns {sorted(row)}"
    )


def _truncate_list(items, cap=10):
    """Format a list capped at ``cap`` items with a ``(... and N more)`` suffix.

    Used for not-found errors that enumerate the registered alternatives --
    a workbook with 100+ table_ids would otherwise produce an unreadable
    multi-line traceback.
    """
    items = list(items)
    if len(items) <= cap:
        return repr(items)
    extra = len(items) - cap
    return f"{items[:cap]!r} (... and {extra} more)"


# Axes a rate table may carry, in the order they index the internal grid.
# A sheet may include any subset; missing axes broadcast (the rate is held
# flat over that axis at lookup time). ``age`` (attained) is mutually
# exclusive with ``issue_age`` / ``duration`` (select-and-ultimate schema).
# ``issue_class`` is the at-issue classification axis (직업class / UW
# class) -- absent from most tables, broadcasts to a no-op when absent.
# ``elapsed`` is the semi-Markov sojourn axis (state-duration in years
# since entering the source state) -- carried by re-incidence /
# post-event mortality tables; broadcasts to a no-op when absent.
_RATE_AXES = ("sex", "issue_age", "duration", "age", "issue_class", "elapsed")


def _flex_rate_table(ws, *, value_col="rate"):
    """Schema-detecting rate-table reader -- returns ``{table_id: callable}``.

    The sheet may carry any subset of ``_RATE_AXES`` plus ``table_id`` and
    ``value_col`` (``rate`` or ``amount``). The reader detects which axes are
    present and returns a callable per table with the standard
    ``(sex, issue_age, duration)`` signature; axes not in the sheet broadcast
    (the rate is held flat over them), and lookups past the table's range
    clip to the edge.

    Supported schemas (any subset of ``{sex, age, issue_age, duration}``):

    * ``[rate]``                          -- flat scalar
    * ``[age, rate]``                     -- by attained age, sex broadcast
    * ``[sex, age, rate]``                -- by sex x age (the historical default)
    * ``[duration, rate]``                -- by duration, sex / age broadcast (lapse)
    * ``[sex, issue_age, duration, rate]`` -- full select-and-ultimate

    A sheet mixing ``age`` (attained) with ``issue_age`` / ``duration``
    (select schema) is rejected -- pick one parameterisation.
    """
    rows = list(_sheet_dicts(ws))
    if not rows:
        return {}
    header = set(rows[0].keys())
    if "table_id" not in header:
        raise ValueError(
            f"sheet {ws.title!r} is missing required column 'table_id'; "
            f"row has columns {sorted(header)}"
        )
    if value_col not in header:
        raise ValueError(
            f"sheet {ws.title!r} is missing required column {value_col!r} "
            f"(every rate-table row carries the rate in this column); "
            f"row has columns {sorted(header)}"
        )
    axes = tuple(a for a in _RATE_AXES if a in header)
    if "age" in axes and ("issue_age" in axes or "duration" in axes):
        raise ValueError(
            f"sheet {ws.title!r} mixes 'age' (attained) with "
            "'issue_age' / 'duration' (select schema) -- pick one"
        )

    by_id: dict[str, dict[tuple, float]] = {}
    for r in rows:
        tid = str(r["table_id"]).strip()
        try:
            key = tuple(int(r[a]) for a in axes)
        except KeyError as exc:
            raise ValueError(
                f"sheet {ws.title!r} table {tid!r}: row is missing axis "
                f"column {exc.args[0]!r} (header declares axes {axes!r}, "
                "so every row must populate them)"
            ) from None
        bucket = by_id.setdefault(tid, {})
        if key in bucket:
            raise ValueError(
                f"sheet {ws.title!r} table {tid!r}: duplicate row at "
                f"{dict(zip(axes, key))} -- a rate table must have one "
                "entry per axis combination (the last row would silently "
                "overwrite the first)"
            )
        bucket[key] = float(r[value_col])
    return {tid: _build_rate_callable(axes, list(entries.items()), ws.title, tid)
            for tid, entries in by_id.items()}


def _build_rate_callable(axes, entries, sheet_title, table_id):
    """Pack rows into a dense numpy grid and wrap in a lookup closure."""
    if not axes:
        # Flat scalar table -- one row, one rate.
        if len(entries) != 1:
            raise ValueError(
                f"sheet {sheet_title!r} table {table_id!r}: a flat (axis-less) "
                f"table must have exactly one row, got {len(entries)}"
            )
        val = entries[0][1]

        def rate(sex, issue_age, duration, issue_class, elapsed):
            shape = np.broadcast_shapes(
                np.asarray(sex).shape, np.asarray(issue_age).shape,
                np.asarray(duration).shape, np.asarray(issue_class).shape,
                np.asarray(elapsed).shape,
            )
            return np.full(shape, val, dtype=np.float64)
        rate._fcf_table_id = table_id
        rate._fcf_sheet = sheet_title
        rate._fcf_modifiers = ()
        return rate

    keys = np.array([k for k, _ in entries], dtype=np.int64)
    values = np.array([v for _, v in entries], dtype=np.float64)
    mins = keys.min(axis=0)
    maxs = keys.max(axis=0)
    shape = tuple(int(maxs[i] - mins[i] + 1) for i in range(len(axes)))
    grid = np.full(shape, np.nan, dtype=np.float64)
    for k, v in zip(keys, values):
        idx = tuple(int(k[i] - mins[i]) for i in range(len(axes)))
        grid[idx] = v
    if np.isnan(grid).any():
        raise ValueError(
            f"sheet {sheet_title!r} table {table_id!r} is not dense over its "
            f"axes {axes} -- some cells in the cartesian product are missing"
        )

    def rate(sex, issue_age, duration, issue_class, elapsed):
        sex = np.asarray(sex, dtype=np.int64)
        issue_age = np.asarray(issue_age, dtype=np.int64)
        duration = np.asarray(duration, dtype=np.int64)
        issue_class = np.asarray(issue_class, dtype=np.int64)
        elapsed = np.asarray(elapsed, dtype=np.int64)
        # One index array per axis present in the table.
        idxs = []
        for i, a in enumerate(axes):
            if a == "sex":
                raw = sex
            elif a == "age":
                raw = issue_age + duration                # attained age
            elif a == "issue_age":
                raw = issue_age
            elif a == "issue_class":
                raw = issue_class
            elif a == "elapsed":
                raw = elapsed
            else:                                          # duration
                raw = duration
            idxs.append(np.clip(raw - int(mins[i]), 0, shape[i] - 1))
        # Broadcast each index to the input's full broadcast shape so that
        # numpy fancy-indexing returns a result of that shape (axes absent
        # from the table contribute through broadcast, not indexing).
        target = np.broadcast_shapes(
            sex.shape, issue_age.shape, duration.shape,
            issue_class.shape, elapsed.shape,
        )
        return grid[tuple(np.broadcast_to(ix, target) for ix in idxs)]
    rate._fcf_table_id = table_id
    rate._fcf_sheet = sheet_title
    rate._fcf_modifiers = ()
    return rate


def _read_expense_tables(ws) -> dict[str, tuple[ExpenseItem, ...]]:
    """Read the optional ``expense_tables`` sheet.

    Each row is one ``ExpenseItem`` -- the item-form expense ledger the
    engine dispatches on. Columns: ``table_id``, ``expense_type``,
    ``basis``, ``value``. The same ``table_id`` may span multiple rows
    (an acquisition row plus a maintenance row, plus an LAE row, ...).
    Returns ``{table_id: tuple[ExpenseItem, ...]}`` for the
    segments-side ``expense_table`` lookup to consume. Inflation is
    *not* a row attribute -- it lives on the segment as the global
    economic ``expense_inflation`` curve (see :data:`inflation_tables`
    sheet).
    """
    by_id: dict[str, list[ExpenseItem]] = {}
    first = True
    for r in _sheet_dicts(ws):
        if first:
            _require_row_cols(
                r, ("table_id", "expense_type", "basis", "value"),
                sheet=ws.title,
            )
            first = False
        tid = str(r["table_id"]).strip()
        by_id.setdefault(tid, []).append(ExpenseItem(
            expense_type=str(r["expense_type"]).strip(),
            basis=str(r["basis"]).strip(),
            value=float(r["value"]),
        ))
    return {tid: tuple(rows) for tid, rows in by_id.items()}


def _read_ae_factors(ws):
    """Read the optional ``ae_factors`` sheet.

    Each row is one (product, channel, coverage_code) -> factor (a runtime
    multiplier on the base rate). Optional axis columns
    ``{sex, age, issue_age, duration}`` let the factor vary along those
    dimensions (same schema-detection rules as the base rate tables); missing
    axes broadcast. ``channel`` empty matches the segment whose channel is
    blank (a single-segment workbook).

    Returns ``{(product, channel, coverage_code): callable(sex, issue_age,
    duration) -> factor}``. Missing sheet -> empty dict -> no A/E adjustment.
    """
    rows = list(_sheet_dicts(ws))
    if not rows:
        return {}
    header = set(rows[0].keys())
    _require_row_cols(
        rows[0], ("product_code", "coverage_code", "factor"), sheet=ws.title,
    )
    axes = tuple(a for a in _RATE_AXES if a in header)
    if "age" in axes and ("issue_age" in axes or "duration" in axes):
        raise ValueError(
            f"sheet {ws.title!r} mixes 'age' (attained) with "
            "'issue_age' / 'duration' (select schema) -- pick one"
        )

    by_key: dict[tuple, list] = {}
    for r in rows:
        product_code = str(r["product_code"]).strip()
        ch = r.get("channel_code")
        channel_code = str(ch).strip() if ch not in (None, "") else ""
        coverage_code = str(r["coverage_code"]).strip()
        key = (product_code, channel_code, coverage_code)
        try:
            axes_key = tuple(int(r[a]) for a in axes)
        except KeyError as exc:
            raise ValueError(
                f"sheet {ws.title!r} row for {key!r} is missing axis "
                f"column {exc.args[0]!r} (header declares axes {axes!r})"
            ) from None
        by_key.setdefault(key, []).append((axes_key, float(r["factor"])))
    return {
        key: _build_rate_callable(axes, entries, ws.title, "/".join(key))
        for key, entries in by_key.items()
    }


def _propagate_table_id(wrapper, inner, modifier_tag):
    """Carry the source table_id from an inner rate callable to a wrapper."""
    tid = getattr(inner, "_fcf_table_id", None)
    if tid is None:
        return
    wrapper._fcf_table_id = tid
    wrapper._fcf_sheet = getattr(inner, "_fcf_sheet", None)
    wrapper._fcf_modifiers = getattr(inner, "_fcf_modifiers", ()) + (modifier_tag,)


def _with_improvement(rate_fn, improvement_curve):
    """Wrap a rate callable to multiply by an annual improvement factor.

    ``improvement_curve`` is a ``(n_years,)`` array indexed by policy year
    (= duration). ``factor[0] = 1.0`` typically, decreasing for genuine
    improvement (mortality falls). Held flat past the curve's end.
    ``None`` returns ``rate_fn`` unchanged.
    """
    if rate_fn is None or improvement_curve is None:
        return rate_fn
    n = improvement_curve.shape[0]

    def improved(sex, issue_age, duration, issue_class, elapsed):
        d = np.asarray(duration, dtype=np.int64)
        idx = np.clip(d, 0, n - 1)
        return (rate_fn(sex, issue_age, duration, issue_class, elapsed)
                * improvement_curve[idx])
    _propagate_table_id(improved, rate_fn, "improvement")
    return improved


def _with_ae_factor(rate_fn, factor_fn):
    """Wrap a rate callable to multiply by an A/E factor at call time.

    ``factor_fn`` shares the ``(sex, issue_age, duration) -> array``
    signature; ``None`` (no factor configured for this coverage) returns
    ``rate_fn`` unchanged.
    """
    if factor_fn is None or rate_fn is None:
        return rate_fn

    def adjusted(sex, issue_age, duration, issue_class, elapsed):
        return (rate_fn(sex, issue_age, duration, issue_class, elapsed)
                * factor_fn(sex, issue_age, duration, issue_class, elapsed))
    _propagate_table_id(adjusted, rate_fn, "ae")
    return adjusted


def _with_age_shift(rate_fn, shift):
    """Wrap a rate callable to shift its ``issue_age`` argument by ``shift``.

    A positive shift treats every life as ``shift`` years older when looking
    up the base table; negative shifts make them younger. Returns ``rate_fn``
    unchanged when ``shift == 0`` (no allocation cost). ``rate_fn`` may be
    ``None`` (an optional rate the segment did not configure), in which case
    the wrapper is a no-op too.
    """
    if rate_fn is None or shift == 0:
        return rate_fn

    def shifted(sex, issue_age, duration, issue_class, elapsed):
        return rate_fn(sex, issue_age + shift, duration, issue_class, elapsed)
    _propagate_table_id(shifted, rate_fn, f"shift{shift:+d}")
    return shifted


def _axis_tables(ws, axis, *, value_col="rate"):
    """``{table_id: value array}`` from a sheet keyed by ``axis`` (0-based).

    ``value_col`` names the column carrying the per-axis value -- ``"rate"``
    for rate / probability sheets, ``"amount"`` for currency sheets
    (maintenance expense). The column-name distinction documents units;
    a probability and a currency amount should not share a column name.
    """
    by_id: dict[str, dict] = {}
    first = True
    for r in _sheet_dicts(ws):
        if first:
            _require_row_cols(
                r, ("table_id", axis, value_col), sheet=ws.title,
            )
            first = False
        tid = str(r["table_id"]).strip()
        try:
            by_id.setdefault(tid, {})[int(r[axis])] = float(r[value_col])
        except KeyError as exc:
            raise ValueError(
                f"sheet {ws.title!r} table {tid!r}: row is missing "
                f"column {exc.args[0]!r} (row has columns {sorted(r)})"
            ) from None
    return {tid: np.asarray([by_k[k] for k in range(len(by_k))], np.float64)
            for tid, by_k in by_id.items()}


#: Assumptions schema versions this reader knows how to consume. A
#: workbook with no ``_meta`` sheet (or no ``schema_version`` key) is
#: treated as ``v1`` so older sample files keep working. Add the new
#: version here when a breaking schema change ships.
_SUPPORTED_SCHEMA_VERSIONS = frozenset({"v1"})


def _check_schema_version(wb) -> None:
    """Read the optional ``_meta`` sheet's ``schema_version`` and reject
    versions this build does not understand.

    The sheet shape is two columns -- ``key`` / ``value`` -- so additional
    metadata (workbook owner, generation date, etc.) can land in later
    rows without breaking the reader.
    """
    if "_meta" not in wb.sheetnames:
        return                                            # legacy = v1
    ws = wb["_meta"]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return
    meta = {str(r[0]).strip(): r[1] for r in rows[1:]
            if r and r[0] is not None}
    version = str(meta.get("schema_version", "v1")).strip()
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported assumptions schema_version {version!r}; this "
            f"build understands {sorted(_SUPPORTED_SCHEMA_VERSIONS)}. "
            "Upgrade fastcashflow or downgrade the workbook."
        )


def read_assumptions(path: Path | str) -> dict[tuple[str, str], Assumptions]:
    """Read the assumptions workbook into a per-segment ``Assumptions`` dict.

    ``path`` is a single ``assumptions.xlsx`` workbook holding both the rate
    tables and the segment mapping (see the module header for the sheet
    layout). The ``segments`` sheet maps each (product, channel) to which
    tables it uses plus scalar parameters, with a ``defaults`` row whose
    values blank cells inherit; the ``coverages`` sheet attaches
    rate-driven coverages to products.

    Returns ``{(product, channel): Assumptions}`` -- one basis per segment.

    v1: the discount and inflation tables are read but used flat (their
    first entry); the per-segment dict is returned for the caller to value
    segment by segment.

    A workbook may optionally carry a ``_meta`` sheet (``key | value``
    layout) with a ``schema_version`` row. When absent the reader assumes
    ``v1``. The version gates breaking schema changes -- a future ``v2``
    that renames a column will be rejected by a ``v1``-only reader. The
    sample workbook ships with ``schema_version = v1``.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    _check_schema_version(wb)

    def optional(sheet, reader):
        return reader(wb[sheet]) if sheet in wb.sheetnames else {}

    mortality_t = _flex_rate_table(_require_sheet(wb, "mortality_tables"))
    incidence_rate_t = optional("incidence_rate_tables", _flex_rate_table)
    waiver_t = optional("waiver_tables", _flex_rate_table)
    lapse_t = _flex_rate_table(_require_sheet(wb, "lapse_tables"))
    discount_t = _axis_tables(_require_sheet(wb, "discount_tables"), "year")
    inflation_t = optional(
        "inflation_tables", lambda w: _axis_tables(w, "year"),
    )
    ae_factors = optional("ae_factors", _read_ae_factors)
    improvement_t = optional(
        "improvement_tables",
        lambda w: _axis_tables(w, "year", value_col="factor"),
    )
    # Surrender value curves -- per-month factor applied to cumulative
    # premium paid. Optional; absent means lapse has no payout.
    surrender_t = optional(
        "surrender_value_tables",
        lambda w: _axis_tables(w, "duration_month", value_col="factor"),
    )
    # Expense ledger -- item form. Optional; per-segment ``expense_table``
    # in the segments sheet selects which table_id to attach.
    expense_t = optional("expense_tables", _read_expense_tables)

    defaults: dict = {}
    segments: list = []
    seg_rows = list(_sheet_dicts(_require_sheet(wb, "segments")))
    if seg_rows:
        header = set(seg_rows[0].keys())
        for new, legacy in (("product_code", "product"),
                            ("channel_code", "channel")):
            if new not in header and legacy in header:
                raise ValueError(
                    f"segments sheet has column {legacy!r} but not {new!r} "
                    f"-- did you mean {new!r}? (the column was renamed; "
                    "see docs/naming-conventions.md)"
                )
    for r in seg_rows:
        if str(r.get("product_code", "") or "").strip().lower() == "defaults":
            defaults = r
        else:
            segments.append(r)
    # Coverages registry -- global, one row per coverage_code. The same code
    # plugs into any segment's contracts (a HEALTH policy and a TERM_LIFE
    # policy that both attach `CANCER` share the same incidence rate). When
    # a company genuinely needs product-specific calibrations of the same
    # disease, give them different coverage_codes (e.g. CANCER_HEALTH vs
    # CANCER_WHOLELIFE) -- the engine then treats them as separate coverages.
    #
    # Plan B (3-file split): this sheet carries only ``coverage_code`` +
    # ``rate_table`` -- the rate-driven entries. The pattern taxonomy
    # (DEATH / MORBIDITY / DIAGNOSIS / ANNUITY / MATURITY) moves to a
    # separate ``calculation_methods.csv`` file consumed by
    # :func:`read_model_points`, so the assumptions workbook is purely the
    # actuarial basis and the company catalogue lives elsewhere. Survival
    # entries (ANNUITY, MATURITY) never carry a ``rate_table`` and so do
    # not appear here. A death coverage's ``rate_table`` cell may point to
    # either an ``incidence_rate_tables`` entry or a ``mortality_tables``
    # entry: the engine reads the table as a per-coverage payment rate
    # independently of how the same id is also used for in-force decrement.
    rate_driven_coverages: list[tuple[str, str]] = []
    if "coverages" in wb.sheetnames:
        for r in _sheet_dicts(wb["coverages"]):
            rt = r.get("rate_table")
            code = str(r["coverage_code"]).strip()
            if rt in (None, ""):
                raise ValueError(
                    f"coverages row {code!r} has no rate_table; the "
                    "assumptions workbook only lists rate-driven coverages "
                    "(survival entries belong in calculation_methods.csv, "
                    "not here)"
                )
            rate_driven_coverages.append((code, str(rt).strip()))

    result = {}
    for seg in segments:
        product_code = str(seg["product_code"]).strip()
        channel_code = str(seg.get("channel_code", "") or "").strip()
        where = f"segments row ({product_code} / {channel_code})"

        def cell(col):
            v = seg.get(col)
            if v is None or (isinstance(v, str) and not v.strip()):
                v = defaults.get(col)
            return None if (isinstance(v, str) and not v.strip()) else v

        def lookup(registry, col, optional_ref=False):
            tid = cell(col)
            tid = str(tid).strip() if tid is not None else None
            if tid is None:
                if optional_ref:
                    return None
                raise ValueError(f"{where}: {col!r} is required")
            if tid not in registry:
                raise ValueError(f"{where}: {col}={tid!r} is not registered")
            return registry[tid]

        def scalar(col, required=False):
            v = cell(col)
            if v is None and required:
                raise ValueError(f"{where}: {col!r} is required")
            return None if v is None else float(v)

        shift_mort = int(scalar("mortality_age_shift") or 0)
        shift_morb = int(scalar("morbidity_age_shift") or 0)
        shift_wvr = int(scalar("waiver_age_shift") or 0)

        def ae(coverage_code):
            return ae_factors.get((product_code, channel_code, coverage_code))

        coverage_list = []
        for code, rate_table in rate_driven_coverages:
            # Death coverages share the mortality_tables namespace -- a
            # rate_table cell may name either an incidence_rate_tables or
            # a mortality_tables entry. Incidence wins on collision (the
            # convention rare in practice; calibrating the same code with
            # different tables under the same name is a workbook error).
            if rate_table in incidence_rate_t:
                rate_fn = incidence_rate_t[rate_table]
                shift = shift_morb
            elif rate_table in mortality_t:
                rate_fn = mortality_t[rate_table]
                shift = shift_mort
            else:
                raise ValueError(
                    f"coverage {code!r} of product {product_code!r}: "
                    f"rate_table {rate_table!r} is not registered. "
                    f"incidence_rate_tables has "
                    f"{_truncate_list(sorted(incidence_rate_t))}; "
                    f"mortality_tables has "
                    f"{_truncate_list(sorted(mortality_t))}"
                )
            rate_fn = _with_age_shift(rate_fn, shift)
            rate_fn = _with_ae_factor(rate_fn, ae(code))
            coverage_list.append(CoverageRate(code=code, rate=rate_fn))

        mortality_fn = lookup(mortality_t, "mortality_table")
        mortality_fn = _with_age_shift(mortality_fn, shift_mort)
        improvement_curve = lookup(
            improvement_t, "mortality_improvement_table", optional_ref=True,
        )
        mortality_fn = _with_improvement(mortality_fn, improvement_curve)

        waiver = lookup(waiver_t, "waiver_table", optional_ref=True)
        waiver_fn = _with_age_shift(waiver, shift_wvr)

        surrender_curve = lookup(
            surrender_t, "surrender_value_table", optional_ref=True,
        )
        # Row-form expense ledger -- the segments row points an
        # ``expense_table`` cell at one entry of the ``expense_tables``
        # sheet. Blank cell = no expense (empty ``expense_items`` tuple,
        # zero-expense projection).
        expense_items = lookup(expense_t, "expense_table", optional_ref=True)
        # Global economic inflation -- the segments row points an
        # ``inflation_table`` cell at one named scenario in the
        # ``inflation_tables`` sheet (analogous to ``discount_table``).
        # Blank cell = zero inflation (recurring expense items stay flat).
        inflation_curve = lookup(
            inflation_t, "inflation_table", optional_ref=True,
        )
        kwargs: dict = dict(
            mortality_annual=mortality_fn,
            lapse_annual=lookup(lapse_t, "lapse_table"),
            waiver_incidence_annual=waiver_fn,
            # Pass the full per-year discount through -- the engine
            # expands it to a per-month curve via fastcashflow.curves.
            # A one-row table reproduces the flat-scalar behaviour.
            discount_annual=lookup(discount_t, "discount_table"),
            expense_items=expense_items or (),
            expense_inflation=(
                inflation_curve if inflation_curve is not None else 0.0
            ),
            ra_confidence=scalar("ra_confidence", required=True),
            mortality_cv=scalar("mortality_cv", required=True),
            coverages=tuple(coverage_list),
            surrender_value_curve=surrender_curve,
        )
        for opt_col in ("morbidity_cv", "longevity_cv", "disability_cv",
                        "expense_cv", "cost_of_capital_rate",
                        "investment_return", "fund_fee"):
            v = scalar(opt_col)
            if v is not None:
                kwargs[opt_col] = v
        method = cell("ra_method")
        if method is not None:
            kwargs["ra_method"] = str(method).strip()
        # Optional state_model column -- non-programmer actuary picks a
        # bundled topology by its registry key (e.g. "WAIVER"). Blank cell
        # leaves Assumptions.state_model = None; an unknown key is an
        # error with a hint listing the registered keys.
        state_model_key = cell("state_model")
        if state_model_key is not None:
            key = str(state_model_key).strip()
            try:
                kwargs["state_model"] = STATE_MODELS[key]
            except KeyError:
                raise ValueError(
                    f"{where}: state_model={key!r} is not in STATE_MODELS "
                    f"(known: {sorted(STATE_MODELS)})"
                ) from None
        result[(product_code, channel_code)] = Assumptions(**kwargs)
    return result


# ---------------------------------------------------------------------------
# Model points -- wide and long-form
# ---------------------------------------------------------------------------

def _read_state(col: pl.Series) -> np.ndarray:
    """Convert a model-point ``state`` column to engine state codes.

    Accepts the readable names a practitioner edits in a spreadsheet --
    ``active`` / ``waiver`` / ``paidup`` -- or the integer codes directly.
    Case, spaces, hyphens and underscores are ignored, so ``Paid-up`` and
    ``paid up`` read the same. A blank cell means an ordinary active contract.
    """
    if col.dtype == pl.String:
        # Normalised lookup -- canonical STATE_NAMES keys ("ACTIVE", "WAIVER",
        # "PAID_UP") are uppercase, but any spelling (case, spaces, hyphens,
        # underscores ignored) of the canonical name maps to the same code.
        normalised = {
            k.lower().replace("_", "").replace("-", "").replace(" ", ""): v
            for k, v in STATE_NAMES.items()
        }
        out = np.empty(len(col), dtype=np.int64)
        for i, v in enumerate(col):
            name = "" if v is None else str(v).strip().lower()
            name = name.replace(" ", "").replace("-", "").replace("_", "")
            if name == "":
                out[i] = STATE_ACTIVE
            elif name in normalised:
                out[i] = normalised[name]
            else:
                raise ValueError(
                    f"unknown contract state {v!r}; "
                    f"expected one of {sorted(STATE_NAMES)}"
                )
        return out
    out = col.fill_null(STATE_ACTIVE).to_numpy().astype(np.int64)
    valid = set(STATE_NAMES.values())
    bad = sorted(set(int(v) for v in out) - valid)
    if bad:
        raise ValueError(
            f"state column has unknown integer value(s) {bad}; "
            f"expected one of {sorted(valid)} (see STATE_NAMES)"
        )
    return out


def _warn_if_elapsed_months(columns) -> None:
    """Warn that ``elapsed_months`` on a policies frame is silently dropped.

    The static-spec policies frame holds inception-time facts (issue_age,
    term, sex ...). The in-force closing state lives in a separate
    ``inforce_state`` file -- :func:`read_inforce_state` is the only
    surface that fills the ``elapsed_months`` field of :class:`ModelPoints`.
    A column on the policies side is a common mistake (mixed roles) and
    would be silently ignored; the warning makes the source-of-truth
    boundary explicit.
    """
    if "elapsed_months" in columns:
        warnings.warn(
            "policies frame has 'elapsed_months' column; this reader "
            "ignores it. elapsed_months belongs in the in-force state file "
            "(see read_inforce_state / apply_inforce_state) -- the policies "
            "frame is the inception-time static spec.",
            UserWarning,
            stacklevel=3,
        )


def _parse_calculation_methods(path: Path | str) -> dict[str, CalculationMethod]:
    """Read a ``calculation_methods.csv`` taxonomy file into a dict.

    The file has two required columns -- ``coverage_code`` and
    ``calculation_method`` -- plus an optional human-friendly
    ``coverage_name`` (read but not retained, since the engine routes by
    code). Returns ``{coverage_code: CalculationMethod}``. Raises
    :class:`ValueError` for an unknown pattern (V1) and a duplicate code
    (V2); the messages name the offending row so the operator can fix
    the file without scrolling through it.
    """
    df = _read_frame(path)
    for need in ("coverage_code", "calculation_method"):
        if need not in df.columns:
            raise ValueError(
                f"the calculation_methods file is missing required column "
                f"{need!r}"
            )
    result: dict[str, CalculationMethod] = {}
    valid = ", ".join(p.value for p in CalculationMethod)
    for row in df.iter_rows(named=True):
        code = str(row["coverage_code"]).strip()
        raw = str(row["calculation_method"]).strip()
        try:
            pattern = CalculationMethod(raw)
        except ValueError as exc:
            raise ValueError(
                f"calculation_methods row {code!r}: calculation_method={raw!r} "
                f"is not one of {{{valid}}}"
            ) from exc
        if code in result:
            raise ValueError(
                f"calculation_methods row {code!r}: duplicate coverage_code "
                "(every code may appear exactly once in the taxonomy)"
            )
        result[code] = pattern
    return result


def _wide_model_points(df: pl.DataFrame,
                       calculation_methods=None) -> ModelPoints:
    """Build a ``ModelPoints`` from a wide frame -- one row per policy, each
    coverage a ``<coverage_code>_benefit`` column. Reads without any
    assumptions: the coverage codes come from the column names, ordered by
    the ``calculation_methods`` catalogue when given (else column order).
    The engine aligns ``Assumptions.coverages`` to that order at measure
    time."""
    for need in ("issue_age", "term_months"):
        if need not in df.columns:
            raise ValueError(
                f"the model-point file is missing required column {need!r}"
            )
    _warn_if_elapsed_months(df.columns)
    n_mp = df.height
    fields: dict[str, object] = dict(
        issue_age=df["issue_age"].to_numpy(),
        term_months=df["term_months"].to_numpy(),
        level_premium=(df["level_premium"].to_numpy()
                       if "level_premium" in df.columns
                       else np.zeros(n_mp)),
    )
    for opt in ("sex", "count", "single_premium", "premium_term_months",
                "premium_frequency_months", "annuity_frequency_months",
                "maturity_benefit", "annuity_payment",
                "disability_income", "disability_benefit", "account_value",
                "guaranteed_credit_rate"):
        if opt in df.columns:
            fields[opt] = df[opt].to_numpy()
    # Segment metadata -- optional string columns; route to value_segmented.
    for opt in ("product_code", "channel_code"):
        if opt in df.columns:
            fields[opt] = df[opt].to_numpy()
    if "state" in df.columns:
        fields["state"] = _read_state(df["state"])

    # Candidate coverage codes come from the ``<code>_benefit`` columns.
    present_cols: dict[str, np.ndarray] = {}
    for col in df.columns:
        if not col.endswith("_benefit") or col in _NAMED_WIDE:
            continue
        present_cols[col[: -len("_benefit")]] = df[col].to_numpy()
    # Coverage order: the catalogue's rate-driven order when a catalogue is
    # given (and every benefit column must be a rate-driven catalogue code),
    # else the column order -- in which case the pattern is auto-inferred
    # from the code name at measure time (coverage.coverage_arrays).
    if calculation_methods is not None:
        wide_ctypes = {k: CalculationMethod(v)
                       for k, v in calculation_methods.items()}
        rate_driven = [c for c, m in wide_ctypes.items()
                       if m in RATE_DRIVEN_PATTERNS]
        candidate_codes = set(rate_driven)
    else:
        rate_driven = None
        candidate_codes = set(present_cols)
    # Guard the reserved-name collision: a coverage_code whose
    # ``<code>_benefit`` column name shadows a fixed-meaning column
    # (maturity_benefit, disability_benefit, ...) would be silently routed
    # to the scalar field rather than into the CSR. Catch at read time --
    # against the catalogue codes when given (a registered code whose
    # benefit column would be eaten by the reserved scalar).
    reserved_codes = {n[: -len("_benefit")] for n in _NAMED_WIDE
                      if n.endswith("_benefit")}
    bad = sorted(candidate_codes & reserved_codes)
    if bad:
        raise ValueError(
            f"coverage code(s) {bad} collide with reserved wide-form "
            f"column name(s) {[c + '_benefit' for c in bad]} -- rename "
            "the coverage_code"
        )
    if rate_driven is not None:
        unknown = sorted(c for c in present_cols if c not in candidate_codes)
        if unknown:
            raise ValueError(
                f"wide column(s) {[c + '_benefit' for c in unknown]} name "
                f"coverage(s) {unknown} that are not rate-driven in the "
                "calculation_methods catalogue"
            )
        ordered = [c for c in rate_driven if c in present_cols]
    else:
        ordered = list(present_cols)
    code_to_cov_idx = {c: i for i, c in enumerate(ordered)}
    benefits = {code_to_cov_idx[c]: present_cols[c] for c in ordered}
    if benefits:
        fields["benefits"] = benefits
    if calculation_methods is not None:
        fields["calculation_methods"] = calculation_methods
    if ordered:
        fields["coverage_codes"] = tuple(ordered)
    return ModelPoints(**fields)


def _long_model_points(pol: pl.DataFrame, cov: pl.DataFrame,
                       calculation_methods=None) -> ModelPoints:
    """Build a ``ModelPoints`` from a long-form policies + coverages pair.

    The rate-driven coverage order is taken from the ``calculation_methods``
    catalogue, so the portfolio is read without the actuarial basis. The
    engine aligns its coverages to that order at measure time.
    """
    if calculation_methods is None:
        raise ValueError(
            "long-form model points need the calculation_methods taxonomy -- "
            "the per-code pattern routes survival rows (ANNUITY / MATURITY) "
            "to scalar fields and rate-driven rows to the coverage CSR. "
            "Pass a calculation_methods.csv path to read_model_points."
        )
    for need in ("mp_id", "issue_age", "term_months"):
        if need not in pol.columns:
            raise ValueError(
                f"the policies frame is missing required column {need!r}"
            )
    for need in ("mp_id", "coverage_code", "amount"):
        if need not in cov.columns:
            raise ValueError(
                f"the coverages frame is missing required column {need!r}"
            )
    _warn_if_elapsed_months(pol.columns)
    n_mp = pol.height
    # mp_id uniqueness -- a duplicate id would fan out the coverages join
    # (one-to-many) and silently inflate per-policy benefits.
    if pol["mp_id"].n_unique() != n_mp:
        dups = (pol.group_by("mp_id").len()
                   .filter(pl.col("len") > 1)["mp_id"].to_list())
        raise ValueError(
            f"policies frame has duplicate mp_id value(s) {dups[:10]}"
            f"{' (...)' if len(dups) > 10 else ''} -- mp_id must be unique"
        )
    # Premium double-source -- if both the coverages frame's ``premium``
    # column and the policies frame's ``level_premium`` are present, the
    # cov-side branch silently wins below. Reject up front so the operator
    # picks one source.
    if "premium" in cov.columns and "level_premium" in pol.columns:
        raise ValueError(
            "premium is specified twice -- 'premium' in the coverages "
            "frame and 'level_premium' in the policies frame. Pick one: "
            "the coverages-side column sums per coverage to the policy, "
            "the policies-side column is a flat per-policy amount."
        )
    # Coverage-rule columns -- waiting / reduction_end / reduction_factor
    # must arrive together. A reduction_factor without a reduction_end is
    # silently inert (the factor applies for ``t < 0`` months, i.e. never)
    # and almost certainly a user oversight.
    has_rend = "reduction_end" in cov.columns
    has_rfac = "reduction_factor" in cov.columns
    if has_rfac and not has_rend:
        raise ValueError(
            "coverages frame has 'reduction_factor' without 'reduction_end' "
            "-- the factor would never fire (reduction_end defaults to 0). "
            "Add a reduction_end column (months) or drop the factor column."
        )
    ctypes = {k: CalculationMethod(v) for k, v in calculation_methods.items()}
    # Rate-driven coverage order comes from the *catalogue* (calculation_methods),
    # not the assumptions -- so reading the portfolio needs no assumptions.
    # coverage_index integers index this order; the engine aligns
    # Assumptions.coverages to it at measure time (coverage.align_coverages).
    # Only the rate-driven codes that actually appear in this portfolio are
    # kept, in catalogue order.
    present_codes = set(cov["coverage_code"].to_list())
    rate_driven_codes = [c for c, m in ctypes.items()
                         if m in RATE_DRIVEN_PATTERNS and c in present_codes]
    code_to_cov_idx = {c: i for i, c in enumerate(rate_driven_codes)}

    # Resolve every coverage row to its policy index and coverage type.
    pol = pol.with_row_index("_mp")
    cmap = pl.DataFrame({
        "coverage_code": list(ctypes.keys()),
        "_type": [str(v) for v in ctypes.values()],
        "_cov_idx": [code_to_cov_idx.get(c, -1) for c in ctypes],
    })
    cov = (cov.join(pol.select("mp_id", "_mp"), on="mp_id", how="left")
              .join(cmap, on="coverage_code", how="left"))
    if cov["_mp"].null_count():
        bad = sorted({v for v in cov.filter(pl.col("_mp").is_null())
                                    ["mp_id"].to_list() if v is not None})
        raise ValueError(
            f"coverages frame references {len(bad)} unknown mp_id "
            f"value(s) not present in the policies frame: "
            f"{_truncate_list(bad)}"
        )
    if cov["_type"].null_count():
        bad = sorted({v for v in cov.filter(pl.col("_type").is_null())
                                    ["coverage_code"].to_list() if v is not None})
        raise ValueError(
            f"coverages frame references {len(bad)} coverage_code "
            f"value(s) not in the calculation_methods taxonomy: "
            f"{_truncate_list(bad)}"
        )

    mp = cov["_mp"].to_numpy()
    ctype = cov["_type"].to_numpy()
    cov_idx = cov["_cov_idx"].to_numpy().astype(np.int64)
    amount = cov["amount"].to_numpy().astype(np.float64)

    fields: dict[str, object] = dict(
        issue_age=pol["issue_age"].to_numpy(),
        term_months=pol["term_months"].to_numpy(),
    )
    for opt in ("sex", "count", "single_premium", "premium_term_months",
                "premium_frequency_months", "annuity_frequency_months",
                "disability_income", "disability_benefit"):
        if opt in pol.columns:
            fields[opt] = pol[opt].to_numpy()
    for opt in ("product_code", "channel_code"):
        if opt in pol.columns:
            fields[opt] = pol[opt].to_numpy()
    if "state" in pol.columns:
        fields["state"] = _read_state(pol["state"])

    def _by_policy(mask) -> np.ndarray:
        return np.bincount(mp[mask], weights=amount[mask], minlength=n_mp)

    fields["maturity_benefit"] = _by_policy(ctype == CalculationMethod.MATURITY)
    fields["annuity_payment"] = _by_policy(ctype == CalculationMethod.ANNUITY)

    # Premium -- the coverages frame carries it per coverage; sum to the policy.
    if "premium" in cov.columns:
        prem = cov["premium"].fill_null(0.0).to_numpy().astype(np.float64)
        fields["level_premium"] = np.bincount(mp, weights=prem, minlength=n_mp)
    elif "level_premium" in pol.columns:
        fields["level_premium"] = pol["level_premium"].to_numpy()
    else:
        # Neither source provided -- premium is silently zero. A genuine
        # paid-up portfolio is one valid case; a forgotten column is the
        # other. Warn so the latter doesn't slip through.
        warnings.warn(
            "long-form model points have no premium source -- neither "
            "'premium' on the coverages frame nor 'level_premium' on the "
            "policies frame was found. level_premium defaults to zero; "
            "if this portfolio is not fully paid-up, add the column.",
            UserWarning,
            stacklevel=3,
        )
        fields["level_premium"] = np.zeros(n_mp)

    # Coverage list: the rate-driven coverages (codes 0..n-1 indexing
    # ``coverage_codes`` below). annuity / maturity are survival scalars and
    # not part of the CSR. Every rate-driven present code is in
    # ``rate_driven_codes`` by construction, so ``cov_idx >= 0`` here; a code
    # absent from the catalogue was already rejected (the ``_type`` null
    # check above). Whether the assumptions register a rate for each code is
    # checked at measure time (coverage.align_coverages, the V4 guard).
    is_cov = np.isin(ctype, RATE_DRIVEN_PATTERNS)
    order = np.argsort(mp[is_cov], kind="stable")
    cov_mp = mp[is_cov][order]
    fields["coverage_index"] = cov_idx[is_cov][order]
    fields["coverage_amount"] = amount[is_cov][order]

    # Optional per-coverage benefit rules -- a waiting period and a
    # reduced-benefit period, each CSR-aligned with coverage_index.
    for col, field, default in (("waiting", "coverage_waiting", 0),
                                ("reduction_end", "coverage_reduction_end", 0),
                                ("reduction_factor", "coverage_reduction_factor", 1.0)):
        if col in cov.columns:
            rule = cov[col].fill_null(default).to_numpy()
            fields[field] = rule[is_cov][order]

    fields["coverage_offset"] = np.concatenate((
        np.zeros(1, np.int64),
        np.cumsum(np.bincount(cov_mp, minlength=n_mp), dtype=np.int64),
    ))
    fields["calculation_methods"] = ctypes
    # The catalogue order the coverage_index integers were built against.
    # The engine aligns Assumptions.coverages to this at measure time.
    fields["coverage_codes"] = tuple(rate_driven_codes)
    return ModelPoints(**fields)


def read_model_points(
    path: Path | str,
    coverages: Path | str | None = None,
    calculation_methods: Path | str | dict[str, CalculationMethod] | None = None,
) -> ModelPoints:
    """Read model points from a parquet, CSV, Excel or feather file.

    Reads the portfolio **without any assumptions** -- the model points and
    the actuarial basis are separate inputs. The basis enters only at the
    engine call (``measure`` / ``value``), which aligns its coverages to the
    portfolio's coverage order.

    Two forms:

    * **wide** -- ``read_model_points(path)``. One row per policy.
      ``issue_age`` and ``term_months`` are required; ``sex``, ``count``,
      ``state``, ``level_premium``, ``single_premium``,
      ``premium_term_months``, ``premium_frequency_months``,
      ``annuity_frequency_months``, ``maturity_benefit`` and
      ``annuity_payment`` are read if present. Each
      ``<coverage_code>_benefit`` column adds that coverage; the coverage
      order is the ``calculation_methods`` catalogue order when given, else
      the column order (pattern then auto-inferred from the code name).
    * **long-form** -- ``read_model_points(policies,
      coverages=coverages_path, calculation_methods=calculation_methods_path)``.
      A policies frame (``mp_id``, ``issue_age``, ``term_months``,
      optional ``sex`` / ``count`` / ``state`` / ``premium_term_months``)
      and a coverages frame (``mp_id``, ``coverage_code``, ``amount``, and
      optional ``premium`` / ``waiting`` / ``reduction_end`` /
      ``reduction_factor``), one coverage row per policy x coverage.
      A single ``.xlsx`` with ``policies`` and ``coverages`` sheets is read
      as long-form too. ``calculation_methods`` is the company taxonomy file
      (CSV / parquet / feather / xlsx) -- the third side of the Plan-B
      split between *portfolio* (policies + coverages), *basis*
      (assumptions.xlsx) and *catalogue* (calculation_methods.csv).

    The policies frame is the **inception-time static spec** -- issue_age,
    term, sex, and so on. The in-force closing state (elapsed_months,
    prior_csm, lock_in_rate) belongs in a separate file read by
    :func:`read_inforce_state`. An ``elapsed_months`` column on the
    policies side is ignored and a :class:`UserWarning` is emitted; do
    not encode the as-of date by mixing it into the static spec.
    """
    if isinstance(calculation_methods, (str, Path)):
        patterns_dict = _parse_calculation_methods(calculation_methods)
    else:
        patterns_dict = calculation_methods
    p = str(path)
    if coverages is None and p.endswith(".xlsx"):
        wb = openpyxl.load_workbook(p, read_only=True)
        sheets = wb.sheetnames
        wb.close()
        if "policies" in sheets and "coverages" in sheets:
            return _long_model_points(
                pl.read_excel(p, sheet_name="policies", engine="openpyxl"),
                pl.read_excel(p, sheet_name="coverages", engine="openpyxl"),
                patterns_dict,
            )
    pol = _read_frame(path)
    if coverages is not None:
        return _long_model_points(
            pol, _read_frame(coverages), patterns_dict,
        )
    return _wide_model_points(pol, patterns_dict)


def read_inforce_policies(
    path: Path | str,
    coverages: Path | str | None = None,
    calculation_methods: "Path | str | dict[str, CalculationMethod] | None" = None,
) -> "tuple[ModelPoints, InforceState]":
    """Read a single combined policies + in-force state file.

    A self-contained snapshot at a settlement date -- one file per
    valuation date, one row per surviving contract. Columns combine the
    permanent contract spec (``issue_age``, ``sex``, ``term_months``,
    premiums, benefits, ...) and the closing state from the prior period
    (``elapsed_months``, ``count``, ``prior_csm``, ``lock_in_rate``). This
    matches the Korean industry "보유계약 마감파일" pattern -- one
    self-contained snapshot per period, no separate state file to keep
    in sync.

    Returns a ``(ModelPoints, InforceState)`` tuple. The ``ModelPoints``
    has the state's ``elapsed_months`` and ``count`` already folded in;
    the ``InforceState`` carries ``prior_csm`` and ``lock_in_rate`` for
    the in-force valuation call::

        mp, state = fcf.read_inforce_policies(
            "inforce_2026Q1.csv",
            coverages="coverages.csv",
            calculation_methods="calculation_methods.csv",
        )
        val = fcf.value_in_force(
            mp, assumptions, period_months=3,
            prior_csm=state.prior_csm,
            lock_in_rate=state.lock_in_rate,
        )

    For the two-file equivalent (separate ``policies.csv`` +
    ``inforce_state.csv``), see :func:`read_model_points` +
    :func:`read_inforce_state` + :func:`apply_inforce_state`. Both
    workflows produce the same ``ModelPoints`` / ``InforceState`` pair
    and so the same valuation; pick the form that fits the company's
    extract pipeline.

    Required columns: ``mp_id``, ``elapsed_months``, ``count``,
    ``prior_csm``, ``lock_in_rate``, plus whatever the spec side of
    :func:`read_model_points` needs (``issue_age``, ``term_months``,
    optional ``sex``, premiums, ``<code>_benefit`` columns for wide form).
    Variance / movement analysis (:func:`roll_forward`,
    :func:`reconcile`) is unaffected -- mp_id-based matching across
    periods works the same regardless of which reader built each
    snapshot.
    """
    from fastcashflow.modelpoints import (
        InforceState, ModelPoints, apply_inforce_state,
    )

    df = _read_frame(path)
    needed = ("mp_id", "elapsed_months", "count", "prior_csm", "lock_in_rate")
    for col in needed:
        if col not in df.columns:
            raise ValueError(
                f"the in-force policies file is missing required column "
                f"{col!r}. The combined file carries the policies spec "
                "plus the closing-state columns "
                "(elapsed_months, count, prior_csm, lock_in_rate)."
            )
    lock = df["lock_in_rate"].to_numpy().astype(np.float64)
    if lock.size and not np.all(lock == lock[0]):
        raise NotImplementedError(
            "lock_in_rate must be uniform across rows in v1; per-MP "
            "(cohort-aware) lock-in rates are a future extension"
        )
    state = InforceState(
        mp_id=df["mp_id"].to_numpy(),
        elapsed_months=df["elapsed_months"].to_numpy().astype(np.int64),
        count=df["count"].to_numpy().astype(np.float64),
        prior_csm=df["prior_csm"].to_numpy().astype(np.float64),
        lock_in_rate=float(lock[0]) if lock.size else 0.0,
    )

    # Drop the state-only columns before handing the frame to the
    # standard policies reader, which would otherwise warn about
    # ``elapsed_months`` on a policies frame and ignore the rest. ``count``
    # stays -- it is a valid policies column too, and ``apply_inforce_state``
    # will overwrite it with the state value below anyway.
    spec_df = df.drop("elapsed_months", "prior_csm", "lock_in_rate")

    if isinstance(calculation_methods, (str, Path)):
        patterns_dict = _parse_calculation_methods(calculation_methods)
    else:
        patterns_dict = calculation_methods
    if coverages is not None:
        mp = _long_model_points(
            spec_df, _read_frame(coverages), patterns_dict,
        )
    else:
        mp = _wide_model_points(spec_df, patterns_dict)
    mp = apply_inforce_state(mp, state)
    return mp, state


def sample_data_dir() -> Path:
    """Return the on-disk path of the bundled sample data directory.

    The directory contains ``sample_assumptions.xlsx``, ``sample_policies.csv``
    and ``sample_coverages.csv`` -- the inputs behind
    :func:`load_sample_assumptions` and :func:`load_sample_model_points`.
    Use this to open the workbook in Excel and see what a complete
    fastcashflow input looks like before preparing your own.
    """
    return Path(str(resources.files("fastcashflow") / "sample_data"))


def load_sample_assumptions() -> dict[tuple[str, str], Assumptions]:
    """Read fastcashflow's bundled sample assumptions workbook.

    A filled-in workbook packaged with the library, the companion to
    :func:`load_sample_model_points`. See :func:`read_assumptions` for the
    workbook format. The bundled sample has two segments
    (``("term_a", "GA")`` and ``("term_a", "FC")``); pick one to use it as
    a single ``Assumptions``.
    """
    source = resources.files("fastcashflow") / "sample_data" / "sample_assumptions.xlsx"
    with resources.as_file(source) as path:
        return read_assumptions(path)


def load_sample_calculation_methods() -> dict[str, CalculationMethod]:
    """Read fastcashflow's bundled sample benefit-pattern taxonomy.

    The companion to :func:`load_sample_assumptions` and
    :func:`load_sample_model_points` -- the company-level catalogue that
    maps each ``coverage_code`` to its :class:`CalculationMethod`. The same
    file format every portfolio uses (see :func:`read_model_points`
    long-form, ``calculation_methods`` argument).
    """
    source = (
        resources.files("fastcashflow") / "sample_data"
        / "sample_calculation_methods.csv"
    )
    with resources.as_file(source) as path:
        return _parse_calculation_methods(path)


def load_sample_model_points() -> ModelPoints:
    """Read fastcashflow's bundled sample portfolio.

    A small long-form portfolio -- a policies file, a coverages file and
    the benefit-pattern taxonomy -- packaged with the library, so the
    engine can be tried without preparing an input file. See
    :func:`read_model_points` for the file format. The coverage order
    comes from the ``calculation_methods`` catalogue; no assumptions are
    needed to read the portfolio.
    """
    patterns = load_sample_calculation_methods()
    base = resources.files("fastcashflow") / "sample_data"
    with resources.as_file(base / "sample_policies.csv") as policies, \
            resources.as_file(base / "sample_coverages.csv") as coverages:
        return read_model_points(
            policies, coverages=coverages, calculation_methods=patterns,
        )


def load_sample_inforce_state() -> "InforceState":
    """Read fastcashflow's bundled sample in-force state.

    Aligned row-for-row with :func:`load_sample_model_points`. Pair with
    :func:`apply_inforce_state` to fold ``elapsed_months`` and ``count``
    into the sample model points, then call :func:`measure_in_force` or
    :func:`value_in_force` with ``prior_csm`` and ``lock_in_rate`` taken
    from the returned :class:`InforceState`.
    """
    base = resources.files("fastcashflow") / "sample_data"
    with resources.as_file(base / "sample_inforce_state.csv") as path:
        return read_inforce_state(path)


def _drop_sample_table(filename: str, dest: Path | str) -> Path:
    """Drop a packaged single-table sample file at ``dest``, converting to
    whatever format ``dest`` 's extension picks (``.csv`` / ``.parquet`` /
    ``.feather`` / ``.arrow``).

    ``dest`` may be a file path (used as-is) or a directory (file lands
    inside with its original ``sample_*.csv`` name)."""
    src = resources.files("fastcashflow") / "sample_data" / filename
    dest_path = Path(dest)
    if dest_path.is_dir():
        dest_path = dest_path / filename
    with resources.as_file(src) as src_path:
        if dest_path.suffix == src_path.suffix:
            # Same format as the source -- a byte-for-byte copy preserves
            # any formatting the workbook editor cared about.
            import shutil
            shutil.copy2(src_path, dest_path)
        else:
            # Different format -- read the source as a polars frame and
            # let _write_frame route to the right writer by extension.
            _write_frame(_read_frame(src_path), dest_path)
    return dest_path


def save_sample_assumptions(path: Path | str) -> Path:
    """Drop the packaged sample assumptions workbook on disk at ``path``.

    Use this to bootstrap a workbook a reader can open in Excel, inspect,
    and then re-read with :func:`read_assumptions` -- the same call shape
    a real user types against their own file. The bundled sample carries
    seven (product, channel) segments across three products
    (``TERM_LIFE_A``, ``HEALTH_A``, ``WHOLE_LIFE_A``).

    Supported extension: ``.xlsx`` (the workbook carries multiple sheets,
    so single-table formats like CSV are not appropriate here).

    ``path`` may be a file (the workbook lands there) or a directory (the
    workbook lands inside with its original ``sample_assumptions.xlsx``
    name). Returns the resolved destination path.
    """
    import shutil
    src = (resources.files("fastcashflow")
           / "sample_data" / "sample_assumptions.xlsx")
    dest_path = Path(path)
    if dest_path.is_dir():
        dest_path = dest_path / "sample_assumptions.xlsx"
    if dest_path.suffix.lower() != ".xlsx":
        raise ValueError(
            f"save_sample_assumptions: expected an .xlsx path, got "
            f"{str(path)!r}. The assumptions workbook carries multiple "
            "sheets (mortality_tables, lapse_tables, segments, ...) and "
            "single-table formats (csv / parquet / feather) cannot "
            "represent it. Use .xlsx."
        )
    with resources.as_file(src) as src_path:
        shutil.copy2(src_path, dest_path)
    return dest_path


def save_sample_policies(path: Path | str) -> Path:
    """Drop the packaged sample policies file on disk at ``path``.

    The companion to :func:`save_sample_coverages` and
    :func:`save_sample_calculation_methods`. Use the three together with
    :func:`read_model_points` for a copy-paste workflow that mirrors how
    you would read your own files.

    Supported extensions: ``.csv``, ``.xlsx``, ``.parquet``, ``.feather``
    / ``.arrow``. The conversion runs through polars when the requested
    extension differs from the packaged ``.csv`` source. ``.xlsx`` is
    capped at 1,048,576 rows per sheet -- for production-scale
    portfolios use ``.parquet`` or ``.feather``.
    """
    return _drop_sample_table("sample_policies.csv", path)


def save_sample_coverages(path: Path | str) -> Path:
    """Drop the packaged sample coverages file on disk at ``path``.

    Long-form coverage entries -- one row per (model point, coverage_code)
    -- the companion to :func:`save_sample_policies`. A long-form
    portfolio has roughly ``n_mp x avg_coverages_per_mp`` rows here, so
    this is the file most likely to exceed the 1,048,576 row cap of
    ``.xlsx``.

    Supported extensions: ``.csv``, ``.xlsx``, ``.parquet``, ``.feather``
    / ``.arrow``. ``.parquet`` or ``.feather`` for production scale.
    """
    return _drop_sample_table("sample_coverages.csv", path)


def save_sample_calculation_methods(path: Path | str) -> Path:
    """Drop the packaged sample benefit-pattern catalogue on disk at ``path``.

    The company catalogue file -- one row per ``coverage_code`` mapping
    it to its :class:`CalculationMethod`. Tens-to-hundreds of rows in
    practice; ``.xlsx`` row cap never binds.

    Supported extensions: ``.csv``, ``.xlsx``, ``.parquet``, ``.feather``
    / ``.arrow``.
    """
    return _drop_sample_table("sample_calculation_methods.csv", path)


def save_sample_inforce_state(path: Path | str) -> Path:
    """Drop the packaged sample in-force state file on disk at ``path``.

    The dynamic state-at-valuation companion to the static
    :func:`save_sample_policies` file: one row per ``mp_id`` carrying
    the closing state from the prior reporting period
    (``elapsed_months``, ``count``, ``prior_csm``, ``lock_in_rate``).
    Pair the dropped file with :func:`read_inforce_state` and feed the
    result through :func:`apply_inforce_state` before
    :func:`measure_in_force` / :func:`value_in_force` -- the
    subsequent-measurement workflow at each period close.

    Supported extensions: ``.csv``, ``.xlsx``, ``.parquet``, ``.feather``
    / ``.arrow``. One row per contract, so the ``.xlsx`` row cap
    (~1M / sheet) binds at the same scale as the policies file.
    """
    return _drop_sample_table("sample_inforce_state.csv", path)


def save_sample_inforce_policies(path: Path | str) -> Path:
    """Drop a combined policies + in-force state sample file on disk at ``path``.

    The companion to :func:`read_inforce_policies`. Each row carries
    the permanent spec (issue_age, sex, term_months, premium_term_months,
    product_code, channel_code) and the closing state from the prior
    period (elapsed_months, count, prior_csm, lock_in_rate). Built on
    the fly by joining the packaged ``sample_policies.csv`` and
    ``sample_inforce_state.csv`` on ``mp_id``; ``count`` is the state
    value (post-decrement), not the inception count.

    Supported extensions: ``.csv``, ``.xlsx``, ``.parquet``, ``.feather``
    / ``.arrow``. ``.xlsx`` is capped at ~1M rows / sheet.
    """
    base = resources.files("fastcashflow") / "sample_data"
    with resources.as_file(base / "sample_policies.csv") as policies, \
            resources.as_file(base / "sample_inforce_state.csv") as state:
        pol = pl.read_csv(policies)
        st = pl.read_csv(state)
    # ``count`` is on both files; drop the inception count from policies so
    # the state's post-decrement count is the one that survives the join.
    pol = pol.drop("count")
    combined = pol.join(st, on="mp_id", how="inner")
    dest_path = Path(path)
    if dest_path.is_dir():
        dest_path = dest_path / "sample_inforce_policies.csv"
    _write_frame(combined, dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# Economic scenarios
# ---------------------------------------------------------------------------

def read_inforce_state(path: Path | str) -> "InforceState":
    """Read an in-force state file -- the per-MP closing state from the
    prior reporting period.

    The file has one row per model point with columns ``mp_id``,
    ``elapsed_months``, ``count``, ``prior_csm`` and ``lock_in_rate``.
    Reads ``.parquet``, ``.csv``, ``.xlsx`` or ``.feather`` / ``.arrow``
    via :func:`_read_frame`.

    Pair with :func:`apply_inforce_state` to join the state onto a
    :class:`ModelPoints` built from the static policies file, then pass
    the result to :func:`value_in_force` or :func:`measure_in_force` with
    ``prior_csm`` and ``lock_in_rate`` taken from the returned
    :class:`InforceState`.

    ``lock_in_rate`` is required to be uniform across rows in v1 -- the
    engine takes a scalar locked-in rate. Cohort-aware per-MP rates are
    a future extension; for now the reader errors out if the column is
    not constant rather than silently dropping the per-row detail.
    """
    from fastcashflow.modelpoints import InforceState
    df = _read_frame(path)
    needed = ("mp_id", "elapsed_months", "count", "prior_csm", "lock_in_rate")
    for col in needed:
        if col not in df.columns:
            raise ValueError(
                f"the in-force state file is missing required column {col!r}"
            )
    lock = df["lock_in_rate"].to_numpy().astype(np.float64)
    if lock.size and not np.all(lock == lock[0]):
        raise NotImplementedError(
            "lock_in_rate must be uniform across rows in v1; per-MP "
            "(cohort-aware) lock-in rates are a future extension"
        )
    return InforceState(
        mp_id=df["mp_id"].to_numpy(),
        elapsed_months=df["elapsed_months"].to_numpy().astype(np.int64),
        count=df["count"].to_numpy().astype(np.float64),
        prior_csm=df["prior_csm"].to_numpy().astype(np.float64),
        lock_in_rate=float(lock[0]) if lock.size else 0.0,
    )


def read_scenarios(path: Path | str) -> FloatArray:
    """Read a stochastic scenario set from a file.

    The file is a 2-D table -- one row per scenario, one column per
    projection month, every cell a rate or return. Reads ``.parquet``,
    ``.csv``, ``.xlsx`` or ``.feather`` / ``.arrow`` via :func:`_read_frame`.

    Returns a numpy ``float64`` array of shape ``(n_scenarios, n_time)``,
    or ``(n_scenarios,)`` when the file has a single column (flat-rate
    scenarios). The result is what :func:`value_stochastic` and
    :func:`measure_tvog` accept as their ``scenarios`` / ``return_scenarios``
    input.

    Calibration -- Hull-White, Vasicek, regime-switching, climate paths,
    etc. -- is left to a separate scenario-generator step; this reader is
    just the storage / handover layer. For large scenario sets (thousands
    of paths) prefer ``.parquet`` or ``.feather`` over ``.xlsx``.
    """
    df = _read_frame(path)
    arr = df.to_numpy().astype(np.float64)
    if arr.shape[1] == 1:
        return arr[:, 0]
    return arr


# ---------------------------------------------------------------------------
# Valuation results
# ---------------------------------------------------------------------------

def write_valuation(
    valuation: "Valuation",
    path: Path | str,
    *,
    ids: np.ndarray | None = None,
) -> None:
    """Write a ``Valuation`` to a parquet or CSV file.

    One row per model point, in model-point order, with columns ``bel``,
    ``ra``, ``csm`` and ``loss_component``. If ``ids`` is given it is written
    as a leading ``id`` column so the results can be joined back to policies.
    """
    columns: dict[str, np.ndarray] = {}
    if ids is not None:
        columns["id"] = np.asarray(ids)
    columns["bel"] = valuation.bel
    columns["ra"] = valuation.ra
    columns["csm"] = valuation.csm
    columns["loss_component"] = valuation.loss_component
    _write_frame(pl.DataFrame(columns), path)


def value_file(
    input_path: Path | str,
    output_dir: Path | str,
    assumptions: Assumptions,
    *,
    coverages: Path | str | None = None,
    calculation_methods: Path | str | dict[str, CalculationMethod] | None = None,
    chunk_size: int = 20_000_000,
    backend: str = "cpu",
    id_column: str | None = None,
) -> int:
    """Stream a valuation through a parquet file one chunk at a time.

    Reads the input in chunks of ``chunk_size`` model points, values each
    chunk with :func:`value`, and writes the results as a parquet dataset --
    one ``part-NNNNN.parquet`` file per chunk -- under ``output_dir``. Peak
    memory is a single chunk, so this scales past what an in-memory run could
    hold.

    The input format mirrors :func:`read_model_points`:

    * **wide** -- ``input_path`` is a wide parquet file; ``coverages`` is
      ``None``. Each chunk of rows is a self-contained set of model points.
    * **long-form** -- ``input_path`` is the policies parquet and
      ``coverages`` the coverages parquet. Each chunk of policies pulls its
      coverage rows by ``mp_id``, so sorting the coverages file by
      ``mp_id`` lets the parquet reader prune row groups.

    Returns the total number of model points processed.
    """
    # Lazy import -- only ``value_file`` actually drives a valuation, so we
    # keep the engine import off the I/O hot path. A script that only reads
    # model points or writes results never pays the engine import cost.
    from fastcashflow.engine import value

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if input_path.suffix != ".parquet":
        raise ValueError(
            f"value_file streams parquet input only; got {str(input_path)!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("part-*.parquet")):
        raise ValueError(
            f"output directory {str(output_dir)!r} already contains part "
            "files; use a fresh directory"
        )

    if isinstance(calculation_methods, (str, Path)):
        patterns_dict = _parse_calculation_methods(calculation_methods)
    else:
        patterns_dict = calculation_methods
    scan = pl.scan_parquet(input_path)
    n_total = scan.select(pl.len()).collect().item()
    processed = 0

    if coverages is not None:
        # long-form: chunk the policies, pull each chunk's coverage rows.
        cov_scan = pl.scan_parquet(Path(coverages))
        for part, offset in enumerate(range(0, n_total, chunk_size)):
            pol = scan.slice(offset, chunk_size).collect()
            ids = pol["mp_id"]
            cov = cov_scan.join(
                pol.lazy().select("mp_id"), on="mp_id", how="semi"
            ).collect()
            model_points = _long_model_points(pol, cov, patterns_dict)
            write_valuation(
                value(model_points, assumptions, backend=backend),
                output_dir / f"part-{part:05d}.parquet",
                ids=ids.to_numpy(),
            )
            processed += model_points.n_mp
        return processed

    # wide: each chunk of rows is a self-contained set of model points.
    available = scan.collect_schema().names()
    for need in ("issue_age", "term_months"):
        if need not in available:
            raise ValueError(
                f"{str(input_path)!r} is missing required column {need!r}"
            )
    if id_column is not None and id_column not in available:
        raise ValueError(f"{str(input_path)!r} has no id column {id_column!r}")
    columns = [c for c in available
               if c in _NAMED_WIDE or c.endswith("_benefit") or c == id_column]
    projected = scan.select(columns)
    for part, offset in enumerate(range(0, n_total, chunk_size)):
        chunk = projected.slice(offset, chunk_size).collect()
        model_points = _wide_model_points(
            chunk.drop(id_column) if id_column is not None else chunk,
            patterns_dict,
        )
        ids = chunk[id_column].to_numpy() if id_column is not None else None
        write_valuation(
            value(model_points, assumptions, backend=backend),
            output_dir / f"part-{part:05d}.parquet",
            ids=ids,
        )
        processed += model_points.n_mp

    return processed
