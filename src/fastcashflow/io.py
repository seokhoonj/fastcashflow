"""File I/O for model points, the actuarial basis and valuation results.

Model points and results go through polars; the actuarial basis -- read by
:func:`read_assumptions` -- comes from an Excel workbook via openpyxl.

Model points come in two shapes, both producing the same ``ModelPoints``:

* **wide** -- one row per policy, every benefit a column: ``death_benefit``,
  ``maturity_benefit``, ``annuity_payment`` and a ``<rider_code>_benefit``
  column per rider. The convenient form for a single, homogeneous product.
* **long-form** -- a policies frame (contract attributes) plus a coverages
  frame, one row per policy x rider carrying ``amount`` and ``premium``. The
  form for a heterogeneous, multi-product portfolio.

:func:`read_model_points` reads either; ``ModelPoints.to_wide`` /
``ModelPoints.to_long`` convert between them.

The core engine stays identifier-free: the kernel never needs a policy id, so
none is carried through ``ModelPoints`` or ``Valuation``. Identifiers are a
file-boundary concern -- pass them to :func:`write_valuation` (or via
``value_file``'s ``id_column``) to join results back to policies.
"""
from __future__ import annotations

import importlib.resources as resources
from pathlib import Path

import numpy as np
import openpyxl
import polars as pl

from fastcashflow.assumptions import Assumptions, RiderRate
from fastcashflow.coverage import (
    RATE_DRIVEN_TYPES, RISK_MORBIDITY, RISK_MORTALITY, TYPE_ANNUITY,
    TYPE_DEATH, TYPE_DEATH_MAIN, TYPE_DIAGNOSIS, TYPE_MATURITY,
)
from fastcashflow.engine import Valuation, value
from fastcashflow.modelpoints import STATE_ACTIVE, STATE_NAMES, ModelPoints

# Wide model-point columns with a fixed meaning. Any other ``*_benefit``
# column names a rider by its rider code.
_NAMED_WIDE = frozenset((
    "policy_id", "product", "issue_age", "term_months", "sex", "count",
    "state", "level_premium", "single_premium", "premium_term_months",
    "premium_frequency_months", "annuity_frequency_months",
    "death_benefit", "maturity_benefit", "annuity_payment",
    "disability_income", "disability_benefit",
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
    else:
        raise ValueError(
            f"unsupported file type: {path!r} "
            "(expected .parquet, .csv or .feather)"
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
#   * ``riders``         -- (product) -> rider_code, type, optional rate_table.
#   * ``mortality_tables``, ``rider_rate_tables``, ``waiver_tables``,
#     ``lapse_tables``, ``maintenance_tables``, ``discount_tables``,
#     ``inflation_tables`` -- the named rate tables the segments reference.
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


# Axes a rate table may carry, in the order they index the internal grid.
# A sheet may include any subset; missing axes broadcast (the rate is held
# flat over that axis at lookup time). ``age`` (attained) is mutually
# exclusive with ``issue_age`` / ``duration`` (select-and-ultimate schema).
_RATE_AXES = ("sex", "issue_age", "duration", "age")


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
    axes = tuple(a for a in _RATE_AXES if a in header)
    if "age" in axes and ("issue_age" in axes or "duration" in axes):
        raise ValueError(
            f"sheet {ws.title!r} mixes 'age' (attained) with "
            "'issue_age' / 'duration' (select schema) -- pick one"
        )

    by_id: dict[str, list] = {}
    for r in rows:
        tid = str(r["table_id"]).strip()
        key = tuple(int(r[a]) for a in axes)
        by_id.setdefault(tid, []).append((key, float(r[value_col])))
    return {tid: _build_rate_callable(axes, entries, ws.title, tid)
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

        def rate(sex, issue_age, duration):
            shape = np.broadcast_shapes(
                np.asarray(sex).shape, np.asarray(issue_age).shape,
                np.asarray(duration).shape,
            )
            return np.full(shape, val, dtype=np.float64)
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

    def rate(sex, issue_age, duration):
        sex = np.asarray(sex, dtype=np.int64)
        issue_age = np.asarray(issue_age, dtype=np.int64)
        duration = np.asarray(duration, dtype=np.int64)
        # One index array per axis present in the table.
        idxs = []
        for i, a in enumerate(axes):
            if a == "sex":
                raw = sex
            elif a == "age":
                raw = issue_age + duration                # attained age
            elif a == "issue_age":
                raw = issue_age
            else:                                          # duration
                raw = duration
            idxs.append(np.clip(raw - int(mins[i]), 0, shape[i] - 1))
        # Broadcast each index to the input's full broadcast shape so that
        # numpy fancy-indexing returns a result of that shape (axes absent
        # from the table contribute through broadcast, not indexing).
        target = np.broadcast_shapes(sex.shape, issue_age.shape, duration.shape)
        return grid[tuple(np.broadcast_to(ix, target) for ix in idxs)]
    return rate


def _read_ae_factors(ws):
    """Read the optional ``ae_factors`` sheet.

    Each row is one (product, channel, rider_code) -> factor (a runtime
    multiplier on the base rate). Optional axis columns
    ``{sex, age, issue_age, duration}`` let the factor vary along those
    dimensions (same schema-detection rules as the base rate tables); missing
    axes broadcast. ``channel`` empty matches the segment whose channel is
    blank (a single-segment workbook).

    Returns ``{(product, channel, rider_code): callable(sex, issue_age,
    duration) -> factor}``. Missing sheet -> empty dict -> no A/E adjustment.
    """
    rows = list(_sheet_dicts(ws))
    if not rows:
        return {}
    header = set(rows[0].keys())
    axes = tuple(a for a in _RATE_AXES if a in header)
    if "age" in axes and ("issue_age" in axes or "duration" in axes):
        raise ValueError(
            f"sheet {ws.title!r} mixes 'age' (attained) with "
            "'issue_age' / 'duration' (select schema) -- pick one"
        )

    by_key: dict[tuple, list] = {}
    for r in rows:
        product = str(r["product"]).strip()
        ch = r.get("channel")
        channel = str(ch).strip() if ch not in (None, "") else ""
        rider_code = str(r["rider_code"]).strip()
        key = (product, channel, rider_code)
        axes_key = tuple(int(r[a]) for a in axes)
        by_key.setdefault(key, []).append((axes_key, float(r["factor"])))
    return {
        key: _build_rate_callable(axes, entries, ws.title, "/".join(key))
        for key, entries in by_key.items()
    }


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

    def improved(sex, issue_age, duration):
        d = np.asarray(duration, dtype=np.int64)
        idx = np.clip(d, 0, n - 1)
        return rate_fn(sex, issue_age, duration) * improvement_curve[idx]
    return improved


def _with_ae_factor(rate_fn, factor_fn):
    """Wrap a rate callable to multiply by an A/E factor at call time.

    ``factor_fn`` shares the ``(sex, issue_age, duration) -> array``
    signature; ``None`` (no factor configured for this rider) returns
    ``rate_fn`` unchanged.
    """
    if factor_fn is None or rate_fn is None:
        return rate_fn

    def adjusted(sex, issue_age, duration):
        return rate_fn(sex, issue_age, duration) * factor_fn(sex, issue_age, duration)
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

    def shifted(sex, issue_age, duration):
        return rate_fn(sex, issue_age + shift, duration)
    return shifted


def _axis_tables(ws, axis, *, value_col="rate"):
    """``{table_id: value array}`` from a sheet keyed by ``axis`` (0-based).

    ``value_col`` names the column carrying the per-axis value -- ``"rate"``
    for rate / probability sheets, ``"amount"`` for currency sheets
    (maintenance expense). The column-name distinction documents units;
    a probability and a currency amount should not share a column name.
    """
    by_id: dict[str, dict] = {}
    for r in _sheet_dicts(ws):
        tid = str(r["table_id"]).strip()
        by_id.setdefault(tid, {})[int(r[axis])] = float(r[value_col])
    return {tid: np.asarray([by_k[k] for k in range(len(by_k))], np.float64)
            for tid, by_k in by_id.items()}


def read_assumptions(path):
    """Read the assumptions workbook into a per-segment ``Assumptions`` dict.

    ``path`` is a single ``assumptions.xlsx`` workbook holding both the rate
    tables and the segment mapping (see the module header for the sheet
    layout). The ``segments`` sheet maps each (product, channel) to which
    tables it uses plus scalar parameters, with a ``defaults`` row whose
    values blank cells inherit; the ``riders`` sheet attaches riders to
    products.

    Returns ``{(product, channel): Assumptions}`` -- one basis per segment.

    v1: the discount, inflation and maintenance tables are read but used
    flat (their first entry); the per-segment dict is returned for the
    caller to value segment by segment.
    """
    wb = openpyxl.load_workbook(path, data_only=True)

    def optional(sheet, reader):
        return reader(wb[sheet]) if sheet in wb.sheetnames else {}

    mortality_t = _flex_rate_table(wb["mortality_tables"])
    rider_rate_t = optional("rider_rate_tables", _flex_rate_table)
    waiver_t = optional("waiver_tables", _flex_rate_table)
    lapse_t = _flex_rate_table(wb["lapse_tables"])
    maint_t = optional("maintenance_tables",
                       lambda w: _axis_tables(w, "duration", value_col="amount"))
    discount_t = _axis_tables(wb["discount_tables"], "year")
    inflation_t = _axis_tables(wb["inflation_tables"], "year")
    ae_factors = optional("ae_factors", _read_ae_factors)
    improvement_t = optional(
        "improvement_tables",
        lambda w: _axis_tables(w, "year", value_col="factor"),
    )

    defaults: dict = {}
    segments: list = []
    for r in _sheet_dicts(wb["segments"]):
        if str(r.get("product", "") or "").strip().lower() == "defaults":
            defaults = r
        else:
            segments.append(r)
    riders_by_product: dict[str, list] = {}
    if "riders" in wb.sheetnames:
        for r in _sheet_dicts(wb["riders"]):
            rt = r.get("rate_table")
            riders_by_product.setdefault(str(r["product"]).strip(), []).append((
                str(r["rider_code"]).strip(), str(r["type"]).strip(),
                str(rt).strip() if rt not in (None, "") else None,
            ))

    result = {}
    for seg in segments:
        product = str(seg["product"]).strip()
        channel = str(seg.get("channel", "") or "").strip()
        where = f"segments row ({product} / {channel})"

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

        def ae(rider_code):
            return ae_factors.get((product, channel, rider_code))

        riders = []
        coverage_types: dict[str, str] = {}
        for code, rtype, rate_table in riders_by_product.get(product, []):
            coverage_types[code] = rtype
            if rtype not in RATE_DRIVEN_TYPES:
                continue
            if rate_table is None or rate_table not in rider_rate_t:
                raise ValueError(
                    f"rider {code!r} of product {product!r} is rate-driven "
                    f"({rtype}) but rate_table {rate_table!r} is not in "
                    "rider_rate_tables"
                )
            rate_fn = rider_rate_t[rate_table]
            rate_fn = _with_age_shift(rate_fn, shift_morb)
            rate_fn = _with_ae_factor(rate_fn, ae(code))
            riders.append(RiderRate(
                code=code,
                rate=rate_fn,
                is_diagnosis=(rtype == TYPE_DIAGNOSIS),
                risk=RISK_MORTALITY if rtype == TYPE_DEATH else RISK_MORBIDITY,
            ))

        mortality_fn = lookup(mortality_t, "mortality_table")
        mortality_fn = _with_age_shift(mortality_fn, shift_mort)
        mortality_fn = _with_ae_factor(mortality_fn, ae("dth_main"))
        improvement_curve = lookup(
            improvement_t, "mortality_improvement_table", optional_ref=True,
        )
        mortality_fn = _with_improvement(mortality_fn, improvement_curve)

        waiver = lookup(waiver_t, "waiver_table", optional_ref=True)
        waiver_fn = _with_age_shift(waiver, shift_wvr)

        maint = lookup(maint_t, "maintenance_table", optional_ref=True)
        kwargs: dict = dict(
            mortality_annual=mortality_fn,
            lapse_annual=lookup(lapse_t, "lapse_table"),
            waiver_inception_annual=waiver_fn,
            # Pass the full per-year arrays through -- the engine expands
            # them to per-month curves via fastcashflow.curves. A one-row
            # table reproduces the original flat-scalar behaviour.
            discount_annual=lookup(discount_t, "discount_table"),
            expense_inflation=lookup(inflation_t, "inflation_table"),
            expense_maintenance_annual=(0.0 if maint is None else maint),
            expense_acquisition=scalar("expense_acquisition", required=True),
            ra_confidence=scalar("ra_confidence", required=True),
            mortality_cv=scalar("mortality_cv", required=True),
            riders=tuple(riders),
            coverage_types=coverage_types or None,
        )
        for opt_col in ("morbidity_cv", "longevity_cv", "disability_cv",
                        "expense_cv", "cost_of_capital_rate",
                        "investment_return", "fund_fee",
                        "guaranteed_credit_rate"):
            v = scalar(opt_col)
            if v is not None:
                kwargs[opt_col] = v
        method = cell("ra_method")
        if method is not None:
            kwargs["ra_method"] = str(method).strip()
        result[(product, channel)] = Assumptions(**kwargs)
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
        out = np.empty(len(col), dtype=np.int64)
        for i, v in enumerate(col):
            name = "" if v is None else str(v).strip().lower()
            name = name.replace(" ", "").replace("-", "").replace("_", "")
            if name == "":
                out[i] = STATE_ACTIVE
            elif name in STATE_NAMES:
                out[i] = STATE_NAMES[name]
            else:
                raise ValueError(
                    f"unknown contract state {v!r}; "
                    f"expected one of {sorted(STATE_NAMES)}"
                )
        return out
    return col.fill_null(STATE_ACTIVE).to_numpy().astype(np.int64)


def _wide_model_points(df: pl.DataFrame, assumptions) -> ModelPoints:
    """Build a ``ModelPoints`` from a wide frame -- one row per policy, each
    rider a ``<rider_code>_benefit`` column."""
    for need in ("issue_age", "term_months"):
        if need not in df.columns:
            raise ValueError(
                f"the model-point file is missing required column {need!r}"
            )
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
                "death_benefit", "maturity_benefit", "annuity_payment",
                "disability_income", "disability_benefit", "account_value"):
        if opt in df.columns:
            fields[opt] = df[opt].to_numpy()
    # Segment metadata -- optional string columns; route to value_segmented.
    for opt in ("product", "channel"):
        if opt in df.columns:
            fields[opt] = df[opt].to_numpy()
    if "state" in df.columns:
        fields["state"] = _read_state(df["state"])

    code_to_kind = {r.code: i + 1 for i, r in enumerate(
        assumptions.riders if assumptions is not None else ())}
    benefits: dict[int, np.ndarray] = {}
    for col in df.columns:
        if not col.endswith("_benefit") or col in _NAMED_WIDE:
            continue
        code = col[: -len("_benefit")]
        if code not in code_to_kind:
            raise ValueError(
                f"wide column {col!r} names rider {code!r}, which is not a "
                "rate-driven rider in the assumptions"
            )
        benefits[code_to_kind[code]] = df[col].to_numpy()
    if benefits:
        fields["benefits"] = benefits
    return ModelPoints(**fields)


def _long_model_points(pol: pl.DataFrame, cov: pl.DataFrame,
                       assumptions) -> ModelPoints:
    """Build a ``ModelPoints`` from a long-form policies + coverages pair."""
    if assumptions is None:
        raise ValueError(
            "long-form model points need the assumptions -- the rider-code "
            "registry maps coverage rows to engine codes"
        )
    for need in ("policy_id", "issue_age", "term_months"):
        if need not in pol.columns:
            raise ValueError(
                f"the policies frame is missing required column {need!r}"
            )
    for need in ("policy_id", "rider_code", "amount"):
        if need not in cov.columns:
            raise ValueError(
                f"the coverages frame is missing required column {need!r}"
            )
    n_mp = pol.height
    ctypes = assumptions.coverage_types or {}
    code_to_kind = {r.code: i + 1 for i, r in enumerate(assumptions.riders)}

    # Resolve every coverage row to its policy index and coverage type.
    pol = pol.with_row_index("_mp")
    cmap = pl.DataFrame({
        "rider_code": list(ctypes.keys()),
        "_type": list(ctypes.values()),
        "_kind": [code_to_kind.get(c, 0) for c in ctypes],
    })
    cov = (cov.join(pol.select("policy_id", "_mp"), on="policy_id", how="left")
              .join(cmap, on="rider_code", how="left"))
    if cov["_mp"].null_count():
        raise ValueError("a coverage row references an unknown policy_id")
    if cov["_type"].null_count():
        raise ValueError("a coverage row references an unregistered rider_code")

    mp = cov["_mp"].to_numpy()
    ctype = cov["_type"].to_numpy()
    kind = cov["_kind"].to_numpy().astype(np.int64)
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
    for opt in ("product", "channel"):
        if opt in pol.columns:
            fields[opt] = pol[opt].to_numpy()
    if "state" in pol.columns:
        fields["state"] = _read_state(pol["state"])

    def _by_policy(mask) -> np.ndarray:
        return np.bincount(mp[mask], weights=amount[mask], minlength=n_mp)

    fields["maturity_benefit"] = _by_policy(ctype == TYPE_MATURITY)
    fields["annuity_payment"] = _by_policy(ctype == TYPE_ANNUITY)

    # Premium -- the coverages frame carries it per rider; sum to the policy.
    if "premium" in cov.columns:
        prem = cov["premium"].fill_null(0.0).to_numpy().astype(np.float64)
        fields["level_premium"] = np.bincount(mp, weights=prem, minlength=n_mp)
    elif "level_premium" in pol.columns:
        fields["level_premium"] = pol["level_premium"].to_numpy()
    else:
        fields["level_premium"] = np.zeros(n_mp)

    # Coverage list: the main-contract death (code 0) and the rate-driven
    # riders (codes 1..n). annuity / maturity are survival scalars, not here.
    is_cov = (ctype == TYPE_DEATH_MAIN) | np.isin(ctype, RATE_DRIVEN_TYPES)
    order = np.argsort(mp[is_cov], kind="stable")
    cov_mp = mp[is_cov][order]
    fields["cov_kind"] = kind[is_cov][order]
    fields["cov_amount"] = amount[is_cov][order]

    # Optional per-coverage benefit rules -- a waiting period and a
    # reduced-benefit period, each CSR-aligned with cov_kind.
    for col, field, default in (("waiting", "cov_waiting", 0),
                                ("reduction_end", "cov_reduction_end", 0),
                                ("reduction_factor", "cov_reduction_factor", 1.0)):
        if col in cov.columns:
            rule = cov[col].fill_null(default).to_numpy()
            fields[field] = rule[is_cov][order]

    fields["cov_offset"] = np.concatenate((
        np.zeros(1, np.int64),
        np.cumsum(np.bincount(cov_mp, minlength=n_mp), dtype=np.int64),
    ))
    return ModelPoints(**fields)


def read_model_points(path, assumptions=None, coverages=None) -> ModelPoints:
    """Read model points from a parquet, CSV, Excel or feather file.

    Two forms:

    * **wide** -- ``read_model_points(path, assumptions)``. One row per
      policy. ``issue_age`` and ``term_months`` are required; ``sex``,
      ``count``, ``state``, ``level_premium``, ``single_premium``,
      ``premium_term_months``, ``premium_frequency_months``,
      ``annuity_frequency_months``, ``death_benefit``, ``maturity_benefit``
      and ``annuity_payment`` are read if present. A ``<rider_code>_benefit``
      column adds that rider's coverage; the ``assumptions`` resolve the
      rider code to an engine code.
    * **long-form** -- ``read_model_points(policies, assumptions,
      coverages=coverages_path)``. A policies frame (``policy_id``,
      ``issue_age``, ``term_months``, optional ``sex`` / ``count`` /
      ``state`` / ``premium_term_months``) and a coverages frame
      (``policy_id``, ``rider_code``,
      ``amount``, and
      optional ``premium`` / ``waiting`` / ``reduction_end`` /
      ``reduction_factor``), one coverage row per policy x rider. A single
      ``.xlsx``
      with ``policies`` and ``coverages`` sheets is read as long-form too.

    ``assumptions`` is optional only for a wide file with no rider columns.
    """
    p = str(path)
    if coverages is None and p.endswith(".xlsx"):
        wb = openpyxl.load_workbook(p, read_only=True)
        sheets = wb.sheetnames
        wb.close()
        if "policies" in sheets and "coverages" in sheets:
            return _long_model_points(
                pl.read_excel(p, sheet_name="policies", engine="openpyxl"),
                pl.read_excel(p, sheet_name="coverages", engine="openpyxl"),
                assumptions,
            )
    pol = _read_frame(path)
    if coverages is not None:
        return _long_model_points(pol, _read_frame(coverages), assumptions)
    return _wide_model_points(pol, assumptions)


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


def load_sample_model_points() -> ModelPoints:
    """Read fastcashflow's bundled sample portfolio.

    A small long-form portfolio -- a policies file and a coverages file --
    packaged with the library, so the engine can be tried without preparing
    an input file. See :func:`read_model_points` for the file format. The
    coverage list comes from any segment's ``Assumptions`` -- all bundled
    segments share the same product and so the same rider master.
    """
    basis = load_sample_assumptions()
    assumptions = next(iter(basis.values()))
    base = resources.files("fastcashflow") / "sample_data"
    with resources.as_file(base / "sample_policies.csv") as policies, \
            resources.as_file(base / "sample_coverages.csv") as coverages:
        return read_model_points(policies, assumptions, coverages=coverages)


# ---------------------------------------------------------------------------
# Economic scenarios
# ---------------------------------------------------------------------------

def read_scenarios(path) -> np.ndarray:
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

def write_valuation(valuation: Valuation, path, *, ids=None) -> None:
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
    input_path,
    output_dir,
    assumptions: Assumptions,
    *,
    coverages=None,
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
      coverage rows by ``policy_id``, so sorting the coverages file by
      ``policy_id`` lets the parquet reader prune row groups.

    Returns the total number of model points processed.
    """
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

    scan = pl.scan_parquet(input_path)
    n_total = scan.select(pl.len()).collect().item()
    processed = 0

    if coverages is not None:
        # long-form: chunk the policies, pull each chunk's coverage rows.
        cov_scan = pl.scan_parquet(Path(coverages))
        for part, offset in enumerate(range(0, n_total, chunk_size)):
            pol = scan.slice(offset, chunk_size).collect()
            ids = pol["policy_id"]
            cov = cov_scan.join(
                pol.lazy().select("policy_id"), on="policy_id", how="semi"
            ).collect()
            model_points = _long_model_points(pol, cov, assumptions)
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
            assumptions,
        )
        ids = chunk[id_column].to_numpy() if id_column is not None else None
        write_valuation(
            value(model_points, assumptions, backend=backend),
            output_dir / f"part-{part:05d}.parquet",
            ids=ids,
        )
        processed += model_points.n_mp

    return processed
