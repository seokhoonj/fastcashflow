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
# column names a rider by its 특약코드.
_NAMED_WIDE = frozenset((
    "policy_id", "product", "issue_age", "term_months", "sex", "count",
    "state", "monthly_premium", "single_premium", "premium_term_months",
    "death_benefit", "maturity_benefit", "annuity_payment",
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
# Actuarial basis -- the assumption workbook
# ---------------------------------------------------------------------------

def _read_rate_grid(ws):
    """Read a long-form ``sex, age, rate`` sheet into a sex x age grid.

    Each row after the header is a sex (0 male, 1 female), an attained age,
    and the annual rate at that age. Returns the grid shaped ``(2, n_ages)``
    and the minimum age; the ages are the same contiguous set for both sexes.
    """
    by_sex: dict[int, dict[int, float]] = {0: {}, 1: {}}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None or row[1] is None:
            continue
        by_sex[int(row[0])][int(row[1])] = float(row[2])
    ages = sorted(by_sex[0])
    grid = np.asarray(
        [[by_sex[s][a] for a in ages] for s in (0, 1)], dtype=np.float64
    )
    return grid, ages[0]


def _rate_closure(grid, age_min):
    """Wrap a ``(2, n_ages)`` annual-rate grid in a monthly-rate lookup.

    The returned callable has the ``(sex, issue_age, duration)`` signature
    ``Assumptions`` expects; it reads the rate at the attained age
    ``issue_age + duration``, clipping to the grid.
    """
    n_sex, n_ages = grid.shape

    def rate(sex, issue_age, duration):
        s = np.clip(np.asarray(sex, np.int64), 0, n_sex - 1)
        attained = (np.asarray(issue_age, np.int64)
                    + np.asarray(duration, np.int64))
        a = np.clip(attained - age_min, 0, n_ages - 1)
        return 1.0 - (1.0 - grid[s, a]) ** (1.0 / 12.0)

    return rate


def _read_rates_sheet(ws):
    """Read the long-form ``rates`` sheet -- ``rider_code, sex, age, rate`` --
    into a per-rider sex x age grid keyed by 특약코드."""
    by_code: dict[str, dict[int, dict[int, float]]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None or row[1] is None or row[2] is None:
            continue
        code = str(row[0]).strip()
        by_code.setdefault(code, {0: {}, 1: {}})[int(row[1])][int(row[2])] = (
            float(row[3])
        )
    grids: dict[str, tuple] = {}
    for code, by_sex in by_code.items():
        ages = sorted(by_sex[0])
        grid = np.asarray(
            [[by_sex[s][a] for a in ages] for s in (0, 1)], dtype=np.float64
        )
        grids[code] = (grid, ages[0])
    return grids


def _read_riders_sheet(ws):
    """Read the riders master sheet -- ``product, rider_code, rider_name,
    type`` -- returning ``(code, type)`` pairs in sheet order."""
    riders = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[1] is None or row[3] is None:
            continue
        riders.append((str(row[1]).strip(), str(row[3]).strip()))
    return riders


def read_assumptions(path) -> Assumptions:
    """Read an actuarial basis from an Excel workbook into ``Assumptions``.

    Required sheets:

    * ``parameters`` -- two columns, name and value: ``discount_annual``,
      the expense scalars, the risk-adjustment scalars, and so on.
    * ``mortality`` -- long-form ``sex, age, rate``: the base mortality, one
      shared table. It drives the in-force decrement and the main-contract
      death claim.
    * ``lapse`` -- two columns, duration (completed policy years) and the
      annual lapse rate.

    A product with riders adds two more sheets:

    * ``riders`` -- the riders master: ``product, rider_code, rider_name,
      type``, one row per 특약코드. ``type`` is one of ``death_main``,
      ``death``, ``morbidity``, ``diagnosis``, ``annuity``, ``maturity``.
    * ``rates`` -- long-form ``rider_code, sex, age, rate`` for every
      rate-driven rider (``death`` / ``morbidity`` / ``diagnosis``).

    An optional ``waiver`` sheet -- long-form ``sex, age, rate``, the same
    shape as ``mortality`` -- gives the waiver-inception rate: the rate at
    which active in-force transitions to the premium-waived state. Absent,
    no waiver transitions occur.

    Annual rates are converted to monthly and wrapped in the lookup callables
    ``Assumptions`` expects. The bundled sample basis
    (:func:`load_sample_assumptions`) is a filled-in template to copy.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    for sheet in ("parameters", "mortality", "lapse"):
        if sheet not in wb.sheetnames:
            raise ValueError(f"{path!r} has no '{sheet}' sheet")

    # parameters -- a name/value sheet
    params: dict[str, object] = {}
    for name, value, *_ in wb["parameters"].iter_rows(min_row=2, values_only=True):
        if name is not None:
            params[str(name).strip()] = value

    mortality_monthly = _rate_closure(*_read_rate_grid(wb["mortality"]))

    # waiver -- optional sex/age waiver-inception table, like mortality
    waiver_inception_monthly = None
    if "waiver" in wb.sheetnames:
        waiver_inception_monthly = _rate_closure(*_read_rate_grid(wb["waiver"]))

    # lapse -- duration -> annual rate
    lapse_by_dur: dict[int, float] = {}
    for dur, rate, *_ in wb["lapse"].iter_rows(min_row=2, values_only=True):
        if dur is not None:
            lapse_by_dur[int(dur)] = float(rate)
    lapse_arr = np.asarray(
        [lapse_by_dur[d] for d in range(len(lapse_by_dur))], dtype=np.float64
    )

    def lapse_monthly(duration):
        d = np.clip(np.asarray(duration, np.int64), 0, lapse_arr.shape[0] - 1)
        return 1.0 - (1.0 - lapse_arr[d]) ** (1.0 / 12.0)

    # riders master + rates -> the rate-driven coverage list
    riders: tuple[RiderRate, ...] = ()
    coverage_types: dict[str, str] | None = None
    if "riders" in wb.sheetnames:
        rider_rows = _read_riders_sheet(wb["riders"])
        coverage_types = {code: rtype for code, rtype in rider_rows}
        rate_grids = (_read_rates_sheet(wb["rates"])
                      if "rates" in wb.sheetnames else {})
        rider_list = []
        for code, rtype in rider_rows:
            if rtype not in RATE_DRIVEN_TYPES:
                continue          # death_main / annuity / maturity carry no rate
            if code not in rate_grids:
                raise ValueError(
                    f"rider {code!r} is rate-driven ({rtype}) but has no rows "
                    "in the 'rates' sheet"
                )
            rider_list.append(RiderRate(
                code=code,
                rate=_rate_closure(*rate_grids[code]),
                is_diagnosis=(rtype == TYPE_DIAGNOSIS),
                risk=RISK_MORTALITY if rtype == TYPE_DEATH else RISK_MORBIDITY,
            ))
        riders = tuple(rider_list)

    required = ("discount_annual", "expense_acquisition",
                "expense_maintenance_annual", "expense_inflation",
                "ra_confidence", "mortality_cv")
    missing = [k for k in required if params.get(k) is None]
    if missing:
        raise ValueError(f"the 'parameters' sheet is missing: {missing}")
    kwargs: dict[str, object] = dict(
        mortality_monthly=mortality_monthly,
        lapse_monthly=lapse_monthly,
        waiver_inception_monthly=waiver_inception_monthly,
        riders=riders,
        **{k: float(params[k]) for k in required},
    )
    if coverage_types is not None:
        kwargs["coverage_types"] = coverage_types
    for opt in ("longevity_cv", "morbidity_cv", "expense_cv",
                "cost_of_capital_rate", "investment_return", "fund_fee",
                "guaranteed_credit_rate"):
        if params.get(opt) is not None:
            kwargs[opt] = float(params[opt])
    if params.get("ra_method") is not None:
        kwargs["ra_method"] = str(params["ra_method"]).strip()
    return Assumptions(**kwargs)


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
        monthly_premium=(df["monthly_premium"].to_numpy()
                         if "monthly_premium" in df.columns
                         else np.zeros(n_mp)),
    )
    for opt in ("sex", "count", "single_premium", "premium_term_months",
                "death_benefit", "maturity_benefit", "annuity_payment",
                "account_value"):
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
            "long-form model points need the assumptions -- the 특약코드 "
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
    for opt in ("sex", "count", "single_premium", "premium_term_months"):
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
        fields["monthly_premium"] = np.bincount(mp, weights=prem, minlength=n_mp)
    elif "monthly_premium" in pol.columns:
        fields["monthly_premium"] = pol["monthly_premium"].to_numpy()
    else:
        fields["monthly_premium"] = np.zeros(n_mp)

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
      ``count``, ``state``, ``monthly_premium``, ``single_premium``,
      ``premium_term_months``, ``death_benefit``, ``maturity_benefit`` and
      ``annuity_payment`` are read if present. A ``<rider_code>_benefit``
      column adds that rider's coverage; the ``assumptions`` resolve the
      특약코드 to an engine code.
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


def load_sample_assumptions() -> Assumptions:
    """Read fastcashflow's bundled sample actuarial basis.

    A filled-in basis packaged with the library, the companion to
    :func:`load_sample_model_points`. See :func:`read_assumptions` for the
    workbook format.
    """
    source = resources.files("fastcashflow") / "sample_data" / "sample_assumptions.xlsx"
    with resources.as_file(source) as path:
        return read_assumptions(path)


def load_sample_model_points() -> ModelPoints:
    """Read fastcashflow's bundled sample portfolio.

    A small long-form portfolio -- a policies file and a coverages file --
    packaged with the library, so the engine can be tried without preparing
    an input file. See :func:`read_model_points` for the file format.
    """
    assumptions = load_sample_assumptions()
    base = resources.files("fastcashflow") / "sample_data"
    with resources.as_file(base / "sample_policies.csv") as policies, \
            resources.as_file(base / "sample_coverages.csv") as coverages:
        return read_model_points(policies, assumptions, coverages=coverages)


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
