"""File I/O for model points and valuation results -- the polars layer.

Reading and writing go through polars, which parses parquet and CSV in
parallel and hands columns to numpy near-zero-copy.

* :func:`read_model_points` / :func:`write_valuation` are eager -- they hold
  the whole file in memory, which is fine up to ~1e8 model points.
* :func:`value_file` streams a parquet file chunk by chunk (read, value,
  write), so peak memory is one chunk and the portfolio size is bounded by
  disk, not RAM -- the path to ~1e9 model points and beyond.

The core engine stays identifier-free: the kernel never needs a policy id, so
none is carried through ``ModelPointSet`` or ``Valuation``. Identifiers are a
file-boundary concern -- pass them to :func:`write_valuation` (or via
``value_file``'s ``id_column``) to join results back to policies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from fastcashflow.assumptions import Assumptions
from fastcashflow.coverage import INPATIENT, OUTPATIENT, SURGERY
from fastcashflow.engine import Valuation, value
from fastcashflow.modelpoint import ModelPointSet

_REQUIRED_COLUMNS = ("issue_age", "death_benefit", "monthly_premium", "term_months")
_OPTIONAL_COLUMNS = ("maturity_benefit", "annuity_payment", "single_premium")
# Health benefit columns -- each maps to a morbidity coverage kind.
_BENEFIT_COLUMNS = {
    "inpatient_benefit": INPATIENT,
    "surgery_benefit": SURGERY,
    "outpatient_benefit": OUTPATIENT,
}


def _read_frame(path) -> pl.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        return pl.read_parquet(p)
    if p.endswith(".csv"):
        return pl.read_csv(p)
    raise ValueError(f"unsupported file type: {path!r} (expected .parquet or .csv)")


def _write_frame(df: pl.DataFrame, path) -> None:
    p = str(path)
    if p.endswith(".parquet"):
        df.write_parquet(p)
    elif p.endswith(".csv"):
        df.write_csv(p)
    else:
        raise ValueError(
            f"unsupported file type: {path!r} (expected .parquet or .csv)"
        )


def read_model_points(path) -> ModelPointSet:
    """Read model points from a parquet or CSV file into a ``ModelPointSet``.

    The file must contain the columns ``issue_age``, ``death_benefit``,
    ``monthly_premium`` and ``term_months``. The optional columns
    ``maturity_benefit``, ``annuity_payment`` and ``single_premium`` are read
    if present, else default to zero. Health benefit columns --
    ``inpatient_benefit``, ``surgery_benefit``, ``outpatient_benefit`` -- are
    read into the coverage list if present. Any other column (a policy
    identifier, say) is ignored -- to carry an identifier through to the
    results, read it separately and pass it to :func:`write_valuation`.
    """
    df = _read_frame(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path!r} is missing required column(s): {missing}")
    fields = dict(
        issue_age=df["issue_age"].to_numpy(),
        death_benefit=df["death_benefit"].to_numpy(),
        monthly_premium=df["monthly_premium"].to_numpy(),
        term_months=df["term_months"].to_numpy(),
    )
    for optional in _OPTIONAL_COLUMNS:
        if optional in df.columns:
            fields[optional] = df[optional].to_numpy()
    benefits = {kind: df[col].to_numpy()
                for col, kind in _BENEFIT_COLUMNS.items() if col in df.columns}
    if benefits:
        fields["benefits"] = benefits
    return ModelPointSet(**fields)


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
    chunk_size: int = 20_000_000,
    backend: str = "cpu",
    id_column: str | None = None,
) -> int:
    """Stream a valuation through a parquet file one chunk at a time.

    Reads ``input_path`` in chunks of ``chunk_size`` model points, values each
    chunk with :func:`value`, and writes the results as a parquet dataset --
    one ``part-NNNNN.parquet`` file per chunk -- under ``output_dir``. Peak
    memory is a single chunk, so this scales past what an in-memory run could
    hold (portfolios on the order of 1e9 model points and beyond).

    The kernel is per-model-point independent, so the chunked result is
    identical to valuing the whole file at once.

    Parameters
    ----------
    input_path :
        Parquet file with the model-point columns (see
        :func:`read_model_points`). CSV is not supported here -- it has no
        row-group metadata for chunked reads.
    output_dir :
        Directory for the result parts; created if absent, and must not
        already contain ``part-*.parquet`` files. Read the results back with
        ``polars.read_parquet(f"{output_dir}/part-*.parquet")``.
    chunk_size :
        Model points per chunk. The default keeps a chunk near 1-2 GB.
    backend :
        Passed to :func:`value` -- ``"cpu"`` or ``"gpu"``.
    id_column :
        Name of an identifier column in the input file; when given it is read
        per chunk and written alongside each result part.

    Returns
    -------
    int
        The total number of model points processed.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if input_path.suffix != ".parquet":
        raise ValueError(
            f"value_file streams parquet input only; got {str(input_path)!r}"
        )

    columns = list(_REQUIRED_COLUMNS)
    if id_column is not None:
        columns = [id_column, *columns]

    scan = pl.scan_parquet(input_path)
    available = scan.collect_schema().names()
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"{str(input_path)!r} is missing required column(s): {missing}"
        )
    optional = [c for c in _OPTIONAL_COLUMNS if c in available]
    benefit_cols = [c for c in _BENEFIT_COLUMNS if c in available]
    columns = [*columns, *optional, *benefit_cols]

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("part-*.parquet")):
        raise ValueError(
            f"output directory {str(output_dir)!r} already contains part "
            "files; use a fresh directory"
        )

    n_total = scan.select(pl.len()).collect().item()
    projected = scan.select(columns)

    processed = 0
    for part, offset in enumerate(range(0, n_total, chunk_size)):
        chunk = projected.slice(offset, chunk_size).collect()
        fields = dict(
            issue_age=chunk["issue_age"].to_numpy(),
            death_benefit=chunk["death_benefit"].to_numpy(),
            monthly_premium=chunk["monthly_premium"].to_numpy(),
            term_months=chunk["term_months"].to_numpy(),
        )
        for name in optional:
            fields[name] = chunk[name].to_numpy()
        benefits = {_BENEFIT_COLUMNS[c]: chunk[c].to_numpy() for c in benefit_cols}
        if benefits:
            fields["benefits"] = benefits
        mps = ModelPointSet(**fields)
        ids = chunk[id_column].to_numpy() if id_column is not None else None
        write_valuation(
            value(mps, assumptions, backend=backend),
            output_dir / f"part-{part:05d}.parquet",
            ids=ids,
        )
        processed += mps.n_mp

    return processed
