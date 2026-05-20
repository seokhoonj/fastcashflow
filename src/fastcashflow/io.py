"""File I/O for model points and valuation results -- the polars layer.

Reading and writing go through polars, which parses parquet and CSV in
parallel and hands columns to numpy near-zero-copy, so this path scales to
portfolios on the order of 1e8 model points.

The core engine stays identifier-free: the kernel never needs a policy id, so
none is carried through ``ModelPointSet`` or ``Valuation``. Identifiers are a
file-boundary concern -- pass them to :func:`write_valuation` to join results
back to policies.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from fastcashflow.engine import Valuation
from fastcashflow.modelpoint import ModelPointSet

_REQUIRED_COLUMNS = ("issue_age", "sum_assured", "monthly_premium", "term_months")


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

    The file must contain the columns ``issue_age``, ``sum_assured``,
    ``monthly_premium`` and ``term_months``; any other column (a policy
    identifier, say) is ignored. To carry an identifier through to the
    results, read it separately and pass it to :func:`write_valuation`.
    """
    df = _read_frame(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path!r} is missing required column(s): {missing}")
    return ModelPointSet(
        issue_age=df["issue_age"].to_numpy(),
        sum_assured=df["sum_assured"].to_numpy(),
        monthly_premium=df["monthly_premium"].to_numpy(),
        term_months=df["term_months"].to_numpy(),
    )


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
