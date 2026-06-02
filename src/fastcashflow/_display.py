"""Shared human-readable formatting for measurement results.

The measurement dataclasses (``GMMMeasurement`` / ``PAAMeasurement`` /
``VFAMeasurement``) hold ``(n_mp,)`` headline arrays plus optional
``(n_mp, n_time)`` trajectories. The default dataclass ``repr`` dumps every
array -- hundreds of thousands of characters for a real portfolio. These
helpers give a compact one-line ``repr`` (portfolio totals) and a readable
multi-line ``str`` (per-model-point rows, capped, then the total), so
``m`` in a REPL and ``print(m)`` are both useful.

Each measurement class delegates ``__repr__`` / ``__str__`` here, passing its
own ``(label, array)`` columns -- the labels differ per model (GMM:
BEL/RA/CSM/loss; PAA: LRC/loss; VFA adds fee/TVOG).
"""
from __future__ import annotations

import numpy as np

_COL_W = 14
_MAX_ROWS = 10


def measurement_repr(cls_name: str, columns) -> str:
    """Compact one-line repr -- ``Name(n_mp=N, BEL=..., RA=..., ...)`` totals."""
    cols = [(name, np.asarray(vals)) for name, vals in columns]
    n_mp = cols[0][1].shape[0] if cols else 0
    body = ", ".join(f"{name}={a.sum():,.0f}" for name, a in cols)
    return f"{cls_name}(n_mp={n_mp}, {body})"


def measurement_str(cls_name: str, columns, max_rows: int = _MAX_ROWS) -> str:
    """Multi-line table -- per-model-point rows (capped) then the total."""
    cols = [(name, np.asarray(vals)) for name, vals in columns]
    names = [name for name, _ in cols]
    arrs = [a for _, a in cols]
    n_mp = arrs[0].shape[0] if arrs else 0
    title = f"{cls_name} -- {n_mp} model point" + ("s" if n_mp != 1 else "")
    header = f"{'':>8}" + "".join(f"{name:>{_COL_W}}" for name in names)
    lines = [title, header]
    shown = min(n_mp, max_rows)
    for i in range(shown):
        lines.append(
            f"{'mp ' + str(i):>8}"
            + "".join(f"{a[i]:>{_COL_W},.0f}" for a in arrs)
        )
    if n_mp > shown:
        lines.append(f"{'...':>8}  ({n_mp - shown} more model points)")
    lines.append(
        f"{'Total':>8}" + "".join(f"{a.sum():>{_COL_W},.0f}" for a in arrs)
    )
    return "\n".join(lines)
