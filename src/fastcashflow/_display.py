"""Shared human-readable formatting for the input and result objects.

``ModelPoints`` and the per-model ``Measurement`` dataclasses (one per model:
``fcf.gmm.Measurement`` / ``fcf.paa.Measurement`` / ``fcf.vfa.Measurement``)
hold per-model-point arrays. The
default dataclass ``repr`` dumps every array -- hundreds of thousands of
characters for a real portfolio. These helpers give a compact one-line
``repr`` and a readable multi-line ``str`` instead, so the object in a REPL
and ``print(...)`` are both useful.

Measurement classes delegate ``__repr__`` / ``__str__`` here, passing their
own ``(label, array)`` columns (GMM: BEL/RA/CSM/loss; PAA: LRC/loss; VFA adds
fee/TVOG). ``ModelPoints`` delegates to the ``model_points_*`` helpers, which
summarise the portfolio (counts, distributions, ranges) rather than the rows.
"""
from __future__ import annotations

import numpy as np

_COL_W = 14
_MAX_ROWS = 10


def model_points_repr(mp) -> str:
    """Compact one-line repr for ModelPoints -- counts, not the arrays."""
    n = mp.n_mp
    parts = [f"{n} model point" + ("s" if n != 1 else "")]
    if mp.product is not None:
        n_prod = len(set(mp.product))
        parts.append(f"{n_prod} product" + ("s" if n_prod != 1 else ""))
    if mp.coverage_codes:
        n_cov = len(mp.coverage_codes)
        parts.append(f"{n_cov} coverage code" + ("s" if n_cov != 1 else ""))
    elif np.any(np.asarray(mp.account_value)):
        parts.append("account-value")
    return f"<ModelPoints: {', '.join(parts)}>"


def model_points_str(mp) -> str:
    """Multi-line summary -- distributions and ranges, not the row arrays."""
    n = mp.n_mp
    lines = [f"<ModelPoints -- {n} model point" + ("s" if n != 1 else "") + ">"]

    def _dist(arr):
        vals, counts = np.unique(np.asarray(arr), return_counts=True)
        return ", ".join(f"{v} ({c})" for v, c in zip(vals, counts))

    def _row(label, value):
        lines.append(f"  {label:<9}: {value}")

    if mp.product is not None:
        _row("products", _dist(mp.product))
    if mp.channel is not None:
        _row("channels", _dist(mp.channel))
    if mp.coverage_codes:
        _row("coverages", ", ".join(mp.coverage_codes))
    age = np.asarray(mp.issue_age)
    _row("issue_age", f"{age.min():.0f}..{age.max():.0f}")
    term = np.asarray(mp.term_months)
    _row("term", f"{int(term.min())}..{int(term.max())} months")
    av = np.asarray(mp.account_value)
    if np.any(av):
        _row("account", f"{av[av > 0].min():,.0f}..{av.max():,.0f}")
    _row("count", f"{np.asarray(mp.count).sum():,.0f}")
    return "\n".join(lines)


def measurement_repr(cls_name: str, columns) -> str:
    """Compact one-line repr -- ``<Name: n_mp=N, BEL=..., RA=..., ...>`` totals."""
    cols = [(name, np.asarray(vals)) for name, vals in columns]
    n_mp = cols[0][1].shape[0] if cols else 0
    body = ", ".join(f"{name}={a.sum():,.0f}" for name, a in cols)
    return f"<{cls_name}: n_mp={n_mp}, {body}>"


def measurement_str(cls_name: str, columns, max_rows: int = _MAX_ROWS) -> str:
    """Multi-line table -- per-model-point rows (capped) then the total."""
    cols = [(name, np.asarray(vals)) for name, vals in columns]
    names = [name for name, _ in cols]
    arrs = [a for _, a in cols]
    n_mp = arrs[0].shape[0] if arrs else 0
    title = f"<{cls_name} -- {n_mp} model point" + ("s" if n_mp != 1 else "") + ">"
    header = f"{'':>8}" + "".join(f"{name:>{_COL_W}}" for name in names)
    lines = [title, header]
    shown = min(n_mp, max_rows)
    for i in range(shown):
        lines.append(
            f"{'mp ' + str(i):>8}"
            + "".join(f"{a[i]:>{_COL_W},.0f}" for a in arrs)
        )
    if n_mp > shown:
        more = n_mp - shown
        lines.append(f"{'...':>8}  ({more} more model point{'s' if more != 1 else ''})")
    lines.append(
        f"{'Total':>8}" + "".join(f"{a.sum():>{_COL_W},.0f}" for a in arrs)
    )
    return "\n".join(lines)
