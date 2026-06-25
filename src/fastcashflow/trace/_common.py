"""Shared rendering and diff-formatting helpers for the trace views.

Low-level primitives used by every per-model trace module: ASCII-tree
emission, rate-callable formatting, scalar rate evaluation, column sizing,
dict-basis routing, and the assumption-change ("diff") formatters. No model
logic lives here -- only the presentation helpers the four trace modules
(gmm / vfa / paa / reinsurance) share.
"""
from __future__ import annotations

import numpy as np

from fastcashflow.basis import Basis, BasisRouter
from fastcashflow.model_points import ModelPoints


def _emit_tree(items: list[object], out: list[str], prefix: str) -> None:
    """Render a list of (str | (header, sub_lines)) as ASCII tree rows."""
    n = len(items)
    for i, item in enumerate(items):
        last = (i == n - 1)
        head = "`- " if last else "+- "
        child = prefix + ("    " if last else "|   ")
        if isinstance(item, tuple):
            header, subs = item
            out.append(f"{prefix}{head}{header}")
            _emit_tree(subs, out, child)
        else:
            out.append(f"{prefix}{head}{item}")


def _fmt_callable(fn: object) -> str:
    """Format a rate callable, surfacing its source table_id when known."""
    tid = getattr(fn, "_fcf_table_id", None)
    if tid is None:
        return "<callable>"
    mods = getattr(fn, "_fcf_modifiers", ())
    suffix = f" (+{', +'.join(mods)})" if mods else ""
    return f"{tid}{suffix}"


def _eval_rate(
    fn, sex: int, issue_age: float, duration: int,
    issue_class: int, elapsed: int,
) -> float:
    """Evaluate a 5-arg rate callable at scalar inputs and return a float."""
    if fn is None:
        return 0.0
    s = np.array([sex], dtype=np.int64)
    a = np.array([issue_age], dtype=np.float64)
    d = np.array([duration], dtype=np.int64)
    ic = np.array([issue_class], dtype=np.int64)
    em = np.array([elapsed], dtype=np.int64)
    return float(np.asarray(fn(s, a, d, ic, em)).flat[0])


def _key_months(term: int, n_time: int) -> list[int]:
    """Months at which to sample the trajectory in the printed tree.

    A few anchor points across the run-off -- inception, the early years,
    the half-way point and the last year before term -- are enough for a
    sanity check without flooding the output.
    """
    raw = sorted({0, 12, 60, 120, max(0, term - 12), term})
    return [t for t in raw if 0 <= t <= n_time]


def _colw(values: object, spec: str = ",.2f", min_width: int = 0) -> int:
    """Column width = max(``min_width``, widest formatted value).

    Expand-only: a normal-magnitude column keeps its usual ``min_width``
    (so existing output is unchanged), but once a value needs more digits
    (e.g. an excess of 10,000,000 is 13 chars and would overflow a width-12
    field, pushing every later column out of alignment) the field widens to
    fit. Sizing up to the data -- never down -- keeps small cases stable.
    """
    vals = list(values)
    return max(min_width, max((len(format(float(v), spec)) for v in vals), default=1))


def _resolve_basis(
    basis: Basis | dict, model_points: ModelPoints, i: int,
) -> Basis:
    """Return the :class:`Basis` to use for row ``i``.

    Mirrors the dict-routing behaviour of :func:`show_trace`. Factored
    out so the diff variant can resolve two bases the same way.
    """
    if not isinstance(basis, BasisRouter):
        return basis
    if model_points.product is None or model_points.channel is None:
        raise ValueError(
            "model_points has no product / channel columns -- a "
            "BasisRouter cannot be routed; pass a single Basis instead"
        )
    key = (str(model_points.product[i]), str(model_points.channel[i]))
    try:
        return basis.segments[key]
    except KeyError:
        raise KeyError(
            f"no basis for segment {key}; available: {list(basis.segments)}"
        ) from None


def _money_delta(a: float, b: float, *, width: int = 14) -> str:
    """Format ``a -> b   (diff, %diff)`` for two money amounts."""
    d = b - a
    if abs(a) > 1e-12:
        pct = 100.0 * d / a
        pct_s = f"{pct:>+8.2f}%"
    else:
        pct_s = "       --"
    return f"{a:>{width},.2f}  ->  {b:>{width},.2f}   ({d:>+{width-1},.2f}, {pct_s})"


def _rate_delta(a: float, b: float) -> str:
    """Format ``a -> b   (diff)`` for two annual rates (6-dp probabilities)."""
    d = b - a
    if abs(a) > 1e-12:
        pct = 100.0 * d / a
        pct_s = f"{pct:>+8.2f}%"
    else:
        pct_s = "       --"
    return f"{a:>10.6f}  ->  {b:>10.6f}   ({d:>+10.6f}, {pct_s})"


def _diff_scalar(name: str, va, vb) -> str | None:
    """Compare two non-callable values; return a one-line diff or None.

    Returns ``None`` when the values are equal -- the diff view only
    surfaces fields that *changed*.
    """
    if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
        if np.array_equal(np.asarray(va), np.asarray(vb)):
            return None
        a_str = f"ndarray len={np.asarray(va).size}"
        b_str = f"ndarray len={np.asarray(vb).size}"
        return f"{name:<22} = {a_str} -> {b_str}"
    if va == vb:
        return None
    return f"{name:<22} = {va!r} -> {vb!r}"


def _diff_callable(name: str, fa, fb) -> str | None:
    """Compare two rate callables by their source ``_fcf_table_id`` and
    modifier chain. Same identity / same metadata -> no diff line."""
    if fa is fb:
        return None
    return f"{name:<22} : {_fmt_callable(fa)}  ->  {_fmt_callable(fb)}"


def _basis_diff_lines(a: Basis, b: Basis) -> list[object]:
    """The 'what changed' lines between two bases -- model-agnostic.

    Surfaces only the fields that differ: the rate callables (by source id /
    modifier chain), the scalar economic / risk parameters (including the VFA
    ``investment_return`` / ``fund_fee``), the expense ledger length, and each
    coverage's rate. Shared by every ``trace_diff`` so the assumption-change
    view reads the same across GMM / VFA / PAA / reinsurance.
    """
    lines: list[object] = []
    for name in ("mortality_annual", "lapse_annual", "waiver_incidence_annual"):
        line = _diff_callable(name, getattr(a, name), getattr(b, name))
        if line is not None:
            lines.append(line)
    for name in ("discount_annual", "expense_inflation", "ra_method",
                 "ra_confidence", "cost_of_capital_rate", "mortality_cv",
                 "morbidity_cv", "longevity_cv", "disability_cv", "expense_cv",
                 "investment_return", "fund_fee"):
        line = _diff_scalar(name, getattr(a, name), getattr(b, name))
        if line is not None:
            lines.append(line)
    if a.expense_items != b.expense_items:
        lines.append(
            f"expense_items           : len {len(a.expense_items)} -> "
            f"len {len(b.expense_items)}")
    codes_a = [r.code for r in a.coverages]
    codes_b = [r.code for r in b.coverages]
    if codes_a != codes_b:
        lines.append(f"coverages (codes)      : {codes_a} -> {codes_b}")
    else:
        for ra, rb in zip(a.coverages, b.coverages):
            line = _diff_callable(f"coverage[{ra.code}].rate", ra.rate, rb.rate)
            if line is not None:
                lines.append(line)
    if not lines:
        lines.append("(no changes in tracked fields)")
    return lines


def _diff_mp_header(model_points: ModelPoints, sub: ModelPoints, i: int,
                    tag: str) -> str:
    """The ``diff[-tag] mp[i] (...)`` identity line shared by the trace_diffs."""
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "M" if sex_v == 0 else "F"
    age = float(sub.issue_age[0])
    term = int(sub.term_months[0])
    count = float(sub.count[0]) if sub.count is not None else 1.0
    product = (str(model_points.product[i])
               if model_points.product is not None else "-")
    channel = (str(model_points.channel[i])
               if model_points.channel is not None else "-")
    return (f"diff{tag} mp[{i}]  ({product}/{channel}, sex={sex_label}, "
            f"issue_age={age:g}, term={term}m, count={count:g})")
