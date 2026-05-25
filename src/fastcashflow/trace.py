"""Step-by-step calculation trace for a single model point.

:func:`show_trace` renders the BEL / RA / CSM build of one contract as an
ASCII tree: which segment-level tables apply, which rates were looked up
year by year, what cash flows came out, how those discount and roll
forward to the headline numbers. Intended for hand-checking against an
external pricing system or an actuary's own spreadsheet -- find the step
where the engine and the expectation diverge.

The function makes no new calculations; it slices the result of
:func:`fastcashflow.engine.measure` and prints it.
"""
from __future__ import annotations

import sys
from typing import IO

import numpy as np

from fastcashflow.assumptions import Assumptions
from fastcashflow.engine import measure
from fastcashflow.modelpoints import ModelPoints


def _emit_tree(items: list[object], out: list[str], prefix: str) -> None:
    """Render a list of (str | (header, sub_lines)) as ASCII tree rows."""
    n = len(items)
    for i, item in enumerate(items):
        last = (i == n - 1)
        head = "└─ " if last else "├─ "
        child = prefix + ("    " if last else "│   ")
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


def show_trace(
    mp_index: int,
    model_points: ModelPoints,
    assumptions: Assumptions | dict,
    *,
    file: IO | None = None,
) -> None:
    """Print a tree of how one model point's BEL / RA / CSM is computed.

    Parameters
    ----------
    mp_index
        0-based row index in ``model_points``.
    model_points
        Portfolio :class:`ModelPoints`. The function slices to a single
        row before running :func:`measure`, so a 1M-row portfolio does
        not pay for the trace of one contract.
    assumptions
        A single :class:`Assumptions`, or the dict returned by
        :func:`fastcashflow.io.read_assumptions` /
        :func:`fastcashflow.io.load_sample_assumptions`. With the dict
        form the function looks up the segment via the model point's
        ``(product, channel)``.
    file
        Where to write. ``None`` writes to ``sys.stdout``.

    Use it when an engine result disagrees with a hand calculation: the
    tree shows the segment, tables, rate values, cash flows and roll-
    forward step by step, so the diverging step is visible at a glance.
    """
    out: list[str] = []
    if file is None:
        file = sys.stdout

    n_mp = model_points.n_mp
    if not 0 <= mp_index < n_mp:
        raise IndexError(
            f"mp_index {mp_index} out of range for n_mp={n_mp}"
        )
    i = mp_index

    # Multi-segment dict basis: route to the right segment by (product, channel).
    if isinstance(assumptions, dict):
        if model_points.product is None or model_points.channel is None:
            raise ValueError(
                "model_points has no product / channel columns -- a dict "
                "basis cannot be routed; pass a single Assumptions instead"
            )
        key = (str(model_points.product[i]), str(model_points.channel[i]))
        try:
            assumptions = assumptions[key]
        except KeyError:
            raise KeyError(
                f"no assumptions for segment {key}; "
                f"available: {list(assumptions)}"
            ) from None

    # Single-row slice + measure. Subsetting first keeps the trace cost
    # proportional to one MP, not the whole portfolio.
    sub = model_points.subset([i])
    m = measure(sub, assumptions)

    # ---- Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "남" if sex_v == 0 else "여"
    age = float(sub.issue_age[0])
    term = int(sub.term_months[0])
    prem_term = (int(sub.premium_term_months[0])
                 if sub.premium_term_months is not None else term)
    count = float(sub.count[0]) if sub.count is not None else 1.0
    product = (str(model_points.product[i])
               if model_points.product is not None else "-")
    channel = (str(model_points.channel[i])
               if model_points.channel is not None else "-")
    header = (
        f"mp[{i}]  ({product}/{channel}, sex={sex_label}, issue_age={age:g}, "
        f"term={term}m, premium_term={prem_term}m, count={count:g})"
    )

    # ---- Assumptions (segment-level)
    asmp_lines: list[object] = []
    asmp_lines.append(
        f"mortality_annual     -> {_fmt_callable(assumptions.mortality_annual)}"
    )
    asmp_lines.append(
        f"lapse_annual         -> {_fmt_callable(assumptions.lapse_annual)}"
    )
    if assumptions.waiver_incidence_annual is not None:
        asmp_lines.append(
            f"waiver_incidence     -> {_fmt_callable(assumptions.waiver_incidence_annual)}"
        )
    d = assumptions.discount_annual
    if np.ndim(d) == 0:
        asmp_lines.append(f"discount_annual      = {float(d):g} (flat)")
    else:
        arr = np.asarray(d)
        asmp_lines.append(
            f"discount_annual      = ndarray len={arr.size} "
            f"[{arr.flat[0]:g}, ..., {arr.flat[-1]:g}]"
        )
    asmp_lines.append(
        f"alpha_pct / flat     = {assumptions.alpha_pct:g} / {assumptions.alpha_flat:g}"
    )
    asmp_lines.append(f"beta_pct             = {assumptions.beta_pct:g}")
    g = assumptions.gamma_flat
    if np.ndim(g) == 0:
        asmp_lines.append(f"gamma_flat           = {float(g):g}")
    else:
        asmp_lines.append(f"gamma_flat           = ndarray len={np.asarray(g).size}")
    inf = assumptions.expense_inflation
    if np.ndim(inf) == 0:
        asmp_lines.append(f"expense_inflation    = {float(inf):g}")
    else:
        asmp_lines.append(
            f"expense_inflation    = ndarray len={np.asarray(inf).size}"
        )
    asmp_lines.append(
        f"ra: method={assumptions.ra_method!r}, conf={assumptions.ra_confidence:g}"
    )
    asmp_lines.append(
        f"cv: mort={assumptions.mortality_cv:g} morb={assumptions.morbidity_cv:g} "
        f"long={assumptions.longevity_cv:g} disab={assumptions.disability_cv:g}"
    )

    # ---- Coverages (rate-driven riders)
    cov_lines: list[object] = []
    for r in assumptions.coverages:
        cov_lines.append(
            f"{r.code!r:14} risk={r.risk}  is_diagnosis={r.is_diagnosis}  "
            f"rate -> {_fmt_callable(r.rate)}"
        )
    if not cov_lines:
        cov_lines.append("(none -- main-coverage death only)")

    # ---- Rates per policy year, evaluated at this MP's axes
    n_years = (term + 11) // 12
    issue_class_v = (int(sub.issue_class[0])
                     if sub.issue_class is not None else 0)
    elapsed_v = (int(sub.elapsed_months[0])
                 if sub.elapsed_months is not None else 0)
    year_picks = sorted({0, 1, 2, 3, 4, n_years - 1, max(0, n_years // 2)})
    year_picks = [y for y in year_picks if 0 <= y < n_years]

    rate_lines: list[object] = []
    rate_lines.append(
        f"axes: sex={sex_v}, issue_age={age:g}, issue_class={issue_class_v}, "
        f"elapsed_at_issue={elapsed_v}m"
    )
    has_waiver = assumptions.waiver_incidence_annual is not None
    cov_headers = [f"{r.code}(an)" for r in assumptions.coverages]
    head_row = ["year", "mort(an)", "lapse(an)"]
    if has_waiver:
        head_row.append("waiver(an)")
    head_row.extend(cov_headers)
    rate_lines.append("  ".join(f"{h:>12}" for h in head_row))
    for y in year_picks:
        row = [f"{y:>12d}"]
        row.append(
            f"{_eval_rate(assumptions.mortality_annual, sex_v, age, y, issue_class_v, elapsed_v):>12.6f}"
        )
        row.append(
            f"{_eval_rate(assumptions.lapse_annual, sex_v, age, y, issue_class_v, elapsed_v):>12.6f}"
        )
        if has_waiver:
            row.append(
                f"{_eval_rate(assumptions.waiver_incidence_annual, sex_v, age, y, issue_class_v, elapsed_v):>12.6f}"
            )
        for r in assumptions.coverages:
            row.append(
                f"{_eval_rate(r.rate, sex_v, age, y, issue_class_v, elapsed_v):>12.6f}"
            )
        rate_lines.append("  ".join(row))

    # ---- Cash flows (annual sums of the monthly trajectory)
    cf = m.cashflows
    bel = m.bel[0]
    ra = m.ra[0]
    csm = m.csm[0]
    csm_acc = m.csm_accretion[0]
    csm_rel = m.csm_release[0]
    lc = m.loss_component[0]
    discount_start = m.discount_start

    cf_lines: list[object] = []
    cf_headers = ["year", "premium", "claim", "morbidity", "expense",
                  "annuity", "surrender", "disability"]
    cf_lines.append("  ".join(f"{h:>12}" for h in cf_headers))
    for y in range(n_years):
        a0 = y * 12
        a1 = min(a0 + 12, cf.n_time)
        if a1 <= a0:
            break
        row = [f"{y:>12d}"]
        for name in ("premium_cf", "claim_cf", "morbidity_cf", "expense_cf",
                     "annuity_cf", "surrender_cf", "disability_cf"):
            arr = getattr(cf, name)[0]
            row.append(f"{arr[a0:a1].sum():>12,.0f}")
        cf_lines.append("  ".join(row))
    if float(cf.maturity_cf[0]) != 0.0:
        cf_lines.append(
            f"maturity benefit at t={term}m: {float(cf.maturity_cf[0]):,.0f}"
        )

    # ---- Discount factors at key months
    picks = _key_months(term, discount_start.shape[0] - 1)
    disc_lines: list[object] = [
        f"t={t:>4d}m: ds={discount_start[t]:.6f}" for t in picks
    ]

    # ---- BEL roll-forward at key months
    bel_lines: list[object] = [
        "BEL[t] = annuity[t] - premium[t] + (claim+morbidity+disability+"
        "expense+surrender)[t] * (1+i)^(-1/2) + BEL[t+1] * (1+i)^(-1)",
        f"BEL[{term:>4d}] = maturity = {float(cf.maturity_cf[0]):>15,.2f} "
        "(seed -- a single payment at term)",
    ]
    for t in reversed(picks):
        if t == term:
            continue
        bel_lines.append(f"BEL[{t:>4d}] = {bel[t]:>15,.2f}")

    # ---- CSM roll-forward at key months
    fcf0 = float(bel[0] + ra[0])
    csm_lines: list[object] = [
        f"FCF[0]    = BEL[0] + RA[0] = {bel[0]:,.2f} + {ra[0]:,.2f} = {fcf0:,.2f}",
        f"CSM[0]    = max(0, -FCF[0]) = {csm[0]:,.2f}",
        f"loss_comp = max(0,  FCF[0]) = {lc:,.2f}",
        "csm[t+1] = csm[t] + accretion[t] - release[t]",
    ]
    for t in picks:
        if t >= csm_acc.shape[0]:
            csm_lines.append(
                f"t={t:>4d}m: csm={csm[t]:>14,.2f}  "
                f"(past last accretion month)"
            )
        else:
            csm_lines.append(
                f"t={t:>4d}m: csm={csm[t]:>14,.2f}  "
                f"acc={csm_acc[t]:>10,.2f}  rel={csm_rel[t]:>10,.2f}"
            )

    # ---- Final headline
    final_lines: list[object] = [
        f"BEL              = {bel[0]:>15,.2f}",
        f"RA               = {ra[0]:>15,.2f}",
        f"FCF = BEL + RA   = {fcf0:>15,.2f}",
        f"CSM = max(0,-FCF)= {csm[0]:>15,.2f}",
        f"loss_component   = {lc:>15,.2f}",
    ]

    # Assemble the tree
    out.append(header)
    tree_items: list[object] = [
        ("Assumptions (segment-level)", asmp_lines),
        (f"Coverages (rate-driven riders, n={len(assumptions.coverages)})",
         cov_lines),
        ("Rates (annual, evaluated for this MP)", rate_lines),
        (f"Cash flows (annual sum over {cf.n_time}m horizon)", cf_lines),
        ("Discount factors (key months)", disc_lines),
        ("BEL roll-forward (key months)", bel_lines),
        ("CSM roll-forward (key months)", csm_lines),
        ("Final (headline numbers, per policy)", final_lines),
    ]
    _emit_tree(tree_items, out, "")
    file.write("\n".join(out) + "\n")
