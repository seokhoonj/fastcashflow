"""Step-by-step calculation trace for a single model point.

:func:`show_trace` renders the BEL / RA / CSM build of one contract as an
ASCII tree: which segment-level tables apply, which rates were looked up
year by year, what cash flows came out, how those discount and roll
forward to the headline numbers. Intended for hand-checking against an
external pricing system or an actuary's own spreadsheet -- find the step
where the engine and the expectation diverge.

:func:`show_trace_diff` is the two-basis variant: it shows, at each step,
how a change of assumption propagates -- which rate moved, by how much,
which cash flow shifted, and what the net effect on BEL / RA / CSM is.
The right tool for shock analysis and what-if questions.

Both functions make no new calculations; they slice the result of
:func:`fastcashflow.engine.measure` and print it.
"""
from __future__ import annotations

import sys
from typing import IO

import numpy as np

from fastcashflow.assumptions import Assumptions
from fastcashflow.coverage import CalculationMethod, method_attrs
from fastcashflow.curves import discount_monthly_curve
from fastcashflow._typing import FloatArray
from fastcashflow.engine import measure
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.vfa import measure_vfa


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
        if model_points.product_code is None or model_points.channel_code is None:
            raise ValueError(
                "model_points has no product / channel columns -- a dict "
                "basis cannot be routed; pass a single Assumptions instead"
            )
        key = (str(model_points.product_code[i]), str(model_points.channel_code[i]))
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
    product = (str(model_points.product_code[i])
               if model_points.product_code is not None else "-")
    channel = (str(model_points.channel_code[i])
               if model_points.channel_code is not None else "-")
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
    infl = assumptions.expense_inflation
    if np.ndim(infl) == 0:
        asmp_lines.append(f"expense_inflation    = {float(infl):g} (flat)")
    else:
        arr = np.asarray(infl)
        asmp_lines.append(
            f"expense_inflation    = ndarray len={arr.size} "
            f"[{arr.flat[0]:g}, ..., {arr.flat[-1]:g}]"
        )
    rows = assumptions.expense_items
    if not rows:
        asmp_lines.append("expense_items        = ()  (no expense)")
    else:
        row_lines: list[object] = [
            (f"ExpenseItem({r.expense_type!r}, basis={r.basis!r}, "
             f"value={r.value:g})")
            for r in rows
        ]
        asmp_lines.append(
            (f"expense_items        = tuple  (len={len(rows)})", row_lines)
        )
    asmp_lines.append(
        f"ra: method={assumptions.ra_method!r}, conf={assumptions.ra_confidence:g}"
    )
    asmp_lines.append(
        f"cv: mort={assumptions.mortality_cv:g} morb={assumptions.morbidity_cv:g} "
        f"long={assumptions.longevity_cv:g} disab={assumptions.disability_cv:g}"
    )

    # ---- Coverages (rate-driven)
    cov_lines: list[object] = []
    methods = model_points.calculation_methods or {}
    for r in assumptions.coverages:
        method = methods.get(r.code, CalculationMethod.MORBIDITY)
        is_diag, risk = method_attrs(method)
        # Pad each field to the longest possible value so columns line up:
        # code   -> repr widest sample code in practice (~15 incl. quotes)
        # method -> 9 (longest CalculationMethod member name = MORBIDITY)
        # is_diagnosis -> 5 (longest str(bool) = "False")
        cov_lines.append(
            f"{r.code!r:<16}  method={str(method):<9}  risk={risk}  "
            f"is_diagnosis={str(is_diag):<5}  rate -> {_fmt_callable(r.rate)}"
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

    # ---- Diagnosis pool undiagnosed share at key months (only when
    # DIAGNOSIS coverages exist). `undiagnosed` is the kernel's per-coverage
    # scalar (`# fraction of the in-force still undiagnosed`) updated each
    # month as ``undiagnosed *= (1 - monthly_rate)``. Storing the trajectory
    # here makes the depleting-pool mechanism visible alongside the in-force
    # trajectory it composes with.
    picks = _key_months(term, discount_start.shape[0] - 1)
    diag_pool_lines: list[object] = []
    for r in assumptions.coverages:
        method = methods.get(r.code, CalculationMethod.MORBIDITY)
        is_diag, _ = method_attrs(method)
        if not is_diag:
            continue
        # Simulate the kernel's `undiagnosed` update. Annual rate is held
        # flat across the year so the closed-form within-year ramp is
        # (1 - q_annual) ** (months_in_year / 12), and full-year multipliers
        # compound at year boundaries.
        traj = np.empty(term + 1, dtype=np.float64)
        traj[0] = 1.0
        running = 1.0
        for y in range(n_years):
            q_annual = _eval_rate(
                r.rate, sex_v, age, y, issue_class_v, elapsed_v,
            )
            months_in_year = min(12, term - y * 12)
            year_start = y * 12
            for m in range(months_in_year):
                running *= (1.0 - q_annual) ** (1.0 / 12.0)
                traj[year_start + m + 1] = running
        cov_pool_lines: list[object] = [
            f"t={t:>4d}m: undiagnosed={traj[t]:.6f}" for t in picks
        ]
        diag_pool_lines.append((f"{r.code!r}:", cov_pool_lines))

    disc_lines: list[object] = [
        f"t={t:>4d}m: ds={discount_start[t]:.6f}" for t in picks
    ]

    # ---- BEL roll-forward at key months
    bel_lines: list[object] = [
        "BEL[t] = annuity[t] - premium[t] + (claim+morbidity+disability+"
        "expense+surrender)[t] * (1+i)^(-1/2) + BEL[t+1] * (1+i)^(-1)",
        # Keep the value column aligned with the rows below by putting the
        # "maturity seed" annotation after the number, not in the middle.
        f"BEL[{term:>4d}] = {float(cf.maturity_cf[0]):>15,.2f}  "
        "(maturity seed -- a single payment at term)",
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
        (f"Coverages (rate-driven, n={len(assumptions.coverages)})",
         cov_lines),
        ("Rates (annual, evaluated for this MP)", rate_lines),
        (f"Cash flows (annual sum over {cf.n_time}m horizon)", cf_lines),
    ]
    if diag_pool_lines:
        tree_items.append(
            ("Undiagnosed share (key months, per coverage)",
             diag_pool_lines)
        )
    tree_items.extend([
        ("Discount factors (key months)", disc_lines),
        ("BEL roll-forward (key months)", bel_lines),
        ("CSM roll-forward (key months)", csm_lines),
        ("Final (headline numbers, per policy)", final_lines),
    ])
    _emit_tree(tree_items, out, "")
    file.write("\n".join(out) + "\n")


def show_trace_vfa(
    mp_index: int,
    model_points: ModelPoints,
    assumptions: Assumptions | dict,
    *,
    return_scenarios: FloatArray | None = None,
    file: IO | None = None,
) -> None:
    """Print a tree of how one VFA model point's BEL / RA / CSM is computed.

    The VFA (variable-fee, account-value) counterpart of :func:`show_trace`.
    It slices to a single row, runs :func:`measure_vfa`, and shows the
    account-value trajectory, the GMDB / GMAB floors (where the guarantee
    bites), the variable fee and the BEL / RA / CSM -- plus the guarantee
    time value (TVOG) when ``return_scenarios`` is supplied. Use it on
    direct-participation contracts; :func:`show_trace` traces the GMM
    ``measure`` and does not cover the account-value mechanic.
    """
    out: list[str] = []
    if file is None:
        file = sys.stdout
    n_mp = model_points.n_mp
    if not 0 <= mp_index < n_mp:
        raise IndexError(f"mp_index {mp_index} out of range for n_mp={n_mp}")
    i = mp_index
    assumptions = _resolve_basis(assumptions, model_points, i)
    sub = model_points.subset([i])
    m = measure_vfa(sub, assumptions, return_scenarios=return_scenarios)

    # ---- Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "남" if sex_v == 0 else "여"
    age = float(sub.issue_age[0])
    term = int(sub.term_months[0])
    count = float(sub.count[0]) if sub.count is not None else 1.0
    product = (str(model_points.product_code[i])
               if model_points.product_code is not None else "-")
    channel = (str(model_points.channel_code[i])
               if model_points.channel_code is not None else "-")
    av0 = float(sub.account_value[0])
    gcr = float(sub.guaranteed_credit_rate[0])
    gdb = float(sub.guaranteed_death_benefit[0])
    gab = float(sub.guaranteed_accumulation_benefit[0])
    header = (
        f"mp[{i}]  VFA  ({product}/{channel}, sex={sex_label}, "
        f"issue_age={age:g}, term={term}m, count={count:g})"
    )

    # ---- VFA inputs
    vfa_lines: list[object] = [
        f"account_value                          = {av0:>15,.2f}",
        f"guaranteed_credit_rate                 = {gcr:g}",
        f"guaranteed_death_benefit (GMDB)        = {gdb:>15,.2f}",
        f"guaranteed_accumulation_benefit (GMAB) = {gab:>15,.2f}",
        f"investment_return = {assumptions.investment_return:g}  (VFA 할인/적립 basis)",
        f"fund_fee          = {assumptions.fund_fee:g}  (= 이익원)",
        f"mortality_annual  -> {_fmt_callable(assumptions.mortality_annual)}",
        f"lapse_annual      -> {_fmt_callable(assumptions.lapse_annual)}",
        f"ra: method={assumptions.ra_method!r} conf={assumptions.ra_confidence:g} "
        f"expense_cv={assumptions.expense_cv:g}",
    ]

    # ---- Trajectories (from the VFA measurement)
    av = m.account_value[0]
    cf = m.cashflows
    inforce = cf.inforce[0]
    deaths = cf.deaths[0]
    survivors = float(cf.maturity_survivors[0])
    n_time = cf.n_time
    picks = _key_months(term, n_time)

    av_lines: list[object] = []
    for t in picks:
        if t >= av.shape[0]:
            continue
        inf_v = inforce[t] if t < n_time else 0.0
        av_lines.append(f"t={t:>4d}m: AV={av[t]:>15,.2f}  inforce={inf_v:.6f}")

    # ---- Guarantee floors (where they bite)
    floor_lines: list[object] = [
        "death[t] = max(AV[t], GDB);  maturity = max(AV[term-1], GAB)",
    ]
    for t in picks:
        if t >= n_time:
            continue
        excess = max(0.0, gdb - av[t])
        floor_lines.append(
            f"t={t:>4d}m: death=max(AV,GDB)={max(av[t], gdb):>15,.2f}  "
            f"excess={excess:>12,.2f}  deaths={deaths[t]:.6f}"
        )
    ti = max(0, term - 1)
    floor_lines.append(
        f"maturity@t={ti}m: max(AV,GAB)={max(av[ti], gab):>15,.2f}  "
        f"excess={max(0.0, gab - av[ti]):>12,.2f}  survivors={survivors:.6f}"
    )

    # ---- BEL / CSM trajectory + roll-forward
    bel = m.bel[0]
    ra = m.ra[0]
    csm = m.csm[0]
    csm_acc = m.csm_accretion[0]
    csm_rel = m.csm_release[0]
    belcsm_lines: list[object] = [
        f"t={t:>4d}m: BEL={bel[t]:>15,.2f}  CSM={csm[t]:>15,.2f}"
        for t in picks if t < bel.shape[0]
    ]
    csm_lines: list[object] = ["csm[t+1] = csm[t] + accretion[t] - release[t]"]
    for t in picks:
        if t >= csm_acc.shape[0]:
            csm_lines.append(f"t={t:>4d}m: csm={csm[t]:>14,.2f}  (past last accretion)")
        else:
            csm_lines.append(
                f"t={t:>4d}m: csm={csm[t]:>14,.2f}  "
                f"acc={csm_acc[t]:>10,.2f}  rel={csm_rel[t]:>10,.2f}"
            )

    # ---- Final headline
    fee = float(m.variable_fee[0])
    tv = float(m.time_value[0])
    lc = float(m.loss_component[0])
    tv_note = ("(시나리오 없음 -> intrinsic 만)" if return_scenarios is None
               else "(보증 시간가치 -- CSM 흡수)")
    final_lines: list[object] = [
        f"variable_fee     = {fee:>15,.2f}  (수수료 PV = 이익원)",
        f"BEL              = {bel[0]:>15,.2f}",
        f"RA               = {ra[0]:>15,.2f}",
        f"TVOG (time_value)= {tv:>15,.2f}  {tv_note}",
        f"CSM              = {csm[0]:>15,.2f}",
        f"loss_component   = {lc:>15,.2f}",
    ]

    out.append(header)
    tree_items: list[object] = [
        ("VFA inputs", vfa_lines),
        ("Account value & in-force (key months)", av_lines),
        ("Guarantee floors (GMDB / GMAB)", floor_lines),
        ("BEL / CSM trajectory (key months)", belcsm_lines),
        ("CSM roll-forward (key months)", csm_lines),
        ("Final (headline numbers, per policy)", final_lines),
    ]
    _emit_tree(tree_items, out, "")
    file.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# show_trace_diff -- two-basis comparison
# ---------------------------------------------------------------------------

def _resolve_basis(
    assumptions: Assumptions | dict, model_points: ModelPoints, i: int,
) -> Assumptions:
    """Return the :class:`Assumptions` to use for row ``i``.

    Mirrors the dict-routing behaviour of :func:`show_trace`. Factored
    out so the diff variant can resolve two bases the same way.
    """
    if not isinstance(assumptions, dict):
        return assumptions
    if model_points.product_code is None or model_points.channel_code is None:
        raise ValueError(
            "model_points has no product / channel columns -- a dict "
            "basis cannot be routed; pass a single Assumptions instead"
        )
    key = (str(model_points.product_code[i]), str(model_points.channel_code[i]))
    try:
        return assumptions[key]
    except KeyError:
        raise KeyError(
            f"no assumptions for segment {key}; available: {list(assumptions)}"
        ) from None


def _money_delta(a: float, b: float, *, width: int = 14) -> str:
    """Format ``a -> b   (Δ, %Δ)`` for two money amounts."""
    d = b - a
    if abs(a) > 1e-12:
        pct = 100.0 * d / a
        pct_s = f"{pct:>+8.2f}%"
    else:
        pct_s = "       --"
    return f"{a:>{width},.2f}  ->  {b:>{width},.2f}   ({d:>+{width-1},.2f}, {pct_s})"


def _rate_delta(a: float, b: float) -> str:
    """Format ``a -> b   (Δ)`` for two annual rates (6-dp probabilities)."""
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


def show_trace_diff(
    mp_index: int,
    model_points: ModelPoints,
    basis_a: Assumptions | dict,
    basis_b: Assumptions | dict,
    *,
    label_a: str = "before",
    label_b: str = "after",
    file: IO | None = None,
) -> None:
    """Print a tree of how the BEL / RA / CSM of one model point moves
    when assumptions change.

    Parameters
    ----------
    mp_index
        0-based row index in ``model_points``.
    model_points
        Portfolio :class:`ModelPoints`. Subset to the single row before
        each :func:`measure` so the diff cost stays proportional to one MP.
    basis_a, basis_b
        Two assumptions to compare. Either a :class:`Assumptions` or the
        dict from :func:`fastcashflow.io.read_assumptions`. With dicts,
        each is routed independently by the model point's
        ``(product, channel)`` -- comparing two segments is also fine.
    label_a, label_b
        Short labels for the two columns in the printed diff (e.g.
        ``"baseline"`` vs ``"mortality+10%"``). Default ``"before"`` /
        ``"after"``.
    file
        Where to write. ``None`` writes to ``sys.stdout``.

    Sections
    --------
    1. Header (the model-point identity)
    2. Labels (the two basis names)
    3. Assumption changes -- only the fields that differ
    4. Rate deltas -- year-by-year annual rates side by side
    5. Cash flow deltas -- annual sum of each cash-flow component
    6. Discount factor deltas at anchor months
    7. BEL / CSM deltas at anchor months
    8. Final -- BEL / RA / FCF / CSM / loss_component, with absolute and
       percentage change

    Equal values are suppressed from the assumption-change section so
    the eye lands on what actually moved.
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

    asmp_a = _resolve_basis(basis_a, model_points, i)
    asmp_b = _resolve_basis(basis_b, model_points, i)

    sub = model_points.subset([i])
    ma = measure(sub, asmp_a)
    mb = measure(sub, asmp_b)

    # ---- Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "남" if sex_v == 0 else "여"
    age = float(sub.issue_age[0])
    term = int(sub.term_months[0])
    prem_term = (int(sub.premium_term_months[0])
                 if sub.premium_term_months is not None else term)
    count = float(sub.count[0]) if sub.count is not None else 1.0
    product = (str(model_points.product_code[i])
               if model_points.product_code is not None else "-")
    channel = (str(model_points.channel_code[i])
               if model_points.channel_code is not None else "-")
    header = (
        f"diff mp[{i}]  ({product}/{channel}, sex={sex_label}, "
        f"issue_age={age:g}, term={term}m, premium_term={prem_term}m, "
        f"count={count:g})"
    )
    labels_line = f"labels: {label_a!r}  ->  {label_b!r}"

    # ---- Assumption changes (suppressed when equal)
    asmp_diffs: list[object] = []
    for name in ("mortality_annual", "lapse_annual",
                 "waiver_incidence_annual"):
        line = _diff_callable(name,
                              getattr(asmp_a, name),
                              getattr(asmp_b, name))
        if line is not None:
            asmp_diffs.append(line)
    for name in ("discount_annual", "expense_inflation", "ra_method",
                 "ra_confidence", "cost_of_capital_rate", "mortality_cv",
                 "morbidity_cv", "longevity_cv", "disability_cv",
                 "expense_cv"):
        line = _diff_scalar(name,
                            getattr(asmp_a, name),
                            getattr(asmp_b, name))
        if line is not None:
            asmp_diffs.append(line)
    # ExpenseItem ledger -- detect added / dropped / changed rows.
    rows_a = asmp_a.expense_items
    rows_b = asmp_b.expense_items
    if rows_a != rows_b:
        asmp_diffs.append(
            f"expense_items           : len {len(rows_a)} -> len {len(rows_b)}"
        )
    # Coverages: per-coverage rate table change.
    codes_a = [r.code for r in asmp_a.coverages]
    codes_b = [r.code for r in asmp_b.coverages]
    if codes_a != codes_b:
        asmp_diffs.append(
            f"coverages (codes)      : {codes_a} -> {codes_b}"
        )
    else:
        for ra, rb in zip(asmp_a.coverages, asmp_b.coverages):
            line = _diff_callable(f"coverage[{ra.code}].rate",
                                   ra.rate, rb.rate)
            if line is not None:
                asmp_diffs.append(line)
    if not asmp_diffs:
        asmp_diffs.append("(no changes in tracked fields)")

    # ---- Rate deltas at sampled years
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
    def _maybe_row(label: str, av: float, bv: float) -> str | None:
        return None if abs(av - bv) <= 1e-12 else f"{label}  {_rate_delta(av, bv)}"

    for y in year_picks:
        block: list[object] = []
        row = _maybe_row(
            "mortality(annual)",
            _eval_rate(asmp_a.mortality_annual, sex_v, age, y,
                       issue_class_v, elapsed_v),
            _eval_rate(asmp_b.mortality_annual, sex_v, age, y,
                       issue_class_v, elapsed_v),
        )
        if row is not None:
            block.append(row)
        row = _maybe_row(
            "lapse(annual)    ",
            _eval_rate(asmp_a.lapse_annual, sex_v, age, y,
                       issue_class_v, elapsed_v),
            _eval_rate(asmp_b.lapse_annual, sex_v, age, y,
                       issue_class_v, elapsed_v),
        )
        if row is not None:
            block.append(row)
        if (asmp_a.waiver_incidence_annual is not None
                or asmp_b.waiver_incidence_annual is not None):
            row = _maybe_row(
                "waiver(annual)   ",
                _eval_rate(asmp_a.waiver_incidence_annual, sex_v, age, y,
                           issue_class_v, elapsed_v),
                _eval_rate(asmp_b.waiver_incidence_annual, sex_v, age, y,
                           issue_class_v, elapsed_v),
            )
            if row is not None:
                block.append(row)
        if codes_a == codes_b:
            for ra, rb in zip(asmp_a.coverages, asmp_b.coverages):
                row = _maybe_row(
                    f"{ra.code}(annual)".ljust(17),
                    _eval_rate(ra.rate, sex_v, age, y,
                               issue_class_v, elapsed_v),
                    _eval_rate(rb.rate, sex_v, age, y,
                               issue_class_v, elapsed_v),
                )
                if row is not None:
                    block.append(row)
        if block:
            rate_lines.append((f"year {y:>2d}", block))
    if len(rate_lines) == 1:                              # only the axes line
        rate_lines.append("(no rate changes at sampled years)")

    # ---- Cash flow deltas
    cf_a = ma.cashflows
    cf_b = mb.cashflows
    cf_lines: list[object] = []
    headers = ["year", "stream", f"sum({label_a})", f"sum({label_b})",
               "Δ", "%Δ"]
    cf_lines.append("  ".join(f"{h:>14}" for h in headers))
    for y in range(n_years):
        a0 = y * 12
        a1 = min(a0 + 12, cf_a.n_time)
        if a1 <= a0:
            break
        for name in ("premium_cf", "claim_cf", "morbidity_cf",
                     "expense_cf", "annuity_cf", "surrender_cf",
                     "disability_cf"):
            sa = float(getattr(cf_a, name)[0, a0:a1].sum())
            sb = float(getattr(cf_b, name)[0, a0:a1].sum())
            if abs(sa) + abs(sb) < 0.5:
                continue                          # both effectively zero
            if abs(sb - sa) < 0.5:
                continue                          # change rounds to nothing
            d = sb - sa
            pct = (100.0 * d / sa) if abs(sa) > 1e-12 else float("nan")
            pct_s = f"{pct:>+8.2f}%" if not np.isnan(pct) else "      --"
            cf_lines.append(
                f"{y:>14d}  {name[:-3]:>14}  "
                f"{sa:>14,.0f}  {sb:>14,.0f}  {d:>+14,.0f}  {pct_s:>14}"
            )
    if len(cf_lines) == 1:                                # only the header
        cf_lines.append("(no cash flow changes)")
    if float(cf_a.maturity_cf[0]) != 0.0 or float(cf_b.maturity_cf[0]) != 0.0:
        sa = float(cf_a.maturity_cf[0])
        sb = float(cf_b.maturity_cf[0])
        cf_lines.append(
            f"maturity benefit at t={term}m: {sa:,.0f}  ->  {sb:,.0f}  "
            f"({sb - sa:+,.0f})"
        )

    # ---- Discount factor deltas
    picks = _key_months(term, ma.discount_start.shape[0] - 1)
    disc_lines: list[object] = [
        f"t={t:>4d}m: ds  {ma.discount_start[t]:.6f}  ->  "
        f"{mb.discount_start[t]:.6f}  ({mb.discount_start[t] - ma.discount_start[t]:+.6f})"
        for t in picks
    ]

    # ---- BEL / CSM deltas at anchor months
    bel_a, bel_b = ma.bel[0], mb.bel[0]
    csm_a, csm_b = ma.csm[0], mb.csm[0]
    bel_lines: list[object] = [
        f"BEL[{t:>4d}]   {_money_delta(float(bel_a[t]), float(bel_b[t]))}"
        for t in picks
    ]
    csm_lines: list[object] = [
        f"CSM[{t:>4d}]   {_money_delta(float(csm_a[t]), float(csm_b[t]))}"
        for t in picks
    ]

    # ---- Final headline
    ra_a, ra_b = float(ma.ra[0, 0]), float(mb.ra[0, 0])
    bel_a0, bel_b0 = float(bel_a[0]), float(bel_b[0])
    fcf_a, fcf_b = bel_a0 + ra_a, bel_b0 + ra_b
    csm_a0, csm_b0 = float(csm_a[0]), float(csm_b[0])
    lc_a, lc_b = float(ma.loss_component[0]), float(mb.loss_component[0])
    final_lines: list[object] = [
        f"BEL              {_money_delta(bel_a0, bel_b0)}",
        f"RA               {_money_delta(ra_a, ra_b)}",
        f"FCF = BEL+RA     {_money_delta(fcf_a, fcf_b)}",
        f"CSM = max(0,-FCF){_money_delta(csm_a0, csm_b0)}",
        f"loss_component   {_money_delta(lc_a, lc_b)}",
    ]

    # ---- Assemble
    out.append(header)
    out.append(labels_line)
    tree_items: list[object] = [
        ("Assumption changes", asmp_diffs),
        ("Rate deltas (per policy year)", rate_lines),
        (f"Cash flow deltas (annual sum, non-zero rows only)", cf_lines),
        ("Discount factor deltas (key months)", disc_lines),
        ("BEL deltas (key months)", bel_lines),
        ("CSM deltas (key months)", csm_lines),
        ("Final (headline change, per policy)", final_lines),
    ]
    _emit_tree(tree_items, out, "")
    file.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# show_bel_step -- term-by-term unrolling of the BEL backward recursion
# ---------------------------------------------------------------------------

def show_bel_step(
    mp_index: int,
    model_points: ModelPoints,
    assumptions: Assumptions | dict,
    *,
    months: list[int] | None = None,
    file: IO | None = None,
) -> None:
    """Print, term by term, how one model point's BEL[t] is built.

    The kernel runs the IFRS 17 backward recursion::

        BEL[t] = annuity[t] - premium[t]
               + (claim + morbidity + disability + expense + surrender)[t]
                 * (1 + i[t])^(-1/2)
               + BEL[t+1] * (1 + i[t])^(-1)

    seeded by ``BEL[term] = maturity_benefit``. This function unrolls the
    equation at chosen months: prints each cash-flow component at ``t``,
    the half-month and full-month discount factors, the mid-term piece
    (cash flows at mid-month) and the tail piece (carry from the next
    month), then the resulting ``BEL[t]`` against the engine's value.
    When the printed identity holds the engine and a hand calculation
    are in agreement; when it does not, the offending term is right
    there in the row.

    Parameters
    ----------
    mp_index
        0-based row index in ``model_points``.
    model_points
        Portfolio :class:`ModelPoints`. Subset to the single row before
        running :func:`measure`.
    assumptions
        A :class:`Assumptions` or the dict from
        :func:`fastcashflow.io.read_assumptions` (routed by the row's
        ``(product, channel)`` like :func:`show_trace`).
    months
        Anchor months at which to unroll the recursion. ``None`` uses
        ``{0, 12, term//2, term-1, term}`` -- inception, end of year 1,
        the half-way point, the last recursion step, and the seed.
        Out-of-range entries are ignored.
    file
        Where to write. ``None`` -> ``sys.stdout``.
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
    asmp = _resolve_basis(assumptions, model_points, i)
    sub = model_points.subset([i])
    m = measure(sub, asmp)

    term = int(sub.term_months[0])
    n_time = m.cashflows.n_time
    if months is None:
        months = sorted({0, 12, term // 2, term - 1, term})
    months = [int(t) for t in months if 0 <= int(t) <= term]

    # Monthly discount rate curve -- the kernel reads this directly. We
    # recover ``i[t]`` from the same curve here so the printed identity
    # uses the engine's numbers, not a parallel computation that could
    # drift.
    monthly_rate = discount_monthly_curve(asmp, n_time)

    # Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "남" if sex_v == 0 else "여"
    age = float(sub.issue_age[0])
    product = (str(model_points.product_code[i])
               if model_points.product_code is not None else "-")
    channel = (str(model_points.channel_code[i])
               if model_points.channel_code is not None else "-")
    header = (
        f"mp[{i}] BEL step-by-step  ({product}/{channel}, sex={sex_label}, "
        f"issue_age={age:g}, term={term}m)"
    )

    recursion_lines: list[object] = [
        "BEL[t] = annuity[t] - premium[t]",
        "       + (claim + morbidity + disability + expense + surrender)[t]"
        " * (1 + i[t])^(-1/2)",
        "       + BEL[t+1] * (1 + i[t])^(-1)",
        f"seed:   BEL[{term}] = maturity_benefit = "
        f"{float(m.cashflows.maturity_cf[0]):,.2f}",
    ]

    step_blocks: list[object] = []
    bel_engine = m.bel[0]
    cf = m.cashflows
    for t in months:
        if t == term:
            step_blocks.append((
                f"t={t:>4d}  (seed -- no recursion below)",
                [f"BEL[{t}] = {float(bel_engine[t]):>15,.2f}  "
                 "(= maturity_benefit)"]
            ))
            continue
        # Inside the recursion: 0 <= t < term.
        rate = float(monthly_rate[t])
        half = (1.0 + rate) ** (-0.5)
        full = 1.0 / (1.0 + rate)
        prem = float(cf.premium_cf[0, t])
        ann = float(cf.annuity_cf[0, t])
        claim = float(cf.claim_cf[0, t])
        morb = float(cf.morbidity_cf[0, t])
        disab = float(cf.disability_cf[0, t])
        exp = float(cf.expense_cf[0, t])
        surr = float(cf.surrender_cf[0, t])
        mid_sum = claim + morb + disab + exp + surr
        mid_piece = mid_sum * half
        tail_piece = float(bel_engine[t + 1]) * full
        bel_recompute = ann - prem + mid_piece + tail_piece
        residual = bel_recompute - float(bel_engine[t])
        block: list[object] = [
            f"i[t]                      = {rate:.6f}",
            f"half = (1+i)^(-1/2)       = {half:.6f}",
            f"full = (1+i)^(-1)         = {full:.6f}",
            f"premium[t]                = {prem:>15,.2f}",
            f"annuity[t]                = {ann:>15,.2f}",
            f"claim[t]                  = {claim:>15,.2f}",
            f"morbidity[t]              = {morb:>15,.2f}",
            f"disability[t]             = {disab:>15,.2f}",
            f"expense[t]                = {exp:>15,.2f}",
            f"surrender[t]              = {surr:>15,.2f}",
            f"mid-month sum             = {mid_sum:>15,.2f}",
            f"mid-month piece (×half)   = {mid_piece:>15,.2f}",
            f"BEL[t+1]                  = {float(bel_engine[t + 1]):>15,.2f}",
            f"tail piece (BEL[t+1]×full)= {tail_piece:>15,.2f}",
            f"recomputed BEL[t]         = {bel_recompute:>15,.2f}",
            f"engine BEL[t]             = {float(bel_engine[t]):>15,.2f}  "
            f"(residual {residual:+.4e})",
        ]
        step_blocks.append((f"t={t:>4d}", block))

    out.append(header)
    tree_items: list[object] = [
        ("Recursion (back-pass)", recursion_lines),
        ("Steps", step_blocks),
        ("Inception BEL", [
            f"BEL[0] = {float(bel_engine[0]):>15,.2f}",
        ]),
    ]
    _emit_tree(tree_items, out, "")
    file.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# show_csm_step -- term-by-term unrolling of the CSM forward recursion
# ---------------------------------------------------------------------------

def show_csm_step(
    mp_index: int,
    model_points: ModelPoints,
    assumptions: Assumptions | dict,
    *,
    months: list[int] | None = None,
    file: IO | None = None,
) -> None:
    """Print, term by term, how one model point's CSM[t] is built.

    The kernel runs the forward recursion::

        csm[0]   = max(0, -(BEL[0] + RA[0]))
        csm[t]   = csm[t-1] + accretion[t-1] - release[t-1]
        accretion[t-1] = csm[t-1] * i[t-1]
        release[t-1]   = (csm[t-1] + accretion[t-1])
                         * coverage_units[t-1] / sum(coverage_units[t-1:])

    ``coverage_units`` is the in-force survival series. This function
    unrolls the step at chosen months: prints the prior CSM, the
    monthly rate, the accretion, the coverage-unit share consumed in
    that month, the release amount, and the resulting CSM[t] against
    the engine's value.

    For an onerous contract (``csm[0] == 0``), every subsequent step is
    zero too -- the printed trace then visibly says so, which is itself
    useful when checking that the engine is honouring the floor.

    Parameters
    ----------
    mp_index
        0-based row index in ``model_points``.
    model_points, assumptions, file
        Same shape as :func:`show_bel_step`.
    months
        Months at which to unroll the step (each row shows the
        computation that produced ``csm[t]`` from ``csm[t-1]``).
        ``None`` uses ``{1, 12, term//2, term}``. Out-of-range entries
        are ignored. ``t = 0`` is the seed and is always printed
        regardless of ``months``.
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
    asmp = _resolve_basis(assumptions, model_points, i)
    sub = model_points.subset([i])
    m = measure(sub, asmp)

    term = int(sub.term_months[0])
    n_time = m.cashflows.n_time
    if months is None:
        months = sorted({1, 12, term // 2, term})
    # Recursion produces csm[t] for t in 1..n_time. t = 0 is the seed.
    months = [int(t) for t in months if 1 <= int(t) <= n_time]

    monthly_rate = discount_monthly_curve(asmp, n_time)
    inforce = m.cashflows.inforce[0]              # (n_time,)
    bel0 = float(m.bel[0, 0])
    ra0 = float(m.ra[0, 0])
    fcf0 = bel0 + ra0
    csm = m.csm[0]
    csm_acc = m.csm_accretion[0]
    csm_rel = m.csm_release[0]

    # Pre-compute the reverse-cumulative coverage-unit tail the kernel
    # uses for the release fraction; printing it makes the share that
    # falls into ``t`` explicit.
    cu_tail = np.empty(n_time)
    running = 0.0
    for s in range(n_time - 1, -1, -1):
        running += inforce[s]
        cu_tail[s] = running

    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "남" if sex_v == 0 else "여"
    age = float(sub.issue_age[0])
    product = (str(model_points.product_code[i])
               if model_points.product_code is not None else "-")
    channel = (str(model_points.channel_code[i])
               if model_points.channel_code is not None else "-")
    header = (
        f"mp[{i}] CSM step-by-step  ({product}/{channel}, sex={sex_label}, "
        f"issue_age={age:g}, term={term}m)"
    )

    recursion_lines: list[object] = [
        "csm[0]   = max(0, -(BEL[0] + RA[0]))",
        "csm[t]   = csm[t-1] + accretion[t-1] - release[t-1]",
        "accretion[t-1] = csm[t-1] * i[t-1]",
        "release[t-1]   = (csm[t-1] + accretion[t-1])"
        " * coverage_units[t-1] / sum(coverage_units[t-1:])",
    ]

    seed_lines: list[object] = [
        f"BEL[0]               = {bel0:>15,.2f}",
        f"RA[0]                = {ra0:>15,.2f}",
        f"FCF[0] = BEL + RA    = {fcf0:>15,.2f}",
        f"csm[0] = max(0,-FCF) = {float(csm[0]):>15,.2f}",
    ]
    onerous = float(csm[0]) <= 0.0
    if onerous:
        seed_lines.append(
            "onerous contract -- csm = 0 throughout; release/accretion "
            "are 0 by construction."
        )

    step_blocks: list[object] = []
    for t in months:
        prior_csm = float(csm[t - 1])
        rate = float(monthly_rate[t - 1])
        acc = float(csm_acc[t - 1])
        accreted = prior_csm + acc
        cu = float(inforce[t - 1])
        cu_rem = float(cu_tail[t - 1])
        rel = float(csm_rel[t - 1])
        if cu_rem > 1e-12:
            rel_frac = cu / cu_rem
            frac_line = (
                f"release fraction          = cov_units / cu_tail "
                f"= {cu:.6f} / {cu_rem:.6f} = {rel_frac:.6f}"
            )
        else:
            frac_line = (
                "release fraction          = 0 (cu_tail below epsilon -- "
                "all in-force already exited)"
            )
        recomputed = accreted - rel
        residual = recomputed - float(csm[t])
        block: list[object] = [
            f"csm[t-1]                  = {prior_csm:>15,.2f}",
            f"i[t-1]                    = {rate:.6f}",
            f"accretion[t-1] = csm*i    = {acc:>15,.2f}",
            f"accreted = csm + acc      = {accreted:>15,.2f}",
            f"coverage_units[t-1]       = {cu:.6f}",
            f"cu_tail[t-1] = sum(cu[t-1:]) = {cu_rem:.6f}",
            frac_line,
            f"release[t-1] = accreted * frac = {rel:>15,.2f}",
            f"recomputed csm[t]         = {recomputed:>15,.2f}",
            f"engine csm[t]             = {float(csm[t]):>15,.2f}  "
            f"(residual {residual:+.4e})",
        ]
        step_blocks.append((f"t={t:>4d}", block))

    out.append(header)
    tree_items: list[object] = [
        ("Recursion (forward-pass)", recursion_lines),
        ("Seed (t = 0)", seed_lines),
        ("Steps", step_blocks if step_blocks else
         ["(no recursion months requested in range)"]),
        ("End CSM", [f"csm[{n_time}] = {float(csm[n_time]):>15,.2f}"]),
    ]
    _emit_tree(tree_items, out, "")
    file.write("\n".join(out) + "\n")
