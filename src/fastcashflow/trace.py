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

from fastcashflow.basis import Basis, BasisRouter
from fastcashflow.coverage import CalculationMethod, method_attrs
from fastcashflow.curves import discount_factors, discount_monthly_curve
from fastcashflow._typing import FloatArray
from fastcashflow.engine import measure
from fastcashflow.model_points import ModelPoints, NO_GUARANTEE_RATE
from fastcashflow.numerics import _norm_ppf
from fastcashflow._paa import measure_paa
from fastcashflow._reinsurance import QuotaShare, Treaty, measure_reinsurance
from fastcashflow._vfa import measure_vfa


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


def show_trace(
    mp_index: int,
    model_points: ModelPoints,
    basis: Basis | dict,
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
    basis
        A single :class:`Basis`, or the :class:`~fastcashflow.BasisRouter`
        returned by :func:`fastcashflow.io.read_basis` /
        :func:`fastcashflow.io.load_sample_basis`. With the router
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

    # Multi-segment BasisRouter: route to the right segment by (product, channel).
    if isinstance(basis, BasisRouter):
        if model_points.product is None or model_points.channel is None:
            raise ValueError(
                "model_points has no product / channel columns -- a "
                "BasisRouter cannot be routed; pass a single Basis instead"
            )
        key = (str(model_points.product[i]), str(model_points.channel[i]))
        try:
            basis = basis.segments[key]
        except KeyError:
            raise KeyError(
                f"no basis for segment {key}; "
                f"available: {list(basis.segments)}"
            ) from None

    # Single-row slice + measure. Subsetting first keeps the trace cost
    # proportional to one MP, not the whole portfolio.
    sub = model_points.subset([i])
    m = measure(sub, basis)

    # ---- Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "M" if sex_v == 0 else "F"
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

    # ---- Basis (segment-level)
    basis_lines: list[object] = []
    basis_lines.append(
        f"mortality_annual     -> {_fmt_callable(basis.mortality_annual)}"
    )
    basis_lines.append(
        f"lapse_annual         -> {_fmt_callable(basis.lapse_annual)}"
    )
    if basis.waiver_incidence_annual is not None:
        basis_lines.append(
            f"waiver_incidence     -> {_fmt_callable(basis.waiver_incidence_annual)}"
        )
    d = basis.discount_annual
    if np.ndim(d) == 0:
        basis_lines.append(f"discount_annual      = {float(d):g} (flat)")
    else:
        arr = np.asarray(d)
        basis_lines.append(
            f"discount_annual      = ndarray len={arr.size} "
            f"[{arr.flat[0]:g}, ..., {arr.flat[-1]:g}]"
        )
    infl = basis.expense_inflation
    if np.ndim(infl) == 0:
        basis_lines.append(f"expense_inflation    = {float(infl):g} (flat)")
    else:
        arr = np.asarray(infl)
        basis_lines.append(
            f"expense_inflation    = ndarray len={arr.size} "
            f"[{arr.flat[0]:g}, ..., {arr.flat[-1]:g}]"
        )
    rows = basis.expense_items
    if not rows:
        basis_lines.append("expense_items        = ()  (no expense)")
    else:
        row_lines: list[object] = [
            (f"ExpenseItem({r.expense_type!r}, basis={r.basis!r}, "
             f"value={r.value:g})")
            for r in rows
        ]
        basis_lines.append(
            (f"expense_items        = tuple  (len={len(rows)})", row_lines)
        )
    basis_lines.append(
        f"ra: method={basis.ra_method!r}, conf={basis.ra_confidence:g}"
    )
    basis_lines.append(
        f"cv: mort={basis.mortality_cv:g} morb={basis.morbidity_cv:g} "
        f"long={basis.longevity_cv:g} disab={basis.disability_cv:g}"
    )

    # ---- Coverages (rate-driven)
    cov_lines: list[object] = []
    methods = model_points.calculation_methods or {}
    for r in basis.coverages:
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
    has_waiver = basis.waiver_incidence_annual is not None
    cov_headers = [f"{r.code}(an)" for r in basis.coverages]
    head_row = ["year", "mort(an)", "lapse(an)"]
    if has_waiver:
        head_row.append("waiver(an)")
    head_row.extend(cov_headers)
    rate_lines.append("  ".join(f"{h:>12}" for h in head_row))
    for y in year_picks:
        row = [f"{y:>12d}"]
        row.append(
            f"{_eval_rate(basis.mortality_annual, sex_v, age, y, issue_class_v, elapsed_v):>12.6f}"
        )
        row.append(
            f"{_eval_rate(basis.lapse_annual, sex_v, age, y, issue_class_v, elapsed_v):>12.6f}"
        )
        if has_waiver:
            row.append(
                f"{_eval_rate(basis.waiver_incidence_annual, sex_v, age, y, issue_class_v, elapsed_v):>12.6f}"
            )
        for r in basis.coverages:
            row.append(
                f"{_eval_rate(r.rate, sex_v, age, y, issue_class_v, elapsed_v):>12.6f}"
            )
        rate_lines.append("  ".join(row))

    # ---- Cash flows (annual sums of the monthly trajectory)
    cf = m.cashflows
    bel = m.bel_path[0]
    ra = m.ra_path[0]
    csm = m.csm_path[0]
    csm_acc = m.csm_accretion[0]
    csm_rel = m.csm_release[0]
    lc = m.loss_component[0]
    discount_bom = m.discount_bom

    cf_lines: list[object] = []
    cf_names = ["premium_cf", "claim_cf", "morbidity_cf", "expense_cf",
                "annuity_cf", "surrender_cf", "disability_cf"]
    cf_heads = ["premium", "claim", "morbidity", "expense",
                "annuity", "surrender", "disability"]
    cf_rows: list[object] = []  # (year, [column sums])
    for y in range(n_years):
        a0 = y * 12
        a1 = min(a0 + 12, cf.n_time)
        if a1 <= a0:
            break
        cf_rows.append((y, [getattr(cf, nm)[0][a0:a1].sum() for nm in cf_names]))
    # Per-column width = max(header, 12, widest value) -- expand only, so a
    # large-portfolio sum that needs >12 digits does not break the columns.
    cw = [max(len(h), _colw((vals[j] for _, vals in cf_rows), ",.0f", 12))
          for j, h in enumerate(cf_heads)]
    cf_lines.append("  ".join(
        [f"{'year':>12}"] + [f"{h:>{w}}" for h, w in zip(cf_heads, cw)]))
    for y, vals in cf_rows:
        cf_lines.append("  ".join(
            [f"{y:>12d}"] + [f"{v:>{w},.0f}" for v, w in zip(vals, cw)]))
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
    picks = _key_months(term, discount_bom.shape[0] - 1)
    diag_pool_lines: list[object] = []
    for r in basis.coverages:
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
        f"t={t:>4d}m: ds={discount_bom[t]:.6f}" for t in picks
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
    _gcw = _colw((csm[t] for t in picks if t < csm.shape[0]), ",.2f", 14)
    _gaw = _colw([csm_acc[t] for t in picks if t < csm_acc.shape[0]] or [0.0], ",.2f", 10)
    _grw = _colw([csm_rel[t] for t in picks if t < csm_rel.shape[0]] or [0.0], ",.2f", 10)
    for t in picks:
        if t >= csm_acc.shape[0]:
            csm_lines.append(
                f"t={t:>4d}m: csm={csm[t]:>{_gcw},.2f}  "
                f"(past last accretion month)"
            )
        else:
            csm_lines.append(
                f"t={t:>4d}m: csm={csm[t]:>{_gcw},.2f}  "
                f"acc={csm_acc[t]:>{_gaw},.2f}  rel={csm_rel[t]:>{_grw},.2f}"
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
        ("Basis (segment-level)", basis_lines),
        (f"Coverages (rate-driven, n={len(basis.coverages)})",
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
    basis: Basis | dict,
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
    basis = _resolve_basis(basis, model_points, i)
    sub = model_points.subset([i])
    m = measure_vfa(sub, basis, return_scenarios=return_scenarios)

    # ---- Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "M" if sex_v == 0 else "F"
    age = float(sub.issue_age[0])
    term = int(sub.term_months[0])
    count = float(sub.count[0]) if sub.count is not None else 1.0
    product = (str(model_points.product[i])
               if model_points.product is not None else "-")
    channel = (str(model_points.channel[i])
               if model_points.channel is not None else "-")
    av0 = float(sub.account_value[0])
    gcr = float(sub.minimum_crediting_rate[0])
    # NO_GUARANTEE_RATE is a sentinel, not a rate -- render it as "none" rather
    # than the bare -1.0. A 0.0 prints as a real 0% floor.
    gcr_str = "none" if gcr == NO_GUARANTEE_RATE else f"{gcr:g}"
    gmdb = float(sub.minimum_death_benefit[0])
    gmab = float(sub.minimum_accumulation_benefit[0])
    header = (
        f"mp[{i}]  VFA  ({product}/{channel}, sex={sex_label}, "
        f"issue_age={age:g}, term={term}m, count={count:g})"
    )

    # ---- VFA inputs
    # Label column sized to the longest label so the value column stays
    # aligned regardless of field-name length; rate scalars share the
    # right-aligned value column with the amounts.
    _w = 28  # len("minimum_accumulation_benefit")
    _vw = max(_colw([av0, gmdb, gmab], ",.2f", 15),
              _colw([gcr, basis.investment_return, basis.fund_fee], "g", 15))
    vfa_lines: list[object] = [
        f"{'account_value':<{_w}} = {av0:>{_vw},.2f}",
        f"{'minimum_crediting_rate':<{_w}} = {gcr_str:>{_vw}}",
        f"{'minimum_death_benefit':<{_w}} = {gmdb:>{_vw},.2f}  (GMDB)",
        f"{'minimum_accumulation_benefit':<{_w}} = {gmab:>{_vw},.2f}  (GMAB)",
        f"{'investment_return':<{_w}} = {basis.investment_return:>{_vw}g}  (VFA discount / accrual basis)",
        f"{'fund_fee':<{_w}} = {basis.fund_fee:>{_vw}g}  (= source of profit)",
        f"{'mortality_annual':<{_w}} -> {_fmt_callable(basis.mortality_annual)}",
        f"{'lapse_annual':<{_w}} -> {_fmt_callable(basis.lapse_annual)}",
        f"{'ra':<{_w}} -> method={basis.ra_method!r} conf={basis.ra_confidence:g} "
        f"expense_cv={basis.expense_cv:g}",
    ]

    # ---- Trajectories (from the VFA measurement)
    av = m.account_value_path[0]
    cf = m.cashflows
    inforce = cf.inforce[0]
    deaths = cf.deaths[0]
    survivors = float(cf.maturity_survivors[0])
    n_time = cf.n_time
    picks = _key_months(term, n_time)

    _avw = _colw((av[t] for t in picks if t < av.shape[0]), ",.2f", 15)
    av_lines: list[object] = []
    for t in picks:
        if t >= av.shape[0]:
            continue
        inf_v = inforce[t] if t < n_time else 0.0
        av_lines.append(f"t={t:>4d}m: AV={av[t]:>{_avw},.2f}  inforce={inf_v:.6f}")

    # ---- Guarantee floors (where they bite)
    # Build rows as (left, amount, excess, rate_label, rate); the left and
    # rate-label columns are padded to a common width so the amount / excess /
    # rate columns line up across the death rows and the maturity row.
    ti = max(0, term - 1)
    floor_rows = [
        (f"t={t:>4d}m: death=max(AV,GMDB)", max(av[t], gmdb),
         max(0.0, gmdb - av[t]), "deaths", float(deaths[t]))
        for t in picks if t < n_time
    ]
    floor_rows.append(
        (f"maturity@t={ti}m: max(AV,GMAB)", max(av[ti], gmab),
         max(0.0, gmab - av[ti]), "survivors", float(survivors))
    )
    lw = max(len(r[0]) for r in floor_rows)
    rw = max(len(r[3]) for r in floor_rows)
    aw = _colw((r[1] for r in floor_rows), ",.2f", 15)  # amount: expand from 15
    ew = _colw((r[2] for r in floor_rows), ",.2f", 12)  # excess: expand from 12
    floor_lines: list[object] = [
        "death[t] = max(AV[t], GMDB);  maturity = max(AV[term-1], GMAB)",
    ]
    floor_lines += [
        f"{left:<{lw}} ={amt:>{aw},.2f}  excess={ex:>{ew},.2f}  {rl:>{rw}}={rate:.6f}"
        for left, amt, ex, rl, rate in floor_rows
    ]

    # ---- BEL / CSM trajectory + roll-forward
    bel = m.bel_path[0]
    ra = m.ra_path[0]
    csm = m.csm_path[0]
    csm_acc = m.csm_accretion[0]
    csm_rel = m.csm_release[0]
    _pbc = [t for t in picks if t < bel.shape[0]]
    _bw = _colw((bel[t] for t in _pbc), ",.2f", 15)
    _cw = _colw((csm[t] for t in _pbc), ",.2f", 15)
    belcsm_lines: list[object] = [
        f"t={t:>4d}m: BEL={bel[t]:>{_bw},.2f}  CSM={csm[t]:>{_cw},.2f}"
        for t in _pbc
    ]
    _csw = _colw((csm[t] for t in picks if t < csm.shape[0]), ",.2f", 14)
    _acw = _colw([csm_acc[t] for t in picks if t < csm_acc.shape[0]] or [0.0], ",.2f", 10)
    _rew = _colw([csm_rel[t] for t in picks if t < csm_rel.shape[0]] or [0.0], ",.2f", 10)
    csm_lines: list[object] = ["csm[t+1] = csm[t] + accretion[t] - release[t]"]
    for t in picks:
        if t >= csm_acc.shape[0]:
            csm_lines.append(f"t={t:>4d}m: csm={csm[t]:>{_csw},.2f}  (past last accretion)")
        else:
            csm_lines.append(
                f"t={t:>4d}m: csm={csm[t]:>{_csw},.2f}  "
                f"acc={csm_acc[t]:>{_acw},.2f}  rel={csm_rel[t]:>{_rew},.2f}"
            )

    # ---- Final headline
    # The guarantee time value is carried into the fulfilment cash flows:
    # FCF = BEL + RA + TVOG, then CSM = max(0, -FCF) and loss = max(0, FCF).
    # Showing the FCF line makes the CSM / loss split reconcile -- otherwise
    # a reader cannot see why a negative BEL still leaves CSM = 0.
    fee = float(m.variable_fee[0])
    tv = float(m.time_value[0])
    lc = float(m.loss_component[0])
    fcf0 = float(bel[0] + ra[0] + tv)
    if return_scenarios is None:
        fcf_label, tv_note = "FCF = BEL + RA   ", "(no scenarios -> intrinsic only)"
    else:
        fcf_label, tv_note = "FCF = BEL+RA+TVOG", "(time value of the guarantee)"
    outcome = ("-> onerous (TVOG exceeds the unearned fee)" if lc > 0.0
               else "-> profitable (CSM absorbs it)")
    _fw = _colw([fee, float(bel[0]), float(ra[0]), tv, fcf0, float(csm[0]), lc], ",.2f", 15)
    final_lines: list[object] = [
        f"variable_fee     = {fee:>{_fw},.2f}  (fee PV = source of profit)",
        f"BEL              = {bel[0]:>{_fw},.2f}",
        f"RA               = {ra[0]:>{_fw},.2f}",
        f"TVOG (time_value)= {tv:>{_fw},.2f}  {tv_note}",
        f"{fcf_label}= {fcf0:>{_fw},.2f}",
        f"CSM = max(0,-FCF)= {csm[0]:>{_fw},.2f}",
        f"loss_component   = {lc:>{_fw},.2f}  {outcome}",
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


def show_trace_paa(
    mp_index: int,
    model_points: ModelPoints,
    basis: Basis | dict,
    *,
    revenue_basis: str = "time",
    file: IO | None = None,
) -> None:
    """Print a tree of how one PAA model point's LRC / revenue / LIC is built.

    The PAA (Premium Allocation Approach, the short-duration simplification)
    counterpart of :func:`show_trace`. PAA has no CSM -- the liability for
    remaining coverage (LRC) is an unearned-premium-style balance -- so the
    tree shows the LRC roll-forward (premium in, revenue released), the
    insurance service result (revenue less service expense) and the
    liability for incurred claims (LIC). Use it on PAA contracts;
    :func:`show_trace` traces the GMM ``measure`` (BEL / RA / CSM).
    """
    out: list[str] = []
    if file is None:
        file = sys.stdout
    n_mp = model_points.n_mp
    if not 0 <= mp_index < n_mp:
        raise IndexError(f"mp_index {mp_index} out of range for n_mp={n_mp}")
    i = mp_index
    basis = _resolve_basis(basis, model_points, i)
    sub = model_points.subset([i])
    m = measure_paa(sub, basis, revenue_basis=revenue_basis)

    # ---- Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "M" if sex_v == 0 else "F"
    age = float(sub.issue_age[0])
    term = int(sub.term_months[0])
    count = float(sub.count[0]) if sub.count is not None else 1.0
    product = (str(model_points.product[i])
               if model_points.product is not None else "-")
    channel = (str(model_points.channel[i])
               if model_points.channel is not None else "-")
    header = (
        f"mp[{i}]  PAA  ({product}/{channel}, sex={sex_label}, "
        f"issue_age={age:g}, term={term}m, count={count:g})"
    )

    cf = m.cashflows
    premium = cf.premium_cf[0]
    n_time = cf.n_time
    picks = _key_months(term, n_time)
    lrc = m.lrc_path[0]
    revenue = m.revenue[0]
    svc_exp = m.service_expense[0]
    svc_result = m.service_result[0]
    lic = m.lic[0]
    lc = float(m.loss_component[0])

    sp = basis.settlement_pattern
    sp_desc = ("None (no payment lag -> LIC=0)" if sp is None
               else f"len={np.asarray(sp).size}")
    basis_desc = ("B126(a) time-based" if revenue_basis == "time"
                  else "B126(b) claims-based")

    # ---- PAA inputs
    paa_lines: list[object] = [
        f"premium_total      = {float(premium.sum()):>15,.2f}",
        f"revenue_basis      = {revenue_basis!r}  ({basis_desc})",
        f"settlement_pattern = {sp_desc}  (payment spread of incurred claims = LIC)",
        f"mortality_annual   -> {_fmt_callable(basis.mortality_annual)}",
        f"lapse_annual       -> {_fmt_callable(basis.lapse_annual)}",
        f"ra: method={basis.ra_method!r} conf={basis.ra_confidence:g} "
        f"(for the onerous test)",
    ]

    # ---- LRC roll-forward
    lrc_lines: list[object] = [
        "LRC[t+1] = LRC[t] + premium[t] - revenue[t]   (LRC[0] = 0)",
    ]
    _pw = _colw((premium[t] for t in picks if t < n_time), ",.2f", 13)
    _rw = _colw((revenue[t] for t in picks if t < n_time), ",.2f", 13)
    _lw = _colw((lrc[t] for t in picks if t < n_time), ",.2f", 15)
    for t in picks:
        if t >= n_time:
            continue
        lrc_lines.append(
            f"t={t:>4d}m: prem={premium[t]:>{_pw},.2f}  rev={revenue[t]:>{_rw},.2f}  "
            f"LRC[t]={lrc[t]:>{_lw},.2f}"
        )

    # ---- Insurance service result
    result_lines: list[object] = [
        "service_result[t] = revenue[t] - service_expense[t]",
    ]
    _sw = _colw((svc_exp[t] for t in picks if t < n_time), ",.2f", 13)
    _rsw = _colw((svc_result[t] for t in picks if t < n_time), ",.2f", 13)
    for t in picks:
        if t >= n_time:
            continue
        result_lines.append(
            f"t={t:>4d}m: rev={revenue[t]:>{_rw},.2f}  svc_exp={svc_exp[t]:>{_sw},.2f}  "
            f"result={svc_result[t]:>{_rsw},.2f}"
        )

    # ---- LIC
    lic_lines: list[object] = [
        f"t={t:>4d}m: LIC={lic[t]:>15,.2f}" for t in picks if t < lic.shape[0]
    ]

    # ---- Final headline
    final_lines: list[object] = [
        f"LRC[0]                = {lrc[0]:>15,.2f}  (= 0, before premium inflow)",
        f"total revenue         = {float(revenue.sum()):>15,.2f}  (= total premium)",
        f"total service_expense = {float(svc_exp.sum()):>15,.2f}",
        f"insurance svc result  = {float(svc_result.sum()):>15,.2f}",
        f"loss_component        = {lc:>15,.2f}  (onerous; from the GMM FCF)",
        f"LIC (peak)            = {float(lic.max()):>15,.2f}",
        "(PAA has no CSM -- LRC is the unearned-premium balance)",
    ]

    out.append(header)
    tree_items: list[object] = [
        ("PAA inputs", paa_lines),
        ("LRC roll-forward (key months)", lrc_lines),
        ("Insurance service result (key months)", result_lines),
        ("LIC -- liability for incurred claims (key months)", lic_lines),
        ("Final (headline numbers, per policy)", final_lines),
    ]
    _emit_tree(tree_items, out, "")
    file.write("\n".join(out) + "\n")


def show_trace_reinsurance(
    mp_index: int,
    model_points: ModelPoints,
    basis: Basis | dict,
    *,
    treaty: Treaty,
    file: IO | None = None,
) -> None:
    """Print a tree of how one reinsurance-held model point's BEL / RA / CSM is built.

    The reinsurance-held counterpart of :func:`show_trace`. The ``treaty`` (e.g.
    :class:`~fastcashflow.reinsurance.QuotaShare`) cedes the direct portfolio's
    claims and premiums; the tree shows the ceded flows (recovery, reinsurance
    premium), the BEL = PV(reinsurance premium) - PV(recovery) (a net cost when
    positive), the RA = risk transferred (the margin on the ceded claims, IFRS 17
    para 64), and the CSM = -(BEL - RA) -- the net cost or gain of the cover,
    which may be negative and carries no loss component (para 65). Use it on a
    reinsurance held; :func:`show_trace` traces the direct GMM ``measure``.
    """
    out: list[str] = []
    if file is None:
        file = sys.stdout
    n_mp = model_points.n_mp
    if not 0 <= mp_index < n_mp:
        raise IndexError(f"mp_index {mp_index} out of range for n_mp={n_mp}")
    i = mp_index
    basis = _resolve_basis(basis, model_points, i)
    sub = model_points.subset([i])
    m = measure_reinsurance(sub, basis, treaty=treaty)

    # ---- Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "M" if sex_v == 0 else "F"
    age = float(sub.issue_age[0])
    term = int(sub.term_months[0])
    count = float(sub.count[0]) if sub.count is not None else 1.0
    product = (str(model_points.product[i])
               if model_points.product is not None else "-")
    channel = (str(model_points.channel[i])
               if model_points.channel is not None else "-")
    header = (
        f"mp[{i}]  Reinsurance  ({product}/{channel}, sex={sex_label}, "
        f"issue_age={age:g}, term={term}m, count={count:g})"
    )

    proj = m.cashflows
    n_time = proj.n_time
    picks = _key_months(term, n_time)
    # Re-cede to split the recovery by risk (the RA weights the two cv's).
    ceded_mort, ceded_morb, reins_prem = treaty.cede(proj)
    ceded_mort, ceded_morb, reins_prem = ceded_mort[0], ceded_morb[0], reins_prem[0]
    recovery = m.recovery[0]
    discount_bom, discount_mid = discount_factors(basis, n_time)
    pv_recovery = float((recovery * discount_mid).sum())
    pv_reins_prem = float((reins_prem * discount_bom[:-1]).sum())
    pv_ceded_mort = float((ceded_mort * discount_mid).sum())
    pv_ceded_morb = float((ceded_morb * discount_mid).sum())
    z = _norm_ppf(basis.ra_confidence)
    bel = float(m.bel[0])
    ra = float(m.ra[0])
    csm0 = float(m.csm[0])
    csm = m.csm_path[0]
    acc = m.csm_accretion[0]
    rel = m.csm_release[0]

    cession_desc = (f"QuotaShare cession={treaty.cession:g}"
                    if isinstance(treaty, QuotaShare) else type(treaty).__name__)

    # ---- Treaty / inputs
    treaty_lines: list[object] = [
        f"treaty             = {cession_desc}",
        "recovery           = cession x (claim_cf + morbidity_cf)  (ceded share of direct claims)",
        "reinsurance_premium= cession x premium_cf  (reinsurance premium)",
        f"mortality_annual   -> {_fmt_callable(basis.mortality_annual)}",
        f"lapse_annual       -> {_fmt_callable(basis.lapse_annual)}",
        f"ra: conf={basis.ra_confidence:g}  mort_cv={basis.mortality_cv:g}  "
        f"morb_cv={basis.morbidity_cv:g}  (risk transferred; no loss component)",
    ]

    # ---- Ceded cash flows
    flow_lines: list[object] = []
    _rw = _colw((recovery[t] for t in picks if t < n_time), ",.2f", 13)
    _pw = _colw((reins_prem[t] for t in picks if t < n_time), ",.2f", 13)
    for t in picks:
        if t >= n_time:
            continue
        flow_lines.append(
            f"t={t:>4d}m: recovery={recovery[t]:>{_rw},.2f}  "
            f"reins_prem={reins_prem[t]:>{_pw},.2f}"
        )

    # ---- Discount factors
    disc_lines: list[object] = [
        f"t={t:>4d}m: bom={discount_bom[t]:.6f}"
        + (f"  mid={discount_mid[t]:.6f}" if t < n_time else "")
        for t in picks if t <= n_time
    ]

    # ---- BEL / RA / CSM build
    build_lines: list[object] = [
        f"PV(reinsurance_premium) = {pv_reins_prem:>15,.2f}",
        f"PV(recovery)            = {pv_recovery:>15,.2f}",
        f"BEL = PV(reins_prem) - PV(recovery) = {bel:>15,.2f}  (positive = net cost)",
        f"PV(ceded mortality)     = {pv_ceded_mort:>15,.2f}",
        f"PV(ceded morbidity)     = {pv_ceded_morb:>15,.2f}",
        f"RA = z({basis.ra_confidence:g})={z:.4f} x "
        f"(mort_cv*PV_mort + morb_cv*PV_morb) = {ra:>15,.2f}",
        f"CSM[0] = -(BEL - RA) = {csm0:>15,.2f}  (negative when a net cost; para 65)",
    ]

    # ---- CSM roll-forward
    csm_lines: list[object] = [
        "csm[t+1] = csm[t] + accretion[t] - release[t]",
    ]
    _cw = _colw((csm[t] for t in picks if t < n_time), ",.2f", 15)
    for t in picks:
        if t >= n_time:
            continue
        csm_lines.append(
            f"t={t:>4d}m: csm={csm[t]:>{_cw},.2f}  acc={acc[t]:>12,.2f}  "
            f"rel={rel[t]:>12,.2f}"
        )

    # ---- Final headline
    final_lines: list[object] = [
        f"BEL = {bel:>15,.2f}  (PV reinsurance premium - PV recoveries; net cost)",
        f"RA  = {ra:>15,.2f}  (risk transferred to the reinsurer, para 64)",
        f"CSM = {csm0:>15,.2f}  (net cost / gain; may be negative, para 65, no loss component)",
    ]

    out.append(header)
    tree_items: list[object] = [
        ("Treaty / inputs", treaty_lines),
        ("Ceded cash flows (key months)", flow_lines),
        ("Discount factors (key months)", disc_lines),
        ("BEL / RA / CSM build", build_lines),
        ("CSM roll-forward (key months)", csm_lines),
        ("Final (headline numbers, per policy)", final_lines),
    ]
    _emit_tree(tree_items, out, "")
    file.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# show_trace_diff -- two-basis comparison
# ---------------------------------------------------------------------------

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


def show_trace_diff(
    mp_index: int,
    model_points: ModelPoints,
    basis_a: Basis | dict,
    basis_b: Basis | dict,
    *,
    label_a: str = "before",
    label_b: str = "after",
    file: IO | None = None,
) -> None:
    """Print a tree of how the BEL / RA / CSM of one model point moves
    when basis change.

    Parameters
    ----------
    mp_index
        0-based row index in ``model_points``.
    model_points
        Portfolio :class:`ModelPoints`. Subset to the single row before
        each :func:`measure` so the diff cost stays proportional to one MP.
    basis_a, basis_b
        Two basis to compare. Either a :class:`Basis` or the
        :class:`~fastcashflow.BasisRouter` from :func:`fastcashflow.io.read_basis`.
        With a router, each is routed independently by the model point's
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

    resolved_a = _resolve_basis(basis_a, model_points, i)
    resolved_b = _resolve_basis(basis_b, model_points, i)

    sub = model_points.subset([i])
    ma = measure(sub, resolved_a)
    mb = measure(sub, resolved_b)

    # ---- Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "M" if sex_v == 0 else "F"
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
        f"diff mp[{i}]  ({product}/{channel}, sex={sex_label}, "
        f"issue_age={age:g}, term={term}m, premium_term={prem_term}m, "
        f"count={count:g})"
    )
    labels_line = f"labels: {label_a!r}  ->  {label_b!r}"

    # ---- Assumption changes (suppressed when equal)
    basis_diffs = _basis_diff_lines(resolved_a, resolved_b)
    # codes used again below to gate the per-coverage rate-delta rows
    codes_a = [r.code for r in resolved_a.coverages]
    codes_b = [r.code for r in resolved_b.coverages]

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
            _eval_rate(resolved_a.mortality_annual, sex_v, age, y,
                       issue_class_v, elapsed_v),
            _eval_rate(resolved_b.mortality_annual, sex_v, age, y,
                       issue_class_v, elapsed_v),
        )
        if row is not None:
            block.append(row)
        row = _maybe_row(
            "lapse(annual)    ",
            _eval_rate(resolved_a.lapse_annual, sex_v, age, y,
                       issue_class_v, elapsed_v),
            _eval_rate(resolved_b.lapse_annual, sex_v, age, y,
                       issue_class_v, elapsed_v),
        )
        if row is not None:
            block.append(row)
        if (resolved_a.waiver_incidence_annual is not None
                or resolved_b.waiver_incidence_annual is not None):
            row = _maybe_row(
                "waiver(annual)   ",
                _eval_rate(resolved_a.waiver_incidence_annual, sex_v, age, y,
                           issue_class_v, elapsed_v),
                _eval_rate(resolved_b.waiver_incidence_annual, sex_v, age, y,
                           issue_class_v, elapsed_v),
            )
            if row is not None:
                block.append(row)
        if codes_a == codes_b:
            for ra, rb in zip(resolved_a.coverages, resolved_b.coverages):
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
               "diff", "%diff"]
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
    picks = _key_months(term, ma.discount_bom.shape[0] - 1)
    disc_lines: list[object] = [
        f"t={t:>4d}m: ds  {ma.discount_bom[t]:.6f}  ->  "
        f"{mb.discount_bom[t]:.6f}  ({mb.discount_bom[t] - ma.discount_bom[t]:+.6f})"
        for t in picks
    ]

    # ---- BEL / CSM deltas at anchor months
    bel_a, bel_b = ma.bel_path[0], mb.bel_path[0]
    csm_a, csm_b = ma.csm_path[0], mb.csm_path[0]
    bel_lines: list[object] = [
        f"BEL[{t:>4d}]   {_money_delta(float(bel_a[t]), float(bel_b[t]))}"
        for t in picks
    ]
    csm_lines: list[object] = [
        f"CSM[{t:>4d}]   {_money_delta(float(csm_a[t]), float(csm_b[t]))}"
        for t in picks
    ]

    # ---- Final headline
    ra_a, ra_b = float(ma.ra_path[0, 0]), float(mb.ra_path[0, 0])
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
        ("Assumption changes", basis_diffs),
        ("Rate deltas (per policy year)", rate_lines),
        (f"Cash flow deltas (annual sum, non-zero rows only)", cf_lines),
        ("Discount factor deltas (key months)", disc_lines),
        ("BEL deltas (key months)", bel_lines),
        ("CSM deltas (key months)", csm_lines),
        ("Final (headline change, per policy)", final_lines),
    ]
    _emit_tree(tree_items, out, "")
    file.write("\n".join(out) + "\n")


def show_trace_diff_vfa(
    mp_index: int,
    model_points: ModelPoints,
    basis_a: Basis | dict,
    basis_b: Basis | dict,
    *,
    return_scenarios: FloatArray | None = None,
    label_a: str = "before",
    label_b: str = "after",
    file: IO | None = None,
) -> None:
    """Diff one VFA model point's headline (BEL / RA / fee / CSM / TVOG / loss)
    across two bases, with the assumption changes that drive it.

    The VFA counterpart of :func:`show_trace_diff` -- a headline-level diff
    (assumption changes plus the metric deltas), not the per-month path deltas
    the GMM version also prints. ``return_scenarios`` (if given) is applied to
    both measures so the time-value (TVOG) delta is meaningful.
    """
    if file is None:
        file = sys.stdout
    if not 0 <= mp_index < model_points.n_mp:
        raise IndexError(
            f"mp_index {mp_index} out of range for n_mp={model_points.n_mp}")
    i = mp_index
    ra_basis = _resolve_basis(basis_a, model_points, i)
    rb_basis = _resolve_basis(basis_b, model_points, i)
    sub = model_points.subset([i])
    ma = measure_vfa(sub, ra_basis, return_scenarios=return_scenarios)
    mb = measure_vfa(sub, rb_basis, return_scenarios=return_scenarios)

    def g(m, name):
        return float(getattr(m, name)[0])
    final_lines: list[object] = [
        f"BEL  {_money_delta(g(ma,'bel'), g(mb,'bel'))}",
        f"RA   {_money_delta(g(ma,'ra'), g(mb,'ra'))}",
        f"fee  {_money_delta(g(ma,'variable_fee'), g(mb,'variable_fee'))}",
        f"CSM  {_money_delta(g(ma,'csm'), g(mb,'csm'))}",
        f"TVOG {_money_delta(g(ma,'time_value'), g(mb,'time_value'))}",
        f"loss {_money_delta(g(ma,'loss_component'), g(mb,'loss_component'))}",
    ]
    out = [_diff_mp_header(model_points, sub, i, "-vfa"),
           f"labels: {label_a!r}  ->  {label_b!r}"]
    _emit_tree([("Assumption changes", _basis_diff_lines(ra_basis, rb_basis)),
                ("Final (headline change, per policy)", final_lines)], out, "")
    file.write("\n".join(out) + "\n")


def show_trace_diff_paa(
    mp_index: int,
    model_points: ModelPoints,
    basis_a: Basis | dict,
    basis_b: Basis | dict,
    *,
    revenue_basis: str = "time",
    label_a: str = "before",
    label_b: str = "after",
    file: IO | None = None,
) -> None:
    """Diff one PAA model point's headline (LRC / revenue / service result /
    LIC / loss) across two bases, with the assumption changes that drive it.

    The PAA counterpart of :func:`show_trace_diff` -- a headline-level diff. PAA
    has no CSM; the metrics are the unearned-premium LRC, the recognised
    revenue, the insurance service result, the incurred-claims liability peak
    and the loss component.
    """
    if file is None:
        file = sys.stdout
    if not 0 <= mp_index < model_points.n_mp:
        raise IndexError(
            f"mp_index {mp_index} out of range for n_mp={model_points.n_mp}")
    i = mp_index
    ra_basis = _resolve_basis(basis_a, model_points, i)
    rb_basis = _resolve_basis(basis_b, model_points, i)
    sub = model_points.subset([i])
    ma = measure_paa(sub, ra_basis, revenue_basis=revenue_basis)
    mb = measure_paa(sub, rb_basis, revenue_basis=revenue_basis)

    final_lines: list[object] = [
        f"LRC[0]          {_money_delta(float(ma.lrc[0]), float(mb.lrc[0]))}",
        f"total revenue   {_money_delta(float(ma.revenue[0].sum()), float(mb.revenue[0].sum()))}",
        f"svc result      {_money_delta(float(ma.service_result[0].sum()), float(mb.service_result[0].sum()))}",
        f"LIC (peak)      {_money_delta(float(ma.lic[0].max()), float(mb.lic[0].max()))}",
        f"loss_component  {_money_delta(float(ma.loss_component[0]), float(mb.loss_component[0]))}",
    ]
    out = [_diff_mp_header(model_points, sub, i, "-paa"),
           f"labels: {label_a!r}  ->  {label_b!r}"]
    _emit_tree([("Assumption changes", _basis_diff_lines(ra_basis, rb_basis)),
                ("Final (headline change, per policy)", final_lines)], out, "")
    file.write("\n".join(out) + "\n")


def show_trace_diff_reinsurance(
    mp_index: int,
    model_points: ModelPoints,
    basis_a: Basis | dict,
    basis_b: Basis | dict,
    *,
    treaty: Treaty,
    label_a: str = "before",
    label_b: str = "after",
    file: IO | None = None,
) -> None:
    """Diff one reinsurance-held model point's headline (BEL / RA / CSM) across
    two bases (same ``treaty``), with the assumption changes that drive it.

    The reinsurance counterpart of :func:`show_trace_diff` -- a headline-level
    diff. The CSM is the net cost / gain of the cover and may be negative; there
    is no loss component (Sec. 65).
    """
    if file is None:
        file = sys.stdout
    if not 0 <= mp_index < model_points.n_mp:
        raise IndexError(
            f"mp_index {mp_index} out of range for n_mp={model_points.n_mp}")
    i = mp_index
    ra_basis = _resolve_basis(basis_a, model_points, i)
    rb_basis = _resolve_basis(basis_b, model_points, i)
    sub = model_points.subset([i])
    ma = measure_reinsurance(sub, ra_basis, treaty=treaty)
    mb = measure_reinsurance(sub, rb_basis, treaty=treaty)

    final_lines: list[object] = [
        f"BEL  {_money_delta(float(ma.bel[0]), float(mb.bel[0]))}",
        f"RA   {_money_delta(float(ma.ra[0]), float(mb.ra[0]))}",
        f"CSM  {_money_delta(float(ma.csm[0]), float(mb.csm[0]))}",
    ]
    out = [_diff_mp_header(model_points, sub, i, "-reinsurance"),
           f"labels: {label_a!r}  ->  {label_b!r}"]
    _emit_tree([("Assumption changes", _basis_diff_lines(ra_basis, rb_basis)),
                ("Final (headline change, per policy)", final_lines)], out, "")
    file.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# show_trace_bel_step -- term-by-term unrolling of the BEL backward recursion
# ---------------------------------------------------------------------------

def show_trace_bel_step(
    mp_index: int,
    model_points: ModelPoints,
    basis: Basis | dict,
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
    basis
        A :class:`Basis` or the :class:`~fastcashflow.BasisRouter` from
        :func:`fastcashflow.io.read_basis` (routed by the row's
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
    basis = _resolve_basis(basis, model_points, i)
    sub = model_points.subset([i])
    m = measure(sub, basis)

    term = int(sub.term_months[0])
    n_time = m.cashflows.n_time
    if months is None:
        months = sorted({0, 12, term // 2, term - 1, term})
    # A whole-month anchor only -- int(t) would silently truncate 12.5 to 12.
    bad = [t for t in months if float(t) != int(t)]
    if bad:
        raise ValueError(f"months must be whole-month integers; got {bad}")
    months = [int(t) for t in months if 0 <= int(t) <= term]

    # Monthly discount rate curve -- the kernel reads this directly. We
    # recover ``i[t]`` from the same curve here so the printed identity
    # uses the engine's numbers, not a parallel computation that could
    # drift.
    monthly_rate = discount_monthly_curve(basis, n_time)

    # Header
    sex_v = int(sub.sex[0]) if sub.sex is not None else 0
    sex_label = "M" if sex_v == 0 else "F"
    age = float(sub.issue_age[0])
    product = (str(model_points.product[i])
               if model_points.product is not None else "-")
    channel = (str(model_points.channel[i])
               if model_points.channel is not None else "-")
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
    bel_engine = m.bel_path[0]
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
            f"mid-month piece (*half)   = {mid_piece:>15,.2f}",
            f"BEL[t+1]                  = {float(bel_engine[t + 1]):>15,.2f}",
            f"tail piece (BEL[t+1]*full)= {tail_piece:>15,.2f}",
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
# show_trace_csm_step -- term-by-term unrolling of the CSM forward recursion
# ---------------------------------------------------------------------------

def show_trace_csm_step(
    mp_index: int,
    model_points: ModelPoints,
    basis: Basis | dict,
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
    model_points, basis, file
        Same shape as :func:`show_trace_bel_step`.
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
    basis = _resolve_basis(basis, model_points, i)
    sub = model_points.subset([i])
    m = measure(sub, basis)

    term = int(sub.term_months[0])
    n_time = m.cashflows.n_time
    if months is None:
        months = sorted({1, 12, term // 2, term})
    # Recursion produces csm[t] for t in 1..n_time. t = 0 is the seed.
    bad = [t for t in months if float(t) != int(t)]
    if bad:
        raise ValueError(f"months must be whole-month integers; got {bad}")
    months = [int(t) for t in months if 1 <= int(t) <= n_time]

    monthly_rate = discount_monthly_curve(basis, n_time)
    inforce = m.cashflows.inforce[0]              # (n_time,)
    bel0 = float(m.bel_path[0, 0])
    ra0 = float(m.ra_path[0, 0])
    fcf0 = bel0 + ra0
    csm = m.csm_path[0]
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
    sex_label = "M" if sex_v == 0 else "F"
    age = float(sub.issue_age[0])
    product = (str(model_points.product[i])
               if model_points.product is not None else "-")
    channel = (str(model_points.channel[i])
               if model_points.channel is not None else "-")
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
