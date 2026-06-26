"""Step-by-step calculation trace for a single GMM model point.

:func:`trace` renders the BEL / RA / CSM build of one contract as an
ASCII tree: which segment-level tables apply, which rates were looked up
year by year, what cash flows came out, how those discount and roll
forward to the headline numbers. Intended for hand-checking against an
external pricing system or an actuary's own spreadsheet -- find the step
where the engine and the expectation diverge.

:func:`trace_diff` is the two-basis variant: it shows, at each step,
how a change of assumption propagates -- which rate moved, by how much,
which cash flow shifted, and what the net effect on BEL / RA / CSM is.
The right tool for shock analysis and what-if questions.

Both functions make no new calculations; they slice the result of
:func:`fastcashflow._measurement.gmm.measure` and print it.
"""
from __future__ import annotations

import sys
from typing import IO

import numpy as np

from fastcashflow.basis import Basis, BasisRouter
from fastcashflow.coverage import CalculationMethod, method_attrs
from fastcashflow.curves import discount_monthly_curve
from fastcashflow.model_points import ModelPoints
from fastcashflow._measurement.gmm import measure
from fastcashflow._trace.common import (
    _emit_tree, _fmt_callable, _eval_rate, _key_months, _colw,
    _resolve_basis, _money_delta, _rate_delta, _basis_diff_lines,
)


def trace(
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
            (f"ExpenseItem({r.category!r}, {r.base!r}, "
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
    discount_factor_bom = m.discount_factor_bom

    cf_lines: list[object] = []
    cf_names = ["premium_cf", "mortality_cf", "morbidity_cf", "expense_cf",
                "annuity_cf", "surrender_cf", "disability_cf"]
    cf_heads = ["premium", "mortality", "morbidity", "expense",
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
    # month as ``undiagnosed *= (1 - discount_monthly)``. Storing the trajectory
    # here makes the depleting-pool mechanism visible alongside the in-force
    # trajectory it composes with.
    picks = _key_months(term, discount_factor_bom.shape[0] - 1)
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
        f"t={t:>4d}m: ds={discount_factor_bom[t]:.6f}" for t in picks
    ]

    # ---- Universal-life account mechanic (only when the contract is an
    # account book -- cf.account is the AccountTrajectory sidecar the shared
    # projection populates). Gated on account is not None so a non-account
    # contract's trace stays byte-identical. This mirrors trace_vfa's
    # account section, but for the GMM-measured UL: the account value carried
    # forward, the COI charged, the net amount at risk the COI prices, the
    # in-force-weighted fund the BEL nets, and that the death benefit tops the
    # account up to the face (max(av_mid, face)).
    account_lines: list[object] = []
    acct = cf.account
    # Conversion month for an annuitizing UL (0 = ordinary account, never
    # converts). Always an array post-construction, so the index is safe.
    A_annz = (int(sub.annuitization_months[0])
              if sub.annuitization_months is not None else 0)
    if acct is not None:
        av = acct.av[0]            # (n_time+1,) month-start AV (col 0 = av0)
        av_mid = acct.av_mid[0]    # (n_time,)   half-month-credited AV
        coi = acct.coi[0]          # (n_time,)   cost-of-insurance charged
        fund = acct.fund[0]        # (n_time+1,) in-force-weighted AV held
        face = (float(sub.minimum_death_benefit[0])
                if sub.minimum_death_benefit is not None else 0.0)
        n_t = cf.n_time
        load = float(basis.premium_load)
        inv = float(basis.investment_return)
        account_lines.append(f"account_value0 (av0)  = {av[0]:>15,.2f}")
        account_lines.append(f"face (min_death_ben)  = {face:>15,.2f}")
        account_lines.append(
            f"premium_load          = {load:>15g}  "
            "(prem_to_av = premium * (1 - load))"
        )
        account_lines.append(
            f"investment_return     = {inv:>15g}  (account crediting basis)"
        )
        account_lines.append(
            "coi_annual            -> " + _fmt_callable(basis.coi_annual)
        )
        account_lines.append(
            "death = max(av_mid, face);  NAR = max(0, face - av_mid);  "
            "COI = coi_m * NAR"
        )
        # Annuitizing UL: the account roll stops at the conversion month, so the
        # av rows below read 0 past it -- flag that here rather than leave the
        # reader puzzling over the zeros (the conversion is detailed in its own
        # section).
        if A_annz > 0:
            account_lines.append(
                f"annuitizes at t={A_annz}m -- account is spent at conversion; "
                f"av is 0 after (see the annuitization section)"
            )
        # Per-month account rows at the key months. av / fund carry the extra
        # month-end column (index up to n_time); av_mid / coi are (n_time,) so
        # they are shown only where t < n_time.
        a_picks = [t for t in picks if t <= n_t]
        _aw = _colw((av[t] for t in a_picks if t < av.shape[0]), ",.2f", 15)
        _mw = _colw((av_mid[t] for t in a_picks if t < n_t), ",.2f", 15)
        _cw2 = _colw((coi[t] for t in a_picks if t < n_t), ",.2f", 12)
        _nw = _colw(
            (max(0.0, face - av_mid[t]) for t in a_picks if t < n_t), ",.2f", 15)
        _fw2 = _colw((fund[t] for t in a_picks if t < fund.shape[0]), ",.2f", 15)
        for t in a_picks:
            if t < n_t:
                nar = max(0.0, face - av_mid[t])
                death = max(av_mid[t], face)
                account_lines.append(
                    f"t={t:>4d}m: av={av[t]:>{_aw},.2f}  "
                    f"av_mid={av_mid[t]:>{_mw},.2f}  coi={coi[t]:>{_cw2},.2f}  "
                    f"nar={nar:>{_nw},.2f}  death={death:>{_aw},.2f}  "
                    f"fund={fund[t]:>{_fw2},.2f}"
                )
            else:
                # Month-end boundary column: av / fund have it, av_mid / coi
                # do not (no within-month roll past the contract boundary).
                account_lines.append(
                    f"t={t:>4d}m: av={av[t]:>{_aw},.2f}  "
                    f"{'(boundary)':>{_mw}}  {'':>{_cw2}}  {'':>{_nw}}  "
                    f"{'':>{_aw}}  fund={fund[t]:>{_fw2},.2f}"
                )

    # ---- Universal-life annuitization (conversion + payout) -- only when the
    # contract carries a conversion month. The account roll above stops at A;
    # here the balance is converted to a guaranteed survival income: the balance
    # carried into month A, floored at the GMAB, times the locked GAO rate. The
    # payout (phase 2) pays annuity-due on the surviving in-force, with no
    # further premium / COI / surrender and no maturity lump.
    annz_lines: list[object] = []
    if acct is not None and A_annz > 0:
        av = acct.av[0]
        annuity = cf.annuity_cf[0]
        inforce = cf.inforce[0]
        annz_rate = float(sub.annuitization_rate[0])
        gmab_acc = (float(sub.minimum_accumulation_benefit[0])
                    if sub.minimum_accumulation_benefit is not None else 0.0)
        bal_in = float(av[A_annz])            # balance carried into month A
        converted = bal_in if bal_in > gmab_acc else gmab_acc
        locked = converted * annz_rate
        _zw = _colw([bal_in, gmab_acc, converted, locked], ",.2f", 15)
        annz_lines.append(
            f"annuitization_months  = {A_annz:>{_zw}d}  "
            "(account stops, converts to income)")
        annz_lines.append(
            f"balance at conversion = {bal_in:>{_zw},.2f}  "
            f"(av[{A_annz}], no month-{A_annz} credit)")
        annz_lines.append(
            f"GMAB floor            = {gmab_acc:>{_zw},.2f}  "
            "(minimum_accumulation_benefit)")
        annz_lines.append(
            f"converted_balance     = {converted:>{_zw},.2f}  (= max(balance, GMAB))")
        annz_lines.append(
            f"annuitization_rate    = {annz_rate:>{_zw}g}  (GAO rate, locked once)")
        annz_lines.append(
            f"locked_annuity_payment= {locked:>{_zw},.2f}  "
            "(= converted_balance x rate)")
        annz_lines.append(
            "phase 2: annuity-due on surviving in-force; no premium / COI / "
            "surrender, no maturity lump")
        # Phase-2 annuity payments at the payout key months (annuity_cf is the
        # paid amount; nonzero only on a payment month). The conversion month and
        # a payout midpoint are added so a short pick list still shows the income
        # starting and decrementing with survival.
        p_picks = sorted(
            {t for t in picks if A_annz <= t < cf.n_time}
            | {A_annz, (A_annz + cf.n_time) // 2})
        p_picks = [t for t in p_picks if A_annz <= t < cf.n_time]
        if p_picks:
            _pw = _colw((annuity[t] for t in p_picks), ",.2f", 15)
            for t in p_picks:
                annz_lines.append(
                    f"t={t:>4d}m: annuity={annuity[t]:>{_pw},.2f}  "
                    f"inforce={inforce[t]:.6f}")

    # ---- BEL roll-forward at key months
    bel_lines: list[object] = [
        "BEL[t] = annuity[t] - premium[t] + (mortality+morbidity+disability+"
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
    if account_lines:
        tree_items.append(
            ("Universal-life account (key months)", account_lines)
        )
    if annz_lines:
        tree_items.append(
            ("Universal-life annuitization (conversion + payout)", annz_lines)
        )
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


def trace_diff(
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
        for name in ("premium_cf", "mortality_cf", "morbidity_cf",
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
    picks = _key_months(term, ma.discount_factor_bom.shape[0] - 1)
    disc_lines: list[object] = [
        f"t={t:>4d}m: ds  {ma.discount_factor_bom[t]:.6f}  ->  "
        f"{mb.discount_factor_bom[t]:.6f}  ({mb.discount_factor_bom[t] - ma.discount_factor_bom[t]:+.6f})"
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


def trace_bel_step(
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
               + (mortality + morbidity + disability + expense + surrender)[t]
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
        ``(product, channel)`` like :func:`trace`).
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
    discount_monthly = discount_monthly_curve(basis, n_time)

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
        "       + (mortality + morbidity + disability + expense + surrender)[t]"
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
        rate = float(discount_monthly[t])
        half = (1.0 + rate) ** (-0.5)
        full = 1.0 / (1.0 + rate)
        prem = float(cf.premium_cf[0, t])
        ann = float(cf.annuity_cf[0, t])
        claim = float(cf.mortality_cf[0, t])
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
            f"mortality[t]              = {claim:>15,.2f}",
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
# trace_csm_step -- term-by-term unrolling of the CSM forward recursion
# ---------------------------------------------------------------------------


def trace_csm_step(
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
        Same shape as :func:`trace_bel_step`.
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

    discount_monthly = discount_monthly_curve(basis, n_time)
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
        rate = float(discount_monthly[t - 1])
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
