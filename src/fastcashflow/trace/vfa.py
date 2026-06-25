"""Step-by-step calculation trace for one VFA (variable-fee) contract.

:func:`show_trace_vfa` renders the account-value trajectory, the GMDB / GMAB
floor bites, the variable fee and the BEL / RA / CSM build as an ASCII tree;
:func:`show_trace_diff_vfa` is its two-basis assumption-change variant. The
GMM tracer (:func:`fastcashflow.trace.gmm.show_trace`) measures GMM contracts,
so a variable contract uses this module instead.
"""
from __future__ import annotations

import sys
from typing import IO

import numpy as np

from fastcashflow.basis import Basis
from fastcashflow._typing import FloatArray
from fastcashflow.model_points import ModelPoints, NO_GUARANTEE_RATE
from fastcashflow._measurement import vfa as _vfa
from fastcashflow.trace._common import (
    _emit_tree, _fmt_callable, _key_months, _colw, _resolve_basis,
    _money_delta, _basis_diff_lines, _diff_mp_header,
)


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
    It slices to a single row, runs :func:`_vfa.measure`, and shows the
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
    m = _vfa.measure(sub, basis, return_scenarios=return_scenarios)

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

    # ---- Universal-life annuitization (conversion + payout) -- only when the
    # contract carries a conversion month. The account roll stops at A and the
    # balance converts to a survival income: the balance carried into month A,
    # floored at the GMAB, times the locked annuitization rate (the initial
    # payment). A variable payout then re-floats that level each elapsed month by
    # (1+fund)/(1+air) (the annuity-unit value); a fixed payout keeps it level.
    # No maturity lump is paid (the balance was already converted), so the
    # maturity row in the floors section above is moot for an annuitizing book.
    annz_lines: list[object] = []
    A_annz = (int(sub.annuitization_months[0])
              if sub.annuitization_months is not None else 0)
    if A_annz > 0 and A_annz < av.shape[0]:
        annuity = cf.annuity_cf[0]
        annz_rate = float(sub.annuitization_rate[0])
        gmab_acc = (float(sub.minimum_accumulation_benefit[0])
                    if sub.minimum_accumulation_benefit is not None else 0.0)
        bal_in = float(av[A_annz])            # balance carried into month A
        converted = bal_in if bal_in > gmab_acc else gmab_acc
        locked = converted * annz_rate
        air = (float(sub.annuity_air_annual[0])
               if sub.annuity_air_annual is not None else float("nan"))
        variable = bool(np.isfinite(air))
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
            f"annuitization_rate    = {annz_rate:>{_zw}g}  (initial income rate)")
        annz_lines.append(
            f"locked_annuity_payment= {locked:>{_zw},.2f}  "
            "(= converted_balance x rate; the initial payment)")
        if variable:
            annz_lines.append(
                f"annuity_air_annual    = {air:>{_zw}g}  (AIR; variable payout)")
            annz_lines.append(
                "phase 2: VARIABLE payout -- re-floats by ((1+fund)/(1+air))^k, "
                "k = t-A; annuity-due on surviving in-force, no maturity lump")
        else:
            annz_lines.append(
                "phase 2: FIXED payout -- annuity-due on surviving in-force; no "
                "premium / COI / surrender, no maturity lump")
        # Phase-2 payments at the payout key months (annuity_cf is the in-force-
        # weighted paid amount). The conversion month and a payout midpoint are
        # added so a short pick list still shows the income starting / moving.
        p_picks = sorted(
            {t for t in picks if A_annz <= t < n_time}
            | {A_annz, (A_annz + n_time) // 2})
        p_picks = [t for t in p_picks if A_annz <= t < n_time]
        if p_picks:
            _pw = _colw((annuity[t] for t in p_picks), ",.2f", 15)
            for t in p_picks:
                annz_lines.append(
                    f"t={t:>4d}m: annuity={annuity[t]:>{_pw},.2f}  "
                    f"inforce={inforce[t]:.6f}")

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
    ]
    if annz_lines:
        tree_items.append(
            ("Universal-life annuitization (conversion + payout)", annz_lines))
    tree_items += [
        ("BEL / CSM trajectory (key months)", belcsm_lines),
        ("CSM roll-forward (key months)", csm_lines),
        ("Final (headline numbers, per policy)", final_lines),
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
    ma = _vfa.measure(sub, ra_basis, return_scenarios=return_scenarios)
    mb = _vfa.measure(sub, rb_basis, return_scenarios=return_scenarios)

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
