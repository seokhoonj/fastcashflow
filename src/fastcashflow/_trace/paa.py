"""Step-by-step calculation trace for one PAA (short-duration) contract.

:func:`trace` renders the LRC roll-forward (premium received,
revenue recognised), the insurance service result and the LIC as an ASCII
tree; :func:`trace_diff` is its two-basis assumption-change variant.
The PAA carries no CSM, so the tree shows the LRC / revenue movement rather
than the BEL / CSM build.
"""
from __future__ import annotations

import sys
from typing import IO

import numpy as np

from fastcashflow.basis import Basis
from fastcashflow.model_points import ModelPoints
from fastcashflow._measurement import paa as _paa
from fastcashflow._trace.common import (
    _emit_tree, _fmt_callable, _key_months, _colw, _resolve_basis,
    _money_delta, _basis_diff_lines, _diff_mp_header,
)


def trace(
    mp_index: int,
    model_points: ModelPoints,
    basis: Basis | dict,
    *,
    revenue_basis: str = "time",
    file: IO | None = None,
) -> None:
    """Print a tree of how one PAA model point's LRC / revenue / LIC is built.

    The PAA (Premium Allocation Approach, the short-duration simplification)
    counterpart of :func:`trace`. PAA has no CSM -- the liability for
    remaining coverage (LRC) is an unearned-premium-style balance -- so the
    tree shows the LRC roll-forward (premium in, revenue released), the
    insurance service result (revenue less service expense) and the
    liability for incurred claims (LIC). Use it on PAA contracts;
    :func:`trace` traces the GMM ``measure`` (BEL / RA / CSM).
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
    m = _paa.measure(sub, basis, revenue_basis=revenue_basis)

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
    lic_path = m.lic_path[0]
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
        f"t={t:>4d}m: LIC={lic_path[t]:>15,.2f}" for t in picks if t < lic_path.shape[0]
    ]

    # ---- Final headline
    final_lines: list[object] = [
        f"LRC[0]                = {lrc[0]:>15,.2f}  (= 0, before premium inflow)",
        f"total revenue         = {float(revenue.sum()):>15,.2f}  (= total premium)",
        f"total service_expense = {float(svc_exp.sum()):>15,.2f}",
        f"insurance svc result  = {float(svc_result.sum()):>15,.2f}",
        f"loss_component        = {lc:>15,.2f}  (onerous; from the GMM FCF)",
        f"LIC (peak)            = {float(lic_path.max()):>15,.2f}",
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


def trace_diff(
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

    The PAA counterpart of :func:`trace_diff` -- a headline-level diff. PAA
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
    ma = _paa.measure(sub, ra_basis, revenue_basis=revenue_basis)
    mb = _paa.measure(sub, rb_basis, revenue_basis=revenue_basis)

    final_lines: list[object] = [
        f"LRC[0]          {_money_delta(float(ma.lrc[0]), float(mb.lrc[0]))}",
        f"total revenue   {_money_delta(float(ma.revenue[0].sum()), float(mb.revenue[0].sum()))}",
        f"svc result      {_money_delta(float(ma.service_result[0].sum()), float(mb.service_result[0].sum()))}",
        f"LIC (peak)      {_money_delta(float(ma.lic_path[0].max()), float(mb.lic_path[0].max()))}",
        f"loss_component  {_money_delta(float(ma.loss_component[0]), float(mb.loss_component[0]))}",
    ]
    out = [_diff_mp_header(model_points, sub, i, "-paa"),
           f"labels: {label_a!r}  ->  {label_b!r}"]
    _emit_tree([("Assumption changes", _basis_diff_lines(ra_basis, rb_basis)),
                ("Final (headline change, per policy)", final_lines)], out, "")
    file.write("\n".join(out) + "\n")
