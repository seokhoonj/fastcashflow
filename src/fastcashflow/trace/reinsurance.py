"""Step-by-step calculation trace for one held reinsurance contract.

:func:`show_trace_reinsurance` renders the ceded cash flows (recovery,
reinsurance premium), the BEL = PV(premium) - PV(recovery), the
risk-transfer RA and the CSM build (net cost / gain, no loss component) as an
ASCII tree; :func:`show_trace_diff_reinsurance` is its two-basis variant.
"""
from __future__ import annotations

import sys
from typing import IO

from fastcashflow.basis import Basis
from fastcashflow.curves import discount_factors
from fastcashflow.model_points import ModelPoints
from fastcashflow.numerics import _norm_ppf
from fastcashflow._measurement.reinsurance import QuotaShare, Treaty
from fastcashflow._measurement import reinsurance as _reinsurance
from fastcashflow.trace._common import (
    _emit_tree, _fmt_callable, _key_months, _colw, _resolve_basis,
    _money_delta, _basis_diff_lines, _diff_mp_header,
)


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
    paragraph 64), and the CSM = -(BEL - RA) -- the net cost or gain of the cover,
    which may be negative and carries no loss component (paragraph 65). Use it on a
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
    m = _reinsurance.measure(sub, basis, treaty=treaty)

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
    discount_factor_bom, discount_factor_mid = discount_factors(basis, n_time)
    pv_recovery = float((recovery * discount_factor_mid).sum())
    pv_reins_prem = float((reins_prem * discount_factor_bom[:-1]).sum())
    pv_ceded_mort = float((ceded_mort * discount_factor_mid).sum())
    pv_ceded_morb = float((ceded_morb * discount_factor_mid).sum())
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
        "recovery           = cession x (mortality_cf + morbidity_cf)  (ceded share of direct claims)",
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
        f"t={t:>4d}m: bom={discount_factor_bom[t]:.6f}"
        + (f"  mid={discount_factor_mid[t]:.6f}" if t < n_time else "")
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
        f"CSM[0] = -(BEL - RA) = {csm0:>15,.2f}  (negative when a net cost; paragraph 65)",
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
        f"RA  = {ra:>15,.2f}  (risk transferred to the reinsurer, paragraph 64)",
        f"CSM = {csm0:>15,.2f}  (net cost / gain; may be negative, paragraph 65, no loss component)",
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
    is no loss component (paragraph 65).
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
    ma = _reinsurance.measure(sub, ra_basis, treaty=treaty)
    mb = _reinsurance.measure(sub, rb_basis, treaty=treaty)

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
