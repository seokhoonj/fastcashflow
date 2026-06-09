"""IFRS 17 Variable Fee Approach (VFA) -- direct-participation contracts.

The VFA is IFRS 17's measurement model for insurance contracts with direct
participation features -- contracts where the policyholder's benefit is a
share of a pool of *underlying items* (a fund). It is the model for
unit-linked and with-profits business.

This module measures a single-premium account-value contract: a premium is
paid into an account at issue; the account value grows at the underlying-
items return less a variable fee; the benefit is the account value on
surrender, max(account value, a guaranteed minimum death benefit) on death,
and max(account value, a guaranteed minimum accumulation benefit) at
maturity. The entity's profit is the *variable fee* it deducts, its share
of the underlying items.

Under the VFA the financial result flows through the CSM rather than profit
or loss, so the account-value cash flows are discounted, and the CSM is
accreted, at the underlying-items return -- not a locked-in rate.
fastcashflow is deterministic (a single scenario), so the VFA's hallmark --
the CSM absorbing the variability of the underlying items -- reduces here to
that return-rate accretion. A minimum guaranteed credited rate is supported
-- the account is credited ``max(return, guarantee)`` each period. Its
intrinsic cost appears in this deterministic measurement; the time value of
the guarantee, the extra cost from return volatility, is measured
stochastically by ``measure_tvog``. Surrender penalties and a lapse-risk
adjustment are left for later; the risk adjustment here covers expense risk
-- the main non-financial risk an account-value contract carries, with the
policyholder bearing the investment risk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.basis import Basis, _single_basis
from fastcashflow.io import write_measurement, _write_measurement_columns
from fastcashflow.numerics import (
    _csm_kernel,
    _norm_ppf,
    _settlement_factor,
    _settlement_lic,
)
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows
from fastcashflow.statemodel import resolve_state_model
from fastcashflow.tvog import (
    guarantee_floor_time_value, tvog_weights, _validate_return_scenarios,
)
# In-force helpers shared with the GMM path (engine does not import _vfa, and io
# imports engine lazily, so this top-level import is cycle-free).
from fastcashflow.engine import _reconcile_state, _inforce_rescale


# What the ``csm`` on a VFAMeasurement represents -- so downstream accounting
# output cannot mistake an in-force carry-only figure for a settlement CSM. The
# discriminator is carried on the result (not just the docstring) and the
# accounting-output entry points (roll_forward / report / group / serialise)
# guard on it via _require_settlement_csm.
CSM_BASIS_INITIAL = "initial_measurement"           # inception headline (csm0)
CSM_BASIS_PROJECTED_RUNOFF = "projected_runoff"     # inception full trajectory
CSM_BASIS_CARRY_ONLY = "carry_only"                 # measure_inforce: prior CSM
#                                                     rolled at the basis return,
#                                                     paragraph-45 unlock deferred
CSM_BASIS_PARAGRAPH_45 = "paragraph_45_settlement"  # future: real subsequent meas.
CSM_BASES = (CSM_BASIS_INITIAL, CSM_BASIS_PROJECTED_RUNOFF,
             CSM_BASIS_CARRY_ONLY, CSM_BASIS_PARAGRAPH_45)


def _require_settlement_csm(measurement: "VFAMeasurement", op: str) -> None:
    """Reject a carry-only in-force CSM where a settlement CSM is required.

    ``vfa.measure_inforce`` returns a carry-only CSM (the prior CSM rolled at the
    basis return, with the IFRS 17 paragraph-45 remeasurement deferred) -- a
    diagnostic headline, not an accounting settlement figure. Accounting-output
    functions call this guard so that figure cannot be silently rolled forward,
    reported, grouped or serialised as if it were paragraph-45 compliant.
    """
    if measurement.csm_basis == CSM_BASIS_CARRY_ONLY:
        raise ValueError(
            f"{op}: this VFAMeasurement carries a carry-only in-force CSM "
            f"(csm_basis='{CSM_BASIS_CARRY_ONLY}', from vfa.measure_inforce -- "
            "the prior CSM rolled at the basis return, with the paragraph-45 "
            "remeasurement deferred). It is not a settlement figure, so it "
            f"cannot be used by {op}. The real paragraph-45 VFA subsequent "
            "measurement (opening->closing movement) is a separate, later step.")


@dataclass(frozen=True, slots=True, eq=False)
class VFAMeasurement:
    """VFA measurement of a direct-participation (account-value) portfolio.

    ``account_value``, ``bel``, ``ra`` and ``csm`` are ``(n_mp, n_time+1)``
    trajectories -- column 0 is the inception figure, the RA being a
    confidence-level margin for expense risk. The BEL is reported net of the
    account value the entity holds. The CSM is accreted at the
    underlying-items return and released by coverage units::

        csm[:, t+1] = csm[:, t] + csm_accretion[:, t] - csm_release[:, t]

    ``variable_fee`` is the present value of the entity's fee -- its share
    of the underlying items. ``loss_component`` and ``time_value`` are
    ``(n_mp,)`` inception figures; the time value of the guarantee drives
    the CSM but is reported separately from ``bel``.
    """

    # headline -- always present, shape (n_mp,)
    bel: FloatArray              # inception BEL (net of account value)
    ra: FloatArray               # inception RA (expense risk)
    csm: FloatArray              # inception CSM
    variable_fee: FloatArray     # PV of the entity's fee
    time_value: FloatArray       # guarantee TVOG at inception
    loss_component: FloatArray   # onerous loss at inception
    # trajectory -- full only (None on the headline-only path)
    bel_path: FloatArray | None = None            # (n_mp, n_time+1) -- BEL trajectory
    ra_path: FloatArray | None = None             # (n_mp, n_time+1) -- RA trajectory
    csm_path: FloatArray | None = None            # (n_mp, n_time+1) -- CSM trajectory
    account_value_path: FloatArray | None = None  # (n_mp, n_time+1) -- account-value trajectory
    csm_accretion: FloatArray | None = None       # (n_mp, n_time)
    csm_release: FloatArray | None = None          # (n_mp, n_time)
    lic: FloatArray | None = None                 # (n_mp, n_time+1)
    discount_bom: FloatArray | None = None      # (n_time+1,), or (n_mp, n_time+1) when portfolio-stitched
    cashflows: "Cashflows | None" = None
    model_points: "ModelPoints | None" = None     # stamped by measure_vfa, for group axes
    group_labels: "np.ndarray | None" = None       # per-group label on a grouped result
    group_sizes: IntArray | None = None         # model points per group, aligned with labels
    csm_basis: str = CSM_BASIS_PROJECTED_RUNOFF  # what the csm represents (see CSM_BASES)

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm),
                ("fee", self.variable_fee), ("TVOG", self.time_value),
                ("loss", self.loss_component)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr("VFAMeasurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str("VFAMeasurement", self._columns())


@dataclass(frozen=True, slots=True, eq=False)
class VFAAggregate:
    """Portfolio-aggregate VFA view -- a scalable sum of measured model-point
    results, holding no per-model-point row. Inception totals plus the run-off
    trajectories summed over the model-point axis. Computed in bounded memory, so
    it works where a per-model-point ``measure_vfa(full=True)`` would OOM. Not an
    IFRS group remeasurement and not a group re-floor engine: ``csm`` /
    ``loss_component`` are the sum of each contract's floored figure, matching the
    headline -- not a group-level re-floor.
    """

    bel: float                       # portfolio inception BEL total
    ra: float                        # portfolio inception RA total
    csm: float                       # portfolio inception CSM total
    variable_fee: float              # portfolio variable-fee total
    time_value: float                # portfolio guarantee TVOG total
    loss_component: float            # portfolio inception loss-component total
    bel_path: FloatArray             # (n_time+1,) -- aggregate BEL trajectory
    ra_path: FloatArray              # (n_time+1,) -- aggregate RA trajectory
    csm_path: FloatArray             # (n_time+1,) -- aggregate CSM trajectory
    lic: FloatArray                  # (n_time+1,) -- aggregate liability for incurred claims
    # No account_value_path: the account value is a per-policy level (its
    # closed-form growth never terminates at the contract boundary, so summing it
    # is horizon-dependent, not a clean aggregate) -- the group() VFA result drops
    # it for the same reason. The group's fund would be sum(inforce x av), a
    # different quantity, not modelled here.


@write_measurement.register
def _(measurement: VFAMeasurement, path, *, ids=None):
    _require_settlement_csm(measurement, "write_measurement")
    _write_measurement_columns(
        {"bel": measurement.bel, "ra": measurement.ra, "csm": measurement.csm,
         "variable_fee": measurement.variable_fee,
         "time_value": measurement.time_value,
         "loss_component": measurement.loss_component}, path, ids)


def _scatter_vfa_headline(n_mp, results):
    """Scatter per-chunk headline-only VFAMeasurements into one ``(n_mp,)`` result.

    ``results`` is ``[(idx, VFAMeasurement)]`` from ``measure_vfa(..., full=False)``
    over row-blocks; only the headline ``bel`` / ``ra`` / ``csm`` /
    ``variable_fee`` / ``time_value`` / ``loss_component`` are laid back, the
    trajectory fields staying ``None``. The portfolio orchestrator uses this on
    its ``full=False`` path so a chunked VFA partition costs ``O(n_mp)`` retained,
    not ``O(n_mp x n_time)``.
    """
    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    variable_fee = np.empty(n_mp)
    time_value = np.empty(n_mp)
    loss_component = np.empty(n_mp)
    for idx, m in results:
        bel[idx] = m.bel
        ra[idx] = m.ra
        csm[idx] = m.csm
        variable_fee[idx] = m.variable_fee
        time_value[idx] = m.time_value
        loss_component[idx] = m.loss_component
    return VFAMeasurement(
        bel=bel, ra=ra, csm=csm, variable_fee=variable_fee,
        time_value=time_value, loss_component=loss_component)


def _stitch_vfa_measurements(n_mp, sub_results):
    """Scatter per-segment VFAMeasurements into one ``(n_mp, ...)`` result.

    ``sub_results`` is ``[(idx, VFAMeasurement)]`` -- each segment's headline and
    trajectories are laid into the portfolio arrays at its rows and zero-padded
    on the right to the portfolio's longest horizon (a contract carries no BEL /
    CSM past its term). Like the GMM stitch, ``discount_bom`` becomes per-MP 2-D:
    VFA discounts at each segment's underlying-items return, so the curve is a
    property of the row, not one curve for the book. The padded tail repeats
    each row's last factor (a flat curve -> zero forward rate) so a rate read off
    it is finite, not a 0/0. The mixed-portfolio orchestrator
    (``fcf.portfolio.measure``) uses this to combine a VFA partition that spans
    several routing segments into one ``VFAMeasurement``.
    """
    n_time = max(m.bel_path.shape[1] - 1 for _, m in sub_results)

    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    variable_fee = np.empty(n_mp)
    time_value = np.empty(n_mp)
    loss_component = np.empty(n_mp)
    bel_path = np.zeros((n_mp, n_time + 1))
    ra_path = np.zeros((n_mp, n_time + 1))
    csm_path = np.zeros((n_mp, n_time + 1))
    account_value_path = np.zeros((n_mp, n_time + 1))
    csm_accretion = np.zeros((n_mp, n_time))
    csm_release = np.zeros((n_mp, n_time))
    lic = np.zeros((n_mp, n_time + 1))
    discount_bom = np.ones((n_mp, n_time + 1))

    cf_2d = ("inforce", "deaths", "premium_cf", "claim_cf", "morbidity_cf",
             "expense_cf", "annuity_cf", "disability_cf", "surrender_cf")
    cf_arrays = {name: np.zeros((n_mp, n_time)) for name in cf_2d}
    maturity_cf = np.zeros(n_mp)
    maturity_survivors = np.zeros(n_mp)

    for idx, m in sub_results:
        t = m.bel_path.shape[1] - 1
        bel[idx] = m.bel
        ra[idx] = m.ra
        csm[idx] = m.csm
        variable_fee[idx] = m.variable_fee
        time_value[idx] = m.time_value
        loss_component[idx] = m.loss_component
        bel_path[idx, :t + 1] = m.bel_path
        ra_path[idx, :t + 1] = m.ra_path
        csm_path[idx, :t + 1] = m.csm_path
        account_value_path[idx, :t + 1] = m.account_value_path
        csm_accretion[idx, :t] = m.csm_accretion
        csm_release[idx, :t] = m.csm_release
        lic[idx, :t + 1] = m.lic
        # Per-MP discount: lay the segment's 1-D curve across its rows, then
        # flat-fill the tail so the padded months read a zero forward rate.
        discount_bom[idx, :t + 1] = m.discount_bom
        discount_bom[idx, t + 1:] = m.discount_bom[-1]
        cf = m.cashflows
        for name in cf_2d:
            arr = getattr(cf, name)
            cf_arrays[name][idx, :arr.shape[1]] = arr
        maturity_cf[idx] = cf.maturity_cf
        maturity_survivors[idx] = cf.maturity_survivors

    cashflows = type(sub_results[0][1].cashflows)(
        maturity_cf=maturity_cf, maturity_survivors=maturity_survivors,
        **cf_arrays,
    )
    return VFAMeasurement(
        bel=bel, ra=ra, csm=csm, variable_fee=variable_fee,
        time_value=time_value, loss_component=loss_component,
        bel_path=bel_path, ra_path=ra_path, csm_path=csm_path,
        account_value_path=account_value_path,
        csm_accretion=csm_accretion, csm_release=csm_release, lic=lic,
        discount_bom=discount_bom, cashflows=cashflows,
    )


@dataclass(frozen=True, slots=True)
class _VFAProjection:
    """AV-anchored VFA projection shared by :func:`measure_vfa` (inception) and
    ``vfa.measure_inforce`` (subsequent measurement).

    Holds the trajectory building blocks that do not depend on the CSM choice:
    the account-value path, the BEL / RA trajectories, the variable-fee PV
    *trajectory* (PV of the fee from each month onward, so an in-force valuation
    can slice the remaining fee at the valuation date), the LIC, the cash flows,
    the coverage units and the underlying-items return. Each caller derives its
    own CSM (inception ``csm0`` vs a carried ``prior_csm``) and headline.
    """

    bel: FloatArray                  # (n_mp, n_time+1) BEL trajectory
    ra: FloatArray                   # (n_mp, n_time+1) RA trajectory
    variable_fee_path: FloatArray    # (n_mp, n_time+1) PV of the fee from each t
    time_value: FloatArray           # (n_mp,) guarantee TVOG at the anchor
    lic: FloatArray                  # (n_mp, n_time+1)
    cashflows: "Cashflows"
    inforce: FloatArray              # (n_mp, n_time) coverage units
    r_m: float                       # monthly underlying-items return
    av: FloatArray                   # (n_mp, n_time+1) account-value path
    disc_start: FloatArray           # (n_time+1,) start-of-month discount
    guarantee_excess_pv: FloatArray  # (n_mp, n_time+1) PV of the GMDB/GMAB excess over AV
    expense_pv: FloatArray           # (n_mp, n_time+1) PV of expenses


def _vfa_project(
    model_points: ModelPoints,
    basis: Basis,
    return_scenarios: FloatArray | None = None,
    *,
    elapsed_months: IntArray | None = None,
    account_value: FloatArray | None = None,
) -> "_VFAProjection":
    """Project VFA cash flows / trajectories, optionally re-anchoring the AV.

    With ``elapsed_months`` / ``account_value`` left ``None`` (the default) this
    is the inception projection: the account value grows from the model point's
    ``account_value`` at issue. For subsequent measurement, pass each contract's
    ``elapsed_months`` and its **observed** per-MP fund value (``account_value``)
    -- the account-value path is re-anchored so it equals the observed value at
    that duration (``av[t] = observed * growth ** (t - elapsed_months)``). The
    decrements come from the inception projection (they depend on policy
    duration, not the fund), and the AV-dependent benefits / fees recompute from
    the re-anchored path; the caller slices the trajectories at the valuation
    date.
    """
    basis = _single_basis(basis, entry="measure_vfa")
    # The VFA death money is ``deaths * death_benefit`` (below), computed from
    # the occupancy decrement -- it never reads the GMM death-claim factor. A
    # state-conditioned death benefit or occupancy exit would be silently
    # ignored here, so reject it rather than mis-measure the guarantee.
    state_model = resolve_state_model(basis)
    if any(s.death_benefit_factor != 1.0 for s in state_model.states):
        raise NotImplementedError(
            "state-conditioned death benefit (State.death_benefit_factor) is "
            "not supported on the VFA path; measure_vfa pays the GMDB/GMAB "
            "floor on the occupancy decrement, which the GMM death-claim "
            "factor does not reach."
        )
    if any(tr.after_sojourn_months
           for s in state_model.states for tr in s.transitions):
        raise NotImplementedError(
            "a deterministic transition (Transition.after_sojourn_months) is not "
            "supported on the VFA path."
        )
    proj = project_cashflows(model_points, basis)
    inforce = proj.inforce
    n_mp, n_time = inforce.shape

    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0
    # A minimum guarantee credits max(return, guarantee) to the account. The
    # guarantee is a per-policy contract term (locked at issue, cohort-aware),
    # carried on the model point -- zero means no effective guarantee when the
    # central return is non-negative.
    g_m = (1.0 + model_points.minimum_crediting_rate) ** (1.0 / 12.0) - 1.0
    credit_m = np.maximum(r_m, g_m)                            # (n_mp,)
    growth = (1.0 + credit_m) * (1.0 - f_m)                    # (n_mp,)

    # Account-value trajectory -- per-policy closed form, anchored at the
    # valuation date. At inception elapsed_months = 0 and account_value is the
    # model point's, so this is exactly av0 * growth^t (the original path);
    # for an in-force valuation the path is re-seeded to the observed fund value
    # at t = elapsed_months.
    periods = np.arange(n_time + 1)
    em = (np.zeros(n_mp, dtype=np.int64) if elapsed_months is None
          else np.asarray(elapsed_months, dtype=np.int64))
    av_anchor = (model_points.account_value if account_value is None
                 else np.asarray(account_value, dtype=np.float64))
    av = av_anchor[:, None] * (
        growth[:, None] ** (periods[None, :] - em[:, None])
    )

    # Every policy eventually exits and receives its account value -- except
    # that a death exit pays max(account value, guaranteed minimum death
    # benefit). ``deaths`` is the mortality portion of the decrement; the
    # remainder (lapse, maturity) takes the account value. With the default
    # zero GMDB this reduces exactly to ``exits * av`` (max(AV, 0) = AV).
    inforce_pad = np.concatenate([inforce, np.zeros((n_mp, 1))], axis=1)
    exits = inforce_pad[:, :-1] - inforce_pad[:, 1:]      # (n_mp, n_time)
    deaths = proj.deaths                                  # (n_mp, n_time)
    death_benefit = np.maximum(
        av[:, :n_time], model_points.minimum_death_benefit[:, None]
    )
    rows = np.arange(n_mp)
    # Maturity survivors are part of the exit flow at the maturity column, but
    # unlike a mid-month death or lapse (paid the start-of-month account value)
    # they reach the maturity date (time = term) with the *matured* account
    # value av[term] -- the value after the final month's growth. Split them out
    # of the start-of-month exit payout and pay them separately below, so the
    # base account-value half and the GMAB floor share one maturity date and
    # value (the GMM convention: maturity paid at time term).
    #
    # Maturity is realised only when the term falls within *that contract's own*
    # Sec. 34 boundary -- the projection runs each contract over range(boundary),
    # so maturity_survivors is non-zero only when term <= boundary. Decide
    # eligibility per contract, NOT from the portfolio-wide horizon n_time (= max
    # boundary): in a mixed book a short contract would otherwise read
    # term - 1 < n_time as "within" off another contract's longer horizon. Past a
    # boundary cut there is no maturity, so clamp the index (the zero maturity
    # weight makes the clamped cell harmless) rather than index out of bounds. At
    # boundary == term (the common case) term_idx is term - 1.
    boundary_idx = model_points.contract_boundary_months - 1
    within = (model_points.term_months - 1) <= boundary_idx
    term_idx = np.where(within, model_points.term_months - 1, boundary_idx)
    maturity_survivors = np.where(within, proj.maturity_survivors, 0.0)
    # Non-maturity exits (mid-month deaths and lapses) keep the start-of-month
    # account value; remove the maturity survivors from that flow.
    non_maturity_exits = exits - deaths
    non_maturity_exits[rows, term_idx] -= maturity_survivors
    benefit_cf = deaths * death_benefit + non_maturity_exits * av[:, :n_time]
    # Maturity survivors are paid max(account value, GMAB) on the matured value
    # av[term] (index term_idx + 1; av is width n_time + 1, so this is always in
    # range). Default zero GMAB pays the plain matured account value.
    av_at_maturity = av[rows, term_idx + 1]
    maturity_benefit = maturity_survivors * np.maximum(
        av_at_maturity, model_points.minimum_accumulation_benefit)
    # The GMAB excess (the guarantee's lift over the matured account value),
    # exposed separately for the paragraph-45 guarantee-cost disclosure below.
    maturity_excess = maturity_survivors * np.maximum(
        0.0, model_points.minimum_accumulation_benefit - av_at_maturity)
    # The maturity benefit enters benefit_cf at the term - 1 exit column
    # nominally, so the LIC settlement (and any other benefit_cf consumer) sees
    # the incurred amount. It is paid at maturity (time = term), one month past
    # that column, so the present-value path below discounts it the extra month.
    benefit_cf[rows, term_idx] += maturity_benefit
    # The extra-month discount, applied to the PV path only (the LIC keeps the
    # nominal amount). disc_start[term] / disc_start[term - 1] = 1/(1 + r_m), so
    # anchoring the maturity payout at time term scales its term - 1 cell by that
    # factor; this matches the GMM maturity convention (discount at the boundary
    # index). The guarantee-excess disclosure path takes the same shift on its
    # (smaller) GMAB-only amount.
    def _mat_shift(amount):
        shift = np.zeros((n_mp, n_time))
        shift[rows, term_idx] = amount * (1.0 / (1.0 + r_m) - 1.0)
        return shift
    mat_pv_shift = _mat_shift(maturity_benefit)
    g_pv_shift = _mat_shift(maturity_excess)
    # Variable fee -- the entity's share, deducted from the grown account value.
    fee_cf = inforce * av[:, :n_time] * (1.0 + credit_m)[:, None] * f_m
    # Liability for incurred claims -- exit benefits settled over the pattern.
    if basis.settlement_pattern is None:
        lic = np.zeros((n_mp, n_time + 1))
    else:
        lic = _settlement_lic(benefit_cf, basis.settlement_pattern)

    # Discount at the underlying-items return -- the VFA basis. Benefits are
    # discounted start-of-month, consistent with the account value, so a
    # zero fee leaves no profit.
    base = 1.0 + r_m
    disc_start = base ** (-np.arange(n_time + 1))
    disc_mid = base ** (-(np.arange(n_time) + 0.5))

    # Present-value trajectories -- the PV at each month t of the cash flows
    # from t onward, by a reverse cumulative discounted sum.
    def _pv_trajectory(cashflow: FloatArray, discount: FloatArray) -> FloatArray:
        tail = np.cumsum((cashflow * discount)[:, ::-1], axis=1)[:, ::-1]
        pv = np.zeros((n_mp, n_time + 1))
        pv[:, :n_time] = tail
        return pv / disc_start

    # A settlement pattern pays the exit benefit over later months -- so
    # discount it to those payment dates in the present value.
    benefit_for_pv = benefit_cf + mat_pv_shift
    if basis.settlement_pattern is not None:
        benefit_for_pv = benefit_for_pv * _settlement_factor(
            basis.settlement_pattern, r_m
        )
    pv_benefits = _pv_trajectory(benefit_for_pv, disc_start[:n_time])
    pv_expenses = _pv_trajectory(proj.expense_cf, disc_mid)
    # Variable fee as a PV *trajectory* (PV of the fee from each month onward).
    # Column 0 is the inception total (= the old scalar sum); an in-force
    # valuation slices the remaining fee at the valuation date.
    variable_fee_path = _pv_trajectory(fee_cf, disc_mid)
    # Guarantee-excess PV (the GMDB/GMAB cost over the account value) and the
    # expense PV, exposed for the paragraph-45 settlement movement's
    # future-service change (c) = -(dG + dE + dRA). The GMDB excess each month is
    # deaths*(max(av,gmdb)-av); the GMAB excess sits at the maturity column. Same
    # settlement-pattern discounting as the total benefit path (above).
    guarantee_excess_cf = deaths * (death_benefit - av[:, :n_time])
    guarantee_excess_cf[rows, term_idx] += maturity_excess
    g_for_pv = guarantee_excess_cf + g_pv_shift
    if basis.settlement_pattern is not None:
        g_for_pv = g_for_pv * _settlement_factor(
            basis.settlement_pattern, r_m)
    guarantee_excess_pv = _pv_trajectory(g_for_pv, disc_start[:n_time])

    # The deterministic BEL carries the guarantee's intrinsic value only.
    # Given return scenarios, fold in its time value too -- under the VFA
    # the CSM absorbs it.
    time_value = np.zeros(n_mp)
    if return_scenarios is not None:
        return_scenarios = np.asarray(return_scenarios, dtype=np.float64)
        if return_scenarios.ndim != 2 or return_scenarios.shape[1] != n_time:
            raise ValueError(
                f"return_scenarios must be 2-D (n_scenarios, {n_time}) -- "
                "the projection horizon"
            )
        return_scenarios = _validate_return_scenarios(return_scenarios)
        # tvog_weights is portfolio-level in v1, so it expects a uniform
        # guarantee across model points; per-MP varying guarantees with
        # stochastic returns are a future extension.
        g_unique = np.unique(model_points.minimum_crediting_rate)
        if g_unique.size > 1:
            raise NotImplementedError(
                "return_scenarios with per-MP varying minimum_crediting_rate "
                "is not supported yet; the time-value pass uses a scalar "
                "guarantee in v1"
            )
        time_value = model_points.account_value * (
            exits @ tvog_weights(
                minimum_crediting_rate=float(g_unique[0]),
                fund_fee=basis.fund_fee,
                investment_return=basis.investment_return,
                return_scenarios=return_scenarios,
            )
        )
        # The credit-rate guarantee above lifts the account growth; the GMDB
        # and GMAB are put-option floors on the account value. Add their time
        # value too -- each guarantee's intrinsic value is already in the BEL.
        time_value = time_value + guarantee_floor_time_value(
            account_value=model_points.account_value,
            deaths=proj.deaths,
            maturity_survivors=proj.maturity_survivors,
            term_index=model_points.term_months - 1,
            minimum_death_benefit=model_points.minimum_death_benefit,
            minimum_accumulation_benefit=model_points.minimum_accumulation_benefit,
            minimum_crediting_rate=float(g_unique[0]),
            fund_fee=basis.fund_fee,
            investment_return=basis.investment_return,
            return_scenarios=return_scenarios,
        )

    # BEL and RA as trajectories. The BEL is reported net of the account
    # value the entity holds -- a smooth, modest figure that at inception
    # nets to benefits and expenses less the premium (= the account value).
    fund = inforce_pad * av
    bel = pv_benefits + pv_expenses - fund
    # RA -- a confidence-level margin for expense risk, the non-financial
    # risk an account-value contract carries (mortality risk on the amount
    # is near zero, every exit paying the account value).
    ra = _norm_ppf(basis.ra_confidence) * basis.expense_cv * pv_expenses
    return _VFAProjection(
        bel=bel, ra=ra, variable_fee_path=variable_fee_path,
        time_value=time_value, lic=lic, cashflows=proj, inforce=inforce,
        r_m=r_m, av=av, disc_start=disc_start,
        guarantee_excess_pv=guarantee_excess_pv, expense_pv=pv_expenses,
    )


def measure_vfa(
    model_points: ModelPoints,
    basis: Basis,
    return_scenarios: FloatArray | None = None,
    *,
    full: bool = True,
) -> VFAMeasurement:
    """Measure a direct-participation portfolio under the Variable Fee Approach.

    The account value rolls forward as
    ``AV[t+1] = AV[t] * (1 + max(r, g)) * (1 - f)`` -- the credited rate (the
    underlying-items return ``r`` floored at any guaranteed rate ``g``) less
    the variable fee ``f`` -- from ``AV[0]`` = the model point's
    ``account_value``. A surrender pays the account value; a death exit pays
    ``max(account value, minimum_death_benefit)`` (GMDB) and the survivors
    reaching term pay ``max(account value, minimum_accumulation_benefit)``
    (GMAB), so the excess over the account value is each guarantee's intrinsic
    cost. When ``return_scenarios`` is given, each guarantee's *time value*
    (the extra cost from return volatility) is folded into the CSM too -- the
    credit-rate guarantee through the account-value growth, the GMDB and GMAB
    floors as put options on the account value.

    BEL is the present value of benefits and expenses less the premium, all
    at the underlying-items return; the CSM is ``max(0, -(BEL + RA))`` -- the
    entity's unearned variable fee -- accreted at the same return and
    released by coverage units. The RA is a confidence-level margin for
    expense risk.

    ``full=True`` (default) returns the BEL / RA / CSM / account-value
    trajectories; ``full=False`` fills only the headline ``bel`` / ``ra`` /
    ``csm`` / ``variable_fee`` / ``time_value`` / ``loss_component`` (the
    inception CSM is ``csm0``, so the release kernel is skipped) and leaves the
    trajectory and cash-flow fields ``None`` -- the building block the portfolio
    orchestrator chunks to bound memory.

    BEL, RA and CSM are returned as month-by-month trajectories. The
    deterministic BEL carries the guarantee's intrinsic value only; when
    ``return_scenarios`` -- an ``(n_scenarios, n_time)`` array of monthly
    underlying-items returns -- is supplied, the time value of the guarantee
    enters the inception fulfilment cash flows too, so the CSM absorbs it,
    and ``time_value`` records that amount per model point.
    """
    p = _vfa_project(model_points, basis, return_scenarios)
    n_time = p.inforce.shape[1]
    variable_fee = p.variable_fee_path[:, 0]
    # The inception fulfilment cash flows -- with the guarantee time value --
    # drive the CSM and the loss component.
    fcf = p.bel[:, 0] + p.ra[:, 0] + p.time_value
    loss_component = np.maximum(0.0, fcf)
    csm0 = np.maximum(0.0, -fcf)

    if not full:
        # Headline only: the inception CSM is csm0 (the full path's csm[:, 0]),
        # so the release kernel is skipped; the trajectory and cash-flow fields
        # are dropped so the chunked portfolio path retains O(n_mp) per row.
        return VFAMeasurement(
            bel=p.bel[:, 0], ra=p.ra[:, 0], csm=csm0, variable_fee=variable_fee,
            time_value=p.time_value, loss_component=loss_component,
            model_points=model_points)

    # VFA accretes at the underlying-items return -- flat across time in
    # the deterministic measurement; broadcast to the per-month curve the
    # kernel consumes.
    csm, csm_accretion, csm_release = _csm_kernel(
        csm0, p.inforce, np.full(n_time, p.r_m),
    )

    return VFAMeasurement(
        bel=p.bel[:, 0],
        ra=p.ra[:, 0],
        csm=csm[:, 0],
        variable_fee=variable_fee,
        time_value=p.time_value,
        loss_component=loss_component,
        bel_path=p.bel,
        ra_path=p.ra,
        csm_path=csm,
        account_value_path=p.av,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        lic=p.lic,
        discount_bom=p.disc_start,
        cashflows=p.cashflows,
        model_points=model_points,
    )


def measure_aggregate(
    model_points: ModelPoints,
    basis: Basis,
    *,
    chunk_size: int = 200_000,
) -> VFAAggregate:
    """Portfolio-aggregate VFA measurement in bounded memory.

    The VFA analogue of :func:`fastcashflow.gmm.measure_aggregate`: BEL / RA /
    CSM / variable fee / time value are additive across contracts, so the
    portfolio's run-off is the per-model-point trajectories summed over the
    model-point axis. Runs ``measure_vfa(..., full=True)`` over row-blocks of
    ``chunk_size`` model points and accumulates only the ``(n_time+1,)`` sums, so
    peak memory is ``O(chunk_size x n_time)`` regardless of ``n_mp``.

    Returns a :class:`VFAAggregate` (scalar totals + aggregate ``bel_path`` /
    ``ra_path`` / ``csm_path`` / ``lic``). ``account_value`` does not carry: it is
    a per-policy level whose closed-form growth never terminates at the boundary,
    so summing it is horizon-dependent (the ``group`` VFA result drops it for the
    same reason). The deterministic intrinsic value only -- the guarantee time
    value over return scenarios is a per-contract analysis (:func:`vfa.tvog`), not
    aggregated here. ``basis`` is a single :class:`Basis` (mixed / routed
    portfolios go through :func:`fastcashflow.portfolio.measure_aggregate`).
    """
    if chunk_size < 1:
        # Guard before the chunk loop: chunk_size <= 0 would skip every block and
        # return zero aggregates (silently wrong) instead of measuring anything.
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    n_mp = model_points.n_mp
    n_time = int(np.asarray(model_points.contract_boundary_months).max())
    bel_path = np.zeros(n_time + 1)
    ra_path = np.zeros(n_time + 1)
    csm_path = np.zeros(n_time + 1)
    lic = np.zeros(n_time + 1)
    bel = ra = csm = variable_fee = time_value = loss = 0.0
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        m = measure_vfa(model_points.subset(idx), basis, full=True)
        nt = m.bel_path.shape[1]
        bel_path[:nt] += m.bel_path.sum(axis=0)
        ra_path[:nt] += m.ra_path.sum(axis=0)
        csm_path[:nt] += m.csm_path.sum(axis=0)
        lic[:nt] += m.lic.sum(axis=0)
        bel += float(m.bel.sum())
        ra += float(m.ra.sum())
        csm += float(m.csm.sum())
        variable_fee += float(m.variable_fee.sum())
        time_value += float(m.time_value.sum())
        loss += float(m.loss_component.sum())
    return VFAAggregate(
        bel=bel, ra=ra, csm=csm, variable_fee=variable_fee,
        time_value=time_value, loss_component=loss, bel_path=bel_path,
        ra_path=ra_path, csm_path=csm_path, lic=lic)


def measure_inforce(
    model_points: ModelPoints,
    state: "InforceState",
    basis: Basis,
    *,
    period_months: int | None = None,
) -> VFAMeasurement:
    """In-force subsequent measurement of a VFA book at the valuation date.

    Unlike the PAA (no CSM) and like the GMM settlement, the prior period's
    closing CSM is carried forward; unlike the GMM, the VFA CSM accretes at the
    underlying-items return (not a locked-in rate), so the carry is
    ``_csm_kernel(state.prior_csm, coverage_units, r_m)`` -- the GMM settlement
    roll with the return substituted for the lock-in rate. The fulfilment cash
    flows (BEL / RA / variable fee / guarantee intrinsic value) are re-measured
    from the **observed** fund value at the valuation date
    (``state.account_value``): the account-value path is re-anchored at that
    observed value (:func:`_vfa_project`), the decrements come from the inception
    projection (they depend on policy duration, not the fund), and the result is
    sliced at each contract's ``elapsed_months`` and re-based by
    ``count / inforce[elapsed]`` (exact for cash flows linear in the in-force).

    ``state`` (an :class:`~fastcashflow.InforceState`) supplies the period-close
    ``elapsed_months`` / ``count`` (reconciled onto ``model_points`` by
    :func:`~fastcashflow.apply_inforce_state`), the carried ``prior_csm``, and
    the observed ``account_value`` (required here -- a VFA in-force needs the
    real fund value, not the modelled one). ``period_months`` (default 12) is the
    length of the period the prior CSM is rolled.

    v1 returns the **as-of valuation-date headline** (``bel`` / ``ra`` / ``csm``
    / ``variable_fee`` / ``time_value`` / ``loss_component``); the trajectory
    fields are ``None``. Deferred, with the same defer-the-unlocking stance as
    ``gmm.measure_inforce``: the paragraph-45 remeasurement (the change in the
    entity's share of the underlying-items fair value and the guarantee
    future-service cash flows), the loss-component path (``loss_component`` is
    zero), the stochastic time value of the guarantee (``time_value`` is zero --
    the deterministic intrinsic value only), and the full movement trajectory
    (use a period-close roll-forward, a later phase).

    .. warning::

       The CSM is a **carry-only approximation**: it is the prior CSM accreted
       at the basis underlying-items return and released by the *expected*
       (inception-run) coverage units. Unlike the BEL / RA / variable fee, it
       does **not** respond to the observed account value -- the paragraph-45
       adjustment for the change in the entity's share of the underlying-items
       fair value is deferred. So the fulfilment-cash-flow figures are
       observed-AV-consistent but the CSM is not yet a paragraph-45-compliant
       settlement CSM. A book whose fund diverged materially from the central
       return will show a CSM that lags that move until the paragraph-45
       remeasurement lands.

       The result is tagged ``csm_basis = 'carry_only'`` (see
       :data:`fastcashflow.vfa.CSM_BASES`), and the accounting-output entry
       points -- :func:`~fastcashflow.roll_forward`,
       :func:`~fastcashflow.report`, :func:`~fastcashflow.group`,
       :func:`~fastcashflow.group_of_contracts` and
       :func:`~fastcashflow.write_measurement` -- reject it, so this figure
       cannot be silently consumed as a settlement CSM.
    """
    basis = _single_basis(basis, entry="vfa.measure_inforce")
    # Reorder state to model-points order and reject a stale snapshot whose
    # elapsed_months / count disagree (same guard as the GMM/PAA paths).
    state = _reconcile_state(model_points, state)
    if state.account_value is None:
        raise ValueError(
            "vfa.measure_inforce needs the observed fund value at the valuation "
            "date: set InforceState.account_value. (The modelled account value "
            "would assume the fund followed the central return; a settlement "
            "must use the real fund. GMM / PAA in-force do not need it.)")
    n_mp = model_points.n_mp
    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    # The valuation date must lie strictly within each contract's own Sec. 34
    # boundary. The projection runs each contract over range(boundary), so its
    # in-force is filled for months 0 .. boundary-1; at or beyond the boundary
    # there is no remaining coverage to value. Decide this **per contract** --
    # not against the portfolio-wide horizon -- so a short contract in a mixed
    # book is judged on its own boundary, not another contract's longer one.
    # em < boundary <= n_time, so inforce[em] is always a valid, live column.
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    runoff = em >= boundary
    if np.any(runoff):
        bad = int(np.argmax(runoff))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} >= "
            f"contract_boundary_months[{bad}]={int(boundary[bad])} (the Sec. 34 "
            "horizon); the contract has no remaining coverage at the valuation "
            "date. Value it strictly before its boundary.")

    # Re-anchor the account-value path at the observed fund value at em, then
    # re-base the sliced headline to the valuation date (see _inforce_rescale).
    p = _vfa_project(model_points, basis,
                     elapsed_months=em, account_value=state.account_value)
    rows = np.arange(n_mp)
    rescale = _inforce_rescale(p, model_points, em, rows)
    bel = p.bel[rows, em] * rescale
    ra = p.ra[rows, em] * rescale
    variable_fee = p.variable_fee_path[rows, em] * rescale
    time_value = np.zeros(n_mp)            # stochastic TVOG deferred (intrinsic only)

    # Carry the prior closing CSM forward one period: accrete at the
    # underlying-items return r_m and release by coverage units, from
    # t = em - period_months to t = em -- the GMM settlement roll with r_m in
    # place of the lock-in rate (the VFA CSM has no locked-in rate). The carry
    # needs no rescale: the coverage-unit release is an in-force *fraction*.
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    prior_t = em - period
    if np.any(prior_t < 0):
        bad = int(np.argmin(prior_t))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} < period_months={period}; "
            "the prior closing date precedes inception, which has no CSM to "
            "carry forward")
    inforce = p.inforce
    n_time_total = inforce.shape[1]
    max_len = n_time_total - int(prior_t.min())
    col_offsets = np.arange(max_len)
    src_cols = prior_t[:, None] + col_offsets[None, :]
    mask = src_cols < n_time_total
    src_cols_safe = np.where(mask, src_cols, 0)
    inforce_seg = np.ascontiguousarray(
        np.where(mask, inforce[rows[:, None], src_cols_safe], 0.0))
    csm_traj, _, _ = _csm_kernel(
        np.asarray(state.prior_csm, dtype=np.float64),
        inforce_seg, np.full(max_len, p.r_m))
    csm = csm_traj[:, period]
    loss_component = np.zeros(n_mp)        # paragraph-45 onerous unlocking deferred

    return VFAMeasurement(
        bel=bel, ra=ra, csm=csm, variable_fee=variable_fee,
        time_value=time_value, loss_component=loss_component,
        model_points=model_points, csm_basis=CSM_BASIS_CARRY_ONLY)
