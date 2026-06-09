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
from fastcashflow.tvog import guarantee_floor_time_value, tvog_weights


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
    benefit_cf = deaths * death_benefit + (exits - deaths) * av[:, :n_time]
    # GMAB: the survivors reaching each policy's term receive max(account
    # value, minimum_accumulation_benefit). They sit in the (exits - deaths)
    # account-value payout at the maturity (term - 1) column; lift them by the
    # excess over the account value there. Default zero GMAB adds nothing.
    rows = np.arange(n_mp)
    term_idx = model_points.term_months - 1
    av_at_maturity = av[rows, term_idx]
    maturity_excess = proj.maturity_survivors * np.maximum(
        0.0, model_points.minimum_accumulation_benefit - av_at_maturity
    )
    benefit_cf[rows, term_idx] += maturity_excess
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
    benefit_for_pv = benefit_cf
    if basis.settlement_pattern is not None:
        benefit_for_pv = benefit_cf * _settlement_factor(
            basis.settlement_pattern, r_m
        )
    pv_benefits = _pv_trajectory(benefit_for_pv, disc_start[:n_time])
    pv_expenses = _pv_trajectory(proj.expense_cf, disc_mid)
    # Variable fee as a PV *trajectory* (PV of the fee from each month onward).
    # Column 0 is the inception total (= the old scalar sum); an in-force
    # valuation slices the remaining fee at the valuation date.
    variable_fee_path = _pv_trajectory(fee_cf, disc_mid)

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
