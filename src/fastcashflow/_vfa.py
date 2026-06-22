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
that return-rate accretion. A minimum guaranteed crediting rate is supported
-- the account is credited ``max(return, guarantee)`` each period. Its
intrinsic cost appears in this deterministic measurement; the time value of
the guarantee, the extra cost from return volatility, is measured
stochastically by ``measure_tvog``. A surrender charge (a fraction of the
account withheld on surrender, ``Basis.surrender_charge_annual``) is supported;
a lapse-risk adjustment is left for later. The risk adjustment here covers
expense risk -- the main non-financial risk an account-value contract carries,
with the policyholder bearing the investment risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow._measurement_basis import (
    MEASUREMENT_BASIS_INCEPTION,
    MEASUREMENT_BASIS_SETTLEMENT,
    MEASUREMENT_BASIS_SETTLEMENT_CARRY,
    _inforce_marker_columns,
)
from fastcashflow.basis import Basis, _single_basis
from fastcashflow.io import (
    write_measurement, _write_measurement_columns,
    _stream_single_file, _vfa_model_points_from_frame)
from fastcashflow.numerics import (
    _carry_lic_residual,
    _csm_kernel,
    _norm_ppf,
    _settlement_factor,
    _settlement_lic,
    _settlement_lic_discounted,
)
from fastcashflow.model_points import ModelPoints, NO_GUARANTEE_RATE
from fastcashflow.projection import Cashflows, project_cashflows
from fastcashflow.state_model import resolve_state_model
from fastcashflow.tvog import (
    guarantee_floor_time_value, ul_guarantee_floor_time_value,
    measure_tvog,
    tvog_weights, tvog_term_weight,
    _validate_return_scenarios,
    credited_monthly_rate,
)
# In-force helpers shared with the GMM path (engine does not import _vfa, and io
# imports engine lazily, so this top-level import is cycle-free).
from fastcashflow.engine import _reconcile_state, _inforce_rescale


def moneyness_lapse_multiplier(moneyness, sensitivity, *, floor: float = 0.0,
                               cap=None):
    """Dynamic-lapse multiplier as a function of account-value moneyness.

    ``moneyness`` is the account value divided by the guarantee (the guaranteed
    floor / GMAB): ``> 1`` means the guarantee is out-of-the-money (the account
    already exceeds it), ``< 1`` means it is in-the-money (the floor is valuable).
    The fixed linear form ``1 + sensitivity * (moneyness - 1)`` LIFTS lapse when the
    guarantee is worthless (policyholders surrender to chase the account) and LOWERS
    it when the floor bites (they hold the valuable guarantee), clamped to
    ``[floor, cap]`` (``cap=None`` is no upper bound). At-the-money (``moneyness ==
    1``) the multiplier is 1 (no adjustment).

    ``sensitivity`` is the injected behavioural elasticity -- the seam; the FORM is
    fixed, so a moneyness PATH (the account value is exogenous to lapse, a per-policy
    level) resolves to a per-period multiplier array up front, with no per-step
    callback. This is the account-value counterpart to
    :func:`fastcashflow.solvency.dynamic_lapse_multiplier` (the parallel rate form).
    It is a behavioural PRIMITIVE; feeding the resulting lapse back into the in-force
    projection is a separate integration step. Accepts a scalar or an array and
    returns the same shape."""
    m = np.asarray(moneyness, dtype=np.float64)
    mult = np.clip(1.0 + sensitivity * (m - 1.0), floor,
                   np.inf if cap is None else cap)
    return float(mult) if m.ndim == 0 else mult


def moneyness_lapse_scale(model_points: ModelPoints, basis: Basis,
                          sensitivity: float, *, floor: float = 0.0,
                          cap=None) -> FloatArray:
    """Per-policy-year lapse multiplier from the account-value moneyness path.

    Resolves the dynamic-lapse seam UP FRONT into the ``(n_mp, n_years)`` array
    :func:`fastcashflow.projection.project_cashflows` consumes as ``lapse_scale``.
    At the start of each policy year the account value -- the closed-form VFA
    growth path ``av0 * growth ** month`` -- is divided by the GMAB
    (``minimum_accumulation_benefit``, the guarantee a surrender forgoes) to get
    the moneyness, then :func:`moneyness_lapse_multiplier` (``sensitivity``,
    ``floor``, ``cap``) maps it to a lapse factor for that year. A contract with
    no GMAB (guarantee ``<= 0``) gets a flat ``1.0`` -- no floor to weigh, so the
    lapse is untouched.

    The account value is exogenous to lapse (a per-policy level, not a count), so
    the whole moneyness path is known before the decrement projection -- the
    factor array is built once here and the kernel sees only the scaled lapse (no
    per-step callback). This is the inception path (``elapsed_months = 0``); an
    in-force re-anchored moneyness is a later extension. It is the account-value
    counterpart to :func:`fastcashflow.solvency.interest_with_dynamic_lapse` (the
    parallel-rate scalar form)."""
    basis = _single_basis(basis, entry="moneyness_lapse_scale")
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0
    credit_m = credited_monthly_rate(r_m, model_points.minimum_crediting_rate)
    growth = (1.0 + credit_m) * (1.0 - f_m)                       # (n_mp,)
    n_time = int(model_points.contract_boundary_months.max())
    n_years = (n_time + 11) // 12
    year_start = np.arange(n_years) * 12                          # 0, 12, 24, ...
    av0 = np.asarray(model_points.account_value, dtype=np.float64)
    av_year = av0[:, None] * growth[:, None] ** year_start[None, :]   # (n_mp, n_years)
    return _moneyness_scale_from_av_year(
        av_year, model_points.minimum_accumulation_benefit, sensitivity, floor, cap)


def _moneyness_scale_from_av_year(av_year, gmab, sensitivity, floor, cap):
    """The shared moneyness -> per-policy-year lapse factor: ``av_year / GMAB`` mapped
    through :func:`moneyness_lapse_multiplier`, with a flat ``1.0`` where the GMAB is
    absent (``<= 0``). ``av_year`` is the account value at each policy-year start --
    the closed-form path for the variable-annuity case, the rolled account path for
    the universal-life case."""
    gmab = np.asarray(gmab, dtype=np.float64)
    has_g = gmab > 0.0
    safe_g = np.where(has_g, gmab, 1.0)[:, None]
    scale = moneyness_lapse_multiplier(av_year / safe_g, sensitivity,
                                       floor=floor, cap=cap)
    scale = np.where(has_g[:, None], scale, 1.0)
    return np.ascontiguousarray(scale)


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
CSM_BASIS_PARAGRAPH_45 = "paragraph_45_settlement"  # vfa.settle: subsequent meas.
CSM_BASES = (CSM_BASIS_INITIAL, CSM_BASIS_PROJECTED_RUNOFF,
             CSM_BASIS_CARRY_ONLY, CSM_BASIS_PARAGRAPH_45)

# The VFA keeps csm_basis as the stored field (single source of truth) and
# derives the cross-model measurement_basis from it (see _measurement_basis).
_CSM_TO_MEASUREMENT_BASIS = {
    CSM_BASIS_INITIAL: MEASUREMENT_BASIS_INCEPTION,
    CSM_BASIS_PROJECTED_RUNOFF: MEASUREMENT_BASIS_INCEPTION,
    CSM_BASIS_CARRY_ONLY: MEASUREMENT_BASIS_SETTLEMENT_CARRY,
    CSM_BASIS_PARAGRAPH_45: MEASUREMENT_BASIS_SETTLEMENT,
}


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
            f"cannot be used by {op}. For the real paragraph-45 subsequent "
            "measurement (the opening->closing settlement movement) use "
            "fastcashflow.vfa.settle(...).")


@dataclass(frozen=True, slots=True, eq=False)
class VFAMeasurement:
    """VFA measurement of a direct-participation (account-value) portfolio.

    The headline ``bel``, ``ra``, ``csm``, ``variable_fee``, ``time_value`` and
    ``loss_component`` are ``(n_mp,)`` as-of figures -- at inception for
    ``measure``, at the valuation date for ``measure_inforce`` (the RA a
    confidence-level margin for expense risk; the BEL net of the account value
    the entity holds; ``variable_fee`` the present value of the entity's fee --
    its share of the underlying items). The full path adds the
    ``(n_mp, n_time+1)`` trajectories ``bel_path`` / ``ra_path`` / ``csm_path`` /
    ``account_value_path`` (column 0 the as-of figure), ``None`` on the
    headline-only path; a grouped result also leaves ``account_value_path``
    ``None`` (the account value is a per-policy level, not a group quantity). The
    CSM is accreted at the underlying-items return and released by coverage
    units::

        csm_path[:, t+1] = csm_path[:, t] + csm_accretion[:, t] - csm_release[:, t]

    The guarantee time value drives the CSM but is reported separately in
    ``time_value``, not folded into ``bel``.
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
    lic_path: FloatArray | None = None            # (n_mp, n_time+1) -- liability for incurred claims.
    # The entity's own-pocket insurance cash flows, retained for the asset-liability
    # gap (a unit-linked book's account-value benefits are funded by the unit fund;
    # only the guarantee excess over the account value lands on the entity's general
    # account). Full VA path only -- None on the headline / aggregate / UL paths.
    guarantee_excess_cf: FloatArray | None = None  # (n_mp, n_time) GMDB/GMAB excess over AV
    benefit_cf: FloatArray | None = None           # (n_mp, n_time) gross incurred benefit (AV + excess)
    fee_cf: FloatArray | None = None               # (n_mp, n_time) variable fee skimmed (entity inflow)
    # The terminal column holds the residual of claims whose settlement tail
    # runs past the horizon (stays non-zero by design, not a leak).
    discount_factor_bom: FloatArray | None = None      # (n_time+1,), or (n_mp, n_time+1) when portfolio-stitched
    cashflows: "Cashflows | None" = None
    model_points: "ModelPoints | None" = None     # stamped by measure_vfa, for group axes
    group_labels: "np.ndarray | None" = None       # per-group label on a grouped result
    group_sizes: IntArray | None = None         # model points per group, aligned with labels
    csm_basis: str = CSM_BASIS_PROJECTED_RUNOFF  # what the csm represents (see CSM_BASES)

    @property
    def measurement_basis(self) -> str:
        """Cross-model time-basis discriminator, derived from ``csm_basis``
        (the VFA's stored field stays the single source of truth)."""
        return _CSM_TO_MEASUREMENT_BASIS[self.csm_basis]

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
    lic_path: FloatArray             # (n_time+1,) -- aggregate liability for incurred claims
    # No account_value_path: the account value is a per-policy level (its
    # closed-form growth never terminates at the contract boundary, so summing it
    # is horizon-dependent, not a clean aggregate) -- the group() VFA result drops
    # it for the same reason. The group's fund would be sum(inforce x av), a
    # different quantity, not modelled here.


@write_measurement.register
def _(measurement: VFAMeasurement, path, *, ids=None):
    _require_settlement_csm(measurement, "write_measurement")
    cols = {"bel": measurement.bel, "ra": measurement.ra,
            "csm": measurement.csm,
            "variable_fee": measurement.variable_fee,
            "time_value": measurement.time_value,
            "loss_component": measurement.loss_component}
    # A paragraph-45 closing balance gets the same marker columns as the
    # other models' non-inception output; inception output is unchanged.
    cols.update(_inforce_marker_columns(measurement, measurement.bel.shape[0]))
    _write_measurement_columns(cols, path, ids)


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
    CSM past its term). Like the GMM stitch, ``discount_factor_bom`` becomes per-MP 2-D:
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
    lic_path = np.zeros((n_mp, n_time + 1))
    discount_factor_bom = np.ones((n_mp, n_time + 1))

    cf_2d = ("inforce", "deaths", "premium_cf", "mortality_cf", "morbidity_cf",
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
        lic_path[idx, :t + 1] = m.lic_path
        _carry_lic_residual(lic_path, idx, t, n_time, m.lic_path)
        # Per-MP discount: lay the segment's 1-D curve across its rows, then
        # flat-fill the tail so the padded months read a zero forward rate.
        discount_factor_bom[idx, :t + 1] = m.discount_factor_bom
        discount_factor_bom[idx, t + 1:] = m.discount_factor_bom[-1]
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
        csm_accretion=csm_accretion, csm_release=csm_release, lic_path=lic_path,
        discount_factor_bom=discount_factor_bom, cashflows=cashflows,
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
    lic_path: FloatArray             # (n_mp, n_time+1)
    benefit_cf: FloatArray           # (n_mp, n_time) incurred benefit claims (builds the LIC)
    guarantee_excess_cf: FloatArray  # (n_mp, n_time) GMDB/GMAB excess over AV (the insurance claim, IC-excluded)
    fee_cf: FloatArray               # (n_mp, n_time) variable fee skimmed (the entity's inflow)
    cashflows: "Cashflows"
    inforce: FloatArray              # (n_mp, n_time) coverage units
    r_m: float                       # monthly underlying-items return
    av: FloatArray                   # (n_mp, n_time+1) account-value path
    discount_factor_bom: FloatArray  # (n_time+1,) start-of-month discount
    guarantee_excess_pv: FloatArray  # (n_mp, n_time+1) PV of the GMDB/GMAB excess over AV
    expense_pv: FloatArray           # (n_mp, n_time+1) PV of expenses


def _vfa_project(
    model_points: ModelPoints,
    basis: Basis,
    return_scenarios: FloatArray | None = None,
    *,
    elapsed_months: IntArray | None = None,
    account_value: FloatArray | None = None,
    tv_reduce: bool = True,
    _proj: "Cashflows | None" = None,
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

    ``_proj`` (private) lets a caller running this twice on the same book
    (``vfa.settle``: an expected and an observed leg) share one decrement
    projection -- the legs differ only in the AV anchor, and sharing makes
    the in-force / coverage-unit arrays bit-identical between them.
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
    proj = _proj if _proj is not None else project_cashflows(model_points, basis)
    inforce = proj.inforce
    n_mp, n_time = inforce.shape

    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0
    # A minimum guarantee credits max(return, guarantee) to the account. The
    # guarantee is a per-policy contract term (locked at issue, cohort-aware),
    # carried on the model point: NO_GUARANTEE_RATE means the account follows
    # the bare return, 0.0 is a real 0% floor (see credited_monthly_rate). r_m
    # is a scalar and the guarantee is per model point, so credit_m is (n_mp,).
    credit_m = credited_monthly_rate(r_m, model_points.minimum_crediting_rate)
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
    # nominal amount). discount_factor_bom[term] / discount_factor_bom[term - 1] = 1/(1 + r_m), so
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
    # Variable fee -- the entity's share, skimmed from the grown account value.
    # Charged only on the policies in the fund THROUGH month-end, which incur
    # that month's credit-and-fee growth: the start-of-month in-force less this
    # month's mid-month exits (deaths and non-maturity lapses). Those exits
    # leave at the start of the month with the un-grown av[t] -- the same
    # convention benefit_cf uses above (lines 425-429) -- so they take no part
    # in the growth or the fee. The maturity survivors are NOT in
    # non_maturity_exits (removed at term_idx above): they reach the maturity
    # date with the matured av[term] and so correctly keep the final month's
    # fee. A Sec. 34 boundary cut sets maturity_survivors = 0, so a censored
    # contract pays no final-month fee, matching its un-grown start-of-month
    # payout. Written subtractively (== inforce_pad[:, 1:] + maturity_survivors
    # at term_idx) so it never MUTATES inforce_pad: the slice inforce_pad[:, 1:]
    # is a view, so scattering the maturity add-back in place would corrupt the
    # BEL fund (fund = inforce_pad * av) computed below -- the subtractive form
    # needs no defensive copy.
    fee_base = inforce - deaths - non_maturity_exits
    fee_cf = fee_base * av[:, :n_time] * (1.0 + credit_m)[:, None] * f_m
    # Liability for incurred claims -- exit benefits settled over the pattern.
    if basis.settlement_pattern is None:
        lic_path = np.zeros((n_mp, n_time + 1))
    else:
        lic_path = _settlement_lic(benefit_cf, basis.settlement_pattern)

    # Discount at the underlying-items return -- the VFA basis. Benefits are
    # discounted start-of-month, consistent with the account value, so a
    # zero fee leaves no profit.
    base = 1.0 + r_m
    discount_factor_bom = base ** (-np.arange(n_time + 1))
    disc_mid = base ** (-(np.arange(n_time) + 0.5))

    # Present-value trajectories -- the PV at each month t of the cash flows
    # from t onward, by a reverse cumulative discounted sum.
    def _pv_trajectory(cashflow: FloatArray, discount: FloatArray) -> FloatArray:
        tail = np.cumsum((cashflow * discount)[:, ::-1], axis=1)[:, ::-1]
        pv = np.zeros((n_mp, n_time + 1))
        pv[:, :n_time] = tail
        return pv / discount_factor_bom

    # A settlement pattern pays the exit benefit over later months -- so
    # discount it to those payment dates in the present value.
    benefit_for_pv = benefit_cf + mat_pv_shift
    if basis.settlement_pattern is not None:
        benefit_for_pv = benefit_for_pv * _settlement_factor(
            basis.settlement_pattern, r_m
        )
    pv_benefits = _pv_trajectory(benefit_for_pv, discount_factor_bom[:n_time])
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
    guarantee_excess_pv = _pv_trajectory(g_for_pv, discount_factor_bom[:n_time])

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
        # The credit-rate TVOG weights are portfolio-level (one crediting
        # guarantee), so the time-value pass PARTITIONS the book by distinct
        # minimum_crediting_rate: each group is uniform and shares the weights,
        # and a mixed-guarantee book sums the per-group per-MP results (the
        # credit-rate TVOG and the GMDB/GMAB floor are independent per model
        # point). The GMDB/GMAB floors themselves already vary per model point.
        g_arr = np.asarray(model_points.minimum_crediting_rate, dtype=np.float64)
        n_scen = return_scenarios.shape[0]
        av0 = model_points.account_value
        time_value = np.zeros(n_mp) if tv_reduce else np.zeros((n_mp, n_scen))
        for g in np.unique(g_arr):
            m = g_arr == g                                # this group's model points
            gf = float(g)
            w = tvog_weights(
                minimum_crediting_rate=gf, fund_fee=basis.fund_fee,
                investment_return=basis.investment_return,
                return_scenarios=return_scenarios, reduce=tv_reduce)
            w_term = tvog_term_weight(
                minimum_crediting_rate=gf, fund_fee=basis.fund_fee,
                investment_return=basis.investment_return,
                return_scenarios=return_scenarios, reduce=tv_reduce)
            # Maturity survivors sit in `exits` at term_idx (= term - 1) but exit
            # at time = term, credited the full final month -- peel them off that
            # column weight w[term_idx] and re-seat one month later (the GMAB /
            # Sec. B119 handling). w_ext = [w, w_term] selects by term_idx + 1.
            if tv_reduce:
                w_ext = np.append(w, w_term)
                credit = av0[m] * (
                    exits[m] @ w
                    - maturity_survivors[m] * w[term_idx[m]]
                    + maturity_survivors[m] * w_ext[term_idx[m] + 1])
            else:
                # Per-scenario forms: w is (n_scen, n_time), w_term (n_scen,) ->
                # the reduced expression vectorised over the scenario axis.
                w_ext = np.concatenate([w, w_term[:, None]], axis=1)
                credit = av0[m][:, None] * (
                    exits[m] @ w.T
                    - maturity_survivors[m][:, None] * w[:, term_idx[m]].T
                    + maturity_survivors[m][:, None] * w_ext[:, term_idx[m] + 1].T)
            # The GMDB and GMAB are put-option floors on the account value; add
            # their time value (each guarantee's intrinsic value is in the BEL).
            floor = guarantee_floor_time_value(
                account_value=av0[m], deaths=proj.deaths[m],
                maturity_survivors=maturity_survivors[m], term_index=term_idx[m],
                minimum_death_benefit=model_points.minimum_death_benefit[m],
                minimum_accumulation_benefit=model_points.minimum_accumulation_benefit[m],
                minimum_crediting_rate=gf, fund_fee=basis.fund_fee,
                investment_return=basis.investment_return,
                return_scenarios=return_scenarios, reduce=tv_reduce)
            time_value[m] = credit + floor

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
        time_value=time_value, lic_path=lic_path, benefit_cf=benefit_cf,
        guarantee_excess_cf=guarantee_excess_cf, fee_cf=fee_cf,
        cashflows=proj, inforce=inforce,
        r_m=r_m, av=av, discount_factor_bom=discount_factor_bom,
        guarantee_excess_pv=guarantee_excess_pv, expense_pv=pv_expenses,
    )


def measure_vfa(
    model_points: ModelPoints,
    basis: Basis,
    return_scenarios: FloatArray | None = None,
    *,
    full: bool = True,
    lapse_sensitivity: float | None = None,
    lapse_floor: float = 0.0,
    lapse_cap=None,
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
    ``basis`` must resolve to a single :class:`Basis`; multi-segment routers are not
    accepted.

    BEL, RA and CSM are returned as month-by-month trajectories. The
    deterministic BEL carries the guarantee's intrinsic value only; when
    ``return_scenarios`` -- an ``(n_scenarios, n_time)`` array of monthly
    underlying-items returns -- is supplied, the time value of the guarantee
    enters the inception fulfilment cash flows too, so the CSM absorbs it,
    and ``time_value`` records that amount per model point.

    ``lapse_sensitivity`` (default ``None`` -- a static lapse) turns on a dynamic
    lapse driven by the account-value moneyness: the lapse decrement is scaled by
    the per-policy-year moneyness factor (``lapse_sensitivity`` the elasticity,
    clamped to ``[lapse_floor, lapse_cap]``), which lifts surrenders when the GMAB is
    out-of-the-money and lowers them when the floor bites. The factor keys on the
    GMAB; both the closed-form variable-annuity path (the account value is the
    closed-form growth path) and the account-backed universal-life path (the value
    is read from the rolled account) are supported.
    """
    basis = _single_basis(basis, entry="measure_vfa")
    # Variable universal life: an account-backed (universal-life) book is
    # measured through the SHARED recursive account roll (the projection fold,
    # identical to the GMM path), discounted at the underlying-items return --
    # the only thing the VFA model changes. The closed-form _vfa_project
    # (variable-annuity, no cost-of-insurance) handles the account-flag-absent
    # case. Branch STRICTLY on the coverage flags, never account_value (the
    # variable-annuity product carries an account value too).
    from fastcashflow.engine import _measure_full, _portfolio_has_account
    if _portfolio_has_account(model_points, basis):
        r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
        n_time = int(model_points.contract_boundary_months.max())
        lapse_scale = None
        if lapse_sensitivity is not None:
            # Moneyness dynamic lapse on the universal-life path. Roll the account
            # once (static lapse) to read its value path, resolve the per-policy-year
            # moneyness factor against the GMAB, then re-measure with the scaled
            # lapse. The account value is a per-policy level (independent of the
            # decrement count), so the static-roll path is the right driver.
            m0 = _measure_full(model_points, basis,
                               discount_monthly=np.full(n_time, r_m))
            n_years = (n_time + 11) // 12
            year_start = np.minimum(np.arange(n_years) * 12, n_time)
            av_year = m0.cashflows.account.av[:, year_start]
            lapse_scale = _moneyness_scale_from_av_year(
                av_year, model_points.minimum_accumulation_benefit,
                lapse_sensitivity, lapse_floor, lapse_cap)
        m = _measure_full(model_points, basis,
                          discount_monthly=np.full(n_time, r_m),
                          lapse_scale=lapse_scale)
        zeros = np.zeros_like(m.bel)  # UL has no asset-based variable fee
        if return_scenarios is None:
            # Deterministic: the BEL carries the guarantee's intrinsic value only.
            time_value = zeros
        else:
            # Guarantee time value: re-roll the account under the return scenarios
            # and price the GMDB / GMAB floors (mean cost less the central
            # intrinsic), partitioned by the crediting guarantee. v1 still rejects
            # an annuitizing book (the conversion floor differs from the
            # GMDB/GMAB-at-maturity option) -- raised inside the helper.
            return_scenarios = np.asarray(return_scenarios, dtype=np.float64)
            if return_scenarios.ndim != 2 or return_scenarios.shape[1] != n_time:
                raise ValueError(
                    f"return_scenarios must be 2-D (n_scenarios, {n_time}) -- "
                    "the projection horizon")
            time_value = _ul_floor_time_value(
                model_points, basis, m.cashflows.deaths,
                m.cashflows.maturity_survivors, return_scenarios, reduce=True)
        # Fold the time value into the inception FCF; the CSM absorbs it (with
        # time_value == 0 this reproduces the deterministic m.csm / m.loss_component).
        fcf = m.bel + m.ra + time_value
        loss_component = np.maximum(0.0, fcf)
        csm0 = np.maximum(0.0, -fcf)
        if not full:
            return VFAMeasurement(
                bel=m.bel, ra=m.ra, csm=csm0, variable_fee=zeros,
                time_value=time_value, loss_component=loss_component,
                model_points=model_points)
        if return_scenarios is None:
            csm_path, csm_accretion, csm_release = (
                m.csm_path, m.csm_accretion, m.csm_release)
        else:
            csm_path, csm_accretion, csm_release = _csm_kernel(
                csm0, m.cashflows.inforce, np.full(n_time, r_m),
                basis.coverage_unit_discount)
        return VFAMeasurement(
            bel=m.bel, ra=m.ra, csm=csm_path[:, 0], variable_fee=zeros,
            time_value=time_value, loss_component=loss_component,
            bel_path=m.bel_path, ra_path=m.ra_path, csm_path=csm_path,
            account_value_path=m.cashflows.account.av,
            csm_accretion=csm_accretion, csm_release=csm_release,
            lic_path=m.lic_path, discount_factor_bom=m.discount_factor_bom,
            cashflows=m.cashflows, model_points=model_points)

    proj = None
    if lapse_sensitivity is not None:
        scale = moneyness_lapse_scale(model_points, basis, lapse_sensitivity,
                                      floor=lapse_floor, cap=lapse_cap)
        proj = project_cashflows(model_points, basis, lapse_scale=scale)
    p = _vfa_project(model_points, basis, return_scenarios, _proj=proj)
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
        basis.coverage_unit_discount,
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
        lic_path=p.lic_path,
        discount_factor_bom=p.discount_factor_bom,
        cashflows=p.cashflows,
        guarantee_excess_cf=p.guarantee_excess_cf,
        benefit_cf=p.benefit_cf,
        fee_cf=p.fee_cf,
        model_points=model_points,
    )


def measure_vfa_stochastic(model_points: ModelPoints, basis: Basis,
                           return_scenarios: FloatArray):
    """The VFA liability distribution over fund-return scenarios -- the VFA
    counterpart of :func:`fastcashflow.gmm.stochastic`.

    For each ``return_scenarios`` path the full guarantee cost (the credited-rate
    floor plus the GMDB / GMAB account-value floors) is realised and folded into
    the liability, giving a per-scenario BEL / RA / CSM / loss component. Read the
    distribution off the returned
    :class:`~fastcashflow.stochastic.StochasticResult` with ``mean`` /
    ``percentile``. The mean BEL reconciles to ``vfa.measure(...,
    return_scenarios).bel + .time_value`` (the risk-neutral price); the mean CSM
    differs from the pooled-scenario CSM because the loss-component floor is convex
    -- that convexity is the reason to read the distribution.

    ``return_scenarios`` is an ``(n_scenarios, n_time)`` array of monthly
    underlying-items returns (e.g. :attr:`fastcashflow.esg.EconomicScenarios.returns`).

    Vectorised over the scenario axis: the deterministic projection (the intrinsic
    BEL and RA) is scenario-independent and runs once; only the guarantee time
    value varies by scenario, computed as one ``(n_mp, n_scenarios)`` matrix. The
    per-scenario realised BEL is the intrinsic BEL plus that path's time value, and
    the CSM / loss re-floor the fulfilment cash flow (BEL + RA + time value) per
    scenario -- identical to running :func:`measure_vfa` once per path, but without
    re-projecting."""
    from fastcashflow.stochastic import StochasticResult
    rs = np.asarray(return_scenarios, dtype=np.float64)
    if rs.ndim != 2:
        raise ValueError("return_scenarios must be 2-D (n_scenarios, n_time)")
    if rs.shape[0] == 0:
        raise ValueError("return_scenarios is empty; need at least one scenario")
    n_scen = rs.shape[0]
    basis = _single_basis(basis, entry="measure_vfa_stochastic")

    bel_mp, ra_mp, tv = _vfa_time_value_by_scenario(model_points, basis, rs)
    # tv is (n_mp, n_scenarios); bel_mp / ra_mp are the scenario-independent
    # intrinsic per-MP BEL and RA. The realised liability per path is the intrinsic
    # BEL plus that path's guarantee time value; the CSM / loss re-floor the FCF
    # (BEL + RA + time value) per scenario -- exactly the per-scenario measure_vfa.
    fcf = (bel_mp + ra_mp)[:, None] + tv               # (n_mp, n_scenarios)
    bel = float(bel_mp.sum()) + tv.sum(axis=0)         # (n_scenarios,)
    ra = np.full(n_scen, float(ra_mp.sum()))
    csm = np.maximum(0.0, -fcf).sum(axis=0)
    loss = np.maximum(0.0, fcf).sum(axis=0)
    return StochasticResult(bel=bel, ra=ra, csm=csm, loss_component=loss)


def _ul_floor_time_value(model_points: ModelPoints, basis: Basis,
                         deaths: FloatArray, maturity_survivors: FloatArray,
                         return_scenarios: FloatArray, *, reduce: bool = True):
    """Per-MP guarantee time value of a universal-life account book, PARTITIONED by
    distinct crediting guarantee (each group is uniform; the GMDB / GMAB floors
    themselves vary per model point). Re-rolls the account under the return
    scenarios and prices the floors (mean cost less the central intrinsic). Returns
    ``(n_mp,)`` (``reduce``) or ``(n_mp, n_scenarios)``."""
    from fastcashflow.engine import _account_roll_inputs
    am = getattr(model_points, "annuitization_months", None)
    if am is not None and np.any(np.asarray(am) > 0):
        raise NotImplementedError(
            "return_scenarios (guarantee time value) is not yet supported "
            "for an annuitizing universal-life book.")
    (av0, face, prem_to_av, coi_rate_m, admin_fee, account_charge,
     gmab, _g, _sc) = _account_roll_inputs(model_points, basis)
    g_arr = np.asarray(model_points.minimum_crediting_rate, dtype=np.float64)
    boundary = np.asarray(model_points.contract_boundary_months, np.int64)
    n_mp = av0.shape[0]
    n_scen = return_scenarios.shape[0]
    out = np.zeros(n_mp) if reduce else np.zeros((n_mp, n_scen))
    for g in np.unique(g_arr):
        m = g_arr == g                                    # this group's model points
        out[m] = ul_guarantee_floor_time_value(
            account_value0=av0[m], face=face[m], prem_to_av=prem_to_av[m],
            coi_rate_m=coi_rate_m[m], admin_fee=admin_fee,    # admin_fee is shared (n_time,)
            account_charge=account_charge[m], gmab=gmab[m],
            minimum_crediting_rate=float(g), deaths=deaths[m],
            maturity_survivors=maturity_survivors[m], boundary=boundary[m],
            investment_return=basis.investment_return,
            return_scenarios=return_scenarios, reduce=reduce)
    return out


def _vfa_time_value_by_scenario(model_points: ModelPoints, basis: Basis,
                                return_scenarios: FloatArray):
    """The scenario-independent intrinsic per-MP BEL / RA and the
    ``(n_mp, n_scenarios)`` guarantee time-value matrix -- the vectorised core of
    :func:`measure_vfa_stochastic`. The universal-life (account) book re-rolls the
    account under every scenario in one pass; the variable-annuity closed form
    builds the credit-rate and floor time-value matrices directly."""
    from fastcashflow.engine import _measure_full, _portfolio_has_account
    rs = np.asarray(return_scenarios, dtype=np.float64)
    if _portfolio_has_account(model_points, basis):
        r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
        n_time = int(model_points.contract_boundary_months.max())
        if rs.shape[1] != n_time:
            raise ValueError(
                f"return_scenarios must be 2-D (n_scenarios, {n_time}) -- "
                "the projection horizon")
        m = _measure_full(model_points, basis, discount_monthly=np.full(n_time, r_m))
        tv = _ul_floor_time_value(model_points, basis, m.cashflows.deaths,
                                  m.cashflows.maturity_survivors, rs, reduce=False)
        return m.bel, m.ra, tv
    # Variable-annuity closed form.
    p = _vfa_project(model_points, basis, rs, tv_reduce=False)
    return p.bel[:, 0], p.ra[:, 0], p.time_value


def measure_stream(
    input_path,
    output_dir,
    basis: Basis,
    *,
    calculation_methods=None,
    chunk_size: int = 20_000_000,
    id_column: str | None = None,
    validate_unique_mp_id: bool = True,
    return_scenarios: FloatArray | None = None,
) -> int:
    """Stream a VFA (account-value) valuation through a parquet file, chunk by chunk.

    The VFA counterpart of :func:`~fastcashflow.gmm.measure_stream`. The VFA base
    is a single policies frame (account value + guarantee floors, no coverages),
    so it reads ``input_path`` in ``chunk_size`` blocks, measures each with
    ``vfa.measure(..., full=False)``, and writes per-chunk ``part-NNNNN.parquet``
    results (bel / ra / csm / variable_fee / time_value). Returns the model points
    processed. ``basis`` is a single :class:`Basis`.

    ``return_scenarios`` -- an ``(n_scenarios, horizon)`` array of monthly
    underlying-items returns -- prices the guarantee time value (TVOG) of each
    chunk, exactly as :func:`measure`. The per-model-point time value depends only
    on a contract's own account path over its own term, so it is additive across
    chunks and invariant to the chunk a row lands in: each chunk slices the
    scenario prefix it needs (its own horizon ``contract_boundary_months.max()``),
    and the result equals an in-memory ``measure(..., return_scenarios=...)`` over
    the whole book. ``horizon`` must therefore cover the longest contract in the
    file. Omitted (the default), the stream values deterministically and
    ``time_value`` is 0.

    Marginal benefit note: streaming is for portfolios too large to hold in
    memory (a GMM book of 1e8 rows). Variable books are typically far smaller, so
    :func:`measure` / :func:`measure_aggregate` usually suffice; this exists for
    API symmetry.
    """
    basis = _single_basis(basis, entry="vfa.measure_stream")
    rs = None if return_scenarios is None else np.asarray(
        return_scenarios, dtype=np.float64)

    def measure_fn(mp):
        if rs is None:
            return measure_vfa(mp, basis, full=False)
        # Slice the scenario prefix to this chunk's horizon -- a row's time value
        # is decided within its own term, so the prefix is exact (and a too-short
        # scenario surfaces as the shape check inside measure_vfa).
        n_time = int(mp.contract_boundary_months.max())
        return measure_vfa(mp, basis, full=False, return_scenarios=rs[:, :n_time])

    return _stream_single_file(
        input_path, output_dir, chunk_size=chunk_size, id_column=id_column,
        validate_unique_mp_id=validate_unique_mp_id,
        build_mp=lambda frame: _vfa_model_points_from_frame(frame, calculation_methods),
        measure_fn=measure_fn,
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
    ``ra_path`` / ``csm_path`` / ``lic_path``). ``account_value`` does not carry: it is
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
    lic_path = np.zeros(n_time + 1)
    bel = ra = csm = variable_fee = time_value = loss = 0.0
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        m = measure_vfa(model_points.subset(idx), basis, full=True)
        nt = m.bel_path.shape[1]
        bel_path[:nt] += m.bel_path.sum(axis=0)
        ra_path[:nt] += m.ra_path.sum(axis=0)
        csm_path[:nt] += m.csm_path.sum(axis=0)
        lic_path[:nt] += m.lic_path.sum(axis=0)
        bel += float(m.bel.sum())
        ra += float(m.ra.sum())
        csm += float(m.csm.sum())
        variable_fee += float(m.variable_fee.sum())
        time_value += float(m.time_value.sum())
        loss += float(m.loss_component.sum())
    return VFAAggregate(
        bel=bel, ra=ra, csm=csm, variable_fee=variable_fee,
        time_value=time_value, loss_component=loss, bel_path=bel_path,
        ra_path=ra_path, csm_path=csm_path, lic_path=lic_path)


def measure_inforce(
    model_points: ModelPoints,
    state: "InforceState",
    basis: Basis,
    *,
    period_months: int | None = None,
) -> VFAMeasurement:
    """In-force diagnostic / runoff valuation of a VFA book at a single date.

    Unlike the PAA (no CSM) and like the GMM carry
    (:func:`fastcashflow.gmm.measure_inforce`), the prior period's
    closing CSM is carried forward; unlike the GMM, the VFA CSM accretes at the
    underlying-items return (not a locked-in rate), so the carry is
    ``_csm_kernel(state.prior_csm, coverage_units, r_m)`` -- the GMM carry
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
    fields are ``None``. Deferred here: the paragraph-45 remeasurement (the
    change in the entity's share of the underlying-items fair value and the
    guarantee future-service cash flows) and the loss-component movement --
    **both live in** :func:`~fastcashflow.vfa.settle`, the paragraph-45
    settlement of the same state -- plus the stochastic time value of the
    guarantee (``time_value`` is zero -- the deterministic intrinsic value
    only) and the full movement trajectory.

    .. warning::

       The CSM is a **carry-only approximation**: it is the prior CSM accreted
       at the basis underlying-items return and released by the *expected*
       (inception-run) coverage units. Unlike the BEL / RA / variable fee, it
       does **not** respond to the observed account value -- the paragraph-45
       adjustment for the change in the entity's share of the underlying-items
       fair value is deferred. So the fulfilment-cash-flow figures are
       observed-AV-consistent but the CSM is not yet a paragraph-45-compliant
       settlement CSM; for that, settle the same state with
       :func:`~fastcashflow.vfa.settle`.

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
    rescale = _inforce_rescale(p.cashflows.inforce, model_points, em, rows)
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
        inforce_seg, np.full(max_len, p.r_m), basis.coverage_unit_discount)
    csm = csm_traj[:, period]
    loss_component = np.zeros(n_mp)        # paragraph-45 onerous unlocking deferred

    return VFAMeasurement(
        bel=bel, ra=ra, csm=csm, variable_fee=variable_fee,
        time_value=time_value, loss_component=loss_component,
        model_points=model_points, csm_basis=CSM_BASIS_CARRY_ONLY)


# The paragraph-48/50(b) CSM / loss-component step is shared with the GMM
# settlement (gmm.settle) and lives in numerics; re-exported here for the
# existing VFA call sites and tests.
from fastcashflow.numerics import _paragraph45_csm_algebra  # noqa: E402,F401


def settle(
    model_points: ModelPoints,
    state: "InforceState",
    basis: Basis,
    *,
    period_months: int | None = None,
    premium_experience_future_fraction: float | FloatArray = 0.0,
) -> "VFASettlementMovement":
    """Paragraph-45 subsequent-measurement settlement of a VFA in-force book
    (period close).

    The real IFRS 17 paragraph-45 opening -> closing movement, replacing the
    carry-only ``vfa.measure_inforce`` headline for settlement / disclosure:
    the CSM is adjusted for the change in the entity's share of the fair
    value of the underlying items (45(b)) and for the changes in fulfilment
    cash flows relating to future service (45(c)), the loss component is
    reversed / recognised per paragraphs 48 and 50(b), and one paragraph-B119
    coverage-unit release is taken on the post-adjustment balance. Returns a
    :class:`~fastcashflow.movement.VFASettlementMovement` whose blocks
    reconcile exactly and whose :meth:`closing_measurement
    <fastcashflow.movement.VFASettlementMovement.closing_measurement>` is a
    settlement-grade closing balance sheet
    (``csm_basis='paragraph_45_settlement'``).

    ``state`` is the closing-dated :class:`~fastcashflow.InforceState`
    extended with the prior reporting date's figures, all at month
    ``elapsed_months - period_months``: ``prior_csm`` (the opening CSM),
    ``prior_count``, ``prior_account_value`` and (optionally)
    ``prior_loss_component``; plus the closing snapshot's ``count`` and
    observed ``account_value``. The same state feeds ``measure_inforce``
    (the fast carry-only diagnostic) and this settlement.

    How it computes: two forward projections of the same book share one
    decrement run -- the EXPECTED leg anchors the account value at the
    opening observation and advances it under the basis return; the OBSERVED
    leg anchors at the closing observation -- and both are sliced at the
    closing date. The paragraph-45 future-service change is **minus the
    observed-vs-expected difference of the engine's own BEL and RA** (exact
    with respect to every engine convention, including the end-of-month fee
    timing and a binding minimum-crediting-rate floor); the 45(b) /
    45(c) split is a disclosure decomposition of that exact total, the
    45(b) line being the fund-consistent (end-of-month-weighted)
    variable-fee PV difference. The CSM accretes by direct compounding at
    the underlying-items return over the period and is released once at
    period end over the coverage units provided in the period against those
    provided plus expected from the opening date. With no experience
    deviation this reproduces the ``measure_inforce`` monthly carry exactly
    (a telescoping identity of the release weights).

    A contract whose Sec. 34 boundary falls inside the period is a **final
    settlement**: allowed, provided its closing ``count`` and
    ``account_value`` are zero (validated) -- its remaining CSM releases in
    full. The opening date must lie strictly within every contract's
    boundary.

    An onerous book amortises its loss component through the paragraph-50(a)/51
    incurred-service channel (``loss_component_finance`` /
    ``loss_component_amortised``): the period's guarantee-excess + expense
    release (the claims+expenses pool, which for VFA excludes the account-value
    investment component) is split on the loss-component ratio, running the loss
    component to zero by the end of coverage (52). The B96(c) investment-
    component experience (``csm_investment_experience`` -- the expected less the
    actual account value returned on exits) adjusts the CSM when
    ``state.actual_investment_component`` is given.

    v1 scope (see :class:`~fastcashflow.movement.VFASettlementMovement` for
    the full statement): deterministic, single basis rate (the
    finance-not-adjusting-CSM split is an exact zero residual), intrinsic
    guarantee only, no assumption change, within-period experience assumed
    equal to expected (only the closing count and observed account value
    deviate), per-model-point floors. A basis with a ``settlement_pattern`` is supported:
    the movement carries the liability for incurred claims (``lic_opening`` /
    ``claims_incurred`` / ``claims_paid`` / ``lic_closing``, paragraphs 40(b) /
    42 / 103(b)) -- benefit claims build it up as incurred and run it off over
    the pattern, undiscounted and at the expected scale, reconstructed from the
    projection each period (the same v1 cuts as gmm.settle: no 42(b)/(c)).
    """
    from fastcashflow.movement import VFASettlementMovement

    basis = _single_basis(basis, entry="vfa.settle")
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    # Reorder state to model-points order and reject a stale snapshot whose
    # elapsed_months / count disagree (same guard as measure_inforce).
    state = _reconcile_state(model_points, state)
    if state.account_value is None:
        raise ValueError(
            "vfa.settle needs the observed fund value at the closing date: "
            "set InforceState.account_value.")
    if state.prior_account_value is None:
        raise ValueError(
            "vfa.settle needs the observed fund value at the opening date: "
            "set InforceState.prior_account_value (the prior reporting "
            "date's observation, like prior_csm).")
    if state.prior_count is None:
        raise ValueError(
            "vfa.settle needs the in-force count at the opening date: set "
            "InforceState.prior_count (the prior reporting date's count, "
            "like prior_csm).")
    n_mp = model_points.n_mp
    prior_csm = np.asarray(state.prior_csm, dtype=np.float64)
    if np.any(prior_csm < 0.0):
        raise ValueError(
            "InforceState.prior_csm must be >= 0 for vfa.settle; a VFA CSM "
            "is floored at zero (an onerous balance is the loss component)")
    lc_open = (np.zeros(n_mp) if state.prior_loss_component is None
               else np.asarray(state.prior_loss_component, dtype=np.float64))
    both = np.minimum(prior_csm, lc_open) > 0.0
    if np.any(both):
        bad = np.flatnonzero(both)[:5].tolist()
        raise ValueError(
            f"prior_csm and prior_loss_component are both positive at "
            f"row(s) {bad}; an IFRS 17 group carries a CSM or a loss "
            "component, never both")
    em_close = np.asarray(state.elapsed_months, dtype=np.int64)
    em_open = em_close - period
    if np.any(em_open < 0):
        bad = int(np.argmin(em_open))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em_close[bad])} < "
            f"period_months={period}; the opening date precedes inception, "
            "which has no state to settle from")
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    dead_at_open = em_open >= boundary
    if np.any(dead_at_open):
        bad = int(np.argmax(dead_at_open))
        raise ValueError(
            f"elapsed_months[{bad}] - period_months = {int(em_open[bad])} >= "
            f"contract_boundary_months[{bad}]={int(boundary[bad])} (the "
            "Sec. 34 horizon); the contract had no remaining coverage at the "
            "opening date, so there is nothing to settle.")
    # A boundary inside the period is a final settlement -- legitimate, but
    # the closing snapshot must agree that the contract is gone.
    count = np.asarray(state.count, dtype=np.float64)
    observed_av = np.asarray(state.account_value, dtype=np.float64)
    runoff = em_close >= boundary
    bad_runoff = runoff & ((count != 0.0) | (observed_av != 0.0))
    if np.any(bad_runoff):
        bad = int(np.argmax(bad_runoff))
        raise ValueError(
            f"row {bad} reaches its contract boundary within the period "
            f"(elapsed_months={int(em_close[bad])} >= "
            f"contract_boundary_months={int(boundary[bad])}) -- a final "
            "settlement -- but its closing count / account_value is not "
            "zero. A matured contract must close with count=0 and "
            "account_value=0.")
    # Per-MP clamped closing column: value-neutral (a contract's trajectory
    # columns at/past its own boundary are zero), needed only so a mixed
    # book's short contract never indexes off the shared arrays.
    em_c = np.minimum(em_close, boundary)

    # Two forward projections, one shared decrement run. The legs take their
    # anchors EXPLICITLY from the state -- never from the seated model points,
    # whose elapsed_months / count are the closing snapshot. The projection
    # itself is seeded at count = 1 (the survival curve): the seated count is
    # the CLOSING observation, and seeding with it would zero the whole
    # in-force for a matured or mass-lapsed row -- the re-basing factors
    # below carry the real opening / closing counts instead.
    from dataclasses import replace as _replace
    proj_mp = _replace(model_points, count=np.ones(n_mp))
    proj = project_cashflows(proj_mp, basis)
    p_exp = _vfa_project(proj_mp, basis, elapsed_months=em_open,
                         account_value=state.prior_account_value, _proj=proj)
    p_obs = _vfa_project(proj_mp, basis, elapsed_months=em_close,
                         account_value=observed_av, _proj=proj)
    rows = np.arange(n_mp)
    inforce = p_exp.inforce                       # the ONE shared array
    n_time = inforce.shape[1]
    inforce_pad = np.concatenate([inforce, np.zeros((n_mp, 1))], axis=1)
    r_m = p_exp.r_m
    v_half = (1.0 + r_m) ** -0.5

    # Re-basing factors -- per-MP scalars from the state's own opening /
    # closing counts (computed inline: _inforce_rescale reads
    # model_points.count, which is the CLOSING count). A dead-mid-life
    # cohort (count = 0 on a live column) gets k_obs = 0 -- full
    # derecognition -- not the dead-column no-op.
    if_open = inforce[rows, em_open]
    k_exp = (np.asarray(state.prior_count, dtype=np.float64)
             / np.where(if_open > 0.0, if_open, 1.0))
    if_close = inforce_pad[rows, em_c]
    k_obs = np.where(if_close > 0.0,
                     count / np.where(if_close > 0.0, if_close, 1.0), 1.0)

    def _fcf_block(exp_path, obs_path):
        """The five movement lines of one FCF block (BEL or RA).

        One residual -- the release line (the expected run-off, the
        roll-forward convention); every other line is computed directly, so
        an error cannot hide in two places at once.
        """
        opening = k_exp * exp_path[rows, em_open]
        offsets = np.arange(period)
        src = em_open[:, None] + offsets[None, :]
        mask = src <= n_time
        vals = np.where(mask, exp_path[rows[:, None], np.where(mask, src, 0)],
                        0.0)
        interest = r_m * k_exp * vals.sum(axis=1)
        expected_close = k_exp * exp_path[rows, em_c]
        release = opening + interest - expected_close
        closing = k_obs * obs_path[rows, em_c]
        experience = closing - expected_close
        return opening, interest, release, experience, closing

    (bel_opening, bel_interest, bel_release,
     bel_experience, bel_closing) = _fcf_block(p_exp.bel, p_obs.bel)
    (ra_opening, ra_interest, ra_release,
     ra_experience, ra_closing) = _fcf_block(p_exp.ra, p_obs.ra)

    # The paragraph-45 future-service change: exactly minus the
    # observed-vs-expected FCF difference. The 45(b)/45(c) lines are a
    # disclosure split of this exact total -- the fee line weighted
    # end-of-month (the fee is skimmed on the grown account value, while
    # variable_fee_path discounts mid-month), so a pure account-value move
    # with no guarantee and no crediting floor lands entirely in 45(b).
    x = -(bel_experience + ra_experience)
    fee_exp = p_exp.variable_fee_path[rows, em_c]
    fee_obs = p_obs.variable_fee_path[rows, em_c]
    csm_fv_share = v_half * k_obs * (fee_obs - fee_exp)
    csm_future_service = x - csm_fv_share

    # B96(a)/B97(c) premium experience -- the VFA mirror of the gmm.settle
    # split. The actual premium received less the expected (on-track) premium
    # over the period splits by the entity's future-service fraction: the
    # future leg adjusts the CSM (B96(a), a NEW future-service change with no
    # BEL/RA counterpart, so it stays OUTSIDE the csm_fv_share / csm_future_
    # service cross-tie -- it enters the paragraph-45 algebra on top of x); the
    # current/past leg is a P&L memo (B97(c)). Absent actual_premium => zero on
    # both lines (byte-identical to the pre-feature settle).
    pe_frac = np.asarray(premium_experience_future_fraction, dtype=np.float64)
    if pe_frac.ndim > 1 or (pe_frac.ndim == 1 and pe_frac.shape[0] != n_mp):
        raise ValueError(
            "premium_experience_future_fraction must be a scalar or one entry "
            f"per model point ({n_mp}), got shape {pe_frac.shape}")
    if (not np.all(np.isfinite(pe_frac))
            or np.any(pe_frac < 0.0) or np.any(pe_frac > 1.0)):
        raise ValueError(
            "premium_experience_future_fraction must be finite and lie in "
            "[0, 1] (the entity's split of the premium experience between "
            "future service -> CSM and current/past service -> P&L); got "
            f"{premium_experience_future_fraction}")
    if state.actual_premium is not None:
        pe_off = np.arange(period)
        pe_src = em_open[:, None] + pe_off[None, :]
        pe_mask = pe_src < n_time
        exp_premium = k_exp * np.where(
            pe_mask, proj.premium_cf[rows[:, None], np.where(pe_mask, pe_src, 0)],
            0.0).sum(axis=1)
        premium_experience = (np.asarray(state.actual_premium,
                                         dtype=np.float64) - exp_premium)
    else:
        premium_experience = np.zeros(n_mp)
    csm_premium_experience = pe_frac * premium_experience
    premium_experience_revenue = (1.0 - pe_frac) * premium_experience

    # paragraph 50(a)/51 incurred-service loss-component channel (the VFA
    # mirror of gmm.settle). paragraph 51(a) allocates claims and expenses
    # excluding investment components -- for VFA the investment component IS the
    # account value, so the insurance "claims" are the guarantee excess (the
    # GMDB/GMAB benefit above the account value). The pool is therefore
    # guarantee_excess + expenses + RA (inherently IC-excluded). The loss
    # component accretes r x the pool interest (51c) and amortises r x the pool
    # release (50(a)); r = lc_open / pool_open is re-derived every period so the
    # LC runs to zero by the end of coverage (52). A profitable book
    # (lc_open == 0) is byte-identical. The algebra below acts on the
    # POST-amortisation loss component.
    ge_o, ge_i, ge_r, _, _ = _fcf_block(p_exp.guarantee_excess_pv,
                                        p_exp.guarantee_excess_pv)
    ex_o, ex_i, ex_r, _, _ = _fcf_block(p_exp.expense_pv, p_exp.expense_pv)
    pool_open = ge_o + ex_o + ra_opening
    lc_ratio = np.where(pool_open > 0.0,
                        lc_open / np.where(pool_open > 0.0, pool_open, 1.0), 0.0)
    loss_component_finance = lc_ratio * (ge_i + ex_i + ra_interest)
    loss_component_amortised = lc_ratio * (ge_r + ex_r + ra_release)
    lc_after_incurred = lc_open + loss_component_finance - loss_component_amortised

    # B96(c) investment-component experience (VFA): the investment component is
    # the account value returned on exits (benefit_cf - guarantee_excess_cf).
    # The expected less the actual account value payable over the period adjusts
    # the CSM (the whole difference, no fraction -- B96(c) is entirely future
    # service); the account value does not touch insurance revenue. Absent
    # actual_investment_component => zero (byte-identical).
    ic_cf = p_exp.benefit_cf - p_exp.guarantee_excess_cf
    ic_src = em_open[:, None] + np.arange(period)[None, :]
    ic_mask = ic_src < n_time
    expected_ic = k_exp * np.where(
        ic_mask, ic_cf[rows[:, None], np.where(ic_mask, ic_src, 0)],
        0.0).sum(axis=1)
    if state.actual_investment_component is not None:
        actual_ic = np.asarray(state.actual_investment_component,
                               dtype=np.float64)
        csm_investment_experience = expected_ic - actual_ic
    else:
        csm_investment_experience = np.zeros(n_mp)

    # CSM / loss-component algebra (paragraphs 45 / 48 / 50(b)). Accrete by
    # direct compounding (NOT the monthly roll -- that would interleave
    # releases and double-count against the single B119 release below). The
    # premium- and investment-experience future legs are new future-service
    # changes with no BEL/RA counterpart, so they enter here on top of x.
    accreted = prior_csm * (1.0 + r_m) ** period
    csm_accretion = accreted - prior_csm
    csm_after, lc_reversed, lc_recognised, lc_closing = (
        _paragraph45_csm_algebra(
            accreted, x + csm_premium_experience + csm_investment_experience,
            lc_after_incurred))

    # Paragraph-B119 release, once, on the post-adjustment balance: units
    # provided in the period (expected basis) over those provided plus
    # expected from the OPENING date (the closing-date remainder at the
    # observed count). With on-track experience this equals the monthly
    # carry's telescoped release exactly; at a final settlement the future
    # units are zero and the whole balance releases.
    cu_tail = np.concatenate(
        [np.cumsum(inforce[:, ::-1], axis=1)[:, ::-1], np.zeros((n_mp, 1))],
        axis=1)
    cu_period = k_exp * (cu_tail[rows, em_open] - cu_tail[rows, em_c])
    cu_future = k_obs * cu_tail[rows, em_c]
    denom = cu_period + cu_future
    eps = 1e-12 * np.where(denom > 0.0, denom, 1.0)
    live = denom > eps
    # No coverage units at all (full derecognition -- a mass surrender / all
    # exits) releases the WHOLE remaining CSM (B119 / paragraph 76), frac=1,
    # matching gmm.settle; a 0.0 fallback would strand the CSM of a group that
    # no longer provides coverage.
    frac = np.where(live, cu_period / np.where(live, denom, 1.0), 1.0)
    csm_release = csm_after * frac
    csm_closing = csm_after - csm_release

    # Liability for incurred claims (paragraphs 40(b) / 42(c) / 103(b)) -- the
    # VFA mirror of the GMM/PAA LIC block. Benefit claims build it up as
    # incurred (42(a)) and run it off over the settlement pattern. The LIC is
    # measured at fulfilment cash flows -- the discounted PV of the unpaid
    # run-off (42(c)). It carries NO risk adjustment: the VFA RA prices expense
    # risk only (z x expense_cv x pv_expenses), the benefit risk sitting in the
    # variable fee, so the incurred benefits carry no RA in the LIC either. This
    # INHERITS the engine's VFA-RA-is-expense-only convention -- a benefit RA on
    # the LIC (paragraphs 32/37) would first require pricing benefit RA in the
    # VFA LRC; adding it here alone would make the LIC and LRC inconsistent.
    # claims_incurred and claims_paid stay NOMINAL (claims_paid the residual on
    # the undiscounted trajectory p_exp.lic_path); lic_finance is the reconciling
    # residual -- the 42(c) discount unwind plus the discounting measurement
    # effect. p_exp.lic_path is all-zero without a settlement_pattern (claims paid as
    # incurred, the LIC zero at both dates and lic_finance zero).
    #
    # DISCOUNT RATE -- the LIC discounts at basis.discount_monthly (the IFRS 17
    # discount curve), NOT the underlying-items return r_m the rest of the VFA
    # measurement uses. This is deliberate: an INCURRED claim is a fixed,
    # determined amount that no longer varies with the underlying items, so its
    # liability discounts at a rate that does not reflect underlying-item
    # variability (B74) -- the same rate the GMM/PAA LIC uses. r_m is for the
    # not-yet-incurred, account-value-linked future benefits. (Pinned by an
    # off-diagonal test where discount_annual != investment_return.)
    offsets = np.arange(period)
    inc_src = em_open[:, None] + offsets[None, :]
    inc_mask = inc_src < n_time
    benefit = p_exp.benefit_cf
    claims_incurred = k_exp * np.where(
        inc_mask, benefit[rows[:, None], np.where(inc_mask, inc_src, 0)],
        0.0).sum(axis=1)
    claims_paid = (k_exp * p_exp.lic_path[rows, em_open] + claims_incurred
                   - k_exp * p_exp.lic_path[rows, em_c])
    if basis.settlement_pattern is not None:
        lic_disc = _settlement_lic_discounted(
            benefit, basis.settlement_pattern, basis.discount_monthly)
        lic_opening = k_exp * lic_disc[rows, em_open]
        lic_closing = k_exp * lic_disc[rows, em_c]
    else:
        lic_opening = k_exp * p_exp.lic_path[rows, em_open]
        lic_closing = k_exp * p_exp.lic_path[rows, em_c]
    lic_finance = lic_closing - lic_opening - claims_incurred + claims_paid

    # B97(b)/(c) within-period claims and expense experience (the gmm.settle
    # mirror): the actual benefits / expenses incurred over the period less the
    # expected, recognised in the insurance service result (P&L memos -- not the
    # CSM, no balance recursion). Expected claims are claims_incurred (above);
    # expected expenses are the period expense run at the expected scale. Absent
    # state.actual_claims / state.actual_expenses => zero (byte-identical).
    exp_expenses = k_exp * np.where(
        inc_mask, proj.expense_cf[rows[:, None], np.where(inc_mask, inc_src, 0)],
        0.0).sum(axis=1)
    if state.actual_claims is not None:
        claims_experience = (np.asarray(state.actual_claims, dtype=np.float64)
                             - claims_incurred)
    else:
        claims_experience = np.zeros(n_mp)
    if state.actual_expenses is not None:
        expense_experience = (np.asarray(state.actual_expenses, dtype=np.float64)
                              - exp_expenses)
    else:
        expense_experience = np.zeros(n_mp)

    return VFASettlementMovement(
        period_months=period,
        bel_opening=bel_opening,
        bel_interest=bel_interest,
        bel_release=bel_release,
        bel_experience=bel_experience,
        bel_closing=bel_closing,
        ra_opening=ra_opening,
        ra_interest=ra_interest,
        ra_release=ra_release,
        ra_experience=ra_experience,
        ra_closing=ra_closing,
        csm_opening=prior_csm,
        csm_accretion=csm_accretion,
        csm_fv_share=csm_fv_share,
        csm_future_service=csm_future_service,
        csm_premium_experience=csm_premium_experience,
        premium_experience_revenue=premium_experience_revenue,
        csm_investment_experience=csm_investment_experience,
        claims_experience=claims_experience,
        expense_experience=expense_experience,
        loss_component_reversed=lc_reversed,
        loss_component_recognised=lc_recognised,
        csm_release=csm_release,
        csm_closing=csm_closing,
        loss_component_opening=lc_open,
        loss_component_finance=loss_component_finance,
        loss_component_amortised=loss_component_amortised,
        loss_component_closing=lc_closing,
        variable_fee_closing=k_obs * fee_obs,
        coverage_units_provided=cu_period,
        coverage_units_future=cu_future,
        account_value_closing=observed_av,
        lic_opening=lic_opening,
        claims_incurred=claims_incurred,
        lic_finance=lic_finance,
        claims_paid=claims_paid,
        lic_closing=lic_closing,
        lock_in_rate=float(state.lock_in_rate),
        model_points=model_points,
    )


def settle_aggregate(
    model_points: ModelPoints,
    state: "InforceState",
    basis: Basis,
    *,
    period_months: int | None = None,
    chunk_size: int = 200_000,
    premium_experience_future_fraction: float | FloatArray = 0.0,
) -> "VFASettlementAggregate":
    """Portfolio-total paragraph-45 settlement in bounded memory.

    :func:`settle` materialises ``(n_mp, n_time)`` projection intermediates
    -- two forward projections over the whole book -- so a million-policy
    close would peak far beyond memory. Every line of the settlement
    movement is additive across contracts, so this runs :func:`settle`
    over row blocks of ``chunk_size`` model points and accumulates only the
    scalar line totals; peak memory is ``O(chunk_size x n_time)``
    regardless of ``n_mp``.

    Returns a :class:`~fastcashflow.movement.VFASettlementAggregate`: the
    movement's lines summed, movement-positive (``reconcile`` applies the
    display negation and reproduces the per-MP movement's table exactly).
    The aggregate cannot be chained -- ``closing_inputs()`` raises; chain
    per-MP movements instead. ``state`` joins ``model_points`` by mp_id
    once, before chunking, so a period-close file in its own row order
    never pairs one contract's rows with another's prior balances.
    """
    from fastcashflow.movement import (
        _VFA_SETTLEMENT_LINES, VFASettlementAggregate)

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period}")
    # One global mp_id join (and stale-snapshot check) BEFORE slicing, so a
    # chunk's model points and state rows always belong to the same
    # contracts; the per-chunk settle re-checks the aligned pair (a no-op).
    state = _reconcile_state(model_points, state)
    n_mp = model_points.n_mp
    # A per-MP fraction is sliced per chunk so the aggregate equals the per-MP
    # settle sum even when the split varies by contract (the per-chunk settle
    # re-validates the value range / finiteness).
    pe_frac = np.asarray(premium_experience_future_fraction, dtype=np.float64)
    if pe_frac.ndim > 1 or (pe_frac.ndim == 1 and pe_frac.shape[0] != n_mp):
        raise ValueError(
            "premium_experience_future_fraction must be a scalar or one entry "
            f"per model point ({n_mp}), got shape {pe_frac.shape}")
    # Per-chunk partial sums, combined with fsum so the total does not
    # depend on the chunking (compensated summation: chunk_size is a memory
    # knob, never a numbers knob).
    parts: dict[str, list[float]] = {n: [] for n in _VFA_SETTLEMENT_LINES}
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        frac_arg = (float(pe_frac) if pe_frac.ndim == 0 else pe_frac[idx])
        mv = settle(model_points.subset(idx), state.subset(idx), basis,
                    period_months=period,
                    premium_experience_future_fraction=frac_arg)
        for name in _VFA_SETTLEMENT_LINES:
            parts[name].append(float(getattr(mv, name).sum()))
    return VFASettlementAggregate(
        period_months=period, lock_in_rate=float(state.lock_in_rate),
        **{name: math.fsum(vals) for name, vals in parts.items()})


def settle_stream(
    input_path,
    output_dir,
    basis: Basis,
    *,
    calculation_methods=None,
    state_path=None,
    period_months: int | None = None,
    chunk_size: int = 200_000,
    id_column: str | None = None,
    validate_unique_mp_id: bool = True,
) -> int:
    """Stream a paragraph-45 period close through a parquet file, chunk by
    chunk.

    The out-of-core variant of :func:`settle` and the VFA counterpart of
    :func:`fastcashflow.gmm.settle_stream`. The VFA base is a single
    policies frame (account value + guarantee floors, no coverages), so
    ``input_path`` is read in ``chunk_size`` blocks, each block's
    ``(ModelPoints, InforceState)`` pair is settled, and the per-MP
    settlement movements land as one ``part-NNNNN.parquet`` per chunk under
    ``output_dir`` -- every movement line plus the ``measurement_basis``
    marker. Returns the model points processed.

    Input layouts, as in the GMM variant: ONE combined file (policies spec
    plus the closing-state columns, including the observed ``account_value``
    and the ``prior_count`` / ``prior_account_value`` /
    ``prior_loss_component`` figures :func:`settle` needs) or TWO files
    (policies parquet + ``state_path`` state parquet, semi-joined per chunk
    on ``mp_id`` with the global id sets validated bidirectionally).
    ``lock_in_rate`` must be uniform across the book (v1 scalar; the VFA
    carries it as a state echo only).

    **Chaining on disk**: each part carries ``count``, ``lock_in_rate``,
    ``elapsed_months``, ``account_value_closing`` and the closing balances,
    so the next period's state file is assembled from the parts alone
    (``prior_csm <- csm_closing``, ``prior_loss_component <-
    loss_component_closing``, ``prior_count <- count``,
    ``prior_account_value <- account_value_closing``, then advance to the
    next observation) -- the disk side of
    :meth:`VFASettlementMovement.closing_inputs()
    <fastcashflow.movement.VFASettlementMovement.closing_inputs>`.
    """
    from fastcashflow.io import _settle_stream_driver

    basis = _single_basis(basis, entry="vfa.settle_stream")
    return _settle_stream_driver(
        input_path, output_dir, state_path=state_path, chunk_size=chunk_size,
        id_column=id_column, validate_unique_mp_id=validate_unique_mp_id,
        build_mp=lambda spec: _vfa_model_points_from_frame(
            spec, calculation_methods),
        settle_fn=lambda mp, st: settle(mp, st, basis,
                                        period_months=period_months),
        entry="vfa.settle_stream",
    )


def recognition_schedule(
    model_points: ModelPoints,
    state: "InforceState",
    basis: Basis,
    *,
    band_edges_months=(12, 36, 60),
    period_months: int | None = None,
):
    """Paragraph-109 maturity-band disclosure for a settled VFA book.

    The VFA counterpart of :func:`~fastcashflow.gmm.recognition_schedule`:
    allocates the :func:`settle` closing CSM to maturity bands by each
    contract's forward coverage-unit (in-force) fraction, so the bands sum to
    the closing CSM -- when, in maturity terms, the remaining CSM is expected
    to be recognised in profit or loss. Onerous contracts carry no CSM and
    contribute nothing. ``band_edges_months`` are the band boundaries in months
    from the valuation date (default 12 / 36 / 60); ``period_months`` is the
    settlement period (default 12).
    """
    from fastcashflow.engine import (
        _validate_band_edges, _build_recognition_schedule)
    edges = _validate_band_edges(band_edges_months)
    mv = settle(model_points, state, basis, period_months=period_months)
    inforce = measure_vfa(model_points, basis, full=True).cashflows.inforce
    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    return _build_recognition_schedule(
        np.asarray(mv.csm_closing, dtype=np.float64), inforce, em, boundary,
        edges)


@dataclass(frozen=True, slots=True)
class GuaranteeTVOG:
    """Total time value of a VFA / universal-life book's guarantees.

    A participating account can carry two economically distinct guarantees that
    bite on disjoint regions of the account value, so their time values add:

    * ``credited_rate_floor`` -- the minimum-crediting-rate guarantee (the account
      is credited ``max(return, floor)`` each month), measured by
      :func:`measure_tvog`. It lifts the account value itself, realised across the
      account exits. Zero when the book carries no crediting guarantee.
    * ``account_floor`` -- the GMDB / GMAB account-value floors (a death pays
      ``max(account, GMDB)``, a maturity ``max(account, GMAB)``), measured through
      :func:`measure_vfa` and summed over the model points. They pay the SHORTFALL
      when the account falls below the guaranteed benefit.

    The crediting floor lifts the account from below the credited rate; the GMDB /
    GMAB floors top a benefit up when the account is short -- disjoint payoffs, so
    :attr:`total` is their sum, the book's full guarantee time value.
    """

    credited_rate_floor: float   # crediting-guarantee TVOG (portfolio)
    account_floor: float         # GMDB / GMAB floor TVOG (portfolio, summed over MP)

    @property
    def total(self) -> float:
        """The full guarantee time value -- crediting floor plus account floors."""
        return self.credited_rate_floor + self.account_floor


def guarantee_tvog(
    model_points: ModelPoints, basis: Basis, return_scenarios: FloatArray
) -> GuaranteeTVOG:
    """Total guarantee time value of a VFA / universal-life book.

    Sums the two guarantees a participating account can carry -- the
    minimum-crediting-rate floor (:func:`measure_tvog`) and the GMDB / GMAB
    account-value floors (:func:`measure_vfa`, summed over the model points).
    They bite on disjoint regions of the account value, so the sum is the book's
    full guarantee time value. A book with no crediting guarantee (every
    ``minimum_crediting_rate`` is :data:`NO_GUARANTEE_RATE`) contributes a zero
    crediting floor rather than raising. ``return_scenarios`` is the shared
    ``(n_scenarios, n_time)`` set of monthly underlying-items returns.
    """
    g = np.asarray(model_points.minimum_crediting_rate, dtype=np.float64)
    if g.size == 0 or np.all(g == NO_GUARANTEE_RATE):
        credited_rate_floor = 0.0
    else:
        credited_rate_floor = measure_tvog(
            model_points, basis, return_scenarios).time_value
    account_floor = float(np.sum(measure_vfa(
        model_points, basis, return_scenarios=return_scenarios).time_value))
    return GuaranteeTVOG(
        credited_rate_floor=credited_rate_floor, account_floor=account_floor)
