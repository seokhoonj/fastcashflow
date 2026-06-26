"""Model-neutral valuation core -- project cash flows into the BEL / RA bundle.

The prefix every GMM-family model shares: :func:`valued_projection` projects the
cash flows and rolls them into a :class:`ValuedProjection` (BEL / RA trajectories
plus discount context, NO CSM, no model identity). Each model then assembles its
own measurement (CSM / LRC) on top. Extracted from the GMM engine so VFA / PAA /
reinsurance share this core directly rather than borrowing from a GMM module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.curves import (
    discount_factors_from_curve,
    discount_monthly_curve,
)
from fastcashflow._numerics import (
    _cost_of_capital_ra,
    _norm_ppf,
    _risk_adjustment,
    _roll_forward_kernel,
    _settlement_factor,
    _settlement_lic,
)
from fastcashflow.model_points import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows


def _account_risk_adjustment(model_points, basis, proj, discount_monthly):
    """Universal-life risk adjustment -- priced on the net amount at risk.

    The insurance risk of an account-backed death leg is the mortality borne on
    the NET AMOUNT AT RISK (the death benefit above the account,
    ``deaths * max(0, face - av_mid)``) -- the account portion returns the
    policyholder's own money and bears no insurance risk -- plus expense risk,
    plus the morbidity risk of any cost-deducting rider (a fixed health benefit
    funded from the account). This BYPASSES :func:`_risk_adjustment` and its
    ``expense_cv != 0`` guard (a UL RA legitimately prices ``expense_cv``): run
    the at-risk claim, the morbidity claim and the expense through one
    roll-forward pass, then the confidence margin ``z(ra_confidence) *
    (mortality_cv*pv_nar + morbidity_cv*pv_morbidity + expense_cv*pv_expense)``,
    cost-of-capital-wrapped per ``ra_method``.
    """
    face = model_points.minimum_death_benefit
    n_mp, n_time = proj.mortality_cf.shape
    zeros_t = np.zeros((n_mp, n_time))
    zeros_mp = np.zeros(n_mp)
    nar_claim = np.ascontiguousarray(
        proj.deaths * np.maximum(0.0, face[:, None] - proj.account.av_mid))
    # The annuity payout (an annuitizing UL contract, phase 2) bears longevity
    # risk -- the insurer pays the income for as long as the annuitant lives --
    # so its PV is priced through longevity_cv, alongside the at-risk mortality
    # and expense. The annuity stream rides the survival slot of the
    # roll-forward (position 6); a non-annuitizing account book has annuity_cf
    # == 0, so pv_annuity == 0 and this term vanishes (byte-identical). The
    # account maturity lump is the return of the policyholder's own balance (an
    # investment component) and bears no insurance risk, so it is deliberately
    # NOT longevity-priced (it stays out, the maturity slot is zero here).
    # The morbidity claim of a cost-deducting rider (funds from the account, but
    # pays a fixed health benefit -- not the balance) bears morbidity risk; it
    # rides the DISABILITY slot of the roll-forward (position 3, otherwise empty
    # for an account book) purely to harvest its PV. A book with no such rider
    # has morbidity_cf == 0, so pv_morbidity == 0 and the term vanishes
    # (byte-identical). expense_cf rides the morbidity slot for the same reason.
    _, pv_nar, pv_expense, pv_morbidity, pv_annuity = _roll_forward_kernel(
        nar_claim, proj.expense_cf, proj.morbidity_cf, zeros_t, zeros_t,
        proj.annuity_cf, zeros_mp, zeros_t,
        model_points.contract_boundary_months, discount_monthly)
    z = _norm_ppf(basis.ra_confidence)
    confidence_margin = z * (basis.mortality_cv * pv_nar
                             + basis.morbidity_cv * pv_morbidity
                             + basis.expense_cv * pv_expense
                             + basis.longevity_cv * pv_annuity)
    if basis.ra_method == "cost_of_capital":
        return _cost_of_capital_ra(
            confidence_margin, discount_monthly, basis.cost_of_capital_rate)
    return confidence_margin


@dataclass(frozen=True, slots=True, eq=False)
class ValuedProjection:
    """Neutral valuation bundle -- the model-agnostic prefix of a full
    measurement.

    The projected cash flows valued into BEL / RA trajectories plus the discount
    context, with NO CSM and NO model identity. Produced by
    :func:`valued_projection` (downstream of the cash-flow projection). Each
    model's full measurement assembles its own result from this bundle plus its
    own CSM / LRC machinery, so no model borrows another's measurement
    container. ``bel`` / ``ra`` are the ``(n_mp,)`` inception headline (column 0
    of the trajectories).
    """

    bel_path: FloatArray              # (n_mp, n_time+1) -- BEL trajectory
    ra_path: FloatArray               # (n_mp, n_time+1) -- RA trajectory
    lic_path: FloatArray              # (n_mp, n_time+1) -- liability for incurred claims
    discount_factor_bom: FloatArray   # beginning-of-month discount factors
    discount_factor_mid: FloatArray   # mid-of-month discount factors
    discount_monthly: FloatArray      # per-month discount / CSM-accretion rate curve
    cashflows: Cashflows              # the underlying projection

    @property
    def bel(self) -> FloatArray:
        return self.bel_path[:, 0]

    @property
    def ra(self) -> FloatArray:
        return self.ra_path[:, 0]


def valued_projection(model_points: ModelPoints, basis: Basis, *,
                      discount_monthly: FloatArray | None = None,
                      lapse_scale: FloatArray | None = None) -> ValuedProjection:
    """Value a cash-flow projection into the neutral BEL / RA bundle.

    The model-agnostic core of a full measurement: project the cash flows, then
    roll them forward into the BEL trajectory and price the RA, returning a
    :class:`ValuedProjection` (no CSM, no model identity). This is the prefix
    every GMM-family model shares; each model then adds its own CSM / LRC on top.

    ``discount_monthly`` overrides the discount / CSM-accretion curve (default:
    the locked-in ``discount_monthly_curve``). ``vfa.measure`` passes the flat
    underlying-items return here to value a universal-life account book under the
    VFA model -- the account roll (generation) is identical to GMM, only the
    discount rate differs. The override is only used by the account path, which
    carries no ``settlement_pattern``, so the settlement factor below (keyed on
    ``basis.discount_monthly``) is never reached together with an override.
    """
    proj = project_cashflows(model_points, basis, lapse_scale=lapse_scale)
    mortality_cf, morbidity_cf = proj.mortality_cf, proj.morbidity_cf
    if discount_monthly is None:
        discount_monthly = discount_monthly_curve(basis, proj.n_time)
    if basis.settlement_pattern is None:
        lic_path = np.zeros((mortality_cf.shape[0], proj.n_time + 1))
    else:
        lic_path = _settlement_lic(mortality_cf + morbidity_cf, basis.settlement_pattern)
        # Claims are paid over the pattern, not at incurrence -- discount
        # them to their payment dates in the fulfilment cash flows. With a
        # discount curve we use the in-year scalar (paragraph 40 / B71 -- the
        # rate at the month of incurrence is the right reference); the
        # full-curve treatment would require a time-varying settlement
        # factor inside the kernel, deferred.
        factor = _settlement_factor(basis.settlement_pattern, basis.discount_monthly)
        mortality_cf = mortality_cf * factor
        morbidity_cf = morbidity_cf * factor
    discount_factor_bom, discount_factor_mid = discount_factors_from_curve(discount_monthly)

    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _roll_forward_kernel(
        mortality_cf, morbidity_cf, proj.disability_cf, proj.expense_cf,
        proj.premium_cf, proj.annuity_cf, proj.maturity_cf, proj.surrender_cf,
        model_points.contract_boundary_months, discount_monthly,
    )
    if proj.account is not None:
        # Universal-life account-backed measurement. The BEL nets the account
        # value the entity holds (fund) -- premium is the lone gross inflow
        # (counted once in the roll-forward), and the account it builds is held
        # as fund and subtracted ONCE post-PV. The RA prices the mortality risk
        # on the NET AMOUNT AT RISK (the death benefit above the account) plus
        # expense risk, bypassing the slot-RA machinery (which hard-raises on
        # expense_cv and would price mortality on the full death benefit).
        bel = bel - proj.account.fund
        ra = _account_risk_adjustment(model_points, basis, proj, discount_monthly)
    else:
        pv_survival_ra = pv_survival
        ac = proj.annuity_certain_cf
        if ac is not None and np.any(ac != 0.0):
            # The guaranteed (certain) annuity payments are paid regardless of
            # survival, so they carry no longevity risk -- remove their PV from
            # the survival PV that feeds the longevity RA (the BEL still includes
            # them via the full annuity_cf). A second roll-forward over the
            # certain stream alone (everything else zero) yields its PV in the
            # pv_survival slot, with the kernel's exact start-of-month discount.
            zero = np.zeros_like(ac)
            zero_mat = np.zeros(ac.shape[0])
            _, _, _, _, pv_certain = _roll_forward_kernel(
                zero, zero, zero, zero, zero, ac, zero_mat, zero,
                model_points.contract_boundary_months, discount_monthly,
            )
            pv_survival_ra = pv_survival - pv_certain
        ra = _risk_adjustment(basis, pv_claims, pv_morbidity, pv_disability,
                              pv_survival_ra, discount_monthly)
    return ValuedProjection(
        bel_path=bel,
        ra_path=ra,
        lic_path=lic_path,
        discount_factor_bom=discount_factor_bom,
        discount_factor_mid=discount_factor_mid,
        discount_monthly=discount_monthly,
        cashflows=proj,
    )
