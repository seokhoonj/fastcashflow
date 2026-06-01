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

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.numerics import (
    _csm_kernel,
    _norm_ppf,
    _settlement_factor,
    _settlement_lic,
)
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows
from fastcashflow.tvog import guarantee_floor_time_value, tvog_weights


@dataclass(frozen=True, slots=True)
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

    account_value: FloatArray    # (n_mp, n_time+1) -- account-value trajectory
    bel: FloatArray              # (n_mp, n_time+1) -- BEL trajectory
    ra: FloatArray               # (n_mp, n_time+1) -- RA trajectory (expense risk)
    csm: FloatArray              # (n_mp, n_time+1) -- CSM trajectory
    csm_accretion: FloatArray    # (n_mp, n_time)   -- CSM accreted each month
    csm_release: FloatArray      # (n_mp, n_time)   -- CSM released each month
    variable_fee: FloatArray     # (n_mp,)          -- PV of the entity's fee
    loss_component: FloatArray   # (n_mp,)          -- onerous loss at inception
    time_value: FloatArray       # (n_mp,)          -- guarantee TVOG at inception
    lic: FloatArray              # (n_mp, n_time+1) -- liability for incurred claims
    discount_start: FloatArray   # (n_time+1,)      -- start-of-month discount factors
    cashflows: Cashflows


def measure_vfa(
    model_points: ModelPoints,
    basis: Basis,
    return_scenarios: FloatArray | None = None,
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

    BEL, RA and CSM are returned as month-by-month trajectories. The
    deterministic BEL carries the guarantee's intrinsic value only; when
    ``return_scenarios`` -- an ``(n_scenarios, n_time)`` array of monthly
    underlying-items returns -- is supplied, the time value of the guarantee
    enters the inception fulfilment cash flows too, so the CSM absorbs it,
    and ``time_value`` records that amount per model point.
    """
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

    # Account-value trajectory -- per-policy closed form.
    periods = np.arange(n_time + 1)
    av = model_points.account_value[:, None] * (
        growth[:, None] ** periods[None, :]
    )

    # Every policy eventually exits and receives its account value -- except
    # that a death exit pays max(account value, guaranteed minimum death
    # benefit). ``deaths`` is the mortality portion of the decrement; the
    # remainder (lapse, maturity) takes the account value. With the default
    # zero GDB this reduces exactly to ``exits * av`` (max(AV, 0) = AV).
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
    # excess over the account value there. Default zero GAB adds nothing.
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
    variable_fee = (fee_cf * disc_mid).sum(axis=1)

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
    # The inception fulfilment cash flows -- with the guarantee time value --
    # drive the CSM and the loss component.
    fcf = bel[:, 0] + ra[:, 0] + time_value
    loss_component = np.maximum(0.0, fcf)
    csm0 = np.maximum(0.0, -fcf)
    # VFA accretes at the underlying-items return -- flat across time in
    # the deterministic measurement; broadcast to the per-month curve the
    # kernel consumes.
    csm, csm_accretion, csm_release = _csm_kernel(
        csm0, inforce, np.full(n_time, r_m),
    )

    return VFAMeasurement(
        account_value=av,
        bel=bel,
        ra=ra,
        csm=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        variable_fee=variable_fee,
        loss_component=loss_component,
        time_value=time_value,
        lic=lic,
        discount_start=disc_start,
        cashflows=proj,
    )
