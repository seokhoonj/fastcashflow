"""IFRS 17 Variable Fee Approach (VFA) -- direct-participation contracts.

The VFA is IFRS 17's measurement model for insurance contracts with direct
participation features -- contracts where the policyholder's benefit is a
share of a pool of *underlying items* (a fund). It is the model for
unit-linked and with-profits business.

This module measures a single-premium account-value contract: a premium is
paid into an account at issue; the account value grows at the underlying-
items return less a variable fee; and the benefit -- paid on death,
surrender or maturity alike -- is the account value at that time. The
entity's profit is the *variable fee* it deducts, its share of the
underlying items.

Under the VFA the financial result flows through the CSM rather than profit
or loss, so the account-value cash flows are discounted, and the CSM is
accreted, at the underlying-items return -- not a locked-in rate.
fastcashflow is deterministic (a single scenario), so the VFA's hallmark --
the CSM absorbing the variability of the underlying items -- reduces here to
that return-rate accretion. Guarantees and surrender penalties, where the
real non-financial risk of VFA business sits, are left for a later phase,
so the v1 RA is zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.gmm import _csm_kernel
from fastcashflow.modelpoint import ModelPointSet
from fastcashflow.projection import Cashflows, project_cashflows


@dataclass(frozen=True, slots=True)
class VFAMeasurement:
    """VFA measurement of a direct-participation (account-value) portfolio.

    ``account_value`` is the ``(n_mp, n_time+1)`` account-value trajectory.
    ``bel`` and ``loss_component`` are ``(n_mp,)`` inception figures. ``csm``
    is the ``(n_mp, n_time+1)`` trajectory, accreted at the underlying-items
    return and released by coverage units::

        csm[:, t+1] = csm[:, t] + csm_accretion[:, t] - csm_release[:, t]

    ``variable_fee`` is the present value of the entity's fee -- its share
    of the underlying items.
    """

    account_value: FloatArray    # (n_mp, n_time+1) -- account-value trajectory
    bel: FloatArray              # (n_mp,)          -- inception BEL
    csm: FloatArray              # (n_mp, n_time+1) -- CSM trajectory
    csm_accretion: FloatArray    # (n_mp, n_time)   -- CSM accreted each month
    csm_release: FloatArray      # (n_mp, n_time)   -- CSM released each month
    variable_fee: FloatArray     # (n_mp,)          -- PV of the entity's fee
    loss_component: FloatArray   # (n_mp,)          -- onerous loss at inception
    cashflows: Cashflows


def measure_vfa(mps: ModelPointSet, asmp: Assumptions) -> VFAMeasurement:
    """Measure a direct-participation portfolio under the Variable Fee Approach.

    The account value rolls forward as ``AV[t+1] = AV[t] * (1 + r) * (1 - f)``
    -- the underlying-items return ``r`` less the variable fee ``f`` -- from
    ``AV[0]`` = the model point's ``account_value``. The benefit on every exit
    (death, surrender, maturity) is the account value at that time.

    BEL is the present value of benefits and expenses less the premium, all
    at the underlying-items return; the CSM is ``max(0, -BEL)`` -- the
    entity's unearned variable fee -- accreted at the same return and
    released by coverage units.
    """
    proj = project_cashflows(mps, asmp)
    inforce = proj.inforce
    n_mp, n_time = inforce.shape

    r_m = (1.0 + asmp.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + asmp.fund_fee) ** (1.0 / 12.0) - 1.0
    growth = (1.0 + r_m) * (1.0 - f_m)

    # Account-value trajectory -- flat rates, so a closed form.
    av = mps.account_value[:, None] * growth ** np.arange(n_time + 1)[None, :]

    # Every policy eventually exits and receives its account value.
    inforce_pad = np.concatenate([inforce, np.zeros((n_mp, 1))], axis=1)
    exits = inforce_pad[:, :-1] - inforce_pad[:, 1:]      # (n_mp, n_time)
    benefit_cf = exits * av[:, :n_time]
    # Variable fee -- the entity's share, deducted from the grown account value.
    fee_cf = inforce * av[:, :n_time] * (1.0 + r_m) * f_m

    # Discount at the underlying-items return -- the VFA basis. Benefits are
    # discounted start-of-month, consistent with the account value, so a
    # zero fee leaves no profit.
    base = 1.0 + r_m
    disc_start = base ** (-np.arange(n_time))
    disc_mid = base ** (-(np.arange(n_time) + 0.5))

    pv_benefits = (benefit_cf * disc_start).sum(axis=1)
    pv_expenses = (proj.expense_cf * disc_mid).sum(axis=1)
    variable_fee = (fee_cf * disc_mid).sum(axis=1)

    bel = pv_benefits + pv_expenses - mps.account_value
    loss_component = np.maximum(0.0, bel)                 # RA is zero in v1
    csm0 = np.maximum(0.0, -bel)
    csm, csm_accretion, csm_release = _csm_kernel(csm0, inforce, r_m)

    return VFAMeasurement(
        account_value=av,
        bel=bel,
        csm=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        variable_fee=variable_fee,
        loss_component=loss_component,
        cashflows=proj,
    )
