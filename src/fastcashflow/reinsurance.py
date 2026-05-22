"""IFRS 17 reinsurance contracts held -- a quota-share treaty.

A cedant buys reinsurance to transfer risk. This module measures a
proportional (quota-share) reinsurance contract held over a direct
portfolio: the cedant cedes a fixed fraction of its claims and pays the
same fraction of its premiums to the reinsurer.

IFRS 17 measures reinsurance held with the general model but with two
modifications (paragraphs 60-70):

* The risk adjustment is the amount of risk *transferred* to the reinsurer
  (paragraph 64) -- here, the margin on the ceded claims.
* There is no unearned profit; the CSM is instead the net cost or net gain
  of buying the cover (paragraph 65). So the CSM may be negative -- a net
  cost is deferred and amortised, not expensed -- and there is no loss
  component.

v1 scope: a single quota-share cession rate over the portfolio, with no
ceding commission. The reinsurer's non-performance risk and the
loss-recovery component (for onerous underlying contracts) are left for
later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.gmm import _csm_kernel, _norm_ppf, discount_factors
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows


@dataclass(frozen=True, slots=True)
class ReinsuranceMeasurement:
    """Measurement of a reinsurance contract held.

    ``bel`` and ``ra`` are ``(n_mp,)`` inception figures -- ``bel`` is the
    present value of reinsurance premiums less recoveries (a net cost when
    positive), ``ra`` is the risk transferred. ``csm`` is the
    ``(n_mp, n_time+1)`` trajectory of the net cost or gain of the cover,
    which may be negative; it reconciles as
    ``csm[:, t+1] = csm[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    """

    bel: FloatArray            # (n_mp,) -- PV(reins. premiums) - PV(recoveries)
    ra: FloatArray             # (n_mp,) -- risk transferred to the reinsurer
    csm: FloatArray            # (n_mp, n_time+1) -- net cost/gain trajectory
    csm_accretion: FloatArray  # (n_mp, n_time)
    csm_release: FloatArray    # (n_mp, n_time)
    recovery: FloatArray       # (n_mp, n_time) -- recoveries received
    reins_premium: FloatArray  # (n_mp, n_time) -- reinsurance premiums paid
    cashflows: Cashflows


def measure_reinsurance(
    model_points: ModelPoints, assumptions: Assumptions, cession_rate: float
) -> ReinsuranceMeasurement:
    """Measure a quota-share reinsurance contract held over a direct portfolio.

    ``cession_rate`` (in ``[0, 1]``) is the fraction of claims ceded; the
    cedant pays the same fraction of its premiums as reinsurance premium and
    recovers that fraction of its claims.

    The BEL is the present value of reinsurance premiums less recoveries; the
    RA is the margin on the ceded claims (the risk transferred). The CSM is
    ``-(BEL - RA)`` -- the net cost or gain of the cover -- and may be
    negative; it is accreted and released by coverage units like a direct
    contract's CSM, but with no loss component (paragraph 65).
    """
    if not 0.0 <= cession_rate <= 1.0:
        raise ValueError(f"cession_rate must be in [0, 1], got {cession_rate}")

    proj = project_cashflows(model_points, assumptions)
    discount_start, discount_mid = discount_factors(assumptions, proj.n_time)

    # The cedant cedes a fraction of claims (recovered) and of premiums (paid).
    recovery = cession_rate * (proj.claim_cf + proj.morbidity_cf)
    reins_premium = cession_rate * proj.premium_cf

    pv_recovery = (recovery * discount_mid).sum(axis=1)
    pv_reins_premium = (reins_premium * discount_start[:-1]).sum(axis=1)
    bel = pv_reins_premium - pv_recovery

    # RA -- the risk transferred, i.e. the margin on the ceded claims.
    z = _norm_ppf(assumptions.ra_confidence)
    pv_ceded_mortality = (cession_rate * proj.claim_cf * discount_mid).sum(axis=1)
    pv_ceded_morbidity = (cession_rate * proj.morbidity_cf * discount_mid).sum(axis=1)
    ra = z * (assumptions.mortality_cv * pv_ceded_mortality
              + assumptions.morbidity_cv * pv_ceded_morbidity)

    # CSM -- the net cost or gain of the cover. No loss component: a net cost
    # is a negative CSM, deferred and amortised over the coverage.
    csm0 = -(bel - ra)
    csm, csm_accretion, csm_release = _csm_kernel(
        csm0, proj.inforce, assumptions.discount_monthly
    )

    return ReinsuranceMeasurement(
        bel=bel,
        ra=ra,
        csm=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        recovery=recovery,
        reins_premium=reins_premium,
        cashflows=proj,
    )
