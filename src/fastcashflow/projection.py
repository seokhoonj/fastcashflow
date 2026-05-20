"""Monthly cash flow projection -- the BaseProj layer.

Sign convention (liability perspective, used consistently across the engine):

    premium_cf : insurer INFLOW  -- reduces the insurance liability
    claim_cf   : insurer OUTFLOW -- increases the insurance liability

Getting this convention consistent everywhere is the single most error-prone
part of a GMM engine, so it is stated once here and never re-decided.

Timing convention (monthly steps, month ``t`` spans ``[t, t+1)``):

    inforce[t]  : policies in force at the START of month t (per policy)
    premium     : charged at the start of month t, on inforce[t]
    deaths[t]   : occur during month t -- inforce[t] * monthly mortality
    lapses      : occur during month t, on the mortality survivors
    claim       : death benefit for deaths during month t
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.modelpoint import ModelPointSet


@dataclass(frozen=True, slots=True)
class CashflowProjection:
    """Projected monthly cash flows. Every array is shaped ``(n_mp, n_time)``."""

    inforce: FloatArray      # policies in force at the start of each month
    deaths: FloatArray       # deaths during each month
    premium_cf: FloatArray   # premium inflow per month
    claim_cf: FloatArray     # claim outflow per month

    @property
    def n_time(self) -> int:
        """Number of monthly projection steps."""
        return int(self.inforce.shape[1])


def project_cashflows(mps: ModelPointSet, asmp: Assumptions) -> CashflowProjection:
    """Project monthly cash flows for every model point.

    Vectorised over the model-point axis; the time axis is a sequential loop
    because the in-force recursion is genuinely sequential in time.
    """
    n_mp = mps.n_mp
    n_time = int(mps.term_months.max())     # months 0 .. n_time-1
    lapse = asmp.lapse_monthly

    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))

    # active[mp, t] is True while the contract is within its coverage term.
    active = np.arange(n_time)[None, :] < mps.term_months[:, None]
    inforce[:, 0] = 1.0                     # per-policy basis

    for t in range(n_time):
        attained_age = mps.issue_age + t // 12
        mortality = np.where(active[:, t], asmp.mortality_monthly(attained_age), 0.0)
        deaths[:, t] = inforce[:, t] * mortality
        if t + 1 < n_time:
            survivors = inforce[:, t] * (1.0 - mortality) * (1.0 - lapse)
            inforce[:, t + 1] = np.where(active[:, t], survivors, 0.0)

    premium_cf = inforce * mps.monthly_premium[:, None] * active
    claim_cf = deaths * mps.sum_assured[:, None]

    return CashflowProjection(
        inforce=inforce,
        deaths=deaths,
        premium_cf=premium_cf,
        claim_cf=claim_cf,
    )
