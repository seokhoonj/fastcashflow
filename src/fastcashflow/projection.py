"""Monthly cash flow projection -- the BaseProj layer.

Sign convention (liability perspective, used consistently across the engine):

    premium_cf : insurer INFLOW  -- reduces the insurance liability
    claim_cf   : insurer OUTFLOW -- increases the insurance liability
    expense_cf : insurer OUTFLOW -- increases the insurance liability

Getting this convention consistent everywhere is the single most error-prone
part of a GMM engine, so it is stated once here and never re-decided.

Timing convention (monthly steps, month ``t`` spans ``[t, t+1)``):

    inforce[t]  : policies in force at the START of month t (per policy)
    premium     : charged at the start of month t, on inforce[t]
    deaths[t]   : occur during month t -- inforce[t] * monthly mortality
    lapses      : occur during month t, on the mortality survivors
    claim       : death benefit for deaths during month t
    expense     : acquisition at t = 0; maintenance every in-force month

Two layers: a compiled kernel (``_project_kernel``) does the raw time loop;
a Pythonic wrapper (``project_cashflows``) prepares its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

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
    expense_cf: FloatArray   # expense outflow per month

    @property
    def n_time(self) -> int:
        """Number of monthly projection steps."""
        return int(self.inforce.shape[1])


@njit(cache=True)
def _project_kernel(mortality, term_months, lapse, monthly_premium, sum_assured,
                    expense_acquisition, maint_monthly, inflation):
    """Compiled time-loop kernel -- raw numpy arrays and scalars only.

    The model-point axis is the independent (outer) loop; the time axis is
    the sequential (inner) loop, because the in-force recursion depends on
    the previous month.
    """
    n_mp, n_time = mortality.shape
    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))
    premium_cf = np.zeros((n_mp, n_time))
    claim_cf = np.zeros((n_mp, n_time))
    expense_cf = np.zeros((n_mp, n_time))

    for mp in range(n_mp):
        term = term_months[mp]
        inforce[mp, 0] = 1.0
        for t in range(term):
            ift = inforce[mp, t]
            q = mortality[mp, t]
            deaths[mp, t] = ift * q
            premium_cf[mp, t] = ift * monthly_premium[mp]
            claim_cf[mp, t] = ift * q * sum_assured[mp]
            acquisition = expense_acquisition if t == 0 else 0.0
            expense_cf[mp, t] = acquisition + ift * maint_monthly * inflation[t]
            if t + 1 < term:
                inforce[mp, t + 1] = ift * (1.0 - q) * (1.0 - lapse)

    return inforce, deaths, premium_cf, claim_cf, expense_cf


def project_cashflows(mps: ModelPointSet, asmp: Assumptions) -> CashflowProjection:
    """Project monthly cash flows for every model point.

    The Pythonic wrapper: it extracts raw arrays from the inputs, evaluates
    the mortality assumption once, and hands everything to the kernel.
    """
    n_time = int(mps.term_months.max())     # months 0 .. n_time-1
    months = np.arange(n_time)

    # Evaluate mortality once, outside the kernel, as an (n_mp, n_time) array.
    attained_age = mps.issue_age[:, None] + (months // 12)[None, :]
    mortality = np.ascontiguousarray(
        asmp.mortality_monthly(attained_age), dtype=np.float64
    )
    inflation = (1.0 + asmp.expense_inflation) ** (months / 12.0)

    inforce, deaths, premium_cf, claim_cf, expense_cf = _project_kernel(
        mortality,
        mps.term_months,
        asmp.lapse_monthly,
        mps.monthly_premium,
        mps.sum_assured,
        asmp.expense_acquisition,
        asmp.expense_maintenance_annual / 12.0,
        inflation,
    )
    return CashflowProjection(
        inforce=inforce,
        deaths=deaths,
        premium_cf=premium_cf,
        claim_cf=claim_cf,
        expense_cf=expense_cf,
    )
