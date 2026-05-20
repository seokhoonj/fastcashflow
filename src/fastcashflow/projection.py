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

Two layers: a compiled, parallel kernel (``_project_kernel``) runs the raw
time loop; a Pythonic wrapper (``project_cashflows``) prepares its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.modelpoint import ModelPointSet


@dataclass(frozen=True, slots=True)
class Cashflows:
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


@njit(parallel=True, cache=True)
def _project_kernel(rates_by_year, term_months, lapse_by_year, monthly_premium,
                    sum_assured, expense_acquisition, maint_monthly,
                    inflation, n_time):
    """Compiled, parallel time-loop kernel -- raw numpy arrays and scalars only.

    The model-point axis is the independent (outer) loop, run in parallel
    across cores; the time axis is the sequential (inner) loop, because the
    in-force recursion depends on the previous month.

    Mortality and lapse are supplied per policy year (``rates_by_year``,
    ``lapse_by_year``); both change only once every twelve months.
    """
    n_mp = rates_by_year.shape[0]
    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))
    premium_cf = np.zeros((n_mp, n_time))
    claim_cf = np.zeros((n_mp, n_time))
    expense_cf = np.zeros((n_mp, n_time))

    for mp in prange(n_mp):
        term = term_months[mp]
        inforce[mp, 0] = 1.0
        for t in range(term):
            ift = inforce[mp, t]
            year = t // 12
            q = rates_by_year[mp, year]
            deaths[mp, t] = ift * q
            premium_cf[mp, t] = ift * monthly_premium[mp]
            claim_cf[mp, t] = ift * q * sum_assured[mp]
            acquisition = expense_acquisition if t == 0 else 0.0
            expense_cf[mp, t] = acquisition + ift * maint_monthly * inflation[t]
            if t + 1 < term:
                inforce[mp, t + 1] = ift * (1.0 - q) * (1.0 - lapse_by_year[year])

    return inforce, deaths, premium_cf, claim_cf, expense_cf


def project_cashflows(mps: ModelPointSet, asmp: Assumptions) -> Cashflows:
    """Project monthly cash flows for every model point.

    The Pythonic wrapper: it extracts raw arrays from the inputs and
    evaluates the assumptions. Mortality and lapse are evaluated on the
    per-policy-year grid, not the full ``(n_mp, n_time)`` grid -- both
    change only once a year, so this is an identical result for a twelfth
    of the work.
    """
    n_time = int(mps.term_months.max())     # months 0 .. n_time-1
    n_years = (n_time + 11) // 12
    months = np.arange(n_time)
    durations = np.arange(n_years)

    issue_age_grid, duration_grid = np.meshgrid(
        mps.issue_age, durations, indexing="ij"
    )
    rates_by_year = np.ascontiguousarray(
        asmp.mortality_monthly(issue_age_grid, duration_grid), dtype=np.float64
    )
    lapse_by_year = np.ascontiguousarray(
        asmp.lapse_monthly(durations), dtype=np.float64
    )
    inflation = (1.0 + asmp.expense_inflation) ** (months / 12.0)

    inforce, deaths, premium_cf, claim_cf, expense_cf = _project_kernel(
        rates_by_year,
        mps.term_months,
        lapse_by_year,
        mps.monthly_premium,
        mps.sum_assured,
        asmp.expense_acquisition,
        asmp.expense_maintenance_annual / 12.0,
        inflation,
        n_time,
    )
    return Cashflows(
        inforce=inforce,
        deaths=deaths,
        premium_cf=premium_cf,
        claim_cf=claim_cf,
        expense_cf=expense_cf,
    )
