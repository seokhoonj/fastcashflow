"""Monthly cash flow projection -- the BaseProj layer.

Sign convention (liability perspective, used consistently across the engine):

    premium_cf : insurer INFLOW  -- reduces the insurance liability
    claim_cf   : insurer OUTFLOW -- increases the insurance liability
    expense_cf : insurer OUTFLOW -- increases the insurance liability
    annuity_cf : insurer OUTFLOW -- increases the insurance liability
    maturity_cf: insurer OUTFLOW -- increases the insurance liability

Getting this convention consistent everywhere is the single most error-prone
part of a GMM engine, so it is stated once here and never re-decided.

Timing convention (monthly steps, month ``t`` spans ``[t, t+1)``):

    inforce[t]  : policies in force at the START of month t (per policy)
    premium     : charged at the start of month t, on inforce[t]
                  (the single premium, if any, is added at t = 0)
    annuity     : paid at the start of month t, on inforce[t]
    deaths[t]   : occur during month t -- inforce[t] * monthly mortality
    lapses      : occur during month t, on the mortality survivors
    claim       : sum of the policy's coverages for deaths during month t
    expense     : acquisition at t = 0; maintenance every in-force month
    maturity    : maturity benefit at time = term, paid to the survivors

Two layers: a compiled, parallel kernel (``_project_kernel``) runs the raw
time loop; a Pythonic wrapper (``project_cashflows``) prepares its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.coverage import coverage_rates
from fastcashflow.modelpoint import ModelPointSet


@dataclass(frozen=True, slots=True)
class Cashflows:
    """Projected cash flows.

    The per-month arrays are shaped ``(n_mp, n_time)``; ``maturity_cf`` is
    ``(n_mp,)`` -- one payment per policy, at that policy's term.
    """

    inforce: FloatArray      # policies in force at the start of each month
    deaths: FloatArray       # deaths during each month
    premium_cf: FloatArray   # premium inflow per month (single premium at t=0)
    claim_cf: FloatArray     # death-benefit outflow per month (all coverages)
    expense_cf: FloatArray   # expense outflow per month
    annuity_cf: FloatArray   # annuity (survival income) outflow per month
    maturity_cf: FloatArray  # (n_mp,) maturity benefit, paid at time = term

    @property
    def n_time(self) -> int:
        """Number of monthly projection steps."""
        return int(self.inforce.shape[1])


@njit(parallel=True, cache=True)
def _project_kernel(mortality, term_months, lapse_by_year, monthly_premium,
                    single_premium, cov_kind, cov_amount, cov_offset, cov_rates,
                    maturity_benefit, annuity_payment, expense_acquisition,
                    maint_monthly, inflation, n_time):
    """Compiled, parallel time-loop kernel -- raw numpy arrays and scalars only.

    The model-point axis is the independent (outer) loop, run in parallel
    across cores; the time axis is the sequential (inner) loop, because the
    in-force recursion depends on the previous month.

    Each policy's death claim is the sum over its coverage list: coverage
    ``k`` pays ``cov_amount[k]`` at rate ``cov_rates[cov_kind[k], mp, year]``.
    Mortality (the decrement) and lapse are supplied per policy year, both
    changing only once every twelve months. The maturity benefit is paid to
    the in-force survivors at time = term.
    """
    n_mp = mortality.shape[0]
    inforce = np.zeros((n_mp, n_time))
    deaths = np.zeros((n_mp, n_time))
    premium_cf = np.zeros((n_mp, n_time))
    claim_cf = np.zeros((n_mp, n_time))
    expense_cf = np.zeros((n_mp, n_time))
    annuity_cf = np.zeros((n_mp, n_time))
    maturity_cf = np.zeros(n_mp)

    for mp in prange(n_mp):
        term = term_months[mp]
        c_start = cov_offset[mp]
        c_end = cov_offset[mp + 1]
        inforce[mp, 0] = 1.0
        last_year = -1
        claim_rate = 0.0      # aggregate claim per unit in-force, current year
        for t in range(term):
            ift = inforce[mp, t]
            year = t // 12
            # Coverage rates change only once a year, so the per-coverage sum
            # is rebuilt on a year boundary, not every month.
            if year != last_year:
                claim_rate = 0.0
                for k in range(c_start, c_end):
                    claim_rate += cov_rates[cov_kind[k], mp, year] * cov_amount[k]
                last_year = year
            q = mortality[mp, year]
            deaths[mp, t] = ift * q
            single = single_premium[mp] if t == 0 else 0.0
            premium_cf[mp, t] = ift * monthly_premium[mp] + single
            claim_cf[mp, t] = ift * claim_rate
            annuity_cf[mp, t] = ift * annuity_payment[mp]
            acquisition = expense_acquisition if t == 0 else 0.0
            expense_cf[mp, t] = acquisition + ift * maint_monthly * inflation[t]
            survivors = ift * (1.0 - q) * (1.0 - lapse_by_year[year])
            if t + 1 < term:
                inforce[mp, t + 1] = survivors
            else:
                maturity_cf[mp] = survivors * maturity_benefit[mp]

    return (inforce, deaths, premium_cf, claim_cf, expense_cf, annuity_cf,
            maturity_cf)


def project_cashflows(mps: ModelPointSet, asmp: Assumptions) -> Cashflows:
    """Project cash flows for every model point.

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
    mortality = np.ascontiguousarray(
        asmp.mortality_monthly(issue_age_grid, duration_grid), dtype=np.float64
    )
    lapse_by_year = np.ascontiguousarray(
        asmp.lapse_monthly(durations), dtype=np.float64
    )
    cov_rates = coverage_rates(mortality)
    inflation = (1.0 + asmp.expense_inflation) ** (months / 12.0)

    (inforce, deaths, premium_cf, claim_cf, expense_cf, annuity_cf,
     maturity_cf) = _project_kernel(
        mortality,
        mps.term_months,
        lapse_by_year,
        mps.monthly_premium,
        mps.single_premium,
        mps.cov_kind,
        mps.cov_amount,
        mps.cov_offset,
        cov_rates,
        mps.maturity_benefit,
        mps.annuity_payment,
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
        annuity_cf=annuity_cf,
        maturity_cf=maturity_cf,
    )
