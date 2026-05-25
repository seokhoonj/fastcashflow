"""Shared numerical primitives -- numpy arrays in, numpy arrays out.

These functions know nothing about ``Assumptions`` / ``Cashflows`` / any
other domain dataclass. The orchestration layer (engine, PAA, VFA, movement)
unpacks domain objects to raw arrays + scalars and calls in here. The split
keeps these primitives numba-friendly and unit-testable on bare numpy.

The orchestration-specific ``@njit`` kernels (``_project_kernel`` in
projection.py, ``_value_kernel`` in engine.py) stay next to their callers --
they have only one call site each. The primitives below are the ones that
genuinely cross modules.

Contents:

* settlement-pattern helpers (``_settlement_lic``, ``_settlement_factor``)
* the standard-normal inverse CDF (``_norm_ppf``)
* the cost-of-capital RA accumulator (``_cost_of_capital_ra``)
* the BEL / RA / CSM time-loop kernels (``_rollforward_kernel``,
  ``_csm_kernel``)
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray


def _settlement_lic(
    incurred: FloatArray, settlement_pattern: FloatArray
) -> FloatArray:
    """Liability for incurred claims over a claims settlement pattern.

    ``incurred`` is the ``(n_mp, n_time)`` claims incurred each month;
    ``settlement_pattern`` is the run-off pattern, summing to 1. Returns the
    ``(n_mp, n_time+1)`` LIC trajectory -- claims build it up as incurred and
    run it off as paid -- held undiscounted.
    """
    incurred = np.asarray(incurred, dtype=np.float64)
    pattern = np.asarray(settlement_pattern, dtype=np.float64)
    if not np.isclose(pattern.sum(), 1.0):
        raise ValueError(f"settlement_pattern must sum to 1, got {pattern.sum()}")
    n_mp, n_time = incurred.shape
    paid = np.zeros_like(incurred)
    for k, weight in enumerate(pattern):
        if k < n_time:
            paid[:, k:] += weight * incurred[:, :n_time - k]
    lic = np.zeros((n_mp, n_time + 1))
    lic[:, 1:] = np.cumsum(incurred - paid, axis=1)
    return lic


def _settlement_factor(
    settlement_pattern: FloatArray, monthly_rate: float
) -> float:
    """Present-value factor for a claim spread over a settlement pattern.

    The present value, at the month a claim is incurred, of paying a unit
    claim over ``settlement_pattern`` -- discounted at ``monthly_rate``.
    A pattern that pays everything immediately gives 1.
    """
    pattern = np.asarray(settlement_pattern, dtype=np.float64)
    if not np.isclose(pattern.sum(), 1.0):
        raise ValueError(f"settlement_pattern must sum to 1, got {pattern.sum()}")
    months = np.arange(pattern.shape[0])
    return float(np.sum(pattern / (1.0 + monthly_rate) ** months))


# Coefficients of Acklam's rational approximation of the standard-normal
# inverse CDF -- the published constants of the algorithm.
_ACKLAM_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACKLAM_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
_ACKLAM_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACKLAM_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00)


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (quantile function).

    Rational-approximation algorithm (Acklam), accuracy ~1e-9. Implemented
    from the published algorithm; avoids a scipy dependency for a value the
    engine needs only once per run.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in the open interval (0, 1)")

    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return ((((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
                / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0))
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return ((((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
                / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0))
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return (-(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
            / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0))


def _cost_of_capital_ra(cl_margin, monthly_rate, coc_rate):
    """Cost-of-capital RA -- the cost of holding the confidence-level margin
    as non-financial-risk capital over the contract's run-off.

    The capital required at each future month is taken as the confidence-
    level margin there; the RA at month ``t`` is the cost-of-capital rate
    times the present value, at ``t``, of that capital over months ``t``
    onward. ``monthly_rate`` is the per-month rate curve, shape
    ``(n_time,)``; a flat rate and a yield curve share the same form.
    """
    full = 1.0 / (1.0 + monthly_rate)             # (n_time,)
    cap_pv = np.empty_like(cl_margin)
    cap_pv[:, -1] = cl_margin[:, -1]
    for t in range(cl_margin.shape[1] - 2, -1, -1):
        cap_pv[:, t] = cl_margin[:, t] + full[t] * cap_pv[:, t + 1]
    return coc_rate * cap_pv


@njit(parallel=True, cache=True)
def _rollforward_kernel(claim_cf, morbidity_cf, disability_cf, expense_cf,
                        premium_cf, annuity_cf, maturity_cf, surrender_cf,
                        term_months, monthly_rate):
    """Backward pass -- the BEL and the four RA present-value trajectories.

    ``BEL[t]`` is the present value, at month boundary ``t``, of the cash
    flows from month ``t`` onward, built by a backward recursion. Premiums
    and annuity payments fall at the start of the month, claims, expenses
    and surrender mid-month::

        BEL[t] = annuity[t] - premium[t]
                 + (claim[t] + morbidity[t] + disability[t]
                    + expense[t] + surrender[t]) * (1+i[t])^-0.5
                 + BEL[t+1] * (1+i[t])^-1

    ``monthly_rate`` is the per-month rate curve, shape ``(n_time,)``, so
    the locked-in rate can be flat or a yield curve. The maturity benefit
    is a single payment at ``t = term``, so it seeds ``BEL[term]``.
    ``BEL[:, 0]`` is then the inception BEL.

    Four more trajectories feed the Risk Adjustment, one per risk class:
    ``pv_claims`` (death claims -- mortality risk), ``pv_morbidity`` (health
    claims -- morbidity risk), ``pv_disability`` (disability income and the
    lump sum -- disability risk) and ``pv_survival`` (annuity payments and
    the maturity benefit -- longevity risk). All five trajectories have shape
    ``(n_mp, n_time+1)``.
    """
    n_mp, n_time = claim_cf.shape
    bel = np.zeros((n_mp, n_time + 1))
    pv_claims = np.zeros((n_mp, n_time + 1))
    pv_morbidity = np.zeros((n_mp, n_time + 1))
    pv_disability = np.zeros((n_mp, n_time + 1))
    pv_survival = np.zeros((n_mp, n_time + 1))

    half = (1.0 + monthly_rate) ** (-0.5)
    full = 1.0 / (1.0 + monthly_rate)

    for mp in prange(n_mp):
        term = term_months[mp]
        bel[mp, term] = maturity_cf[mp]
        pv_survival[mp, term] = maturity_cf[mp]
        for t in range(term - 1, -1, -1):
            claim = claim_cf[mp, t]
            morbidity = morbidity_cf[mp, t]
            disability = disability_cf[mp, t]
            annuity = annuity_cf[mp, t]
            surrender = surrender_cf[mp, t]
            bel[mp, t] = (
                annuity - premium_cf[mp, t]
                + (claim + morbidity + disability
                   + expense_cf[mp, t] + surrender) * half[t]
                + bel[mp, t + 1] * full[t]
            )
            pv_claims[mp, t] = claim * half[t] + pv_claims[mp, t + 1] * full[t]
            pv_morbidity[mp, t] = morbidity * half[t] + pv_morbidity[mp, t + 1] * full[t]
            pv_disability[mp, t] = disability * half[t] + pv_disability[mp, t + 1] * full[t]
            pv_survival[mp, t] = annuity + pv_survival[mp, t + 1] * full[t]

    return bel, pv_claims, pv_morbidity, pv_disability, pv_survival


@njit(parallel=True, cache=True)
def _csm_kernel(csm0, coverage_units, monthly_rate):
    """Compiled CSM roll-forward kernel -- raw numpy arrays only.

    Per model point (run in parallel across cores): interest accretion at the
    locked-in rate -- a per-month curve ``monthly_rate`` of length
    ``n_time``, so flat scalar and yield curve share the kernel -- then
    release proportional to coverage units. The coverage-unit tail sum is
    built in a single backward pass so the roll-forward stays linear in
    time. Monthly interest and release are returned too, so the roll-forward
    is fully decomposable: ``csm[t+1] = csm[t] + accretion[t] - release[t]``.
    """
    n_mp, n_time = coverage_units.shape
    csm = np.zeros((n_mp, n_time + 1))
    accretion = np.zeros((n_mp, n_time))
    release = np.zeros((n_mp, n_time))

    for mp in prange(n_mp):
        csm[mp, 0] = csm0[mp]

        cu_tail = np.empty(n_time)          # cu_tail[s] = sum of coverage_units[mp, s:]
        running = 0.0
        for s in range(n_time - 1, -1, -1):
            running += coverage_units[mp, s]
            cu_tail[s] = running

        for t in range(1, n_time + 1):
            interest = csm[mp, t - 1] * monthly_rate[t - 1]
            accreted = csm[mp, t - 1] + interest
            cu_remaining = cu_tail[t - 1]
            if cu_remaining > 0.0:
                rel = accreted * coverage_units[mp, t - 1] / cu_remaining
            else:
                rel = 0.0
            accretion[mp, t - 1] = interest
            release[mp, t - 1] = rel
            csm[mp, t] = accreted - rel

    return csm, accretion, release
