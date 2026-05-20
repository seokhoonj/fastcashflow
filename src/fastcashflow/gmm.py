"""IFRS 17 General Measurement Model -- BEL, RA and CSM.

References are to the IFRS 17 standard:

    BEL   estimates of future cash flows, discounted   (Sec. 33-35)
    RA    risk adjustment for non-financial risk       (Sec. 37)
    CSM   contractual service margin
            - initial recognition                      (Sec. 38)
            - subsequent measurement / roll-forward    (Sec. 44)
            - release by coverage units                (Sec. B119)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.projection import CashflowProjection


def discount_factors(asmp: Assumptions, n_time: int) -> FloatArray:
    """Monthly discount factors back to time 0. Shape ``(n_time,)``.

    Simplification: every month-t cash flow is discounted with the same
    factor ``(1 + i)^-t``, i.e. claims are treated as start-of-month.
    """
    t = np.arange(n_time)
    return (1.0 + asmp.discount_monthly) ** (-t)


def _pv(cashflow: FloatArray, discount: FloatArray) -> FloatArray:
    """Present value of a monthly cash flow stream, per model point.

    ``cashflow`` is ``(n_mp, n_time)`` and ``discount`` is ``(n_time,)``;
    the result is ``(n_mp,)``.
    """
    return (cashflow * discount).sum(axis=1)


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


def compute_bel(proj: CashflowProjection, discount: FloatArray) -> FloatArray:
    """Best Estimate of Liability, per model point. Shape ``(n_mp,)``.

    ``BEL = PV(claims) + PV(expenses) - PV(premiums)``. A negative BEL means
    the contract is profitable -- premium inflows outweigh the outflows.
    """
    return (_pv(proj.claim_cf, discount)
            + _pv(proj.expense_cf, discount)
            - _pv(proj.premium_cf, discount))


def compute_ra(
    proj: CashflowProjection,
    discount: FloatArray,
    ra_confidence: float,
    claims_cv: float,
) -> FloatArray:
    """Risk Adjustment, per model point. Shape ``(n_mp,)``.

    Confidence-level method: the RA is the margin that lifts the liability
    from its best estimate to the ``ra_confidence`` percentile, under a
    normal approximation::

        RA = z(ra_confidence) * claims_cv * PV(claims)

    where ``z`` is the standard-normal quantile. A cost-of-capital RA, which
    needs a capital projection, is left for a later phase.
    """
    z = _norm_ppf(ra_confidence)
    return z * claims_cv * _pv(proj.claim_cf, discount)


@dataclass(frozen=True, slots=True)
class CSMResult:
    """Outcome of the CSM measurement."""

    csm: FloatArray             # (n_mp, n_time+1) -- CSM at each month boundary
    release: FloatArray         # (n_mp, n_time)   -- CSM released each month
    loss_component: FloatArray  # (n_mp,)          -- loss component at inception


def compute_csm(
    bel: FloatArray,
    ra: FloatArray,
    proj: CashflowProjection,
    asmp: Assumptions,
) -> CSMResult:
    """CSM at initial recognition (Sec. 38) and deterministic roll-forward (Sec. 44).

    Fulfilment cash flows ``FCF = BEL + RA``.

    Initial recognition:
        ``CSM_0 = max(0, -FCF)``           -- profitable contract
        ``loss_component = max(0, FCF)``   -- onerous contract

    Roll-forward (deterministic -- no assumption changes):
        - interest accretion at the locked-in monthly rate
        - release proportional to coverage units (in-force is the coverage unit)
    """
    n_mp = bel.shape[0]
    n_time = proj.n_time
    monthly_rate = asmp.discount_monthly

    fcf = bel + ra                          # Fulfilment Cash Flows
    csm = np.zeros((n_mp, n_time + 1))
    release = np.zeros((n_mp, n_time))
    csm[:, 0] = np.maximum(0.0, -fcf)
    loss_component = np.maximum(0.0, fcf)

    coverage_units = proj.inforce           # shape (n_mp, n_time)
    # Tail sum of coverage units: cu_tail[:, t] == coverage_units[:, t:].sum(1).
    # Precomputed once in O(n) so the roll-forward loop stays linear, not O(n^2).
    cu_tail = np.cumsum(coverage_units[:, ::-1], axis=1)[:, ::-1]

    for t in range(1, n_time + 1):
        accreted = csm[:, t - 1] * (1.0 + monthly_rate)
        cu_now = coverage_units[:, t - 1]
        cu_remaining = cu_tail[:, t - 1]
        released_fraction = np.divide(
            cu_now, cu_remaining,
            out=np.zeros_like(cu_now),
            where=cu_remaining > 0.0,
        )
        release[:, t - 1] = accreted * released_fraction
        csm[:, t] = accreted - release[:, t - 1]

    return CSMResult(csm=csm, release=release, loss_component=loss_component)
