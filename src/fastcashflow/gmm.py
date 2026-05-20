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

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.projection import CashflowProjection


def discount_factors(asmp: Assumptions, n_time: int) -> FloatArray:
    """Monthly discount factors back to time 0. Shape ``(n_time,)``.

    Phase 0 simplification: every month-t cash flow is discounted with the
    same factor ``(1 + i)^-t``, i.e. claims are treated as start-of-month.
    """
    t = np.arange(n_time)
    return (1.0 + asmp.discount_monthly) ** (-t)


def _pv(cashflow: FloatArray, discount: FloatArray) -> FloatArray:
    """Present value of a monthly cash flow stream, per model point.

    ``cashflow`` is ``(n_mp, n_time)`` and ``discount`` is ``(n_time,)``;
    the result is ``(n_mp,)``.
    """
    return (cashflow * discount).sum(axis=1)


def compute_bel(proj: CashflowProjection, discount: FloatArray) -> FloatArray:
    """Best Estimate of Liability, per model point. Shape ``(n_mp,)``.

    ``BEL = PV(claim outflow) - PV(premium inflow)``. A negative BEL means
    the contract is profitable -- premium inflows outweigh claim outflows.
    """
    return _pv(proj.claim_cf, discount) - _pv(proj.premium_cf, discount)


def compute_ra(proj: CashflowProjection, discount: FloatArray, ra_rate: float) -> FloatArray:
    """Risk Adjustment, per model point. Shape ``(n_mp,)``.

    Phase 0 placeholder: ``RA = ra_rate * PV(claims)``. Phase 1 replaces this
    with a confidence-level or cost-of-capital method.
    """
    return ra_rate * _pv(proj.claim_cf, discount)


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

    Roll-forward (Phase 0 -- deterministic, no assumption changes):
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
