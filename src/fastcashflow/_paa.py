"""IFRS 17 Premium Allocation Approach (PAA) -- the simplified measurement.

The PAA is the simplified model the standard permits for short-coverage
contracts -- IFRS 17 paragraphs 53-59 (eligibility, Sec. 53; the liability
for remaining coverage, Sec. 55; insurance revenue, Sec. B126). Instead of
the GMM's BEL / RA / CSM, the Liability for Remaining Coverage (LRC) is
measured like an unearned premium: premiums build it up, insurance revenue
draws it down as coverage is provided. There is no CSM -- profit emerges
as revenue is earned.

Scope and simplifications, each with the standard's basis:

* Acquisition cash flows are expensed as incurred -- the Sec. 59(a) option,
  available when the coverage period is one year or less -- so they are not
  held in the LRC.
* The LRC is held undiscounted: Sec. 56 does not require a financing
  adjustment when the time between providing service and the related
  premium due date is one year or less.
* Insurance revenue is allocated by ``revenue_basis``: Sec. B126(a)
  (passage of time -- premium earned straight-line over the coverage
  period, the default) or Sec. B126(b) (the expected timing of incurred
  claims and expenses).
* The onerous test (Sec. 57-58) is applied at inception. The loss is
  ``max(0, fulfilment cash flows for remaining coverage - LRC)``, which at
  inception equals ``max(0, the GMM fulfilment cash flows)``. It is
  reported separately rather than folded into the LRC carrying amount.
* The Liability for Incurred Claims (Sec. 59(b)) runs off a claims
  settlement pattern; with no pattern set, claims settle when incurred and
  it is zero. It is held undiscounted -- Sec. 59(b) permits this when
  claims are paid within a year of being incurred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.curves import discount_monthly_curve
from fastcashflow.numerics import _norm_ppf, _rollforward_kernel, _settlement_lic
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows


@dataclass(frozen=True, slots=True)
class PAAMeasurement:
    """PAA measurement -- the Liability for Remaining Coverage and the
    underwriting result released from it.

    ``lrc`` is an ``(n_mp, n_time+1)`` trajectory; column 0 is the inception
    LRC. ``revenue`` and ``service_expense`` are ``(n_mp, n_time)`` -- the
    insurance revenue earned and the insurance service expense incurred each
    month. ``service_result`` (a property) is their difference. ``lic`` is
    the ``(n_mp, n_time+1)`` liability for incurred claims -- claims build it
    up as they are incurred and run it off as they are paid.
    """

    # headline -- always present, shape (n_mp,)
    lrc: FloatArray              # inception Liability for Remaining Coverage
    loss_component: FloatArray   # onerous-contract loss at inception
    # trajectory -- full only (None on the headline-only path)
    lrc_path: FloatArray | None = None         # (n_mp, n_time+1) -- LRC trajectory
    revenue: FloatArray | None = None          # (n_mp, n_time)   -- insurance revenue earned
    service_expense: FloatArray | None = None  # (n_mp, n_time)   -- claims + expenses incurred
    lic: FloatArray | None = None              # (n_mp, n_time+1) -- liability for incurred claims
    cashflows: "Cashflows | None" = None

    @property
    def service_result(self) -> FloatArray:
        """Insurance service result -- revenue less service expense."""
        return self.revenue - self.service_expense


def measure_paa(
    model_points: ModelPoints,
    basis: Basis,
    *,
    revenue_basis: str = "time",
) -> PAAMeasurement:
    """Measure a portfolio under the Premium Allocation Approach.

    The LRC rolls forward as ``LRC[t+1] = LRC[t] + premium[t] - revenue[t]``
    from ``LRC[0] = 0`` -- premiums received build it up, insurance revenue
    releases it. A single-premium contract gives the textbook pro-rata
    unearned premium reserve.

    ``revenue_basis`` selects the Sec. B126 allocation of insurance revenue,
    which always sums to the total premium:

    * ``"time"``   -- B126(a), passage of time: the premium earned
      straight-line over the coverage period (the default).
    * ``"claims"`` -- B126(b), the expected timing of incurred claims and
      expenses; for when the release of risk differs significantly from the
      passage of time. A policy with no service expense has no such pattern
      and falls back to ``"time"``.

    The onerous test reuses the GMM fulfilment cash flows: a contract whose
    inception fulfilment cash flows are a net outflow carries that outflow
    as a loss component.
    """
    proj = project_cashflows(model_points, basis)

    premium_total = proj.premium_cf.sum(axis=1)          # (n_mp,)
    service_expense = proj.claim_cf + proj.morbidity_cf + proj.expense_cf

    # Liability for incurred claims -- claims incurred build it up, claims
    # paid (spread over the settlement pattern) run it off. Held
    # undiscounted, consistent with the LRC.
    incurred = proj.claim_cf + proj.morbidity_cf
    if basis.settlement_pattern is None:
        lic = np.zeros((incurred.shape[0], incurred.shape[1] + 1))
    else:
        lic = _settlement_lic(incurred, basis.settlement_pattern)

    # Insurance revenue -- total premium allocated across the periods of
    # service (Sec. B126), so total revenue equals total premium.
    if revenue_basis == "time":
        # B126(a): premium earned straight-line over the coverage period.
        in_coverage = np.arange(proj.n_time)[None, :] < model_points.term_months[:, None]
        weight = in_coverage.astype(np.float64)
    elif revenue_basis == "claims":
        weight = service_expense.copy()                  # B126(b)
        empty = weight.sum(axis=1) == 0.0                # no pattern -> B126(a)
        weight[empty] = proj.inforce[empty]
    else:
        raise ValueError(
            f"revenue_basis must be 'time' or 'claims', got {revenue_basis!r}"
        )
    w_sum = weight.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum == 0.0, 1.0, w_sum)   # safe divide; weight=0 → revenue=0
    revenue = premium_total[:, None] * weight / w_sum

    # LRC roll-forward -- premiums build it up, revenue releases it.
    net = proj.premium_cf - revenue
    n_mp, n_time = net.shape
    lrc = np.zeros((n_mp, n_time + 1))
    lrc[:, 1:] = np.cumsum(net, axis=1)

    # Onerous test -- the GMM inception fulfilment cash flows.
    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _rollforward_kernel(
        proj.claim_cf, proj.morbidity_cf, proj.disability_cf, proj.expense_cf,
        proj.premium_cf, proj.annuity_cf, proj.maturity_cf, proj.surrender_cf,
        model_points.term_months,
        discount_monthly_curve(basis, proj.n_time),
    )
    z = _norm_ppf(basis.ra_confidence)
    ra0 = z * (basis.mortality_cv * pv_claims[:, 0]
               + basis.morbidity_cv * pv_morbidity[:, 0]
               + basis.disability_cv * pv_disability[:, 0]
               + basis.longevity_cv * pv_survival[:, 0])
    loss_component = np.maximum(0.0, bel[:, 0] + ra0)

    return PAAMeasurement(
        lrc=lrc[:, 0],
        loss_component=loss_component,
        lrc_path=lrc,
        revenue=revenue,
        service_expense=service_expense,
        lic=lic,
        cashflows=proj,
    )
