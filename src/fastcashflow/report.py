"""IFRS 17 financial-statement disclosure -- the insurance service result.

Turns a GMM measurement into the period-by-period IFRS 17 reporting figures:
the insurance service result and its build-up (insurance revenue and service
expense), the insurance finance expense, and the contractual service margin
analysis of change.

Insurance revenue follows IFRS 17 paragraphs B120-B124: the revenue of a
period is the reduction in the liability for remaining coverage that relates
to services -- the insurance service expenses expected to be incurred, plus
the release of the risk adjustment, plus the release of the CSM. The
insurance service result is revenue less the service expenses actually
incurred; in a deterministic projection expected equals actual, so the
result is exactly the RA release plus the CSM release -- the profit emerging
as service is provided.

v1 scope: investment-component benefits (maturity, annuity and surrender
values) are excluded from the revenue and service-expense lines -- a full
investment-component split is left for later, as is the liability-for-
incurred-claims reconciliation and the loss-component detail.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.engine import Measurement


def _to_years(monthly: FloatArray) -> FloatArray:
    """Sum a per-month series into policy years."""
    n_time = monthly.shape[0]
    n_years = (n_time + 11) // 12
    padded = np.zeros(n_years * 12)
    padded[:n_time] = monthly
    return padded.reshape(n_years, 12).sum(axis=1)


@dataclass(frozen=True, slots=True)
class Report:
    """IFRS 17 disclosure figures, period by period.

    Every array is shaped ``(n_mp, n_time)`` -- one row per model point, one
    column per month. ``insurance_service_result`` is revenue less service
    expense; ``insurance_finance_expense`` is signed (positive an expense,
    negative income). The CSM analysis of change reconciles as
    ``csm_opening + csm_accretion - csm_release = csm_closing``.
    """

    insurance_revenue: FloatArray
    insurance_service_expense: FloatArray
    insurance_service_result: FloatArray
    insurance_finance_expense: FloatArray
    csm_opening: FloatArray
    csm_accretion: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray

    def annual(self) -> dict[str, FloatArray]:
        """Portfolio totals aggregated to policy years.

        Each line item is summed across model points and then across the
        twelve months of each policy year, giving a ``(n_years,)`` array.
        """
        return {
            name: _to_years(getattr(self, name).sum(axis=0))
            for name in (
                "insurance_revenue", "insurance_service_expense",
                "insurance_service_result", "insurance_finance_expense",
                "csm_accretion", "csm_release",
            )
        }


def report(measurement: Measurement) -> Report:
    """Assemble the IFRS 17 disclosure from a GMM measurement.

    See the module docstring for the basis (IFRS 17 paragraphs B120-B124).
    """
    bel, ra, csm = measurement.bel, measurement.ra, measurement.csm
    cf = measurement.cashflows
    monthly_rate = 1.0 / measurement.discount_start[1] - 1.0
    full = 1.0 / (1.0 + monthly_rate)

    # Insurance service expenses incurred in the period -- death and health
    # claims and expenses (investment-component benefits excluded, see the
    # module docstring).
    service_expense = cf.claim_cf + cf.morbidity_cf + cf.expense_cf

    # The risk adjustment and CSM released to profit or loss in the period.
    ra_release = ra[:, :-1] - ra[:, 1:] * full
    csm_release = measurement.csm_release

    # Insurance revenue (B124): the service expenses plus the RA and CSM
    # released. The service result is revenue less the expenses incurred.
    insurance_revenue = service_expense + ra_release + csm_release
    insurance_service_result = ra_release + csm_release

    # Insurance finance expense -- the unwind of discount on the fulfilment
    # cash flows plus the interest accreted on the CSM.
    insurance_finance_expense = (
        monthly_rate * (bel[:, :-1] + ra[:, :-1]) + measurement.csm_accretion
    )

    return Report(
        insurance_revenue=insurance_revenue,
        insurance_service_expense=service_expense,
        insurance_service_result=insurance_service_result,
        insurance_finance_expense=insurance_finance_expense,
        csm_opening=csm[:, :-1],
        csm_accretion=measurement.csm_accretion,
        csm_release=csm_release,
        csm_closing=csm[:, 1:],
    )
