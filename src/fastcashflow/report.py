"""IFRS 17 reporting -- the insurance service result.

``report`` turns a measurement -- GMM, PAA or VFA -- into the period-by-period
IFRS 17 reporting figures: the insurance service result and its build-up
(insurance revenue and service expense), the insurance finance expense, the
loss component of onerous contracts, and the contractual service margin
analysis of change. Producing the same ``Report`` shape for every measurement
model lets a mixed portfolio's report be compared and consolidated.

Insurance revenue follows IFRS 17 paragraphs B120-B124: the revenue of a
period is the reduction in the liability for remaining coverage that relates
to services -- the insurance service expenses expected to be incurred, plus
the release of the risk adjustment and of the CSM. The insurance service
result is revenue less the service expenses actually incurred.

v1 scope: investment-component benefits (maturity, annuity, surrender and
account values) are excluded from the revenue and service-expense lines.
The liability for incurred claims is zero -- the engine settles claims when
incurred, with no settlement lag -- so it carries no separate reconciliation.
The loss component is reported at inception; its release trajectory and the
full incurred-claims movement are left for later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.engine import Measurement
from fastcashflow.paa import PAAMeasurement
from fastcashflow.vfa import VFAMeasurement


def _to_years(monthly: FloatArray) -> FloatArray:
    """Sum a per-month series into policy years."""
    n_time = monthly.shape[0]
    n_years = (n_time + 11) // 12
    padded = np.zeros(n_years * 12)
    padded[:n_time] = monthly
    return padded.reshape(n_years, 12).sum(axis=1)


@dataclass(frozen=True, slots=True)
class Report:
    """IFRS 17 reporting figures, period by period.

    Each flow array is shaped ``(n_mp, n_time)`` -- one row per model point,
    one column per month; ``loss_component`` is ``(n_mp,)`` -- the onerous
    loss at inception. ``insurance_service_result`` is revenue less service
    expense; ``insurance_finance_expense`` is signed (positive an expense).
    The CSM analysis of change reconciles as
    ``csm_opening + csm_accretion - csm_release = csm_closing`` (the CSM
    columns are zero for a PAA measurement, which has no CSM).
    """

    insurance_revenue: FloatArray
    insurance_service_expense: FloatArray
    insurance_service_result: FloatArray
    insurance_finance_expense: FloatArray
    loss_component: FloatArray
    csm_opening: FloatArray
    csm_accretion: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray

    def annual(self) -> dict[str, FloatArray]:
        """Portfolio totals aggregated to policy years.

        Each per-period line item is summed across model points and then
        across the twelve months of each policy year.
        """
        return {
            name: _to_years(getattr(self, name).sum(axis=0))
            for name in (
                "insurance_revenue", "insurance_service_expense",
                "insurance_service_result", "insurance_finance_expense",
                "csm_accretion", "csm_release",
            )
        }

    def __str__(self) -> str:
        annual = self.annual()
        n_years = len(annual["insurance_revenue"])
        shown = min(n_years, 5)
        rows = (
            ("Insurance revenue", annual["insurance_revenue"]),
            ("Service expense",   annual["insurance_service_expense"]),
            ("Service result",    annual["insurance_service_result"]),
            ("Finance expense",   annual["insurance_finance_expense"]),
            ("CSM accretion",     annual["csm_accretion"]),
            ("CSM release",       annual["csm_release"]),
        )
        title = "IFRS 17 report -- annual portfolio totals"
        if n_years > shown:
            title += f" (first {shown} of {n_years} years)"
        header = f"{'':18}" + "".join(
            f"{f'Year {y + 1}':>12}" for y in range(shown)
        )
        lines = [title, header]
        for name, series in rows:
            lines.append(
                f"{name:18}"
                + "".join(f"{series[y]:>12,.0f}" for y in range(shown))
            )
        return "\n".join(lines)


def report(measurement: Measurement | PAAMeasurement | VFAMeasurement) -> Report:
    """Assemble the IFRS 17 report from a GMM, PAA or VFA measurement.

    See the module docstring for the basis (IFRS 17 paragraphs B120-B124).
    """
    if isinstance(measurement, Measurement):
        return _report_gmm(measurement)
    if isinstance(measurement, PAAMeasurement):
        return _report_paa(measurement)
    if isinstance(measurement, VFAMeasurement):
        return _report_vfa(measurement)
    raise TypeError(
        "report() expects a GMM, PAA or VFA measurement, got "
        f"{type(measurement).__name__}"
    )


def _report_gmm(m: Measurement) -> Report:
    """GMM: revenue grosses up the RA release and the CSM release."""
    bel, ra, csm = m.bel, m.ra, m.csm
    cf = m.cashflows
    monthly_rate = 1.0 / m.discount_start[1] - 1.0
    full = 1.0 / (1.0 + monthly_rate)

    service_expense = cf.claim_cf + cf.morbidity_cf + cf.expense_cf
    ra_release = ra[:, :-1] - ra[:, 1:] * full
    csm_release = m.csm_release

    return Report(
        insurance_revenue=service_expense + ra_release + csm_release,
        insurance_service_expense=service_expense,
        insurance_service_result=ra_release + csm_release,
        insurance_finance_expense=(
            monthly_rate * (bel[:, :-1] + ra[:, :-1]) + m.csm_accretion
        ),
        loss_component=m.loss_component,
        csm_opening=csm[:, :-1],
        csm_accretion=m.csm_accretion,
        csm_release=csm_release,
        csm_closing=csm[:, 1:],
    )


def _report_paa(m: PAAMeasurement) -> Report:
    """PAA: the service result is already revenue less expense; no CSM."""
    zeros = np.zeros_like(m.revenue)
    return Report(
        insurance_revenue=m.revenue,
        insurance_service_expense=m.service_expense,
        insurance_service_result=m.revenue - m.service_expense,
        insurance_finance_expense=zeros,          # LRC held undiscounted
        loss_component=m.loss_component,
        csm_opening=zeros,
        csm_accretion=zeros,
        csm_release=zeros,
        csm_closing=zeros,
    )


def _report_vfa(m: VFAMeasurement) -> Report:
    """VFA: profit emerges as the CSM releases; the RA covers expense risk."""
    csm = m.csm
    service_expense = m.cashflows.expense_cf       # account value is investment comp.
    csm_release = m.csm_release
    # Release the expense-risk RA over the coverage period, in proportion to
    # the coverage units (in-force).
    inforce = m.cashflows.inforce
    ra0 = m.ra[:, 0]                                # inception RA
    ra_release = ra0[:, None] * inforce / inforce.sum(axis=1, keepdims=True)
    return Report(
        insurance_revenue=service_expense + ra_release + csm_release,
        insurance_service_expense=service_expense,
        insurance_service_result=ra_release + csm_release,
        insurance_finance_expense=m.csm_accretion,
        loss_component=m.loss_component,
        csm_opening=csm[:, :-1],
        csm_accretion=m.csm_accretion,
        csm_release=csm_release,
        csm_closing=csm[:, 1:],
    )
