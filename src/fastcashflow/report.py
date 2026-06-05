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
from functools import singledispatch

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.engine import GMMMeasurement
from fastcashflow._paa import PAAMeasurement
from fastcashflow._vfa import VFAMeasurement


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

    ``insurance_finance_expense`` is also disaggregated by source (IFRS 17
    B130-B136) into ``bel_finance_expense`` (finance on the estimates of
    future cash flows), ``ra_finance_expense`` (finance on the risk
    adjustment) and ``csm_finance_expense`` (the CSM interest accreted at the
    locked-in rate, B72). The three sum to ``insurance_finance_expense`` up to
    floating-point rounding (the aggregate is kept as its own expression, so
    the parts may differ from it by a rounding step rather than re-deriving
    it). The split is the structural basis for a later P&L / OCI allocation.
    """

    insurance_revenue: FloatArray
    insurance_service_expense: FloatArray
    insurance_service_result: FloatArray
    insurance_finance_expense: FloatArray
    bel_finance_expense: FloatArray   # B130-B136: finance on the FCF estimates
    ra_finance_expense: FloatArray    # B130-B136: finance on the risk adjustment
    csm_finance_expense: FloatArray   # B130-B136: CSM interest at the locked-in rate (B72)
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


@singledispatch
def report(measurement) -> Report:
    """Assemble the IFRS 17 report from a GMM, PAA or VFA measurement.

    See the module docstring for the basis (IFRS 17 paragraphs B120-B124).
    Dispatches on the measurement type; a new model registers its own report
    with ``@report.register``.
    """
    raise TypeError(
        "report() expects a GMM, PAA or VFA measurement, got "
        f"{type(measurement).__name__}"
    )


@report.register
def _(measurement: GMMMeasurement) -> Report:
    return _report_gmm(measurement)


@report.register
def _(measurement: PAAMeasurement) -> Report:
    return _report_paa(measurement)


@report.register
def _(measurement: VFAMeasurement) -> Report:
    return _report_vfa(measurement)


def _report_gmm(m: GMMMeasurement) -> Report:
    """GMM: revenue grosses up the RA release and the CSM release."""
    if m.bel_path is None:
        raise ValueError(
            "report() requires a full=True measurement; the trajectory fields "
            "are None on the full=False fast path. Call measure(..., full=True)."
        )
    bel, ra, csm = m.bel_path, m.ra_path, m.csm_path
    cf = m.cashflows
    # Per-month forward rate from the discount-factor curve, so that a
    # non-flat curve accretes the FCF and discounts the RA release at the
    # right rate in every month -- the same pattern movement.py uses. The last
    # axis is time: (n_time,) for a single basis, (n_mp, n_time) for a segmented
    # measurement; the array maths below broadcast over either shape.
    ds = m.discount_bom
    monthly_rate = ds[..., :-1] / ds[..., 1:] - 1.0
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
        # Disaggregated by source (B130-B136). Computed as separate values --
        # NOT a refactor of the aggregate above -- so the aggregate expression
        # stays byte-identical (a*b + a*c is not bit-identical to a*(b+c)).
        bel_finance_expense=monthly_rate * bel[:, :-1],
        ra_finance_expense=monthly_rate * ra[:, :-1],
        csm_finance_expense=m.csm_accretion,
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
        bel_finance_expense=zeros,
        ra_finance_expense=zeros,
        csm_finance_expense=zeros,
        loss_component=m.loss_component,
        csm_opening=zeros,
        csm_accretion=zeros,
        csm_release=zeros,
        csm_closing=zeros,
    )


def _report_vfa(m: VFAMeasurement) -> Report:
    """VFA: profit emerges as the CSM releases; the RA covers expense risk."""
    csm = m.csm_path
    service_expense = m.cashflows.expense_cf       # account value is investment comp.
    csm_release = m.csm_release
    # Release the expense-risk RA over the coverage period, in proportion to
    # the coverage units (in-force).
    inforce = m.cashflows.inforce
    ra0 = m.ra_path[:, 0]                                # inception RA
    ra_release = ra0[:, None] * inforce / inforce.sum(axis=1, keepdims=True)
    return Report(
        insurance_revenue=service_expense + ra_release + csm_release,
        insurance_service_expense=service_expense,
        insurance_service_result=ra_release + csm_release,
        insurance_finance_expense=m.csm_accretion,
        # VFA finance is the CSM accretion only -- the account value is the
        # investment component, and the variable-fee (B132) disaggregation is
        # out of scope. So the whole finance line sits on the CSM component.
        bel_finance_expense=np.zeros_like(m.csm_accretion),
        ra_finance_expense=np.zeros_like(m.csm_accretion),
        csm_finance_expense=m.csm_accretion,
        loss_component=m.loss_component,
        csm_opening=csm[:, :-1],
        csm_accretion=m.csm_accretion,
        csm_release=csm_release,
        csm_closing=csm[:, 1:],
    )
