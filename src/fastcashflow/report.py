"""IFRS 17 reporting -- the insurance service result.

``report`` turns a measurement -- GMM, PAA or VFA -- into the period-by-period
IFRS 17 reporting figures: the insurance service result and its build-up
(insurance revenue and service expense), the insurance finance expense, the
loss component of onerous contracts, and the contractual service margin
analysis of change. Producing the same ``Report`` shape for every measurement
model lets a mixed portfolio's report be compared and consolidated. A
reinsurance-held measurement instead returns a :class:`ReinsuranceReport` --
the mirror layout (IFRS 17 paragraphs 82 + 86), with premiums paid and amounts
recovered disaggregated and the net presentation left a property.

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

from typing import ClassVar

from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from fastcashflow._measurement_model import (
    GMM, REINSURANCE, model_tag, supported_model_tags,
)
from fastcashflow._typing import FloatArray
from fastcashflow.curves import forward_rates
from fastcashflow.engine import _require_full
from fastcashflow._measurement_basis import _require_inception
from fastcashflow._paa import _require_full_paa
from fastcashflow._vfa import _require_settlement_csm
import fastcashflow._gmm as _gmm
import fastcashflow._paa as _paa
import fastcashflow._vfa as _vfa
import fastcashflow._reinsurance as _reinsurance
from fastcashflow.solvency_assessment import DynamicSolvency


def _to_years(monthly: FloatArray) -> FloatArray:
    """Sum a per-month series into policy years."""
    n_time = monthly.shape[0]
    n_years = (n_time + 11) // 12
    padded = np.zeros(n_years * 12)
    padded[:n_time] = monthly
    return padded.reshape(n_years, 12).sum(axis=1)


# The flow lines a Report buckets into reporting periods (balances -- csm_opening
# / csm_closing -- are not summed; the period view reports flows).
_PERIOD_LINES = (
    "insurance_revenue", "insurance_service_expense", "insurance_service_result",
    "insurance_finance_expense", "bel_finance_expense", "ra_finance_expense",
    "csm_finance_expense", "csm_accretion", "csm_release",
)
_REINSURANCE_PERIOD_LINES = (
    "reinsurance_premium_allocated", "amounts_recovered",
    "reinsurance_service_result", "ra_release", "reinsurance_finance_expense",
    "bel_finance_expense", "ra_finance_expense", "csm_finance_expense",
    "csm_accretion", "csm_release",
)


def _period_offsets(basis: str, inception_month, n_mp: int) -> FloatArray:
    """The per-model-point calendar offset of policy-month 0.

    ``elapsed`` aligns every cohort at its own inception (offset 0): the period
    view sums by elapsed policy time, the generalisation of :meth:`annual`.
    ``calendar`` shifts each model point by its ``inception_month`` -- the
    calendar-month index (0-based, relative to the report origin) at which that
    cohort's coverage begins -- so flows fall in the calendar period they occur
    in, the basis a multi-cohort close reports on.
    """
    if basis == "elapsed":
        if inception_month is not None:
            raise ValueError(
                "by_period: 'elapsed' basis takes no inception_month "
                "(every cohort is aligned at its own inception)")
        return np.zeros(n_mp, dtype=np.int64)
    if basis == "calendar":
        if inception_month is None:
            raise ValueError(
                "by_period: 'calendar' basis needs inception_month -- the "
                "calendar-month index of each model point's inception")
        offsets = np.asarray(inception_month, dtype=np.int64)
        if offsets.shape != (n_mp,):
            raise ValueError(
                f"by_period: inception_month has {offsets.shape} entries for "
                f"{n_mp} model points")
        if (offsets < 0).any():
            raise ValueError("by_period: inception_month must be non-negative")
        return offsets
    raise ValueError(
        f"by_period: basis must be 'elapsed' or 'calendar', got {basis!r}")


def _chunk(collapsed: FloatArray, period_months: int, n_periods: int) -> FloatArray:
    """Sum a per-month portfolio series into ``n_periods`` reporting periods --
    the generalisation of :func:`_to_years` to an arbitrary period length."""
    padded = np.zeros(n_periods * period_months)
    padded[:collapsed.shape[0]] = collapsed
    return padded.reshape(n_periods, period_months).sum(axis=1)


def _by_period(report, line_names, period_months, basis, inception_month,
               loss_component):
    """Sum a report's flow lines into reporting periods of ``period_months``.

    Returns ``dict[str, FloatArray]`` -- each line's portfolio total per period
    (period 0 first). ``loss_component`` (an inception per-MP scalar), when
    given, is placed in the period containing each cohort's inception."""
    if period_months <= 0:
        raise ValueError(
            f"by_period: period_months must be positive, got {period_months}")
    n_mp, n_time = getattr(report, line_names[0]).shape
    offsets = _period_offsets(basis, inception_month, n_mp)

    if basis == "elapsed":
        # Every cohort is aligned at policy-month 0, so sum across model points
        # FIRST and then chunk -- the exact order :meth:`annual` uses, so
        # by_period(12) reproduces annual() bit for bit (a scatter-order sum
        # would diverge under floating-point cancellation).
        n_periods = (n_time + period_months - 1) // period_months
        result = {name: _chunk(getattr(report, name).sum(axis=0),
                               period_months, n_periods)
                  for name in line_names}
        if loss_component is not None:
            totals = np.zeros(n_periods)
            totals[0] = loss_component.sum()    # recognised at inception
            result["loss_component"] = totals
        return result

    # Calendar basis: cohorts have different inception months, so a flow at
    # (mp, month) lands in period (offset[mp] + month) // period_months. There
    # is no annual() counterpart, so the scatter accumulation is the form.
    period_index = ((offsets[:, None] + np.arange(n_time)[None, :])
                    // period_months).ravel()
    n_periods = int(period_index.max()) + 1
    result = {}
    for name in line_names:
        totals = np.zeros(n_periods)
        np.add.at(totals, period_index, getattr(report, name).ravel())
        result[name] = totals
    if loss_component is not None:
        totals = np.zeros(n_periods)
        np.add.at(totals, offsets // period_months, loss_component)
        result["loss_component"] = totals
    return result


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

    model: ClassVar[str] = GMM

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

    def by_period(self, period_months: int = 12, *, basis: str = "elapsed",
                  inception_month=None) -> dict[str, FloatArray]:
        """Portfolio totals bucketed into reporting periods of ``period_months``.

        The general form of :meth:`annual` (which is ``by_period(12)`` on the
        elapsed basis for its six lines): every flow line of the report --
        revenue, service expense, service result, the finance expense and its
        B130-B136 split, and the CSM accretion / release -- summed across model
        points into each reporting period, plus ``loss_component`` placed in the
        period of each cohort's inception. ``basis='elapsed'`` (default) buckets
        by elapsed policy time; ``basis='calendar'`` shifts each cohort by its
        ``inception_month`` so flows fall in the calendar period they occur in
        (see :func:`_period_offsets`). Returns ``dict[str, FloatArray]``, each
        array one entry per period (period 0 first).
        """
        return _by_period(self, _PERIOD_LINES, period_months, basis,
                          inception_month, self.loss_component)

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


@dataclass(frozen=True, slots=True)
class ReinsuranceReport:
    """IFRS 17 reporting figures for a reinsurance contract held, period by period.

    Reinsurance held is the *mirror* of an issued contract (IFRS 17 paragraph
    82): the cedant pays reinsurance premiums (an outflow) and receives
    recoveries of incurred claims (an inflow). Paragraph 86 lets the entity
    present the premiums paid net against the amounts recovered, or separately,
    so this report exposes the disaggregated components and leaves net-vs-gross
    a presentation choice -- ``net_reinsurance_result`` (a property) is the
    paragraph-86 net, not a stored field.

    Each flow array is shaped ``(n_mp, n_time)`` -- one row per model point, one
    column per month. ``reinsurance_premium_allocated`` (the systematic
    allocation of premiums paid, the cost side) and ``amounts_recovered``
    (recoveries of incurred claims, the income side) are both positive, matching
    the measurement's outflow-positive premium and inflow-positive recovery.
    ``reinsurance_service_result`` is the analog of the issuer service result --
    the release of the risk transferred plus the release of the CSM
    (``ra_release + csm_release``, IFRS 17 paragraphs 82 + B119), *not* the gross
    recovery-less-premium netting (that is ``net_reinsurance_result``).
    ``ra_release`` is the period release of the risk transferred (paragraph 64)
    excluding interest -- the same revenue-earned form as the issuer
    ``_report_gmm`` (the RA interest is in the finance line). The report (a P&L
    view) and the :class:`~fastcashflow.reinsurance.ReinsuranceReconciliation` (a liability
    roll-forward) decompose the same opening->closing transition differently, so
    ``ra_release`` here is the revenue-earned amount, not the reconciliation's
    movement residual; the finance lines and the CSM release do tie out.

    ``reinsurance_finance_expense`` is the interest unwind on the BEL and RA
    plus the CSM accretion at the locked-in rate, disaggregated by source (IFRS
    17 B130-B136) into ``bel_finance_expense`` (finance on the estimates of
    reinsurance cash flows), ``ra_finance_expense`` (finance on the risk
    transferred) and ``csm_finance_expense`` (the CSM interest, B72). The three
    sum to ``reinsurance_finance_expense`` up to floating-point rounding (the
    aggregate is kept as its own expression, so the parts may differ from it by
    a rounding step rather than re-deriving it).

    The CSM analysis of change reconciles as
    ``csm_opening + csm_accretion - csm_release = csm_closing``. There is no
    loss component (IFRS 17 Sec. 65): the CSM is the net cost or gain of the
    cover and may be negative -- a net cost is deferred and amortised, with no
    floor -- so the trajectory carries any negative value through as-is.
    """

    model: ClassVar[str] = REINSURANCE

    reinsurance_premium_allocated: FloatArray   # systematic allocation of premiums paid (cost side)
    amounts_recovered: FloatArray               # recoveries of incurred claims (income side)
    reinsurance_service_result: FloatArray      # ra_release + csm_release (paragraphs 82 + B119)
    ra_release: FloatArray                      # period unwind of the risk transferred (paragraph 64)
    reinsurance_finance_expense: FloatArray     # interest on BEL + RA + CSM accretion
    bel_finance_expense: FloatArray   # B130-B136: finance on the reinsurance FCF estimates
    ra_finance_expense: FloatArray    # B130-B136: finance on the risk transferred
    csm_finance_expense: FloatArray   # B130-B136: CSM interest at the locked-in rate (B72)
    csm_opening: FloatArray
    csm_accretion: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray

    @property
    def net_reinsurance_result(self) -> FloatArray:
        """IFRS 17 paragraph 86 net presentation: recoveries less premiums paid.

        ``amounts_recovered - reinsurance_premium_allocated`` -- positive when
        recoveries exceed the premiums allocated to the period. Paragraph 86
        permits this net presentation or the two disaggregated line items;
        the property supports the net choice without baking it into the report.
        This is *not* the service result (which is ``ra_release + csm_release``).
        """
        return self.amounts_recovered - self.reinsurance_premium_allocated

    def annual(self) -> dict[str, FloatArray]:
        """Portfolio totals aggregated to policy years.

        Each per-period line item is summed across model points and then
        across the twelve months of each policy year.
        """
        return {
            name: _to_years(getattr(self, name).sum(axis=0))
            for name in (
                "reinsurance_premium_allocated", "amounts_recovered",
                "reinsurance_service_result", "reinsurance_finance_expense",
                "csm_accretion", "csm_release",
            )
        }

    def by_period(self, period_months: int = 12, *, basis: str = "elapsed",
                  inception_month=None) -> dict[str, FloatArray]:
        """Portfolio totals bucketed into reporting periods of ``period_months``.

        The reinsurance-held counterpart of :meth:`Report.by_period`: premiums
        paid, amounts recovered, the service result, the RA release, and the
        finance expense with its B130-B136 split, summed across model points
        into each reporting period. There is no loss component (IFRS 17 Sec. 65).
        ``basis`` and ``inception_month`` behave as in :meth:`Report.by_period`.
        """
        return _by_period(self, _REINSURANCE_PERIOD_LINES, period_months, basis,
                          inception_month, None)

    def __str__(self) -> str:
        annual = self.annual()
        n_years = len(annual["reinsurance_premium_allocated"])
        shown = min(n_years, 5)
        rows = (
            ("Reinsurance premium", annual["reinsurance_premium_allocated"]),
            ("Amounts recovered",   annual["amounts_recovered"]),
            ("Net result",          annual["amounts_recovered"]
                                    - annual["reinsurance_premium_allocated"]),
            ("Service result",      annual["reinsurance_service_result"]),
            ("Finance expense",     annual["reinsurance_finance_expense"]),
            ("CSM accretion",       annual["csm_accretion"]),
            ("CSM release",         annual["csm_release"]),
        )
        title = "IFRS 17 reinsurance-held report -- annual portfolio totals"
        if n_years > shown:
            title += f" (first {shown} of {n_years} years)"
        header = f"{'':20}" + "".join(
            f"{f'Year {y + 1}':>14}" for y in range(shown)
        )
        lines = [title, header]
        for name, series in rows:
            lines.append(
                f"{name:20}"
                + "".join(f"{series[y]:>14,.0f}" for y in range(shown))
            )
        return "\n".join(lines)


def _ratio(r: float) -> str:
    """Format a solvency ratio as a percentage, or ``n/a`` when not finite (a
    non-positive required capital gives an unbounded ratio)."""
    return "n/a" if not np.isfinite(r) else f"{r * 100:,.1f}%"


@dataclass(frozen=True, slots=True)
class DynamicSolvencyReport:
    """Formatted view of a :func:`~fastcashflow.dynamic_solvency` scenario overlay.

    Lays out the static t=0 picture (available capital, required capital, ratio),
    the coupled rate / dynamic-lapse scenario (the mark-to-market revaluation and
    the forced-sale friction), and the after-scenario surplus and ratio. When the
    scenario forces a sale, a liquidation block shows the total sold, the realized
    loss and any unfunded shortfall. Output is ASCII English -- part of the API
    surface a global user sees."""

    result: DynamicSolvency

    def __str__(self) -> str:
        d = self.result
        s, it, liq = d.static, d.interaction, d.liquidation
        w = 30

        def row(label: str, value: float) -> str:
            return f"  {label:{w}}{value:>18,.0f}"

        lines = [
            "Dynamic solvency -- coupled rate / dynamic-lapse scenario overlay",
            "  -- static (t=0) --",
            row("Available capital", s.available_capital),
            row("Required capital (SCR)", s.total_scr),
            f"  {'Solvency ratio':{w}}{_ratio(s.solvency_ratio):>18}",
            "  -- scenario --",
            row("Base NAV", it.base_nav),
            row("Stressed NAV", it.stressed_nav),
            row("Revaluation loss", it.revaluation_loss),
            row("Forced-sale loss", it.forced_sale_loss),
            row("Total interaction loss", it.total_loss),
            "  -- after scenario --",
            row("Stressed available capital", d.stressed_available_capital),
            f"  {'Stressed solvency ratio':{w}}{_ratio(d.stressed_ratio):>18}",
        ]
        if liq.total_realized_loss or liq.total_unfunded:
            lines += [
                "  -- liquidation --",
                row("Forced sale (total)", float(liq.forced_sale.sum())),
                row("Realized loss (total)", liq.total_realized_loss),
                row("Unfunded (total)", liq.total_unfunded),
            ]
        return "\n".join(lines)


@singledispatch
def report(measurement) -> Report:
    """Assemble the IFRS 17 report from a GMM, PAA, VFA or reinsurance measurement.

    See the module docstring for the basis (IFRS 17 paragraphs B120-B124).
    Dispatches on the measurement type; a new model registers its own report
    with ``@report.register``. A reinsurance-held measurement returns a
    :class:`ReinsuranceReport` (the mirror layout, IFRS 17 paragraphs 82 + 86),
    not a :class:`Report`. A mixed-portfolio container
    (:class:`~fastcashflow.portfolio.PortfolioMeasurement` or
    :class:`~fastcashflow.portfolio.PortfolioGroups`) is also accepted: each
    model slot is reported on its own measurement and a
    :class:`~fastcashflow.portfolio.PortfolioReport` is returned (a GMM, PAA and
    VFA report are never merged).
    """
    raise TypeError(
        "report() expects one of "
        f"{', '.join(supported_model_tags(report))}, got "
        f"{model_tag(measurement)}"
    )


@report.register
def _(measurement: _gmm.Measurement) -> Report:
    _require_inception(measurement, "report()")
    return _report_gmm(measurement)


@report.register
def _(measurement: _paa.Measurement) -> Report:
    _require_inception(measurement, "report()")
    return _report_paa(measurement)


@report.register
def _(measurement: _vfa.Measurement) -> Report:
    _require_settlement_csm(measurement, "report")
    return _report_vfa(measurement)


@report.register
def _(measurement: _reinsurance.Measurement) -> ReinsuranceReport:
    _require_inception(measurement, "report()")
    return _report_reinsurance(measurement)


@report.register
def _(result: DynamicSolvency) -> DynamicSolvencyReport:
    """A dynamic-solvency scenario overlay reports its before / after picture."""
    return DynamicSolvencyReport(result=result)


def _report_gmm(m: _gmm.Measurement) -> Report:
    """GMM: revenue grosses up the RA release and the CSM release."""
    _require_full(m, "report()")
    bel, ra, csm = m.bel_path, m.ra_path, m.csm_path
    cf = m.cashflows
    # Per-month forward rate from the discount-factor curve, so that a
    # non-flat curve accretes the FCF and discounts the RA release at the
    # right rate in every month -- the same pattern movement.py uses. The last
    # axis is time: (n_time,) for a single basis, (n_mp, n_time) for a segmented
    # measurement; the array maths below broadcast over either shape.
    discount_monthly = forward_rates(m.discount_factor_bom)
    monthly_discount = 1.0 / (1.0 + discount_monthly)

    # Insurance service expense is the incurred protection benefit + expenses
    # (B120-B124). disability_cf -- the semi-Markov disability income / lump-sum
    # benefit -- is a protection claim, not an investment component, so it
    # belongs here; omitting it dropped the disability flow from a DI book's
    # revenue and service result entirely.
    service_expense = (cf.mortality_cf + cf.morbidity_cf + cf.disability_cf
                       + cf.expense_cf)
    ra_release = ra[:, :-1] - ra[:, 1:] * monthly_discount
    csm_release = m.csm_release

    return Report(
        insurance_revenue=service_expense + ra_release + csm_release,
        insurance_service_expense=service_expense,
        insurance_service_result=ra_release + csm_release,
        insurance_finance_expense=(
            discount_monthly * (bel[:, :-1] + ra[:, :-1]) + m.csm_accretion
        ),
        # Disaggregated by source (B130-B136). Computed as separate values --
        # NOT a refactor of the aggregate above -- so the aggregate expression
        # stays byte-identical (a*b + a*c is not bit-identical to a*(b+c)).
        bel_finance_expense=discount_monthly * bel[:, :-1],
        ra_finance_expense=discount_monthly * ra[:, :-1],
        csm_finance_expense=m.csm_accretion,
        loss_component=m.loss_component,
        csm_opening=csm[:, :-1],
        csm_accretion=m.csm_accretion,
        csm_release=csm_release,
        csm_closing=csm[:, 1:],
    )


def _report_reinsurance(m: _reinsurance.Measurement) -> ReinsuranceReport:
    """Reinsurance held: the cedant's premiums paid and recoveries received.

    IFRS 17 paragraphs 82 + 86 present income or expenses from reinsurance
    contracts held separately from issued contracts. The service result is the
    release of the risk transferred (paragraph 64) plus the release of the CSM
    (the net cost / gain of the cover, Sec. 65) -- the same release-based build
    as the issuer ``_report_gmm`` service result, with the premiums paid and
    recoveries received exposed as disaggregated line items (the paragraph-86
    net is then a presentation choice, computed by the report property).

    The RA release is the change in the risk transferred excluding interest
    (``opening - closing discounted``), the same revenue-earned form as the
    issuer ``_report_gmm`` -- so the RA interest goes to the finance line and the
    service result is the pure release. The report (a P&L view) and the
    reconciliation (a liability roll-forward) decompose the same opening->closing
    transition differently, so this ``ra_release`` is the revenue-earned amount,
    not the reconciliation's movement residual; the finance lines and the CSM
    release, however, do tie out to the reconciliation. The finance expense is
    the interest on the BEL and RA at the locked-in rate plus the CSM accretion,
    disaggregated by source (B130-B136). There is no loss component (Sec. 65);
    the CSM may be negative and the trajectory carries it through.
    """
    _require_full(m, "report()")
    bel, ra, csm = m.bel_path, m.ra_path, m.csm_path
    # Per-month forward rate from the discount-factor curve -- the same pattern
    # _report_gmm and movement.py use, so the finance unwind matches in every
    # month. The last axis is time: (n_time,) for a single basis, (n_mp, n_time)
    # for a segmented measurement; the maths below broadcast over either shape.
    discount_monthly = forward_rates(m.discount_factor_bom)
    monthly_discount = 1.0 / (1.0 + discount_monthly)

    # The RA release the same form as the issuer _report_gmm -- the change in
    # the risk transferred EXCLUDING interest (opening - closing discounted) --
    # so the RA interest sits in the finance line and the service result is the
    # pure release. The report (a P&L view) and the reconciliation (a liability
    # roll-forward) are different decompositions of the same opening->closing
    # transition, so this revenue-earned ra_release is NOT the reconciliation's
    # movement residual (opening + interest - closing); the finance lines and the
    # CSM release do tie out to the reconciliation, the RA run-off line does not.
    ra_release = ra[:, :-1] - ra[:, 1:] * monthly_discount
    csm_release = m.csm_release

    return ReinsuranceReport(
        # Disaggregated cash flows (paragraph 86): premiums paid (outflow,
        # cost side) and recoveries received (inflow, income side), both positive.
        reinsurance_premium_allocated=m.reinsurance_premium,
        amounts_recovered=m.recovery,
        # Service result -- release of risk transferred + CSM release, mirroring
        # the issuer service result (_report_gmm: ra_release + csm_release).
        reinsurance_service_result=ra_release + csm_release,
        ra_release=ra_release,
        reinsurance_finance_expense=(
            discount_monthly * (bel[:, :-1] + ra[:, :-1]) + m.csm_accretion
        ),
        # Disaggregated by source (B130-B136). Computed as separate values --
        # NOT a refactor of the aggregate above -- so the aggregate expression
        # stays byte-identical (a*b + a*c is not bit-identical to a*(b+c)).
        bel_finance_expense=discount_monthly * bel[:, :-1],
        ra_finance_expense=discount_monthly * ra[:, :-1],
        csm_finance_expense=m.csm_accretion,
        csm_opening=csm[:, :-1],
        csm_accretion=m.csm_accretion,
        csm_release=csm_release,
        csm_closing=csm[:, 1:],
    )


def _report_paa(m: _paa.Measurement) -> Report:
    """PAA: the service result is already revenue less expense; no CSM."""
    _require_full_paa(m, "report()")
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


def _report_vfa(m: _vfa.Measurement) -> Report:
    """VFA: profit emerges as the CSM releases; the RA covers expense risk."""
    _require_full(m, "report()")
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
