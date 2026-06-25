"""Profit testing -- value and emergence of new business.

A thin pricing layer over the GMM measurement: the present-value metrics (new
business value, profit margin) and the period-by-period profit signature, plus
the rate metrics (IRR, break-even) on a shareholder cash-flow stream. It assembles
already-computed pieces -- the inception BEL/RA/CSM, ``report`` (the IFRS 17 P&L
emergence) and the discount curve -- rather than re-projecting anything.

Conventions (v1, pre-tax, pre-required-capital):
* New business value (NBV) = CSM + RA - loss component = the present value, at
  issue, of the profit a contract is expected to release. It equals -BEL.
* Profit signature = the per-period insurance service result -- on a
  best-estimate run that is the CSM release plus the RA release recognised each
  period; its present value at the locked-in rate is the NBV.
* IRR / break-even act on a shareholder cash-flow stream the caller supplies
  (the profit signature net of the day-0 new-business strain); they only carry
  meaning once that strain makes the stream change sign (the statutory profit
  test, where the reserve strain is explicit).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.reporting.report import report as _report


@dataclass(frozen=True, slots=True, eq=False)
class ProfitSignature:
    """Period-by-period shareholder profit emergence of a book.

    ``profit`` is ``(n_periods,)`` -- the profit recognised in each reporting
    period of ``period_months`` months (the portfolio total). ``month_end`` is
    the elapsed month at the end of each period. ``present_value`` discounts the
    stream; ``total`` is its undiscounted sum.
    """

    period_months: int
    month_end: IntArray           # (n_periods,) elapsed month at each period end
    profit: FloatArray            # (n_periods,) profit recognised per period

    @property
    def total(self) -> float:
        """Undiscounted lifetime profit."""
        return float(np.sum(self.profit))

    def present_value(self, annual_rate: float) -> float:
        """Present value of the profit stream at a flat annual ``annual_rate``,
        each period discounted to issue from its mid-point (the standard mid-year
        convention -- the period's profit emerges over the period). This
        approximately reconciles to :func:`nbv`; the exact new-business value is
        the NBV (``CSM + RA``), not this aggregated-and-re-discounted stream."""
        mid = (np.asarray(self.month_end, np.float64)
               - 0.5 * self.period_months) / 12.0
        return float(np.sum(self.profit / (1.0 + annual_rate) ** mid))


def nbv(measurement) -> FloatArray:
    """New business value per model point -- the present value at issue of the
    profit the contract is expected to release: ``CSM + RA - loss component``
    (equivalently ``-BEL``). Pre-tax and pre-required-capital."""
    return measurement.csm + measurement.ra - measurement.loss_component


def _pv_premium(measurement) -> FloatArray:
    """Present value of the premium stream per model point, on the locked-in
    discount curve, beginning-of-month (premiums fall at the start of a month)."""
    if measurement.cashflows is None or measurement.discount_factor_bom is None:
        raise ValueError(
            "profit metrics need a full=True measurement (the cash flows and "
            "discount factors); the headline-only fast path does not carry them.")
    premium = measurement.cashflows.premium_cf            # (n_mp, n_time)
    dfb = measurement.discount_factor_bom                 # (n_time+1,) or (n_mp, n_time+1)
    df = dfb[..., :premium.shape[1]]                      # align to n_time
    return np.sum(premium * df, axis=1)


def profit_margin(measurement) -> FloatArray:
    """Profit margin per model point -- new business value over the present value
    of premiums (the PVNBP margin). Zero-premium contracts return 0."""
    pvp = _pv_premium(measurement)
    safe = np.where(pvp != 0.0, pvp, 1.0)
    return np.where(pvp != 0.0, nbv(measurement) / safe, 0.0)


def signature(measurement, period_months: int = 12) -> ProfitSignature:
    """The IFRS 17 profit signature -- the per-period insurance service result
    (CSM release + RA release on a best-estimate run), summed over the book.

    Built from :func:`~fastcashflow.reporting.report`; the present value of the signature
    at the locked-in rate approximately reconciles to the portfolio :func:`nbv`
    total (the exact new-business value is the NBV; the annual signature is an
    aggregated presentation that re-discounts a year's profit from its mid-point).
    """
    rep = _report(measurement)
    by = rep.by_period(period_months)
    profit = np.asarray(by["insurance_service_result"], np.float64)
    n_periods = profit.shape[0]
    month_end = (np.arange(1, n_periods + 1) * period_months).astype(np.int64)
    return ProfitSignature(period_months=period_months, month_end=month_end,
                           profit=profit)


def irr(cashflows: FloatArray, *, period_months: int = 12,
        low: float = -0.99, high: float = 10.0) -> float:
    """Internal rate of return of a shareholder cash-flow stream (one entry per
    period of ``period_months`` months, period 0 first).

    The rate ``r`` (annual) at which the net present value is zero, found by
    bisection. The stream must change sign (a day-0 outgo / strain followed by
    profit), else there is no internal rate and a ``ValueError`` is raised -- an
    all-positive IFRS 17 signature has none; pair it with the new-business strain
    (the statutory profit test) for a meaningful IRR.
    """
    cf = np.asarray(cashflows, np.float64)
    step = period_months / 12.0
    t = np.arange(cf.shape[0]) * step

    def npv(r):
        return float(np.sum(cf / (1.0 + r) ** t))

    f_lo, f_hi = npv(low), npv(high)
    if f_lo == 0.0:
        return low
    if f_lo * f_hi > 0.0:
        raise ValueError(
            "no internal rate of return in [-0.99, 10]: the cash-flow stream "
            "does not change sign (an IRR needs an outgo followed by income).")
    for _ in range(200):
        mid = 0.5 * (low + high)
        f_mid = npv(mid)
        if abs(f_mid) < 1e-10 or (high - low) < 1e-12:
            return mid
        if f_lo * f_mid < 0.0:
            high = mid
        else:
            low, f_lo = mid, f_mid
    return 0.5 * (low + high)


def break_even_year(cashflows: FloatArray, *, period_months: int = 12) -> int:
    """The first period (1-based, in ``period_months`` units) at which the
    cumulative shareholder cash flow turns non-negative -- the payback point.
    Returns -1 if it never recovers."""
    cum = np.cumsum(np.asarray(cashflows, np.float64))
    hit = np.nonzero(cum >= 0.0)[0]
    return int(hit[0] + 1) if hit.size else -1


__all__ = ["ProfitSignature", "nbv", "profit_margin", "signature", "irr",
           "break_even_year"]
