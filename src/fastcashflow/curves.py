"""``fastcashflow.curves`` -- the interest-rate / time-value curve domain.

One namespace for everything about the term-structure curve, in two activities:

* CURVE CONSTRUCTION -- fit a market spot curve from observed data:
  :func:`smith_wilson` / :func:`smith_wilson_alpha` / :func:`smith_wilson_prices`
  (Smith-Wilson interpolation/extrapolation) and :func:`nelson_siegel` /
  :func:`nelson_siegel_svensson` / :func:`fit_nelson_siegel_svensson`
  (Nelson-Siegel-Svensson parametric fit). The result is an annual spot curve the
  user writes onto ``Basis.discount_annual``.
* CURVE TRANSFORM -- the three equivalent term-structure representations and the
  maps between them. Spot/forward rates and discount factors are the same curve
  in different views (``discount_factors`` <-> ``forward_rates`` are inverses):
  :func:`discount_monthly_curve` / :func:`discount_factors` /
  :func:`discount_factors_from_curve` / :func:`forward_rates`, plus the expense
  multiplier :func:`inflation_index`. These read ``Basis`` (or a raw rate array)
  and broadcast a per-year curve to per-month length, holding the last value flat
  past the end.

The discount/transform helpers are defined here; the builders are re-exported
from :mod:`fastcashflow._smith_wilson` / :mod:`fastcashflow._nelson_siegel` so the
whole curve domain has one home. ``project_cashflows`` (which CONSUMES a curve to
project cash flows) lives in :mod:`fastcashflow.core`, not here.
"""
from __future__ import annotations

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
# Curve builders -- re-exported so fcf.curves is the single home for the domain.
from fastcashflow._smith_wilson import (
    smith_wilson, smith_wilson_alpha, smith_wilson_prices)
from fastcashflow._nelson_siegel import (
    nelson_siegel, nelson_siegel_svensson, fit_nelson_siegel_svensson,
    NelsonSiegelSvensson)

__all__ = [
    # construction (builders)
    "smith_wilson", "smith_wilson_alpha", "smith_wilson_prices",
    "nelson_siegel", "nelson_siegel_svensson", "fit_nelson_siegel_svensson",
    "NelsonSiegelSvensson",
    # transform (the three representations + expense inflation)
    "discount_monthly_curve", "discount_factors", "discount_factors_from_curve",
    "forward_rates", "inflation_index",
]


def _per_year_to_per_month(
    annual: float | FloatArray, n_time: int, name: str,
) -> FloatArray:
    """Expand a scalar or per-year annual value to a ``(n_time,)`` per-month array.

    Per-month entry ``t`` carries the annual value for policy year
    ``t // 12``; if the per-year input is shorter than the projection it is
    held flat at its last value -- consistent with the per-duration lapse
    handling. Used for discount / inflation / maintenance fields.
    """
    if np.ndim(annual) == 0:
        return np.full(n_time, float(annual))
    arr = np.asarray(annual, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar or 1-D array, got shape {arr.shape}"
        )
    idx = np.minimum(np.arange(n_time) // 12, arr.shape[0] - 1)
    return arr[idx]


def discount_monthly_curve(basis: Basis, n_time: int) -> FloatArray:
    """Per-month locked-in monthly discount rate, shape ``(n_time,)``.

    Locked-in basis (Sec. 36) is held either as a flat annual rate or a
    per-year annual curve on ``basis.discount_annual``. Within a
    policy year a constant-force conversion turns the annual to monthly
    (twelve monthly applications reproduce the annual exactly).
    """
    annual = _per_year_to_per_month(
        basis.discount_annual, n_time, "discount_annual",
    )
    # ``(1+annual)**(1/12)`` is NaN when ``annual <= -1.0`` (a non-positive
    # base raised to a fractional power). Reject so a silently-NaN discount
    # curve does not propagate to BEL.
    if np.any(annual <= -1.0):
        bad = float(np.min(annual))
        raise ValueError(
            f"discount_annual must be > -1.0 (a rate <= -100% has no "
            f"monthly equivalent), got min {bad!r}"
        )
    return (1.0 + annual) ** (1.0 / 12.0) - 1.0


def inflation_index(basis: Basis, n_time: int) -> FloatArray:
    """Per-month expense-inflation multiplier, shape ``(n_time,)``.

    A flat ``Basis.expense_inflation = i`` reproduces the
    closed-form ``(1+i)^(t/12)`` growth. A per-year curve compounds
    annual factors across completed policy years and applies the
    in-year fractional ramp on the current year. Held flat past the
    end of the curve. ``derive_expense_components`` multiplies the
    recurring expense items (``gamma_fixed``, ``lae_pro_rata``) by
    this curve.
    """
    annual = _per_year_to_per_month(
        basis.expense_inflation, n_time, "expense_inflation",
    )
    months = np.arange(n_time)
    in_year_ramp = (1.0 + annual) ** ((months % 12) / 12.0)
    # Compounded annual factors across completed years. Twelve months
    # within a year share the same prior compounding, so a per-year
    # cumprod over the year-boundary slice is enough.
    annual_per_year = annual[::12]
    compounded = np.empty(annual_per_year.shape[0] + 1)
    compounded[0] = 1.0
    np.cumprod(1.0 + annual_per_year, out=compounded[1:])
    return compounded[months // 12] * in_year_ramp


def discount_factors(basis: Basis, n_time: int) -> tuple[FloatArray, FloatArray]:
    """Discount factors back to time 0, by cash-flow timing.

    Returns ``(discount_factor_bom, discount_factor_mid)``:

    * ``discount_factor_bom[t]`` -- shape ``(n_time+1,)`` -- start-of-month flows
      (premiums) and the maturity benefit at time = term.
    * ``discount_factor_mid[t]`` -- shape ``(n_time,)`` -- mid-month flows
      (claims and expenses, which arise during the month).

    The discount basis is the locked-in rate or rate curve carried on
    ``basis`` (Sec. 36); a flat scalar gives the closed-form ``(1+i)^-t``
    expression and a per-year curve gives the cumulative-product form.
    """
    return discount_factors_from_curve(discount_monthly_curve(basis, n_time))


def discount_factors_from_curve(
    discount_monthly: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Discount factors from a per-month rate curve.

    ``discount_monthly`` is a ``(n_time,)`` array of monthly forward rates --
    the rate applied across each projection month. Returns the same
    ``(discount_factor_bom, discount_factor_mid)`` pair as :func:`discount_factors`; a
    constant curve reproduces it bar floating-point rounding.
    """
    discount_monthly = np.asarray(discount_monthly, dtype=np.float64)
    discount_factor_bom = np.empty(discount_monthly.shape[0] + 1)
    discount_factor_bom[0] = 1.0
    np.cumprod(1.0 / (1.0 + discount_monthly), out=discount_factor_bom[1:])
    discount_factor_mid = discount_factor_bom[:-1] / np.sqrt(1.0 + discount_monthly)
    return discount_factor_bom, discount_factor_mid


def forward_rates(discount_factor_bom: FloatArray) -> FloatArray:
    """The per-month forward rate implied by a beginning-of-month discount curve.

    ``discount_factor_bom[..., t]`` discounts to the start of month ``t``; the one-month
    forward rate over month ``t`` is ``discount_factor_bom[t] / discount_factor_bom[t+1] - 1``
    -- the inverse of :func:`discount_factors_from_curve`. The trailing axis is
    time, so ``[..., :-1]`` / ``[..., 1:]`` serves a single ``(n_time+1,)`` curve
    and a per-MP ``(n_mp, n_time+1)`` one alike. The ellipsis is load-bearing:
    on a segmented (per-MP) curve a bare ``[:-1]`` would slice the model-point
    axis, not time -- the silent-wrong bug class this helper retires.
    """
    return discount_factor_bom[..., :-1] / discount_factor_bom[..., 1:] - 1.0
