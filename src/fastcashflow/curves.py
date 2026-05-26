"""Time-axis curves derived from an ``Assumptions`` set.

The orchestration layer (engine, PAA, VFA) calls these helpers to turn the
high-level assumption object into the concrete per-month / per-year arrays
the numerical primitives consume. Keeping these in their own layer lets the
numerical primitives stay domain-object-free (numpy arrays only) and the
``Assumptions`` dataclass stay math-free (just inputs).

Two groups of helpers:

* discount factors -- :func:`discount_factors`,
  :func:`discount_factors_from_curve`, :func:`discount_monthly_curve`.
  These read ``Assumptions.discount_annual`` (scalar or per-year array)
  and broadcast a per-year array to per-month length, holding the last
  value flat past the end.
* (internal) :func:`_per_year_to_per_month` -- the shared broadcast helper.
"""
from __future__ import annotations

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions


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


def discount_monthly_curve(assumptions: Assumptions, n_time: int) -> FloatArray:
    """Per-month locked-in monthly discount rate, shape ``(n_time,)``.

    Locked-in basis (Sec. 36) is held either as a flat annual rate or a
    per-year annual curve on ``assumptions.discount_annual``. Within a
    policy year a constant-force conversion turns the annual to monthly
    (twelve monthly applications reproduce the annual exactly).
    """
    annual = _per_year_to_per_month(
        assumptions.discount_annual, n_time, "discount_annual",
    )
    return (1.0 + annual) ** (1.0 / 12.0) - 1.0


def discount_factors(assumptions: Assumptions, n_time: int) -> tuple[FloatArray, FloatArray]:
    """Discount factors back to time 0, by cash-flow timing.

    Returns ``(discount_start, discount_mid)``:

    * ``discount_start[t]`` -- shape ``(n_time+1,)`` -- start-of-month flows
      (premiums) and the maturity benefit at time = term.
    * ``discount_mid[t]`` -- shape ``(n_time,)`` -- mid-month flows
      (claims and expenses, which arise during the month).

    The discount basis is the locked-in rate or rate curve carried on
    ``assumptions`` (Sec. 36); a flat scalar gives the closed-form ``(1+i)^-t``
    expression and a per-year curve gives the cumulative-product form.
    """
    return discount_factors_from_curve(discount_monthly_curve(assumptions, n_time))


def discount_factors_from_curve(
    monthly_rates: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Discount factors from a per-month rate curve.

    ``monthly_rates`` is a ``(n_time,)`` array of monthly forward rates --
    the rate applied across each projection month. Returns the same
    ``(discount_start, discount_mid)`` pair as :func:`discount_factors`; a
    constant curve reproduces it bar floating-point rounding.
    """
    monthly_rates = np.asarray(monthly_rates, dtype=np.float64)
    discount_start = np.empty(monthly_rates.shape[0] + 1)
    discount_start[0] = 1.0
    np.cumprod(1.0 / (1.0 + monthly_rates), out=discount_start[1:])
    discount_mid = discount_start[:-1] / np.sqrt(1.0 + monthly_rates)
    return discount_start, discount_mid
