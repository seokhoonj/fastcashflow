"""Time-axis curves derived from an ``Basis`` set.

The orchestration layer (engine, PAA, VFA) calls these helpers to turn the
high-level assumption object into the concrete per-month / per-year arrays
the numerical primitives consume. Keeping these in their own layer lets the
numerical primitives stay domain-object-free (numpy arrays only) and the
``Basis`` dataclass stay math-free (just inputs).

Three helpers exposed:

* discount factors -- :func:`discount_factors`,
  :func:`discount_factors_from_curve`, :func:`discount_monthly_curve`.
  Read ``Basis.discount_annual`` (scalar or per-year curve) and
  broadcast a per-year array to per-month length, holding the last
  value flat past the end.
* expense inflation -- :func:`inflation_index`. Reads
  ``Basis.expense_inflation`` (same scalar-or-curve shape) and
  returns the cumulative ``(1+i)`` multiplier curve consumed by
  :func:`fastcashflow.basis.derive_expense_components`.
* (internal) :func:`_per_year_to_per_month` -- the shared broadcast helper.
"""
from __future__ import annotations

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis


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

    Returns ``(discount_bom, discount_mid)``:

    * ``discount_bom[t]`` -- shape ``(n_time+1,)`` -- start-of-month flows
      (premiums) and the maturity benefit at time = term.
    * ``discount_mid[t]`` -- shape ``(n_time,)`` -- mid-month flows
      (claims and expenses, which arise during the month).

    The discount basis is the locked-in rate or rate curve carried on
    ``basis`` (Sec. 36); a flat scalar gives the closed-form ``(1+i)^-t``
    expression and a per-year curve gives the cumulative-product form.
    """
    return discount_factors_from_curve(discount_monthly_curve(basis, n_time))


def discount_factors_from_curve(
    monthly_rates: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Discount factors from a per-month rate curve.

    ``monthly_rates`` is a ``(n_time,)`` array of monthly forward rates --
    the rate applied across each projection month. Returns the same
    ``(discount_bom, discount_mid)`` pair as :func:`discount_factors`; a
    constant curve reproduces it bar floating-point rounding.
    """
    monthly_rates = np.asarray(monthly_rates, dtype=np.float64)
    discount_bom = np.empty(monthly_rates.shape[0] + 1)
    discount_bom[0] = 1.0
    np.cumprod(1.0 / (1.0 + monthly_rates), out=discount_bom[1:])
    discount_mid = discount_bom[:-1] / np.sqrt(1.0 + monthly_rates)
    return discount_bom, discount_mid


def forward_rates(discount_bom: FloatArray) -> FloatArray:
    """The per-month forward rate implied by a beginning-of-month discount curve.

    ``discount_bom[..., t]`` discounts to the start of month ``t``; the one-month
    forward rate over month ``t`` is ``discount_bom[t] / discount_bom[t+1] - 1``
    -- the inverse of :func:`discount_factors_from_curve`. The trailing axis is
    time, so ``[..., :-1]`` / ``[..., 1:]`` serves a single ``(n_time+1,)`` curve
    and a per-MP ``(n_mp, n_time+1)`` one alike. The ellipsis is load-bearing:
    on a segmented (per-MP) curve a bare ``[:-1]`` would slice the model-point
    axis, not time -- the silent-wrong bug class this helper retires.
    """
    return discount_bom[..., :-1] / discount_bom[..., 1:] - 1.0
