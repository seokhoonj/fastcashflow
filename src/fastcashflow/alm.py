"""Asset-liability management foundations -- duration, DV01, key-rate duration.

The practical entry point to ALM is the deterministic interest-rate sensitivity
of the liability and the assets backing it -- duration and DV01 -- not a full
dynamic asset-liability projection. This module computes those from pieces the
engine already produces: the liability cash flows (`full=True` measurement), the
discount curve, and a re-measure under a shocked curve.

Two metrics, one unit:

* **DV01** -- the change in present value per 1 basis-point rise in the curve.
  Defined for any present value (positive, negative, near zero), so it is the
  robust headline for the LIABILITY, whose best-estimate value can be small or
  negative for a profitable book. Computed by a parallel curve bump and
  re-measure (so it captures any rate-dependent cash flows the engine models).
* **Macaulay / Modified duration** -- the present-value-weighted average time and
  its yield sensitivity. Clean and textbook for a single-sign cash-flow stream (a
  BOND); reported as an effective modified duration for the liability, guarded
  when the present value is near zero.

DV01 is the common unit that lets an asset book and the liability be compared --
the asset-liability DV01 gap (zero = immunised against a parallel rate move).

Scope (v1): liability DV01 / effective duration / key-rate duration, a bond
duration, and the DV01 gap. A full asset-portfolio projection (rolling,
reinvestment) -- dynamic ALM -- a real-world scenario generator, convexity, and
credit-spread sensitivity are out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.engine import measure
from fastcashflow.model_points import ModelPoints

_BP = 1e-4    # one basis point


@dataclass(frozen=True, slots=True)
class DurationResult:
    """Interest-rate sensitivity of a present value.

    ``pv`` is the present value (the BEL for a liability, the market value for a
    bond). ``macaulay`` / ``modified`` are durations in years (``macaulay`` is
    ``nan`` where it is not well defined -- a mixed-sign liability stream).
    ``dv01`` is the decrease in ``pv`` for a +1bp parallel rise in the curve
    (positive for a normal positive-duration instrument)."""

    pv: float
    macaulay: float
    modified: float
    dv01: float


def _bel(model_points: ModelPoints, basis: Basis, discount_annual) -> float:
    """Portfolio BEL under a discount curve override (fast path)."""
    m = measure(model_points, replace(basis, discount_annual=discount_annual),
                full=False)
    return float(m.bel.sum())


def _base_curve(basis: Basis, n_years: int) -> FloatArray:
    """The basis discount curve as a per-year array of length ``n_years`` (a
    scalar is broadcast); the tail is held flat past the supplied curve."""
    base = np.asarray(basis.discount_annual, dtype=np.float64)
    if base.ndim == 0:
        return np.full(n_years, float(base))
    if base.shape[0] >= n_years:
        return base[:n_years].copy()
    return np.concatenate([base, np.full(n_years - base.shape[0], base[-1])])


def net_liability_cashflows(measurement) -> tuple[FloatArray, FloatArray]:
    """The portfolio net liability cash flow per month, in the engine's timing.

    Returns ``(flow_bom, flow_mid)``: begin-of-month flows ``(n_time+1,)`` --
    ``annuity - premium`` plus the maturity benefit placed at each contract's
    boundary -- and mid-month flows ``(n_time,)`` -- death / morbidity /
    disability / expense / surrender claims. Dotting these with the measurement's
    ``discount_factor_bom`` / ``discount_factor_mid`` reproduces the BEL. The
    premium is the only inflow (a minus); every claim is an outflow (a plus).

    Requires a ``full=True`` measurement. Account-value (universal-life) books are
    rejected: their BEL is netted of the account fund after discounting, so the
    raw cash flows do not reconstruct it.
    """
    cf = measurement.cashflows
    if cf is None:
        raise ValueError(
            "net_liability_cashflows needs a full=True measurement (the cash "
            "flows); the headline-only fast path does not carry them.")
    if getattr(cf, "account", None) is not None:
        raise NotImplementedError(
            "net_liability_cashflows does not support account-value (UL) books -- "
            "their BEL nets the account fund after discounting, so the raw cash "
            "flows do not reconstruct it.")
    n_time = cf.premium_cf.shape[1]
    flow_mid = (cf.mortality_cf + cf.morbidity_cf + cf.disability_cf
                + cf.expense_cf + cf.surrender_cf).sum(axis=0)
    flow_bom = np.zeros(n_time + 1, dtype=np.float64)
    flow_bom[:n_time] = (cf.annuity_cf - cf.premium_cf).sum(axis=0)
    boundary = np.asarray(measurement.model_points.contract_boundary_months,
                          dtype=np.int64)
    np.add.at(flow_bom, np.minimum(boundary, n_time), np.asarray(cf.maturity_cf, float))
    return flow_bom, flow_mid


def liability_dv01(model_points: ModelPoints, basis: Basis, *,
                   bump: float = _BP) -> float:
    """The liability DV01 -- the decrease in BEL for a +1bp parallel rise in the
    discount curve, by central difference (re-measure at ``+/-bump``).

    Robust for any BEL (positive, negative, near zero). ``bump`` is the parallel
    rate shift used for the finite difference (default 1bp); the result is scaled
    to a per-1bp figure."""
    base = np.asarray(basis.discount_annual, dtype=np.float64)
    up = _bel(model_points, basis, base + bump)
    dn = _bel(model_points, basis, base - bump)
    return -(up - dn) / (2.0 * bump) * _BP


def liability_duration(model_points: ModelPoints, basis: Basis, *,
                       bump: float = _BP) -> DurationResult:
    """The liability's interest-rate sensitivity -- ``pv`` (BEL), ``dv01`` and an
    effective ``modified`` duration (``= dv01 / (|pv| * 1bp)``). ``macaulay`` is
    ``nan`` (the mixed-sign liability stream has no clean Macaulay time);
    ``modified`` is ``nan`` when ``|pv|`` is negligible (the ratio is then
    ill-conditioned -- read the ``dv01`` instead)."""
    pv = _bel(model_points, basis, np.asarray(basis.discount_annual, np.float64))
    dv01 = liability_dv01(model_points, basis, bump=bump)
    modified = dv01 / (abs(pv) * _BP) if abs(pv) > 1.0 else float("nan")
    return DurationResult(pv=pv, macaulay=float("nan"), modified=modified, dv01=dv01)


def key_rate_durations(model_points: ModelPoints, basis: Basis, *,
                       bump: float = _BP) -> FloatArray:
    """Key-rate DV01s -- the liability DV01 attributed to each policy-year bucket
    of the curve, by bumping one year of the per-year discount curve at a time
    (central difference). Returns ``(n_years,)``; the buckets sum to approximately
    the parallel :func:`liability_dv01` (the key-rate decomposition of it)."""
    n_years = int(np.ceil(
        float(np.asarray(model_points.contract_boundary_months).max()) / 12.0))
    base = _base_curve(basis, n_years)
    krd = np.empty(n_years, dtype=np.float64)
    for k in range(n_years):
        up_curve = base.copy(); up_curve[k] += bump
        dn_curve = base.copy(); dn_curve[k] -= bump
        up = _bel(model_points, basis, up_curve)
        dn = _bel(model_points, basis, dn_curve)
        krd[k] = -(up - dn) / (2.0 * bump) * _BP
    return krd


__all__ = [
    "DurationResult", "net_liability_cashflows",
    "liability_dv01", "liability_duration", "key_rate_durations",
]
