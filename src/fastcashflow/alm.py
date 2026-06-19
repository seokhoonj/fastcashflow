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


# ---------------------------------------------------------------------------
# Bonds -- the asset side's interest-rate sensitivity (single-sign cash flows,
# so the textbook Macaulay / Modified duration applies cleanly).
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Bond:
    """A fixed-coupon bullet bond. ``coupon_rate`` is the annual coupon as a
    fraction of ``face``; ``frequency`` is the number of coupons per year.

    ``credit_rating`` (external / S&P scale: AAA, AA, A, BBB, BB, B, CCC, D, or
    "unrated") and ``exposure_class`` ("corporate", "public", "securitisation")
    drive the credit-risk SCR (:func:`fastcashflow.credit_scr`); ``currency`` (ISO
    code, "KRW" for domestic) drives the FX SCR (:func:`fastcashflow.fx_scr`). None
    of these affect the price or duration; the market value is in the reporting
    currency."""

    face: float
    coupon_rate: float
    maturity_years: float
    frequency: int = 1
    credit_rating: str = "AA"
    exposure_class: str = "corporate"
    currency: str = "KRW"


def bond_cashflows(bond: Bond) -> tuple[FloatArray, FloatArray]:
    """The bond's ``(times_years, amounts)`` -- a coupon at each period and the
    face repaid with the final coupon."""
    n = int(round(bond.maturity_years * bond.frequency))
    times = np.arange(1, n + 1, dtype=np.float64) / bond.frequency
    coupon = bond.face * bond.coupon_rate / bond.frequency
    amounts = np.full(n, coupon, dtype=np.float64)
    amounts[-1] += bond.face
    return times, amounts


def effective_maturity(bond: Bond) -> float:
    """The cash-flow-weighted average maturity ``sum(t * CF_t) / sum(CF_t)``
    (K-ICS effective maturity, undiscounted as written in the standard). Used to
    pick the credit-risk maturity bucket. A coupon bond's effective maturity is
    shorter than its final maturity (early coupons pull the weight in)."""
    t, a = bond_cashflows(bond)
    total = float(a.sum())
    return float((t * a).sum() / total) if total > 0.0 else 0.0


def _annual_df(times: FloatArray, discount_annual) -> FloatArray:
    """Annual-compounding discount factors at ``times`` (years) for a flat scalar
    rate or a per-year rate array (the spot, year by year, held flat past its
    end). Constant-force monthly discounting agrees with this at the year grid."""
    times = np.asarray(times, dtype=np.float64)
    c = np.asarray(discount_annual, dtype=np.float64)
    if c.ndim == 0:
        return (1.0 + float(c)) ** (-times)
    n_max = int(np.ceil(times.max())) if times.size else 0
    rates = np.array([c[min(k, c.shape[0] - 1)] for k in range(n_max)])
    cum = np.concatenate([[0.0], np.cumsum(np.log1p(rates))])   # cum[n] = sum_{k<n} ln(1+c_k)
    floor = np.floor(times).astype(np.int64)
    frac = times - floor
    last_ln = np.array([np.log1p(c[min(k, c.shape[0] - 1)]) for k in floor])
    return np.exp(-(cum[floor] + frac * last_ln))


def bond_value(bond: Bond, discount_annual) -> float:
    """Market value of the bond -- its cash flows discounted at the curve."""
    t, a = bond_cashflows(bond)
    return float((a * _annual_df(t, discount_annual)).sum())


def _bond_irr(times: FloatArray, amounts: FloatArray, pv: float) -> float:
    """The flat annual yield reproducing ``pv`` (bisection; price falls in yield).

    The bracket ``(-0.99, 100)`` contains any realistic bond yield -- a
    positive-cash-flow bond has price ``-> +inf`` as the yield approaches -100%
    and ``-> 0`` as it grows, so the root is always inside. Raises if the price is
    not bracketed (e.g. non-positive or non-monotone cash flows)."""
    lo, hi = -0.99, 100.0

    def f(y: float) -> float:
        return float((amounts * (1.0 + y) ** (-times)).sum()) - pv

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            "bond yield is not bracketed in (-0.99, 100) -- check the bond cash "
            "flows and price")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < 1e-10 or (hi - lo) < 1e-13:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def bond_duration(bond: Bond, discount_annual) -> DurationResult:
    """The bond's market value, Macaulay / Modified duration and DV01. Macaulay is
    the present-value-weighted time; Modified is ``Macaulay / (1 + y)`` with ``y``
    the flat-equivalent yield; DV01 is ``Modified * value * 1bp`` (the value drop
    per +1bp)."""
    t, a = bond_cashflows(bond)
    pv_t = a * _annual_df(t, discount_annual)
    pv = float(pv_t.sum())
    macaulay = float((t * pv_t).sum() / pv)
    y = _bond_irr(t, a, pv)
    modified = macaulay / (1.0 + y)
    return DurationResult(pv=pv, macaulay=macaulay, modified=modified,
                          dv01=modified * pv * _BP)


def alm_gap(asset_dv01: float, liability_dv01: float) -> dict:
    """The asset-liability DV01 gap -- ``asset_dv01 - liability_dv01``. Zero means
    the net value is immunised against a small parallel rate move (the asset and
    liability fall by the same amount per 1bp). Both inputs are DV01s on the same
    curve (e.g. summed :func:`bond_duration` DV01s and :func:`liability_dv01`)."""
    return {"asset_dv01": asset_dv01, "liability_dv01": liability_dv01,
            "dv01_gap": asset_dv01 - liability_dv01}


__all__ = [
    "DurationResult", "Bond", "net_liability_cashflows",
    "liability_dv01", "liability_duration", "key_rate_durations",
    "bond_cashflows", "bond_value", "bond_duration", "effective_maturity",
    "alm_gap",
]
