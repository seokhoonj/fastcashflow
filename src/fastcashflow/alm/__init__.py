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

from dataclasses import replace

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow._duration import DurationResult, _BP
from fastcashflow.basis import Basis
from fastcashflow._measurement.gmm import measure
from fastcashflow._measurement.account import _portfolio_has_account
from fastcashflow.model_points import ModelPoints


def _reject_account_book_alm(model_points: ModelPoints, basis: Basis,
                             entry: str) -> None:
    """Route an account-value (universal-life / variable) book away from the
    GMM-style discount-bump interest metrics.

    Such a book discounts its liability at the underlying-items return, not the
    risk-free curve, so bumping ``discount_annual`` is not its interest
    sensitivity. The VFA layer carries the symmetric tools: the interest
    sensitivity is :func:`fastcashflow.vfa.liability_duration` / :func:`fastcashflow.vfa.liability_dv01`
    (``fcf.vfa.liability_duration``, which bumps the underlying-items return), the
    rate capital is the VFA interest sub-risk
    (:func:`fastcashflow.vfa.interest_scr`), and the asset-liability cash-flow
    ladder is :func:`fastcashflow.vfa.cashflow_gap` over
    :func:`fastcashflow.vfa.net_liability_cashflows`.
    """
    if _portfolio_has_account(model_points, basis):
        raise NotImplementedError(
            f"{entry} does not apply to an account-value (universal-life / "
            "variable) book -- its liability is discounted at the underlying-"
            "items return, not the risk-free curve, so a discount-curve bump is "
            "not its interest sensitivity. Use fcf.vfa.liability_duration / "
            "fcf.vfa.liability_dv01 for the VFA interest sensitivity, "
            "fcf.vfa.interest_scr for the rate capital, and "
            "fcf.vfa.cashflow_gap / fcf.vfa.net_liability_cashflows for "
            "the asset-liability cash-flow ladder.")


def _bel(model_points: ModelPoints, basis: Basis, discount_annual) -> float:
    """Portfolio BEL under a discount curve override (fast path)."""
    _reject_account_book_alm(
        model_points, basis,
        "the ALM liability interest metrics (liability_dv01 / liability_duration "
        "/ key_rate_dv01s)")
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

    Requires a ``full=True`` measurement. This is the GROSS-benefit ladder for a
    non-account book; an account-value (universal-life / variable) book has its
    own entity net-liability ladder -- use :func:`fastcashflow.vfa.net_liability_cashflows`
    (and :func:`fastcashflow.vfa.cashflow_gap` for the asset-liability
    gap), which net the account fund the entity holds. This function rejects an
    account book rather than return a gross ladder its net BEL would not match.
    """
    cf = measurement.cashflows
    if cf is None:
        raise ValueError(
            "net_liability_cashflows needs a full=True measurement (the cash "
            "flows); the headline-only fast path does not carry them.")
    if getattr(cf, "account", None) is not None:
        raise NotImplementedError(
            "net_liability_cashflows is the gross-benefit ladder for a "
            "non-account book; an account-value (universal-life / variable) book "
            "nets the account fund after discounting, so the raw flows do not "
            "reconstruct its net BEL. Use fcf.vfa.net_liability_cashflows "
            "(the entity net-liability ladder) / fcf.vfa.cashflow_gap "
            "instead.")
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
    """The liability's interest-rate sensitivity -- ``pv`` (BEL), ``dv01``, an
    effective ``modified`` duration (``= dv01 / (|pv| * 1bp)``) and an effective
    ``convexity`` (the second central difference of the BEL under a parallel curve
    shift, ``(BEL(+b) + BEL(-b) - 2 BEL(0)) / (|pv| * b^2)``). ``macaulay`` is
    ``nan`` (the mixed-sign liability stream has no clean Macaulay time);
    ``modified`` / ``convexity`` are ``nan`` when ``|pv|`` is negligible (the
    ratios are then ill-conditioned -- read the ``dv01`` instead)."""
    base = np.asarray(basis.discount_annual, np.float64)
    pv = _bel(model_points, basis, base)
    dv01 = liability_dv01(model_points, basis, bump=bump)
    if abs(pv) > 1.0:
        modified = dv01 / (abs(pv) * _BP)
        up = _bel(model_points, basis, base + bump)
        dn = _bel(model_points, basis, base - bump)
        convexity = (up + dn - 2.0 * pv) / (abs(pv) * bump * bump)
    else:
        modified = convexity = float("nan")
    return DurationResult(pv=pv, macaulay=float("nan"), modified=modified,
                          dv01=dv01, convexity=convexity)


def key_rate_dv01s(model_points: ModelPoints, basis: Basis, *,
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


def gap(asset_dv01: float, liability_dv01: float) -> dict:
    """The asset-liability DV01 gap -- ``asset_dv01 - liability_dv01``. Zero means
    the net value is immunised against a small parallel rate move (the asset and
    liability fall by the same amount per 1bp). Both inputs are DV01s on the same
    curve (e.g. summed :func:`bond_duration` DV01s and :func:`liability_dv01`)."""
    return {"asset_dv01": asset_dv01, "liability_dv01": liability_dv01,
            "dv01_gap": asset_dv01 - liability_dv01}


def duration_gap(asset_duration: float, asset_value: float,
                 liability_duration: float, liability_value: float) -> dict:
    """The value-weighted (modified) duration gap of the surplus.

    ``duration_gap = D_A - (L/A) * D_L`` with ``D_A`` / ``D_L`` the asset /
    liability modified durations, ``A`` the asset market value and ``L`` the
    liability value (the BEL). ``leverage = L/A`` is the liability-to-asset ratio
    that scales the liability duration onto the asset base, because the surplus
    ``E = A - L`` moves by ``dE = -(D_A*A - D_L*L)*dy = -A*duration_gap*dy``. So a
    zero gap immunises the surplus against a small parallel yield move; a positive
    gap (assets longer than the leveraged liabilities) means the surplus FALLS when
    yields rise. ``surplus_dv01 = A * duration_gap * 1bp`` is that fall per +1bp --
    the same quantity as the :func:`gap` ``dv01_gap`` when the durations and
    values are mutually consistent (``dv01 = modified * value * 1bp``).

    Durations are modified (per unit yield); take them from
    :attr:`DurationResult.modified` and the values from :attr:`DurationResult.pv`
    (e.g. :func:`liability_duration` and a summed :func:`bond_duration`)."""
    leverage = liability_value / asset_value
    gap = asset_duration - leverage * liability_duration
    return {"asset_duration": asset_duration, "liability_duration": liability_duration,
            "leverage": leverage, "duration_gap": gap,
            "surplus_dv01": asset_value * gap * _BP}


__all__ = [
    "DurationResult", "net_liability_cashflows",
    "liability_dv01", "liability_duration", "key_rate_dv01s",
    "gap", "duration_gap",
]
