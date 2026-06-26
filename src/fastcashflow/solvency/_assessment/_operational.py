"""Solvency assessment -- operational-risk SCR.

A liability-side factor charge (premiums and BEL), added on top of the BSCR. A
leaf sub-risk module: no asset model, no dependence on the other sub-risks.
"""
from __future__ import annotations

from fastcashflow.basis import Basis
from fastcashflow._measurement.gmm import measure
from fastcashflow.model_points import ModelPoints


# ---------------------------------------------------------------------------
# Operational risk -- a liability-side factor charge, added on top of the BSCR.
# K-ICS (table 40): max(premium exposure x 3.5%, current-estimate liability x
# 0.4%) for general life / long-term. Solvency II (Article 204): min(0.3 x BSCR,
# max(0.04 x premiums, 0.0045 x technical provisions)) + 0.25 x unit-linked
# expenses. Computed from the liability (premiums and BEL) -- no asset model.
# ---------------------------------------------------------------------------

_OPERATIONAL_CALIBRATION = {
    "K-ICS": {"method": "kics", "premium_factor": 0.035, "bel_factor": 0.004},
    "Solvency II": {"method": "sii", "premium_factor": 0.04, "bel_factor": 0.0045,
                    "cap_bscr": 0.30, "expul_factor": 0.25},
}


def _operational_cal(regime):
    try:
        return _OPERATIONAL_CALIBRATION[regime.name]
    except KeyError:
        raise ValueError(
            f"no operational-risk calibration for regime {regime.name!r}")


def operational_scr(model_points: ModelPoints, basis: Basis, regime, *,
                    bscr: float | None = None, measure_fn=None) -> float:
    """The operational-risk SCR -- a factor on the liability (premiums and BEL).

    K-ICS: ``max(premium x 3.5%, BEL x 0.4%)``. Solvency II: ``min(0.3 x bscr,
    max(0.04 x premium, 0.0045 x BEL)) + 0.25 x unit-linked expenses`` -- pass
    ``bscr`` (the basic SCR) for the cap; unit-linked expenses are 0 in v1. The
    premium exposure is the first projection year's earned premium; the BEL
    exposure is floored at zero.

    ``measure_fn`` is the liability measurement (default the GMM
    :func:`~fastcashflow._measurement.gmm.measure`); pass :func:`~fastcashflow.vfa.measure`
    for a variable book's BEL / premium exposure."""
    cal = _operational_cal(regime)
    mf = measure_fn if measure_fn is not None else measure
    m = mf(model_points, basis, full=True)
    bel = max(0.0, float(m.bel.sum()))
    premium = max(0.0, float(m.cashflows.premium_cf[:, :12].sum()))
    op = max(premium * cal["premium_factor"], bel * cal["bel_factor"])
    if cal["method"] == "sii":
        if bscr is not None:
            op = min(cal["cap_bscr"] * bscr, op)
        op += cal["expul_factor"] * 0.0          # unit-linked expenses (v1: 0)
    return op
