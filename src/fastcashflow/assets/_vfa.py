"""VFA (account-value) asset-liability tools -- the variable book's cash-flow gap,
exposed prefix-free under ``fcf.vfa.*``.

The VFA counterpart of :func:`fastcashflow.assets.cashflow_gap`: it nets the
projected asset cash flows against the VFA entity net-liability ladder
(:func:`fastcashflow.vfa.net_liability_cashflows`) FOR an account-value book.
"""
from __future__ import annotations

import numpy as np

from fastcashflow.assets import CashflowGap, Portfolio, portfolio_cashflows
from fastcashflow.alm import _vfa as _alm_vfa


def cashflow_gap(portfolio: Portfolio, measurement) -> CashflowGap:
    """The asset-liability cash-flow gap for a variable (VFA) book.

    The VFA counterpart of :func:`fastcashflow.assets.cashflow_gap`: nets the projected asset cash
    flows (:func:`portfolio_cashflows`) against the VFA entity net liability
    cash flow (:func:`fastcashflow.vfa.net_liability_cashflows`) -- the
    guarantee top-up plus expenses less the entity income (the variable fee for a
    variable annuity, the account charges for universal life), on the engine's
    monthly grid. Unlike ``cashflow_gap`` this is FOR account-value books, not
    against them: the account-value benefit is funded by the unit fund, so the gap
    pits the entity's own assets only against the guarantee-excess basis.
    Undiscounted liquidity ladder -- where the general account throws off / must find
    cash to carry the guarantee. Requires a ``full=True`` measurement; both the
    variable-annuity and universal-life paths are supported."""
    net = _alm_vfa.net_liability_cashflows(measurement)      # (n_time,)
    n_time = net.shape[0]
    liability_cf = np.zeros(n_time + 1, dtype=np.float64)
    liability_cf[:n_time] = net
    asset_cf = portfolio_cashflows(portfolio, n_time)
    return CashflowGap(asset_cf=asset_cf, liability_cf=liability_cf)
