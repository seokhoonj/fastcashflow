"""Time value of options and guarantees (TVOG) for VFA business.

A direct-participation (VFA) contract may carry a *minimum guaranteed
credited rate*: the account is credited ``max(return, guarantee)`` each
period, so the entity funds the shortfall whenever the return falls below
the guarantee.

Because ``max`` is convex, a deterministic projection at the central return
understates the guarantee's cost -- it sees only the *intrinsic value*, the
cost if the future equals the central scenario. The extra cost that comes
from return volatility -- the *time value* -- appears only when the contract
is valued over many scenarios. Their sum is the guarantee's total cost::

    total value  =  intrinsic value  +  time value (TVOG)

``measure_tvog`` values the account-value benefits under N return scenarios
and reports this split. The scenarios are an input -- fastcashflow is the
engine, not an economic scenario generator.

This is where a stochastic valuation genuinely changes the answer. For
linear protection business the stochastic mean equals the deterministic
result; a guarantee's time value, by contrast, is invisible to any single
deterministic run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.modelpoint import ModelPointSet
from fastcashflow.projection import project_cashflows


@dataclass(frozen=True, slots=True)
class TVOGResult:
    """The cost of a minimum guarantee, split into intrinsic and time value.

    ``guarantee_cost`` is the ``(n_scenarios,)`` present value of the
    guarantee under each scenario -- its distribution. ``intrinsic_value``
    is the cost in the central (deterministic) scenario; ``time_value`` is
    the TVOG, the mean cost in excess of the intrinsic value; and
    :attr:`total_value` is their sum, the guarantee's full economic cost.
    """

    guarantee_cost: FloatArray   # (n_scenarios,) -- PV cost of the guarantee
    intrinsic_value: float       # cost in the central deterministic scenario
    time_value: float            # TVOG -- mean cost less the intrinsic value

    @property
    def total_value(self) -> float:
        """Intrinsic value plus time value -- the guarantee's full cost."""
        return self.intrinsic_value + self.time_value


def _pv_account_benefits(
    exit_value: FloatArray,
    monthly_credit: FloatArray,
    monthly_return: FloatArray,
    fund_fee_m: float,
) -> FloatArray:
    """PV of account-value exit benefits given monthly credit and return paths.

    ``exit_value`` is the ``(n_time,)`` portfolio total of account value times
    exiting policies per month, before any growth. ``monthly_credit`` and
    ``monthly_return`` are ``(n_scenarios, n_time)`` paths: the account grows
    at the credited rate, and the benefits discount at the underlying return.
    Returns the ``(n_scenarios,)`` present value.
    """
    growth = (1.0 + monthly_credit) * (1.0 - fund_fee_m)
    n = monthly_credit.shape[0]
    ones = np.ones((n, 1))
    # av_factor[s, t] and discount[s, t] are products over months before t.
    av_factor = np.concatenate([ones, np.cumprod(growth, axis=1)[:, :-1]], axis=1)
    discount = np.concatenate(
        [ones, np.cumprod(1.0 / (1.0 + monthly_return), axis=1)[:, :-1]], axis=1
    )
    return (av_factor * discount) @ exit_value


def measure_tvog(
    mps: ModelPointSet, asmp: Assumptions, return_scenarios: FloatArray
) -> TVOGResult:
    """Measure the time value of a VFA contract's minimum guarantee.

    ``return_scenarios`` is an ``(n_scenarios, n_time)`` array of monthly
    underlying-items returns -- one path per scenario, ``n_time`` being the
    projection horizon. ``asmp`` must set ``guaranteed_credit_rate``; the
    account is credited ``max(return, guarantee)`` and the entity funds any
    shortfall.

    The guarantee cost is the present value of account-value benefits in
    excess of the no-guarantee benefits. Its mean over the scenarios is the
    total value; the cost in the central scenario (``investment_return``) is
    the intrinsic value; the difference is the time value (TVOG).
    """
    if asmp.guaranteed_credit_rate is None:
        raise ValueError("measure_tvog requires asmp.guaranteed_credit_rate to be set")

    return_scenarios = np.asarray(return_scenarios, dtype=np.float64)
    if return_scenarios.ndim != 2:
        raise ValueError("return_scenarios must be 2-D (n_scenarios, n_time)")

    proj = project_cashflows(mps, asmp)
    inforce = proj.inforce
    n_mp, n_time = inforce.shape
    if return_scenarios.shape[1] != n_time:
        raise ValueError(
            f"return_scenarios must have {n_time} columns (the projection "
            f"horizon), got {return_scenarios.shape[1]}"
        )

    # Portfolio total of (account value x policies exiting) per month -- the
    # benefit base before any return growth.
    inforce_pad = np.concatenate([inforce, np.zeros((n_mp, 1))], axis=1)
    exits = inforce_pad[:, :-1] - inforce_pad[:, 1:]
    exit_value = (mps.account_value[:, None] * exits).sum(axis=0)   # (n_time,)

    f_m = (1.0 + asmp.fund_fee) ** (1.0 / 12.0) - 1.0
    g_m = (1.0 + asmp.guaranteed_credit_rate) ** (1.0 / 12.0) - 1.0

    # Without a guarantee the return cancels between growth and discount, so
    # the no-guarantee benefit is identical in every scenario.
    no_guarantee = float(np.sum(exit_value * (1.0 - f_m) ** np.arange(n_time)))

    # Stochastic: credit max(return, guarantee), discount at the return.
    credit = np.maximum(return_scenarios, g_m)
    pv_stochastic = _pv_account_benefits(exit_value, credit, return_scenarios, f_m)
    guarantee_cost = pv_stochastic - no_guarantee

    # Deterministic central scenario -- a flat return path.
    r_m = (1.0 + asmp.investment_return) ** (1.0 / 12.0) - 1.0
    central = np.full((1, n_time), r_m)
    pv_central = _pv_account_benefits(
        exit_value, np.maximum(central, g_m), central, f_m
    )[0]
    intrinsic_value = float(pv_central - no_guarantee)

    time_value = float(guarantee_cost.mean() - intrinsic_value)
    return TVOGResult(
        guarantee_cost=guarantee_cost,
        intrinsic_value=intrinsic_value,
        time_value=time_value,
    )
