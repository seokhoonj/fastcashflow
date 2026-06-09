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
from fastcashflow.basis import Basis
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import project_cashflows


@dataclass(frozen=True, slots=True, eq=False)
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


def _validate_return_scenarios(return_scenarios: FloatArray) -> FloatArray:
    """Reject scenario sets the time-value kernel cannot price.

    An empty set reduces the scenario mean to NaN (which flows into the CSM and
    loss component); a non-finite return propagates NaN/inf; a monthly return of
    -100% or worse sign-flips the ``1 / (1 + r)`` discount and returns a
    plausible-looking wrong number. Returns the validated float array.
    """
    rs = np.asarray(return_scenarios, dtype=np.float64)
    if rs.shape[0] < 1:
        raise ValueError("return_scenarios must contain at least one scenario row")
    if not np.all(np.isfinite(rs)):
        raise ValueError("return_scenarios must be finite")
    if np.any(rs <= -1.0):
        raise ValueError(
            "return_scenarios must be greater than -1 -- a monthly return of "
            "-100% or worse is invalid"
        )
    return rs


def _av_and_discount(
    monthly_credit: FloatArray, monthly_return: FloatArray, fund_fee_m: float
) -> tuple[FloatArray, FloatArray]:
    """Return ``(av_factor, discount)``, each ``(n_scenarios, n_time)``.

    ``av_factor[s, t]`` is the account-value multiplier at the start of month
    ``t`` -- the product of ``(1 + credit)(1 - fee)`` over the months before
    ``t``. ``discount[s, t]`` is the product of ``1 / (1 + return)`` over those
    months. Kept separate so a non-linear payoff (a guarantee floor on the
    account value) can use the account-value path on its own; their product is
    :func:`_discounted_growth`.
    """
    growth = (1.0 + monthly_credit) * (1.0 - fund_fee_m)
    n = monthly_credit.shape[0]
    ones = np.ones((n, 1))
    av_factor = np.concatenate([ones, np.cumprod(growth, axis=1)[:, :-1]], axis=1)
    discount = np.concatenate(
        [ones, np.cumprod(1.0 / (1.0 + monthly_return), axis=1)[:, :-1]], axis=1
    )
    return av_factor, discount


def _discounted_growth(
    monthly_credit: FloatArray, monthly_return: FloatArray, fund_fee_m: float
) -> FloatArray:
    """The ``(n_scenarios, n_time)`` factor ``av_factor * discount``.

    The account grows each month at the credited rate net of the fund fee;
    the exit benefit discounts at the underlying-items return. Entry
    ``[s, t]`` is the product of those two over the months before ``t``.
    """
    av_factor, discount = _av_and_discount(monthly_credit, monthly_return, fund_fee_m)
    return av_factor * discount


def _pv_account_benefits(
    exit_value: FloatArray,
    monthly_credit: FloatArray,
    monthly_return: FloatArray,
    fund_fee_m: float,
) -> FloatArray:
    """PV of account-value exit benefits given monthly credit and return paths.

    ``exit_value`` is the ``(n_time,)`` portfolio total of account value times
    exiting policies per month, before any growth. Returns the
    ``(n_scenarios,)`` present value.
    """
    return _discounted_growth(monthly_credit, monthly_return, fund_fee_m) @ exit_value


def tvog_weights(
    *,
    minimum_crediting_rate: float,
    fund_fee: float,
    investment_return: float,
    return_scenarios: FloatArray,
) -> FloatArray:
    """Per-month weights for the time value of a minimum guarantee.

    Returns a ``(n_time,)`` vector ``w`` for which the TVOG of a book equals
    ``w @ e``, where ``e[t]`` is the book's total of account value times
    policies exiting in month ``t``. The weight is the mean over scenarios of
    the guaranteed discounted-growth factor less that factor in the central
    scenario -- the extra account-value cost the convex guarantee adds once
    returns vary.

    The guarantee is taken as a scalar: TVOG is a portfolio-level aggregate
    in v1, so per-MP varying guarantees are not yet supported here.
    """
    return_scenarios = _validate_return_scenarios(return_scenarios)
    n_time = return_scenarios.shape[1]
    f_m = (1.0 + fund_fee) ** (1.0 / 12.0) - 1.0
    g_m = (1.0 + minimum_crediting_rate) ** (1.0 / 12.0) - 1.0
    r_m = (1.0 + investment_return) ** (1.0 / 12.0) - 1.0
    stochastic = _discounted_growth(
        np.maximum(return_scenarios, g_m), return_scenarios, f_m
    ).mean(axis=0)
    central = np.full((1, n_time), r_m)
    central_factor = _discounted_growth(np.maximum(central, g_m), central, f_m)[0]
    return stochastic - central_factor


def guarantee_floor_time_value(
    *,
    account_value: FloatArray,
    deaths: FloatArray,
    maturity_survivors: FloatArray,
    term_index: FloatArray,
    minimum_death_benefit: FloatArray,
    minimum_accumulation_benefit: FloatArray,
    minimum_crediting_rate: float,
    fund_fee: float,
    investment_return: float,
    return_scenarios: FloatArray,
) -> FloatArray:
    """Per-model-point time value of the GMDB and GMAB account-value floors.

    A death pays ``max(account value, GMDB)`` and a maturity pays
    ``max(account value, GMAB)`` -- put options on the account value, struck at
    the guarantee. The deterministic projection prices only their intrinsic
    value (the cost in the central scenario); the *time value* -- the extra
    cost convexity adds once returns vary -- is the mean cost over the return
    scenarios less that central cost. Returns a ``(n_mp,)`` array, the amount
    the CSM additionally absorbs at inception. Discounting is at the underlying
    return -- the VFA basis, not a risk-neutral measure -- so this time value
    is *not* sign-constrained: a deep in-the-money floor can carry a negative
    time value, since volatility there mostly lets scenarios escape the floor.

    The account value path uses the credited rate ``max(return, guarantee)``;
    ``minimum_crediting_rate`` is the (scalar, v1) crediting guarantee. The
    GMDB / GMAB floors themselves may vary by model point.
    """
    return_scenarios = _validate_return_scenarios(return_scenarios)
    n_time = return_scenarios.shape[1]
    f_m = (1.0 + fund_fee) ** (1.0 / 12.0) - 1.0
    g_m = (1.0 + minimum_crediting_rate) ** (1.0 / 12.0) - 1.0
    r_m = (1.0 + investment_return) ** (1.0 / 12.0) - 1.0

    # Account-value multiplier and discount under each scenario, and under the
    # central flat-return path (whose floor cost is the intrinsic value).
    av_s, disc_s = _av_and_discount(
        np.maximum(return_scenarios, g_m), return_scenarios, f_m
    )
    central = np.full((1, n_time), r_m)
    av_c, disc_c = _av_and_discount(np.maximum(central, g_m), central, f_m)
    av_c, disc_c = av_c[0], disc_c[0]

    # The GMAB strikes at maturity (time = term) -- the *matured* account value
    # after the final month's growth -- one month past the width-n_time path the
    # GMDB walks (whose entries are start-of-month values). Extend each path by
    # that final month's growth and discount so the maturity index (term) reads
    # the matured value and discounts to time term, matching the deterministic
    # intrinsic value.
    av_s_mat = np.concatenate(
        [av_s, (av_s[:, -1] * (1.0 + np.maximum(return_scenarios[:, -1], g_m))
                * (1.0 - f_m))[:, None]], axis=1)
    disc_s_mat = np.concatenate(
        [disc_s, (disc_s[:, -1] / (1.0 + return_scenarios[:, -1]))[:, None]],
        axis=1)
    av_c_mat = np.append(av_c, av_c[-1] * (1.0 + max(r_m, g_m)) * (1.0 - f_m))
    disc_c_mat = np.append(disc_c, disc_c[-1] / (1.0 + r_m))

    n_mp = account_value.shape[0]
    time_value = np.zeros(n_mp)
    for mp in range(n_mp):
        av0 = account_value[mp]
        ti = int(term_index[mp])
        # Maturity index (time = term). ``term_index`` is the deterministic
        # path's boundary-clamped maturity column, so ti + 1 <= n_time already;
        # the min is a defensive belt against a caller passing an unclamped
        # term - 1 (a boundary-cut contract then has zero maturity survivors, so
        # the clamped read is harmless either way).
        mi = min(ti + 1, n_time)
        # GMDB: floor excess on the death exits each month (start-of-month
        # account value); GMAB: on the maturity survivors at the matured value.
        # Cost per scenario, then the mean less the central (intrinsic) cost.
        gdb_excess_s = np.maximum(0.0, minimum_death_benefit[mp] - av0 * av_s)
        cost_s = (deaths[mp] * gdb_excess_s * disc_s).sum(axis=1)
        gab_excess_s = np.maximum(
            0.0, minimum_accumulation_benefit[mp] - av0 * av_s_mat[:, mi]
        )
        cost_s = cost_s + maturity_survivors[mp] * gab_excess_s * disc_s_mat[:, mi]

        gdb_excess_c = np.maximum(0.0, minimum_death_benefit[mp] - av0 * av_c)
        cost_c = float((deaths[mp] * gdb_excess_c * disc_c).sum())
        gab_excess_c = max(
            0.0, minimum_accumulation_benefit[mp] - av0 * av_c_mat[mi]
        )
        cost_c += maturity_survivors[mp] * gab_excess_c * disc_c_mat[mi]

        time_value[mp] = float(cost_s.mean()) - cost_c
    return time_value


def measure_tvog(
    model_points: ModelPoints, basis: Basis, return_scenarios: FloatArray
) -> TVOGResult:
    """Measure the time value of a VFA contract's minimum-crediting-rate guarantee.

    Values the credited-rate floor only -- the guarantee that the account is
    credited ``max(return, minimum_crediting_rate)`` each month. The GMDB / GMAB
    account-value floors are NOT included here; their time value is folded into
    ``vfa.measure(..., return_scenarios).time_value`` instead. This function is
    the standalone credited-rate analysis.

    ``return_scenarios`` is an ``(n_scenarios, n_time)`` array of monthly
    underlying-items returns -- one path per scenario, ``n_time`` being the
    projection horizon. The model points must carry an explicit
    ``minimum_crediting_rate`` > 0; the always-on 0% crediting floor
    (``max(return, 0)``) and the GMDB / GMAB account-value floors are valued by
    ``vfa.measure(..., return_scenarios).time_value``, not here. In v1 the rate
    is taken as a portfolio-wide scalar (per-MP varying rates with stochastic
    returns are a future extension), so the column is required to be uniform
    across rows.

    The guarantee cost is the present value of account-value benefits in
    excess of the no-guarantee benefits. Its mean over the scenarios is the
    total value; the cost in the central scenario (``investment_return``) is
    the intrinsic value; the difference is the time value (TVOG).
    """
    g_unique = np.unique(np.asarray(model_points.minimum_crediting_rate,
                                     dtype=np.float64))
    if g_unique.size > 1:
        raise NotImplementedError(
            "measure_tvog requires a uniform minimum_crediting_rate across "
            "model points in v1; per-MP varying rates with stochastic "
            "returns are a future extension"
        )
    if g_unique.size == 0 or float(g_unique[0]) == 0.0:
        raise ValueError(
            "measure_tvog values an explicit minimum-crediting-rate guarantee "
            "(rate > 0), and this contract sets minimum_crediting_rate == 0. The "
            "always-on 0% crediting floor and the GMDB / GMAB account-value "
            "floors are valued by vfa.measure(..., return_scenarios).time_value "
            "instead."
        )
    g_annual = float(g_unique[0])

    return_scenarios = np.asarray(return_scenarios, dtype=np.float64)
    if return_scenarios.ndim != 2:
        raise ValueError("return_scenarios must be 2-D (n_scenarios, n_time)")
    return_scenarios = _validate_return_scenarios(return_scenarios)

    proj = project_cashflows(model_points, basis)
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
    exit_value = (model_points.account_value[:, None] * exits).sum(axis=0)   # (n_time,)

    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0
    g_m = (1.0 + g_annual) ** (1.0 / 12.0) - 1.0

    # Without a guarantee the return cancels between growth and discount, so
    # the no-guarantee benefit is identical in every scenario.
    no_guarantee = float(np.sum(exit_value * (1.0 - f_m) ** np.arange(n_time)))

    # Stochastic: credit max(return, guarantee), discount at the return.
    credit = np.maximum(return_scenarios, g_m)
    pv_stochastic = _pv_account_benefits(exit_value, credit, return_scenarios, f_m)
    guarantee_cost = pv_stochastic - no_guarantee

    # Deterministic central scenario -- a flat return path.
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
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
