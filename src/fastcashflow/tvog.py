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
from fastcashflow.model_points import (
    ModelPoints, NO_GUARANTEE_RATE, validate_crediting_rate,
)
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


def credited_monthly_rate(
    monthly_return: FloatArray, annual_rate: FloatArray
) -> FloatArray:
    """The monthly rate credited to a VFA account under its minimum guarantee.

    Each month the account is credited ``max(monthly_return, monthly_floor)``,
    where ``monthly_floor`` is the monthly equivalent of the annual
    ``minimum_crediting_rate``. A rate of :data:`NO_GUARANTEE_RATE` carries no
    crediting guarantee -- the bare ``monthly_return`` is credited, which may be
    negative -- while ``0.0`` is a real 0% floor (principal protection). A
    positive rate floors the credited rate at its monthly equivalent.

    ``annual_rate`` is a scalar or an array broadcastable against
    ``monthly_return`` (a per-model-point vector, a scenario matrix, or the
    central path). This is the single site the crediting floor is applied:
    every deterministic, stochastic and maturity path routes through here, so
    no path can silently omit the floor. Callers validate the rate domain
    (:func:`validate_crediting_rate`) at the boundary, so the only sub-zero
    value reaching here is the exact sentinel.
    """
    annual_rate = np.asarray(annual_rate, dtype=np.float64)
    monthly_floor = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
    # No guarantee -> floor at -inf, so the maximum leaves the return untouched.
    floor = np.where(annual_rate == NO_GUARANTEE_RATE, -np.inf, monthly_floor)
    return np.maximum(monthly_return, floor)


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
    validate_crediting_rate(minimum_crediting_rate)
    return_scenarios = _validate_return_scenarios(return_scenarios)
    n_time = return_scenarios.shape[1]
    if minimum_crediting_rate == NO_GUARANTEE_RATE:
        # No crediting guarantee -> the credited rate is the bare return, so the
        # account growth and the discount cancel exactly and the credit-rate
        # TVOG is identically zero. Return it directly rather than route through
        # the separate cumulative growth / discount products, which over/underflow
        # (yielding 0 * inf = NaN) for an extreme-but-valid near-ruin return path.
        return np.zeros(n_time)
    f_m = (1.0 + fund_fee) ** (1.0 / 12.0) - 1.0
    r_m = (1.0 + investment_return) ** (1.0 / 12.0) - 1.0
    central = np.full((1, n_time), r_m)
    stochastic = _discounted_growth(
        credited_monthly_rate(return_scenarios, minimum_crediting_rate),
        return_scenarios, f_m
    ).mean(axis=0)
    central_factor = _discounted_growth(
        credited_monthly_rate(central, minimum_crediting_rate), central, f_m
    )[0]
    return stochastic - central_factor


def tvog_term_weight(
    *,
    minimum_crediting_rate: float,
    fund_fee: float,
    investment_return: float,
    return_scenarios: FloatArray,
) -> float:
    """The maturity term-column credit-rate TVOG weight (one scalar).

    A maturity survivor exits at time = term -- one month past the width-n_time
    grid :func:`tvog_weights` covers (whose entries weight start-of-month
    exits). It stayed in the fund that final month, credited
    ``max(return, guarantee)`` and charged the fee, so it carries one more month
    of the guarantee's discounted-growth factor. Returns the mean-over-scenarios
    less-central weight at the matured term, extending each path by that final
    month exactly as :func:`guarantee_floor_time_value` does for the GMAB
    (Sec. B119), so the deterministic GMAB, the floor time value and the
    credit-rate TVOG agree on the maturity date. ``0.0`` when there is no
    crediting guarantee -- the short-circuit avoids forming the
    over/underflowing cumulative product (0 * inf = NaN) on an
    extreme-but-valid near-ruin return path.
    """
    validate_crediting_rate(minimum_crediting_rate)
    return_scenarios = _validate_return_scenarios(return_scenarios)
    n_time = return_scenarios.shape[1]
    if minimum_crediting_rate == NO_GUARANTEE_RATE:
        return 0.0
    f_m = (1.0 + fund_fee) ** (1.0 / 12.0) - 1.0
    r_m = (1.0 + investment_return) ** (1.0 / 12.0) - 1.0
    central = np.full((1, n_time), r_m)
    av_s, disc_s = _av_and_discount(
        credited_monthly_rate(return_scenarios, minimum_crediting_rate),
        return_scenarios, f_m
    )
    av_c, disc_c = _av_and_discount(
        credited_monthly_rate(central, minimum_crediting_rate), central, f_m
    )
    av_c, disc_c = av_c[0], disc_c[0]
    # One extra month -- the final month's credited growth net of fee on the AV
    # factor and one more month of return-discount -- so the factor lands on the
    # matured term column (mirror the GMAB extension in guarantee_floor_time_value).
    ext_s = (av_s[:, -1]
             * (1.0 + credited_monthly_rate(return_scenarios[:, -1],
                                            minimum_crediting_rate))
             * (1.0 - f_m)) * (disc_s[:, -1] / (1.0 + return_scenarios[:, -1]))
    ext_c = (av_c[-1]
             * (1.0 + credited_monthly_rate(r_m, minimum_crediting_rate))
             * (1.0 - f_m)) * (disc_c[-1] / (1.0 + r_m))
    return float(ext_s.mean() - ext_c)


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
    validate_crediting_rate(minimum_crediting_rate)
    return_scenarios = _validate_return_scenarios(return_scenarios)
    n_time = return_scenarios.shape[1]
    f_m = (1.0 + fund_fee) ** (1.0 / 12.0) - 1.0
    r_m = (1.0 + investment_return) ** (1.0 / 12.0) - 1.0
    central = np.full((1, n_time), r_m)

    # Account-value multiplier and discount under each scenario, and under the
    # central flat-return path (whose floor cost is the intrinsic value). The
    # account is credited max(return, guarantee), or the bare return when the
    # crediting guarantee is off (NO_GUARANTEE_RATE).
    av_s, disc_s = _av_and_discount(
        credited_monthly_rate(return_scenarios, minimum_crediting_rate),
        return_scenarios, f_m
    )
    av_c, disc_c = _av_and_discount(
        credited_monthly_rate(central, minimum_crediting_rate), central, f_m
    )
    av_c, disc_c = av_c[0], disc_c[0]

    # The GMAB strikes at maturity (time = term) -- the *matured* account value
    # after the final month's growth -- one month past the width-n_time path the
    # GMDB walks (whose entries are start-of-month values). Extend each path by
    # that final month's growth and discount so the maturity index (term) reads
    # the matured value and discounts to time term, matching the deterministic
    # intrinsic value.
    av_s_mat = np.concatenate(
        [av_s, (av_s[:, -1]
                * (1.0 + credited_monthly_rate(return_scenarios[:, -1],
                                               minimum_crediting_rate))
                * (1.0 - f_m))[:, None]], axis=1)
    disc_s_mat = np.concatenate(
        [disc_s, (disc_s[:, -1] / (1.0 + return_scenarios[:, -1]))[:, None]],
        axis=1)
    av_c_mat = np.append(
        av_c, av_c[-1]
        * (1.0 + credited_monthly_rate(r_m, minimum_crediting_rate))
        * (1.0 - f_m))
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
    projection horizon. The model points must carry a crediting guarantee --
    a ``minimum_crediting_rate`` of ``0.0`` (a real 0% floor, ``max(return, 0)``)
    or a positive rate. A contract with no crediting guarantee
    (``NO_GUARANTEE_RATE``) is rejected: it has no credited-rate time value to
    measure (use ``vfa.measure(..., return_scenarios).time_value`` for the
    GMDB / GMAB floor time value instead). In v1 the rate is taken as a
    portfolio-wide scalar (per-MP varying rates with stochastic returns are a
    future extension), so the column is required to be uniform across rows.

    The guarantee cost is the present value of account-value benefits in
    excess of the no-guarantee benefits. Its mean over the scenarios is the
    total value; the cost in the central scenario (``investment_return``) is
    the intrinsic value; the difference is the time value (TVOG).

    Maturity survivors are weighted at the matured term (time = term, one month
    past their term - 1 exit column), the same re-seat the folded credit-rate
    TVOG in :func:`vfa.measure` uses, so the two agree on a mixed-term book.
    """
    validate_crediting_rate(model_points.minimum_crediting_rate)
    g_unique = np.unique(np.asarray(model_points.minimum_crediting_rate,
                                     dtype=np.float64))
    if g_unique.size > 1:
        raise NotImplementedError(
            "measure_tvog requires a uniform minimum_crediting_rate across "
            "model points in v1; per-MP varying rates with stochastic "
            "returns are a future extension"
        )
    if g_unique.size == 0 or float(g_unique[0]) == NO_GUARANTEE_RATE:
        raise ValueError(
            "measure_tvog values a credited-rate guarantee, and this contract "
            "carries none (minimum_crediting_rate == NO_GUARANTEE_RATE). A 0.0 "
            "rate is a real 0% floor and is valued here; the GMDB / GMAB "
            "account-value floors are valued by "
            "vfa.measure(..., return_scenarios).time_value instead."
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

    # Maturity survivors exit at time = term, not term - 1; weight them at the
    # matured-term discounted-growth column (one month past the n_time grid) so
    # this standalone path matches the folded measure_vfa credit-rate TVOG
    # (test_vfa_tvog_matches_measure_tvog). Boundary-clamp the maturity column
    # exactly as the deterministic VFA path. Peel the per-MP maturity
    # account-value mass out of exit_value BEFORE the term re-seat (a mixed-term
    # book has the maturity slice at different columns per model point).
    boundary_idx = model_points.contract_boundary_months - 1
    within = (model_points.term_months - 1) <= boundary_idx
    term_idx = np.where(within, model_points.term_months - 1, boundary_idx)
    maturity_survivors = np.where(within, proj.maturity_survivors, 0.0)
    mat_value = model_points.account_value * maturity_survivors          # (n_mp,)
    nm_exit_value = exit_value.copy()
    np.add.at(nm_exit_value, term_idx, -mat_value)

    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0

    # Without a guarantee the return cancels between growth and discount, so the
    # no-guarantee benefit is identical in every scenario. Non-maturity exits on
    # the n_time fee grid; the maturity slice takes one extra (1 - f_m) month (it
    # stays in the fund to time = term), keeping the intrinsic value an
    # apples-to-apples central comparison.
    fee_grid = (1.0 - f_m) ** np.arange(n_time)
    no_guarantee = float(nm_exit_value @ fee_grid
                         + (mat_value * (1.0 - f_m) ** (term_idx + 1)).sum())

    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    central = np.full((1, n_time), r_m)

    # Stochastic: credit max(return, guarantee), discount at the return. Extend
    # each path one month so the maturity slice reads the matured term column.
    dg_s = _discounted_growth(
        credited_monthly_rate(return_scenarios, g_annual), return_scenarios, f_m)
    ext_s = (dg_s[:, -1]
             * (1.0 + credited_monthly_rate(return_scenarios[:, -1], g_annual))
             * (1.0 - f_m) / (1.0 + return_scenarios[:, -1]))
    dg_s_mat = np.concatenate([dg_s, ext_s[:, None]], axis=1)     # (n_scen, n_time+1)
    pv_stochastic = dg_s @ nm_exit_value + dg_s_mat[:, term_idx + 1] @ mat_value
    guarantee_cost = pv_stochastic - no_guarantee

    # Deterministic central scenario -- a flat return path, the intrinsic value.
    dg_c = _discounted_growth(
        credited_monthly_rate(central, g_annual), central, f_m)[0]
    ext_c = (dg_c[-1] * (1.0 + credited_monthly_rate(r_m, g_annual))
             * (1.0 - f_m) / (1.0 + r_m))
    dg_c_mat = np.append(dg_c, ext_c)                            # (n_time+1,)
    pv_central = float(dg_c @ nm_exit_value + dg_c_mat[term_idx + 1] @ mat_value)
    intrinsic_value = float(pv_central - no_guarantee)

    time_value = float(guarantee_cost.mean() - intrinsic_value)
    return TVOGResult(
        guarantee_cost=guarantee_cost,
        intrinsic_value=intrinsic_value,
        time_value=time_value,
    )
