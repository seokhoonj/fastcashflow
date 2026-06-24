"""Solvency capital -- a regime-agnostic required-capital (SCR) layer.

A risk-based solvency regime (K-ICS, Solvency II) sets required capital by a
shock-and-aggregate recipe: stress one best-estimate assumption, re-measure the
liability, and take the increase in the liability (the loss in net asset value)
as that sub-risk's capital; then combine the sub-risks through a correlation
matrix. The two regimes share this structure exactly -- only the shock
magnitudes, the correlation cells, and the risk-margin method differ -- so this
module is the regime-agnostic engine and a :class:`RegimeSpec` carries the
calibration (the K-ICS and Solvency II specs live in this module).

The capital for one sub-risk is::

    C_i = max( BEL(stressed) - BEL(base), 0 )

computed by re-running :func:`fastcashflow.gmm.measure` on a stressed
:class:`~fastcashflow.basis.Basis` / :class:`~fastcashflow.model_points.ModelPoints`
(BEL is liability-positive, so an adverse stress raises it). The module capital is
``sqrt(c^T R c)`` over the sub-risk capital vector ``c`` and the regime
correlation matrix ``R``. The result feeds
:func:`fastcashflow.embedded_value` as the required-capital input.

Scope (v1): the liability-side modules a cash-flow engine can shock -- life /
long-term underwriting sub-risks (mortality, longevity, morbidity / disability,
lapse incl. mass lapse, expense) and interest-rate risk. Out of scope (no asset
model): equity / property / credit / operational risk, available capital, and the
top-level inter-module BSCR matrix. Catastrophe (a factor charge under K-ICS) is
also out of scope. These are deferred, not approximated silently.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.coverage import CalculationMethod
from fastcashflow.curves import discount_monthly_curve
from fastcashflow.gmm._engine import measure
from fastcashflow._measurement.inforce import inforce_surrender_value
from fastcashflow.model_points import ModelPoints
from fastcashflow.numerics import _cost_of_capital_ra

Transform = Callable[[ModelPoints, Basis], "tuple[ModelPoints, Basis]"]
# A scalar BEL adjustment a stress adds AFTER re-measurement, evaluated on the
# BASE (model_points, basis) -- used for a one-time shock-date cash flow the
# (model_points, basis) transform cannot express (e.g. the mass-lapse surrender
# value paid to the policies that leave at the valuation date).
BelAddon = Callable[[ModelPoints, Basis], float]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Stress:
    """One prescribed shock -- a labelled ``(model_points, basis)`` transform.

    ``apply`` returns a stressed ``(model_points, basis)`` pair; re-measuring it
    and differencing against the base gives the shock's loss in net asset value.

    ``bel_addon`` (optional) adds a scalar to the stressed BEL after
    re-measurement -- a one-time shock-date outflow the transform cannot carry
    (the mass-lapse surrender value). It is evaluated on the BASE
    ``(model_points, basis)``; ``None`` for a pure re-measure shock.
    """

    name: str
    apply: Transform
    bel_addon: BelAddon | None = None


@dataclass(frozen=True, slots=True)
class SubRisk:
    """A capital sub-risk: one or more stress variants and how to combine them.

    ``combine`` is ``"single"`` (one variant) or ``"worst_of"`` (the largest loss
    over the variants -- e.g. lapse takes the worst of up / down / mass lapse).
    ``name`` is the canonical key and MUST match the regime correlation axis.
    """

    name: str
    variants: tuple[Stress, ...]
    combine: str = "single"


@dataclass(frozen=True, slots=True, eq=False)
class RegimeSpec:
    """A solvency regime's calibration (K-ICS, Solvency II).

    ``sub_risks`` order is locked to the ``correlation`` matrix axes.
    ``interest_curves`` is a tuple of interest-rate :class:`Stress` (worst taken),
    or ``None`` when the regime's interest scenarios are caller-supplied. The
    risk margin is either ``"percentile"`` (``insurance_scr * risk_margin_factor``)
    or ``"cost_of_capital"`` (``risk_margin_coc_rate`` over the capital run-off).

    ``catastrophe_correlation`` and ``property_correlation`` are the catastrophe /
    long-term-property sub-risks' correlation with each of the ``sub_risks`` (same
    order), used when a caller passes a catastrophe charge / property coverage codes
    to :func:`required_capital`; ``None`` means the regime does not fold that
    (extra) sub-risk into its insurance module.
    """

    name: str
    sub_risks: tuple[SubRisk, ...]
    correlation: FloatArray
    interest_curves: tuple[Stress, ...] | None = None
    risk_margin_method: str = "percentile"
    risk_margin_factor: float = 0.0
    risk_margin_coc_rate: float = 0.06
    catastrophe_correlation: tuple[float, ...] | None = None
    property_correlation: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        # The capital vector is built positionally from sub_risks, so the
        # correlation axes must line up by position. Validate the structural
        # invariants (a transposed-but-same-size reorder is the author's
        # contract -- sub_risks order MUST match the matrix axis order).
        R = np.asarray(self.correlation, dtype=np.float64)
        k = len(self.sub_risks)
        if R.shape != (k, k):
            raise ValueError(
                f"correlation must be ({k}, {k}) to match {k} sub_risks, got {R.shape}")
        if not np.allclose(R, R.T):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(R), 1.0):
            raise ValueError("correlation diagonal must be 1.0")


@dataclass(frozen=True, slots=True, eq=False)
class SCRResult:
    """The required-capital breakdown for a portfolio under one regime.

    ``sub_risk_capital`` is the per-sub-risk capital (pre-aggregation);
    ``insurance_scr`` the correlation-aggregated insurance module;
    ``interest_capital`` the worst-of interest stress; ``total_scr`` their sum
    (v1: no inter-module diversification). ``scr_path`` is the projected capital
    run-off used for a cost-of-capital risk margin (``None`` for percentile).
    """

    regime: str
    sub_risk_capital: dict[str, float]
    insurance_scr: float
    interest_capital: float
    total_scr: float
    risk_margin: float
    base_bel: float
    scr_path: FloatArray | None = None


# ---------------------------------------------------------------------------
# Stress constructors -- each returns a Stress whose apply() rebuilds the inputs
# ---------------------------------------------------------------------------

def _scaled(fn, factor: float):
    """Wrap a 5-arg rate callable, scaling its output by ``factor`` and clamping
    to 1.0 (rates are probabilities). The factor is captured by closure -- never
    as a default argument (the engine reads a 4th positional as issue_class)."""
    def wrapped(sex, issue_age, duration, issue_class, elapsed):
        return np.minimum(fn(sex, issue_age, duration, issue_class, elapsed) * factor, 1.0)
    return wrapped


def _added_first_year(fn, addend: float):
    """Wrap a 5-arg rate callable, ADDING ``addend`` in the first policy year only
    (``duration == 0``) and clamping to 1.0. The time axis is the per-year duration
    grid, so ``duration == 0`` is the next 12 months -- the window the Solvency II
    life-catastrophe shock (Art 143) applies to. The factor is captured by closure."""
    def wrapped(sex, issue_age, duration, issue_class, elapsed):
        base = fn(sex, issue_age, duration, issue_class, elapsed)
        bump = np.where(np.asarray(duration) == 0, addend, 0.0)
        return np.minimum(base + bump, 1.0)
    return wrapped


def _split_first_year(fn, first_factor: float, later_factor: float):
    """Wrap a 5-arg rate callable, scaling its output by ``first_factor`` in the
    first policy year (``duration == 0``) and by ``later_factor`` thereafter,
    clamping to 1.0. The per-year duration grid means ``duration == 0`` is the
    next 12 months -- the shape of the Solvency II disability inception shock
    (Art 139: +35% over the next year, +25% thereafter). Factors by closure."""
    def wrapped(sex, issue_age, duration, issue_class, elapsed):
        base = fn(sex, issue_age, duration, issue_class, elapsed)
        factor = np.where(np.asarray(duration) == 0, first_factor, later_factor)
        return np.minimum(base * factor, 1.0)
    return wrapped


def _classified_methods(model_points: ModelPoints, basis: Basis) -> dict:
    """The coverage -> :class:`CalculationMethod` map, validated to cover EVERY
    coverage code, so a mortality / morbidity stress can never silently skip an
    unclassified coverage (which would understate the capital). Returns ``{}`` when
    the basis has no coverages; otherwise raises if the map is missing or
    incomplete -- the engine itself needs the same classification to measure."""
    codes = {r.code for r in basis.coverages}
    if not codes:
        return {}
    methods = model_points.calculation_methods
    if methods is None:
        raise ValueError(
            "this stress needs ModelPoints.calculation_methods to classify the "
            f"coverages {sorted(codes)} (cannot tell which are DEATH / morbidity)")
    missing = codes - set(methods)
    if missing:
        raise ValueError(
            f"coverages {sorted(missing)} are not in calculation_methods; cannot "
            "classify them for the stress")
    return methods


def scale_mortality(factor: float) -> Stress:
    """Stress mortality up by ``factor`` -- BOTH the in-force decrement
    (``mortality_annual``) and every DEATH coverage's claim rate. Scaling only the
    decrement would raise survivorship without raising death claims; the death
    claim is driven by a separate coverage rate. Raises if the coverages cannot be
    classified (no silent under-scaling)."""
    def apply(mp: ModelPoints, basis: Basis):
        methods = _classified_methods(mp, basis)
        coverages = tuple(
            replace(r, rate=_scaled(r.rate, factor))
            if methods.get(r.code) == CalculationMethod.DEATH else r
            for r in basis.coverages)
        new_basis = replace(basis,
                            mortality_annual=_scaled(basis.mortality_annual, factor),
                            coverages=coverages)
        return mp, new_basis
    return Stress(name=f"mortality x{factor:g}", apply=apply)


def catastrophe_mortality(addend: float = 0.0015) -> Stress:
    """Solvency II life-catastrophe stress (Delegated Regulation Art 143) -- an
    instantaneous ABSOLUTE addition of ``addend`` (0.15 percentage points = 0.0015
    by default) to the mortality rates over the next 12 months (the first policy
    year). Like :func:`scale_mortality` it lifts BOTH the in-force decrement
    (``mortality_annual``) and every DEATH coverage's claim rate, but the shift is
    absolute and one-year, not a permanent relative scaling. Raises if the
    coverages cannot be classified (no silent under-scaling)."""
    def apply(mp: ModelPoints, basis: Basis):
        methods = _classified_methods(mp, basis)
        coverages = tuple(
            replace(r, rate=_added_first_year(r.rate, addend))
            if methods.get(r.code) == CalculationMethod.DEATH else r
            for r in basis.coverages)
        new_basis = replace(
            basis,
            mortality_annual=_added_first_year(basis.mortality_annual, addend),
            coverages=coverages)
        return mp, new_basis
    return Stress(name=f"mortality cat +{addend:g}", apply=apply)


def scale_longevity(factor: float) -> Stress:
    """Stress mortality DOWN by ``factor`` (< 1) on the in-force decrement only --
    the longevity loss arises from survivors drawing annuity / maturity benefits
    longer, not from death claims, so the DEATH coverage rate is left unchanged."""
    def apply(mp: ModelPoints, basis: Basis):
        return mp, replace(basis, mortality_annual=_scaled(basis.mortality_annual, factor))
    return Stress(name=f"longevity x{factor:g}", apply=apply)


def scale_lapse(factor: float) -> Stress:
    """Stress the ongoing lapse / surrender rate by ``factor`` (option-exercise
    lapse up / down)."""
    def apply(mp: ModelPoints, basis: Basis):
        return mp, replace(basis, lapse_annual=_scaled(basis.lapse_annual, factor))
    return Stress(name=f"lapse x{factor:g}", apply=apply)


# Named in-force state-machine transition rates that live as their own Basis
# field. A stress scales one by the name a Transition.rate carries -- e.g. the DI
# recovery edge (disabled -> active) or a morbidity / waiver inception edge.
# mortality and lapse are the core decrements with their own stresses.
_STATE_RATE_FIELD = {
    "waiver_incidence":    "waiver_incidence_annual",
    "ci_incidence":        "ci_incidence_annual",
    "ci_reincidence":      "ci_reincidence_annual",
    "disability_recovery": "disability_recovery_annual",
    "lapse_paidup":        "lapse_paidup_annual",
}


def scale_state_rate(rate_name: str, factor: float) -> Stress:
    """Scale a named in-force state-machine transition rate by ``factor``.

    ``rate_name`` is the string a :class:`~fastcashflow.state_model.Transition`
    carries -- e.g. ``"disability_recovery"`` for the disability-income recovery
    edge (disabled -> active), or ``"waiver_incidence"`` / ``"ci_incidence"`` for
    a disability / morbidity inception edge. The rate lives in its own ``Basis``
    field (mapped here); the stress rebuilds the basis with that one field scaled,
    clamped to 1.0. A basis that does not supply the rate (the field is ``None``)
    is left unchanged -- the shock is a no-op for a product without that
    transition. ``mortality`` and ``lapse`` are the core decrements with their own
    stresses (:func:`scale_mortality` / :func:`scale_lapse`); this is for the
    optional transition rates a multi-state product declares.

    A recovery shock is a DOWN scale (``factor < 1`` -- fewer recoveries, so the
    disabled stay on claim longer and the liability rises); an inception shock is
    an UP scale. Used to build the Solvency II disability sub-risk (Art. 139: a
    decrease in disability recovery rates, an increase in inception rates)."""
    try:
        field = _STATE_RATE_FIELD[rate_name]
    except KeyError:
        raise ValueError(
            f"unknown state-machine rate {rate_name!r}; known rates are "
            f"{sorted(_STATE_RATE_FIELD)}") from None
    def apply(mp: ModelPoints, basis: Basis):
        fn = getattr(basis, field)
        if fn is None:                       # product has no such transition
            return mp, basis
        return mp, replace(basis, **{field: _scaled(fn, factor)})
    return Stress(name=f"{rate_name} x{factor:g}", apply=apply)


def mass_lapse(fraction: float) -> Stress:
    """An instantaneous mass lapse -- ``fraction`` of the in-force surrenders at
    the valuation date.

    Two effects, both captured. (1) The remaining ``(1 - fraction)`` in-force is
    a count haircut on ``ModelPoints.count`` -- its future cash flows fall
    proportionally (the lost-business value). (2) The leaving ``fraction`` is
    paid its surrender value at the shock date -- a one-time outflow the count
    haircut cannot express; it is added via :class:`Stress.bel_addon` as
    ``fraction x sum(count x surrender_value)`` at the valuation date (see
    :func:`fastcashflow._measurement.inforce.inforce_surrender_value`). Together the stressed
    liability is ``BEL(remaining) + surrender_value(leaving)``, so the capital is
    ``fraction x sum(count x (surrender_value - per-policy BEL))``.

    Zero one-time outflow when the basis prices no surrender value (the count
    haircut alone, the historical behaviour). Account-backed (universal-life)
    books pay the account value net of the surrender charge at the shock date --
    ``inforce_surrender_value`` reads it from the account roll."""
    def apply(mp: ModelPoints, basis: Basis):
        n = mp.n_mp
        count = np.ones(n) if mp.count is None else np.asarray(mp.count, float)
        return replace(mp, count=count * (1.0 - fraction)), basis

    def addon(mp: ModelPoints, basis: Basis) -> float:
        return fraction * float(inforce_surrender_value(mp, basis).sum())

    return Stress(name=f"mass lapse {fraction:g}", apply=apply, bel_addon=addon)


def dynamic_lapse_multiplier(rate_shock: float, sensitivity: float) -> float:
    """The dynamic-lapse multiplier for a parallel rate move -- the lapse<->rate
    coupling in one number.

    A fixed linear form ``1 + sensitivity * rate_shock`` (floored at 0): when market
    rates rise, policyholders surrender more to chase the higher yield, so a positive
    ``sensitivity`` lifts the lapse rate (and a rate fall lowers it). ``rate_shock``
    is the parallel shift as a fraction (``+0.01`` = +100bp); ``sensitivity`` is the
    injected coefficient -- the lapse elasticity to a unit rate move. The FORM is
    fixed and only the coefficient varies, so the engine resolves it to ONE scalar
    multiplier up front and the kernel sees only the scaled lapse array -- the seam
    costs nothing in the hot loop (no per-step callback)."""
    return max(0.0, 1.0 + sensitivity * rate_shock)


def interest_with_dynamic_lapse(shift: float, sensitivity: float, *,
                                name: str | None = None) -> Stress:
    """A coupled interest-rate / dynamic-lapse stress -- the asset-liability
    interaction in one ``(model_points, basis)`` transform.

    Applies a parallel ``shift`` (absolute, ``+0.01`` for +100bp) to the discount
    curve AND scales the lapse rate by :func:`dynamic_lapse_multiplier`
    (``shift``, ``sensitivity``) -- the lapse the same rate move induces.
    Re-measuring and differencing against the base gives the liability-side loss
    INCLUDING the lapse<->rate correlation; ``sensitivity = 0`` collapses it to a
    pure parallel rate shock (the uncoupled view), so the coupling's marginal effect
    is the difference between the two. ``sensitivity`` is the seam (the injected
    elasticity of the fixed linear form); a relative regime curve is a separate
    construct (:func:`shock_curve`) -- this is the analysis parallel shift used with
    the dynamic asset engine."""
    mult = dynamic_lapse_multiplier(shift, sensitivity)
    def apply(mp: ModelPoints, basis: Basis):
        base = np.asarray(basis.discount_annual, float)
        return mp, replace(basis, discount_annual=base + shift,
                           lapse_annual=_scaled(basis.lapse_annual, mult))
    nm = name or f"rate {shift:+g} + dyn-lapse x{mult:g}"
    return Stress(name=nm, apply=apply)


def scale_annuity(factor: float) -> Stress:
    """Scale the annuity benefit amount by ``factor`` (revision risk -- the risk
    that in-payment annuity benefits are revised upward). A no-op for model points
    that carry no annuity payment."""
    def apply(mp: ModelPoints, basis: Basis):
        if mp.annuity_payment is None:
            return mp, basis
        return replace(mp, annuity_payment=np.asarray(mp.annuity_payment, float) * factor), basis
    return Stress(name=f"annuity x{factor:g}", apply=apply)


def scale_coverages(factor_by_method: dict[CalculationMethod, float]) -> Stress:
    """Scale each coverage's claim rate by the factor for its
    :class:`CalculationMethod` (e.g. ``{MORBIDITY: 1.10, DIAGNOSIS: 1.13}`` for a
    morbidity / disability stress). Coverages whose method is absent are left
    unchanged. The in-force decrement is untouched."""
    def apply(mp: ModelPoints, basis: Basis):
        methods = _classified_methods(mp, basis)
        coverages = tuple(
            replace(r, rate=_scaled(r.rate, factor_by_method[methods[r.code]]))
            if methods[r.code] in factor_by_method else r
            for r in basis.coverages)
        return mp, replace(basis, coverages=coverages)
    return Stress(name="coverage rates", apply=apply)


def scale_coverages_first_year(first_by_method: dict, later_by_method: dict) -> Stress:
    """Scale each coverage's claim rate by a DURATION-split factor: the first
    policy year (``duration == 0``) by ``first_by_method[method]`` and later years
    by ``later_by_method[method]`` -- the +35% next-12-months / +25%-thereafter
    shape of the Solvency II disability / morbidity inception shock (Art 139).

    The two dicts must carry the same methods (the first-year and steady factors
    for each). Coverages whose method is in neither are left unchanged; the
    in-force decrement is untouched. ``duration == 0`` is the next 12 months only
    for business entering the projection at inception (the same caveat as the
    Art 143 life-catastrophe first-year shock)."""
    if set(first_by_method) != set(later_by_method):
        raise ValueError(
            "first_by_method and later_by_method must carry the same methods; got "
            f"{sorted(first_by_method)} vs {sorted(later_by_method)}")
    def apply(mp: ModelPoints, basis: Basis):
        methods = _classified_methods(mp, basis)
        coverages = tuple(
            replace(r, rate=_split_first_year(
                r.rate, first_by_method[methods[r.code]], later_by_method[methods[r.code]]))
            if methods[r.code] in first_by_method else r
            for r in basis.coverages)
        return mp, replace(basis, coverages=coverages)
    return Stress(name="coverage rates (first-year split)", apply=apply)


def scale_coverage_codes(codes, factor: float) -> Stress:
    """Scale the claim rate of the named coverage ``codes`` by ``factor`` (the
    others untouched). Used for the long-term property / other sub-risk, whose
    coverages are identified by code (the property categorisation is finer than the
    engine's :class:`CalculationMethod`)."""
    targets = set(codes)
    def apply(mp: ModelPoints, basis: Basis):
        coverages = tuple(
            replace(r, rate=_scaled(r.rate, factor)) if r.code in targets else r
            for r in basis.coverages)
        return mp, replace(basis, coverages=coverages)
    return Stress(name="property coverage rates", apply=apply)


def scale_expense(level: float = 1.10, inflation_add: float = 0.01) -> Stress:
    """Stress expenses -- scale every expense item's value by ``level`` and add
    ``inflation_add`` (percentage points, as a decimal) to the expense inflation
    rate."""
    def apply(mp: ModelPoints, basis: Basis):
        items = tuple(replace(it, value=it.value * level) for it in basis.expense_items)
        infl = basis.expense_inflation
        new_infl = (np.asarray(infl, float) + inflation_add
                    if np.ndim(infl) else float(infl) + inflation_add)
        return mp, replace(basis, expense_items=items, expense_inflation=new_infl)
    return Stress(name=f"expense x{level:g}+{inflation_add:g}", apply=apply)


def shock_curve(rel_by_maturity: FloatArray, *, up: bool,
                floor_pp: float = 0.0, zero_floor: bool = False,
                name: str | None = None) -> Stress:
    """Stress the risk-free discount curve by a maturity-dependent RELATIVE shock.

    ``rel_by_maturity`` is the per-year relative shock (e.g. +0.70 at 1y); it is
    held flat past its end. For an up shock the result is floored at
    ``base + floor_pp`` (Solvency II's +1pp minimum). For a down shock,
    ``zero_floor`` leaves already-negative base rates unshocked (Solvency II)."""
    rel = np.asarray(rel_by_maturity, float)

    def apply(mp: ModelPoints, basis: Basis):
        base = np.asarray(basis.discount_annual, float)
        if base.ndim == 0:
            base = np.full(rel.shape[0], float(base))
        m = base.shape[0]
        r = rel[:m] if rel.shape[0] >= m else np.concatenate(
            [rel, np.full(m - rel.shape[0], rel[-1])])
        shocked = base * (1.0 + r)
        if up:
            shocked = np.maximum(shocked, base + floor_pp)
        elif zero_floor:
            shocked = np.where(base < 0.0, base, shocked)
        return mp, replace(basis, discount_annual=shocked)
    return Stress(name=name or ("interest up" if up else "interest down"), apply=apply)


def shock_spread(spread_by_maturity: FloatArray, *, name: str,
                 compounding: str = "annual") -> Stress:
    """Stress the risk-free discount curve by a maturity-by-maturity shock spread.

    ``spread_by_maturity`` is the per-year shock spread (year ``y`` in entry
    ``y - 1``); it is held flat past its last entry. This is the form the K-ICS
    interest-rate scenarios take: the shocked curve is the base risk-free curve
    plus the supervisor-published shock spread (the difference between the adjusted
    risk-free term structures before and after the shock; handbook 4-2.(1)-7), as
    opposed to the relative shock :func:`shock_curve` applies for the Solvency II
    maturity-relative table.

    ``compounding`` selects how the spread meets the annual-compounded curve:
    ``"annual"`` (default) adds it directly, ``base + spread`` (the spread is in
    :attr:`~fastcashflow.Basis.discount_annual` units -- a ``+0.473`` percentage-
    point spread is ``0.00473``); ``"continuous"`` treats the spread as a
    continuous-compounding shock and applies it in that space,
    ``(1 + base) * exp(spread) - 1``. The FSS-published K-ICS shock-spread tables
    are continuous-compounding, so ``compounding="continuous"`` consumes them
    directly; a plain additive application would understate the shock by a level-
    dependent amount (~the spread times the base, per maturity).
    """
    if compounding not in ("annual", "continuous"):
        raise ValueError("compounding must be 'annual' or 'continuous'")
    spread = np.asarray(spread_by_maturity, float)

    def apply(mp: ModelPoints, basis: Basis):
        base = np.asarray(basis.discount_annual, float)
        if base.ndim == 0:
            base = np.full(spread.shape[0], float(base))
        m = base.shape[0]
        s = spread[:m] if spread.shape[0] >= m else np.concatenate(
            [spread, np.full(m - spread.shape[0], spread[-1])])
        shocked = (1.0 + base) * np.exp(s) - 1.0 if compounding == "continuous" else base + s
        return mp, replace(basis, discount_annual=shocked)
    return Stress(name=name, apply=apply)


@dataclass(frozen=True, slots=True)
class KICSInterest:
    """The five K-ICS interest-rate shock scenarios and their aggregation.

    Each field is a :class:`Stress` adding the official maturity-by-maturity shock
    spread to the risk-free discount curve: the level factor ``up`` / ``down``
    (LTFR +/-15bp; handbook 4-2.(1)-6), the slope factor ``flat`` / ``steep``
    (short-end up & long-end down / short-end down & long-end up), and
    ``mean_reversion``. The shock spreads are published by the supervisor
    (handbook 4-2.(1)-7), so build the scenarios from them with
    :meth:`from_spreads`; do not bake a table in.

    The interest-rate capital aggregates the five by the handbook (p.205) formula

        sqrt( max(up, down)^2 + max(flat, steep)^2 ) + mean_reversion

    The level pair and the twist pair are independent (correlation 0), hence the
    sum-of-squares under the root; ``up`` / ``down`` are mutually exclusive
    directions, so the worst of each pair is taken. Each directional amount is the
    net-asset-value DECREASE under that scenario floored at zero
    (``max(Delta BEL, 0)`` for a liability-only book; the asset leg is zero here).
    The mean-reversion amount is SIGNED -- it can raise OR lower the charge
    (handbook 4-2.(1)-5), so it is added outside the root without a floor.
    """

    up: Stress
    down: Stress
    flat: Stress
    steep: Stress
    mean_reversion: Stress

    @classmethod
    def from_spreads(cls, *, up: FloatArray, down: FloatArray, flat: FloatArray,
                     steep: FloatArray, mean_reversion: FloatArray,
                     compounding: str = "annual") -> "KICSInterest":
        """Build the five scenarios from per-maturity shock-spread arrays.

        Each argument is the shock spread by maturity (year 1 in entry 0).
        ``compounding`` is passed to :func:`shock_spread` for every scenario:
        ``"annual"`` (default) adds the spread to the annual-compounded curve;
        ``"continuous"`` applies it in continuous-compounding space, the form the
        FSS-published K-ICS shock-spread tables take. The FSS tables give only the
        ``up`` / ``flat`` scenarios, with ``down = -up`` and ``steep = -flat``.
        """
        return cls(
            up=shock_spread(up, name="interest up", compounding=compounding),
            down=shock_spread(down, name="interest down", compounding=compounding),
            flat=shock_spread(flat, name="interest flat", compounding=compounding),
            steep=shock_spread(steep, name="interest steep", compounding=compounding),
            mean_reversion=shock_spread(mean_reversion, name="interest mean reversion",
                                        compounding=compounding),
        )

    def capital(self, delta) -> tuple[float, dict[str, float]]:
        """The interest-rate capital and its scenario breakdown.

        ``delta`` maps a :class:`Stress` to the net-asset-value change it causes
        (``Delta BEL`` for the liability leg). Returns ``(capital, components)``,
        where ``components`` carries the five floored / signed scenario amounts
        keyed ``interest_up`` ... ``interest_mean_reversion`` (for the SCR
        breakdown). The capital is the handbook (p.205) aggregation."""
        up = max(0.0, delta(self.up))
        down = max(0.0, delta(self.down))
        flat = max(0.0, delta(self.flat))
        steep = max(0.0, delta(self.steep))
        mr = delta(self.mean_reversion)                  # signed -- can be negative
        level = max(up, down)
        twist = max(flat, steep)
        cap = float(np.sqrt(level * level + twist * twist) + mr)
        components = {
            "interest_up": up, "interest_down": down,
            "interest_flat": flat, "interest_steep": steep,
            "interest_mean_reversion": mr,
        }
        return cap, components


# ---------------------------------------------------------------------------
# Aggregation + the public entry point
# ---------------------------------------------------------------------------

def aggregate(capital: dict[str, float], spec: RegimeSpec) -> float:
    """Correlation-aggregate the sub-risk capitals: ``sqrt(c^T R c)`` with the
    capital vector ordered by ``spec.sub_risks`` and the regime matrix ``R``."""
    c = np.array([capital[sr.name] for sr in spec.sub_risks], dtype=np.float64)
    R = np.asarray(spec.correlation, dtype=np.float64)
    return float(np.sqrt(c @ R @ c))


def _aggregate_with_extras(base_caps, base_corr, extras) -> float:
    """``sqrt(c^T R c)`` over the base sub-risks plus the ``extras`` (each a
    ``(name, capital, correlation-vs-base)`` tuple). The base block is
    ``base_corr``; an extra correlates with the base by its vector and with another
    extra via :data:`_EXTRA_CROSS` (0 if unlisted)."""
    n = len(base_caps)
    m = len(extras)
    R = np.eye(n + m)
    R[:n, :n] = np.asarray(base_corr, dtype=np.float64)
    c = list(base_caps)
    for i, (name, cap, corr_base) in enumerate(extras):
        c.append(cap)
        R[:n, n + i] = R[n + i, :n] = np.asarray(corr_base, dtype=np.float64)
        for j in range(i):
            other = extras[j][0]
            rho = _EXTRA_CROSS.get((other, name), _EXTRA_CROSS.get((name, other), 0.0))
            R[n + j, n + i] = R[n + i, n + j] = rho
    c = np.array(c, dtype=np.float64)
    return float(np.sqrt(c @ R @ c))


def ratio(scr: SCRResult, available_capital: float) -> float:
    """The solvency ratio -- available capital over the required capital
    (``available_capital / scr.total_scr``).

    ``available_capital`` is a CALLER INPUT: the market value of assets less the
    market value of liabilities (on the prudential balance sheet), tiered per the
    regime. fastcashflow is a liability engine with no asset model, so it cannot
    produce the available capital itself -- supply it (e.g. from an asset system).
    The denominator is the liability-side required capital this module computes;
    asset-side market-risk modules are out of scope, so for a book with material
    asset risk the ratio is an upper bound on the regulatory one."""
    return available_capital / scr.total_scr


# K-ICS catastrophe factors (handbook 2-8). Pandemic = death sum assured x 0.1%.
# Large accident = death + disability + property (correlation 1, simple sum); each
# is a sum of zone-exposure x max(sum-assured x shock - prior-year claims, 0). The
# catastrophe amount is sqrt(pandemic^2 + large-accident^2) (correlation 0).
_PANDEMIC_FACTOR = 0.001
_ACCIDENT_TERMS = {            # category -> [(zone exposure ratio, shock), ...]
    "death":      ((0.0000711, 0.150), (0.0003733, 0.015)),
    "disability": ((0.0000711, 0.200), (0.0003733, 0.100)),
    "property":   ((0.0000711, 1.000), (0.0002133, 0.250), (0.0000160, 0.100)),
}


def kics_catastrophe(*, pandemic_death: float = 0.0, accident_death: float = 0.0,
                    disability: float = 0.0, property: float = 0.0,
                    prior_year_claims: dict[str, float] | None = None) -> float:
    """The K-ICS catastrophe risk amount (handbook 2-8) -- a factor on sum assured.

    ``pandemic_death`` is the sum assured of pandemic death-exposed coverages
    (charged 0.1%). ``accident_death`` / ``disability`` / ``property`` are the
    large-accident sum-assured buckets, each charged the zone-exposure factors
    against ``max(sum_assured x shock - prior_year_claims[bucket], 0)``. The result
    is ``sqrt(pandemic^2 + large_accident^2)`` (the two are uncorrelated). The
    exposure buckets are caller-supplied (the catastrophe categorisation of a
    coverage is a mapping decision, not derivable from the engine type)."""
    pyc = prior_year_claims or {}
    pandemic = max(0.0, pandemic_death) * _PANDEMIC_FACTOR     # exposure >= 0

    def accident(sa: float, key: str) -> float:
        claims = pyc.get(key, 0.0)
        return sum(ratio * max(sa * shock - claims, 0.0)
                   for ratio, shock in _ACCIDENT_TERMS[key])

    large = (accident(accident_death, "death") + accident(disability, "disability")
             + accident(property, "property"))                  # correlation 1: sum
    return float(np.sqrt(pandemic ** 2 + large ** 2))


_PROPERTY_SHOCK = 1.16         # K-ICS handbook 2-5: long-term property/other +16%


def required_capital(
    model_points: ModelPoints, basis: Basis, *, regime: RegimeSpec,
    catastrophe: float = 0.0, property_codes=(),
    interest_scenarios: KICSInterest | None = None,
    measure_fn: Callable | None = None,
) -> SCRResult:
    """Required capital (SCR) for a portfolio under ``regime``.

    Re-measures the liability under each sub-risk's stress, takes
    ``max(Delta BEL, 0)`` as the sub-risk capital, correlation-aggregates the
    insurance module, adds the interest-rate stress, and computes the regime risk
    margin. v1 is liability-side: the total is ``insurance_scr +
    interest_capital`` (no inter-module diversification). Pass ``SCRResult`` on to
    :func:`fastcashflow.embedded_value` via its ``required_capital`` argument.

    ``measure_fn`` is the liability measurement the stresses re-run (default the
    GMM :func:`~fastcashflow.gmm._engine.measure`); pass
    :func:`~fastcashflow.vfa.measure` to price a variable book's sub-risks on its
    net BEL (see :func:`vfa_required_capital`). It must accept ``(mp, basis,
    full=...)`` and return a result carrying ``bel`` (and ``ra_path`` / ``bel_path``
    for a cost-of-capital risk margin).

    Interest-rate capital comes from ``interest_scenarios`` when supplied -- a
    :class:`KICSInterest` (the five K-ICS shock scenarios, aggregated by the
    handbook p.205 formula); its five scenario amounts also land in
    ``sub_risk_capital`` (keys ``interest_up`` ...). Otherwise it is the worst-of
    ``regime.interest_curves`` (the Solvency II maturity-relative up / down table),
    or zero when neither is present. The K-ICS shock spreads are supervisor-
    published, so they are supplied at call time rather than baked into the regime.

    Two EXTRA insurance sub-risks fold into the module through table 6 when the
    regime supports them: ``property_codes`` (the long-term property / other
    coverages -- a +16% rate shock, re-measured) via ``property_correlation``, and
    ``catastrophe`` (the factor-based amount from :func:`kics_catastrophe`) via
    ``catastrophe_correlation``. The risk margin EXCLUDES catastrophe (handbook: the
    margin is the insurance amount ex-catastrophe), but INCLUDES property.
    """
    mf = measure_fn if measure_fn is not None else measure
    m_base = mf(model_points, basis, full=False)
    base_bel = float(m_base.bel.sum())

    def delta(stress: Stress) -> float:
        mp2, basis2 = stress.apply(model_points, basis)
        d = float(mf(mp2, basis2, full=False).bel.sum()) - base_bel
        if stress.bel_addon is not None:
            d += stress.bel_addon(model_points, basis)
        return d

    capital: dict[str, float] = {}
    for sr in regime.sub_risks:
        capital[sr.name] = max(0.0, max(delta(v) for v in sr.variants))
    base_caps = [capital[sr.name] for sr in regime.sub_risks]

    extras_margin = []          # property is in the risk-margin base; catastrophe is not
    if len(property_codes) and regime.property_correlation is not None:
        prop = max(0.0, delta(scale_coverage_codes(property_codes, _PROPERTY_SHOCK)))
        capital["property"] = prop
        extras_margin.append(("property", prop, regime.property_correlation))
    extras_all = list(extras_margin)
    if catastrophe > 0.0 and regime.catastrophe_correlation is not None:
        capital["catastrophe"] = catastrophe
        extras_all.append(("catastrophe", catastrophe, regime.catastrophe_correlation))

    insurance_ex_cat = _aggregate_with_extras(base_caps, regime.correlation, extras_margin)
    insurance_scr = _aggregate_with_extras(base_caps, regime.correlation, extras_all)

    interest_capital = 0.0
    if interest_scenarios is not None:
        interest_capital, interest_components = interest_scenarios.capital(delta)
        capital.update(interest_components)
    elif regime.interest_curves is not None:
        interest_capital = max(0.0, max(delta(s) for s in regime.interest_curves))

    total_scr = insurance_scr + interest_capital

    if regime.risk_margin_method == "percentile":
        risk_margin = insurance_ex_cat * regime.risk_margin_factor   # ex-catastrophe
        scr_path = None
    elif regime.risk_margin_method == "cost_of_capital":
        # The risk margin covers non-hedgeable (insurance / underwriting) risk;
        # interest-rate risk is excluded from its capital, so the run-off scales
        # the INSURANCE SCR, not the total. The run-off shape is proxied by the
        # confidence-level RA trajectory (the engine's own non-financial
        # risk-capital path). v1 approximation: the SCR run-off shape, not a full
        # SCR re-projection at each future month; clamped non-negative.
        m_full = mf(model_points, basis, full=True)
        driver = m_full.ra_path.sum(axis=0)
        d0 = float(driver[0])
        if d0 <= 0.0:
            driver = np.abs(m_full.bel_path.sum(axis=0))
            d0 = float(driver[0]) if driver[0] != 0.0 else 1.0
        scr_path = np.maximum(insurance_ex_cat * driver / d0, 0.0)
        n_time = scr_path.shape[0] - 1
        disc_m = discount_monthly_curve(basis, n_time)
        risk_margin = float(_cost_of_capital_ra(
            scr_path.reshape(1, -1), disc_m, regime.risk_margin_coc_rate)[0, 0])
    else:
        raise ValueError(
            "risk_margin_method must be 'percentile' or 'cost_of_capital', got "
            f"{regime.risk_margin_method!r}")

    return SCRResult(
        regime=regime.name, sub_risk_capital=capital, insurance_scr=insurance_scr,
        interest_capital=interest_capital, total_scr=total_scr,
        risk_margin=risk_margin, base_bel=base_bel, scr_path=scr_path)


# ---------------------------------------------------------------------------
# Solvency II calibration (Delegated Regulation (EU) 2015/35, primary source).
# Life underwriting sub-risks (Articles 137-143) and the life sub-risk
# correlation matrix (Article 136 / Annex IV). Catastrophe (Art 143, +0.15pp
# absolute mortality over the next 12 months) is the 7th sub-risk; its correlation
# row (Cat vs mortality / longevity / disability / expense / revision / lapse) is
# (0.25, 0, 0.25, 0.25, 0, 0.25). Order is locked to the Article 136 axes for the
# sub-risks present: mortality / longevity / disability / expense / revision /
# lapse / catastrophe.
# ---------------------------------------------------------------------------

_SII_CORRELATION = np.array([
    #  mortality  longevity  disability  expense  revision  lapse   catastrophe
    [   1.00,     -0.25,      0.25,      0.25,     0.00,    0.00,    0.25],   # mortality
    [  -0.25,      1.00,      0.00,      0.25,     0.25,    0.25,    0.00],   # longevity
    [   0.25,      0.00,      1.00,      0.50,     0.00,    0.00,    0.25],   # disability
    [   0.25,      0.25,      0.50,      1.00,     0.50,    0.50,    0.25],   # expense
    [   0.00,      0.25,      0.00,      0.50,     1.00,    0.00,    0.00],   # revision
    [   0.00,      0.25,      0.00,      0.50,     0.00,    1.00,    0.25],   # lapse
    [   0.25,      0.00,      0.25,      0.25,     0.00,    0.25,    1.00],   # catastrophe
])

# Interest-rate stress -- the EIOPA maturity-relative shock table (Art 166 up /
# Art 167 down), interpolated to a per-year array. Up is floored at +1pp; the
# down shock leaves already-negative base rates unshocked.
_SII_RATE_UP = [(1, 0.70), (2, 0.70), (3, 0.64), (5, 0.55),
                (10, 0.42), (15, 0.33), (20, 0.26), (90, 0.20)]
_SII_RATE_DOWN = [(1, -0.75), (2, -0.65), (3, -0.56), (5, -0.46), (10, -0.31),
                  (15, -0.27), (16, -0.28), (20, -0.29), (90, -0.20)]


def _per_year_rel(points, n_years: int = 60) -> FloatArray:
    """Interpolate (maturity, relative-shock) points to a per-year array."""
    mats = np.array([m for m, _ in points], dtype=np.float64)
    vals = np.array([v for _, v in points], dtype=np.float64)
    return np.interp(np.arange(1, n_years + 1), mats, vals)


# Solvency II disability-morbidity sub-risk (Art 139): one scenario applied
# together, then re-measured -- an INCREASE in disability / morbidity INCEPTION
# rates (+35% over the next 12 months, +25% thereafter) AND a -20% DECREASE in
# disability RECOVERY rates. Inception hits both the flat morbidity / diagnosis
# coverage claim rates and the state-machine inception edges (waiver / CI
# incidence, both morbidity inceptions); recovery hits the DI recovery edge
# (disabled -> active). A product without a given rate is left unchanged.
_SII_DISABILITY_INCEPTION_FIRST = 1.35
_SII_DISABILITY_INCEPTION_LATER = 1.25
_SII_DISABILITY_RECOVERY = 0.80
_SII_DISABILITY_INCEPTION_RATES = ("waiver_incidence", "ci_incidence")


def _sii_disability_shock() -> Stress:
    """The combined Art 139 disability shock (inception up, recovery down)."""
    cov = scale_coverages_first_year(
        {CalculationMethod.MORBIDITY: _SII_DISABILITY_INCEPTION_FIRST,
         CalculationMethod.DIAGNOSIS: _SII_DISABILITY_INCEPTION_FIRST},
        {CalculationMethod.MORBIDITY: _SII_DISABILITY_INCEPTION_LATER,
         CalculationMethod.DIAGNOSIS: _SII_DISABILITY_INCEPTION_LATER})

    def apply(mp: ModelPoints, basis: Basis):
        # 1. inception on the flat morbidity / diagnosis coverage claim rates
        mp, basis = cov.apply(mp, basis)
        over = {}
        # 2. inception on the state-machine inception edges (if present)
        for name in _SII_DISABILITY_INCEPTION_RATES:
            field = _STATE_RATE_FIELD[name]
            fn = getattr(basis, field)
            if fn is not None:
                over[field] = _split_first_year(
                    fn, _SII_DISABILITY_INCEPTION_FIRST, _SII_DISABILITY_INCEPTION_LATER)
        # 3. recovery down on the DI recovery edge (if present)
        if basis.disability_recovery_annual is not None:
            over["disability_recovery_annual"] = _scaled(
                basis.disability_recovery_annual, _SII_DISABILITY_RECOVERY)
        if over:
            basis = replace(basis, **over)
        return mp, basis

    return Stress(name="disability", apply=apply)


SII = RegimeSpec(
    name="Solvency II",
    sub_risks=(
        SubRisk("mortality", (scale_mortality(1.15),), "single"),      # +15%
        SubRisk("longevity", (scale_longevity(0.80),), "single"),      # -20%
        SubRisk("disability", (_sii_disability_shock(),), "single"),   # Art 139:
                                                                       #   inception +35%/+25%,
                                                                       #   recovery -20%
        SubRisk("expense", (scale_expense(1.10, 0.01),), "single"),    # +10%, inflation +1pp
        SubRisk("revision", (scale_annuity(1.03),), "single"),         # annuity benefits +3%
        SubRisk("lapse", (scale_lapse(1.50), scale_lapse(0.50),        # option-exercise +/-50%
                          mass_lapse(0.40)), "worst_of"),              # mass lapse 40%
        SubRisk("catastrophe", (catastrophe_mortality(0.0015),),       # Art 143: +0.15pp
                "single"),                                             #   mortality, next 12 months
    ),
    correlation=_SII_CORRELATION,
    interest_curves=(
        shock_curve(_per_year_rel(_SII_RATE_UP), up=True, floor_pp=0.01,
                    name="interest up"),
        shock_curve(_per_year_rel(_SII_RATE_DOWN), up=False, zero_floor=True,
                    name="interest down"),
    ),
    risk_margin_method="cost_of_capital",
    risk_margin_coc_rate=0.06,    # RM = CoC 6% x sum SCR(t)/(1+r)^(t+1) (Art 37, 39)
)


# ---------------------------------------------------------------------------
# K-ICS calibration (K-ICS handbook, primary source). Catastrophe is excluded
# from v1 -- under K-ICS it is a factor charge on sum insured, not a Delta-BEL
# shock, so it sits outside the shock-and-re-measure engine. Sub-risk order is
# locked to the correlation axes (the 5x5 sub-matrix of the life sub-risk
# correlation table for the sub-risks present here: mortality / longevity /
# morbidity / lapse / expense).
# ---------------------------------------------------------------------------

_KICS_CORRELATION = np.array([
    #  mortality  longevity  morbidity  lapse   expense
    [   1.00,     -0.25,      0.25,     0.00,    0.25],   # mortality
    [  -0.25,      1.00,      0.00,     0.25,    0.25],   # longevity
    [   0.25,      0.00,      1.00,     0.00,    0.50],   # morbidity (disability/illness)
    [   0.00,      0.25,      0.00,     1.00,    0.50],   # lapse
    [   0.25,      0.25,      0.50,     0.50,    1.00],   # expense
])

KICS = RegimeSpec(
    name="K-ICS",
    sub_risks=(
        SubRisk("mortality", (scale_mortality(1.125),), "single"),     # mortality +12.5%
        SubRisk("longevity", (scale_longevity(0.825),), "single"),     # mortality -17.5%
        SubRisk("morbidity", (scale_coverages({                        # disability/illness:
            CalculationMethod.DIAGNOSIS: 1.13,                         #   fixed-benefit +13%
            CalculationMethod.MORBIDITY: 1.10,                         #   indemnity    +10%
        }),), "single"),
        SubRisk("lapse", (scale_lapse(1.35), scale_lapse(0.65),        # option-exercise +/-35%
                          mass_lapse(0.30)), "worst_of"),              # mass lapse 30%
        SubRisk("expense", (scale_expense(1.10, 0.01),), "single"),    # expense +10%, inflation +1pp
    ),
    correlation=_KICS_CORRELATION,
    interest_curves=None,    # K-ICS interest shock is AFDNS-model-derived (not a
                             # static table) -- supply the official curve scenarios
                             # via the caller; not baked in.
    risk_margin_method="percentile",
    risk_margin_factor=0.40,  # risk margin = insurance-risk amount x 0.40 (= /Z99.5% x Z85%)
    # table 6 rows vs (mortality, longevity, morbidity, lapse, expense):
    catastrophe_correlation=(0.25, 0.0, 0.25, 0.25, 0.25),
    property_correlation=(0.0, 0.0, 0.0, 0.0, 0.5),   # long-term property/other
)

# Table 6 cross-correlation between the two extra (non-shock-vector) sub-risks.
_EXTRA_CROSS = {("property", "catastrophe"): 0.25}


__all__ = [
    "Stress", "SubRisk", "RegimeSpec", "SCRResult",
    "scale_mortality", "scale_longevity", "scale_lapse", "mass_lapse",
    "scale_state_rate", "catastrophe_mortality",
    "scale_coverages", "scale_coverages_first_year", "scale_coverage_codes",
    "scale_annuity", "scale_expense",
    "dynamic_lapse_multiplier", "interest_with_dynamic_lapse",
    "shock_curve", "shock_spread", "KICSInterest",
    "aggregate",
    "kics_catastrophe", "ratio",
    "SII", "KICS",
]
