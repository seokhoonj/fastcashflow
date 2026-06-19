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
from fastcashflow.engine import measure
from fastcashflow.model_points import ModelPoints
from fastcashflow.numerics import _cost_of_capital_ra

Transform = Callable[[ModelPoints, Basis], "tuple[ModelPoints, Basis]"]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Stress:
    """One prescribed shock -- a labelled ``(model_points, basis)`` transform.

    ``apply`` returns a stressed ``(model_points, basis)`` pair; re-measuring it
    and differencing against the base gives the shock's loss in net asset value.
    """

    name: str
    apply: Transform


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
    """

    name: str
    sub_risks: tuple[SubRisk, ...]
    correlation: FloatArray
    interest_curves: tuple[Stress, ...] | None = None
    risk_margin_method: str = "percentile"
    risk_margin_factor: float = 0.0
    risk_margin_coc_rate: float = 0.06

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


def mass_lapse(fraction: float) -> Stress:
    """An instantaneous mass lapse -- ``fraction`` of the in-force surrenders at
    once, modelled by haircutting ``ModelPoints.count`` to ``(1 - fraction)``.

    v1 simplification: the surrender value paid to the lapsing policies at the
    shock date is NOT captured by a count haircut (the engine has no one-time
    surrender primitive), so the mass-lapse capital can be understated for
    contracts with a material surrender value. Documented; refine when a
    one-time-surrender mechanism exists."""
    def apply(mp: ModelPoints, basis: Basis):
        n = mp.n_mp
        count = np.ones(n) if mp.count is None else np.asarray(mp.count, float)
        return replace(mp, count=count * (1.0 - fraction)), basis
    return Stress(name=f"mass lapse {fraction:g}", apply=apply)


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


# ---------------------------------------------------------------------------
# Aggregation + the public entry point
# ---------------------------------------------------------------------------

def aggregate(capital: dict[str, float], spec: RegimeSpec) -> float:
    """Correlation-aggregate the sub-risk capitals: ``sqrt(c^T R c)`` with the
    capital vector ordered by ``spec.sub_risks`` and the regime matrix ``R``."""
    c = np.array([capital[sr.name] for sr in spec.sub_risks], dtype=np.float64)
    R = np.asarray(spec.correlation, dtype=np.float64)
    return float(np.sqrt(c @ R @ c))


def required_capital(
    model_points: ModelPoints, basis: Basis, *, regime: RegimeSpec,
) -> SCRResult:
    """Required capital (SCR) for a portfolio under ``regime``.

    Re-measures the liability under each sub-risk's stress, takes
    ``max(Delta BEL, 0)`` as the sub-risk capital, correlation-aggregates the
    insurance module, adds the worst-of interest stress, and computes the regime
    risk margin. v1 is liability-side: the total is ``insurance_scr +
    interest_capital`` (no inter-module diversification). Pass ``SCRResult`` on to
    :func:`fastcashflow.embedded_value` via its ``required_capital`` argument.
    """
    m_base = measure(model_points, basis, full=False)
    base_bel = float(m_base.bel.sum())

    def delta(stress: Stress) -> float:
        mp2, basis2 = stress.apply(model_points, basis)
        return float(measure(mp2, basis2, full=False).bel.sum()) - base_bel

    capital: dict[str, float] = {}
    for sr in regime.sub_risks:
        capital[sr.name] = max(0.0, max(delta(v) for v in sr.variants))
    insurance_scr = aggregate(capital, regime)

    interest_capital = 0.0
    if regime.interest_curves is not None:
        interest_capital = max(0.0, max(delta(s) for s in regime.interest_curves))

    total_scr = insurance_scr + interest_capital

    if regime.risk_margin_method == "percentile":
        risk_margin = insurance_scr * regime.risk_margin_factor
        scr_path = None
    elif regime.risk_margin_method == "cost_of_capital":
        # The capital run-off is proxied by the confidence-level RA trajectory
        # (the engine's own non-financial risk-capital path -- non-negative and
        # declining over the run-off). v1 approximation: SCR(t) = total_scr scaled
        # to that shape, not a full SCR re-projection at each future month.
        m_full = measure(model_points, basis, full=True)
        driver = m_full.ra_path.sum(axis=0)
        d0 = float(driver[0])
        if d0 <= 0.0:
            driver = np.abs(m_full.bel_path.sum(axis=0))
            d0 = float(driver[0]) if driver[0] != 0.0 else 1.0
        scr_path = total_scr * driver / d0
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


__all__ = [
    "Stress", "SubRisk", "RegimeSpec", "SCRResult",
    "scale_mortality", "scale_longevity", "scale_lapse", "mass_lapse",
    "scale_coverages", "scale_expense", "shock_curve",
    "aggregate", "required_capital",
]
