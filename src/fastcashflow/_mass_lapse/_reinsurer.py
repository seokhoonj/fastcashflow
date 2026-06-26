"""Mass-lapse reinsurance -- reinsurer side (tail distribution, pricing, IFRS 17).

The cedant side is deterministic (the standard formula fixes the lapse at 40%);
the reinsurer prices over the whole tail of the cumulative excess lapse, so it
needs a distribution F(L) (:class:`LapseDistribution` / :class:`LapseTailDistribution`),
a layer price (:func:`price_treaty`) and the reinsurer's own IFRS 17 measurement
of the assumed treaty (:func:`measure_assumed_treaty`). Builds on the cedant
:class:`LapseXL` treaty type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from fastcashflow.numerics import _norm_ppf
from fastcashflow._mass_lapse._cedant import LapseXL


_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the complementary error function."""
    return 0.5 * math.erfc(-x / _SQRT2)


# ---------------------------------------------------------------------------
# Reinsurer side: the lapse tail distribution and pricing (Phase D).
#
# The cedant side is deterministic because the standard formula fixes the lapse
# at a single point (40%). The reinsurer prices over the WHOLE tail of the
# cumulative excess-over-best-estimate lapse L, so it needs a distribution F(L).
# Pricing depends on F only through ``survival(x) = P(L > x)``: the expected
# layer is the integral of the survival over the layer (a standard identity),
#
#     E[clip(L - a, 0, b - a)] = integral_a^b P(L > x) dx,
#
# so any object exposing ``survival`` (and the ``expected_layer`` it implies) is
# a drop-in F(L). :class:`LapseTailDistribution` is the public baseline,
# calibrated to public Solvency II anchors; a reinsurer's proprietary F(L)
# (dynamic lapse, dependence structure, cross-portfolio tail) replaces it
# without touching the pricing.
# ---------------------------------------------------------------------------

# Public tail anchors for the excess-over-best-estimate lapse L (exceedance
# probabilities): the 40% standard-formula stress is the 1-in-200 (99.5%) point
# (Article 142(6)(b)); EIOPA notes attachment points are typically set around a
# 1-in-30 event (e.g. 15%). The second anchor is a calibration choice, not a
# regulatory law -- override it with the book's own lapse volatility.
SF_LAPSE_TAIL_ANCHOR = (0.40, 1.0 / 200.0)
ATTACHMENT_TAIL_ANCHOR = (0.15, 1.0 / 30.0)


class LapseDistribution:
    """The Engine/Model seam: the distribution F(L) of the cumulative excess-
    over-best-estimate lapse fraction ``L`` that reinsurer pricing integrates
    against.

    This is fastcashflow's plug-in point. The valuable, proprietary part of
    mass-lapse reinsurance is the MODEL that produces F(L) -- calibrated to a
    reinsurer's cross-portfolio lapse experience (its deepest moat), the
    economic-to-lapse link, and channel-level clustering. fastcashflow is the
    ENGINE: it takes ANY F(L) and returns the capital relief, pricing, capital,
    risk adjustment and CSM. To plug in a model, subclass this and implement
    ``survival``; ``expected_layer`` and ``value_at_risk`` then work for free
    (pricing needs nothing else). A subclass MAY override them with closed forms
    (as :class:`LapseTailDistribution` does) for speed and precision."""

    __slots__ = ()

    def survival(self, x: float) -> float:
        """``P(L > x)`` -- the one method a plug-in F(L) must provide."""
        raise NotImplementedError

    def expected_layer(self, attachment: float, detachment: float) -> float:
        """``E[clip(L - attachment, 0, detachment - attachment)]`` -- the expected
        covered fraction, equal to the survival integral over the layer (the
        identity any survival-only F(L) prices through). Numerical default;
        override for a closed form."""
        if not (0.0 <= attachment < detachment):
            raise ValueError("require 0 <= attachment < detachment")
        grid = np.linspace(attachment, detachment, 4001)
        surv = np.array([self.survival(float(x)) for x in grid])
        return float(np.trapezoid(surv, grid))

    def value_at_risk(self, q: float) -> float:
        """The ``q``-quantile of ``L`` -- the lapse level ``x`` with
        ``survival(x) = 1 - q`` -- by bisection on the (decreasing) survival.
        Numerical default; override for a closed form."""
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q}")
        target = 1.0 - q
        lo, hi = 0.0, 1.0
        while self.survival(hi) > target and hi < 1e6:    # expand to bracket
            hi *= 2.0
        for _ in range(100):                              # bisection
            mid = 0.5 * (lo + hi)
            if self.survival(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


@runtime_checkable
class LapseModel(Protocol):
    """A lapse MODEL: produces a :class:`LapseDistribution` for a given context
    (e.g. a book's channel mix or an economic scenario).

    This is the proprietary layer fastcashflow deliberately leaves to the user.
    A reinsurer's model -- calibrated to cross-portfolio tail data and the
    channel-level clustering that actually drives mass lapse -- is its real IP,
    and is never open-sourced. The baseline
    :meth:`LapseTailDistribution.from_anchors` is a trivial context-free model;
    the engine consumes only the returned distribution, so any model satisfying
    this protocol drops in."""

    def distribution(self, context=None) -> LapseDistribution:
        ...


@dataclass(frozen=True, slots=True)
class LapseTailDistribution(LapseDistribution):
    """Lognormal distribution of the cumulative excess-over-best-estimate lapse
    fraction ``L``, the public baseline F(L) for reinsurer pricing.

    Calibrated by :meth:`from_anchors` to two tail exceedance probabilities
    (default: 15% at 1-in-30, 40% at 1-in-200). Pricing depends on a
    distribution ONLY through :meth:`survival`; a reinsurer's proprietary F(L)
    is a drop-in replacement that provides the same method. The closed-form
    :meth:`expected_layer` equals ``integral_attach^detach survival(x) dx`` -- the
    identity any survival-only F(L) can use numerically."""

    mu: float
    sigma: float

    @classmethod
    def from_anchors(cls, lower=ATTACHMENT_TAIL_ANCHOR,
                     upper=SF_LAPSE_TAIL_ANCHOR) -> "LapseTailDistribution":
        """Calibrate the lognormal to two ``(lapse_level, exceedance_prob)``
        anchors. ``P(L > a) = p`` gives ``(ln a - mu)/sigma = z`` with
        ``z = norm_ppf(1 - p)``; two anchors solve ``mu`` and ``sigma``."""
        (a1, p1), (a2, p2) = lower, upper
        if not (0.0 < a1 < a2 and 0.0 < p2 < p1 < 1.0):
            raise ValueError(
                "anchors must satisfy 0 < a1 < a2 and 0 < p2 < p1 < 1 "
                f"(a higher lapse is rarer), got {lower}, {upper}")
        z1, z2 = _norm_ppf(1.0 - p1), _norm_ppf(1.0 - p2)
        sigma = (math.log(a2) - math.log(a1)) / (z2 - z1)
        mu = math.log(a1) - sigma * z1
        return cls(mu=mu, sigma=sigma)

    def survival(self, x: float) -> float:
        """``P(L > x)`` -- the only method pricing requires of an F(L)."""
        if x <= 0.0:
            return 1.0
        return 1.0 - _norm_cdf((math.log(x) - self.mu) / self.sigma)

    @property
    def mean(self) -> float:
        """``E[L]`` -- the expected excess lapse."""
        return math.exp(self.mu + 0.5 * self.sigma * self.sigma)

    def expected_excess(self, k: float) -> float:
        """``E[(L - k)+]`` -- the lognormal stop-loss above ``k``."""
        if k <= 0.0:
            return self.mean
        d1 = (self.mu + self.sigma * self.sigma - math.log(k)) / self.sigma
        d2 = d1 - self.sigma
        return self.mean * _norm_cdf(d1) - k * _norm_cdf(d2)

    def expected_layer(self, attachment: float, detachment: float) -> float:
        """``E[clip(L - attachment, 0, detachment - attachment)]`` -- the expected
        covered excess-lapse fraction (equals the survival integral over the
        layer)."""
        return self.expected_excess(attachment) - self.expected_excess(detachment)

    def value_at_risk(self, q: float) -> float:
        """The ``q``-quantile of ``L`` -- ``exp(mu + sigma x norm_ppf(q))``. At
        ``q = 0.995`` this is the 1-in-200 lapse (the standard-formula stress; by
        calibration it returns the upper anchor)."""
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q}")
        return math.exp(self.mu + self.sigma * _norm_ppf(q))


@dataclass(frozen=True, slots=True)
class ReinsurancePricing:
    """The reinsurer's price and assumed capital for a :class:`LapseXL` treaty.

    ``expected_recovery`` is the pure premium (expected loss) ``S x E[layer]``;
    ``capital`` is the assumed risk capital -- the unexpected loss
    ``VaR(recovery) - expected_recovery`` at ``var_level`` -- after the
    reinsurer's diversification factor; ``premium`` loads the cost of capital on
    top of the pure premium. ``expected_profit`` is the load (premium less
    expected loss)."""

    loss_density: float
    capacity_at_risk: float          # S x capacity -- the most the layer can pay
    expected_recovery: float
    capital: float
    premium: float

    @property
    def expected_profit(self) -> float:
        """Premium less expected loss -- the cost-of-capital load."""
        return self.premium - self.expected_recovery

    @property
    def rate_on_line(self) -> float:
        """Premium as a fraction of the capacity at risk (the market quote
        convention -- typically a low single-digit percent)."""
        return self.premium / self.capacity_at_risk if self.capacity_at_risk else 0.0

    @property
    def loss_on_line(self) -> float:
        """Expected loss as a fraction of the capacity at risk."""
        return (self.expected_recovery / self.capacity_at_risk
                if self.capacity_at_risk else 0.0)


def price_treaty(
    loss_density: float, treaty: LapseXL, distribution: LapseDistribution, *,
    cost_of_capital: float = 0.06, var_level: float = 0.995,
    diversification_factor: float = 1.0,
) -> ReinsurancePricing:
    """Price the treaty from the reinsurer's side over the lapse tail
    ``distribution`` (any object exposing ``expected_layer`` and
    ``value_at_risk``; :class:`LapseTailDistribution` is the public baseline).

    ``expected_recovery = S x E[layer]`` is the pure premium. The assumed capital
    is the unexpected loss ``S x covered_fraction(VaR_level(L)) -
    expected_recovery`` scaled by ``diversification_factor`` (1.0 = standalone;
    a reinsurer diversifies the assumed lapse risk against its own book, so its
    marginal capital is a fraction of standalone -- pass e.g. 0.25). The premium
    loads ``cost_of_capital`` on that capital:

        premium = expected_recovery + cost_of_capital x capital.

    Everything tail-dependent flows through ``distribution`` -- the proprietary
    F(L) the reinsurer plugs in sets the price."""
    S = loss_density
    capacity_at_risk = S * treaty.capacity
    expected_recovery = S * distribution.expected_layer(
        treaty.attachment, treaty.detachment)
    var_lapse = distribution.value_at_risk(var_level)
    unexpected = S * treaty.covered_fraction(var_lapse) - expected_recovery
    capital = max(0.0, unexpected) * diversification_factor
    premium = expected_recovery + cost_of_capital * capital
    return ReinsurancePricing(
        loss_density=S, capacity_at_risk=capacity_at_risk,
        expected_recovery=expected_recovery, capital=capital, premium=premium)


# ---------------------------------------------------------------------------
# Reinsurer-side IFRS 17 measurement of the assumed treaty (Phase D3).
# The treaty is a stream of premium inflows and contingent recovery outflows --
# a portfolio-level structure, not a per-policy projection, so it is measured by
# direct discounting. Sign convention (the engine's): outflow-positive, so the
# recovery the reinsurer pays is positive and the premium it receives is
# negative; BEL = PV(recovery) - PV(premium). For an out-of-the-money treaty the
# premium exceeds the expected recovery, so the BEL is negative (profitable) and
# the unearned profit sits in the CSM.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AssumedTreatyMeasurement:
    """IFRS 17 measurement of the treaty from the reinsurer's (assuming) side.

    ``bel = PV(expected recovery) - PV(premium)`` (outflow-positive);
    ``risk_adjustment`` is the cost-of-capital margin on the assumed capital over
    the treaty. ``fulfilment_cash_flows = bel + risk_adjustment``; the CSM is the
    unearned profit ``max(0, -fcf)`` and the loss component ``max(0, fcf)``
    (General Measurement Model -- IFRS 17 paragraph 38, 47)."""

    pv_premium: float
    pv_expected_recovery: float
    bel: float
    risk_adjustment: float

    @property
    def fulfilment_cash_flows(self) -> float:
        """``BEL + RA`` -- negative for a profitable assumed treaty."""
        return self.bel + self.risk_adjustment

    @property
    def csm(self) -> float:
        """Contractual service margin -- the unearned profit ``max(0, -FCF)``."""
        return max(0.0, -self.fulfilment_cash_flows)

    @property
    def loss_component(self) -> float:
        """Onerous loss component ``max(0, FCF)`` (zero unless the premium is
        below the risk-adjusted expected recovery)."""
        return max(0.0, self.fulfilment_cash_flows)


def measure_assumed_treaty(
    pricing: ReinsurancePricing, *, duration_years: int,
    discount_annual: float = 0.0, risk_adjustment_cost_of_capital: float = 0.06,
) -> AssumedTreatyMeasurement:
    """Measure the assumed treaty over ``duration_years`` annual periods.

    Premium is received in advance (start of each period); the expected recovery
    is paid in arrears (end of each period, when the measurement window closes).
    The risk adjustment is the cost of capital on the assumed capital held each
    period. Discounting is a flat ``discount_annual``.

    ``BEL = PV(recovery) - PV(premium)``, ``RA = ra_coc x capital x annuity``,
    and the CSM / loss component follow the standard sign convention. A treaty
    priced at exactly the cost of capital has BEL + RA ~ 0 (no unearned profit);
    a premium loaded above the cost of capital leaves a positive CSM."""
    if duration_years <= 0:
        raise ValueError(f"duration_years must be positive, got {duration_years}")
    v = 1.0 / (1.0 + discount_annual)
    annuity_advance = sum(v ** t for t in range(duration_years))        # t = 0..n-1
    annuity_arrear = sum(v ** t for t in range(1, duration_years + 1))  # t = 1..n
    pv_premium = pricing.premium * annuity_advance
    pv_recovery = pricing.expected_recovery * annuity_arrear
    bel = pv_recovery - pv_premium
    ra = risk_adjustment_cost_of_capital * pricing.capital * annuity_arrear
    return AssumedTreatyMeasurement(
        pv_premium=pv_premium, pv_expected_recovery=pv_recovery,
        bel=bel, risk_adjustment=ra)
