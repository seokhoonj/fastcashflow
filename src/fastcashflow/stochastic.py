"""Stochastic valuation -- the liability distribution over economic scenarios.

A deterministic run gives one liability from one assumption set. A stochastic
valuation runs the projection under many economic scenarios and reports the
*distribution* of the liability -- which feeds the percentile-based risk and
capital measures a single deterministic run cannot give.

``value_stochastic`` takes the scenarios as input -- fastcashflow is the
engine, not an economic scenario generator -- and values each one with the
fused ``value`` kernel. Running N scenarios over millions of seriatim
policies is precisely what the engine's speed exists for: a slow engine
cannot do seriatim stochastic at scale at all.

v1 scope: each scenario is a flat annual discount rate. Scenario paths (a
rate that varies over the projection) and investment-return scenarios for
participating business are left for later.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.engine import value
from fastcashflow.modelpoint import ModelPointSet


@dataclass(frozen=True, slots=True)
class StochasticResult:
    """Per-scenario portfolio totals from a stochastic valuation.

    Each array is ``(n_scenarios,)`` -- the portfolio total of that figure
    under each scenario. Read the distribution off with :meth:`mean` and
    :meth:`percentile`, or from the arrays directly.
    """

    bel: FloatArray
    ra: FloatArray
    csm: FloatArray
    loss_component: FloatArray

    def mean(self) -> dict[str, float]:
        """The mean of each line across the scenarios."""
        return {name: float(getattr(self, name).mean())
                for name in ("bel", "ra", "csm", "loss_component")}

    def percentile(self, q: float) -> dict[str, float]:
        """The ``q``-th percentile of each line across the scenarios."""
        return {name: float(np.percentile(getattr(self, name), q))
                for name in ("bel", "ra", "csm", "loss_component")}


def value_stochastic(
    mps: ModelPointSet, asmp: Assumptions, scenarios: FloatArray
) -> StochasticResult:
    """Value a portfolio under each economic scenario -- the liability distribution.

    ``scenarios`` is a 1-D array of annual discount rates, one per scenario.
    Each scenario is valued with the fused :func:`value` kernel and the
    portfolio total of every figure is recorded, so the distribution -- mean,
    percentiles -- can be read from the result.
    """
    scenarios = np.asarray(scenarios, dtype=np.float64)
    n = int(scenarios.shape[0])
    bel = np.empty(n)
    ra = np.empty(n)
    csm = np.empty(n)
    loss_component = np.empty(n)
    for s in range(n):
        v = value(mps, replace(asmp, discount_annual=float(scenarios[s])))
        bel[s] = v.bel.sum()
        ra[s] = v.ra.sum()
        csm[s] = v.csm.sum()
        loss_component[s] = v.loss_component.sum()
    return StochasticResult(bel=bel, ra=ra, csm=csm, loss_component=loss_component)
