"""ESG shared core -- the scenario result and model-agnostic Monte-Carlo infra.

The pieces every economic-scenario model reuses, independent of the short-rate
model chosen: the :class:`EconomicScenarios` result, the correlated standard
normal draw (with antithetic variance reduction), and the monthly time step. A
new short-rate model (Vasicek, CIR, G2++, ...) lives in its own ``esg/_<model>.py``
that imports from here; this module imports no model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray

_DT = 1.0 / 12.0                    # one month in years
_SQRT_DT = _DT ** 0.5


@dataclass(frozen=True, slots=True, eq=False)
class EconomicScenarios:
    """Correlated risk-neutral scenarios from :func:`fastcashflow.esg.simulate`.

    ``rates`` is ``(n_scenarios, n_time)`` -- the short rate as an **annual**
    rate for each projection month, the ``scenarios`` input to ``gmm.stochastic``.
    ``returns`` is ``(n_scenarios, n_time)`` -- the lognormal fund **monthly simple
    return**, the ``return_scenarios`` input to ``vfa.measure`` / ``measure_tvog``.
    The fields are used directly; no further unit conversion is needed.
    """

    rates: FloatArray             # (n_scen, n_time) annual short rate per month
    returns: FloatArray           # (n_scen, n_time) monthly fund return
    short_rate: FloatArray        # (n_scen, n_time) continuous short rate r_t
    initial_prices: FloatArray    # (n_time+1,) P(0,t) the model is calibrated to

    def martingale_error(self):
        """The two no-arbitrage check errors, as relative deviations.

        Returns ``(bond_error, equity_error)``:

        * ``bond_error`` -- the worst relative deviation, over all maturities, of
          the Monte-Carlo zero-coupon price ``mean_s exp(-sum_t r/12)`` from the
          calibrated ``P(0, T)``. Zero under perfect calibration / infinite paths.
        * ``equity_error`` -- the relative deviation of the discounted terminal
          fund value ``mean_s [ prod_t (1 + return) * exp(-sum_t r/12) ]`` from 1
          (the fund is a tradeable, so its discounted value is a martingale).
        """
        disc = np.exp(-np.cumsum(self.short_rate, axis=1) * _DT)   # (n_scen, n_time)
        mc_price = disc.mean(axis=0)                               # (n_time,)
        p0 = self.initial_prices[1:]                              # P(0, 1..n_time)
        bond_error = float(np.max(np.abs(mc_price / p0 - 1.0)))
        fund = np.cumprod(1.0 + self.returns, axis=1)             # S_t / S_0
        equity_error = float(np.abs((fund * disc).mean(axis=0)[-1] - 1.0))
        return bond_error, equity_error


def _normals(n_scenarios, n_time, seed, antithetic):
    """Two independent standard-normal matrices ``(n_scenarios, n_time)``,
    deterministic in the seed. With ``antithetic`` the second half of each path
    set is the negation of the first (a variance-reduction pairing); an odd count
    drops the one spare mirror path. Two 2-D draws (not one 3-D array) keep the
    correlation step a plain elementwise combination."""
    rng = np.random.default_rng(seed)
    if antithetic:
        half = (n_scenarios + 1) // 2
        za = rng.standard_normal((half, n_time))
        zb = rng.standard_normal((half, n_time))
        return (np.concatenate([za, -za])[:n_scenarios],
                np.concatenate([zb, -zb])[:n_scenarios])
    return (rng.standard_normal((n_scenarios, n_time)),
            rng.standard_normal((n_scenarios, n_time)))
