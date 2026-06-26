"""Risk-neutral economic scenario generator (ESG) -- ``fcf.esg.*``.

A market-consistent generator that produces the scenarios the measurement
engine consumes -- it does not value anything itself. Two factors:

* a short rate, calibrated so the model reprices the initial risk-free curve (a
  :func:`~fastcashflow.curves.smith_wilson_prices` discount curve); ``.rates``
  feeds ``gmm.stochastic`` (annual rate per month).
* a correlated lognormal (geometric Brownian motion) fund return whose
  risk-neutral drift is the short rate; ``.returns`` feeds ``vfa.measure`` /
  ``measure_tvog`` (monthly simple return).

Correctness is by no-arbitrage, not hand calculation: under the risk-neutral
measure the Monte-Carlo average of the stochastic discount factor reprices the
zero-coupon bond, and the discounted fund value is a martingale --
:meth:`EconomicScenarios.martingale_error` reports both.

Structure: the model-agnostic result + Monte-Carlo infra are in
:mod:`~fastcashflow.esg._core`; each short-rate model is its own private module.
The v1 model is Hull-White one-factor (:mod:`~fastcashflow.esg._hull_white`);
``simulate`` / ``hull_white_rates`` are its entry points. A further model
(Vasicek, CIR, G2++, stochastic-vol, ...) drops in as a new ``esg/_<model>.py``
re-exported here.
"""
from fastcashflow.esg._core import EconomicScenarios
from fastcashflow.esg._hull_white import simulate, hull_white_rates

__all__ = ["EconomicScenarios", "simulate", "hull_white_rates"]
