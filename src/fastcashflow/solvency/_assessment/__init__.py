"""Solvency assessment implementation package.

Private -- the surface is re-exported through :mod:`fastcashflow.solvency`
(``solvency/__init__`` does ``from ...solvency._assessment import *``, so this
package's ``__all__`` IS part of ``fcf.solvency.*``). Layered sub-risks
(_interaction / _market / _credit / _operational / _toplevel) feed the
assembly (_assemble); a one-directional DAG, no cycle.

``__all__`` is unchanged from the flat module -- the nouns fcf.solvency
exposes. The verbs ``assess`` / ``assess_dynamic`` / ``assess_stochastic`` and
the private ``_market_cal`` are imported into the namespace (so the direct
imports in fcf.gmm and solvency/_vfa resolve) but kept OUT of ``__all__``, so
they do not leak into ``fcf.solvency.*`` through the star import.
"""
from fastcashflow.solvency._assessment._interaction import (
    InteractionResult, interaction_loss, net_interest_scr,
    net_interest_kics_scr)
from fastcashflow.solvency._assessment._market import (
    equity_scr, property_scr, fx_scr, concentration_scr, market_scr,
    _market_cal, _MARKET_CORRELATION)
from fastcashflow.solvency._assessment._credit import (
    credit_scr, _credit_bucket, _sii_spread_stress)
from fastcashflow.solvency._assessment._operational import operational_scr
from fastcashflow.solvency._assessment._toplevel import basic_scr
from fastcashflow.solvency._assessment._assemble import (
    Assessment, DynamicAssessment, StochasticAssessment, available_capital,
    assess, assess_dynamic, assess_stochastic)

__all__ = [
    "InteractionResult", "interaction_loss",
    "net_interest_scr", "net_interest_kics_scr",
    "equity_scr", "property_scr", "fx_scr", "concentration_scr",
    "market_scr", "credit_scr", "operational_scr",
    "basic_scr",
    "Assessment", "DynamicAssessment", "StochasticAssessment",
    "available_capital",
]
