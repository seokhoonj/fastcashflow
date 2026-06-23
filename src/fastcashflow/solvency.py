"""Solvency namespace -- ``fcf.solvency.*``.

The full required-capital / solvency-capital surface: the regime-agnostic
SCR engine (stresses, regimes SII/K-ICS, ``required_capital``) and the
balance-sheet assembly (asset/liability SCR modules, ``assess_solvency``,
dynamic and stochastic solvency). A thin facade re-exporting the two private
implementation modules ``_solvency`` (engine) and ``_solvency_assessment``
(assembly); both surfaces are disjoint, so the union is unambiguous.
"""
from fastcashflow._solvency import *            # noqa: F401,F403 (engine)
from fastcashflow._solvency_assessment import *  # noqa: F401,F403 (assembly)
from fastcashflow import _solvency as _engine
from fastcashflow import _solvency_assessment as _assembly

__all__ = list(_engine.__all__) + list(_assembly.__all__)
