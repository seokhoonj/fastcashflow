"""Solvency namespace -- ``fcf.solvency.*``.

The full required-capital / solvency-capital surface: the regime-agnostic
SCR engine (stresses, regimes SII/K-ICS, ``required_capital``) and the
balance-sheet assembly (asset/liability SCR modules, ``assess``,
dynamic and stochastic solvency). A thin facade re-exporting the two private
implementation modules ``_engine`` (SCR engine) and ``_assessment``
(assembly); both surfaces are disjoint, so the union is unambiguous. The
VFA-specific bodies live in ``_vfa`` and are re-exposed through ``fcf.vfa``.
"""
from fastcashflow.solvency._engine import *      # noqa: F401,F403 (engine)
from fastcashflow.solvency._assessment import *  # noqa: F401,F403 (assembly)
from fastcashflow.solvency import _engine, _assessment

__all__ = list(_engine.__all__) + list(_assessment.__all__)
