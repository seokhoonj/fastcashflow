"""Solvency assessment -- top-level BSCR aggregation.

The basic required capital from the disclosed module amounts (insurance /
market / credit / operational), combined through the regime correlation matrix.
A leaf module: it takes the module SCRs as scalars, so it imports no sub-risk.
"""
from __future__ import annotations

import numpy as np


# K-ICS table 3: the top-level correlation among the insurance, market and credit
# risk modules -- all pairwise 0.25 (life-long-term / market / credit).
_TOPLEVEL_CORRELATION = np.array([
    [1.00, 0.25, 0.25],
    [0.25, 1.00, 0.25],
    [0.25, 0.25, 1.00],
])

# Table 3 with the general (P&C) insurance module added: order is
# (life-long-term, general insurance, market, credit). Life-vs-general is 0; all
# other pairs are 0.25. Used when a caller supplies a general-insurance SCR (for a
# life + P&C book); the life-only engine leaves it at zero.
_TOPLEVEL_CORRELATION_4 = np.array([
    [1.00, 0.00, 0.25, 0.25],
    [0.00, 1.00, 0.25, 0.25],
    [0.25, 0.25, 1.00, 0.25],
    [0.25, 0.25, 0.25, 1.00],
])

# Solvency II BSCR top-level correlation (Delegated Regulation (EU) 2015/35, Annex
# IV). Among (life, market, counterparty-default) every pair is 0.25 -- the same
# values as K-ICS table 3, so the 3-module matrix is shared. With the non-life
# (general insurance) module added the only differences from K-ICS are
# non-life <-> default = 0.5 and life <-> non-life = 0 (the rest 0.25); order is
# (life, general/non-life, market, default/credit).
_SII_TOPLEVEL_CORRELATION_4 = np.array([
    [1.00, 0.00, 0.25, 0.25],
    [0.00, 1.00, 0.25, 0.50],
    [0.25, 0.25, 1.00, 0.25],
    [0.25, 0.50, 0.25, 1.00],
])


def basic_scr(insurance: float, market: float, credit: float, *,
                               regime, operational: float = 0.0,
                               general_insurance: float = 0.0) -> float:
    """The basic required capital from disclosed module amounts -- the top-level
    aggregate of the (life) insurance, market and credit modules plus the
    operational charge (added OUTSIDE the aggregate).

    K-ICS uses the table-3 correlation; Solvency II uses the Annex IV BSCR matrix.
    For (life, market, credit) the two coincide (all pairs 0.25), so the 3-module
    aggregate is the same; with ``general_insurance`` (a fourth P&C module) they
    differ only in general-vs-credit (K-ICS 0.25, Solvency II 0.5) and share
    life-vs-general 0. The disclosed ``diversification effect`` is the simple module
    sum minus this aggregate. Use it to reproduce a disclosed basic required capital
    from the published module risk amounts, or for a what-if on the module mix
    without re-running a book."""
    if general_insurance > 0.0:
        c = np.array([insurance, general_insurance, market, credit], dtype=np.float64)
        R = (_SII_TOPLEVEL_CORRELATION_4 if regime.name == "Solvency II"
             else _TOPLEVEL_CORRELATION_4)
    else:
        c = np.array([insurance, market, credit], dtype=np.float64)
        R = _TOPLEVEL_CORRELATION       # 3-module values coincide for K-ICS and SII
    return float(np.sqrt(c @ R @ c)) + operational


def _pct(r: float) -> str:
    """Format a coverage ratio for display -- 'n/a' for a non-finite (risk-free) ratio."""
    return "n/a" if not np.isfinite(r) else f"{r:.1%}"
