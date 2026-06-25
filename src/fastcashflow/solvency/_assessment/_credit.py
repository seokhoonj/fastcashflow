"""Solvency assessment -- asset-side credit-risk SCR.

A factor charge on credit exposures (bonds), read off a rating x effective-
maturity grid (or the Solvency II spread stress). A leaf sub-risk module; it
also owns the rating mapper (:func:`_rating_row`) that :mod:`._market`
concentration risk reuses.
"""
from __future__ import annotations

import math

from fastcashflow.assets import (
    Bond, bond_value, bond_duration, effective_maturity, AssetPortfolio)


# ---------------------------------------------------------------------------
# Credit risk -- an asset-side factor charge on credit exposures (bonds).
# K-ICS (chapter 5): the credit risk factor = default + downgrade charge, read
# off a (rating x effective-maturity) grid that differs by exposure class
# (public / corporate / securitisation -- handbook tables 29 / 30 / 31). The
# factor is a percent of market value (it already embeds the spread shock), so
# the charge is factor x market value -- no re-measure. Effective maturity is the
# cash-flow-weighted average maturity (:func:`fastcashflow.assets.effective_maturity`).
# External (S&P) ratings map to the K-ICS grades AAA/AA -> 1-2, A -> 3, BBB -> 4,
# BB -> 5, B -> 6, CCC and below -> 7, D -> default. Solvency II uses the Art-176
# spread stress (piecewise-linear in modified duration by credit quality step).
# ---------------------------------------------------------------------------

_CREDIT_FACTORS = {            # K-ICS handbook tables 29 / 30 / 31, in PERCENT;
    "public": {                # rows: K-ICS grade, columns: maturity bucket 0-1 .. 14+
        "1-2": (0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.3),
        "3": (0.4, 1, 1.3, 1.5, 1.8, 2, 2.2, 2.4, 2.5, 2.7, 2.8, 2.9, 3, 3, 3.1),
        "4": (1, 2.2, 2.6, 3, 3.3, 3.6, 3.9, 4.1, 4.2, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9),
        "5": (2.5, 5.1, 6, 6.6, 7, 7.3, 7.5, 7.6, 7.6, 7.7, 7.8, 7.8, 7.9, 7.9, 7.9),
        "6": (6.3, 10.8, 11.8, 12.3, 12.5, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7),
        "7": (22, 24.7, 25.2, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3),
        "unrated": (2.5, 5.1, 6, 6.6, 7, 7.3, 7.5, 7.6, 7.6, 7.7, 7.8, 7.8, 7.9, 7.9, 7.9),
        "default": (35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35),
    },
    "corporate": {             # the non-covered-bond ("other") rows for grades 1-2 / 3
        "1-2": (0.2, 0.7, 0.9, 1.2, 1.4, 1.6, 1.7, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.4, 2.5),
        "3": (0.6, 1.3, 1.6, 1.8, 2.1, 2.3, 2.6, 2.8, 3, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7),
        "4": (1.4, 3, 3.6, 4.1, 4.5, 4.9, 5.1, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9, 6, 6),
        "5": (3.6, 7.1, 8.3, 9, 9.4, 9.7, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8),
        "6": (8.9, 14.4, 15.3, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6, 15.6),
        "7": (35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35),
        "unrated": (6.3, 10.7, 11.8, 12.3, 12.5, 12.6, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7),
        "default": (35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35),
    },
    "securitisation": {
        "1-2": (0.2, 0.7, 0.9, 1.2, 1.4, 1.6, 1.7, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.4, 2.5),
        "3": (0.6, 1.3, 1.6, 1.8, 2.1, 2.3, 2.6, 2.8, 3, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7),
        "4": (1.4, 3, 3.6, 4.1, 4.5, 4.9, 5.1, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9, 6, 6),
        "5": (10.8, 21.3, 24.9, 27, 28.2, 29.1, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4),
        "6": (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
        "7": (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
        "unrated": (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
        "default": (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
    },
}

# Solvency II spread risk on bonds (Art 176): the stress factor is piecewise-linear
# in modified duration, a + b x (dur - lower), with (a, b) per credit quality step
# (CQS 0-6) and duration bucket [0-5, 5-10, 10-15, 15-20, 20+].
_SII_SPREAD = {                # CQS -> [(lower, a, b), ...]
    0: [(0, 0.000, 0.009), (5, 0.045, 0.005), (10, 0.070, 0.005), (15, 0.095, 0.005), (20, 0.120, 0.005)],
    1: [(0, 0.000, 0.011), (5, 0.055, 0.006), (10, 0.085, 0.005), (15, 0.110, 0.005), (20, 0.135, 0.005)],
    2: [(0, 0.000, 0.014), (5, 0.070, 0.007), (10, 0.105, 0.005), (15, 0.130, 0.005), (20, 0.155, 0.005)],
    3: [(0, 0.000, 0.025), (5, 0.125, 0.015), (10, 0.200, 0.010), (15, 0.250, 0.010), (20, 0.300, 0.005)],
    4: [(0, 0.000, 0.045), (5, 0.225, 0.025), (10, 0.350, 0.018), (15, 0.440, 0.005), (20, 0.466, 0.005)],
    5: [(0, 0.000, 0.075), (5, 0.375, 0.042), (10, 0.585, 0.005), (15, 0.610, 0.005), (20, 0.635, 0.005)],
}
_SII_CQS = {                   # S&P base letter -> credit quality step (5 = CQS 5 and 6)
    "AAA": 0, "AA": 1, "A": 2, "BBB": 3, "BB": 4, "B": 5, "CCC": 5, "CC": 5, "C": 5,
}


def _sii_spread_stress(rating: str, modified_duration: float) -> float:
    """The Solvency II Art-176 spread stress factor for a bond's rating and modified
    duration; unrated maps to CQS 3 (BBB-equivalent) as a v1 simplification."""
    r = (rating or "").strip().upper()
    base = r.rstrip("+-0123456789")
    cqs = _SII_CQS.get(base, 3)
    d = max(0.0, modified_duration)
    buckets = _SII_SPREAD[cqs]
    idx = min(4, int(d // 5)) if d > 0 else 0
    if d > 20.0:
        idx = 4
    lower, a, b = buckets[idx]
    return min(1.0, a + b * (d - lower))


# Per regime: the K-ICS grid above; Solvency II uses the Art-176 spread stress.
_CREDIT_CALIBRATION = {"K-ICS": _CREDIT_FACTORS, "Solvency II": "sii_spread"}

_RATING_TO_ROW = {             # external (S&P) base letter -> K-ICS factor-table row
    "AAA": "1-2", "AA": "1-2", "A": "3", "BBB": "4", "BB": "5", "B": "6",
    "CCC": "7", "CC": "7", "C": "7", "D": "default",
}


def _rating_row(rating: str) -> str:
    """Map an external rating (e.g. ``"AA+"``, ``"BBB-"``, ``"unrated"``) to its
    K-ICS factor-table row, stripping the +/- and any numeric modifier."""
    r = (rating or "").strip().upper()
    if r in ("UNRATED", "NR", ""):
        return "unrated"
    base = r.rstrip("+-0123456789")
    return _RATING_TO_ROW.get(base, "unrated")


def _credit_bucket(maturity: float) -> int:
    """The maturity-bucket index for the factor grid: bucket k is ``k < m <= k+1``
    (so ``0-1`` is index 0), capped at index 14 (the ``14+`` bucket)."""
    return min(14, max(0, math.ceil(maturity) - 1))


def credit_scr(portfolio: AssetPortfolio, regime, discount_annual) -> float:
    """The credit-risk SCR -- each bond's market value times its credit factor.

    The factor is read off the K-ICS (rating x effective-maturity) grid for the
    bond's ``exposure_class`` (handbook tables 29 / 30 / 31); it is a percent of
    market value and already embeds the default + downgrade charge, so the SCR is
    ``sum(market_value x factor)`` -- no re-measure. ``Cash`` is risk-free and
    equity / property carry market (not credit) risk, so only bonds contribute.
    Solvency II uses the Art-176 spread stress (piecewise-linear in modified
    duration by credit quality step)."""
    factors = _CREDIT_CALIBRATION.get(regime.name)
    if factors is None:
        return 0.0
    total = 0.0
    for h in portfolio.holdings:
        if not isinstance(h, Bond):
            continue
        if factors == "sii_spread":             # Solvency II Art 176
            mod = bond_duration(h, discount_annual).modified
            factor = _sii_spread_stress(h.credit_rating, mod)
        else:                                   # K-ICS rating x maturity grid
            table = factors.get(h.exposure_class)
            if table is None:
                raise ValueError(
                    f"unknown bond exposure_class {h.exposure_class!r}; "
                    f"known: {sorted(factors)}")
            row = table[_rating_row(h.credit_rating)]
            factor = row[_credit_bucket(effective_maturity(h))] / 100.0
        total += bond_value(h, discount_annual) * factor
    return max(0.0, total)
