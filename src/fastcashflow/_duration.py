"""Interest-rate duration result type -- shared by liability and bond metrics.

A neutral leaf module so the asset side (a bond's duration) and the liability
side (a BEL's duration) can both return a :class:`DurationResult` without either
importing the other. It sits at the base of the import graph (it imports nothing
from the package), which keeps the asset / liability / matching layers acyclic.
"""
from __future__ import annotations

from dataclasses import dataclass

_BP = 1e-4    # one basis point


@dataclass(frozen=True, slots=True)
class DurationResult:
    """Interest-rate sensitivity of a present value.

    ``pv`` is the present value (the BEL for a liability, the market value for a
    bond). ``macaulay`` / ``modified`` are durations in years (``macaulay`` is
    ``nan`` where it is not well defined -- a mixed-sign liability stream).
    ``dv01`` is the decrease in ``pv`` for a +1bp parallel rise in the curve
    (positive for a normal positive-duration instrument). ``convexity`` is the
    second-order yield sensitivity ``(1/pv) d2pv/dy2`` in years^2 (the curvature
    that the linear duration misses for a large rate move:
    ``dpv/pv ~ -modified*dy + 0.5*convexity*dy^2``); ``nan`` where it is not well
    defined (a near-zero ``pv``)."""

    pv: float
    macaulay: float
    modified: float
    dv01: float
    convexity: float = float("nan")
