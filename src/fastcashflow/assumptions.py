"""Actuarial assumption set for the Phase 0 deterministic projection."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastcashflow._typing import FloatArray


@dataclass(frozen=True, slots=True)
class Assumptions:
    """Deterministic assumption set -- Phase 0, no assumption changes over time.

    Parameters
    ----------
    mortality_monthly :
        Maps an array of attained ages (years) to an array of monthly
        mortality rates of the same shape.
    lapse_monthly :
        Flat monthly lapse rate.
    discount_annual :
        Flat annual discount rate. Locked in at initial recognition and used
        both for discounting cash flows and for CSM interest accretion.
    ra_rate :
        Phase 0 placeholder -- the Risk Adjustment is this fraction of the
        present value of claims. A proper RA methodology (confidence level /
        cost of capital) replaces this in Phase 1.
    """

    mortality_monthly: Callable[[FloatArray], FloatArray]
    lapse_monthly: float
    discount_annual: float
    ra_rate: float

    @property
    def discount_monthly(self) -> float:
        """Monthly discount rate equivalent to ``discount_annual``."""
        return (1.0 + self.discount_annual) ** (1.0 / 12.0) - 1.0
