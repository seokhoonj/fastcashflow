"""Actuarial assumption set for the deterministic projection."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastcashflow._typing import FloatArray


@dataclass(frozen=True, slots=True)
class Assumptions:
    """Deterministic assumption set -- no assumption changes over time.

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
    expense_acquisition :
        One-off acquisition expense per policy, incurred at t = 0.
    expense_maintenance_annual :
        Annual maintenance expense per in-force policy; one twelfth is
        charged each month.
    expense_inflation :
        Annual inflation rate applied to the maintenance expense.
    ra_confidence :
        Confidence level for the Risk Adjustment (e.g. 0.75). The RA lifts
        the liability from its best estimate to this percentile.
    claims_cv :
        Coefficient of variation of claims, used by the RA.
    """

    mortality_monthly: Callable[[FloatArray], FloatArray]
    lapse_monthly: float
    discount_annual: float
    expense_acquisition: float
    expense_maintenance_annual: float
    expense_inflation: float
    ra_confidence: float
    claims_cv: float

    @property
    def discount_monthly(self) -> float:
        """Monthly discount rate equivalent to ``discount_annual``."""
        return (1.0 + self.discount_annual) ** (1.0 / 12.0) - 1.0
