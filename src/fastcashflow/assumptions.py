"""Actuarial assumption set for the deterministic projection."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastcashflow._typing import FloatArray, IntArray

RateFn = Callable[[FloatArray, IntArray], FloatArray]


@dataclass(frozen=True, slots=True)
class Assumptions:
    """Deterministic assumption set -- no assumption changes over time.

    Parameters
    ----------
    mortality_monthly :
        Maps ``(issue_age, duration_years)`` -- arrays of issue age (years)
        and completed policy years (0-based), of the same shape -- to an
        array of monthly mortality rates. A select-and-ultimate basis is
        expressed by letting the rate depend on duration within the select
        period and on attained age (issue_age + duration) beyond it; the
        select-period logic lives in this callable, not the engine.
    lapse_monthly :
        Maps an array of completed policy years (0-based) to an array of
        monthly lapse rates of the same shape.
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
    mortality_cv :
        Coefficient of variation of death claims -- the mortality-risk
        component of the RA.
    longevity_cv :
        Coefficient of variation of survival benefits (maturity benefits and
        annuity payments) -- the longevity-risk component of the RA. The RA
        components are added (the natural mortality / longevity hedge is not
        credited -- conservative for mixed contracts).
    morbidity_cv :
        Coefficient of variation of morbidity claims (hospitalisation,
        surgery, outpatient) -- the morbidity-risk component of the RA.
    expense_cv :
        Coefficient of variation of expense cash flows -- the expense-risk
        component of the Risk Adjustment for account-value (VFA) contracts.
    ra_method :
        Which Risk Adjustment technique to use -- ``"confidence_level"``
        (the default; a percentile margin on the benefit present values) or
        ``"cost_of_capital"``. The cost-of-capital method is available
        through ``measure``; ``value`` computes the confidence-level RA.
    cost_of_capital_rate :
        Annual cost-of-capital rate for the cost-of-capital RA -- the rate
        charged on the non-financial-risk capital held over the run-off.
    investment_return :
        Annual return earned on the underlying items backing an
        account-value (VFA) contract.
    fund_fee :
        Annual variable-fee rate -- the entity's share of the underlying
        items, deducted from the account value each period (VFA).
    guaranteed_credit_rate :
        Annual minimum credited rate guaranteed on an account-value (VFA)
        contract. The account is credited ``max(return, guarantee)`` each
        period, so the guarantee has a cost whenever the return falls short.
        ``None`` means no guarantee.
    settlement_pattern :
        Claims run-off pattern -- the fractions of an incurred claim paid in
        the month it is incurred, the next month, and so on, summing to 1.
        ``None`` settles every claim immediately. Used by the PAA to measure
        the liability for incurred claims.
    morbidity_rates :
        ``{coverage kind: callable}`` map giving the monthly morbidity rate
        of each health coverage kind (see :mod:`fastcashflow.coverage`). Each
        callable has the same signature as ``mortality_monthly``. Required
        only for the kinds a portfolio actually uses.
    """

    mortality_monthly: RateFn
    lapse_monthly: Callable[[IntArray], FloatArray]
    discount_annual: float
    expense_acquisition: float
    expense_maintenance_annual: float
    expense_inflation: float
    ra_confidence: float
    mortality_cv: float
    longevity_cv: float = 0.0
    morbidity_cv: float = 0.0
    expense_cv: float = 0.0
    ra_method: str = "confidence_level"
    cost_of_capital_rate: float = 0.06
    investment_return: float = 0.0
    fund_fee: float = 0.0
    guaranteed_credit_rate: float | None = None
    settlement_pattern: FloatArray | None = None
    morbidity_rates: dict[int, RateFn] | None = None

    @property
    def discount_monthly(self) -> float:
        """Monthly discount rate equivalent to ``discount_annual``."""
        return (1.0 + self.discount_annual) ** (1.0 / 12.0) - 1.0
