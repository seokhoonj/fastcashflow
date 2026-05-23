"""Actuarial assumption set for the deterministic projection."""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.statemodel import StateModel

RateFn = Callable[[IntArray, FloatArray, IntArray], FloatArray]


def annual_to_monthly(annual_rate: FloatArray) -> FloatArray:
    """Convert an annual decrement / incidence rate to its monthly equivalent.

    Constant-force basis: the rate acts at a constant intensity across the
    year, so twelve monthly applications reproduce the annual rate exactly --
    ``1 - (1 - q_monthly)**12 == q_annual``. This is the conversion
    consistent with the engine's per-policy-year rate grid, where one rate is
    held flat across the year's twelve monthly steps; a within-year varying
    method (uniform distribution of decrements) cannot be expressed on that
    grid.
    """
    annual = np.asarray(annual_rate, dtype=np.float64)
    return 1.0 - (1.0 - annual) ** (1.0 / 12.0)


@dataclass(frozen=True, slots=True)
class RiderRate:
    """One rate-driven rider's assumption -- a coverage code and how it runs.

    Parameters
    ----------
    code :
        The rider's code label. The engine works in the integer grid
        index this factorises to; the label is what the model-point file
        names a coverage by.
    rate :
        Annual-rate callable, the same signature as ``mortality_annual``;
        the engine converts it to a monthly rate (see
        :func:`annual_to_monthly`).
    is_diagnosis :
        True for a single-payment diagnosis benefit -- its claims run off a
        depleting "not yet diagnosed" pool. False for a recurring claim
        (a death-type rider or a multiple-occurrence health benefit).
    risk :
        Risk class for the Risk Adjustment -- ``RISK_MORTALITY`` (a
        death-type rider) or ``RISK_MORBIDITY`` (a health rider).
    """

    code: str
    rate: RateFn
    is_diagnosis: bool
    risk: int


@dataclass(frozen=True, slots=True)
class Assumptions:
    """Deterministic assumption set -- no assumption changes over time.

    Parameters
    ----------
    mortality_annual :
        Maps ``(sex, issue_age, duration_years)`` -- arrays of sex (0 male,
        1 female), issue age (years) and completed policy years (0-based),
        of the same shape -- to an array of annual mortality rates. The
        engine converts each to a monthly rate (see
        :func:`annual_to_monthly`). A select-and-ultimate basis is expressed
        by letting the rate depend on duration within the select period and
        on attained age (issue_age + duration) beyond it; the select-period
        logic lives in this callable, not the engine.
    lapse_annual :
        Same ``(sex, issue_age, duration)`` signature as
        ``mortality_annual``. Typical lapse depends only on duration, but the
        signature also lets the reader pick up a per-sex or per-issue_age
        lapse table when the workbook carries those axes (the engine reads
        the callable on the full sex / age / duration grid either way).
    discount_annual :
        Annual locked-in discount rate (Sec. 36). Either a flat scalar or a
        per-year ``(n_years,)`` array; the engine expands either to a
        per-month rate curve via
        :func:`fastcashflow.curves.discount_monthly_curve`. Used for
        discounting cash flows and for CSM interest accretion.
    expense_acquisition :
        One-off acquisition expense per policy, incurred at t = 0.
    expense_maintenance_annual :
        Annual maintenance expense per in-force policy; one twelfth is
        charged each month. Either a flat scalar (the same amount every
        year) or a per-year ``(n_years,)`` array (a step at each year
        boundary). Held flat past the end of the array.
    expense_inflation :
        Annual inflation rate applied to the maintenance expense. Either a
        flat scalar (closed-form ``(1+i)^(t/12)`` growth) or a per-year
        ``(n_years,)`` array (compounds across years, with the in-year
        fractional ramp on the current year). Held flat past the end.
    ra_confidence :
        Confidence level for the Risk Adjustment (e.g. 0.75). The RA lifts
        the liability from its best estimate to this percentile.
    mortality_cv :
        Coefficient of variation of death claims -- the mortality-risk
        component of the RA.
    waiver_incidence_annual :
        Maps ``(sex, issue_age, duration_years)`` to an array of annual
        waiver-incidence rates -- the rate at which active in-force
        transitions to the premium-waived state. Same signature as
        ``mortality_annual``. ``None`` means no transitions: every model
        point keeps its input state for the whole projection. The
        spelling matches the standard actuarial term ``incidence`` -- a
        per-unit-time event rate -- used by the rest of the engine for
        analogous rates (``ci_incidence_annual``,
        ``ci_reincidence_annual``).
    waiver_inception_annual :
        Deprecated alias for ``waiver_incidence_annual``; still accepted
        for backward compatibility but raises ``DeprecationWarning``. Set
        only one of the two.
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
    disability_cv :
        Coefficient of variation of disability cash flows -- disability
        income and the on-transition lump sum -- the disability-risk
        component of the Risk Adjustment.
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
        ``None`` settles every claim immediately. It measures the liability
        for incurred claims and discounts claims to their payment dates in
        the best-estimate liability.
    riders :
        Ordered tuple of :class:`RiderRate` -- the rate-driven riders
        (death-type, morbidity and diagnosis coverages), one per rider code.
        Their order fixes the integer coverage codes: rider ``i`` is code
        ``i + 1``; code 0 is the main-contract death coverage, driven by
        ``mortality_annual``. Empty for a death-only portfolio.
    coverage_types :
        Map of every rider code to its type string -- the riders master. Set
        by :func:`read_assumptions` and used by :func:`read_model_points`
        to route long-form coverage rows; ``None`` when built in code.
    state_model :
        The product's in-force state machine -- a :class:`~fastcashflow.statemodel.StateModel`
        declaring the transient states, their transitions and which states
        pay premium or a benefit. ``None`` uses the default active / waiver
        model
        (:data:`~fastcashflow.statemodel.WAIVER_MODEL`); the
        ``waiver_incidence_annual`` rate then drives the active -> waiver
        transition. A product with a different state set supplies its own.
    """

    mortality_annual: RateFn
    lapse_annual: RateFn
    discount_annual: float | FloatArray
    expense_acquisition: float
    expense_maintenance_annual: float | FloatArray
    expense_inflation: float | FloatArray
    ra_confidence: float
    mortality_cv: float
    waiver_incidence_annual: RateFn | None = None
    # Deprecated alias retained for source compatibility -- ``__post_init__``
    # routes it to ``waiver_incidence_annual`` with a DeprecationWarning.
    waiver_inception_annual: RateFn | None = None
    # Semi-Markov (Phase (c)) prototype rates. ``ci_incidence_annual`` is the
    # first-cancer diagnosis rate (active -> post_first transition, Markov);
    # ``ci_reincidence_annual`` is the duration-dependent reincidence rate
    # (post_first -> post_second) -- its callable receives an extra
    # ``state_duration`` argument (months since first diagnosis), the
    # natural place to express a 면책 (exclusion) period or any sojourn-
    # time effect.
    ci_incidence_annual: RateFn | None = None
    ci_reincidence_annual: object | None = None    # (sex, age, p_dur, s_dur) -> rate
    longevity_cv: float = 0.0
    morbidity_cv: float = 0.0
    expense_cv: float = 0.0
    disability_cv: float = 0.0
    ra_method: str = "confidence_level"
    cost_of_capital_rate: float = 0.06
    investment_return: float = 0.0
    fund_fee: float = 0.0
    guaranteed_credit_rate: float | None = None
    settlement_pattern: FloatArray | None = None
    riders: tuple[RiderRate, ...] = ()
    coverage_types: dict[str, str] | None = None
    state_model: StateModel | None = None

    def __post_init__(self) -> None:
        # Backward-compatibility: accept the deprecated ``waiver_inception_annual``
        # spelling, warn, route to the canonical ``waiver_incidence_annual``.
        # Setting both forms is an error -- the caller has to pick one.
        if self.waiver_inception_annual is not None:
            if self.waiver_incidence_annual is not None:
                raise ValueError(
                    "set waiver_incidence_annual, not both "
                    "waiver_incidence_annual and waiver_inception_annual"
                )
            warnings.warn(
                "waiver_inception_annual is deprecated; "
                "use waiver_incidence_annual",
                DeprecationWarning, stacklevel=3,
            )
            object.__setattr__(
                self, "waiver_incidence_annual", self.waiver_inception_annual,
            )
            object.__setattr__(self, "waiver_inception_annual", None)

    @property
    def discount_monthly(self) -> float:
        """First-year monthly discount rate, used as a representative scalar.

        Reserved for the few places that need a single rate -- the claims
        settlement-pattern present-value factor (Sec. 40 / B71) -- where the
        in-year rate is the right reference. The per-month rate curve the
        kernels consume is composed by
        :func:`fastcashflow.curves.discount_monthly_curve`, which handles
        both a flat scalar and a per-year curve uniformly.
        """
        d = self.discount_annual
        head = float(d) if np.ndim(d) == 0 else float(np.asarray(d).flat[0])
        return (1.0 + head) ** (1.0 / 12.0) - 1.0
