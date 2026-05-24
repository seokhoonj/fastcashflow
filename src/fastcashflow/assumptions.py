"""Actuarial assumption set for the deterministic projection."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import DurationRateFn, FloatArray, IntArray, RateFn
from fastcashflow.statemodel import StateModel


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
    # ``DurationRateFn`` takes (sex, age, policy_duration, state_duration).
    # The fourth argument is the cohort index (months since entering the
    # source state), the natural place to express a 면책 (exclusion)
    # period or any sojourn-time effect on the rate.
    ci_reincidence_annual: DurationRateFn | None = None
    # ``disability_recovery_annual`` is the duration-dependent recovery
    # rate (disabled -> active). Same DurationRateFn signature -- the
    # state_duration is the standard DI valuation-table axis along which
    # the recovery rate drops off sharply with claim duration. Pair with
    # a Markov inception rate on the active state's transition (any of
    # ``waiver_incidence_annual`` or a custom slot) to model a full DI
    # contract.
    disability_recovery_annual: DurationRateFn | None = None
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
                "waiver_inception_annual is deprecated and will be removed "
                "in the 0.1.0 release; use waiver_incidence_annual",
                DeprecationWarning, stacklevel=2,
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


_DESCRIBE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("상태 전이율 (state transition rate, callable)", (
        "mortality_annual",
        "lapse_annual",
        "waiver_incidence_annual",
        "ci_incidence_annual",
        "ci_reincidence_annual",
        "disability_recovery_annual",
    )),
    ("경제 / 비용", (
        "discount_annual",
        "expense_acquisition",
        "expense_maintenance_annual",
        "expense_inflation",
    )),
    ("위험조정 (RA)", (
        "ra_method",
        "ra_confidence",
        "cost_of_capital_rate",
        "mortality_cv",
        "morbidity_cv",
        "longevity_cv",
        "disability_cv",
        "expense_cv",
    )),
    ("기타 (VFA / 정산)", (
        "investment_return",
        "fund_fee",
        "guaranteed_credit_rate",
        "settlement_pattern",
    )),
)


def _fmt_value(v: object) -> str:
    if v is None:
        return "None"
    if callable(v):
        return "<callable>"
    if isinstance(v, np.ndarray):
        flat = v.flatten()
        if flat.size <= 4:
            preview = "[" + ", ".join(f"{x:g}" for x in flat) + "]"
        else:
            preview = f"[{flat[0]:g}, ..., {flat[-1]:g}]"
        return f"ndarray shape={tuple(v.shape)} {preview}"
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (int, str)):
        return repr(v)
    return repr(v)


def _emit_tree(lines: list[object], out: list[str], prefix: str) -> None:
    """Render a list of (str | (header, sub_lines)) items as ASCII tree rows."""
    n = len(lines)
    for i, item in enumerate(lines):
        last = (i == n - 1)
        head = "└─ " if last else "├─ "
        child = prefix + ("    " if last else "│   ")
        if isinstance(item, tuple):
            header, subs = item
            out.append(f"{prefix}{head}{header}")
            _emit_tree(subs, out, child)
        else:
            out.append(f"{prefix}{head}{item}")


def describe_assumptions(obj, *, file=None) -> None:
    """Print the tree structure of an Assumptions (or read_assumptions dict).

    Groups the fields by role -- rates, economic / expense, risk adjustment,
    riders / coverage types, state machine, other -- so a reader can see
    what is inside the object without scanning every dataclass field.

    Pass a single :class:`Assumptions` to see one segment, or pass the dict
    returned by :func:`fastcashflow.io.read_assumptions` /
    :func:`fastcashflow.io.load_sample_assumptions` to also see the
    ``(product, channel)`` keys.
    """
    import sys
    out_lines: list[str] = []
    if isinstance(obj, dict):
        out_lines.append(
            f"dict[(product, channel), Assumptions]  ({len(obj)} segments)"
        )
        keys = list(obj.keys())
        for i, key in enumerate(keys):
            last = (i == len(keys) - 1)
            head = "└─ " if last else "├─ "
            child = "    " if last else "│   "
            out_lines.append(f"{head}{key!r}  ->  Assumptions")
            if i == 0:
                _describe_assumptions_lines(obj[key], out_lines, prefix=child)
            else:
                out_lines.append(
                    f"{child}└─ (다른 segment 와 동일 구조; 값만 차이)"
                )
    elif isinstance(obj, Assumptions):
        out_lines.append("Assumptions")
        _describe_assumptions_lines(obj, out_lines, prefix="")
    else:
        raise TypeError(
            f"describe_assumptions expects Assumptions or dict, got "
            f"{type(obj).__name__}"
        )
    text = "\n".join(out_lines) + "\n"
    (file or sys.stdout).write(text)


def _describe_assumptions_lines(
    asmp: "Assumptions", out: list[str], *, prefix: str,
) -> None:
    sections: list[tuple[str, list[object]]] = []
    marks = ["1.", "2.", "3.", "4.", "5.", "6."]

    def field_lines(names: tuple[str, ...]) -> list[object]:
        width = max(len(n) for n in names)
        return [f"{n:<{width}}  {_fmt_value(getattr(asmp, n))}" for n in names]

    for i, (title, names) in enumerate(_DESCRIBE_GROUPS[:3]):
        sections.append((f"{marks[i]} {title}", field_lines(names)))

    riders = asmp.riders
    cov = asmp.coverage_types
    rider_lines: list[object] = [
        "(rate 는 워크북 'rider_rate_tables' 시트의 row 를 wrap)",
    ]
    width = max((len(r.code) for r in riders), default=0)
    for r in riders:
        rider_lines.append(
            f"RiderRate(code={r.code!r:{width+2}}, risk={r.risk}, "
            f"is_diagnosis={r.is_diagnosis}, rate=<callable>)"
        )
    cov_lines: list[object] = [f"{k!r:12} -> {v!r}" for k, v in cov.items()]
    sections.append((f"{marks[3]} 특약 / 담보 정의", [
        (f"riders : tuple  (len={len(riders)})", rider_lines),
        (f"coverage_types : dict  (len={len(cov)})", cov_lines),
    ]))

    sm = asmp.state_model
    if sm is None:
        sm_body: list[object] = ["None"]
    else:
        state_items: list[object] = []
        for st in sm.states:
            trs: list[object] = []
            for t in st.transitions:
                target = "exit" if t.to is None else repr(t.to)
                tag = " (lump_sum)" if t.lump_sum else ""
                trs.append(f"{t.rate}  ->  {target}{tag}")
            state_items.append((
                f"State({st.name!r}, premium={st.premium}, "
                f"benefit={st.benefit}, duration_max={st.duration_max})",
                trs,
            ))
        sm_body = [(f"states : tuple  (len={len(sm.states)})", state_items)]
    sections.append((f"{marks[4]} state_model : StateModel", sm_body))

    sections.append((
        f"{marks[5]} {_DESCRIBE_GROUPS[3][0]}",
        field_lines(_DESCRIBE_GROUPS[3][1]),
    ))

    _emit_tree([(t, b) for t, b in sections], out, prefix)
