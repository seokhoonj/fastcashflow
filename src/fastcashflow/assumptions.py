"""Actuarial assumption set for the deterministic projection."""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import DurationRateFn, FloatArray, IntArray, RateFn
from fastcashflow.statemodel import StateModel


# RateFn fields on Assumptions that follow the standard
# ``(sex, issue_age, duration, issue_class, elapsed)`` 5-arg signature
# when a user lambda is written with 4 positional args, the 4th is
# interpreted as ``issue_class`` (the post-Phase-1A shape).
_RATE_FN_FIELDS: tuple[str, ...] = (
    "mortality_annual",
    "lapse_annual",
    "waiver_incidence_annual",
    "ci_incidence_annual",
)

# DurationRateFn-shape fields on Assumptions -- semi-Markov rates whose
# legacy 4-arg user lambdas wrote the 4th argument as the cohort index
# (state-duration since entering the source state). After the 5-arg
# unification these map to the new ``elapsed`` axis (the 5th positional
# argument). The adapter knows to shift the legacy 4th -> 5th for these
# fields specifically.
_DURATION_RATE_FN_FIELDS: tuple[str, ...] = (
    "ci_reincidence_annual",
    "disability_recovery_annual",
)


def _adapt_rate_arity(fn, *, is_duration: bool = False):
    """Wrap a legacy rate callable to the 5-arg unified shape.

    The engine now calls every rate as
    ``(sex, issue_age, duration, issue_class, elapsed)``. User callables
    written before the unification may be:

    * 3-arg ``(sex, age, dur)`` -- the pre-axis-extension shape; the
      wrapper discards ``issue_class`` and ``elapsed``.
    * 4-arg, RateFn shape ``(sex, age, dur, issue_class)`` -- the
      Phase-1A shape; the wrapper discards ``elapsed``.
    * 4-arg, DurationRateFn shape ``(sex, age, dur, cohort_index)`` --
      the pre-unification semi-Markov shape; the wrapper maps the
      original 4th arg to ``elapsed`` (the 5th in the new signature).
      Selected via ``is_duration=True`` (the caller knows the field is
      a DurationRateFn slot).
    * 5-arg or ``*args`` -- already the new shape, returned unchanged.

    ``None`` is also passed through unchanged.
    """
    if fn is None:
        return fn
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn   # builtin / C-level callable -- assume the new shape
    params = list(sig.parameters.values())
    # *args absorbs any arity -- no wrapping needed.
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return fn
    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 5:
        return fn   # already 5-arg (or more) -- pass through
    if len(positional) == 4:
        if is_duration:
            # Legacy DurationRateFn: 4th arg is cohort_index -> shift to elapsed.
            def wrapped(sex, issue_age, duration, issue_class, elapsed):
                return fn(sex, issue_age, duration, elapsed)
        else:
            # Legacy Phase-1A RateFn: 4th arg is issue_class.
            def wrapped(sex, issue_age, duration, issue_class, elapsed):
                return fn(sex, issue_age, duration, issue_class)
    elif len(positional) == 3:
        def wrapped(sex, issue_age, duration, issue_class, elapsed):
            return fn(sex, issue_age, duration)
    else:
        return fn   # unusual arity -- leave alone, let the engine error
    # Preserve the source-table metadata so describe_assumptions still
    # surfaces the table_id when the wrapped fn came from io.py.
    for attr in ("_fcf_table_id", "_fcf_sheet", "_fcf_modifiers"):
        if hasattr(fn, attr):
            setattr(wrapped, attr, getattr(fn, attr))
    return wrapped


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
class CoverageRate:
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
    alpha_pct :
        Acquisition cost as a fraction of annualized premium (alpha,
        신계약비 의 % 부분). Paid at t = 0 on the issued count. Korean
        commission is typically dominated by this. Default 0.
    alpha_flat :
        Acquisition cost as a flat amount per policy issued (alpha 의
        정액 부분, 예: 의료심사 / 발급비). Paid at t = 0 on the issued
        count. Usually small or zero in Korean practice. Default 0.
    beta_pct :
        Premium-based recurring expense as a fraction of annualized
        premium (beta, 수금비 + premium-비례 유지비). Charged each month
        while ``t < premium_term_months``; one twelfth of the annual
        amount per month. Default 0.
    gamma_flat :
        Per-policy maintenance expense as an annual flat amount per
        in-force policy (gamma, per-policy 유지비 — 인건비 / IT / 콜센터).
        Charged each month while ``t < term_months`` (continues after
        premium fully paid). One twelfth per month, inflation applied via
        :data:`expense_inflation`. Default 0.
    expense_inflation :
        Annual inflation rate applied to ``gamma_flat``. Either a flat
        scalar (closed-form ``(1+i)^(t/12)`` growth) or a per-year
        ``(n_years,)`` array (compounds across years, with the in-year
        fractional ramp on the current year). Held flat past the end.
        Does not apply to ``alpha_*`` (one-time at t=0) or ``beta_pct``
        (% of locked premium).
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
        Ordered tuple of :class:`CoverageRate` -- the rate-driven riders
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
    ra_confidence: float
    mortality_cv: float
    alpha_pct: float = 0.0
    alpha_flat: float = 0.0
    beta_pct: float = 0.0
    gamma_flat: float | FloatArray = 0.0
    expense_inflation: float | FloatArray = 0.0
    # Surrender value (해약환급금) curve -- per-month factor applied to the
    # cumulative premium paid. Engine: surrender_cf[t] = lapse_flow[t] x
    # cum_premium[t] x surrender_value_curve[t]. None = no surrender value
    # (lapse silently removes the contract, the historical behaviour).
    surrender_value_curve: FloatArray | None = None
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
    coverages: tuple[CoverageRate, ...] = ()
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

        # Wrap legacy 3-arg / 4-arg rate callables to the unified 5-arg
        # ``(sex, issue_age, duration, issue_class, elapsed)`` shape the
        # engine now passes everywhere. Built-in callables from io.py are
        # already 5-arg (a no-op detection); legacy user lambdas get an
        # issue_class / elapsed-discarding wrapper. RateFn vs DurationRateFn
        # fields differ in how a legacy 4-arg lambda is interpreted -- see
        # ``_adapt_rate_arity``. Runs after the waiver_inception routing so
        # the routed callable is wrapped too.
        for field in _RATE_FN_FIELDS:
            adapted = _adapt_rate_arity(getattr(self, field))
            if adapted is not getattr(self, field):
                object.__setattr__(self, field, adapted)
        for field in _DURATION_RATE_FN_FIELDS:
            adapted = _adapt_rate_arity(getattr(self, field), is_duration=True)
            if adapted is not getattr(self, field):
                object.__setattr__(self, field, adapted)
        # Rider rates take the RateFn shape; wrap each rider's rate too.
        # Riders are a tuple of frozen CoverageRate dataclasses -- rebuild the
        # tuple with the adapted callables.
        new_coverages = tuple(
            (r if r.rate is _adapt_rate_arity(r.rate)
             else CoverageRate(
                 code=r.code,
                 rate=_adapt_rate_arity(r.rate),
                 is_diagnosis=r.is_diagnosis,
                 risk=r.risk,
             ))
            for r in self.coverages
        )
        if any(nr is not r for nr, r in zip(new_coverages, self.coverages)):
            object.__setattr__(self, "coverages", new_coverages)

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
        "alpha_pct",
        "alpha_flat",
        "beta_pct",
        "gamma_flat",
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


def _fmt_callable(v: object) -> str:
    """Format a rate callable, surfacing its source table_id when known."""
    tid = getattr(v, "_fcf_table_id", None)
    if tid is None:
        return "<callable>"
    mods = getattr(v, "_fcf_modifiers", ())
    suffix = f" (+{', +'.join(mods)})" if mods else ""
    return f"<callable -> {tid}{suffix}>"


def _fmt_value(v: object) -> str:
    if v is None:
        return "None"
    if callable(v):
        return _fmt_callable(v)
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
            _describe_assumptions_lines(obj[key], out_lines, prefix=child)
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

    riders = asmp.coverages
    cov = asmp.coverage_types
    coverage_lines: list[object] = []
    width = max((len(r.code) for r in riders), default=0)
    for r in riders:
        coverage_lines.append(
            f"CoverageRate(code={r.code!r:{width+2}}, risk={r.risk}, "
            f"is_diagnosis={r.is_diagnosis}, rate={_fmt_callable(r.rate)})"
        )
    cov_lines: list[object] = [f"{k!r:12} -> {v!r}" for k, v in cov.items()]
    sections.append((f"{marks[3]} 특약 / 담보 정의", [
        (f"coverages : tuple  (len={len(riders)})", coverage_lines),
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
