"""Actuarial assumption set for the deterministic projection."""
from __future__ import annotations

import inspect
import numbers
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from fastcashflow._typing import DurationRateFn, FloatArray, RateFn, RateLike
from fastcashflow.state_model import StateModel


# RateFn fields on Basis that follow the standard
# ``(sex, issue_age, duration, issue_class, elapsed)`` 5-arg signature
# when a user lambda is written with 4 positional args, the 4th is
# interpreted as ``issue_class`` (the post-Phase-1A shape).
_RATE_FN_FIELDS: tuple[str, ...] = (
    "mortality_annual",
    "lapse_annual",
    "lapse_paidup_annual",
    "waiver_incidence_annual",
    "ci_incidence_annual",
    "premium_factor_annual",
    "annuity_factor_annual",
    "surrender_charge_annual",
    "coi_annual",
)

# DurationRateFn-shape fields on Basis -- semi-Markov rates whose
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
    # Preserve the source-table metadata so describe_basis still
    # surfaces the table_id when the wrapped fn came from io.py.
    for attr in ("_fcf_table_id", "_fcf_sheet", "_fcf_modifiers"):
        if hasattr(fn, attr):
            setattr(wrapped, attr, getattr(fn, attr))
    return wrapped


def _const_rate_fn(value: float) -> RateFn:
    """A flat rate -- ``value`` over every (sex, age, duration, ...) grid."""
    val = float(value)

    def rate(sex, issue_age, duration, issue_class, elapsed):
        shape = np.broadcast_shapes(
            np.asarray(sex).shape, np.asarray(issue_age).shape,
            np.asarray(duration).shape, np.asarray(issue_class).shape,
            np.asarray(elapsed).shape,
        )
        return np.full(shape, val, dtype=np.float64)
    return rate


def _array_rate_fn(arr) -> RateFn:
    """Annual rate by policy year -- ``arr[duration]`` (0-based completed year).

    Raises when the projection reaches a duration past the array: the array
    must cover the contract term (``len * 12 >= term_months``). The other axes
    (sex / issue_age / ...) are ignored -- this is the "already resolved rate
    path" form, for a single / homogeneous segment.
    """
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    n = int(arr.shape[0])

    def rate(sex, issue_age, duration, issue_class, elapsed):
        d = np.asarray(duration, dtype=np.int64)
        if d.size and int(d.max()) >= n:
            raise ValueError(
                f"rate array has {n} entries (policy years 0..{n - 1}) but the "
                f"projection reaches policy year {int(d.max())} -- the array "
                "must cover the contract term (len * 12 >= term_months). "
                "Lengthen the array, or pass a scalar / table / callable."
            )
        return arr[d]
    return rate


def _as_rate_fn(spec: RateLike):
    """Normalise a ``RateLike`` rate spec into a ``RateFn`` callable.

    Single entry point for the polymorphic rate input -- mirrors
    ``np.asarray`` for ``ArrayLike``:

    * ``None`` / callable        -> returned unchanged (callable arity is fixed
                                    afterwards by ``_adapt_rate_arity``)
    * ``int`` / ``float``        -> flat rate (``_const_rate_fn``)
    * polars / pandas DataFrame  -> rate table, axes auto-detected from columns
                                    (``io._rate_fn_from_records``); duck-typed
                                    so neither is a hard dependency
    * 1-D sequence               -> annual rate by policy year (``_array_rate_fn``)
    """
    if spec is None or callable(spec):
        return spec
    # scalar (python or numpy real; bool excluded -- True/False is not a rate)
    if isinstance(spec, numbers.Real) and not isinstance(spec, bool):
        return _const_rate_fn(float(spec))
    # DataFrame: polars (iter_rows) or pandas (to_dict). Checked before the
    # array branch -- a DataFrame is not a 1-D sequence.
    if hasattr(spec, "iter_rows"):                       # polars.DataFrame
        from fastcashflow.io import _rate_fn_from_records
        return _rate_fn_from_records(list(spec.iter_rows(named=True)))
    if hasattr(spec, "to_dict"):                         # pandas.DataFrame
        from fastcashflow.io import _rate_fn_from_records
        return _rate_fn_from_records(spec.to_dict("records"))
    # 1-D array-like -> annual rate by policy year
    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim == 1:
        return _array_rate_fn(arr)
    raise TypeError(
        "rate must be a float, 1-D sequence, polars/pandas DataFrame, or a "
        f"RateFn callable; got {type(spec).__name__} (ndim={arr.ndim}). "
        "For a multi-axis (sex x age) table, pass a DataFrame."
    )


def annual_to_monthly(annual_rate: FloatArray) -> FloatArray:
    """Convert an annual decrement / incidence rate to its monthly equivalent.

    Constant-force basis: the rate acts at a constant intensity across the
    year, so twelve monthly applications reproduce the annual rate exactly --
    ``1 - (1 - q_monthly)**12 == q_annual``. This is the conversion
    consistent with the engine's per-policy-year rate grid, where one rate is
    held flat across the year's twelve monthly steps; a within-year varying
    method (uniform distribution of decrements) cannot be expressed on that
    grid.

    Algebraically equivalent to ``1 - (1 - q)**(1/12)`` but written via
    ``-expm1(log1p(-q)/12)`` so that very small annual rates do not lose
    precision to the ``1 - tiny`` catastrophic cancellation in float64.
    """
    annual = np.asarray(annual_rate, dtype=np.float64)
    # A decrement / incidence rate is a probability in [0, 1]. Non-finite or
    # negative inputs otherwise pass through silently: a NaN propagates to a
    # NaN BEL, and a negative rate round-trips (1 - (1 - q)**12 == q) into a
    # negative "probability" that yields a plausible-looking but meaningless
    # liability. Reject up front. (Discount rates, which may be negative, use
    # discount_monthly_curve, not this function.)
    if not np.all(np.isfinite(annual)):
        raise ValueError(
            "annual_to_monthly: annual rate must be finite (a decrement "
            "probability in [0, 1]); got a NaN / inf value"
        )
    if np.any(annual < 0.0):
        bad = float(np.min(annual))
        raise ValueError(
            f"annual_to_monthly: annual rate must be >= 0.0 (decrement "
            f"probability), got min {bad!r}"
        )
    # A probability above 1.0 makes log1p(-annual) take log of a non-positive
    # number, returning NaN that propagates silently through the engine.
    # Reject up front so the operator sees the bad input, not a NaN BEL.
    if np.any(annual > 1.0):
        bad = float(np.max(annual))
        raise ValueError(
            f"annual_to_monthly: annual rate must be <= 1.0 (decrement "
            f"probability), got max {bad!r}"
        )
    # annual == 1.0 lands on log1p(0) = -inf -> monthly_q = 1.0 (everyone
    # decrements within the month), mathematically correct. Silence the
    # accompanying numpy ``divide by zero in log1p`` RuntimeWarning since
    # the result is well-defined.
    with np.errstate(divide="ignore"):
        return -np.expm1(np.log1p(-annual) / 12.0)


def validate_factor(grid, name: str, expected_shape: tuple) -> FloatArray:
    """Guard a materialised premium / annuity factor grid.

    A factor (``premium_factor_annual`` / ``annuity_factor_annual``) is a free
    Basis callable -- it may legitimately exceed 1.0 for an escalating cash
    flow, so it deliberately never passes through ``annual_to_monthly`` and its
    ``<= 1`` guard. That freedom also means the callable can return the wrong
    shape (a scalar, a mis-broadcast array) or a non-finite / negative value,
    any of which would silently mis-index the kernel, flip a premium / annuity
    cash flow's sign, or poison the BEL with a NaN -- bypassing the
    ``premium >= 0`` invariant on ``ModelPoints``. A factor is a finite,
    non-negative multiple of the right shape (0 is a valid premium holiday /
    deferral). Validate it here, where the callable's output is materialised,
    in every kernel path -- as a ``ValueError`` (an input-contract failure),
    not an ``assert`` (which a non-conforming callable should still hit under
    ``python -O``, where asserts are stripped).
    """
    grid = np.ascontiguousarray(np.asarray(grid, dtype=np.float64))
    if grid.shape != expected_shape:
        raise ValueError(
            f"{name} must return an array of shape {expected_shape} (one value "
            f"per grid cell x policy year); got {grid.shape}. Build it from the "
            f"(sex, issue_age, duration, issue_class, elapsed) arrays it is "
            f"called with, e.g. ``1.0 + 0.1 * duration``."
        )
    if not np.all(np.isfinite(grid)):
        raise ValueError(
            f"{name} returned a non-finite value; the factor must be finite"
        )
    if np.any(grid < 0.0):
        raise ValueError(
            f"{name} returned a negative value; the factor is a non-negative "
            f"multiple on the cash flow (a premium / annuity cannot go negative)"
        )
    return grid


def _single_basis(basis, *, entry: str) -> "Basis":
    """Resolve a :class:`Basis` or :class:`BasisRouter` to a single ``Basis``.

    The entry points that do not route segments (``vfa.measure`` /
    ``paa.measure`` / ``reinsurance.measure`` / ``measure_inforce``) accept a
    one-segment :class:`BasisRouter` and unwrap it; a genuinely multi-segment
    router is rejected with an actionable message rather than crashing deep in
    the kernel. A plain :class:`Basis` passes through unchanged. A bare ``dict``
    is no longer accepted -- build a :class:`BasisRouter` (``read_basis`` does).
    """
    if isinstance(basis, Basis):
        return basis
    if isinstance(basis, BasisRouter):
        return basis.resolve_one(entry=entry)
    raise TypeError(
        f"{entry} takes a Basis or a BasisRouter (from read_basis), got "
        f"{type(basis).__name__}"
    )


@dataclass(frozen=True, slots=True)
class ExpenseItem:
    """One typed entry in the expense ledger.

    Each row is dispatched by ``basis`` and contributes its ``value``
    into the kernel-side expense primitives. Inflation is *not* a row
    attribute -- it lives on :class:`Basis` (``expense_inflation``,
    matching the way ``discount_annual`` lives on :class:`Basis`),
    so a company's economic basis is named in one place and every
    inflation-bearing row picks it up automatically.

    Parameters
    ----------
    expense_type
        Free-form label for reporting / audit (e.g. ``"acquisition"``,
        ``"maintenance"``, ``"collection"``, ``"LAE"``,
        ``"overhead"``). Engine ignores it; ``show_trace`` and
        ``describe_basis`` echo it.
    basis
        Dispatch key -- one of :data:`EXPENSE_BASES`. The five values
        follow the Korean actuarial alpha / beta / gamma convention
        plus a dedicated LAE (Loss Adjustment Expense) slot, each split
        into ``pro_rata`` (proportional to a base amount) or ``fixed``
        (per-policy flat).
    value
        Numeric value -- a fraction (0..1) for the ``_pro_rata`` bases,
        an amount per policy for the ``_fixed`` bases.

    Notes
    -----
    The basis decides whether the global ``expense_inflation`` applies:
    ``gamma_fixed`` and ``lae_pro_rata`` recur every month and so
    inflate; the two ``alpha_*`` bases pay once at ``t=0`` and
    ``beta_pro_rata`` rides the premium itself, so a second inflation
    factor would double-count.
    """

    expense_type: str
    basis: str
    value: float

    def __post_init__(self) -> None:
        # Validate at construction, not deep in derive_expense_components at
        # measure time: a typo'd basis ("alpha" for "alpha_fixed") otherwise
        # surfaces late, and a non-finite value silently NaNs the expense leg.
        if self.basis not in EXPENSE_BASES:
            raise ValueError(
                f"unknown expense basis {self.basis!r}; expected one of "
                f"{EXPENSE_BASES}"
            )
        if not np.isfinite(float(self.value)):
            raise ValueError(
                f"ExpenseItem value must be finite, got {self.value!r}"
            )


#: All ``ExpenseItem.basis`` values the engine knows how to dispatch.
#: Follows the Korean actuarial alpha / beta / gamma convention:
#: alpha = acquisition (one-off at t=0), beta = premium-prorated
#: maintenance, gamma = per-policy fixed maintenance; LAE is the
#: claim-prorated Loss Adjustment Expense slot.
EXPENSE_BASES = (
    "alpha_pro_rata",   # acquisition, % of annualised premium, t=0
    "alpha_fixed",      # acquisition, per policy, t=0
    "beta_pro_rata",    # maintenance, % of premium, every paying month
    "gamma_fixed",      # maintenance, per policy, every month
    "lae_pro_rata",     # LAE, % of claim-type outflow, every month
)

# The valid value-sets for the string-typed Basis fields, named once so the
# engine validates against one source instead of scattered string literals.
RA_METHODS = (
    "confidence_level",   # percentile margin on the risk-bearing PV (default)
    "cost_of_capital",    # CoC rate x capital released over the run-off
)
SURRENDER_VALUE_BASES = (
    "cum_premium_factor",  # factor x cumulative premium (sample-grade default)
    "amount_per_policy",   # contractual surrender amount per policy at duration t
    "amount_per_unit",     # per-policy amount x ModelPoints.surrender_base_amount
)


def derive_expense_components(
    expense_items: tuple["ExpenseItem", ...], n_time: int,
    inflation_index: FloatArray | None = None,
) -> tuple[float, float, float, FloatArray, FloatArray]:
    """Project ``expense_items`` onto the five kernel-side primitives.

    Returns ``(alpha_pro_rata, alpha_fixed, beta_pro_rata, gamma_fixed,
    lae_pro_rata)``:

    - ``alpha_pro_rata`` -- sum of ``value`` over ``alpha_pro_rata``
      rows. Paid at ``t=0`` on annualized premium.
    - ``alpha_fixed`` -- sum of ``value`` over ``alpha_fixed`` rows.
      Paid at ``t=0`` per policy.
    - ``beta_pro_rata`` -- sum of ``value`` over ``beta_pro_rata`` rows.
      Charged each premium-paying month on the actual premium.
    - ``gamma_fixed[t]`` -- per-month per-policy maintenance: each
      ``gamma_fixed`` row contributes ``value / 12 *
      inflation_index[t]``.
    - ``lae_pro_rata[t]`` -- LAE (Loss Adjustment Expense) fraction:
      each ``lae_pro_rata`` row contributes
      ``value * inflation_index[t]``. Applied to the month's
      claim + morbidity + disability total.

    ``inflation_index`` is the ``(n_time,)`` per-month inflation
    multiplier produced by :func:`fastcashflow.curves.inflation_index`;
    a scalar economic ``expense_inflation = i`` gives
    ``inflation_index[t] = (1+i)^(t/12)`` and a per-year curve
    compounds across years. Pass ``None`` for a no-inflation basis
    (every month equal to 1.0).
    """
    alpha_pro_rata = 0.0
    alpha_fixed = 0.0
    beta_pro_rata = 0.0
    gamma_fixed = np.zeros(n_time, dtype=np.float64)
    lae_pro_rata = np.zeros(n_time, dtype=np.float64)
    if inflation_index is None:
        inflation_index = np.ones(n_time, dtype=np.float64)
    for row in expense_items:
        if row.basis == "alpha_fixed":
            alpha_fixed += row.value
        elif row.basis == "alpha_pro_rata":
            alpha_pro_rata += row.value
        elif row.basis == "beta_pro_rata":
            beta_pro_rata += row.value
        elif row.basis == "gamma_fixed":
            gamma_fixed += row.value * inflation_index / 12.0
        elif row.basis == "lae_pro_rata":
            lae_pro_rata += row.value * inflation_index
        else:
            raise ValueError(
                f"unknown expense basis {row.basis!r}; expected one of "
                f"{EXPENSE_BASES}"
            )
    return alpha_pro_rata, alpha_fixed, beta_pro_rata, gamma_fixed, lae_pro_rata


@dataclass(frozen=True, slots=True)
class CoverageRate:
    """One rate-driven coverage's assumption -- a coverage code and how it runs.

    Parameters
    ----------
    code :
        The coverage's code label. The engine works in the integer grid
        index this factorises to; the label is what the model-point file
        names a coverage by.
    rate :
        A :data:`RateLike` -- a flat scalar, a per-policy-year array, a
        polars / pandas rate table, or a :data:`RateFn` callable (the same
        signature as ``mortality_annual``). Whatever the form, it is
        normalised to a ``RateFn`` here; the engine converts the annual rate
        to a monthly one (see :func:`annual_to_monthly`).

    funds_from_account :
        Account-chassis interaction flag (universal-life funding mechanism). When
        ``True`` the coverage's monthly risk charge is drawn from the policy's
        account value -- the death leg's cost-of-insurance on the net amount at
        risk. The coverage's ``rate`` is then the COI rate (``coi_annual``).
        Defaults to ``False`` (a plain rate-driven claim).
    pays_account_balance :
        Account-chassis interaction flag. When ``True`` the coverage's benefit
        reads the account balance -- death pays ``max(account value, face)``.
        Such a coverage is EXCLUDED from the aggregate claim-rate accumulator and
        the rule-bearing claim loop; the account death benefit is written once
        from the rolled account value. Defaults to ``False``.

    Notes
    -----
    Whether a coverage runs as a depleting diagnosis pool vs a recurring
    claim, and which risk class the RA prices it as, is *derived* from
    the portfolio-level :class:`CalculationMethod` taxonomy (the
    ``calculation_methods.csv`` file, surfaced as
    :attr:`fastcashflow.model_points.ModelPoints.calculation_methods`). Those
    two flags do not live on :class:`CoverageRate`. The two account-chassis
    flags above DO live here -- they are a contract-level funding choice, not
    a benefit-method routing key.
    """

    code: str
    rate: RateLike
    funds_from_account: bool = False
    pays_account_balance: bool = False

    def __post_init__(self) -> None:
        # Normalise a RateLike (scalar / array / DataFrame) into a RateFn so
        # the engine always sees a callable. A callable is returned unchanged;
        # its arity is fixed later in ``Basis.__post_init__``.
        coerced = _as_rate_fn(self.rate)
        if coerced is not self.rate:
            object.__setattr__(self, "rate", coerced)


@dataclass(frozen=True, slots=True)
class Basis:
    """Deterministic assumption set -- no assumption changes over time.

    Parameters
    ----------
    mortality_annual :
        Annual mortality-rate callable. Like every rate function on
        :class:`Basis`, it takes the unified five positional grids
        ``(sex, issue_age, duration, issue_class, elapsed)`` and returns an
        array of annual rates of the same shape -- see :data:`RateFn` (in
        ``fastcashflow._typing``) for the full contract: ``sex`` (0 male,
        1 female), ``issue_age`` (years), ``duration`` (completed policy years,
        0-based), ``issue_class`` (at-issue / underwriting class), ``elapsed``
        (semi-Markov sojourn). A table without a given axis broadcasts over it.
        The engine converts the annual rate to a monthly one (see
        :func:`annual_to_monthly`). A select-and-ultimate basis lets the rate
        depend on duration within the select period and on attained age
        (issue_age + duration) beyond it; that logic lives in this callable,
        not the engine.

        A legacy three-arg ``(sex, issue_age, duration)`` callable still works
        (it is auto-wrapped to the five-arg shape). WARNING: do not bake a
        constant in as a *fourth* default parameter --
        ``lambda s, a, d, f=factor: ...`` is read as a four-arg rate, and the
        engine passes ``issue_class`` into ``f``, silently overriding it (wrong
        rates, no error). Capture the constant in a closure instead.
    lapse_annual :
        Same five-arg :data:`RateFn` shape as ``mortality_annual``. Typical
        lapse depends only on duration, but the signature also lets a table
        key on sex / issue_age / issue_class when the workbook carries those
        axes (the engine reads the callable on the full grid either way).
    discount_annual :
        Annual locked-in discount rate (paragraph 36). Either a flat scalar or a
        per-year ``(n_years,)`` array; the engine expands either to a
        per-month rate curve via
        :func:`fastcashflow.curves.discount_monthly_curve`. Used for
        discounting cash flows and for CSM interest accretion.
    expense_items :
        Row-form expense ledger -- a tuple of :class:`ExpenseItem`. Each
        row carries an expense type label (acquisition / maintenance /
        collection / LAE / overhead -- free-form), a
        :data:`EXPENSE_BASES` dispatch key and a numeric value. The
        engine projects every row through
        :func:`derive_expense_components` into the kernel-side primitives
        (alpha / beta / gamma / LAE fractions). An empty tuple is the
        no-expense basis.
    expense_inflation :
        Global annual inflation applied to the recurring expense items
        (``gamma_fixed`` and ``lae_pro_rata``). Either a flat scalar
        -- closed-form ``(1+i)^(t/12)`` growth -- or a per-year
        ``(n_years,)`` array (compounds across years, in-year fractional
        ramp on the current year, held flat past the end). Macro-economic
        assumption, defined once per segment; the I/O layer points the
        segments sheet at one named scenario in the ``inflation_tables``
        sheet (analogous to ``discount_annual`` / ``discount_tables``).
        Does not apply to the two ``_init`` bases (one-time at t=0) or
        to ``premium_pct`` (which already rides the premium).
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
        component of the Risk Adjustment. **VFA-only in v1**: ``vfa.measure``
        uses it directly, but the GMM / PAA RA sums the mortality /
        morbidity / disability / longevity components only. Adding the
        expense term to the GMM RA -- and so closing the gap to the
        IFRS 17 non-financial-risk RA -- is future work; a non-zero
        ``expense_cv`` on a GMM / PAA measurement raises ``NotImplementedError``
        rather than silently doing nothing (set it to 0, or use VFA).
    disability_cv :
        Coefficient of variation of disability cash flows -- disability
        income and the on-transition lump sum -- the disability-risk
        component of the Risk Adjustment.
    ra_method :
        Which Risk Adjustment technique to use -- ``"confidence_level"``
        (the default; a percentile margin on the benefit present values) or
        ``"cost_of_capital"``. The cost-of-capital method is available
        through ``measure(..., full=True)``; the fast path (``full=False``)
        computes the confidence-level RA.
    cost_of_capital_rate :
        Annual cost-of-capital rate for the cost-of-capital RA -- the rate
        charged on the non-financial-risk capital held over the run-off.
    investment_return :
        Annual return earned on the underlying items backing an
        account-value (VFA) contract.
    fund_fee :
        Annual variable-fee rate -- the entity's share of the underlying
        items, deducted from the account value each period (VFA).
    coi_annual :
        Universal-life cost-of-insurance charge rate -- the same five-arg
        :data:`RateFn` shape as ``mortality_annual``. The monthly COI deducted
        from a UL account is ``annual_to_monthly(coi_annual(grid)) * NAR``,
        where the net amount at risk ``NAR = max(0, face - account value)``
        and the face is the model point's ``minimum_death_benefit``. It is a
        contractual charge, DISTINCT from the best-estimate ``mortality_annual``
        used to value actual death claims; their spread is the mortality margin
        that drives the UL CSM. ``None`` charges no COI. UL-only.
    premium_load :
        Universal-life premium load -- the fraction (0..1) of each premium
        withheld before crediting to the account
        (``prem_to_av = premium * (1 - premium_load)``). The full premium is
        still the insurer inflow; the load margin emerges in the fulfilment cash
        flows because only the net-of-load amount grows the account. An
        account-mechanics parameter, not an expense-ledger row. UL-only.
    settlement_pattern :
        Claims run-off pattern -- the fractions of an incurred claim paid in
        the month it is incurred, the next month, and so on, summing to 1.
        ``None`` settles every claim immediately. It measures the liability
        for incurred claims and discounts claims to their payment dates in
        the best-estimate liability.
    coverages :
        Ordered tuple of :class:`CoverageRate` -- the rate-driven coverages
        (death-type, morbidity and diagnosis), one per coverage code.
        No code is reserved: entry ``i`` lives at code ``i``, the integer
        the portfolio's ``coverage_index`` CSR uses to index this tuple. A
        contract's death coverage, if any, is just one entry whose
        ``rate_table`` typically references the same mortality table the
        engine uses as the in-force decrement (``mortality_annual``) --
        the two are different mathematical quantities (decrement vs claim
        payout) that happen to share a table in most products. The taxonomy
        side -- whether a coverage code runs as a diagnosis pool vs a
        recurring claim -- lives on the portfolio
        (:attr:`fastcashflow.model_points.ModelPoints.calculation_methods`),
        not here.
    state_model :
        The product's in-force state machine -- a :class:`~fastcashflow.state_model.StateModel`
        declaring the transient states, their transitions and which states
        pay premium or a benefit. ``None`` uses the default active / waiver
        model
        (:data:`~fastcashflow.state_model.WAIVER_MODEL`); the
        ``waiver_incidence_annual`` rate then drives the active -> waiver
        transition. A product with a different state set supplies its own.
    """

    mortality_annual: RateFn
    lapse_annual: RateFn
    discount_annual: float | FloatArray
    ra_confidence: float
    mortality_cv: float
    # Row-form expense ledger -- see ExpenseItem / derive_expense_components.
    # The engine projects every row into the kernel-side alpha / beta /
    # gamma / claim-handling primitives; an empty tuple is the no-expense
    # basis.
    expense_items: tuple[ExpenseItem, ...] = ()
    # Global economic inflation applied to the recurring expense items
    # (per_policy_monthly, claim_pct). Scalar or per-year curve -- same
    # shape contract as discount_annual; the engine expands either to a
    # per-month inflation_index via fastcashflow.curves.
    expense_inflation: float | FloatArray = 0.0
    # Surrender value curve -- per-month value applied at each
    # policy-duration. Its meaning is set by ``surrender_value_basis``.
    # None = no surrender value (lapse silently removes the contract, the
    # historical behaviour).
    surrender_value_curve: FloatArray | None = None
    # How ``surrender_value_curve`` is interpreted:
    #   "cum_premium_factor" (default, back-compat) -- a factor on cumulative
    #       premium: surrender_cf[t] = lapse_flow[t] x cum_premium[t] x
    #       curve[t]. Sample-grade: cum_premium is path-dependent on
    #       pre-valuation premiums, so the in-force figure is not exact.
    #   "amount_per_policy" -- the curve is the contractual per-policy
    #       surrender amount at policy-duration t (months since inception):
    #       surrender_cf[t] = lapse_flow[t] x curve[t]. Linear in the
    #       in-force, so the in-force count / inforce[elapsed] rescale is
    #       exact (no premium reconstruction, no sample-grade warning).
    #   "amount_per_unit" -- as amount_per_policy, additionally scaled by the
    #       per-MP ``surrender_base_amount`` (explicit; no default base).
    surrender_value_basis: str = "cum_premium_factor"
    waiver_incidence_annual: RateFn | None = None
    # Lapse rate for the paid-up state -- used only by a state model
    # whose paid-up state references the ``lapse_paidup`` transition rate
    # (e.g. STATE_MODELS["WAIVER_PAIDUP"]). Paid-up contracts (premium
    # payment finished) typically surrender at a different rate than
    # premium-paying actives -- the Korean post-payment lapse jump. When
    # None the paid-up state falls back to ``lapse_annual``.
    lapse_paidup_annual: RateFn | None = None
    # Premium SHAPE -- a multiplicative factor on the level ``ModelPoints.premium``
    # by ``(sex, issue_age, duration, issue_class, elapsed)`` (the standard 5-arg
    # RateFn). The charge each premium-paying month is
    # ``premium[mp] * premium_factor_annual(.., year)``: a step-rated / renewable
    # premium is ``f(issue_age + duration)``, a step-up premium
    # is ``1 + step * duration``. ``premium[mp]`` stays the scalar SCALE
    # ``solve_premium`` solves for, so FCF stays linear in it. NOTE this is a
    # multiplicative scale, NOT a decrement -- values may exceed 1.0 (step-up)
    # and it is never run through ``annual_to_monthly``. None -> level premium
    # (factor 1.0 everywhere), bit-identical to the no-shape behaviour.
    premium_factor_annual: RateFn | None = None
    # Annuity SHAPE -- the survival-benefit twin of premium_factor_annual: a
    # multiplicative factor on ``ModelPoints.annuity_payment`` by year, for an
    # escalating annuity (e.g. ``lambda s,a,d,ic,el: 1.05 ** d`` for
    # 5%/yr). Same 5-arg RateFn shape; a multiplicative scale, never
    # annual_to_monthly. None -> level annuity (factor 1.0), bit-identical.
    annuity_factor_annual: RateFn | None = None
    # Universal-life SURRENDER CHARGE -- the fraction of the account value the
    # insurer withholds on a surrender, by policy year, to recover acquisition
    # costs (typically large early and declining to zero, e.g. a 5-arg RateFn
    # ``lambda s,a,d,ic,el: max(0.10 - 0.01 * d, 0.0)``). The account surrender
    # value is ``av_mid * (1 - surrender_charge_annual(.., year))``; a rate in
    # ``[0, 1]``, NOT run through ``annual_to_monthly`` (it is a level-by-year
    # fraction, not a decrement). Applies ONLY to account (universal-life) rows;
    # a term row's curve-based surrender is untouched. None -> no charge (the
    # full account value is paid), bit-identical to the prior behaviour.
    surrender_charge_annual: RateFn | None = None
    # Semi-Markov (Phase (c)) prototype rates. ``ci_incidence_annual`` is the
    # first-cancer diagnosis rate (active -> post_first transition, Markov);
    # ``ci_reincidence_annual`` is the duration-dependent reincidence rate
    # (post_first -> post_second) -- its callable receives an extra
    # ``state_duration`` argument (months since first diagnosis), the
    # natural place to express an exclusion period or any sojourn-
    # time effect.
    ci_incidence_annual: RateFn | None = None
    # ``DurationRateFn`` takes (sex, age, policy_duration, state_duration).
    # The fourth argument is the cohort index (months since entering the
    # source state), the natural place to express an exclusion
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
    # Per-state in-force mortality decrement, keyed by the rate name a state
    # declares via ``State.mortality_rate_name`` (default ``"mortality"``). A
    # post-diagnosis state (post-cancer death) carries an elevated death rate
    # without re-declaring its transition: ``State(mortality_rate_name="dth_post")``
    # plus ``state_mortality_annual={"dth_post": fn}``. A name absent from the
    # dict (or a None dict) falls back to the global ``mortality_annual``, so
    # declaring the state without a table preserves behaviour.
    state_mortality_annual: dict[str, RateFn] | None = None
    longevity_cv: float = 0.0
    morbidity_cv: float = 0.0
    expense_cv: float = 0.0
    disability_cv: float = 0.0
    ra_method: str = "confidence_level"
    cost_of_capital_rate: float = 0.06
    # B119 accounting-policy choice: discount future coverage units when
    # allocating the CSM release (True), or leave them undiscounted (False,
    # default). Affects only the CSM roll-forward (full=True / settle /
    # movement), not the inception CSM_0 on the fast path.
    coverage_unit_discount: bool = False
    investment_return: float = 0.0
    fund_fee: float = 0.0
    # Universal-life cost-of-insurance (COI) charge rate -- the standard 5-arg
    # RateFn shape, like mortality_annual. The monthly COI deducted from a UL
    # account is ``annual_to_monthly(coi_annual(grid)) * NAR`` (NAR = net amount
    # at risk = max(0, face - account value)); it is DISTINCT from the
    # best-estimate ``mortality_annual`` used to value actual claims, and their
    # spread is the mortality margin that emerges as the UL CSM. None means no
    # COI charge (a pure-accumulation account, NAR-charge zero). UL-only -- the
    # GMM / VFA / PAA paths ignore it.
    coi_annual: RateFn | None = None
    # Universal-life premium load -- the fraction of each premium withheld
    # before it is credited to the account: ``prem_to_av = premium * (1 -
    # premium_load)``. The full premium is still the insurer inflow; only the
    # net-of-load amount grows the account (and hence the AV-based benefits), so
    # the load margin emerges in the fulfilment cash flows. An account-mechanics
    # parameter, NOT an expense-ledger row (folding it into expense_items would
    # double-count). 0.0 credits the full premium. UL-only.
    premium_load: float = 0.0
    settlement_pattern: FloatArray | None = None
    coverages: tuple[CoverageRate, ...] = ()
    state_model: StateModel | None = None

    def __post_init__(self) -> None:
        # Reject obviously-wrong scalar basis fields at construction time.
        # ra_confidence is a probability; a value at the boundaries makes
        # _norm_ppf hang or return inf.
        if not (0.0 < self.ra_confidence < 1.0):
            raise ValueError(
                f"ra_confidence must be in the open interval (0, 1), "
                f"got {self.ra_confidence!r}"
            )
        # CV / rate scalars: a NaN slips past a bare ``v < 0`` (NaN < 0 is
        # False) and silently NaNs the RA, so check finiteness explicitly.
        for name in ("mortality_cv", "morbidity_cv", "longevity_cv",
                     "disability_cv", "expense_cv", "cost_of_capital_rate",
                     "fund_fee"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {v!r}")
        if not np.isfinite(self.investment_return) or self.investment_return <= -1.0:
            raise ValueError(
                "investment_return must be finite and > -1.0 (a return <= -100% "
                f"has no monthly equivalent / NaNs the VFA account), got "
                f"{self.investment_return!r}")
        # premium_load is a fraction of premium withheld before crediting -- it
        # must be finite and in [0, 1). A load >= 1 would credit nothing (or a
        # negative amount) to the account; a negative load would credit more
        # than the premium paid.
        if not np.isfinite(self.premium_load) or not (0.0 <= self.premium_load < 1.0):
            raise ValueError(
                "premium_load must be finite and in [0, 1) (a fraction of "
                f"premium withheld before crediting), got {self.premium_load!r}")
        # String-enum fields: catch a typo ("amount_policy", "margins") at
        # construction rather than late in a projection / fast-path branch.
        if self.ra_method not in RA_METHODS:
            raise ValueError(
                f"ra_method must be one of {RA_METHODS}, got {self.ra_method!r}"
            )
        if self.surrender_value_basis not in SURRENDER_VALUE_BASES:
            raise ValueError(
                f"surrender_value_basis must be one of {SURRENDER_VALUE_BASES}, "
                f"got {self.surrender_value_basis!r}"
            )
        sp = self.settlement_pattern
        if sp is not None:
            sp_arr = np.asarray(sp, dtype=np.float64)
            # A pattern that sums to 1 can still hold negative or non-finite
            # weights (e.g. [1.2, -0.2]) that distort LIC / BEL; validate the
            # components, not just the total.
            if sp_arr.ndim != 1 or sp_arr.size == 0:
                raise ValueError("settlement_pattern must be a non-empty 1-D array")
            if not np.all(np.isfinite(sp_arr)):
                raise ValueError("settlement_pattern must be finite")
            if np.any(sp_arr < 0):
                raise ValueError(
                    "settlement_pattern weights must be >= 0 (a negative "
                    "settlement weight distorts LIC / BEL)")
            sp_sum = float(sp_arr.sum())
            if abs(sp_sum - 1.0) > 1e-9:
                raise ValueError(
                    f"settlement_pattern must sum to 1.0, got {sp_sum!r}"
                )
            # A settlement pattern combined with a discount *curve* (a per-year
            # term structure) is not supported: discounting each settlement to
            # its payment date would need a time-varying discount factor inside
            # the kernel (deferred). Every GMM / PAA / VFA / stochastic path
            # otherwise falls back to the first-year (in-year) rate, silently
            # approximating -- reject the combination rather than return a wrong
            # number. A scalar discount_annual with a settlement_pattern is fine.
            disc = np.asarray(self.discount_annual, dtype=np.float64)
            if disc.ndim >= 1 and disc.size > 1:
                raise ValueError(
                    "settlement_pattern with a discount curve (a per-year "
                    "discount_annual) is not supported -- settling claims over "
                    "the pattern needs a time-varying discount factor (deferred); "
                    "the engine would discount every settlement at the first-year "
                    "rate. Use a scalar discount_annual with settlement_pattern, "
                    "or drop the settlement_pattern."
                )
        # discount_annual / expense_inflation may be negative (negative rates
        # are valid) but must be finite and > -1 -- a rate <= -100% has no
        # monthly equivalent and produces NaN, and a NaN / inf otherwise
        # propagates to a silently-NaN BEL with no error.
        for name in ("discount_annual", "expense_inflation"):
            v = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(v)):
                raise ValueError(
                    f"{name} must be finite (a NaN / inf propagates to a "
                    f"silently-NaN liability), got {getattr(self, name)!r}"
                )
            if np.any(v <= -1.0):
                raise ValueError(
                    f"{name} must be > -1.0 (a rate <= -100% has no monthly "
                    f"equivalent / produces NaN), got min {float(np.min(v))!r}"
                )
        # Wrap legacy 3-arg / 4-arg rate callables to the unified 5-arg
        # ``(sex, issue_age, duration, issue_class, elapsed)`` shape the
        # engine now passes everywhere. Built-in callables from io.py are
        # already 5-arg (a no-op detection); legacy user lambdas get an
        # issue_class / elapsed-discarding wrapper. RateFn vs DurationRateFn
        # fields differ in how a legacy 4-arg lambda is interpreted -- see
        # ``_adapt_rate_arity``.
        # Each slot is first normalised from a RateLike (scalar / array /
        # DataFrame) into a callable by ``_as_rate_fn``, then a legacy 3-/4-arg
        # callable is wrapped to the 5-arg shape by ``_adapt_rate_arity``.
        for field in _RATE_FN_FIELDS:
            val = getattr(self, field)
            adapted = _adapt_rate_arity(_as_rate_fn(val))
            if adapted is not val:
                object.__setattr__(self, field, adapted)
        for field in _DURATION_RATE_FN_FIELDS:
            val = getattr(self, field)
            adapted = _adapt_rate_arity(_as_rate_fn(val), is_duration=True)
            if adapted is not val:
                object.__setattr__(self, field, adapted)
        # Per-state mortality callables take the standard RateFn shape; adapt
        # each dict value to the 5-arg signature like the named rate fields.
        if self.state_mortality_annual is not None:
            object.__setattr__(self, "state_mortality_annual", {
                name: _adapt_rate_arity(_as_rate_fn(fn))
                for name, fn in self.state_mortality_annual.items()
            })
        # Coverage rates take the RateFn shape; wrap each coverage's rate too.
        # ``coverages`` is a tuple of frozen CoverageRate dataclasses -- rebuild
        # the tuple with the adapted callables.
        # ``replace`` (not ``CoverageRate(code, rate)``) so every other field --
        # notably the account-chassis flags funds_from_account /
        # pays_account_balance -- survives the rate-arity rebuild. Rebuilding
        # with only (code, rate) silently dropped them, disabling UL routing for
        # any coverage whose rate is a callable.
        new_coverages = tuple(
            (r if r.rate is _adapt_rate_arity(r.rate)
             else replace(r, rate=_adapt_rate_arity(r.rate)))
            for r in self.coverages
        )
        if any(nr is not r for nr, r in zip(new_coverages, self.coverages)):
            object.__setattr__(self, "coverages", new_coverages)
        # Coverage code is the key the engine resolves a model point's coverage
        # against (align_coverages -> {r.code: r}); a duplicate code silently
        # keeps only the last rate (the vintage / revision copy-paste mistake).
        codes = [r.code for r in self.coverages]
        if len(set(codes)) != len(codes):
            seen, dup = set(), []
            for c in codes:
                if c in seen and c not in dup:
                    dup.append(c)
                seen.add(c)
            raise ValueError(
                f"Basis.coverages has duplicate coverage code(s) {dup}; each "
                "code must be unique (a duplicate would silently keep only the "
                "last rate)."
            )

    @property
    def discount_monthly(self) -> float:
        """First-year monthly discount rate, used as a representative scalar.

        Reserved for the few places that need a single rate -- the claims
        settlement-pattern present-value factor (paragraph 40 / B71) -- where the
        in-year rate is the right reference. The per-month rate curve the
        kernels consume is composed by
        :func:`fastcashflow.curves.discount_monthly_curve`, which handles
        both a flat scalar and a per-year curve uniformly.
        """
        d = self.discount_annual
        head = float(d) if np.ndim(d) == 0 else float(np.asarray(d).flat[0])
        return (1.0 + head) ** (1.0 / 12.0) - 1.0


_DESCRIBE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("State transition rates (callable)", (
        "mortality_annual",
        "lapse_annual",
        "waiver_incidence_annual",
        "ci_incidence_annual",
        "ci_reincidence_annual",
        "disability_recovery_annual",
    )),
    ("Economic / expense", (
        "discount_annual",
        "expense_inflation",
    )),
    ("Risk adjustment (RA)", (
        "ra_method",
        "ra_confidence",
        "cost_of_capital_rate",
        "mortality_cv",
        "morbidity_cv",
        "longevity_cv",
        "disability_cv",
        "expense_cv",
    )),
    ("Other (VFA / UL / settlement)", (
        "investment_return",
        "fund_fee",
        "coi_annual",
        "premium_load",
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
        head = "`- " if last else "+- "
        child = prefix + ("    " if last else "|   ")
        if isinstance(item, tuple):
            header, subs = item
            out.append(f"{prefix}{head}{header}")
            _emit_tree(subs, out, child)
        else:
            out.append(f"{prefix}{head}{item}")


def describe_basis(obj, *, file=None) -> None:
    """Print the tree structure of a Basis (or read_basis BasisRouter).

    Groups the fields by role -- rates, economic / expense, risk adjustment,
    coverages / coverage types, state machine, other -- so a reader can see
    what is inside the object without scanning every dataclass field.

    Pass a single :class:`Basis` to see one segment, or pass the
    :class:`BasisRouter` returned by :func:`fastcashflow.io.read_basis` /
    :func:`fastcashflow.io.load_sample_basis` to also see the
    ``(product, channel)`` keys.
    """
    import sys
    out_lines: list[str] = []
    if isinstance(obj, BasisRouter):
        out_lines.append(
            f"BasisRouter  ({len(obj.segments)} segments over {obj.segment_axes})"
        )
        keys = list(obj.segments)
        for i, key in enumerate(keys):
            last = (i == len(keys) - 1)
            head = "`- " if last else "+- "
            child = "    " if last else "|   "
            out_lines.append(f"{head}{key!r}  ->  Basis")
            _describe_basis_lines(obj.segments[key], out_lines, prefix=child)
    elif isinstance(obj, Basis):
        out_lines.append("Basis")
        _describe_basis_lines(obj, out_lines, prefix="")
    else:
        raise TypeError(
            f"describe_basis expects Basis or BasisRouter, got "
            f"{type(obj).__name__}"
        )
    text = "\n".join(out_lines) + "\n"
    (file or sys.stdout).write(text)


def _describe_basis_lines(
    basis: "Basis", out: list[str], *, prefix: str,
) -> None:
    sections: list[tuple[str, list[object]]] = []
    marks = ["1.", "2.", "3.", "4.", "5.", "6."]

    def field_lines(names: tuple[str, ...]) -> list[object]:
        width = max(len(n) for n in names)
        return [f"{n:<{width}}  {_fmt_value(getattr(basis, n))}" for n in names]

    for i, (title, names) in enumerate(_DESCRIBE_GROUPS[:3]):
        body = field_lines(names)
        if i == 1:
            rows = basis.expense_items
            row_lines: list[object] = [
                f"ExpenseItem({r.expense_type!r}, basis={r.basis!r}, "
                f"value={r.value:g})"
                for r in rows
            ]
            body.append((f"expense_items : tuple  (len={len(rows)})", row_lines))
        sections.append((f"{marks[i]} {title}", body))

    coverages = basis.coverages
    coverage_lines: list[object] = []
    width = max((len(r.code) for r in coverages), default=0)
    for r in coverages:
        coverage_lines.append(
            f"CoverageRate(code={r.code!r:{width+2}}, "
            f"rate={_fmt_callable(r.rate)})"
        )
    sections.append((f"{marks[3]} Rider / coverage definitions", [
        (f"coverages : tuple  (len={len(coverages)})", coverage_lines),
    ]))

    sm = basis.state_model
    if sm is None:
        sm_body: list[object] = ["None"]
    else:
        state_items: list[object] = []
        for st in sm.states:
            trs: list[object] = []
            for t in st.transitions:
                target = "exit" if t.to is None else repr(t.to)
                tag = " (pays_lump_sum)" if t.pays_lump_sum else ""
                trs.append(f"{t.rate}  ->  {target}{tag}")
            # Show the non-default state knobs only when set, so an ordinary
            # state renders unchanged and a configured one (an elevated death
            # benefit, a capped / exiting benefit state) is visible.
            extras = []
            if st.periodic_benefit_term_months:
                extras.append(f"periodic_benefit_term_months={st.periodic_benefit_term_months}")
            if st.mortality_rate_name != "mortality":
                extras.append(f"mortality_rate_name={st.mortality_rate_name!r}")
            if st.death_benefit_factor != 1.0:
                extras.append(f"death_benefit_factor={st.death_benefit_factor}")
            extra_str = (", " + ", ".join(extras)) if extras else ""
            state_items.append((
                f"State({st.name!r}, pays_premium={st.pays_premium}, "
                f"pays_periodic_benefit={st.pays_periodic_benefit}, sojourn_tracking_months={st.sojourn_tracking_months}{extra_str})",
                trs,
            ))
        sm_body = [(f"states : tuple  (len={len(sm.states)})", state_items)]
    sections.append((f"{marks[4]} state_model : StateModel", sm_body))

    sections.append((
        f"{marks[5]} {_DESCRIBE_GROUPS[3][0]}",
        field_lines(_DESCRIBE_GROUPS[3][1]),
    ))

    _emit_tree([(t, b) for t, b in sections], out, prefix)


#: The IFRS 17 measurement models a segment may declare. Shared by the router,
#: ``read_basis`` workbook validation, and the planned ``portfolio.measure``.
MEASUREMENT_MODELS = ("GMM", "PAA", "VFA")


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    """The full per-segment routing spec: the :class:`Basis` plus its IFRS 17
    measurement model. The canonical thing a :class:`BasisRouter` stores per
    segment -- model-specific policy options (e.g. a PAA revenue basis) are added
    here as they are wired into the routing kernels.
    """

    basis: Basis
    measurement_model: str = "GMM"


class BasisRouter:
    """Routes a model point to its segment's :class:`SegmentSpec`.

    Returned by :func:`fastcashflow.read_basis`. A ``BasisRouter`` is **not** a
    ``Basis`` and **not** a ``dict`` -- it is the routing *policy* that maps a
    segment key (a tuple over :attr:`segment_axes`, e.g. ``("TERM_LIFE_A",
    "GA")``) to that segment's :class:`SegmentSpec` (its ``Basis`` + measurement
    model). ``measure(mp, router)`` reads :attr:`segment_axes` to route each
    model point with no ``segment_by`` argument; an entry point that needs one
    ``Basis`` calls :meth:`resolve_one`.

    Parameters
    ----------
    segments :
        ``{segment-key: Basis}`` mapping. Copied; the router does not alias it.
    segment_axes :
        The axis names a segment key is read over -- ``("product", "channel")``
        by default, or whatever non-assumption columns the segments sheet
        declares.
    measurement_models :
        Optional ``{segment-key: "GMM"|"PAA"|"VFA"}``; every other segment
        defaults to ``"GMM"``. Validated keyed to ``segments`` so a model can
        never name a non-existent segment.

    Notes
    -----
    It deliberately does **not** implement the mapping protocol (no ``[]`` /
    iteration / ``len``) -- reach the underlying mapping explicitly through
    :attr:`segments` (a read-only view of ``{key: Basis}``), or resolve through
    :meth:`resolve` / :meth:`resolve_spec` / :meth:`resolve_one`. Both internal
    stores are immutable, so the per-segment model can never drift from its
    ``Basis`` after construction.
    """

    __slots__ = ("_specs", "_segments", "segment_axes")

    def __init__(self, segments, segment_axes=("product", "channel"),
                 measurement_models=None):
        models = dict(measurement_models or {})
        for key in models:
            if key not in segments:
                raise ValueError(
                    f"measurement_models key {key!r} is not a segment "
                    f"(known: {list(segments)})"
                )
        specs = {}
        for key, basis in segments.items():
            model = models.get(key, "GMM")
            if model not in MEASUREMENT_MODELS:
                raise ValueError(
                    f"unknown measurement_model {model!r} for segment {key}; "
                    f"expected one of {MEASUREMENT_MODELS}"
                )
            specs[key] = SegmentSpec(basis=basis, measurement_model=model)
        self._specs = MappingProxyType(specs)
        self._segments = MappingProxyType(
            {key: spec.basis for key, spec in specs.items()})
        self.segment_axes = tuple(segment_axes)

    @property
    def axes(self) -> tuple:
        """The segment axis names (alias of :attr:`segment_axes`)."""
        return self.segment_axes

    @property
    def segments(self):
        """Read-only ``{segment-key: Basis}`` view (immutable)."""
        return self._segments

    def resolve(self, key) -> "Basis":
        """The :class:`Basis` for one segment key, e.g. ``("TERM_LIFE_A", "GA")``."""
        return self.resolve_spec(key).basis

    def resolve_spec(self, key) -> "SegmentSpec":
        """The full :class:`SegmentSpec` (Basis + measurement model) for a key."""
        try:
            return self._specs[key]
        except KeyError:
            raise KeyError(
                f"no segment {key!r}; known segments {list(self._specs)}"
            ) from None

    def measurement_model_of(self, key) -> str:
        """The IFRS 17 measurement model ('GMM'|'PAA'|'VFA') for one segment."""
        return self.resolve_spec(key).measurement_model

    def resolve_one(self, *, entry: str = "this operation") -> "Basis":
        """The single :class:`Basis` when there is exactly one segment.

        Raises with an actionable message when the router carries more than one
        segment (the caller does not route segments).
        """
        if len(self._specs) == 1:
            return next(iter(self._specs.values())).basis
        raise ValueError(
            f"{entry} takes a single Basis but the router has "
            f"{len(self._specs)} segments ({list(self._specs)}); it does "
            f"not route segments. Measure each segment on its own basis, e.g. "
            f"{entry}(model_points.subset(rows), router.resolve(segment), ...)."
        )

    def __repr__(self) -> str:
        return (f"<BasisRouter: {len(self._specs)} segment(s) over "
                f"{self.segment_axes}>")
