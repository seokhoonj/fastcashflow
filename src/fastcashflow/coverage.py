"""Coverage codes -- the benefit-trigger registry.

A policy's benefits are a variable-length list of *coverages* rather than a
fixed set of fields. Each coverage carries a numeric *code* -- a factorised
coverage identifier that directly indexes the rate-driven coverages the
assumptions register (see :class:`fastcashflow.assumptions.CoverageRate`),
in registration order. No code is reserved: a contract's death coverage,
if any, is just one entry in the user's coverage catalogue, distinguished
by its :class:`CalculationMethod` (``DEATH``).

The base mortality (``Assumptions.mortality_annual``) is a separate engine
input: it drives the in-force decrement only. A death coverage's claim
payout is driven by its own ``rate_table`` -- usually the same mortality
table referenced from the coverages sheet, occasionally a separately
calibrated death-claim experience table. The decrement and the payment
are different mathematical quantities and the engine treats them as such.

The kernels loop the coverage list generically. A coverage's mechanic is
given by two per-code arrays -- ``coverage_is_diagnosis`` (a single-payment
benefit whose claims run off a depleting pool) and ``coverage_risk`` (the
risk class the Risk Adjustment prices) -- built by :func:`coverage_arrays`,
so a new coverage needs no kernel change. The two arrays are *derived* from the
portfolio's ``calculation_methods`` taxonomy (the
:class:`~fastcashflow.modelpoints.ModelPoints` ``calculation_methods`` dict);
the company-level taxonomy is the single source of truth for whether a
coverage is a diagnosis pool or a recurring claim, and which risk class
the RA prices.
"""
from __future__ import annotations

from enum import Enum

import numpy as np


class CalculationMethod(str, Enum):
    """How a benefit pays out -- the engine's calculation routing key.

    Five uniform methods: every rate-driven death coverage (main contract
    or attached, accidental or all-cause, ADB / disease / disaster) is the
    same DEATH method; the rate table is what differentiates them. The
    method is purely a calculation-routing label -- there is no
    "main-contract" method, because the engine has no reserved coverage
    slot.

    ``str, Enum`` -- members compare equal to their string value
    (``CalculationMethod.MORBIDITY == "MORBIDITY"``), so existing numpy
    array comparisons and dict keys keep working unchanged.
    """

    DEATH      = "DEATH"        # death-type coverage; rate-driven; non-decrementing
    MORBIDITY  = "MORBIDITY"    # recurring health claim (inpatient, surgery..)
    DIAGNOSIS  = "DIAGNOSIS"    # single-payment benefit; depleting pool
    # DIAGNOSIS uses an *independent* competing-risks convention: the "not yet
    # diagnosed" pool depletes by mortality / lapse / state transitions *and*
    # by the diagnosis rate, treated as if they were drawn independently each
    # month. That is a simplification -- in reality a diagnosis often
    # *precedes* and triggers correlated mortality / lapse. The independence
    # convention is fine at the rate ranges actuarial tables typically carry
    # (annual incidence well under 1%); the error grows with the rate (a
    # few tens of basis points of BEL difference at very high incidence vs a
    # dependent treatment). Calibrate the diagnosis rate to reflect the
    # convention, or wrap with a coverage rule (waiting / reduction) when the
    # product's mechanic requires it.
    ANNUITY    = "ANNUITY"      # monthly survival income
    MATURITY   = "MATURITY"     # survival benefit paid at the end of the term

    def __str__(self) -> str:
        # Default str() on a (str, Enum) returns "CalculationMethod.MEMBER" in
        # Python 3.11+, which breaks numpy comparisons against string arrays
        # (the value gets stringified to the qualified name before dtype
        # casting). Override to return the bare value so str(member),
        # f-strings, and numpy array casts all yield "DIAGNOSIS", not
        # "CalculationMethod.DIAGNOSIS".
        return self._value_


# Rate-driven methods carry a sex x age rate table and go in the coverage
# list. Survival methods (annuity, maturity) are paid to the in-force
# survivors and need no rate; they are summed into per-policy amounts, not
# the rate grid.
RATE_DRIVEN_METHODS = (
    CalculationMethod.DEATH, CalculationMethod.MORBIDITY, CalculationMethod.DIAGNOSIS,
)

# Risk class of a coverage's claims: 0 mortality, 1 morbidity. The Risk
# Adjustment prices the two with separate coefficients of variation.
RISK_MORTALITY = 0
RISK_MORBIDITY = 1


def method_attrs(method: CalculationMethod) -> tuple[bool, int]:
    """Derive ``(is_diagnosis, risk)`` from a :class:`CalculationMethod`.

    The two flags drive the kernel branch a coverage takes -- a depleting
    diagnosis pool vs a recurring claim, and the RA risk class. They are a
    closed-form function of the method, so the engine derives them at
    call time rather than carrying them as separate fields on
    :class:`~fastcashflow.assumptions.CoverageRate`.
    """
    is_diagnosis = (method == CalculationMethod.DIAGNOSIS)
    risk = (RISK_MORTALITY if method == CalculationMethod.DEATH
            else RISK_MORBIDITY)
    return is_diagnosis, risk


def build_coverage_rates(rate_fns, sex_grid, issue_age_grid,
                         duration_grid, issue_class_grid, elapsed_grid):
    """Stack the per-code rate grids into one ``(n_codes, ..., n_year)`` array.

    A kernel reads a coverage's rate as ``coverage_rates[code, age_or_mp, year]``,
    so the codes share one grid whose first axis is the code. ``rate_fns`` is
    an ordered list of callables, one per coverage in the assumptions'
    registration order; each has the unified ``Assumptions.mortality_annual``
    signature ``(sex, issue_age, duration, issue_class, elapsed)``. The
    annual rates are returned as supplied -- the caller converts the whole
    stack to monthly (see ``assumptions.annual_to_monthly``).

    When ``rate_fns`` is empty the result is an array of shape
    ``(0,) + sex_grid.shape`` -- a zero-claim portfolio; kernel loops over
    ``coverage_index`` are empty for every MP so the leading-axis-zero array
    is never indexed.
    """
    if not rate_fns:
        # No rate-driven coverages. The grid axes are taken from sex_grid
        # so the result has the same ``ndim`` the kernel was compiled
        # against -- numba dispatch keys on shape rank, not the leading
        # axis length.
        return np.zeros((0,) + sex_grid.shape, dtype=np.float64)
    slabs = []
    for rate in rate_fns:
        slabs.append(np.ascontiguousarray(
            rate(sex_grid, issue_age_grid, duration_grid,
                  issue_class_grid, elapsed_grid),
            dtype=np.float64,
        ))
    return np.ascontiguousarray(np.stack(slabs))


def coverage_arrays(coverages, calculation_methods=None):
    """Per-code kernel flag arrays for the coverage list.

    ``coverages`` is the ordered rate-driven coverages, in the same order as
    :attr:`Assumptions.coverages`; ``calculation_methods`` is the portfolio-level
    taxonomy (``{coverage_code: CalculationMethod}``). Each coverage's method
    looked up by code gives the two flags via :func:`method_attrs`.

    Method resolution per coverage:

    1. If ``calculation_methods`` is a dict and the code is a key, use that.
    2. Else, if the code itself is the bare name of a :class:`CalculationMethod`
       member (``"DEATH"``, ``"MORBIDITY"``, ``"DIAGNOSIS"``, ``"ANNUITY"``,
       ``"MATURITY"``), use that method -- the auto-inference convention
       for terse Python construction.
    3. Else, raise :class:`ValueError` naming the unresolved codes. This is a
       deliberate choice: a silent MORBIDITY fallback hides a configuration
       mistake on the most error-prone surface (a DEATH-only contract whose
       claim payouts would otherwise score zero RA against ``mortality_cv``).

    Returns ``(coverage_is_diagnosis, coverage_risk)``, each indexed by
    coverage code.
    """
    flags: list[tuple[bool, int]] = []
    unresolved: list[str] = []
    for r in coverages:
        method = None
        if calculation_methods is not None:
            method = calculation_methods.get(r.code)
        if method is None:
            # Step 2 -- code-as-method auto-inference.
            try:
                method = CalculationMethod(r.code)
            except ValueError:
                method = None
        if method is None:
            unresolved.append(r.code)
            # Append a placeholder so the loop builds a same-length list;
            # the raise below short-circuits the result.
            flags.append((False, RISK_MORBIDITY))
            continue
        flags.append(method_attrs(method))
    if unresolved:
        valid = ", ".join(p.value for p in CalculationMethod)
        raise ValueError(
            f"coverage code(s) {unresolved!r} have no CalculationMethod: pass a "
            "calculation_methods dict on the model points (or load it from a "
            "calculation_methods.csv) mapping each code to one of "
            f"{{{valid}}} -- or rename the coverage to a CalculationMethod "
            "member name (the auto-inference rule)."
        )
    coverage_is_diagnosis = np.array([f[0] for f in flags], np.bool_)
    coverage_risk = np.array([f[1] for f in flags], np.int64)
    return coverage_is_diagnosis, coverage_risk


def align_coverages(coverages, coverage_codes):
    """Reorder ``Assumptions.coverages`` to the model points' coverage order.

    The model points' ``coverage_index`` integers were built against
    ``coverage_codes`` order (the calculation_methods catalogue, or whatever
    order the model points were constructed in). The kernel reads
    ``coverage_rates[coverage_index[k], ...]``, so the rate stack must be
    built in that same order. This looks each code up in the assumptions'
    coverage registry and returns the coverages reordered to match -- so
    *reading the portfolio never has to know the assumptions' internal
    coverage order*. The assumptions enter only here, at the engine call.

    ``coverage_codes`` of ``None`` (model points built with no pinned order,
    e.g. ``ModelPoints.single`` / direct construction whose ``coverage_index``
    already follows ``Assumptions.coverages``) returns ``coverages`` unchanged.

    Raises :class:`ValueError` if a code the model points reference has no
    registered coverage in the assumptions -- the V4 check: every rate-driven
    coverage the portfolio carries needs a ``rate_table`` in the workbook.
    """
    if not coverage_codes:
        return tuple(coverages)
    by_code = {r.code: r for r in coverages}
    missing = [c for c in coverage_codes if c not in by_code]
    if missing:
        raise ValueError(
            f"coverage code(s) {missing} are referenced by the model points "
            "but have no registered coverage in Assumptions.coverages -- add "
            "each code (with its rate_table) to the assumptions workbook's "
            "coverages sheet so the engine has a rate to apply."
        )
    return tuple(by_code[c] for c in coverage_codes)


def validate_csr_codes(coverage_index, n_coverages, *,
                       coverages=None, calculation_methods=None,
                       expected_coverage_codes=None):
    """Check that every ``coverage_index`` value indexes into the coverage list.

    The CSR's ``coverage_index`` is an integer index into
    :attr:`Assumptions.coverages`; the kernel reads
    ``coverage_rates[coverage_index[k], ...]`` directly. An out-of-range index
    would read past the rate-grid into adjacent contiguous memory, producing
    a silently wrong BEL rather than an :class:`IndexError`. This validator
    catches the mistake at engine entry with a clear message naming the
    offending value(s) and the registered coverage count.

    When ``coverages`` and ``calculation_methods`` are both provided, also
    verifies catalogue consistency: every code registered on
    ``Assumptions.coverages`` must appear in the model points'
    ``calculation_methods`` dict. A drift between the two (typically a swap
    of one without the other) lands a coverage with no routing method
    and the engine falls back to MORBIDITY -- silently wrong.

    When ``expected_coverage_codes`` is provided (the rate-driven code tuple
    the model points were built against), also verifies positional order:
    the ``Assumptions.coverages`` order must match exactly. A reorder
    leaves every code present and the catalogue check passes, but the
    ``coverage_index`` integers now point at the wrong rows of the rate
    stack -- DEATH amounts paid out at cancer rates and so on.

    Empty coverage lists are allowed when no CSR row references them.
    """
    if coverage_index.size == 0:
        return
    max_cov_idx = int(coverage_index.max())
    min_cov_idx = int(coverage_index.min())
    if min_cov_idx < 0 or max_cov_idx >= n_coverages:
        bad = sorted({int(k) for k in coverage_index
                      if k < 0 or k >= n_coverages})
        raise ValueError(
            f"coverage_index value(s) {bad} are out of range: assumptions.coverages "
            f"has {n_coverages} entr{'y' if n_coverages == 1 else 'ies'} "
            f"(valid coverage_index range: 0..{max(n_coverages - 1, 0)}). Either "
            "register the missing coverage on Assumptions.coverages or "
            "rebuild ModelPoints.benefits with a coverage_index that maps to a "
            "registered coverage."
        )
    if coverages is not None and calculation_methods is not None:
        registered = {r.code for r in coverages}
        catalogue = set(calculation_methods)
        missing = sorted(registered - catalogue)
        if missing:
            raise ValueError(
                f"coverage code(s) {missing} are registered on "
                "Assumptions.coverages but absent from the model points' "
                "calculation_methods catalogue. The two must agree on every "
                "rate-driven code -- one was swapped without rebuilding "
                "the other."
            )
    if expected_coverage_codes is not None and coverages is not None:
        current = tuple(r.code for r in coverages)
        expected = tuple(expected_coverage_codes)
        if current != expected:
            raise ValueError(
                "Assumptions.coverages order does not match the order the "
                "model points were built against: coverage_index integers "
                "would silently mean different coverages. "
                f"Assumptions.coverages = {list(current)}, "
                f"ModelPoints.coverage_codes = {list(expected)}. "
                "Rebuild the model points against the current Assumptions, "
                "or restore the original coverage ordering."
            )
