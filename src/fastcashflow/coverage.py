"""Coverage codes -- the benefit-trigger registry.

A policy's benefits are a variable-length list of *coverages* rather than a
fixed set of fields. Each coverage carries a numeric *code* -- a factorised
rider identifier. Code 0 is reserved for the main-contract death
benefit, driven by the base mortality so its claim rate matches the in-force
decrement exactly. Codes 1.. are the rate-driven riders the assumptions
register (see :class:`fastcashflow.assumptions.CoverageRate`), in registration
order.

The kernels loop the coverage list generically. A coverage's mechanic is
given by two per-code arrays -- ``cov_is_diagnosis`` (a single-payment
benefit whose claims run off a depleting pool) and ``cov_risk`` (the risk
class the Risk Adjustment prices) -- built by :func:`coverage_arrays`, so a
new rider needs no kernel change. The two arrays are *derived* from the
portfolio's ``benefit_patterns`` taxonomy (the
:class:`~fastcashflow.modelpoints.ModelPoints` ``benefit_patterns`` dict);
the company-level taxonomy is the single source of truth for whether a
coverage is a diagnosis pool or a recurring claim, and which risk class
the RA prices.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

# Code 0 -- the main-contract death coverage, driven by the base mortality.
DEATH = 0


class BenefitPattern(str, Enum):
    """How a benefit pays out -- the engine's calculation routing key.

    Five uniform patterns: every rate-driven death coverage (main-contract
    or rider, accidental or all-cause, ADB / disease / disaster) is the
    same DEATH pattern; the rate table is what differentiates them.
    DEATH_MAIN as a separate pattern is collapsed away -- it was a
    routing detail of the engine's slot 0 (where ``mortality_annual``
    drives both the in-force decrement and the main contract death
    claim), not a kind of benefit. The reserved string code
    ``"DEATH_MAIN"`` in the portfolio's :class:`ModelPoints`
    ``benefit_patterns`` taxonomy is what marks the main-contract slot
    today; future work removes that slot entirely and folds main-contract
    death into the ordinary CSR coverage list.

    ``str, Enum`` -- members compare equal to their string value
    (``BenefitPattern.MORBIDITY == "MORBIDITY"``), so existing numpy
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
        # Default str() on a (str, Enum) returns "BenefitPattern.MEMBER" in
        # Python 3.11+, which breaks numpy comparisons against string arrays
        # (the value gets stringified to the qualified name before dtype
        # casting). Override to return the bare value so str(member),
        # f-strings, and numpy array casts all yield "DIAGNOSIS", not
        # "BenefitPattern.DIAGNOSIS".
        return self._value_


# Reserved coverage_code naming the main-contract death slot. Until the
# engine's code-0 slot is folded into the ordinary CSR coverage list (a
# larger refactor in a later phase), this string code is what marks the
# main-contract death amount on the portfolio: the ``death_benefit``
# field of :class:`~fastcashflow.modelpoints.ModelPoints` (and the
# ``death_benefit`` wide-form column) flow into the CSR slot whose rate
# is :attr:`~fastcashflow.assumptions.Assumptions.mortality_annual`.
MAIN_DEATH_CODE = "DEATH_MAIN"


# Rate-driven patterns carry a sex x age rate table and go in the coverage
# list. Survival patterns (annuity, maturity) are paid to the in-force
# survivors and need no rate; they are summed into per-policy amounts, not
# the rate grid.
RATE_DRIVEN_PATTERNS = (
    BenefitPattern.DEATH, BenefitPattern.MORBIDITY, BenefitPattern.DIAGNOSIS,
)
SURVIVAL_PATTERNS = (BenefitPattern.ANNUITY, BenefitPattern.MATURITY)
BENEFIT_PATTERNS = RATE_DRIVEN_PATTERNS + SURVIVAL_PATTERNS

# Risk class of a coverage's claims: 0 mortality, 1 morbidity. The Risk
# Adjustment prices the two with separate coefficients of variation.
RISK_MORTALITY = 0
RISK_MORBIDITY = 1


def pattern_attrs(pattern: BenefitPattern) -> tuple[bool, int]:
    """Derive ``(is_diagnosis, risk)`` from a :class:`BenefitPattern`.

    The two flags drive the kernel branch a coverage takes -- a depleting
    diagnosis pool vs a recurring claim, and the RA risk class. They are a
    closed-form function of the pattern, so the engine derives them at
    call time rather than carrying them as separate fields on
    :class:`~fastcashflow.assumptions.CoverageRate`.
    """
    is_diagnosis = (pattern == BenefitPattern.DIAGNOSIS)
    risk = (RISK_MORTALITY if pattern == BenefitPattern.DEATH
            else RISK_MORBIDITY)
    return is_diagnosis, risk


def coverage_rates(mortality, rate_fns, sex_grid, issue_age_grid,
                   duration_grid, issue_class_grid, elapsed_grid):
    """Stack the per-code rate grids into one ``(n_codes, ..., n_year)`` array.

    A kernel reads a coverage's rate as ``cov_rates[code, age_or_mp, year]``,
    so the codes share one grid whose first axis is the code. Slab 0 is the
    base ``mortality`` grid (the main-contract death coverage); slabs 1.. are
    the rate-driven riders, evaluated from ``rate_fns`` -- an ordered list of
    callables, each with the unified ``Assumptions.mortality_annual``
    signature ``(sex, issue_age, duration, issue_class, elapsed)``.

    The rates are passed through as supplied -- annual; the caller converts
    the whole stack to monthly (see ``assumptions.annual_to_monthly``).
    """
    slabs = [mortality]
    for rate in rate_fns:
        slabs.append(np.ascontiguousarray(
            rate(sex_grid, issue_age_grid, duration_grid,
                  issue_class_grid, elapsed_grid),
            dtype=np.float64,
        ))
    return np.ascontiguousarray(np.stack(slabs))


def coverage_arrays(riders, benefit_patterns=None):
    """Per-code kernel flag arrays for the coverage list.

    ``riders`` is the ordered rate-driven riders (codes 1..n); code 0, the
    main-contract death coverage, is prepended -- a recurring claim of
    mortality risk. ``benefit_patterns`` is the portfolio-level taxonomy
    (``{coverage_code: BenefitPattern}``); each rider's pattern looked up
    by code gives the two flags via :func:`pattern_attrs`. When
    ``benefit_patterns`` is ``None`` every rider falls back to
    :data:`BenefitPattern.MORBIDITY` -- the conservative default the
    pre-Plan-B engine used for any rate-driven rider that was not flagged
    a diagnosis.

    Returns ``(cov_is_diagnosis, cov_risk)``, each indexed by coverage code.
    """
    flags: list[tuple[bool, int]] = [(False, RISK_MORTALITY)]
    for r in riders:
        pattern = (benefit_patterns.get(r.code) if benefit_patterns is not None
                   else None) or BenefitPattern.MORBIDITY
        flags.append(pattern_attrs(pattern))
    cov_is_diagnosis = np.array([f[0] for f in flags], np.bool_)
    cov_risk = np.array([f[1] for f in flags], np.int64)
    return cov_is_diagnosis, cov_risk


def order_coverages(rate_driven_codes, benefit_patterns):
    """Validate that every rate-driven code is registered in the taxonomy.

    Returns the input sequence unchanged when validation passes -- the
    function name reflects that this is the *ordering* boundary: the
    rate-driven coverages keep their xlsx-coverages-sheet order (which
    becomes the engine's integer code ordering, with code ``i+1`` for
    entry ``i``), but every code must appear in the
    ``benefit_patterns.csv`` taxonomy. Raises :class:`ValueError`
    listing the missing codes when not.
    """
    if benefit_patterns is None:
        return tuple(rate_driven_codes)
    missing = [c for c in rate_driven_codes if c not in benefit_patterns]
    if missing:
        raise ValueError(
            f"rate-driven coverage code(s) {missing!r} are not in the "
            "benefit_patterns taxonomy -- register them in benefit_patterns.csv "
            f"or remove the rate_table cell in the assumptions workbook "
            "(known patterns: "
            f"{', '.join(sorted(set(benefit_patterns.values()), key=str))})"
        )
    return tuple(rate_driven_codes)
