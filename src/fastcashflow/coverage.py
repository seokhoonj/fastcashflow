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
new rider needs no kernel change.
"""
from __future__ import annotations

import numpy as np

# Code 0 -- the main-contract death coverage, driven by the base mortality.
DEATH = 0

# Coverage mechanic types. The riders sheet tags each rider code with one of
# these; the type fixes how the engine drives the coverage.
TYPE_DEATH_MAIN = "DEATH_MAIN"  # main-contract death; base mortality; code 0
TYPE_DEATH      = "DEATH"       # death-type rider; own rate; non-decrementing
TYPE_MORBIDITY  = "MORBIDITY"   # recurring health claim (inpatient, surgery..)
TYPE_DIAGNOSIS  = "DIAGNOSIS"   # single-payment benefit; depleting pool
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
TYPE_ANNUITY    = "ANNUITY"     # monthly survival income
TYPE_MATURITY   = "MATURITY"    # survival benefit paid at the end of the term

# Rate-driven types carry a sex x age rate table and go in the coverage list.
# Survival types (annuity, maturity) are paid to the in-force survivors and
# need no rate; they are summed into per-policy amounts, not the rate grid.
RATE_DRIVEN_TYPES = (TYPE_DEATH, TYPE_MORBIDITY, TYPE_DIAGNOSIS)
SURVIVAL_TYPES = (TYPE_ANNUITY, TYPE_MATURITY)
COVERAGE_TYPES = (TYPE_DEATH_MAIN,) + RATE_DRIVEN_TYPES + SURVIVAL_TYPES

# Risk class of a coverage's claims: 0 mortality, 1 morbidity. The Risk
# Adjustment prices the two with separate coefficients of variation.
RISK_MORTALITY = 0
RISK_MORBIDITY = 1


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


def coverage_arrays(riders):
    """Per-code kernel flag arrays for the coverage list.

    ``riders`` is the ordered rate-driven riders (codes 1..n); code 0, the
    main-contract death coverage, is prepended -- a recurring claim of
    mortality risk. Returns ``(cov_is_diagnosis, cov_risk)``, each indexed
    by coverage code.
    """
    cov_is_diagnosis = np.array(
        [False] + [r.is_diagnosis for r in riders], np.bool_
    )
    cov_risk = np.array(
        [RISK_MORTALITY] + [r.risk for r in riders], np.int64
    )
    return cov_is_diagnosis, cov_risk
