"""Coverage kinds -- the benefit-trigger registry.

A policy's benefits are a variable-length list of *coverages* rather than a
fixed set of fields. Each coverage has a *kind*; the kind selects the rate
that drives it and the risk class it belongs to. New benefit types are added
here as kinds, with no change to the projection kernels -- they loop the
coverage list generically.

The integer value of a kind indexes the first axis of the rate grid the
kernels read (see :func:`coverage_rates`), so the values are stable.
"""
from __future__ import annotations

import numpy as np

from fastcashflow._typing import IntArray

# Coverage kinds. The integer is the rate-grid index.
DEATH = 0        # lump sum on death; driven by the mortality rate
INPATIENT = 1    # inpatient (hospitalisation) benefit; driven by its rate
SURGERY = 2      # surgery benefit; driven by the surgery rate
OUTPATIENT = 3   # outpatient benefit; driven by the outpatient-visit rate

# The morbidity kinds, in rate-grid order. DEATH (kind 0) is the mortality
# kind; these are non-decrementing -- a claim does not remove the policy, so
# the same policy can claim repeatedly (multiple-occurrence health benefits).
MORBIDITY_KINDS = (INPATIENT, SURGERY, OUTPATIENT)
N_KINDS = 1 + len(MORBIDITY_KINDS)

# Risk class of a coverage's claims, indexed by kind: 0 mortality, 1 morbidity.
# The Risk Adjustment prices the two with separate coefficients of variation.
RISK_MORTALITY = 0
RISK_MORBIDITY = 1
COVERAGE_RISK: IntArray = np.array(
    [RISK_MORTALITY, RISK_MORBIDITY, RISK_MORBIDITY, RISK_MORBIDITY], np.int64
)


def coverage_rates(mortality, morbidity_rates, issue_age_grid, duration_grid):
    """Stack the per-kind rate grids into one ``(N_KINDS, ..., n_year)`` array.

    A kernel reads a coverage's rate as ``cov_rates[kind, age_or_mp, year]``,
    so the kinds share one 3-D grid whose first axis is the kind. Kind 0
    (death) is the mortality grid; the morbidity kinds are evaluated from
    ``morbidity_rates`` -- a ``{kind: callable}`` map, each callable having
    the same signature as ``Assumptions.mortality_monthly``. A kind absent
    from the map gets a zero grid; a portfolio must not then use it.
    """
    slabs = [mortality]
    for kind in MORBIDITY_KINDS:
        rate = None if morbidity_rates is None else morbidity_rates.get(kind)
        if rate is None:
            slabs.append(np.zeros_like(mortality))
        else:
            slabs.append(np.ascontiguousarray(
                rate(issue_age_grid, duration_grid), dtype=np.float64
            ))
    return np.ascontiguousarray(np.stack(slabs))
