"""Coverage kinds -- the benefit-trigger registry.

A policy's benefits are a variable-length list of *coverages* rather than a
fixed set of fields. Each coverage has a *kind*; the kind selects the rate
that drives it. New benefit types are added here as kinds, with no change to
the projection kernels -- they loop the coverage list generically.

The integer value of a kind indexes the first axis of the rate grid the
kernels read (see :func:`coverage_rates`).
"""
from __future__ import annotations

import numpy as np

from fastcashflow._typing import FloatArray

# Coverage kinds. The integer is the rate-grid index, so values are stable.
DEATH = 0   # lump sum on death; driven by the mortality rate


def coverage_rates(mortality: FloatArray) -> FloatArray:
    """Stack the per-kind rate grids into one array.

    A kernel reads a coverage's rate as ``cov_rates[kind, age_or_mp, year]``,
    so the kinds share a single 3-D grid whose first axis is the kind. Only
    the death coverage exists so far, so the stack is the mortality grid
    alone; further kinds (hospitalisation, surgery, ...) append their own
    rate grids here, leaving the kernels untouched.
    """
    return np.ascontiguousarray(mortality[np.newaxis, :, :])
