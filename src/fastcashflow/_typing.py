"""Shared type aliases.

``FloatArray`` / ``IntArray`` are the canonical names for the numpy 1D/2D
arrays the engine moves around -- they exist so that an explicit
``NDArray[np.float64]`` doesn't appear in every signature. ``RateFn``
and ``DurationRateFn`` describe the two callable shapes the engine
accepts for rate assumptions.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Per-policy-year rate (the standard shape -- mortality, lapse,
# waiver_incidence, ci_incidence, etc). Called by the engine on the
# ``(sex, issue_age, duration_year, issue_class)`` grid; returns the
# annual rate per cell. The engine constant-force-converts the result to
# monthly. ``issue_class`` is the at-issue classification axis (직업class
# / UW class) the rate table may key on; a table without that axis
# broadcasts over it. Legacy three-arg callables are auto-wrapped to this
# four-arg shape in ``Assumptions.__post_init__``.
RateFn = Callable[[IntArray, FloatArray, IntArray, IntArray], FloatArray]

# Per-policy-year-and-state-duration rate (the semi-Markov shape -- the
# rate depends not only on attained-age duration but also on the sojourn
# time in the current state). ci_reincidence_annual and
# disability_recovery_annual use this signature. The fourth argument is
# the cohort index (months since entering the source state).
DurationRateFn = Callable[
    [IntArray, FloatArray, IntArray, IntArray], FloatArray
]
