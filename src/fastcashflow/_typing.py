"""Shared type aliases.

``FloatArray`` / ``IntArray`` are the canonical names for the numpy 1D/2D
arrays the engine moves around -- they exist so that an explicit
``NDArray[np.float64]`` doesn't appear in every signature. ``RateFn``
describes the unified callable shape every annual rate assumption uses.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# The unified rate callable shape -- every annual rate function in
# Assumptions (mortality, lapse, waiver_incidence, ci_incidence, and the
# duration-dependent ci_reincidence / disability_recovery) takes the same
# five positional grids:
#
#   (sex, issue_age, duration, issue_class, elapsed) -> annual rate
#
# ``issue_class`` is the at-issue classification axis (직업class / UW
# class) the rate table may key on. ``elapsed`` is the semi-Markov sojourn
# axis (state-duration since entering the source state). A table without
# a given axis broadcasts over it; the engine passes zeros for axes a
# particular call does not exercise. Legacy 3-arg or 4-arg user callables
# are auto-wrapped to this five-arg shape in ``Assumptions.__post_init__``
# -- existing user lambdas continue to work without rewriting.
RateFn = Callable[
    [IntArray, FloatArray, IntArray, IntArray, IntArray], FloatArray
]

# ``DurationRateFn`` is kept as an alias for ``RateFn`` after the
# five-arg unification (ci_reincidence_annual / disability_recovery_annual
# previously used a separate four-arg shape). Code that still imports the
# name from earlier versions stays valid.
DurationRateFn = RateFn
