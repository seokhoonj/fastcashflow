"""Shared type aliases.

``FloatArray`` / ``IntArray`` are the canonical names for the numpy 1D/2D
arrays the engine moves around -- they exist so that an explicit
``NDArray[np.float64]`` doesn't appear in every signature. ``RateFn``
describes the unified callable shape every annual rate assumption uses.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# The unified rate callable shape -- every annual rate function in
# Basis (mortality, lapse, waiver_incidence, ci_incidence, and the
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
# are auto-wrapped to this five-arg shape in ``Basis.__post_init__``
# -- existing user lambdas continue to work without rewriting.
RateFn = Callable[
    [IntArray, FloatArray, IntArray, IntArray, IntArray], FloatArray
]

# ``DurationRateFn`` is kept as an alias for ``RateFn`` after the
# five-arg unification (ci_reincidence_annual / disability_recovery_annual
# previously used a separate four-arg shape). Code that still imports the
# name from earlier versions stays valid.
DurationRateFn = RateFn

# ``RateLike`` -- anything a Basis rate slot accepts. Normalised to a
# ``RateFn`` at construction by ``basis._as_rate_fn`` (mirrors numpy's
# ``ArrayLike`` -> ``asarray`` -> ``ndarray``):
#
#   * float / int          -- a flat rate, constant over every axis
#   * Sequence[float]      -- an annual rate by policy year (``arr[duration]``);
#                             must cover the term (len*12 >= term_months) or
#                             the projection raises when it runs past the array
#   * polars / pandas DataFrame -- a rate table; axes auto-detected from the
#                             columns (sex / age / issue_age / duration), the
#                             rate read from the ``rate`` column. Duck-typed
#                             (``iter_rows`` / ``to_dict``), no hard pandas dep
#   * RateFn               -- a callable, used as-is (the escape hatch)
#
# DataFrame inputs cannot be expressed in the static union (no hard import);
# they are accepted at runtime via duck typing.
RateLike = float | int | Sequence[float] | RateFn
