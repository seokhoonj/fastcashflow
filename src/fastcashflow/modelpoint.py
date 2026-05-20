"""Model point data -- the contracts to be projected."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class ModelPointSet:
    """Columnar model point data.

    Every field is a numpy array of length ``n_mp``; the model-point axis is
    the vectorised dimension throughout the engine.

    Phase 0 product: a level-premium fixed-benefit protection contract. All
    monetary amounts are per single policy.
    """

    issue_age: FloatArray        # attained age at issue, in years
    sum_assured: FloatArray      # benefit paid on death
    monthly_premium: FloatArray  # level premium, charged each in-force month
    term_months: IntArray        # coverage term, in months

    def __post_init__(self) -> None:
        # Normalise every field to a numpy array of the right dtype.
        for name, dtype in (
            ("issue_age", np.float64),
            ("sum_assured", np.float64),
            ("monthly_premium", np.float64),
            ("term_months", np.int64),
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))

    @property
    def n_mp(self) -> int:
        """Number of model points."""
        return int(self.issue_age.shape[0])

    @classmethod
    def single(
        cls,
        issue_age: float,
        sum_assured: float,
        monthly_premium: float,
        term_months: int,
    ) -> ModelPointSet:
        """Build a one-policy set -- a Phase 0 convenience for hand checks."""
        return cls(
            issue_age=np.array([issue_age]),
            sum_assured=np.array([sum_assured]),
            monthly_premium=np.array([monthly_premium]),
            term_months=np.array([term_months]),
        )
