"""Model point data -- the contracts to be projected."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class ModelPointSet:
    """Columnar model point data.

    Every field is a numpy array of length ``n_mp``; the model-point axis is
    the vectorised dimension throughout the engine. All monetary amounts are
    per single policy.

    The product is a combination of benefits: a positive ``death_benefit``
    with zero ``maturity_benefit`` is term / whole life; adding a positive
    ``maturity_benefit`` (paid on survival to the end of the term) makes it
    an endowment, and a zero ``death_benefit`` a pure endowment.
    """

    issue_age: FloatArray         # attained age at issue, in years
    death_benefit: FloatArray     # benefit paid on death
    monthly_premium: FloatArray   # level premium, charged each in-force month
    term_months: IntArray         # coverage term, in months
    maturity_benefit: FloatArray | None = None  # benefit on survival to term

    def __post_init__(self) -> None:
        # Normalise the required fields to numpy arrays of the right dtype.
        for name, dtype in (
            ("issue_age", np.float64),
            ("death_benefit", np.float64),
            ("monthly_premium", np.float64),
            ("term_months", np.int64),
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        # maturity_benefit is optional -- absent means a pure protection contract.
        maturity = self.maturity_benefit
        if maturity is None:
            maturity = np.zeros(self.issue_age.shape[0])
        object.__setattr__(
            self, "maturity_benefit", np.asarray(maturity, dtype=np.float64)
        )

    @property
    def n_mp(self) -> int:
        """Number of model points."""
        return int(self.issue_age.shape[0])

    @classmethod
    def single(
        cls,
        issue_age: float,
        death_benefit: float,
        monthly_premium: float,
        term_months: int,
        maturity_benefit: float = 0.0,
    ) -> ModelPointSet:
        """Build a one-policy set -- a convenience for hand checks."""
        return cls(
            issue_age=np.array([issue_age]),
            death_benefit=np.array([death_benefit]),
            monthly_premium=np.array([monthly_premium]),
            term_months=np.array([term_months]),
            maturity_benefit=np.array([maturity_benefit]),
        )
