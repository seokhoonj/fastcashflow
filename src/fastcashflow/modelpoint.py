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

    The product is a combination of benefits and premiums:

    * ``death_benefit``    -- paid on death (term / whole life).
    * ``maturity_benefit`` -- paid on survival to the end of the term
      (endowment; a pure endowment has no death benefit).
    * ``annuity_payment``  -- paid each in-force month while the policyholder
      survives (annuity).
    * ``monthly_premium``  -- level premium charged each in-force month.
    * ``single_premium``   -- one-off premium at t = 0 (single-premium
      products, e.g. an immediate annuity).
    """

    issue_age: FloatArray         # attained age at issue, in years
    death_benefit: FloatArray     # benefit paid on death
    monthly_premium: FloatArray   # level premium, charged each in-force month
    term_months: IntArray         # coverage term, in months
    maturity_benefit: FloatArray | None = None  # benefit on survival to term
    annuity_payment: FloatArray | None = None   # survival income, each month
    single_premium: FloatArray | None = None    # one-off premium at t = 0

    def __post_init__(self) -> None:
        # Normalise the required fields to numpy arrays of the right dtype.
        for name, dtype in (
            ("issue_age", np.float64),
            ("death_benefit", np.float64),
            ("monthly_premium", np.float64),
            ("term_months", np.int64),
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        # The optional benefit / premium fields default to zero (absent).
        n_mp = self.issue_age.shape[0]
        for name in ("maturity_benefit", "annuity_payment", "single_premium"):
            value = getattr(self, name)
            if value is None:
                value = np.zeros(n_mp)
            object.__setattr__(self, name, np.asarray(value, dtype=np.float64))

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
        annuity_payment: float = 0.0,
        single_premium: float = 0.0,
    ) -> ModelPointSet:
        """Build a one-policy set -- a convenience for hand checks."""
        return cls(
            issue_age=np.array([issue_age]),
            death_benefit=np.array([death_benefit]),
            monthly_premium=np.array([monthly_premium]),
            term_months=np.array([term_months]),
            maturity_benefit=np.array([maturity_benefit]),
            annuity_payment=np.array([annuity_payment]),
            single_premium=np.array([single_premium]),
        )
