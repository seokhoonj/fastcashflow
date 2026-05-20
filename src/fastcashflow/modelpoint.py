"""Model point data -- the contracts to be projected."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.coverage import DEATH


@dataclass(frozen=True, slots=True)
class ModelPointSet:
    """Columnar model point data.

    Every scalar field is a numpy array of length ``n_mp``; the model-point
    axis is the vectorised dimension throughout the engine. All monetary
    amounts are per single policy.

    A policy's claim benefits are a variable-length list of *coverages* (see
    :mod:`fastcashflow.coverage`), held in CSR (Compressed Sparse Row) form
    so the kernels loop them generically -- new benefit types add no fields:

    * ``cov_kind[k]``   -- the coverage kind (``DEATH``, ...).
    * ``cov_amount[k]`` -- the benefit amount of coverage ``k``.
    * ``cov_offset``    -- ``(n_mp+1,)``; policy ``mp``'s coverages are the
      slice ``[cov_offset[mp] : cov_offset[mp+1]]``.

    The coverage list is built one of two ways. ``death_benefit`` is the
    convenience for the common case -- it becomes one death coverage per
    policy with a non-zero benefit. For the general case pass the CSR arrays
    ``cov_kind`` / ``cov_amount`` / ``cov_offset`` directly; ``death_benefit``
    is then reconstructed from them so it stays a readable field either way.

    Premiums and survival benefits stay as plain fields -- they do not
    proliferate the way claim benefits do:

    * ``monthly_premium``  -- level premium charged each in-force month.
    * ``single_premium``   -- one-off premium at t = 0.
    * ``maturity_benefit`` -- benefit on survival to the end of the term.
    * ``annuity_payment``  -- survival income paid each in-force month.
    """

    issue_age: FloatArray          # attained age at issue, in years
    monthly_premium: FloatArray    # level premium, charged each in-force month
    term_months: IntArray          # coverage term, in months
    death_benefit: FloatArray | None = None      # convenience -> a death coverage
    maturity_benefit: FloatArray | None = None   # benefit on survival to term
    annuity_payment: FloatArray | None = None    # survival income, each month
    single_premium: FloatArray | None = None     # one-off premium at t = 0
    cov_kind: IntArray | None = None             # CSR: coverage kind
    cov_amount: FloatArray | None = None         # CSR: coverage amount
    cov_offset: IntArray | None = None           # CSR: per-policy slice bounds

    def __post_init__(self) -> None:
        # Normalise the required fields to numpy arrays of the right dtype.
        for name, dtype in (
            ("issue_age", np.float64),
            ("monthly_premium", np.float64),
            ("term_months", np.int64),
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        n_mp = self.issue_age.shape[0]
        # Premiums / survival benefits default to zero (absent).
        for name in ("maturity_benefit", "annuity_payment", "single_premium"):
            value = getattr(self, name)
            value = np.zeros(n_mp) if value is None else np.asarray(value, np.float64)
            object.__setattr__(self, name, value)
        # Coverage CSR: explicit arrays win; otherwise build from death_benefit.
        if self.cov_kind is not None:
            cov_kind = np.asarray(self.cov_kind, np.int64)
            cov_amount = np.asarray(self.cov_amount, np.float64)
            cov_offset = np.asarray(self.cov_offset, np.int64)
            # Reconstruct death_benefit so it stays a readable field.
            mp_of_cov = np.repeat(np.arange(n_mp), np.diff(cov_offset))
            is_death = cov_kind == DEATH
            death_benefit = np.bincount(
                mp_of_cov[is_death], weights=cov_amount[is_death], minlength=n_mp
            )
        else:
            db = self.death_benefit
            death_benefit = np.zeros(n_mp) if db is None else np.asarray(db, np.float64)
            present = death_benefit != 0.0     # a zero benefit needs no entry
            cov_kind = np.full(int(present.sum()), DEATH, np.int64)
            cov_amount = np.ascontiguousarray(death_benefit[present])
            cov_offset = np.concatenate(
                (np.zeros(1, np.int64), np.cumsum(present, dtype=np.int64))
            )
        object.__setattr__(self, "death_benefit", death_benefit)
        object.__setattr__(self, "cov_kind", cov_kind)
        object.__setattr__(self, "cov_amount", cov_amount)
        object.__setattr__(self, "cov_offset", cov_offset)

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
