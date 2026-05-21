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
    axis is the vectorised dimension throughout the engine. Monetary amounts
    are stated per single policy; ``count`` is how many policies the model
    point stands for -- it defaults to one (one row per policy: seriatim),
    and a larger value scales the policy linearly through the projection.

    A policy's claim benefits are a variable-length list of *coverages* (see
    :mod:`fastcashflow.coverage`), held in CSR (Compressed Sparse Row) form
    so the kernels loop them generically -- new benefit types add no fields:

    * ``cov_kind[k]``   -- the coverage kind (``DEATH``, ``HOSPITAL``, ...).
    * ``cov_amount[k]`` -- the benefit amount of coverage ``k``.
    * ``cov_offset``    -- ``(n_mp+1,)``; policy ``mp``'s coverages are the
      slice ``[cov_offset[mp] : cov_offset[mp+1]]``.

    The coverage list is built one of three ways. ``death_benefit`` is the
    shortcut for the common case -- one death coverage per policy with a
    non-zero benefit. ``benefits`` is the general form: a ``{kind: amount
    array}`` map covering any mix of kinds. Or pass the CSR arrays
    ``cov_kind`` / ``cov_amount`` / ``cov_offset`` directly. Whichever is
    used, ``death_benefit`` stays a readable field.

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
    death_benefit: FloatArray | None = None      # shortcut -> a death coverage
    benefits: dict[int, FloatArray] | None = None  # general {kind: amount}
    maturity_benefit: FloatArray | None = None   # benefit on survival to term
    annuity_payment: FloatArray | None = None    # survival income, each month
    single_premium: FloatArray | None = None     # one-off premium at t = 0
    account_value: FloatArray | None = None      # account value at issue (VFA)
    cov_kind: IntArray | None = None             # CSR: coverage kind
    cov_amount: FloatArray | None = None         # CSR: coverage amount
    cov_offset: IntArray | None = None           # CSR: per-policy slice bounds
    count: FloatArray | None = None              # policies the row stands for

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
        for name in ("maturity_benefit", "annuity_payment", "single_premium",
                     "account_value"):
            value = getattr(self, name)
            value = np.zeros(n_mp) if value is None else np.asarray(value, np.float64)
            object.__setattr__(self, name, value)
        # count defaults to one policy per model point (seriatim).
        cnt = self.count
        cnt = np.ones(n_mp) if cnt is None else np.asarray(cnt, np.float64)
        object.__setattr__(self, "count", cnt)
        # Coverage CSR: explicit arrays win; otherwise build from the
        # death_benefit shortcut and/or the general benefits map.
        if self.cov_kind is not None:
            cov_kind = np.asarray(self.cov_kind, np.int64)
            cov_amount = np.asarray(self.cov_amount, np.float64)
            cov_offset = np.asarray(self.cov_offset, np.int64)
        else:
            items = []   # (kind, per-mp amount array), in coverage-list order
            db = self.death_benefit
            db = np.zeros(n_mp) if db is None else np.asarray(db, np.float64)
            items.append((DEATH, db))
            if self.benefits is not None:
                for kind, amount in self.benefits.items():
                    items.append((int(kind), np.asarray(amount, np.float64)))
            cov_kind, cov_amount, cov_offset = _build_csr(items, n_mp)
        # death_benefit stays a readable field, reconstructed from the CSR.
        mp_of_cov = np.repeat(np.arange(n_mp), np.diff(cov_offset))
        is_death = cov_kind == DEATH
        object.__setattr__(self, "death_benefit", np.bincount(
            mp_of_cov[is_death], weights=cov_amount[is_death], minlength=n_mp
        ))
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
        account_value: float = 0.0,
        count: float = 1.0,
        benefits: dict[int, float] | None = None,
    ) -> ModelPointSet:
        """Build a single-model-point set -- a convenience for hand checks."""
        return cls(
            issue_age=np.array([issue_age]),
            death_benefit=np.array([death_benefit]),
            monthly_premium=np.array([monthly_premium]),
            term_months=np.array([term_months]),
            maturity_benefit=np.array([maturity_benefit]),
            annuity_payment=np.array([annuity_payment]),
            single_premium=np.array([single_premium]),
            account_value=np.array([account_value]),
            count=np.array([count]),
            benefits=(
                None if benefits is None
                else {k: np.array([v]) for k, v in benefits.items()}
            ),
        )


def _build_csr(
    items: list[tuple[int, FloatArray]], n_mp: int
) -> tuple[IntArray, FloatArray, IntArray]:
    """Pack ``(kind, per-mp amount)`` items into a coverage CSR.

    A zero amount is no coverage. Coverages are ordered by model point, and
    within a model point by the order the kinds appear in ``items``.
    """
    mp_parts, kind_parts, amount_parts = [], [], []
    for kind, amount in items:
        present = amount != 0.0
        mp_idx = np.nonzero(present)[0]
        mp_parts.append(mp_idx)
        kind_parts.append(np.full(mp_idx.size, kind, np.int64))
        amount_parts.append(amount[present])
    all_mp = np.concatenate(mp_parts)
    all_kind = np.concatenate(kind_parts)
    all_amount = np.concatenate(amount_parts)
    order = np.argsort(all_mp, kind="stable")     # group by mp, keep kind order
    cov_kind = np.ascontiguousarray(all_kind[order])
    cov_amount = np.ascontiguousarray(all_amount[order])
    cov_offset = np.concatenate((
        np.zeros(1, np.int64),
        np.cumsum(np.bincount(all_mp, minlength=n_mp), dtype=np.int64),
    ))
    return cov_kind, cov_amount, cov_offset
