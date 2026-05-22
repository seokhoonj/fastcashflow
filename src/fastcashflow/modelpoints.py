"""Model point data -- the contracts to be projected."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.coverage import (
    DEATH, TYPE_ANNUITY, TYPE_DEATH_MAIN, TYPE_MATURITY,
)

# Contract states -- a model point's in-force valuation state. ACTIVE is the
# ordinary premium-paying contract. WAIVER (premium waived on a triggering
# event) and PAIDUP (the premium-paying term has ended) both keep the
# coverage in force and claims projected, but collect no further premium --
# the fulfilment cash flows reflect the contract's actual terms at the
# measurement date (IFRS 17 Sec. 33-34). WAIVER and PAIDUP differ in cause,
# not in the projected cash flows.
STATE_ACTIVE = 0
STATE_WAIVER = 1
STATE_PAIDUP = 2

# States that collect no premium -- the coverage continues, the premium has
# stopped. The projection zeroes the premium for a model point in any of them.
_NO_PREMIUM_STATES = (STATE_WAIVER, STATE_PAIDUP)

# Names for the file layer -- a model-point ``state`` column reads and writes
# these strings, the readable form a practitioner edits in a spreadsheet.
STATE_NAMES = {"active": STATE_ACTIVE, "waiver": STATE_WAIVER,
               "paidup": STATE_PAIDUP}
STATE_LABELS = {code: name for name, code in STATE_NAMES.items()}


@dataclass(frozen=True, slots=True)
class ModelPoints:
    """Columnar model point data.

    Every scalar field is a numpy array of length ``n_mp``; the model-point
    axis is the vectorised dimension throughout the engine. Monetary amounts
    are stated per single policy; ``count`` is how many policies the model
    point stands for -- it defaults to one (one row per policy: seriatim),
    and a larger value scales the policy linearly through the projection.

    A policy's claim benefits are a variable-length list of *coverages* (see
    :mod:`fastcashflow.coverage`), held in CSR (Compressed Sparse Row) form
    so the kernels loop them generically -- new benefit types add no fields:

    * ``cov_kind[k]``   -- the coverage code (0 = main-contract death, 1..
      the rate-driven riders the assumptions register).
    * ``cov_amount[k]`` -- the benefit amount of coverage ``k``.
    * ``cov_offset``    -- ``(n_mp+1,)``; policy ``mp``'s coverages are the
      slice ``[cov_offset[mp] : cov_offset[mp+1]]``.

    Each coverage may carry a benefit rule: ``cov_waiting`` (months from
    issue with no benefit) and ``cov_reduction_end`` / ``cov_reduction_factor``
    (a benefit multiplier in force until a cut-off month). Both are CSR
    arrays aligned with ``cov_kind`` and default to off -- no waiting, full
    benefit.

    The coverage list is built one of three ways. ``death_benefit`` is the
    shortcut for the common case -- one death coverage per policy with a
    non-zero benefit. ``benefits`` is the general form: a ``{kind: amount
    array}`` map covering any mix of kinds. Or pass the CSR arrays
    ``cov_kind`` / ``cov_amount`` / ``cov_offset`` directly. Whichever is
    used, ``death_benefit`` stays a readable field.

    Premiums and survival benefits stay as plain fields -- they do not
    proliferate the way claim benefits do:

    * ``monthly_premium``     -- level premium charged each in-force month.
    * ``single_premium``      -- one-off premium at t = 0.
    * ``premium_term_months`` -- months the level premium is collected,
      defaulting to the full coverage term.
    * ``maturity_benefit``    -- benefit on survival to the end of the term.
    * ``annuity_payment``     -- survival income paid each in-force month.
    """

    issue_age: FloatArray          # attained age at issue, in years
    monthly_premium: FloatArray    # level premium, charged each in-force month
    term_months: IntArray          # coverage term, in months
    death_benefit: FloatArray | None = None      # shortcut -> a death coverage
    benefits: dict[int, FloatArray] | None = None  # general {kind: amount}
    maturity_benefit: FloatArray | None = None   # benefit on survival to term
    annuity_payment: FloatArray | None = None    # survival income, each month
    single_premium: FloatArray | None = None     # one-off premium at t = 0
    premium_term_months: IntArray | None = None  # months premium is collected
    account_value: FloatArray | None = None      # account value at issue (VFA)
    cov_kind: IntArray | None = None             # CSR: coverage kind
    cov_amount: FloatArray | None = None         # CSR: coverage amount
    cov_offset: IntArray | None = None           # CSR: per-policy slice bounds
    cov_waiting: IntArray | None = None          # CSR: waiting period, months
    cov_reduction_end: IntArray | None = None    # CSR: reduced-benefit end, months
    cov_reduction_factor: FloatArray | None = None  # CSR: reduced-benefit factor
    count: FloatArray | None = None              # policies the row stands for
    sex: IntArray | None = None                  # 0 = male, 1 = female
    state: IntArray | None = None                # contract state (STATE_*)

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
        # sex defaults to 0 (male) for every model point.
        sex = self.sex
        sex = np.zeros(n_mp, np.int64) if sex is None else np.asarray(sex, np.int64)
        object.__setattr__(self, "sex", sex)
        # state defaults to ACTIVE -- an ordinary premium-paying contract.
        state = self.state
        state = (np.zeros(n_mp, np.int64) if state is None
                 else np.asarray(state, np.int64))
        object.__setattr__(self, "state", state)
        # premium_term_months defaults to the full coverage term -- the level
        # premium is collected every in-force month, the ordinary case.
        premium_term = self.premium_term_months
        premium_term = (self.term_months.copy() if premium_term is None
                        else np.asarray(premium_term, np.int64))
        object.__setattr__(self, "premium_term_months", premium_term)
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
        # Per-coverage benefit rules, CSR-aligned with cov_kind. A waiting
        # period (months with no benefit) and a reduced-benefit period (a
        # multiplier until a cut-off month) both default to off.
        n_cov = cov_amount.shape[0]
        cov_waiting = self.cov_waiting
        cov_waiting = (np.zeros(n_cov, np.int64) if cov_waiting is None
                       else np.asarray(cov_waiting, np.int64))
        cov_reduction_end = self.cov_reduction_end
        cov_reduction_end = (np.zeros(n_cov, np.int64) if cov_reduction_end is None
                             else np.asarray(cov_reduction_end, np.int64))
        cov_reduction_factor = self.cov_reduction_factor
        cov_reduction_factor = (np.ones(n_cov) if cov_reduction_factor is None
                                else np.asarray(cov_reduction_factor, np.float64))
        object.__setattr__(self, "cov_waiting", cov_waiting)
        object.__setattr__(self, "cov_reduction_end", cov_reduction_end)
        object.__setattr__(self, "cov_reduction_factor", cov_reduction_factor)

    @property
    def n_mp(self) -> int:
        """Number of model points."""
        return int(self.issue_age.shape[0])

    @property
    def effective_premium(self) -> FloatArray:
        """Monthly premium actually collected -- zero where the premium has
        stopped.

        A waiver-of-premium or paid-up contract (``state`` in
        ``_NO_PREMIUM_STATES``) keeps its coverage in force but collects no
        premium; an active contract pays the stated ``monthly_premium``. The
        projection uses this, not the raw field, so the stated premium is
        preserved for round-trip and pricing.
        """
        no_premium = np.isin(self.state, _NO_PREMIUM_STATES)
        return np.where(no_premium, 0.0, self.monthly_premium)

    @property
    def effective_single_premium(self) -> FloatArray:
        """Single premium actually collected -- zero where the premium has
        stopped (a waiver-of-premium or paid-up contract)."""
        no_premium = np.isin(self.state, _NO_PREMIUM_STATES)
        return np.where(no_premium, 0.0, self.single_premium)

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
        premium_term_months: int | None = None,
        account_value: float = 0.0,
        count: float = 1.0,
        sex: int = 0,
        state: int = STATE_ACTIVE,
        benefits: dict[int, float] | None = None,
    ) -> ModelPoints:
        """Build a single-model-point set -- a convenience for hand checks."""
        return cls(
            issue_age=np.array([issue_age]),
            death_benefit=np.array([death_benefit]),
            monthly_premium=np.array([monthly_premium]),
            term_months=np.array([term_months]),
            maturity_benefit=np.array([maturity_benefit]),
            annuity_payment=np.array([annuity_payment]),
            single_premium=np.array([single_premium]),
            premium_term_months=(None if premium_term_months is None
                                 else np.array([premium_term_months])),
            account_value=np.array([account_value]),
            count=np.array([count]),
            sex=np.array([sex]),
            state=np.array([state]),
            benefits=(
                None if benefits is None
                else {k: np.array([v]) for k, v in benefits.items()}
            ),
        )

    def to_wide(self, assumptions):
        """Convert to a wide polars DataFrame -- one row per policy.

        Every benefit becomes a column: ``death_benefit``,
        ``maturity_benefit``, ``annuity_payment`` and a
        ``<rider_code>_benefit`` column for each rate-driven rider in
        ``assumptions``. The companion to ``read_model_points``'s wide form;
        lossless only for a simple portfolio -- a wide table cannot carry
        per-coverage waiting / reduction rules.
        """
        import polars as pl

        mp_of_cov = np.repeat(np.arange(self.n_mp), np.diff(self.cov_offset))
        cols: dict[str, np.ndarray] = {
            "policy_id": np.arange(self.n_mp),
            "issue_age": self.issue_age,
            "sex": self.sex,
            "term_months": self.term_months,
            "count": self.count,
            "state": np.array([STATE_LABELS[int(s)] for s in self.state]),
            "monthly_premium": self.monthly_premium,
            "single_premium": self.single_premium,
            "premium_term_months": self.premium_term_months,
            "death_benefit": self.death_benefit,
            "maturity_benefit": self.maturity_benefit,
            "annuity_payment": self.annuity_payment,
        }
        for i, rider in enumerate(assumptions.riders):
            mask = self.cov_kind == i + 1
            cols[f"{rider.code}_benefit"] = np.bincount(
                mp_of_cov[mask], weights=self.cov_amount[mask],
                minlength=self.n_mp,
            )
        return pl.DataFrame(cols)

    def to_long(self, assumptions):
        """Convert to a long-form ``(policies, coverages)`` polars pair.

        ``policies`` is one row per policy (contract attributes);
        ``coverages`` is one row per policy x coverage, carrying
        ``rider_code`` and ``amount``. The companion to
        ``read_model_points``'s long-form input.
        """
        import polars as pl

        policies = pl.DataFrame({
            "policy_id": np.arange(self.n_mp),
            "issue_age": self.issue_age,
            "sex": self.sex,
            "term_months": self.term_months,
            "monthly_premium": self.monthly_premium,
            "single_premium": self.single_premium,
            "premium_term_months": self.premium_term_months,
            "count": self.count,
            "state": np.array([STATE_LABELS[int(s)] for s in self.state]),
        })
        # CSR coverages -- code 0 is the main-contract death, 1.. the riders.
        label = {0: _coverage_label(assumptions, TYPE_DEATH_MAIN, "death")}
        for i, rider in enumerate(assumptions.riders):
            label[i + 1] = rider.code
        mp_of_cov = np.repeat(np.arange(self.n_mp), np.diff(self.cov_offset))
        policy_id = [int(m) for m in mp_of_cov]
        rider_code = [label[int(k)] for k in self.cov_kind]
        amount = [float(a) for a in self.cov_amount]
        # Survival benefits are scalar fields -- emit them as coverage rows.
        for ctype, scalar in ((TYPE_ANNUITY, self.annuity_payment),
                              (TYPE_MATURITY, self.maturity_benefit)):
            code = _coverage_label(assumptions, ctype, ctype)
            for mp in np.nonzero(scalar)[0]:
                policy_id.append(int(mp))
                rider_code.append(code)
                amount.append(float(scalar[mp]))
        coverages = pl.DataFrame({
            "policy_id": policy_id, "rider_code": rider_code, "amount": amount,
        })
        return policies, coverages


def _coverage_label(assumptions, ctype, default):
    """The 특약코드 of the first coverage of type ``ctype`` in the
    assumptions' riders master, or ``default`` if none is registered."""
    registry = getattr(assumptions, "coverage_types", None) or {}
    for code, t in registry.items():
        if t == ctype:
            return code
    return default


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
