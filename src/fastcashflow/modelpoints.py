"""Model point data -- the contracts to be projected."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.coverage import (
    DEATH, TYPE_ANNUITY, TYPE_DEATH_MAIN, TYPE_MATURITY,
)

# Contract states -- a model point's in-force state at the valuation date.
# ACTIVE is the ordinary premium-paying contract. WAIVER (premium waived on a
# triggering event) and PAIDUP (the premium-paying term has ended) both keep
# the coverage in force while collecting no premium. The state places the
# model point's starting in-force on the active or the waiver track; during
# the projection active in-force can itself transition to waiver at the
# waiver-inception rate (IFRS 17 Sec. 33-34 -- the fulfilment cash flows
# reflect the contract's actual terms at the measurement date).
STATE_ACTIVE = 0
STATE_WAIVER = 1
STATE_PAIDUP = 2

# Names for the file layer -- a model-point ``state`` column reads and writes
# these strings, the readable form a practitioner edits in a spreadsheet.
STATE_NAMES = {"ACTIVE": STATE_ACTIVE, "WAIVER": STATE_WAIVER,
               "PAID_UP": STATE_PAIDUP}
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

    * ``coverage_kind[k]``   -- the coverage code (0 = main-contract death, 1..
      the rate-driven riders the assumptions register).
    * ``coverage_amount[k]`` -- the benefit amount of coverage ``k``.
    * ``coverage_offset``    -- ``(n_mp+1,)``; policy ``mp``'s coverages are the
      slice ``[coverage_offset[mp] : coverage_offset[mp+1]]``.

    Each coverage may carry a benefit rule: ``coverage_waiting`` (months from
    issue with no benefit) and ``coverage_reduction_end`` / ``coverage_reduction_factor``
    (a benefit multiplier in force until a cut-off month). Both are CSR
    arrays aligned with ``coverage_kind`` and default to off -- no waiting, full
    benefit.

    The coverage list is built one of three ways. ``death_benefit`` is the
    shortcut for the common case -- one death coverage per policy with a
    non-zero benefit. ``benefits`` is the general form: a ``{kind: amount
    array}`` map covering any mix of kinds. Or pass the CSR arrays
    ``coverage_kind`` / ``coverage_amount`` / ``coverage_offset`` directly. Whichever is
    used, ``death_benefit`` stays a readable field.

    Premiums and survival benefits stay as plain fields -- they do not
    proliferate the way claim benefits do:

    * ``level_premium``            -- premium charged each payment occurrence.
    * ``single_premium``           -- one-off premium at t = 0.
    * ``premium_term_months``      -- months the level premium is collected,
      defaulting to the full coverage term.
    * ``premium_frequency_months`` -- months between level-premium payments
      (1 monthly, 3 quarterly, 6 half-yearly, 12 annual), defaulting to 1.
    * ``maturity_benefit``         -- benefit on survival to the end of the term.
    * ``annuity_payment``          -- survival income paid each payout occurrence.
    * ``annuity_frequency_months`` -- months between annuity payouts,
      defaulting to 1.
    * ``disability_income``        -- income paid each month a benefit state
      is occupied (disability income on a disabled state).
    * ``disability_benefit``       -- lump sum paid when a lump-sum transition
      fires (a disability lump sum on becoming disabled).
    """

    issue_age: FloatArray          # attained age at issue, in years
    level_premium: FloatArray      # premium charged each payment occurrence
    term_months: IntArray          # coverage term, in months
    death_benefit: FloatArray | None = None      # shortcut -> a death coverage
    benefits: dict[int, FloatArray] | None = None  # general {kind: amount}
    maturity_benefit: FloatArray | None = None   # benefit on survival to term
    annuity_payment: FloatArray | None = None    # survival income, each month
    disability_income: FloatArray | None = None  # income while in a benefit state
    disability_benefit: FloatArray | None = None # lump sum on a flagged transition
    single_premium: FloatArray | None = None     # one-off premium at t = 0
    premium_term_months: IntArray | None = None  # months premium is collected
    premium_frequency_months: IntArray | None = None  # months between premiums
    annuity_frequency_months: IntArray | None = None  # months between payouts
    account_value: FloatArray | None = None      # account value at issue (VFA)
    # VFA contract terms -- locked at issue, per policy. A guaranteed minimum
    # crediting rate (annual) credited to the account value when the
    # underlying-items return falls short; cohort-dependent (a 4%-guarantee
    # 2010 block vs a 1%-guarantee 2024 block can coexist in one portfolio,
    # which a single Assumptions value could not represent). Default 0.0 = no
    # guarantee; ignored by non-VFA measurements.
    guaranteed_credit_rate: FloatArray | None = None
    coverage_kind: IntArray | None = None             # CSR: coverage kind
    coverage_amount: FloatArray | None = None         # CSR: coverage amount
    coverage_offset: IntArray | None = None           # CSR: per-policy slice bounds
    coverage_waiting: IntArray | None = None          # CSR: waiting period, months
    coverage_reduction_end: IntArray | None = None    # CSR: reduced-benefit end, months
    coverage_reduction_factor: FloatArray | None = None  # CSR: reduced-benefit factor
    count: FloatArray | None = None              # policies the row stands for
    sex: IntArray | None = None                  # 0 = male, 1 = female
    state: IntArray | None = None                # contract state (STATE_*)
    # At-issue classification axis (직업class / UW class) -- one integer per
    # model point, default 0 for every policy. Rate tables that key on
    # ``issue_class`` look up the per-policy value; tables without the axis
    # broadcast over it (no effect).
    issue_class: IntArray | None = None
    # In-force valuation -- months since policy inception at the valuation
    # date. Default 0 reproduces the new-business behaviour (every contract
    # treated as just issued). Set per-MP for an in-force portfolio: each
    # contract has its own inception, so at a single valuation date the
    # array carries different elapsed values across rows. Rate lookups, the
    # premium-paying-window check and surrender's cumulative-premium basis
    # all shift by ``elapsed_months[mp]``.
    elapsed_months: IntArray | None = None
    # Segment metadata -- the (product, channel) keys that map a model point
    # to its assumption set when ``value_segmented`` splits a portfolio.
    # Object arrays of string labels (or None for a single-segment book).
    product: np.ndarray | None = None
    channel: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Normalise the required fields to numpy arrays of the right dtype.
        for name, dtype in (
            ("issue_age", np.float64),
            ("level_premium", np.float64),
            ("term_months", np.int64),
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        n_mp = self.issue_age.shape[0]
        # Premiums / survival benefits default to zero (absent).
        for name in ("maturity_benefit", "annuity_payment", "disability_income",
                     "disability_benefit", "single_premium", "account_value",
                     "guaranteed_credit_rate"):
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
        # issue_class defaults to 0 for every model point -- the conventional
        # 'no class distinction' fallback. Rate tables without an issue_class
        # axis ignore this; tables with the axis look up the per-policy value.
        ic = self.issue_class
        ic = (np.zeros(n_mp, np.int64) if ic is None
              else np.asarray(ic, np.int64))
        object.__setattr__(self, "issue_class", ic)
        # elapsed_months defaults to 0 -- every contract treated as just
        # issued (new-business mode). Non-zero values switch the model
        # point into in-force mode (see the field docstring above).
        em = self.elapsed_months
        em = (np.zeros(n_mp, np.int64) if em is None
              else np.asarray(em, np.int64))
        object.__setattr__(self, "elapsed_months", em)
        # premium_term_months defaults to the full coverage term -- the level
        # premium is collected every in-force month, the ordinary case.
        premium_term = self.premium_term_months
        premium_term = (self.term_months.copy() if premium_term is None
                        else np.asarray(premium_term, np.int64))
        object.__setattr__(self, "premium_term_months", premium_term)
        # Payment frequencies -- months between successive level-premium
        # payments and annuity payouts; default 1 (monthly), must be >= 1.
        for name in ("premium_frequency_months", "annuity_frequency_months"):
            freq = getattr(self, name)
            freq = (np.ones(n_mp, np.int64) if freq is None
                    else np.asarray(freq, np.int64))
            if np.any(freq < 1):
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, freq)
        # Coverage CSR: explicit arrays win; otherwise build from the
        # death_benefit shortcut and/or the general benefits map.
        if self.coverage_kind is not None:
            coverage_kind = np.asarray(self.coverage_kind, np.int64)
            coverage_amount = np.asarray(self.coverage_amount, np.float64)
            coverage_offset = np.asarray(self.coverage_offset, np.int64)
        else:
            items = []   # (kind, per-mp amount array), in coverage-list order
            db = self.death_benefit
            db = np.zeros(n_mp) if db is None else np.asarray(db, np.float64)
            items.append((DEATH, db))
            if self.benefits is not None:
                for kind, amount in self.benefits.items():
                    items.append((int(kind), np.asarray(amount, np.float64)))
            coverage_kind, coverage_amount, coverage_offset = _build_csr(items, n_mp)
        # death_benefit stays a readable field, reconstructed from the CSR.
        mp_of_cov = np.repeat(np.arange(n_mp), np.diff(coverage_offset))
        is_death = coverage_kind == DEATH
        object.__setattr__(self, "death_benefit", np.bincount(
            mp_of_cov[is_death], weights=coverage_amount[is_death], minlength=n_mp
        ))
        object.__setattr__(self, "coverage_kind", coverage_kind)
        object.__setattr__(self, "coverage_amount", coverage_amount)
        object.__setattr__(self, "coverage_offset", coverage_offset)
        # Per-coverage benefit rules, CSR-aligned with coverage_kind. A waiting
        # period (months with no benefit) and a reduced-benefit period (a
        # multiplier until a cut-off month) both default to off.
        n_cov = coverage_amount.shape[0]
        coverage_waiting = self.coverage_waiting
        coverage_waiting = (np.zeros(n_cov, np.int64) if coverage_waiting is None
                       else np.asarray(coverage_waiting, np.int64))
        coverage_reduction_end = self.coverage_reduction_end
        coverage_reduction_end = (np.zeros(n_cov, np.int64) if coverage_reduction_end is None
                             else np.asarray(coverage_reduction_end, np.int64))
        coverage_reduction_factor = self.coverage_reduction_factor
        coverage_reduction_factor = (np.ones(n_cov) if coverage_reduction_factor is None
                                else np.asarray(coverage_reduction_factor, np.float64))
        object.__setattr__(self, "coverage_waiting", coverage_waiting)
        object.__setattr__(self, "coverage_reduction_end", coverage_reduction_end)
        object.__setattr__(self, "coverage_reduction_factor", coverage_reduction_factor)
        # Segment metadata -- normalise to object arrays so they slice with
        # the per-row fields. ``None`` stays None (a single-segment book).
        for name in ("product", "channel"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=object)
                if value.shape != (n_mp,):
                    raise ValueError(
                        f"{name} must have shape ({n_mp},), got {value.shape}"
                    )
            object.__setattr__(self, name, value)

    @property
    def n_mp(self) -> int:
        """Number of model points."""
        return int(self.issue_age.shape[0])

    @classmethod
    def single(
        cls,
        issue_age: float,
        death_benefit: float,
        level_premium: float,
        term_months: int,
        maturity_benefit: float = 0.0,
        annuity_payment: float = 0.0,
        disability_income: float = 0.0,
        disability_benefit: float = 0.0,
        single_premium: float = 0.0,
        premium_term_months: int | None = None,
        premium_frequency_months: int = 1,
        annuity_frequency_months: int = 1,
        account_value: float = 0.0,
        guaranteed_credit_rate: float = 0.0,
        count: float = 1.0,
        sex: int = 0,
        state: int = STATE_ACTIVE,
        benefits: dict[int, float] | None = None,
    ) -> ModelPoints:
        """Build a single-model-point set -- a convenience for hand checks."""
        return cls(
            issue_age=np.array([issue_age]),
            death_benefit=np.array([death_benefit]),
            level_premium=np.array([level_premium]),
            term_months=np.array([term_months]),
            maturity_benefit=np.array([maturity_benefit]),
            annuity_payment=np.array([annuity_payment]),
            disability_income=np.array([disability_income]),
            disability_benefit=np.array([disability_benefit]),
            single_premium=np.array([single_premium]),
            premium_term_months=(None if premium_term_months is None
                                 else np.array([premium_term_months])),
            premium_frequency_months=np.array([premium_frequency_months]),
            annuity_frequency_months=np.array([annuity_frequency_months]),
            account_value=np.array([account_value]),
            guaranteed_credit_rate=np.array([guaranteed_credit_rate]),
            count=np.array([count]),
            sex=np.array([sex]),
            state=np.array([state]),
            benefits=(
                None if benefits is None
                else {k: np.array([v]) for k, v in benefits.items()}
            ),
        )

    def subset(self, indices) -> ModelPoints:
        """Return a new ``ModelPoints`` carrying the rows at ``indices``.

        Per-row fields (issue_age, level_premium, ...) and the segment
        metadata (product, channel) are sliced. The coverage CSR is
        rebuilt: each selected row's coverage slice
        ``coverage_kind[coverage_offset[i]:coverage_offset[i+1]]`` is concatenated, and
        ``coverage_offset`` is reset to the new running cumulative sum. Used by
        :func:`fastcashflow.engine.value_segmented` to split a portfolio
        by (product, channel) before per-segment valuation.
        """
        idx = np.asarray(indices, dtype=np.int64)

        # Per-row scalar fields.
        per_row = (
            "issue_age", "level_premium", "term_months", "death_benefit",
            "maturity_benefit", "annuity_payment", "disability_income",
            "disability_benefit", "single_premium", "premium_term_months",
            "premium_frequency_months", "annuity_frequency_months",
            "account_value", "guaranteed_credit_rate",
            "count", "sex", "state", "issue_class", "elapsed_months",
        )
        kwargs: dict = {name: getattr(self, name)[idx] for name in per_row}

        # CSR coverage arrays -- concatenate each selected row's slice and
        # rebuild coverage_offset as the new cumulative count.
        starts = self.coverage_offset[idx]
        ends = self.coverage_offset[idx + 1]
        cov_idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) \
            if idx.size > 0 else np.zeros(0, dtype=np.int64)
        kwargs["coverage_kind"] = self.coverage_kind[cov_idx]
        kwargs["coverage_amount"] = self.coverage_amount[cov_idx]
        kwargs["coverage_offset"] = np.concatenate(
            ([0], np.cumsum(ends - starts, dtype=np.int64))
        )
        kwargs["coverage_waiting"] = self.coverage_waiting[cov_idx]
        kwargs["coverage_reduction_end"] = self.coverage_reduction_end[cov_idx]
        kwargs["coverage_reduction_factor"] = self.coverage_reduction_factor[cov_idx]

        # Segment metadata -- slice if set; otherwise stay None.
        for name in ("product", "channel"):
            value = getattr(self, name)
            kwargs[name] = None if value is None else value[idx]

        return ModelPoints(**kwargs)

    def to_wide(self, assumptions):
        """Convert to a wide polars DataFrame -- one row per model point.

        Every benefit becomes a column: ``death_benefit``,
        ``maturity_benefit``, ``annuity_payment`` and a
        ``<coverage_code>_benefit`` column for each rate-driven coverage in
        ``assumptions``. The companion to ``read_model_points``'s wide form;
        lossless only for a simple portfolio -- a wide table cannot carry
        per-coverage waiting / reduction rules.
        """
        import polars as pl

        mp_of_cov = np.repeat(np.arange(self.n_mp), np.diff(self.coverage_offset))
        cols: dict[str, np.ndarray] = {
            "mp_id": np.arange(self.n_mp),
            "issue_age": self.issue_age,
            "sex": self.sex,
            "term_months": self.term_months,
            "count": self.count,
            "state": np.array([STATE_LABELS[int(s)] for s in self.state]),
            "level_premium": self.level_premium,
            "single_premium": self.single_premium,
            "premium_term_months": self.premium_term_months,
            "premium_frequency_months": self.premium_frequency_months,
            "annuity_frequency_months": self.annuity_frequency_months,
            "death_benefit": self.death_benefit,
            "maturity_benefit": self.maturity_benefit,
            "annuity_payment": self.annuity_payment,
            "disability_income": self.disability_income,
            "disability_benefit": self.disability_benefit,
        }
        for i, rider in enumerate(assumptions.coverages):
            mask = self.coverage_kind == i + 1
            cols[f"{rider.code}_benefit"] = np.bincount(
                mp_of_cov[mask], weights=self.coverage_amount[mask],
                minlength=self.n_mp,
            )
        return pl.DataFrame(cols)

    def to_long(self, assumptions):
        """Convert to a long-form ``(policies, coverages)`` polars pair.

        ``policies`` is one row per model point (contract attributes);
        ``coverages`` is one row per model point x coverage, carrying
        ``coverage_code`` and ``amount``. The companion to
        ``read_model_points``'s long-form input.
        """
        import polars as pl

        policies = pl.DataFrame({
            "mp_id":                    np.arange(self.n_mp),
            "issue_age":                self.issue_age,
            "sex":                      self.sex,
            "term_months":              self.term_months,
            "level_premium":            self.level_premium,
            "single_premium":           self.single_premium,
            "premium_term_months":      self.premium_term_months,
            "premium_frequency_months": self.premium_frequency_months,
            "annuity_frequency_months": self.annuity_frequency_months,
            "disability_income":        self.disability_income,
            "disability_benefit":       self.disability_benefit,
            "count":                    self.count,
            "state":                    np.array([STATE_LABELS[int(s)] for s in self.state]),
        })
        # CSR coverages -- code 0 is the main-contract death, 1.. the riders.
        label = {0: _coverage_label(assumptions, TYPE_DEATH_MAIN, "death")}
        for i, rider in enumerate(assumptions.coverages):
            label[i + 1] = rider.code
        mp_of_cov = np.repeat(np.arange(self.n_mp), np.diff(self.coverage_offset))
        mp_id = [int(m) for m in mp_of_cov]
        coverage_code = [label[int(k)] for k in self.coverage_kind]
        amount = [float(a) for a in self.coverage_amount]
        # Survival benefits are scalar fields -- emit them as coverage rows.
        for ctype, scalar in ((TYPE_ANNUITY, self.annuity_payment),
                              (TYPE_MATURITY, self.maturity_benefit)):
            code = _coverage_label(assumptions, ctype, ctype)
            for mp in np.nonzero(scalar)[0]:
                mp_id.append(int(mp))
                coverage_code.append(code)
                amount.append(float(scalar[mp]))
        coverages = pl.DataFrame({
            "mp_id": mp_id, "coverage_code": coverage_code, "amount": amount,
        })
        return policies, coverages


@dataclass(frozen=True, slots=True)
class InforceState:
    """Per-MP closing state from the prior reporting period.

    The input layer for in-force / subsequent-measurement workflows. A
    fresh ``inforce_state.csv`` is produced at each period close from the
    company's policy administration system and joined onto the static
    ``policies.csv`` to value the in-force at the next reporting date.

    Fields:

    * ``mp_id`` -- join key, matches the ``mp_id`` column on the policies
      file.
    * ``elapsed_months`` -- months since each contract's inception as of
      the valuation date (= valuation date - inception date).
    * ``count`` -- in-force at the valuation date (the user has already
      scaled it down for past lapses); seats the projection.
    * ``prior_csm`` -- closing CSM at month
      ``elapsed_months - period_months``, the prior reporting date's
      result carried into this period.
    * ``lock_in_rate`` -- annual locked-in discount rate (Sec. B72(b)).
      Scalar in v1; per-MP cohort-aware rates are a future extension.
    """

    mp_id: np.ndarray
    elapsed_months: IntArray
    count: FloatArray
    prior_csm: FloatArray
    lock_in_rate: float


def apply_inforce_state(
    model_points: "ModelPoints", state: InforceState,
) -> "ModelPoints":
    """Return a ``ModelPoints`` with the state's ``elapsed_months`` and
    ``count`` substituted in.

    The two inputs must already be aligned: row ``i`` of the model points
    is the contract whose state is row ``i`` of ``state``. The expected
    workflow is to sort both files by ``mp_id`` upstream; this helper
    enforces only the length check (mp_id alignment is the user's
    responsibility because a generic mp_id-keyed join would force a
    polars / numpy reorganisation of every per-MP array on the
    ``ModelPoints``).
    """
    from dataclasses import replace
    n_mp = int(model_points.issue_age.shape[0])
    if state.elapsed_months.shape[0] != n_mp:
        raise ValueError(
            f"state has {state.elapsed_months.shape[0]} rows; the "
            f"model points have {n_mp}. Align the two files (sort both "
            "by mp_id) before applying."
        )
    return replace(
        model_points,
        elapsed_months=np.asarray(state.elapsed_months, dtype=np.int64),
        count=np.asarray(state.count, dtype=np.float64),
    )


def _coverage_label(assumptions, ctype, default):
    """The rider code of the first coverage of type ``ctype`` in the
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
    coverage_kind = np.ascontiguousarray(all_kind[order])
    coverage_amount = np.ascontiguousarray(all_amount[order])
    coverage_offset = np.concatenate((
        np.zeros(1, np.int64),
        np.cumsum(np.bincount(all_mp, minlength=n_mp), dtype=np.int64),
    ))
    return coverage_kind, coverage_amount, coverage_offset
