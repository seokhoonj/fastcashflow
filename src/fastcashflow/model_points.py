"""Model point data -- the contracts to be projected."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.coverage import CalculationMethod

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
               "PAIDUP": STATE_PAIDUP}
STATE_LABELS = {code: name for name, code in STATE_NAMES.items()}

# A VFA ``minimum_crediting_rate`` of this sentinel means the contract carries
# no crediting guarantee -- the account is credited the bare underlying return
# (which may be negative), not ``max(return, floor)``. It is distinct from a
# rate of ``0.0``, which is a real 0% floor (principal protection: the credited
# rate is ``max(return, 0)``). The sentinel is a negative outside the valid
# rate domain (a real rate < 0 would be a guaranteed loss), so it cannot
# collide with a genuine guarantee; a stray negative that is not this exact
# value is rejected as a data error rather than read as "no guarantee".
NO_GUARANTEE_RATE = -1.0


def validate_crediting_rate(rate: FloatArray) -> None:
    """Reject a ``minimum_crediting_rate`` that is neither a real rate (>= 0)
    nor the :data:`NO_GUARANTEE_RATE` sentinel.

    A stray negative (a ``-0.02`` from a sign error, say) must not be silently
    read as "no guarantee" by the credited-rate floor, so only the exact
    sentinel is permitted below zero.
    """
    rate = np.asarray(rate, dtype=np.float64)
    # A complete boundary check: the scalar TVOG helpers call this in place of
    # the ModelPoints finite check, so a NaN / inf rate must be rejected here too.
    if not np.all(np.isfinite(rate)):
        raise ValueError("minimum_crediting_rate must be finite")
    bad = (rate < 0.0) & (rate != NO_GUARANTEE_RATE)
    if np.any(bad):
        raise ValueError(
            "minimum_crediting_rate must be >= 0 (a real guarantee, 0.0 being "
            "a 0% floor) or the no-guarantee sentinel NO_GUARANTEE_RATE "
            f"({NO_GUARANTEE_RATE}); got a negative rate that is neither "
            "(a sign or data error)"
        )

# When True, ``ModelPoints.__post_init__`` skips the redundant re-validation
# that ``subset`` would otherwise re-run for every segment of a large
# portfolio. ``subset`` slices an already-validated parent, so the slice is
# valid by construction (a subset of unique mp_id is unique, a slice of finite
# amounts is finite); only ``subset`` sets this, synchronously, around the
# construction call. Single-threaded at the Python level, so a module flag is
# safe -- the kernel's parallelism is in the @njit layer, not here.
_TRUST_SLICE = False


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

    * ``coverage_index[k]``   -- the coverage code; an integer index into
      :attr:`Basis.coverages` (entry ``i`` of that tuple lives at
      code ``i``). No code is reserved.
    * ``coverage_amount[k]`` -- the benefit amount of coverage ``k``.
    * ``coverage_offset``    -- ``(n_mp+1,)``; policy ``mp``'s coverages are the
      slice ``[coverage_offset[mp] : coverage_offset[mp+1]]``.

    Each coverage may carry a benefit rule: ``coverage_waiting`` (months from
    issue with no benefit) and ``coverage_reduction_end`` / ``coverage_reduction_factor``
    (a benefit multiplier in force until a cut-off month). Both are CSR
    arrays aligned with ``coverage_index`` and default to off -- no waiting, full
    benefit.

    The coverage list is built one of two ways. ``benefits`` is the general
    form: a ``{cov_idx: amount array}`` map keyed by coverage code (the index
    into :attr:`Basis.coverages`). Or pass the CSR arrays
    ``coverage_index`` / ``coverage_amount`` / ``coverage_offset`` directly --
    the preferred form for a portfolio with per-coverage benefit rules
    (waiting / reduction periods).

    Premiums and survival benefits stay as plain fields -- they do not
    proliferate the way claim benefits do:

    * ``premium``            -- premium charged each payment occurrence.
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
    premium: FloatArray      # premium charged each payment occurrence
    term_months: IntArray          # coverage term, in months
    benefits: dict[int, FloatArray] | None = None  # general {cov_idx: amount}
    maturity_benefit: FloatArray | None = None   # benefit on survival to term
    annuity_payment: FloatArray | None = None    # survival income, each month
    disability_income: FloatArray | None = None  # income while in a benefit state
    disability_benefit: FloatArray | None = None # lump sum on a flagged transition
    # Per-policy base amount the surrender value scales against under
    # surrender_value_basis="amount_per_unit" (e.g. sum insured / basic
    # premium): surrender_cf = lapse_flow * surrender_value_curve[t] *
    # surrender_base_amount. Explicit -- no default base is inferred, since
    # the right base differs by product. None unless that mode is used.
    surrender_base_amount: FloatArray | None = None
    # IFRS 17 contract boundary (Sec. 34): the month past which cash flows
    # leave the current contract (e.g. a step-rated renewable's next renewal,
    # where the insurer can reprice). The projection stops here; the maturity
    # benefit is paid only when the boundary equals the coverage term. None
    # defaults to ``term_months`` -- no boundary cut, the historical behaviour.
    contract_boundary_months: IntArray | None = None
    premium_term_months: IntArray | None = None  # months premium is collected
    premium_frequency_months: IntArray | None = None  # months between premiums
    annuity_frequency_months: IntArray | None = None  # months between payouts
    account_value: FloatArray | None = None      # account value at issue (VFA)
    # VFA contract terms -- locked at issue, per policy. A guaranteed minimum
    # crediting rate (annual) credited to the account value when the
    # underlying-items return falls short; cohort-dependent (a 4%-guarantee
    # 2010 block vs a 1%-guarantee 2024 block can coexist in one portfolio,
    # which a single Basis value could not represent). A rate of 0.0 is a real
    # 0% floor (principal protection: credit = max(return, 0)); the default
    # NO_GUARANTEE_RATE sentinel means no crediting guarantee (the account
    # follows the bare return). Ignored by non-VFA measurements.
    minimum_crediting_rate: FloatArray | None = None
    # Guaranteed minimum death benefit (GMDB) -- the floor the death benefit
    # cannot fall below. On death the VFA pays max(account value, GMDB); the
    # excess over the account value is the guarantee's intrinsic cost. Locked
    # at issue, per policy; cohort-dependent like the credit-rate guarantee.
    # Default 0.0 = no floor (max(AV, 0) = AV); ignored by non-VFA measurements.
    minimum_death_benefit: FloatArray | None = None
    # Guaranteed minimum accumulation benefit (GMAB) -- the floor the maturity
    # benefit cannot fall below. Survivors reaching term receive max(account
    # value, GMAB); the excess over the account value is the guarantee's
    # intrinsic cost. Locked at issue, per policy. Default 0.0 = no floor
    # (max(AV, 0) = AV); ignored by non-VFA measurements.
    minimum_accumulation_benefit: FloatArray | None = None
    # Universal-life annuitization (2-phase annuity): the month -- a policy
    # duration, same clock as term_months -- at which account accumulation ends
    # and the accumulated balance converts to a guaranteed survival-contingent
    # income stream. 0 = no annuitization (the account pays a maturity lump, the
    # ordinary account behaviour). The conversion floors the balance at
    # minimum_accumulation_benefit (GMAB) and pays converted_balance *
    # annuitization_rate every annuity_frequency_months on survival; premiums
    # must cease by this month. Per policy, locked at issue.
    annuitization_months: IntArray | None = None
    # Guaranteed annuity option (GAO) rate -- the balance-to-income conversion
    # rate: locked_annuity_payment = converted_balance * annuitization_rate (a
    # per-period income per unit of accumulated balance, e.g. 0.004 = monthly
    # income of 0.4% of the balance). A financial guarantee locked at issue; NOT
    # a unit-free multiplier. 0 = none. Must be set together with
    # annuitization_months.
    annuitization_rate: FloatArray | None = None
    coverage_index: IntArray | None = None             # CSR: coverage index
    coverage_amount: FloatArray | None = None         # CSR: coverage amount
    coverage_offset: IntArray | None = None           # CSR: per-policy slice bounds
    coverage_waiting: IntArray | None = None          # CSR: waiting period, months
    coverage_reduction_end: IntArray | None = None    # CSR: reduced-benefit end, months
    coverage_reduction_factor: FloatArray | None = None  # CSR: reduced-benefit factor
    coverage_step_month: IntArray | None = None      # CSR: benefit step-up month (0 = none)
    coverage_step_factor: FloatArray | None = None   # CSR: benefit factor from step_month on
    coverage_escalation_annual: FloatArray | None = None  # CSR: annual benefit growth (0 = level)
    coverage_escalation_cap: FloatArray | None = None     # CSR: max benefit multiple (0 = unbounded)
    count: FloatArray | None = None              # policies the row stands for
    sex: IntArray | None = None                  # 0 = male, 1 = female
    state: IntArray | None = None                # contract state (STATE_*)
    # At-issue classification axis (occupational / UW class) -- one integer per
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
    # Segment metadata -- the (product, channel) keys that map a
    # model point to its assumption set when ``measure`` splits a
    # portfolio. Object arrays of string labels (or None for a
    # single-segment book). These are opaque routing keys: the engine
    # never interprets them, so a code, a name, or any custom analysis
    # group is equally valid. A human-friendly ``product_name`` /
    # ``channel_name`` column may sit alongside in the input files for
    # readability, but it is display-only -- the engine ignores it.
    product: np.ndarray | None = None
    channel: np.ndarray | None = None
    # Portfolio-level taxonomy of coverage codes -- ``{coverage:
    # CalculationMethod}``. The dict is the company catalogue (the
    # ``calculation_methods.csv`` file): every code a contract may attach is
    # registered here with its kernel-routing method (DEATH / MORBIDITY /
    # DIAGNOSIS / ANNUITY / MATURITY). The engine derives
    # ``(is_diagnosis, risk)`` from the method via
    # :func:`fastcashflow.coverage.method_attrs`; the I/O reader
    # routes coverage rows by it (annuity / maturity into scalar fields,
    # rate-driven into the CSR). ``None`` lets the engine fall back to its
    # default (every rate-driven coverage treated as a non-diagnosis
    # morbidity claim) -- fine for a hand-written one-MP test that does
    # not need the taxonomy.
    calculation_methods: dict[str, "CalculationMethod"] | None = None
    # Rate-driven coverage codes in registration order, captured at
    # construction time. The integers in ``coverage_index`` are positional
    # indices into this tuple (equivalently, into the ``Basis.coverages``
    # the model points were built against). At engine entry the tuple is
    # matched against the current ``Basis.coverages`` order; a swap or
    # an insertion would silently shift the meaning of every ``coverage_index``
    # value, so a mismatch is refused with a clear error. ``None`` skips the
    # strict check (a hand-written one-MP test that did not pin an
    # basis order); the catalogue-consistency check on
    # ``calculation_methods`` still applies.
    coverage_codes: tuple[str, ...] | None = None
    # Source grouping attributes -- carried for aggregation, never read by the
    # projection kernel. ``issue_date`` is the policy inception date
    # (date-like / numpy datetime64), the source for the annual-cohort axis
    # ``issue_year``. ``attributes`` holds any number of named per-MP label
    # columns -- portfolio_id, profitability_group, risk_class, region,
    # campaign_id, ... -- so :func:`fastcashflow.group` can aggregate on any
    # axis. ``group_of_contracts`` is the IFRS 17 preset over the same machinery.
    issue_date: np.ndarray | None = None
    attributes: dict[str, np.ndarray] | None = None
    # Contract identity -- the mp_id from the policies file, carried so
    # ``apply_inforce_state`` can join the period-close state on it instead of
    # trusting row order. A per-MP label, never read by the projection kernel;
    # ``None`` for a hand-built set. Compared as a string (the uniqueness check
    # and the in-force join both str-key it, so ``1`` and ``"1"`` are the same
    # id); use a consistently-typed id column to avoid surprise.
    mp_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Normalise the required fields to numpy arrays of the right dtype.
        for name, dtype in (
            ("issue_age", np.float64),
            ("premium", np.float64),
            ("term_months", np.int64),
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        n_mp = self.issue_age.shape[0]
        # Per-model-point arrays must all match issue_age's length (which
        # defines n_mp); a mismatch otherwise reads n_mp from one field and
        # silently ignores the rest. The rate / cash-flow inputs must also be
        # finite -- a NaN age or premium yields a NaN BEL with no error.
        for _nm in ("premium", "term_months"):
            _a = getattr(self, _nm)
            if _a.shape != (n_mp,):
                raise ValueError(
                    f"{_nm} has length {_a.size} but n_mp is {n_mp} (from "
                    f"issue_age); per-model-point arrays must match"
                )
        if not np.all(np.isfinite(self.issue_age)):
            raise ValueError("issue_age must be finite")
        if not np.all(np.isfinite(self.premium)):
            raise ValueError(
                "premium must be finite (a NaN premium yields a NaN BEL)"
            )
        # premium is a forward projection assumption (the contractual
        # premium each occurrence), not an accounting ledger entry. Accounting
        # adjustments (refunds, retrospective true-ups) are actual experience
        # and belong in roll_forward / reconcile, not the projection input, so
        # a negative here is a sign / data error.
        if np.any(self.premium < 0):
            raise ValueError(
                "premium must be >= 0 (a negative premium is a sign error; "
                "accounting adjustments belong in movement analysis, not the "
                "projection assumption)"
            )
        # Reject obviously-wrong scalar contract fields at construction time,
        # not at the bottom of a kernel where the error becomes a NaN BEL.
        if np.any(self.issue_age < 0):
            raise ValueError("issue_age must be >= 0")
        # issue_age carries through into the rate-table lookup as an int64
        # (rate grids are indexed by integer year). A fractional input is
        # silently truncated toward zero -- issue_age=40.7 looks up age 40
        # not 41. Warn so a stray .5 from a "midpoint of year" mistake or
        # a date-arithmetic bug does not slip through.
        if np.any(np.modf(self.issue_age)[0] != 0):
            warnings.warn(
                "issue_age has fractional values; the engine truncates "
                "toward zero at rate-table lookup (issue_age=40.7 -> 40). "
                "Round to whole years upstream if integer age was intended.",
                UserWarning,
                stacklevel=2,
            )
        if np.any(self.term_months < 1):
            raise ValueError("term_months must be >= 1")
        # Premiums / survival benefits default to zero (absent).
        for name in ("maturity_benefit", "annuity_payment", "disability_income",
                     "disability_benefit", "account_value",
                     "minimum_crediting_rate", "minimum_death_benefit",
                     "minimum_accumulation_benefit", "annuitization_rate"):
            value = getattr(self, name)
            # minimum_crediting_rate defaults to the no-guarantee sentinel (an
            # absent rate is "no crediting guarantee", not a 0% floor); every
            # other field defaults to zero (absent benefit / premium / amount).
            default = (np.full(n_mp, NO_GUARANTEE_RATE)
                       if name == "minimum_crediting_rate" else np.zeros(n_mp))
            value = default if value is None else np.asarray(value, np.float64)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            # Benefit / premium / account amounts are non-negative; a negative
            # one is a sign or data error that flows silently into the BEL. The
            # crediting rate is a rate (0.0 = 0% floor) and admits the
            # NO_GUARANTEE_RATE sentinel, so it has its own domain check.
            if name == "minimum_crediting_rate":
                validate_crediting_rate(value)
            elif np.any(value < 0):
                raise ValueError(f"{name} must be >= 0 (got a negative amount)")
            object.__setattr__(self, name, value)
        # count defaults to one policy per model point (seriatim).
        cnt = self.count
        cnt = np.ones(n_mp) if cnt is None else np.asarray(cnt, np.float64)
        if cnt.shape != (n_mp,):
            raise ValueError(f"count has length {cnt.size} but n_mp is {n_mp}")
        if not np.all(np.isfinite(cnt)):
            raise ValueError("count must be finite (a NaN count yields a NaN BEL)")
        if np.any(cnt < 0):
            raise ValueError("count must be >= 0")
        object.__setattr__(self, "count", cnt)
        # surrender_base_amount stays None unless provided (amount_per_unit
        # needs it; no default base is inferred). When given it is a per-MP
        # non-negative finite amount.
        sba = self.surrender_base_amount
        if sba is not None:
            sba = np.asarray(sba, np.float64)
            if sba.shape != (n_mp,):
                raise ValueError(
                    f"surrender_base_amount has length {sba.size} but n_mp "
                    f"is {n_mp}")
            if not np.all(np.isfinite(sba)) or np.any(sba < 0):
                raise ValueError(
                    "surrender_base_amount must be finite and >= 0")
            object.__setattr__(self, "surrender_base_amount", sba)
        # sex defaults to 0 (male) for every model point.
        sex = self.sex
        sex = np.zeros(n_mp, np.int64) if sex is None else np.asarray(sex, np.int64)
        if sex.shape != (n_mp,):
            raise ValueError(f"sex has length {sex.size} but n_mp is {n_mp}")
        if np.any((sex != 0) & (sex != 1)):
            raise ValueError("sex must be 0 (male) or 1 (female)")
        object.__setattr__(self, "sex", sex)
        # state defaults to ACTIVE -- an ordinary premium-paying contract.
        state = self.state
        state = (np.zeros(n_mp, np.int64) if state is None
                 else np.asarray(state, np.int64))
        if state.shape != (n_mp,):
            raise ValueError(f"state has length {state.size} but n_mp is {n_mp}")
        if np.any(state < 0):
            raise ValueError(
                "state must be >= 0 (a state index; the valid upper bound is "
                "the Basis state_model's state count, checked at measurement)")
        object.__setattr__(self, "state", state)
        # issue_class defaults to 0 for every model point -- the conventional
        # 'no class distinction' fallback. Rate tables without an issue_class
        # axis ignore this; tables with the axis look up the per-policy value.
        ic = self.issue_class
        ic = (np.zeros(n_mp, np.int64) if ic is None
              else np.asarray(ic, np.int64))
        if ic.shape != (n_mp,):
            raise ValueError(f"issue_class has length {ic.size} but n_mp is {n_mp}")
        if np.any(ic < 0):
            raise ValueError("issue_class must be >= 0")
        object.__setattr__(self, "issue_class", ic)
        # elapsed_months defaults to 0 -- every contract treated as just
        # issued (new-business mode). Non-zero values switch the model
        # point into in-force mode (see the field docstring above).
        em = self.elapsed_months
        em = (np.zeros(n_mp, np.int64) if em is None
              else np.asarray(em, np.int64))
        if em.shape != (n_mp,):
            raise ValueError(
                f"elapsed_months has length {em.size} but n_mp is {n_mp}")
        if np.any(em < 0):
            raise ValueError("elapsed_months must be >= 0")
        object.__setattr__(self, "elapsed_months", em)
        # premium_term_months defaults to the full coverage term -- the level
        # premium is collected every in-force month, the ordinary case.
        premium_term = self.premium_term_months
        premium_term = (self.term_months.copy() if premium_term is None
                        else np.asarray(premium_term, np.int64))
        if premium_term.shape != (n_mp,):
            raise ValueError(
                f"premium_term_months has length {premium_term.size} but "
                f"n_mp is {n_mp}")
        if np.any(premium_term < 0):
            raise ValueError("premium_term_months must be >= 0")
        object.__setattr__(self, "premium_term_months", premium_term)
        # contract_boundary_months defaults to the full coverage term -- no
        # Sec. 34 boundary cut (the historical behaviour). When supplied it
        # must be in [1, term]: the projection runs to the boundary and the
        # maturity benefit is withheld when the boundary is short of the term.
        boundary = self.contract_boundary_months
        boundary = (self.term_months.copy() if boundary is None
                    else np.asarray(boundary, np.int64))
        if np.any(boundary < 1):
            raise ValueError("contract_boundary_months must be >= 1")
        if np.any(boundary > self.term_months):
            raise ValueError(
                "contract_boundary_months must not exceed term_months "
                "(the boundary cannot extend past the coverage term)")
        object.__setattr__(self, "contract_boundary_months", boundary)
        # Payment frequencies -- months between successive level-premium
        # payments and annuity payouts; default 1 (monthly), must be >= 1.
        for name in ("premium_frequency_months", "annuity_frequency_months"):
            freq = getattr(self, name)
            freq = (np.ones(n_mp, np.int64) if freq is None
                    else np.asarray(freq, np.int64))
            if freq.shape != (n_mp,):
                raise ValueError(
                    f"{name} has length {freq.size} but n_mp is {n_mp}")
            if np.any(freq < 1):
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, freq)
        # Universal-life annuitization: the conversion month + GAO rate. Default
        # 0 (no annuitization -- the account pays a maturity lump). When set, the
        # month is a policy duration in [1, term]; the month and rate must be set
        # together (one alone can never produce a payment); premiums must cease
        # by the month; and the maturity lump must be 0 (it is skipped -- the
        # balance converts to the annuity, so a non-zero lump would never pay).
        # The "MP must be account-backed" and "month < contract_boundary"
        # cross-checks need the Basis / boundary and are done at engine entry.
        annz_months = self.annuitization_months
        annz_months = (np.zeros(n_mp, np.int64) if annz_months is None
                       else np.asarray(annz_months, np.int64))
        if annz_months.shape != (n_mp,):
            raise ValueError(
                f"annuitization_months has length {annz_months.size} but n_mp "
                f"is {n_mp}")
        if np.any(annz_months < 0):
            raise ValueError("annuitization_months must be >= 0 (0 = none)")
        if np.any(annz_months > self.term_months):
            raise ValueError(
                "annuitization_months must not exceed term_months")
        object.__setattr__(self, "annuitization_months", annz_months)
        # annuitization_rate was validated finite / >= 0 in the float loop above.
        annz_on = annz_months > 0
        rate_on = self.annuitization_rate > 0
        if np.any(annz_on != rate_on):
            raise ValueError(
                "annuitization_months and annuitization_rate must be set "
                "together (a month with no rate, or a rate with no month, can "
                "never produce a payment)")
        if np.any(annz_on & (self.premium_term_months > annz_months)):
            raise ValueError(
                "premium_term_months must be <= annuitization_months where "
                "annuitization is set (premiums must cease by annuitization)")
        if np.any(annz_on & (self.maturity_benefit > 0)):
            raise ValueError(
                "maturity_benefit must be 0 where annuitization is set (the "
                "maturity lump is skipped -- the balance converts to the "
                "annuity, so a non-zero lump would never pay)")
        # Coverage CSR: explicit arrays win; otherwise build from the
        # general benefits map. With no shortcut field, an empty input
        # yields an empty coverage list -- a portfolio with no rate-driven
        # claim benefits (premiums-only, or one with only survival
        # benefits via maturity_benefit / annuity_payment).
        if self.coverage_index is not None:
            coverage_index = np.asarray(self.coverage_index, np.int64)
            coverage_amount = np.asarray(self.coverage_amount, np.float64)
            coverage_offset = np.asarray(self.coverage_offset, np.int64)
        else:
            items = []   # (position, per-mp amount array), in benefits order
            benefit_codes = None   # the coverage codes the benefits map pins
            if self.benefits is not None:
                keys = list(self.benefits)
                if not all(isinstance(k, str) for k in keys):
                    raise ValueError(
                        "benefits keys must be coverage codes (str, e.g. "
                        "'DEATH'); integer index keys are not supported -- key "
                        "each benefit by its coverage code so the engine aligns "
                        "by code, not by Basis.coverages position."
                    )
                benefit_codes = tuple(keys) if keys else None
                for pos, (key, amount) in enumerate(self.benefits.items()):
                    amt = np.asarray(amount, np.float64)
                    if amt.shape != (n_mp,):
                        raise ValueError(
                            f"benefits[{key!r}] has length {amt.size} but "
                            f"n_mp is {n_mp}"
                        )
                    items.append((pos, amt))
            coverage_index, coverage_amount, coverage_offset = _build_csr(items, n_mp)
            # The benefits map is keyed by coverage code, so the engine aligns by
            # code (order-independent) rather than by Basis.coverages position.
            if benefit_codes is not None and self.coverage_codes is None:
                object.__setattr__(self, "coverage_codes", benefit_codes)
        # Validate the packed benefit amounts in one place, after both input
        # forms (the benefits map and the CSR arrays the file reader fills)
        # land here. The amount feeds straight into the kernel (claim = rate x
        # amount), so a NaN / inf silently NaNs the BEL and a negative flips the
        # claim's sign -- the file-reader path filled the CSR arrays directly
        # and so used to skip this entirely.
        if coverage_amount.size:
            if not np.all(np.isfinite(coverage_amount)):
                raise ValueError(
                    "coverage amounts must be finite (a NaN / inf amount yields "
                    "a silently-NaN BEL)"
                )
            if np.any(coverage_amount < 0):
                bad = int(np.argmax(coverage_amount < 0))
                raise ValueError(
                    "coverage amounts must be >= 0 (a negative flips the claim "
                    f"sign); coverage_amount[{bad}] = {coverage_amount[bad]}"
                )
        object.__setattr__(self, "coverage_index", coverage_index)
        object.__setattr__(self, "coverage_amount", coverage_amount)
        object.__setattr__(self, "coverage_offset", coverage_offset)
        # Per-coverage benefit rules, CSR-aligned with coverage_index. A waiting
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
        # Benefit step-up: the bidirectional partner of the reduction rule.
        # coverage_step_month is the month the benefit steps to coverage_step_factor
        # (an absolute level, 1.2 = benefit x1.2 from that month on); 0 = no step.
        coverage_step_month = self.coverage_step_month
        coverage_step_month = (np.zeros(n_cov, np.int64) if coverage_step_month is None
                          else np.asarray(coverage_step_month, np.int64))
        coverage_step_factor = self.coverage_step_factor
        coverage_step_factor = (np.ones(n_cov) if coverage_step_factor is None
                           else np.asarray(coverage_step_factor, np.float64))
        # Annual benefit escalation: the benefit grows
        # (1 + escalation_annual) ** policy_year, capped at escalation_cap x base
        # (0 = unbounded). 0 growth = level. Compounding %; a step is the
        # separate coverage_step_*.
        coverage_escalation_annual = self.coverage_escalation_annual
        coverage_escalation_annual = (np.zeros(n_cov) if coverage_escalation_annual is None
                                 else np.asarray(coverage_escalation_annual, np.float64))
        coverage_escalation_cap = self.coverage_escalation_cap
        coverage_escalation_cap = (np.zeros(n_cov) if coverage_escalation_cap is None
                              else np.asarray(coverage_escalation_cap, np.float64))
        # Each CSR-aligned rule array carries one entry per coverage. A wrong
        # length would silently drop a coverage's rule (too short) or misread
        # it (too long); a non-finite or negative month / factor would silently
        # mis-time or flip a benefit. Months and multipliers are both >= 0
        # (escalation is the growth axis; the reduction rule handles cuts).
        for name, arr in (
            ("coverage_waiting", coverage_waiting),
            ("coverage_reduction_end", coverage_reduction_end),
            ("coverage_reduction_factor", coverage_reduction_factor),
            ("coverage_step_month", coverage_step_month),
            ("coverage_step_factor", coverage_step_factor),
            ("coverage_escalation_annual", coverage_escalation_annual),
            ("coverage_escalation_cap", coverage_escalation_cap),
        ):
            if arr.shape != (n_cov,):
                raise ValueError(
                    f"{name} must align with the coverage list (one entry per "
                    f"coverage): shape ({n_cov},), got {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")
            if np.any(arr < 0):
                bad = int(np.argmax(arr < 0))
                raise ValueError(f"{name} must be >= 0; {name}[{bad}] = {arr[bad]}")
        object.__setattr__(self, "coverage_waiting", coverage_waiting)
        object.__setattr__(self, "coverage_reduction_end", coverage_reduction_end)
        object.__setattr__(self, "coverage_reduction_factor", coverage_reduction_factor)
        object.__setattr__(self, "coverage_step_month", coverage_step_month)
        object.__setattr__(self, "coverage_step_factor", coverage_step_factor)
        object.__setattr__(self, "coverage_escalation_annual", coverage_escalation_annual)
        object.__setattr__(self, "coverage_escalation_cap", coverage_escalation_cap)
        # Segment metadata + mp_id -- normalise to object arrays so they slice
        # with the per-row fields. ``None`` stays None (a single-segment book).
        for name in ("product", "channel", "mp_id"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=object)
                if value.shape != (n_mp,):
                    raise ValueError(
                        f"{name} must have shape ({n_mp},), got {value.shape}"
                    )
            object.__setattr__(self, name, value)
        # mp_id is the contract identity the in-force / settlement joins key on
        # (apply_inforce_state, group_of_contracts). A duplicate makes that
        # join ambiguous, so reject it when mp_id is supplied. (The file reader
        # already rejects duplicate policy ids; this covers a hand-built set.)
        if self.mp_id is not None and not _TRUST_SLICE:
            # str-key the ids so the check matches apply_inforce_state's string
            # join (1 and "1" are the same id) and so a mixed-type column raises
            # a clear duplicate error, not a np.unique sort TypeError. A set is
            # O(n) -- no sort -- and runs once per build (subset skips it via
            # _TRUST_SLICE).
            keys = [str(v) for v in self.mp_id.tolist()]
            if len(set(keys)) != len(keys):
                seen, dup = set(), []
                for k in keys:
                    if k in seen and k not in dup:
                        dup.append(k)
                    seen.add(k)
                raise ValueError(
                    f"ModelPoints.mp_id must be unique (it is the contract "
                    f"identity / join key); duplicates: {dup[:5]}"
                )
        # Source grouping attributes -- per-row, sliced with the segment keys,
        # untouched by the kernel. issue_date -> datetime64[D]; attributes
        # values -> object label arrays, each of length n_mp.
        if self.issue_date is not None:
            issue_date = np.asarray(self.issue_date, dtype="datetime64[D]")
            if issue_date.shape != (n_mp,):
                raise ValueError(
                    f"issue_date must have shape ({n_mp},), got {issue_date.shape}"
                )
            object.__setattr__(self, "issue_date", issue_date)
        if self.attributes is not None:
            attrs: dict[str, np.ndarray] = {}
            for k, v in self.attributes.items():
                v = np.asarray(v, dtype=object)
                if v.shape != (n_mp,):
                    raise ValueError(
                        f"attributes[{k!r}] must have shape ({n_mp},), got {v.shape}"
                    )
                attrs[str(k)] = v
            object.__setattr__(self, "attributes", attrs)
        # Benefit-pattern taxonomy -- normalise dict values to CalculationMethod
        # members so a CSV-derived ``{"CANCER": "DIAGNOSIS"}`` works the same
        # as a hand-built ``{"CANCER": CalculationMethod.DIAGNOSIS}``.
        bp = self.calculation_methods
        if bp is not None:
            bp = {str(k): CalculationMethod(v) for k, v in bp.items()}
            object.__setattr__(self, "calculation_methods", bp)
        # Registered coverage codes -- normalise to an immutable tuple of str
        # so a hand-built list or a polars Series passes through, and the
        # stored value can never drift out of sync with itself.
        cc = self.coverage_codes
        if cc is not None:
            object.__setattr__(self, "coverage_codes",
                               tuple(str(c) for c in cc))

    @property
    def n_mp(self) -> int:
        """Number of model points."""
        return int(self.issue_age.shape[0])

    def axis(self, name: str) -> np.ndarray:
        """Resolve a grouping axis to a ``(n_mp,)`` label array by name.

        Used by :func:`fastcashflow.group` to aggregate on any axis. Resolution
        order: the derived ``issue_year`` (calendar year of ``issue_date``); the
        named source fields ``product`` / ``channel`` / ``issue_date``; then
        any key in ``attributes``
        (portfolio_id, profitability_group, risk_class, ...). Raises
        :class:`KeyError` listing the available axes when the name is unknown.
        """
        if name == "issue_year":
            if self.issue_date is None:
                raise KeyError("issue_year needs issue_date, which is not set")
            return self.issue_date.astype("datetime64[Y]").astype(int) + 1970
        # Engine-native per-MP fields are axes too, and take precedence over a
        # same-named attribute. ``issue_class`` (risk class), sex, state and
        # elapsed_months default to a filled array, so they always resolve;
        # product / channel / issue_date may be None.
        _fields = ("product", "channel", "issue_date",
                   "issue_class", "sex", "state", "elapsed_months")
        if name in _fields:
            value = getattr(self, name)
            if value is None:
                raise KeyError(f"axis {name!r} is not set on these model points")
            return value
        if self.attributes is not None and name in self.attributes:
            return self.attributes[name]
        available = ["issue_year", *_fields]
        if self.attributes:
            available += list(self.attributes)
        raise KeyError(
            f"unknown grouping axis {name!r}; available: {sorted(set(available))}"
        )

    def __repr__(self) -> str:
        from fastcashflow._display import model_points_repr
        return model_points_repr(self)

    def __str__(self) -> str:
        from fastcashflow._display import model_points_str
        return model_points_str(self)

    @classmethod
    def single(
        cls,
        issue_age: float,
        premium: float,
        term_months: int,
        benefits: dict[str, float] | None = None,
        maturity_benefit: float = 0.0,
        annuity_payment: float = 0.0,
        disability_income: float = 0.0,
        disability_benefit: float = 0.0,
        premium_term_months: int | None = None,
        premium_frequency_months: int = 1,
        annuity_frequency_months: int = 1,
        account_value: float = 0.0,
        minimum_crediting_rate: float | None = None,
        minimum_death_benefit: float = 0.0,
        minimum_accumulation_benefit: float = 0.0,
        annuitization_months: int = 0,
        annuitization_rate: float = 0.0,
        count: float = 1.0,
        sex: int = 0,
        state: int = STATE_ACTIVE,
        calculation_methods: dict[str, "CalculationMethod"] | None = None,
    ) -> ModelPoints:
        """Build a single-model-point set -- a convenience for hand checks.

        ``benefits`` is the per-coverage benefit-amount map keyed by
        coverage CODE (str), e.g. ``{"DEATH": 1_000_000.0}`` -- the engine
        aligns it to ``Basis.coverages`` by code, not by position. Each code
        must also be mapped to a :class:`CalculationMethod` via
        ``calculation_methods`` (no code-as-method auto-inference). None means
        no claim benefits.
        """
        return cls(
            issue_age=np.array([issue_age]),
            premium=np.array([premium]),
            term_months=np.array([term_months]),
            maturity_benefit=np.array([maturity_benefit]),
            annuity_payment=np.array([annuity_payment]),
            disability_income=np.array([disability_income]),
            disability_benefit=np.array([disability_benefit]),
            premium_term_months=(None if premium_term_months is None
                                 else np.array([premium_term_months])),
            premium_frequency_months=np.array([premium_frequency_months]),
            annuity_frequency_months=np.array([annuity_frequency_months]),
            account_value=np.array([account_value]),
            minimum_crediting_rate=(None if minimum_crediting_rate is None
                                    else np.array([minimum_crediting_rate])),
            minimum_death_benefit=np.array([minimum_death_benefit]),
            minimum_accumulation_benefit=np.array([minimum_accumulation_benefit]),
            annuitization_months=np.array([annuitization_months]),
            annuitization_rate=np.array([annuitization_rate]),
            count=np.array([count]),
            sex=np.array([sex]),
            state=np.array([state]),
            benefits=(
                None if benefits is None
                else {k: np.array([v]) for k, v in benefits.items()}
            ),
            calculation_methods=calculation_methods,
        )

    def subset(self, indices) -> ModelPoints:
        """Return a new ``ModelPoints`` carrying the rows at ``indices``.

        Per-row fields (issue_age, premium, ...) and the segment
        metadata (product, channel) are sliced. The coverage CSR is
        rebuilt: each selected row's coverage slice
        ``coverage_index[coverage_offset[i]:coverage_offset[i+1]]`` is concatenated, and
        ``coverage_offset`` is reset to the new running cumulative sum. Used by
        :func:`fastcashflow.gmm.measure` to split a portfolio
        by (product, channel) before per-segment measurement.

        ``indices`` is expected to select **distinct** rows -- it is a row
        selection, not a gather. As an optimisation the result skips the
        re-validation the constructor runs (the parent was already validated),
        so a repeated index (``subset([0, 0])``) would carry a duplicate mp_id
        the constructor would otherwise reject. Every engine caller passes a
        unique segment index, so this is safe on the hot path; pass distinct
        indices when calling it directly.
        """
        idx = np.asarray(indices, dtype=np.int64)

        # Per-row scalar fields.
        per_row = (
            "issue_age", "premium", "term_months",
            "maturity_benefit", "annuity_payment", "disability_income",
            "disability_benefit", "premium_term_months",
            "contract_boundary_months",
            "premium_frequency_months", "annuity_frequency_months",
            "account_value", "minimum_crediting_rate", "minimum_death_benefit",
            "minimum_accumulation_benefit",
            "annuitization_months", "annuitization_rate",
            "count", "sex", "state", "issue_class", "elapsed_months",
        )
        kwargs: dict = {name: getattr(self, name)[idx] for name in per_row}

        # CSR coverage arrays -- gather each selected row's slice and rebuild
        # coverage_offset as the new cumulative count. The gather index
        # ``[start_i .. end_i)`` for every row is built vectorised (a repeat of
        # each start plus a per-row ramp) rather than a Python ``np.arange`` per
        # row, so it stays O(n_cov) with no per-model-point call overhead -- the
        # hot path when ``measure`` subsets a large portfolio per segment.
        starts = self.coverage_offset[idx]
        ends = self.coverage_offset[idx + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total > 0:
            block_start = np.repeat(np.cumsum(lengths) - lengths, lengths)
            ramp = np.arange(total, dtype=np.int64) - block_start
            cov_idx = np.repeat(starts, lengths) + ramp
        else:
            cov_idx = np.zeros(0, dtype=np.int64)
        kwargs["coverage_index"] = self.coverage_index[cov_idx]
        kwargs["coverage_amount"] = self.coverage_amount[cov_idx]
        kwargs["coverage_offset"] = np.concatenate(
            ([0], np.cumsum(ends - starts, dtype=np.int64))
        )
        kwargs["coverage_step_month"] = self.coverage_step_month[cov_idx]
        kwargs["coverage_step_factor"] = self.coverage_step_factor[cov_idx]
        kwargs["coverage_escalation_annual"] = self.coverage_escalation_annual[cov_idx]
        kwargs["coverage_escalation_cap"] = self.coverage_escalation_cap[cov_idx]
        kwargs["coverage_waiting"] = self.coverage_waiting[cov_idx]
        kwargs["coverage_reduction_end"] = self.coverage_reduction_end[cov_idx]
        kwargs["coverage_reduction_factor"] = self.coverage_reduction_factor[cov_idx]

        # Segment metadata + mp_id + optional surrender base -- slice if set;
        # otherwise stay None.
        for name in ("product", "channel", "mp_id",
                     "surrender_base_amount"):
            value = getattr(self, name)
            kwargs[name] = None if value is None else value[idx]
        # Source grouping attributes -- slice if set.
        kwargs["issue_date"] = (None if self.issue_date is None
                                else self.issue_date[idx])
        kwargs["attributes"] = (None if self.attributes is None
                                else {k: v[idx] for k, v in self.attributes.items()})
        # Taxonomy carries through unchanged -- subsetting drops rows, not
        # the company-level catalogue of coverage codes.
        kwargs["calculation_methods"] = self.calculation_methods
        # The registered coverage-code order is a property of the basis
        # the model points were built against, not of the row subset.
        kwargs["coverage_codes"] = self.coverage_codes

        # The slice is valid by construction (the parent was validated), so skip
        # the redundant re-validation -- this is the hot path when measure splits
        # a large portfolio into per-segment subsets.
        global _TRUST_SLICE
        _TRUST_SLICE = True
        try:
            return ModelPoints(**kwargs)
        finally:
            _TRUST_SLICE = False


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
      Usually scalar; per-MP cohort-aware rates are accepted for GoC-grain
      settlement, where the portfolio entry validates uniformity inside each
      group before calling the scalar GMM kernel.
    * ``account_value`` -- observed per-MP fund value at the valuation date
      (``None`` for non-VFA states). VFA subsequent measurement
      (``vfa.measure_inforce``) re-anchors the account-value path at this
      observed value; GMM / PAA ignore it. It stays on the state -- it does not
      overwrite the model point's inception ``account_value``.
    * ``prior_count`` -- in-force at month ``elapsed_months - period_months``,
      the prior reporting date (``None`` unless the state feeds
      ``vfa.settle``). Mirrors ``prior_csm``: a prior-date figure carried on
      the closing-dated state.
    * ``prior_account_value`` -- observed per-MP fund value at the prior
      reporting date (``None`` unless the state feeds ``vfa.settle``).
    * ``prior_loss_component`` -- closing loss component at the prior
      reporting date (``None`` means zero). Read by ``vfa.settle``; the
      paragraph-48/50(b) algebra reverses it on favourable changes before
      rebuilding the CSM.
    * ``profitability`` -- optional inception-frozen profitability class used
      as an explicit group-of-contracts axis at settlement.
    * ``actual_premium`` -- observed per-MP premium cash actually received over
      the reporting period (``None`` means as expected). ``gmm.settle`` splits
      the experience adjustment ``actual_premium - expected_premium`` between
      future service (CSM, Sec. B96(a)) and current/past service (P&L, Sec.
      B97(c)). May be negative (a net refund period, TRG 2018-09 Example B).
    * ``actual_investment_component`` -- observed per-MP investment component
      actually paid over the period (surrender values, annuity / maturity
      repayments -- the amounts repaid regardless of an insured event;
      ``None`` means as expected). ``gmm.settle`` routes the whole difference
      ``expected - actual`` into the CSM (Sec. B96(c)); investment components
      do not affect insurance revenue.
    * ``actual_claims`` / ``actual_expenses`` -- observed per-MP claims incurred
      / expenses incurred over the period (``None`` means as expected). The
      difference from expected is an experience adjustment relating to past /
      current service (Sec. B97(b)/(c)): it is recognised in the insurance
      service result (P&L) and does NOT adjust the CSM. Reported on
      ``gmm.settle`` as ``claims_experience`` / ``expense_experience``.
    * ``prior_lic`` -- closing liability for incurred claims at the prior
      reporting date (``None`` means reconstruct from the in-force). Required to
      settle a PAA pure-LIC-runoff period (the opening date at or past the
      contract boundary): once coverage has ended there is no in-force to scale
      the LIC by, so the carried balance seeds the run-off. ``paa.settle``'s
      ``closing_inputs()`` carries the period's ``lic_closing`` here.
    """

    mp_id: np.ndarray
    elapsed_months: IntArray
    count: FloatArray
    prior_csm: FloatArray
    lock_in_rate: float | FloatArray
    account_value: FloatArray | None = None
    prior_count: FloatArray | None = None
    prior_account_value: FloatArray | None = None
    prior_loss_component: FloatArray | None = None
    profitability: np.ndarray | None = None
    actual_premium: FloatArray | None = None
    actual_investment_component: FloatArray | None = None
    actual_claims: FloatArray | None = None
    actual_expenses: FloatArray | None = None
    prior_lic: FloatArray | None = None

    def __post_init__(self) -> None:
        # Coerce each array to its canonical dtype so a hand-built state
        # (or a reader using a different default dtype) feeds the engine
        # with the dtypes the kernels expect -- without this, an int64
        # ``count`` or a float32 ``elapsed_months`` reaches the kernel and
        # silently triggers a slow path or a numba dispatch error.
        object.__setattr__(
            self, "elapsed_months",
            np.asarray(self.elapsed_months, dtype=np.int64),
        )
        object.__setattr__(
            self, "count", np.asarray(self.count, dtype=np.float64),
        )
        object.__setattr__(
            self, "prior_csm", np.asarray(self.prior_csm, dtype=np.float64),
        )
        lock = np.asarray(self.lock_in_rate, dtype=np.float64)
        object.__setattr__(
            self, "lock_in_rate", float(lock) if lock.ndim == 0 else lock)
        # Validate: a negative elapsed month indexes backwards into the
        # trajectory (silently wrong); a NaN prior CSM / lock-in rate makes the
        # carried-forward CSM NaN with no error; a ragged array reads n from
        # one field and ignores the rest.
        n = self.elapsed_months.shape[0]
        for nm in ("mp_id", "count", "prior_csm"):
            a = np.asarray(getattr(self, nm))
            if a.shape[0] != n:
                raise ValueError(
                    f"InforceState.{nm} has length {a.shape[0]} but "
                    f"elapsed_months has {n}; per-MP arrays must match"
                )
        if np.any(self.elapsed_months < 0):
            raise ValueError("InforceState.elapsed_months must be >= 0")
        if np.any(self.count < 0):
            raise ValueError("InforceState.count must be >= 0")
        if not np.all(np.isfinite(self.prior_csm)):
            raise ValueError("InforceState.prior_csm must be finite")
        lock = np.asarray(self.lock_in_rate, dtype=np.float64)
        if lock.ndim > 0 and lock.shape[0] != n:
            raise ValueError(
                f"InforceState.lock_in_rate has length {lock.shape[0]} but "
                f"elapsed_months has {n}; use a scalar or one rate per model point")
        if not np.all(np.isfinite(lock)):
            raise ValueError("InforceState.lock_in_rate must be finite")
        # mp_id is the identity key the period-close state is joined on
        # (align_inforce_state / apply_inforce_state). A duplicate id makes
        # that join ambiguous -- the dict lookup keeps one row and silently
        # drops the other -- so reject it here. str-key the ids (matching the
        # string join) so a mixed-type column raises a clear duplicate error,
        # not a np.unique sort TypeError.
        keys = [str(v) for v in np.asarray(self.mp_id).tolist()]
        if len(set(keys)) != len(keys):
            seen, dup = set(), []
            for k in keys:
                if k in seen and k not in dup:
                    dup.append(k)
                seen.add(k)
            raise ValueError(
                f"InforceState.mp_id must be unique (it is the join key); "
                f"duplicates: {dup[:5]}"
            )
        # The optional fields (VFA only). When given, each is a per-MP array:
        # coerce, match length, and validate finite / non-negative -- a NaN
        # account value reseeds a NaN account-value path; a negative prior
        # count or loss component is meaningless.
        for nm in ("account_value", "prior_count", "prior_account_value",
                   "prior_loss_component", "prior_lic"):
            value = getattr(self, nm)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape[0] != n:
                raise ValueError(
                    f"InforceState.{nm} has length {arr.shape[0]} but "
                    f"elapsed_months has {n}; per-MP arrays must match"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"InforceState.{nm} must be finite")
            if np.any(arr < 0):
                raise ValueError(f"InforceState.{nm} must be >= 0")
            object.__setattr__(self, nm, arr)
        if self.profitability is not None:
            prof = np.asarray(self.profitability, dtype=object)
            if prof.shape[0] != n:
                raise ValueError(
                    f"InforceState.profitability has length {prof.shape[0]} but "
                    f"elapsed_months has {n}; per-MP arrays must match")
            object.__setattr__(self, "profitability", prof)
        # The settlement within-period experience inputs are finite but may be
        # negative (a net premium refund; a favourable claims / expense / IC
        # experience), so they are validated apart from the >= 0 group -- coerce
        # dtype, match length, and require finiteness (a NaN here silently
        # poisons the experience line it feeds).
        for nm in ("actual_premium", "actual_investment_component",
                   "actual_claims", "actual_expenses"):
            value = getattr(self, nm)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape[0] != n:
                raise ValueError(
                    f"InforceState.{nm} has length {arr.shape[0]} but "
                    f"elapsed_months has {n}; per-MP arrays must match")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"InforceState.{nm} must be finite")
            object.__setattr__(self, nm, arr)

    def subset(self, indices) -> "InforceState":
        """Return a new ``InforceState`` carrying the rows at ``indices``.

        The per-MP fields (``mp_id``, ``elapsed_months``, ``count``,
        ``prior_csm``, and the optional ``account_value`` / ``prior_*`` fields
        when present) are sliced together and the scalar ``lock_in_rate`` is
        carried, so the result stays internally consistent. Use it alongside
        :meth:`ModelPoints.subset` to split a period-close state by segment
        before a per-segment :func:`fastcashflow.gmm.measure_inforce` (slicing
        only ``prior_csm`` would leave the state ragged).
        """
        idx = np.asarray(indices, dtype=np.int64)

        def _opt(value):
            return None if value is None else value[idx]

        lock = np.asarray(self.lock_in_rate, dtype=np.float64)

        return InforceState(
            mp_id=np.asarray(self.mp_id)[idx],
            elapsed_months=self.elapsed_months[idx],
            count=self.count[idx],
            prior_csm=self.prior_csm[idx],
            lock_in_rate=self.lock_in_rate if lock.ndim == 0 else lock[idx],
            account_value=_opt(self.account_value),
            prior_count=_opt(self.prior_count),
            prior_account_value=_opt(self.prior_account_value),
            prior_loss_component=_opt(self.prior_loss_component),
            profitability=_opt(self.profitability),
            actual_premium=_opt(self.actual_premium),
            actual_investment_component=_opt(self.actual_investment_component),
            actual_claims=_opt(self.actual_claims),
            actual_expenses=_opt(self.actual_expenses),
            prior_lic=_opt(self.prior_lic),
        )


def align_inforce_state(
    model_points: "ModelPoints", state: InforceState,
) -> InforceState:
    """Return ``state`` reordered so its rows line up with ``model_points``.

    Every per-MP field of the returned state (``elapsed_months``, ``count``,
    ``prior_csm``, ``mp_id``) is row-for-row aligned with the model points.
    When both carry ``mp_id`` the match is **by mp_id** -- reordered when the
    two files are in different orders, and rejected when their id sets differ
    -- so a misaligned period-close file cannot silently assign one contract's
    state (including its prior CSM) to another. When the model points have no
    ``mp_id`` (a hand-built set), the rows are taken positionally after a
    length check; align them yourself in that case.
    """
    n_mp = int(model_points.issue_age.shape[0])
    if state.elapsed_months.shape[0] != n_mp:
        raise ValueError(
            f"state has {state.elapsed_months.shape[0]} rows; the "
            f"model points have {n_mp}. The state must cover exactly the "
            "valued contracts."
        )
    mp_ids = model_points.mp_id
    if mp_ids is not None:
        mp_ids = np.asarray(mp_ids).astype(str)
        st_ids = np.asarray(state.mp_id).astype(str)
        if set(mp_ids) != set(st_ids):
            missing = sorted(set(mp_ids) - set(st_ids))[:5]
            extra = sorted(set(st_ids) - set(mp_ids))[:5]
            raise ValueError(
                "align_inforce_state: model points and state carry different "
                f"mp_id sets (in model points only: {missing}; in state only: "
                f"{extra}). The state must cover exactly the valued contracts."
            )
        if not np.array_equal(mp_ids, st_ids):       # different order -> join
            pos = {mid: i for i, mid in enumerate(st_ids)}
            state = state.subset(np.array([pos[mid] for mid in mp_ids]))
    return state


def apply_inforce_state(
    model_points: "ModelPoints", state: InforceState,
) -> "ModelPoints":
    """Return a ``ModelPoints`` with the state's ``elapsed_months`` and
    ``count`` substituted in, joined on ``mp_id`` (see
    :func:`align_inforce_state` for the join rules).

    Note this substitutes only ``elapsed_months`` / ``count`` onto the model
    points; the state's ``prior_csm`` rides on the (separately passed)
    :class:`InforceState`. :func:`~fastcashflow.gmm.measure_inforce` re-aligns
    that state by mp_id internally, so prior_csm cannot drift out of order.
    """
    from dataclasses import replace
    state = align_inforce_state(model_points, state)
    return replace(
        model_points,
        elapsed_months=np.asarray(state.elapsed_months, dtype=np.int64),
        count=np.asarray(state.count, dtype=np.float64),
    )


def _build_csr(
    items: list[tuple[int, FloatArray]], n_mp: int
) -> tuple[IntArray, FloatArray, IntArray]:
    """Pack ``(cov_idx, per-mp amount)`` items into a coverage CSR.

    A zero amount is no coverage. Coverages are ordered by model point, and
    within a model point by the order the cov_idx values appear in ``items``.
    An empty ``items`` list yields an empty coverage list -- no claim
    coverages on any policy.
    """
    if not items:
        return (
            np.zeros(0, np.int64),
            np.zeros(0, np.float64),
            np.zeros(n_mp + 1, np.int64),
        )
    mp_parts, cov_idx_parts, amount_parts = [], [], []
    for cov_idx, amount in items:
        present = amount != 0.0
        mp_idx = np.nonzero(present)[0]
        mp_parts.append(mp_idx)
        cov_idx_parts.append(np.full(mp_idx.size, cov_idx, np.int64))
        amount_parts.append(amount[present])
    all_mp = np.concatenate(mp_parts)
    all_cov_idx = np.concatenate(cov_idx_parts)
    all_amount = np.concatenate(amount_parts)
    order = np.argsort(all_mp, kind="stable")     # group by mp, keep cov_idx order
    coverage_index = np.ascontiguousarray(all_cov_idx[order])
    coverage_amount = np.ascontiguousarray(all_amount[order])
    coverage_offset = np.concatenate((
        np.zeros(1, np.int64),
        np.cumsum(np.bincount(all_mp, minlength=n_mp), dtype=np.int64),
    ))
    return coverage_index, coverage_amount, coverage_offset
