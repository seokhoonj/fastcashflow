"""GMM measurement assembly -- result types and the full-measurement builder.

The GMM model owns its measurement here: the result dataclasses
(:class:`Measurement`, :class:`CurrentEstimate`, :class:`Aggregate`), the
CSM orchestration (:func:`_compute_csm`), and the full-measurement assembler
(:func:`_measure_full`) that values a projection into a GMM result. The
assembler builds on the model-agnostic :func:`~fastcashflow._measurement.projection.valued_projection`
bundle and adds the GMM CSM roll, so no other model borrows a GMM container.

The shared valuation kernel (the cash-flow projection, ``valued_projection`` and
the GMM fast ``@njit`` codegen cluster) lives in :mod:`fastcashflow._measurement.gmm.engine`;
this module is imported back by ``engine`` for the result types and the
assembler, while ``valued_projection`` is imported here at call time to keep the
module load acyclic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow._measurement.basis import (
    MEASUREMENT_BASIS_INCEPTION,
    _inforce_marker_columns,
)
from fastcashflow._measurement.model import GMM
from fastcashflow.io import write_measurement, _write_measurement_columns
from fastcashflow._numerics import _csm_kernel


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, eq=False)
class Measurement:
    """IFRS 17 GMM measurement: BEL, RA and CSM.

    The headline fields (``bel``, ``ra``, ``csm``, ``loss_component``) are
    ``(n_mp,)`` inception values and are **always** present.

    The trajectory fields are the roll-forward over time and are populated
    only by ``measure(..., full=True)``; on the headline-only fast path
    (``full=False``) they are ``None``. ``bel_path`` / ``ra_path`` /
    ``csm_path`` are the ``(n_mp, n_time+1)`` trajectories whose column 0 is
    the inception value (so ``bel == bel_path[:, 0]`` when full). The CSM
    roll-forward decomposes as
    ``csm_path[:, t+1] = csm_path[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    ``lic_path`` is the liability for incurred claims -- zero unless a claims
    settlement pattern is set, which also discounts claims to their payment
    dates in the BEL.
    """

    model: ClassVar[str] = GMM

    # Headline -- always present, shape (n_mp,)
    bel: FloatArray              # inception Best Estimate of Liability
    ra: FloatArray               # inception Risk Adjustment
    csm: FloatArray              # inception Contractual Service Margin
    loss_component: FloatArray   # loss component at inception (onerous contracts)

    # Trajectory -- full=True only (None on the headline-only fast path)
    bel_path: FloatArray | None = None        # (n_mp, n_time+1) -- BEL trajectory
    ra_path: FloatArray | None = None         # (n_mp, n_time+1) -- RA trajectory
    csm_path: FloatArray | None = None        # (n_mp, n_time+1) -- CSM trajectory
    csm_accretion: FloatArray | None = None   # (n_mp, n_time)   -- CSM interest accreted
    csm_release: FloatArray | None = None     # (n_mp, n_time)   -- CSM released each month
    lic_path: FloatArray | None = None        # (n_mp, n_time+1) -- liability for incurred claims
    # The terminal column holds the residual of claims whose settlement tail
    # runs past the horizon (stays non-zero by design, not a leak).
    cashflows: "Cashflows | None" = None
    # bom = beginning of month, mom = mid of month: discount factors for a flow
    # at the start vs the middle of each month. Shape (n_time+1,) / (n_time,)
    # for a single basis; (n_mp, n_time+1) / (n_mp, n_time) when measured under
    # a per-segment basis dict, where each row discounts on its own curve.
    discount_factor_bom: FloatArray | None = None  # beginning-of-month discount factors
    discount_factor_mid: FloatArray | None = None  # mid-of-month discount factors
    # Source model points, stamped by ``measure`` so ``group(m, by=[...])`` can
    # resolve axis names without re-passing them. A reference, not a copy; None
    # on a grouped result (its rows are groups, not model points).
    model_points: "ModelPoints | None" = None
    # The per-group composite label (one per row) on a result returned by
    # ``group`` / ``group_of_contracts``; None on a per-model-point measurement.
    group_labels: "np.ndarray | None" = None
    # The number of model points in each group, aligned with ``group_labels``.
    group_sizes: IntArray | None = None
    # Time basis of the result (see _measurement.basis): 'inception' for
    # new-business measure(); in-force results re-base the headline to the
    # valuation date while the trajectories stay on the inception axis, so
    # inception-axis consumers (group / roll_forward / report / transition /
    # plot_*) reject anything else via _require_inception.
    measurement_basis: str = MEASUREMENT_BASIS_INCEPTION

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm),
                ("loss", self.loss_component)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr(f"{self.model}.Measurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str(f"{self.model}.Measurement", self._columns())

    def estimate_at(self, month: int) -> "CurrentEstimate":
        """The current estimate (BEL / RA / CSM / LIC) at a future ``month``.

        This is the deterministic nested-projection view (IFRS 17 paragraph 40): the
        cohort liability the entity would carry at month ``t`` if the central
        best-estimate scenario unfolds to it. It is column ``t`` of the
        trajectories -- so ``estimate_at(0).bel`` equals the inception headline
        ``bel`` -- and equals a fresh in-force measurement at that month,
        ``gmm.measure_inforce(elapsed=t)`` carrying the deterministic survivor
        count (reading the trajectory and re-projecting agree; the tests assert
        this for every ``t``). The returned object's ``per_survivor`` re-bases
        every figure to one surviving policy.

        Requires a ``full=True`` measurement (the trajectory paths); the fast
        path carries only the inception headline. GMM only for now -- VFA / PAA
        carry the same ``*_path`` shape and can gain this later.
        """
        if self.bel_path is None or self.cashflows is None:
            raise ValueError(
                "estimate_at requires a full=True measurement (trajectory "
                "paths); the fast path (full=False) carries only the headline."
            )
        n_time = self.bel_path.shape[1] - 1
        t = int(month)
        if not 0 <= t <= n_time:
            raise ValueError(
                f"month must be in [0, {n_time}] (the projection horizon); "
                f"got {month}."
            )
        # Survivors at the start of month t. The terminal column t == n_time has
        # no in-force column (inforce is (n_mp, n_time)), so the count there is
        # the maturity exit count -- the in-force that reached term.
        if t < n_time:
            inforce = self.cashflows.inforce[:, t]
        else:
            inforce = self.cashflows.maturity_survivors
        lic = (self.lic_path[:, t] if self.lic_path is not None
               else np.zeros_like(self.bel_path[:, t]))
        return CurrentEstimate(
            month=t,
            bel=self.bel_path[:, t],
            ra=self.ra_path[:, t],
            csm=self.csm_path[:, t],
            lic=lic,
            inforce=np.asarray(inforce, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True, eq=False)
class CurrentEstimate:
    """The GMM current estimate at one future month (IFRS 17 paragraph 40).

    Returned by :meth:`Measurement.estimate_at`. The fields are the cohort
    BEL / RA / CSM / LIC at ``month`` -- the liability the entity would carry at
    that date if the central scenario runs to it -- each shape ``(n_mp,)``.
    ``inforce`` is the deterministic survivor count at ``month``. Derived views
    are properties: ``fcf`` = BEL + RA, ``lrc`` = FCF + CSM (the GMM carrying
    amount), and ``per_survivor`` re-bases every money figure to one surviving
    policy (money / ``inforce``).
    """

    model: ClassVar[str] = GMM

    month: int
    bel: FloatArray          # (n_mp,) cohort BEL at `month`
    ra: FloatArray           # (n_mp,) cohort RA at `month`
    csm: FloatArray          # (n_mp,) cohort CSM at `month`
    lic: FloatArray          # (n_mp,) cohort LIC at `month`
    inforce: FloatArray      # (n_mp,) deterministic survivors at `month`

    @property
    def fcf(self) -> FloatArray:
        """Fulfilment cash flows = BEL + RA (IFRS 17 paragraph 32, 37)."""
        return self.bel + self.ra

    @property
    def lrc(self) -> FloatArray:
        """Liability for remaining coverage = FCF + CSM (the GMM carrying amount)."""
        return self.bel + self.ra + self.csm

    @property
    def per_survivor(self) -> "CurrentEstimate":
        """The same estimate re-based to one surviving policy (money / inforce).

        Where ``inforce`` is zero (a fully run-off cohort) the money figures are
        already zero, so the per-policy figure is a zero-safe 0.0.
        """
        inf = np.where(self.inforce > 0.0, self.inforce, 1.0)
        return CurrentEstimate(
            month=self.month,
            bel=self.bel / inf,
            ra=self.ra / inf,
            csm=self.csm / inf,
            lic=self.lic / inf,
            inforce=np.ones_like(self.inforce),
        )

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm),
                ("LIC", self.lic)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr(f"CurrentEstimate(month={self.month})",
                                self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str(f"CurrentEstimate(month={self.month})",
                               self._columns())


@dataclass(frozen=True, slots=True, eq=False)
class Aggregate:
    """Portfolio-aggregate GMM trajectories -- the scalable ``full=True`` view.

    BEL / RA / CSM are additive across contracts, so a large book's liability
    run-off is its per-model-point trajectories *summed over the model-point
    axis*. This holds only that sum: the scalar inception totals plus the
    ``(n_time+1,)`` aggregate ``bel_path`` / ``ra_path`` / ``csm_path`` (column 0
    is inception, so ``bel == bel_path[0]``). It is what
    :func:`~fastcashflow.gmm.measure_aggregate` returns, computed in bounded
    memory so it works where a per-model-point ``measure(full=True)`` would OOM.
    """

    model: ClassVar[str] = GMM

    bel: float                   # portfolio inception BEL total
    ra: float                    # portfolio inception RA total
    csm: float                   # portfolio inception CSM total
    loss_component: float        # portfolio inception loss-component total
    bel_path: FloatArray         # (n_time+1,) -- aggregate BEL trajectory
    ra_path: FloatArray          # (n_time+1,) -- aggregate RA trajectory
    csm_path: FloatArray         # (n_time+1,) -- aggregate CSM trajectory


@dataclass(frozen=True, slots=True, eq=False)
class PeriodMovement:
    """One reporting period's analysis of change.

    The period covers months ``[month_start, month_end)``. Every array is
    ``(n_mp,)``, and each block reconciles exactly::

        bel_opening + bel_assumption_change + bel_experience
            + bel_interest - bel_release == bel_closing

    and likewise for RA and CSM (with ``csm_accretion`` in place of
    ``*_interest``).

    ``*_interest`` / ``csm_accretion`` is the unwind of discount at the
    locked-in rate; ``*_release`` is the expected run-off over the period.
    ``*_assumption_change`` and ``*_experience`` are the effect of an
    assumption revision and of in-force experience -- non-zero only in the
    period the change is recognised. Both relate to future service and so
    adjust the CSM. ``loss_component_recognised`` is the part of an
    unfavourable change beyond the CSM, which falls into the loss component.
    """

    model: ClassVar[str] = GMM

    month_start: int
    month_end: int
    bel_opening: FloatArray
    bel_assumption_change: FloatArray
    bel_experience: FloatArray
    bel_interest: FloatArray
    bel_release: FloatArray
    bel_closing: FloatArray
    ra_opening: FloatArray
    ra_assumption_change: FloatArray
    ra_experience: FloatArray
    ra_interest: FloatArray
    ra_release: FloatArray
    ra_closing: FloatArray
    csm_opening: FloatArray
    csm_assumption_change: FloatArray
    csm_experience: FloatArray
    csm_accretion: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray
    loss_component_recognised: FloatArray


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """An IFRS 17 reconciliation of the insurance contract liability.

    Portfolio totals for one reporting period, in the layout of IFRS 17
    paragraph 101: the estimates of the present value of future cash flows
    (``bel``), the risk adjustment (``ra``) and the CSM each reconcile from
    opening to closing. ``*_future_service`` is the assumption and
    experience effect; ``*_finance`` is the interest unwind; ``*_release``
    is the run-off, shown negative -- so opening plus every row equals
    closing.
    """

    model: ClassVar[str] = GMM

    month_start: int
    month_end: int
    bel_opening: float
    bel_future_service: float
    bel_finance: float
    bel_release: float
    bel_closing: float
    ra_opening: float
    ra_future_service: float
    ra_finance: float
    ra_release: float
    ra_closing: float
    csm_opening: float
    csm_future_service: float
    csm_finance: float
    csm_release: float
    csm_closing: float
    loss_component_recognised: float

    def __str__(self) -> str:
        rows = (
            ("Opening", self.bel_opening, self.ra_opening, self.csm_opening),
            ("Future service", self.bel_future_service,
             self.ra_future_service, self.csm_future_service),
            ("Finance", self.bel_finance, self.ra_finance, self.csm_finance),
            ("Release", self.bel_release, self.ra_release, self.csm_release),
            ("Closing", self.bel_closing, self.ra_closing, self.csm_closing),
        )
        lines = [
            f"Reconciliation -- months {self.month_start + 1}-{self.month_end}",
            f"{'':16}{'BEL':>18}{'RA':>18}{'CSM':>18}",
        ]
        for name, bel, ra, csm in rows:
            lines.append(f"{name:16}{bel:>18,.0f}{ra:>18,.0f}{csm:>18,.0f}")
        if self.loss_component_recognised:
            lines.append(
                f"{'Loss component':16}"
                f"{self.loss_component_recognised:>18,.0f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True, eq=False)
class SettlementMovement:
    """One period's IFRS 17 paragraph-44 settlement movement of a GMM book.

    What :func:`fastcashflow.gmm.settle` returns: the opening -> closing
    movement of the BEL, RA, CSM and loss component over one reporting
    period, per model point. Every measurement array is ``(n_mp,)`` and each
    block reconciles exactly::

        bel_closing == bel_opening + bel_interest - bel_release + bel_experience
        ra_closing  == ra_opening  + ra_interest  - ra_release  + ra_experience
        csm_closing == csm_opening + csm_accretion + csm_experience_unlocking
                       + csm_premium_experience + csm_investment_experience
                       - loss_component_reversed + loss_component_recognised
                       - csm_release
        loss_component_closing == loss_component_opening
                       + loss_component_finance - loss_component_amortised
                       - loss_component_reversed + loss_component_recognised
        lic_closing == lic_opening + claims_incurred + lic_finance - claims_paid

    The GMM cross identity is THREE-term (unlike the VFA's two-term tie)::

        csm_experience_unlocking + finance_wedge
            == -(bel_experience + ra_experience)

    because B72(c) measures the paragraph-44(c) CSM adjustment at the rates
    determined on initial recognition while the BEL block is current-rate
    (B72(a)); the gap is insurance finance income/expense (B97(a)), carried
    as the named ``finance_wedge`` line OUTSIDE the CSM block. The RA part
    of the change has no rate prescription (B96(d)) and enters the CSM at
    its current measure -- a documented accounting policy.

    ``csm_premium_experience`` (B96(a)) and ``premium_experience_revenue``
    (B97(c)) are the two legs of the premium experience adjustment (actual
    premium received over the period less the expected premium), split by the
    entity's future-service fraction. The future-service leg enters the CSM
    block (it is a NEW future-service change with no BEL/RA counterpart, so it
    does NOT appear in the three-term tie above); the current/past leg is a
    P&L memo (insurance revenue), in NO balance recursion, exactly like
    ``finance_wedge``. Both are zero unless ``state.actual_premium`` is given.

    ``claims_experience`` / ``expense_experience`` (B97(b)/(c)) are the
    within-period claims / expense experience -- the actual claims / expenses
    incurred over the period less the expected -- recognised in the insurance
    service result (P&L memos, in NO balance recursion, not the CSM). Zero
    unless ``state.actual_claims`` / ``state.actual_expenses`` are given.

    ``csm_investment_experience`` (B96(c)) is the investment-component
    counterpart: the expected less the actual investment component (surrender /
    annuity repayments) that becomes payable over the period. The WHOLE
    difference enters the CSM (no fraction -- B96(c) is entirely future
    service), a new future-service change outside the three-term tie; the
    investment component does not touch insurance revenue. Zero unless
    ``state.actual_investment_component`` is given.

    ``csm_accretion`` is direct compounding of the prior CSM at the
    locked-in rate (44(b)/B72(b)); ``csm_release`` is the single period-end
    B119 release on the post-adjustment balance, with the coverage-unit
    fraction ``coverage_units_provided / (coverage_units_provided +
    coverage_units_future)`` (em_open denominator, k_exp/k_obs mixed scale).

    ``loss_component_finance`` (B97(a)/51(c)) and ``loss_component_amortised``
    (49/50(a)/51(a)+(b)) are the INCURRED-service channel of an onerous group,
    distinct from the FUTURE-service ``reversed`` / ``recognised`` lines
    (48/50(b)). As coverage is provided the period's paragraph-51 changes are
    split on the systematic loss-component ratio ``r = loss_component_opening /
    pool_opening`` (``pool_opening`` = the opening PV of remaining claims and
    expenses plus the RA): the loss component accretes ``r`` x the pool's
    interest unwind and amortises ``r`` x the pool's release. The amortised
    amount is the paragraph-49/B123(b) loss reversal -- presented in P&L and
    EXCLUDED from insurance revenue (B124(a)(i) / (b)(iii)). Both are zero on a
    profitable book (``r`` = 0) and the cumulative amortisation runs the loss
    component to zero by the end of coverage (paragraph 52), exact because
    ``r`` is re-derived every period. The future-service algebra acts on the
    POST-amortisation loss component, so ``loss_component_reversed`` is capped
    by the loss component net of this channel.

    ``lic_opening`` / ``claims_incurred`` / ``lic_finance`` / ``claims_paid`` /
    ``lic_closing`` are the liability for incurred claims (paragraphs 40(b) / 42
    / 103(b) / 37), meaningful when the basis carries a ``settlement_pattern``:
    claims build the LIC up as incurred (42(a)) and run it off over the pattern.
    The LIC is measured at fulfilment cash flows -- the discounted PV of the
    unpaid run-off plus the risk adjustment (40(b)/42(c)/37). ``claims_incurred``
    and ``claims_paid`` stay NOMINAL cash amounts (``claims_paid`` the nominal
    residual on the undiscounted trajectory); the discounting and RA move only
    the balances, and ``lic_finance`` is the reconciling residual -- the insurance
    finance (42(c) discount unwind) plus the discounting / RA measurement effect
    -- so ``lic_closing == lic_opening + claims_incurred + lic_finance -
    claims_paid``. The block is entirely expected-scale, reconstructed from the
    projection each period. The LIC RA is the confidence-level margin on the
    discounted run-off, split by risk class (a cost-of-capital LIC run-off is a
    refinement). Without a settlement pattern claims are paid as incurred, so the
    LIC is zero at both dates and ``lic_finance`` is zero.

    v1 presentation limitation: ``lic_finance`` is a single reconciling line, so
    the RA run-off / remeasurement is bundled with the 42(c) time-value movement
    rather than separated into its own insurance-service line. The balances
    (``lic_opening`` / ``lic_closing``) are the correct 40(b)/37 fulfilment cash
    flow; a fully separated P&L attribution (pure 42(c) finance vs RA release vs
    the nominal-minus-PV measurement of newly incurred claims) needs a monthly
    finance-accrual decomposition and is a future refinement.
    """

    model: ClassVar[str] = GMM

    bel_opening: FloatArray
    bel_interest: FloatArray
    bel_release: FloatArray
    bel_experience: FloatArray
    bel_closing: FloatArray
    ra_opening: FloatArray
    ra_interest: FloatArray
    ra_release: FloatArray
    ra_experience: FloatArray
    ra_closing: FloatArray
    csm_opening: FloatArray
    csm_accretion: FloatArray            # 44(b)/B72(b): locked-in, direct compounding
    csm_experience_unlocking: FloatArray  # 44(c)/B96(b)(d): locked-in measure
    csm_premium_experience: FloatArray   # B96(a): future-service premium exp, into CSM
    csm_investment_experience: FloatArray  # B96(c): investment-component exp, into CSM
    finance_wedge: FloatArray            # B97(a): current-vs-locked-in gap, not CSM
    premium_experience_revenue: FloatArray  # B97(c): current/past premium exp, P&L memo
    claims_experience: FloatArray        # B97(b)/(c): actual-vs-expected claims, P&L memo
    expense_experience: FloatArray       # B97(b)/(c): actual-vs-expected expenses, P&L memo
    csm_release: FloatArray              # 44(e)/B119: single period-end release
    csm_closing: FloatArray
    loss_component_opening: FloatArray
    loss_component_finance: FloatArray   # 51(c): r x pool interest unwind
    loss_component_amortised: FloatArray  # 50(a)/51(a)+(b): the systematic loss reversal
    loss_component_reversed: FloatArray
    loss_component_recognised: FloatArray
    loss_component_closing: FloatArray
    coverage_units_provided: FloatArray  # k_exp x (tail[em_open] - tail[em_close])
    coverage_units_future: FloatArray    # k_obs x tail[em_close]
    lic_opening: FloatArray              # 40(b)/42/37: discounted PV + RA of incurred claims
    claims_incurred: FloatArray          # 42(a)/103(b)(i): claims incurred this period (nominal)
    lic_finance: FloatArray              # 42(c): discount unwind + discounting/RA measurement
    claims_paid: FloatArray              # the settlement-pattern run-off (nominal residual)
    lic_closing: FloatArray
    period_months: int = 12
    lock_in_rate: float = 0.0
    model_points: object | None = None
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """The closing-date ``(ModelPoints, InforceState)`` pair that seeds
        the next period's settle: ``prior_csm`` / ``prior_loss_component``
        are this period's closing balances and ``prior_count`` the closing
        count. The caller advances the pair to the next observation date
        (``elapsed_months`` / ``count``) before the next call."""
        from fastcashflow.model_points import InforceState
        mp = self.model_points
        if mp is None or mp.mp_id is None:
            raise ValueError(
                "closing_inputs() needs the source model points with mp_id "
                "(the settle entry stamps them; per-MP chaining joins by id)")
        state = InforceState(
            mp_id=mp.mp_id,
            elapsed_months=np.asarray(mp.elapsed_months, dtype=np.int64),
            count=np.asarray(mp.count, dtype=np.float64),
            prior_csm=self.csm_closing,
            lock_in_rate=self.lock_in_rate,
            prior_count=np.asarray(mp.count, dtype=np.float64),
            prior_loss_component=self.loss_component_closing,
        )
        return mp, state


_GMM_RECON_BLOCKS = (
    ("BEL", (
        ("Opening", "bel_opening", "100(a)", False),
        ("Interest accreted", "bel_interest", "B72(a)", False),
        ("Release for service", "bel_release", "B123", False),
        ("Experience", "bel_experience", "B96", False),
        ("Closing", "bel_closing", "100(a)", False),
    )),
    ("RA", (
        ("Opening", "ra_opening", "101(b)", False),
        ("Interest accreted", "ra_interest", "B72(a)", False),
        ("Release for service", "ra_release", "B124", False),
        ("Experience", "ra_experience", "B96(d)", False),
        ("Closing", "ra_closing", "101(b)", False),
    )),
    ("CSM", (
        ("Opening", "csm_opening", "101(c)", False),
        ("Accretion", "csm_accretion", "44(b)/B72(b)", False),
        ("Experience unlocking", "csm_experience_unlocking", "44(c)/B96", False),
        ("Premium experience", "csm_premium_experience", "B96(a)", False),
        ("Investment experience", "csm_investment_experience", "B96(c)", False),
        ("Loss component reversed", "loss_component_reversed", "50(b)", False),
        ("Loss component recognised", "loss_component_recognised", "48", False),
        ("Release for service", "csm_release", "44(e)/B119", False),
        ("Closing", "csm_closing", "101(c)", False),
    )),
    ("Loss component", (
        ("Opening", "loss_component_opening", "49", False),
        ("Finance", "loss_component_finance", "51(c)", False),
        ("Amortised", "loss_component_amortised", "50(a)", False),
        ("Reversed", "loss_component_reversed", "50(b)", False),
        ("Recognised", "loss_component_recognised", "48", False),
        ("Closing", "loss_component_closing", "49", False),
    )),
    ("LIC", (
        ("Opening", "lic_opening", "100(c)", False),
        ("Claims incurred", "claims_incurred", "42(a)", False),
        ("Finance", "lic_finance", "42(c)", False),
        ("Claims paid", "claims_paid", "100(c)", False),
        ("Closing", "lic_closing", "100(c)", False),
    )),
    ("Memo (P&L)", (
        ("Finance wedge", "finance_wedge", "B97(a)", True),
        ("Premium experience (revenue)", "premium_experience_revenue", "B97(c)", True),
        ("Claims experience", "claims_experience", "B97(b)", True),
        ("Expense experience", "expense_experience", "B97(b)", True),
    )),
)


@dataclass(frozen=True, slots=True)
class SettlementReconciliation:
    """Portfolio totals of a :class:`SettlementMovement` -- the
    paragraph-44 settlement table. Release and loss-component-reversed rows
    are stored negative (display convention), so opening plus every row of a
    block equals its closing; ``finance_wedge`` keeps the movement sign (it
    is a P&L line outside the CSM block, not a CSM row)."""

    model: ClassVar[str] = GMM

    period_months: int
    bel_opening: float
    bel_interest: float
    bel_release: float
    bel_experience: float
    bel_closing: float
    ra_opening: float
    ra_interest: float
    ra_release: float
    ra_experience: float
    ra_closing: float
    csm_opening: float
    csm_accretion: float
    csm_experience_unlocking: float
    csm_premium_experience: float
    csm_investment_experience: float
    finance_wedge: float
    premium_experience_revenue: float
    claims_experience: float
    expense_experience: float
    loss_component_finance: float
    loss_component_amortised: float
    loss_component_reversed: float
    loss_component_recognised: float
    csm_release: float
    csm_closing: float
    loss_component_opening: float
    loss_component_closing: float
    lic_opening: float = 0.0
    claims_incurred: float = 0.0
    lic_finance: float = 0.0
    claims_paid: float = 0.0
    lic_closing: float = 0.0

    def __str__(self) -> str:
        from fastcashflow._display import _format_settlement_reconciliation
        return _format_settlement_reconciliation(
            self, "GMM settlement reconciliation", _GMM_RECON_BLOCKS)


_GMM_SETTLEMENT_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_experience_unlocking",
    "csm_premium_experience", "csm_investment_experience",
    "claims_experience", "expense_experience",
    "finance_wedge", "premium_experience_revenue",
    "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_finance",
    "loss_component_amortised", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
    "coverage_units_provided", "coverage_units_future",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
)


@dataclass(frozen=True, slots=True)
class SettlementAggregate:
    """Portfolio totals of the paragraph-44 settlement movement.

    What :func:`fastcashflow.gmm.settle_aggregate` returns: every line of
    :class:`SettlementMovement` summed over the model-point axis, in
    bounded memory. The lines keep the MOVEMENT sign -- the release and
    loss-component-reversed totals are positive run-offs, exactly like the
    per-MP movement; :func:`reconcile` applies the display negation. Each
    block therefore foots in movement form::

        bel_closing == bel_opening + bel_interest - bel_release + bel_experience

    and ``reconcile(aggregate)`` equals the per-MP movement's
    reconciliation table fieldwise.

    An aggregate is not a chaining citizen: the next period's settle needs
    per-MP prior balances, which the sums no longer carry --
    :meth:`closing_inputs` raises ValueError.
    """

    model: ClassVar[str] = GMM

    period_months: int
    lock_in_rate: float
    bel_opening: float
    bel_interest: float
    bel_release: float
    bel_experience: float
    bel_closing: float
    ra_opening: float
    ra_interest: float
    ra_release: float
    ra_experience: float
    ra_closing: float
    csm_opening: float
    csm_accretion: float
    csm_experience_unlocking: float
    csm_premium_experience: float
    csm_investment_experience: float
    finance_wedge: float
    premium_experience_revenue: float
    claims_experience: float
    expense_experience: float
    csm_release: float
    csm_closing: float
    loss_component_opening: float
    loss_component_finance: float
    loss_component_amortised: float
    loss_component_reversed: float
    loss_component_recognised: float
    loss_component_closing: float
    coverage_units_provided: float
    coverage_units_future: float
    lic_opening: float = 0.0
    claims_incurred: float = 0.0
    lic_finance: float = 0.0
    claims_paid: float = 0.0
    lic_closing: float = 0.0
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """Always raises -- see the class docstring."""
        from fastcashflow._measurement.basis import _AGGREGATE_NO_CHAIN
        raise ValueError(_AGGREGATE_NO_CHAIN)


@write_measurement.register
def _(measurement: Measurement, path, *, ids=None):
    cols = {"bel": measurement.bel, "ra": measurement.ra,
            "csm": measurement.csm,
            "loss_component": measurement.loss_component}
    # In-force output gets marker columns so it stays distinguishable from
    # new-business output at the file boundary; inception output is unchanged.
    cols.update(_inforce_marker_columns(measurement, measurement.bel.shape[0]))
    _write_measurement_columns(cols, path, ids)


# ---------------------------------------------------------------------------
# CSM and the full-measurement assembler
# ---------------------------------------------------------------------------

def _compute_csm(bel0, ra0, inforce, discount_monthly, discount_units=False):
    """CSM at initial recognition (paragraph 38) and deterministic roll-forward (paragraph 44).

    Pure-array orchestration: fulfilment cash flows ``FCF = BEL + RA``,
    initial CSM = ``max(0, -FCF)``, loss component = ``max(0, FCF)``, then
    the CSM is rolled forward in :func:`_csm_kernel` (interest accretion at
    the locked-in monthly rate, release proportional to coverage units --
    in-force here).

    ``inforce`` is ``(n_mp, n_time)`` (the coverage-unit series), ``bel0`` /
    ``ra0`` are ``(n_mp,)``. Returns
    ``(csm, accretion, release, loss_component)``.
    """
    fcf = bel0 + ra0
    csm0 = np.maximum(0.0, -fcf)
    loss_component = np.maximum(0.0, fcf)
    csm, accretion, release = _csm_kernel(csm0, inforce, discount_monthly,
                                          discount_units)
    return csm, accretion, release, loss_component


def _measure_full(model_points: "ModelPoints", basis: "Basis", *,
                  discount_monthly: FloatArray | None = None,
                  lapse_scale: FloatArray | None = None) -> Measurement:
    """Full GMM measurement: BEL, RA and CSM rolled forward over time.

    The shared neutral bundle from :func:`~fastcashflow._measurement.projection.valued_projection`
    plus the GMM CSM roll (:func:`_compute_csm`), assembled into a
    :class:`Measurement` that carries both the ``(n_mp,)`` inception headline
    (column 0 of each trajectory) and the ``(n_mp, n_time+1)`` ``*_path``
    trajectories. Reached by ``measure(..., full=True)``. ``discount_monthly`` /
    ``lapse_scale`` are forwarded to :func:`~fastcashflow._measurement.projection.valued_projection`
    (see there for the override semantics).
    """
    from fastcashflow._measurement.projection import valued_projection
    vp = valued_projection(model_points, basis,
                           discount_monthly=discount_monthly,
                           lapse_scale=lapse_scale)
    csm, csm_accretion, csm_release, loss_component = _compute_csm(
        vp.bel, vp.ra, vp.cashflows.inforce, vp.discount_monthly,
        basis.coverage_unit_discount,
    )

    return Measurement(
        bel=vp.bel,
        ra=vp.ra,
        csm=csm[:, 0],
        loss_component=loss_component,
        bel_path=vp.bel_path,
        ra_path=vp.ra_path,
        csm_path=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        lic_path=vp.lic_path,
        cashflows=vp.cashflows,
        discount_factor_bom=vp.discount_factor_bom,
        discount_factor_mid=vp.discount_factor_mid,
    )
