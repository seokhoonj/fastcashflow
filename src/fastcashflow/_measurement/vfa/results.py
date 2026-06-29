"""IFRS 17 Variable Fee Approach (VFA) -- result types.

The VFA model owns its measurement result types here: the headline/trajectory
:class:`Measurement`, the portfolio :class:`Aggregate`, the roll-forward
:class:`PeriodMovement` / :class:`Reconciliation`, the paragraph-45 settlement
:class:`SettlementMovement` / :class:`SettlementReconciliation` /
:class:`SettlementAggregate`, the group-of-contracts :class:`GoCSettlement` and
the guarantee :class:`TVOG`, with the ``CSM_BASIS_*`` vocabulary and
the settlement / reconciliation block specs. These are pure data containers (no
projection logic); the measurement and settlement engine that produces them
lives in :mod:`fastcashflow._measurement.vfa.engine`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow._measurement.model import VFA
from fastcashflow._measurement.basis import (
    MEASUREMENT_BASIS_INCEPTION,
    MEASUREMENT_BASIS_SETTLEMENT,
    MEASUREMENT_BASIS_SETTLEMENT_CARRY,
)

if TYPE_CHECKING:
    from fastcashflow.model_points import ModelPoints
    from fastcashflow.projection import Cashflows


CSM_BASIS_INITIAL = "initial_measurement"           # inception headline (csm0)
CSM_BASIS_PROJECTED_RUNOFF = "projected_runoff"     # inception full trajectory
CSM_BASIS_CARRY_ONLY = "carry_only"                 # measure_inforce: prior CSM
#                                                     rolled at the basis return,
#                                                     paragraph-45 unlock deferred
CSM_BASIS_PARAGRAPH_45 = "paragraph_45_settlement"  # vfa.settle: subsequent meas.
CSM_BASES = (CSM_BASIS_INITIAL, CSM_BASIS_PROJECTED_RUNOFF,
             CSM_BASIS_CARRY_ONLY, CSM_BASIS_PARAGRAPH_45)

# The VFA keeps csm_basis as the stored field (single source of truth) and
# derives the cross-model measurement_basis from it (see _measurement.basis).
_CSM_TO_MEASUREMENT_BASIS = {
    CSM_BASIS_INITIAL: MEASUREMENT_BASIS_INCEPTION,
    CSM_BASIS_PROJECTED_RUNOFF: MEASUREMENT_BASIS_INCEPTION,
    CSM_BASIS_CARRY_ONLY: MEASUREMENT_BASIS_SETTLEMENT_CARRY,
    CSM_BASIS_PARAGRAPH_45: MEASUREMENT_BASIS_SETTLEMENT,
}

@dataclass(frozen=True, slots=True, eq=False)
class Measurement:
    """VFA measurement of a direct-participation (account-value) portfolio.

    The headline ``bel``, ``ra``, ``csm``, ``variable_fee``, ``time_value`` and
    ``loss_component`` are ``(n_mp,)`` as-of figures -- at inception for
    ``measure``, at the valuation date for ``measure_inforce`` (the RA a
    confidence-level margin for expense risk; the BEL net of the account value
    the entity holds; ``variable_fee`` the present value of the entity's fee --
    its share of the underlying items). The full path adds the
    ``(n_mp, n_time+1)`` trajectories ``bel_path`` / ``ra_path`` / ``csm_path`` /
    ``account_value_path`` (column 0 the as-of figure), ``None`` on the
    headline-only path; a grouped result also leaves ``account_value_path``
    ``None`` (the account value is a per-policy level, not a group quantity). The
    CSM is accreted at the underlying-items return and released by coverage
    units::

        csm_path[:, t+1] = csm_path[:, t] + csm_accretion[:, t] - csm_release[:, t]

    The guarantee time value drives the CSM but is reported separately in
    ``time_value``, not folded into ``bel``.
    """

    model: ClassVar[str] = VFA

    # headline -- always present, shape (n_mp,)
    bel: FloatArray              # inception BEL (net of account value)
    ra: FloatArray               # inception RA (expense risk)
    csm: FloatArray              # inception CSM
    variable_fee: FloatArray     # PV of the entity's fee
    time_value: FloatArray       # guarantee TVOG at inception
    loss_component: FloatArray   # onerous loss at inception
    # trajectory -- full only (None on the headline-only path)
    bel_path: FloatArray | None = None            # (n_mp, n_time+1) -- BEL trajectory
    ra_path: FloatArray | None = None             # (n_mp, n_time+1) -- RA trajectory
    csm_path: FloatArray | None = None            # (n_mp, n_time+1) -- CSM trajectory
    account_value_path: FloatArray | None = None  # (n_mp, n_time+1) -- account-value trajectory
    csm_accretion: FloatArray | None = None       # (n_mp, n_time)
    csm_release: FloatArray | None = None          # (n_mp, n_time)
    lic_path: FloatArray | None = None            # (n_mp, n_time+1) -- liability for incurred claims.
    # The entity's own-pocket insurance cash flows, retained for the asset-liability
    # gap (a unit-linked book's account-value benefits are funded by the unit fund;
    # only the guarantee excess over the account value lands on the entity's general
    # account). Full VA path only -- None on the headline / aggregate / UL paths.
    guarantee_excess_cf: FloatArray | None = None  # (n_mp, n_time) GMDB/GMAB excess over AV
    benefit_cf: FloatArray | None = None           # (n_mp, n_time) gross incurred benefit (AV + excess)
    fee_cf: FloatArray | None = None               # (n_mp, n_time) variable fee skimmed (entity inflow)
    # The terminal column holds the residual of claims whose settlement tail
    # runs past the horizon (stays non-zero by design, not a leak).
    discount_factor_bom: FloatArray | None = None      # (n_time+1,), or (n_mp, n_time+1) when portfolio-stitched
    cashflows: "Cashflows | None" = None
    model_points: "ModelPoints | None" = None     # stamped by vfa.measure, for group axes
    group_labels: "np.ndarray | None" = None       # per-group label on a grouped result
    group_sizes: IntArray | None = None         # model points per group, aligned with labels
    csm_basis: str = CSM_BASIS_PROJECTED_RUNOFF  # what the csm represents (see CSM_BASES)

    @property
    def measurement_basis(self) -> str:
        """Cross-model time-basis discriminator, derived from ``csm_basis``
        (the VFA's stored field stays the single source of truth)."""
        return _CSM_TO_MEASUREMENT_BASIS[self.csm_basis]

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm),
                ("fee", self.variable_fee), ("TVOG", self.time_value),
                ("loss", self.loss_component)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr(f"{self.model}.Measurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str(f"{self.model}.Measurement", self._columns())


@dataclass(frozen=True, slots=True, eq=False)
class Aggregate:
    """Portfolio-aggregate VFA view -- a scalable sum of measured model-point
    results, holding no per-model-point row. Inception totals plus the run-off
    trajectories summed over the model-point axis. Computed in bounded memory, so
    it works where a per-model-point ``vfa.measure(full=True)`` would OOM. Not an
    IFRS group remeasurement and not a group re-floor engine: ``csm`` /
    ``loss_component`` are the sum of each contract's floored figure, matching the
    headline -- not a group-level re-floor.
    """

    model: ClassVar[str] = VFA

    bel: float                       # portfolio inception BEL total
    ra: float                        # portfolio inception RA total
    csm: float                       # portfolio inception CSM total
    variable_fee: float              # portfolio variable-fee total
    time_value: float                # portfolio guarantee TVOG total
    loss_component: float            # portfolio inception loss-component total
    bel_path: FloatArray             # (n_time+1,) -- aggregate BEL trajectory
    ra_path: FloatArray              # (n_time+1,) -- aggregate RA trajectory
    csm_path: FloatArray             # (n_time+1,) -- aggregate CSM trajectory
    lic_path: FloatArray             # (n_time+1,) -- aggregate liability for incurred claims
    # No account_value_path: the account value is a per-policy level (its
    # closed-form growth never terminates at the contract boundary, so summing it
    # is horizon-dependent, not a clean aggregate) -- the group() VFA result drops
    # it for the same reason. The group's fund would be sum(inforce x av), a
    # different quantity, not modelled here.


@dataclass(frozen=True, slots=True, eq=False)
class PeriodMovement:
    """One reporting period's movement of the VFA insurance contract liability.

    The period covers months ``[month_start, month_end)``. Every array is
    ``(n_mp,)`` and each block reconciles exactly::

        bel_opening + bel_interest  - bel_release  == bel_closing
        ra_opening  + ra_interest   - ra_release   == ra_closing
        csm_opening + csm_accretion - csm_release  == csm_closing

    ``*_interest`` / ``csm_accretion`` is the unwind at the underlying-items
    return; ``*_release`` is the expected run-off over the period. Under the
    VFA the CSM absorbs the variability of the underlying items, so the
    entity's profit emerges as the CSM is released.
    """

    model: ClassVar[str] = VFA

    month_start: int
    month_end: int
    bel_opening: FloatArray
    bel_interest: FloatArray
    bel_release: FloatArray
    bel_closing: FloatArray
    ra_opening: FloatArray
    ra_interest: FloatArray
    ra_release: FloatArray
    ra_closing: FloatArray
    csm_opening: FloatArray
    csm_accretion: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """An IFRS 17 VFA reconciliation of the insurance contract liability.

    Portfolio totals for one reporting period -- the BEL, RA and CSM each
    reconciled from opening to closing. ``*_finance`` is the unwind at the
    underlying-items return; ``*_release`` is the run-off, shown negative --
    so opening plus every row equals closing.
    """

    model: ClassVar[str] = VFA

    month_start: int
    month_end: int
    bel_opening: float
    bel_finance: float
    bel_release: float
    bel_closing: float
    ra_opening: float
    ra_finance: float
    ra_release: float
    ra_closing: float
    csm_opening: float
    csm_finance: float
    csm_release: float
    csm_closing: float

    def __str__(self) -> str:
        rows = (
            ("Opening", self.bel_opening, self.ra_opening, self.csm_opening),
            ("Finance", self.bel_finance, self.ra_finance, self.csm_finance),
            ("Release", self.bel_release, self.ra_release, self.csm_release),
            ("Closing", self.bel_closing, self.ra_closing, self.csm_closing),
        )
        lines = [
            f"VFA reconciliation -- months {self.month_start + 1}-{self.month_end}",
            f"{'':16}{'BEL':>18}{'RA':>18}{'CSM':>18}",
        ]
        for name, bel, ra, csm in rows:
            lines.append(f"{name:16}{bel:>18,.0f}{ra:>18,.0f}{csm:>18,.0f}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True, eq=False)
class SettlementMovement:
    """One period's IFRS 17 paragraph-45 settlement movement of a VFA book.

    What :func:`fastcashflow.vfa.settle` returns: the opening -> closing
    movement of the BEL, RA, CSM and loss component over one reporting period
    of ``period_months``, per model point. Unlike :class:`PeriodMovement`
    (the *expected* movement sliced from an inception measurement), this is a
    *subsequent measurement*: the closing figures respond to the observed
    account value and in-force count, and the CSM absorbs the future-service
    change per paragraph 45. Every array is ``(n_mp,)`` and each block
    reconciles exactly::

        bel_closing == bel_opening + bel_interest - bel_release + bel_experience
        ra_closing  == ra_opening  + ra_interest  - ra_release  + ra_experience
        csm_closing == csm_opening + csm_accretion + csm_fv_share
                       + csm_future_service + csm_premium_experience
                       + csm_investment_experience
                       - loss_component_reversed
                       + loss_component_recognised - csm_release
        loss_component_closing == loss_component_opening
                       + loss_component_finance - loss_component_amortised
                       - loss_component_reversed + loss_component_recognised
        lic_closing == lic_opening + claims_incurred + lic_finance - claims_paid

    The liability for incurred claims (``lic_opening`` / ``claims_incurred`` /
    ``lic_finance`` / ``claims_paid`` / ``lic_closing``, paragraphs 40(b) /
    42(c) / 103(b)) is present when the basis carries a ``settlement_pattern``:
    benefit claims build it up as incurred and run it off over the pattern. The
    LIC is measured at fulfilment cash flows -- the discounted PV of the unpaid
    run-off (42(c)). It carries NO risk adjustment: the VFA RA prices expense
    risk only (the benefit risk sits in the variable fee), so the incurred
    benefits carry no RA in the LIC either. ``claims_incurred`` and
    ``claims_paid`` stay nominal (``claims_paid`` the residual on the
    undiscounted trajectory); ``lic_finance`` is the reconciling residual (the
    42(c) discount unwind + discounting measurement effect), zero at both dates
    without a pattern. It mirrors the GMM block (which adds the LIC RA).

    and the blocks tie across: ``csm_fv_share + csm_future_service ==
    -(bel_experience + ra_experience)`` -- the paragraph-45 future-service
    change is exactly minus the observed-vs-expected FCF difference.
    ``csm_premium_experience`` (B96(a)) and ``premium_experience_revenue``
    (B97(c)) are the two legs of the premium experience adjustment (actual
    premium received less the expected premium, split by the entity's
    future-service fraction). The future leg is a NEW future-service change with
    no BEL/RA counterpart, so it enters the CSM block but does NOT appear in the
    cross-tie above; the current/past leg is a P&L memo, in no balance
    recursion. Both are zero unless ``state.actual_premium`` is given.
    ``csm_investment_experience`` (B96(c)) is the same for the investment
    component (the account value returned on exits): expected less actual
    account value payable, the whole difference into the CSM, outside the
    cross-tie; zero unless ``state.actual_investment_component`` is given.
    ``loss_component_finance`` / ``loss_component_amortised`` are the
    paragraph-50(a)/51 incurred-service channel of an onerous book -- the
    guarantee-excess + expense release (the claims+expenses pool, excluding the
    account-value investment component) split on the loss-component ratio,
    running the loss component to zero by the end of coverage (52); zero on a
    profitable book.

    Line semantics:

    * ``bel_interest`` / ``ra_interest`` -- the unwind at the underlying-items
      return over the period (the engine's roll-forward convention); the
      fee / crediting wedge of the fund's own growth sits inside the release.
    * ``bel_release`` / ``ra_release`` -- the *expected* run-off, the one
      residual line per block.
    * ``bel_experience`` / ``ra_experience`` -- observed minus expected close,
      the future effect of the account-value and count deviation.
    * ``csm_accretion`` -- ``prior_csm * ((1 + r_m)**period - 1)``, the
      expected financial growth of the CSM. Under the VFA there is no
      paragraph-B72(b) locked-rate accretion; this is the expected part of
      the paragraph-45(b) change, presented jointly with ``csm_fv_share``
      as the financial / entity's-share block.
    * ``csm_fv_share`` -- paragraph 45(b), the change in the entity's share
      of the underlying items: the observed-vs-expected variable-fee PV at
      the closing date (fund-consistent end-of-month weight).
    * ``csm_future_service`` -- paragraph 45(c), every other future-service
      change: the guarantee cost (GMDB / GMAB), the crediting-floor cost and
      the count deviation's future effect.
    * ``loss_component_reversed`` / ``loss_component_recognised`` --
      paragraphs 48 / 50(b): a favourable change reverses the loss component
      before rebuilding the CSM; an unfavourable change beyond the CSM falls
      into the loss component.
    * ``csm_release`` -- paragraph B119, one period-end release of the
      post-adjustment balance over the coverage units provided in the period
      against those provided plus expected from the *opening* date.
    * ``coverage_units_provided`` / ``coverage_units_future`` -- the B119
      numerator and remainder behind that release (expected scale over the
      period, observed scale from the closing date), kept per model point
      so a group-of-contracts settlement can re-pool the release fraction
      at the group grain.
    * ``account_value_closing`` -- the *observed* fund value at the closing
      date, echoed from the input state; with the closing balances it seeds
      the next period's state (:meth:`closing_inputs`).

    v1 limitations (documented, not silent): within-period experience --
    actual deaths, lapses, benefits, expenses, AND the fees actually skimmed
    on the realized fund path -- is assumed equal to expected; only the
    closing count and observed account value deviate. Part of the
    paragraph 45(b) realized entity share is therefore not captured, and the
    period's total comprehensive income is approximate even though
    opening-to-closing balances reconcile. The loss component moves through both
    channels: the paragraph-48/50(b) future-service adjustments (``reversed`` /
    ``recognised``) and the paragraph-50(a)-52 systematic incurred-service
    allocation (``loss_component_finance`` / ``loss_component_amortised``, the
    guarantee-excess + expense pool excluding the account-value investment
    component), which runs the loss component to zero by the end of coverage. An
    opening CSM that embeds a stochastic guarantee time value is accreted
    and released but its time-value component is never remeasured (the
    movement is deterministic, intrinsic-guarantee only). Floors and the
    loss-component algebra operate per model point; within-group offsetting
    between favourable and unfavourable contracts (the group-level CSM floor
    of paragraphs 47-52) is not performed, consistent with the rest of the
    engine.
    """

    model: ClassVar[str] = VFA

    period_months: int
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
    csm_accretion: FloatArray
    csm_fv_share: FloatArray
    csm_future_service: FloatArray
    csm_premium_experience: FloatArray  # B96(a): future-service premium exp, into CSM
    premium_experience_revenue: FloatArray  # B97(c): current/past premium exp, P&L memo
    csm_investment_experience: FloatArray  # B96(c): investment-component exp, into CSM
    claims_experience: FloatArray        # B97(b)/(c): actual-vs-expected claims, P&L memo
    expense_experience: FloatArray       # B97(b)/(c): actual-vs-expected expenses, P&L memo
    loss_component_reversed: FloatArray
    loss_component_recognised: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray
    loss_component_opening: FloatArray
    loss_component_finance: FloatArray   # 51(c): r x pool interest unwind
    loss_component_amortised: FloatArray  # 50(a)/51(a)+(b): the systematic loss reversal
    loss_component_closing: FloatArray
    variable_fee_closing: FloatArray
    coverage_units_provided: FloatArray  # B119 numerator, expected scale
    coverage_units_future: FloatArray    # B119 remainder, observed scale
    account_value_closing: FloatArray    # observed fund value at the close
    lic_opening: FloatArray              # 40(b)/42(c): discounted PV of incurred claims
    claims_incurred: FloatArray          # 42(a)/103(b)(i): claims incurred this period (nominal)
    lic_finance: FloatArray              # 42(c): discount unwind + discounting measurement
    claims_paid: FloatArray              # the settlement-pattern run-off (nominal residual)
    lic_closing: FloatArray
    lock_in_rate: float = 0.0            # state echo only; no VFA locked rate
    model_points: "object | None" = None
    csm_basis: str = CSM_BASIS_PARAGRAPH_45

    @property
    def measurement_basis(self) -> str:
        """Cross-model time-basis discriminator, derived from ``csm_basis``
        (mirrors :class:`~fastcashflow.vfa.Measurement`)."""
        return _CSM_TO_MEASUREMENT_BASIS[self.csm_basis]

    def closing_inputs(self):
        """The closing-date ``(ModelPoints, InforceState)`` pair that seeds
        the next period's settle: ``prior_csm`` / ``prior_loss_component``
        are this period's closing balances, ``prior_count`` the closing
        count and ``prior_account_value`` the observed closing fund value.
        The caller advances the pair to the next observation date
        (``elapsed_months`` / ``count`` / ``account_value``) before the
        next call."""
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
            account_value=self.account_value_closing,
            prior_count=np.asarray(mp.count, dtype=np.float64),
            prior_account_value=self.account_value_closing,
            prior_loss_component=self.loss_component_closing,
        )
        return mp, state

    def closing_measurement(self) -> Measurement:
        """The closing balance sheet as a headline-only
        :class:`~fastcashflow.vfa.Measurement`, tagged
        ``csm_basis='paragraph_45_settlement'`` -- a settlement figure,
        unlike the carry-only diagnostic, so the carry-only guard does not
        reject it: ``write_measurement`` serialises it, and its figures seed
        next period's ``prior_*`` state. (``report`` / ``group`` /
        ``roll_forward`` still need the full trajectories a headline-only
        result does not carry.) ``time_value`` is zero (the movement is
        intrinsic-guarantee only)."""
        return Measurement(
            bel=self.bel_closing,
            ra=self.ra_closing,
            csm=self.csm_closing,
            variable_fee=self.variable_fee_closing,
            time_value=np.zeros_like(self.bel_closing),
            loss_component=self.loss_component_closing,
            model_points=self.model_points,
            csm_basis=CSM_BASIS_PARAGRAPH_45,
        )


_VFA_RECON_BLOCKS = (
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
        ("Accretion", "csm_accretion", "45(b)/B72(b)", False),
        ("Fair value share", "csm_fv_share", "45(b)", False),
        ("Future service", "csm_future_service", "45(c)", False),
        ("Premium experience", "csm_premium_experience", "B96(a)", False),
        ("Investment experience", "csm_investment_experience", "B96(c)", False),
        ("Loss component reversed", "loss_component_reversed", "50(b)", False),
        ("Loss component recognised", "loss_component_recognised", "48", False),
        ("Release for service", "csm_release", "45(e)/B119", False),
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
        ("Premium experience (revenue)", "premium_experience_revenue", "B97(c)", True),
        ("Claims experience", "claims_experience", "B97(b)", True),
        ("Expense experience", "expense_experience", "B97(b)", True),
    )),
)


@dataclass(frozen=True, slots=True)
class SettlementReconciliation:
    """Portfolio totals of a paragraph-45 VFA settlement movement.

    One reporting period of ``period_months`` (the per-MP valuation dates
    are elapsed months, which differ across cohorts -- so the table is
    labelled by the period length, not by a policy-month range). The
    release and loss-component-reversed rows are *stored* negative -- the
    convention of every reconciliation type here -- so within each block
    the opening plus every row equals the closing.
    """

    model: ClassVar[str] = VFA

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
    csm_fv_share: float
    csm_future_service: float
    csm_premium_experience: float
    premium_experience_revenue: float
    csm_investment_experience: float
    claims_experience: float
    expense_experience: float
    loss_component_reversed: float
    loss_component_recognised: float
    csm_release: float
    csm_closing: float
    loss_component_opening: float
    loss_component_finance: float
    loss_component_amortised: float
    loss_component_closing: float
    lic_opening: float = 0.0
    claims_incurred: float = 0.0
    lic_finance: float = 0.0
    claims_paid: float = 0.0
    lic_closing: float = 0.0

    def __str__(self) -> str:
        from fastcashflow._display import _format_settlement_reconciliation
        return _format_settlement_reconciliation(
            self, "VFA settlement reconciliation", _VFA_RECON_BLOCKS)


_VFA_SETTLEMENT_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_fv_share", "csm_future_service",
    "csm_premium_experience", "premium_experience_revenue",
    "csm_investment_experience", "claims_experience", "expense_experience",
    "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_finance",
    "loss_component_amortised", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
    "variable_fee_closing", "account_value_closing",
    "coverage_units_provided", "coverage_units_future",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
)


@dataclass(frozen=True, slots=True)
class SettlementAggregate:
    """Portfolio totals of the paragraph-45 settlement movement.

    What :func:`fastcashflow.vfa.settle_aggregate` returns: every line of
    :class:`SettlementMovement` summed over the model-point axis, in
    bounded memory, movement-positive (the display negation happens in
    :func:`reconcile`). ``reconcile(aggregate)`` equals the per-MP
    movement's reconciliation table fieldwise, and :meth:`closing_inputs`
    raises ValueError -- chaining needs per-MP balances.
    """

    model: ClassVar[str] = VFA

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
    csm_fv_share: float
    csm_future_service: float
    csm_premium_experience: float
    premium_experience_revenue: float
    csm_investment_experience: float
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
    variable_fee_closing: float
    account_value_closing: float
    coverage_units_provided: float
    coverage_units_future: float
    lic_opening: float = 0.0
    claims_incurred: float = 0.0
    lic_finance: float = 0.0
    claims_paid: float = 0.0
    lic_closing: float = 0.0
    lock_in_rate: float = 0.0            # state echo only; no VFA locked rate
    csm_basis: str = CSM_BASIS_PARAGRAPH_45

    @property
    def measurement_basis(self) -> str:
        """Cross-model time-basis discriminator, derived from ``csm_basis``
        (mirrors :class:`SettlementMovement`)."""
        return _CSM_TO_MEASUREMENT_BASIS[self.csm_basis]

    def closing_inputs(self):
        """Always raises -- see the class docstring."""
        from fastcashflow._measurement.basis import _AGGREGATE_NO_CHAIN
        raise ValueError(_AGGREGATE_NO_CHAIN)


_VFA_GOC_SETTLEMENT_LINEAR = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_fv_share", "csm_future_service", "csm_premium_experience",
    "premium_experience_revenue", "csm_investment_experience",
    "claims_experience", "expense_experience",
    "csm_opening", "csm_accretion",
    "variable_fee_closing", "account_value_closing", "loss_component_opening",
    "loss_component_finance", "loss_component_amortised",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
)

_VFA_GOC_SETTLEMENT_NONLINEAR = (
    "csm_release", "csm_closing", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
)


@dataclass(frozen=True, slots=True, eq=False)
class GoCSettlement:
    """Group-of-contracts paragraph-45 settlement movement (VFA).

    The VFA mirror of :class:`GoCSettlement`. Rows are IFRS 17 groups. The
    LINEAR VFA settlement lines are group-summed -- including ``csm_fv_share``
    (45(b)) and ``csm_future_service`` (45(c)), each carrying its own
    ``v_half`` / ``k_obs``, so the group fv_share is the SUM of the per-MP
    fv_shares (not a re-derivation from a re-summed group account value). The
    paragraph-48/50(b) algebra and the single B119 release are applied once at
    group grain on the group-summed inputs (the future-service change is
    ``sum(csm_fv_share + csm_future_service)``). ``closing_inputs()`` allocates
    the group closing CSM / loss component back to model points by closing-
    count pro-rata (or an explicit weight) and carries each contract's observed
    account value forward.
    """

    model: ClassVar[str] = VFA

    group_labels: np.ndarray
    group_sizes: IntArray
    period_months: int
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
    csm_fv_share: FloatArray
    csm_future_service: FloatArray
    csm_premium_experience: FloatArray
    premium_experience_revenue: FloatArray
    csm_investment_experience: FloatArray
    claims_experience: FloatArray
    expense_experience: FloatArray
    csm_opening: FloatArray
    csm_accretion: FloatArray
    variable_fee_closing: FloatArray
    account_value_closing: FloatArray
    loss_component_opening: FloatArray
    loss_component_finance: FloatArray
    loss_component_amortised: FloatArray
    lic_opening: FloatArray
    claims_incurred: FloatArray
    lic_finance: FloatArray
    claims_paid: FloatArray
    lic_closing: FloatArray
    coverage_units_provided: FloatArray
    coverage_units_future: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray
    loss_component_reversed: FloatArray
    loss_component_recognised: FloatArray
    loss_component_closing: FloatArray
    lock_in_rate: FloatArray
    model_points: ModelPoints | None = None
    group_inverse: IntArray | None = None
    lock_in_rate_by_mp: FloatArray | float = 0.0
    profitability_by_mp: np.ndarray | None = None
    account_value_by_mp: FloatArray | None = None
    measurement_basis: str = "settlement"

    _LINEAR: ClassVar[tuple[str, ...]] = _VFA_GOC_SETTLEMENT_LINEAR
    _NONLINEAR: ClassVar[tuple[str, ...]] = _VFA_GOC_SETTLEMENT_NONLINEAR

    def closing_inputs(self, *, allocation=None):
        from fastcashflow.model_points import InforceState
        mp = self.model_points
        inv = self.group_inverse
        if mp is None or inv is None or mp.mp_id is None:
            raise ValueError(
                "closing_inputs() needs the source model points with mp_id and "
                "group membership; use settle_group_of_contracts to create it")
        if self.account_value_by_mp is None:
            raise ValueError(
                "closing_inputs() needs the observed per-MP account value to "
                "carry forward (it is stamped by settle_group_of_contracts)")
        n_mp = mp.n_mp
        if allocation is None:
            weights = np.asarray(mp.count, dtype=np.float64)
        else:
            weights = np.asarray(allocation, dtype=np.float64)
            if weights.shape != (n_mp,):
                raise ValueError(
                    f"allocation must have one entry per model point ({n_mp}), "
                    f"got shape {weights.shape}")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("allocation must be finite and >= 0")
        n_groups = self.group_labels.shape[0]
        denom = np.bincount(inv, weights=weights, minlength=n_groups)
        share = np.zeros(n_mp, dtype=np.float64)
        for g in range(n_groups):
            rows = inv == g
            if denom[g] > 0.0:
                share[rows] = weights[rows] / denom[g]
            else:
                share[rows] = 1.0 / max(1, int(rows.sum()))
        prior_csm = self.csm_closing[inv] * share
        prior_lc = self.loss_component_closing[inv] * share
        av = np.asarray(self.account_value_by_mp, dtype=np.float64)
        state = InforceState(
            mp_id=mp.mp_id,
            elapsed_months=np.asarray(mp.elapsed_months, dtype=np.int64),
            count=np.asarray(mp.count, dtype=np.float64),
            prior_csm=prior_csm,
            lock_in_rate=self.lock_in_rate_by_mp,
            account_value=av,
            prior_count=np.asarray(mp.count, dtype=np.float64),
            prior_account_value=av,
            prior_loss_component=prior_lc,
            profitability=self.profitability_by_mp,
        )
        return mp, state

@dataclass(frozen=True, slots=True)
class TVOG:
    """Total time value of a VFA / universal-life book's guarantees.

    A participating account can carry two economically distinct guarantees that
    bite on disjoint regions of the account value, so their time values add:

    * ``credited_rate_floor`` -- the minimum-crediting-rate guarantee (the account
      is credited ``max(return, floor)`` each month), measured by
      :func:`measure_tvog`. It lifts the account value itself, realised across the
      account exits. Zero when the book carries no crediting guarantee.
    * ``account_floor`` -- the GMDB / GMAB account-value floors (a death pays
      ``max(account, GMDB)``, a maturity ``max(account, GMAB)``), measured through
      :func:`vfa.measure` and summed over the model points. They pay the SHORTFALL
      when the account falls below the guaranteed benefit.

    The crediting floor lifts the account from below the credited rate; the GMDB /
    GMAB floors top a benefit up when the account is short -- disjoint payoffs, so
    :attr:`total` is their sum, the book's full guarantee time value.
    """

    credited_rate_floor: float   # crediting-guarantee TVOG (portfolio)
    account_floor: float         # GMDB / GMAB floor TVOG (portfolio, summed over MP)

    @property
    def total(self) -> float:
        """The full guarantee time value -- crediting floor plus account floors."""
        return self.credited_rate_floor + self.account_floor
