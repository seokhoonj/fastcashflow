"""IFRS 17 Premium Allocation Approach (PAA) -- result types.

The PAA model owns its measurement result types here: the headline/trajectory
:class:`Measurement`, the portfolio :class:`Aggregate`, the roll-forward
:class:`PeriodMovement` / :class:`Reconciliation`, the paragraph-55(b)
settlement :class:`SettlementMovement` / :class:`SettlementReconciliation` /
:class:`SettlementAggregate`, with the reconciliation / settlement block specs.
These are pure data containers (no projection logic); the measurement and
settlement engine that produces them lives in
:mod:`fastcashflow._measurement.paa.engine`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow._measurement.model import PAA
from fastcashflow._measurement.basis import MEASUREMENT_BASIS_INCEPTION

if TYPE_CHECKING:
    from fastcashflow.model_points import ModelPoints
    from fastcashflow.projection import Cashflows


@dataclass(frozen=True, slots=True, eq=False)
class Measurement:
    """PAA measurement -- the Liability for Remaining Coverage and the
    underwriting result released from it.

    ``lrc`` is an ``(n_mp, n_time+1)`` trajectory; column 0 is the inception
    LRC. ``revenue`` and ``service_expense`` are ``(n_mp, n_time)`` -- the
    insurance revenue earned and the insurance service expense incurred each
    month. ``service_result`` (a property) is their difference. ``lic_path`` is
    the ``(n_mp, n_time+1)`` liability for incurred claims -- claims build it
    up as they are incurred and run it off as they are paid.
    """

    model: ClassVar[str] = PAA

    # headline -- always present, shape (n_mp,)
    lrc: FloatArray              # inception Liability for Remaining Coverage
    loss_component: FloatArray   # onerous-contract loss at inception
    # inception fulfilment cash flows for remaining coverage (BEL + RA, signed:
    # negative for a profitable contract). The onerous-test input --
    # loss_component = max(0, fcf) -- kept so grouping can net it on the group
    # aggregate. The PAA liability itself is the LRC, not this.
    fcf: FloatArray | None = None
    # trajectory -- full only (None on the headline-only path)
    lrc_path: FloatArray | None = None         # (n_mp, n_time+1) -- LRC trajectory
    revenue: FloatArray | None = None          # (n_mp, n_time)   -- insurance revenue earned
    service_expense: FloatArray | None = None  # (n_mp, n_time)   -- claims + expenses incurred
    lic_path: FloatArray | None = None         # (n_mp, n_time+1) -- liability for incurred claims
    # The terminal column holds the residual of claims whose settlement tail
    # runs past the horizon (stays non-zero by design, not a leak).
    cashflows: "Cashflows | None" = None
    model_points: "ModelPoints | None" = None  # stamped by measure, for group axes
    group_labels: "np.ndarray | None" = None   # per-group label on a grouped result
    group_sizes: IntArray | None = None     # model points per group, aligned with labels
    # Time basis of the result (see _measurement.basis): the in-force LRC is an
    # as-of re-based headline over inception-axis trajectories, so
    # inception-axis consumers reject it via _require_inception.
    measurement_basis: str = MEASUREMENT_BASIS_INCEPTION

    @property
    def service_result(self) -> FloatArray:
        """Insurance service result -- revenue less service expense."""
        return self.revenue - self.service_expense

    def _columns(self):
        return [("LRC", self.lrc), ("loss", self.loss_component)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr(f"{self.model}.Measurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str(f"{self.model}.Measurement", self._columns())


@dataclass(frozen=True, slots=True, eq=False)
class Aggregate:
    """Portfolio-aggregate PAA view -- a scalable sum of measured model-point
    results, holding no per-model-point row. Inception totals plus the run-off
    trajectories summed over the model-point axis (``lrc`` is the column-0 total).
    Computed in bounded memory, so it works where a per-model-point
    ``measure(full=True)`` would OOM. Not an IFRS group remeasurement and not
    a group re-floor engine: ``loss_component`` is the sum of each contract's
    floored loss, matching the headline -- not a group-level re-floor.
    """

    model: ClassVar[str] = PAA

    lrc: float                   # portfolio inception LRC total
    loss_component: float        # portfolio inception loss-component total
    lrc_path: FloatArray         # (n_time+1,) -- aggregate LRC trajectory
    revenue: FloatArray          # (n_time,)   -- aggregate insurance revenue
    service_expense: FloatArray  # (n_time,)   -- aggregate service expense
    lic_path: FloatArray         # (n_time+1,) -- aggregate liability for incurred claims


@dataclass(frozen=True, slots=True, eq=False)
class PeriodMovement:
    """One reporting period's movement of the PAA insurance contract liability.

    The period covers months ``[month_start, month_end)``. Every array is
    ``(n_mp,)``; the three components each reconcile exactly::

        lrc_opening + premiums        - revenue                == lrc_closing
        loss_component_opening        - loss_component_release == loss_component_closing
        lic_opening + claims_incurred - claims_paid            == lic_closing

    The LRC (liability for remaining coverage) is built up by premiums and
    released by insurance revenue; the loss component runs off over the
    coverage; the LIC (liability for incurred claims) is built up as claims
    are incurred and run off as they are paid. All are held undiscounted.

    When a settlement tail runs past the horizon, the final period's
    ``lic_closing`` stays non-zero -- the parked LIC residual of claims still
    outstanding at the horizon. The invariant above still holds.
    """

    model: ClassVar[str] = PAA

    month_start: int
    month_end: int
    lrc_opening: FloatArray
    premiums: FloatArray
    revenue: FloatArray
    lrc_closing: FloatArray
    loss_component_opening: FloatArray
    loss_component_release: FloatArray
    loss_component_closing: FloatArray
    lic_opening: FloatArray
    claims_incurred: FloatArray
    claims_paid: FloatArray
    lic_closing: FloatArray


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """An IFRS 17 paragraph-100 reconciliation of the PAA liability.

    Portfolio totals for one reporting period, split into the three
    components -- the liability for remaining coverage (excluding the loss
    component), the loss component, and the liability for incurred claims.
    Run-off rows are shown negative, so opening plus every row equals
    closing.
    """

    model: ClassVar[str] = PAA

    month_start: int
    month_end: int
    lrc_opening: float
    premiums: float
    revenue: float
    lrc_closing: float
    loss_component_opening: float
    loss_component_release: float
    loss_component_closing: float
    lic_opening: float
    claims_incurred: float
    claims_paid: float
    lic_closing: float

    def __str__(self) -> str:
        blocks = (
            ("LRC (excluding loss component)", (
                ("Opening", self.lrc_opening),
                ("Premiums received", self.premiums),
                ("Insurance revenue", self.revenue),
                ("Closing", self.lrc_closing),
            )),
            ("Loss component", (
                ("Opening", self.loss_component_opening),
                ("Released", self.loss_component_release),
                ("Closing", self.loss_component_closing),
            )),
            ("Liability for incurred claims", (
                ("Opening", self.lic_opening),
                ("Claims incurred", self.claims_incurred),
                ("Claims paid", self.claims_paid),
                ("Closing", self.lic_closing),
            )),
        )
        lines = [
            f"PAA reconciliation -- months {self.month_start + 1}-{self.month_end}"
        ]
        for title, rows in blocks:
            lines.append(f"  {title}")
            for name, value in rows:
                lines.append(f"    {name:22}{value:>18,.0f}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True, eq=False)
class SettlementMovement:
    """One period's IFRS 17 paragraph-55(b) settlement movement of a PAA book.

    What :func:`fastcashflow.paa.settle` returns: the opening -> closing
    movement of the LRC, loss component, and LIC over one reporting period,
    per model point. Every measurement array is ``(n_mp,)`` and each block
    reconciles exactly::

        lrc_closing == lrc_opening + premiums - revenue + lrc_experience
        loss_component_closing == loss_component_opening
                       + loss_component_recognised - loss_component_reversed
        lic_closing == lic_opening + claims_incurred + lic_finance - claims_paid

    The LRC follows paragraph 55(b), with insurance revenue allocated under
    paragraph B126. The loss component is recalculated under paragraphs 57-58 at each
    date rather than carried, so exactly one of the recognised / reversed
    rows is positive. The LIC block supports settlement-pattern books and
    provides the paragraph 100(c) incurred-claims movement, measured at fulfilment
    cash flows -- the discounted PV of the unpaid run-off plus the risk
    adjustment (40(b)/42(c)/37), exactly like the GMM LIC; ``claims_incurred`` /
    ``claims_paid`` stay nominal and ``lic_finance`` is the reconciling
    residual. (paragraph 59(b) permits omitting the LIC discounting for <=1yr claims;
    discounting is also compliant and kept uniform with the GMM block.) There is
    no CSM block -- the PAA carries no CSM -- and the LRC itself stays
    undiscounted (paragraph 56); the finance line is on the LIC only.
    """

    model: ClassVar[str] = PAA

    lrc_opening: FloatArray
    premiums: FloatArray
    revenue: FloatArray
    lrc_experience: FloatArray
    lrc_closing: FloatArray
    loss_component_opening: FloatArray
    loss_component_recognised: FloatArray
    loss_component_reversed: FloatArray
    loss_component_closing: FloatArray
    lic_opening: FloatArray
    claims_incurred: FloatArray
    lic_finance: FloatArray
    claims_paid: FloatArray
    lic_closing: FloatArray
    claims_experience: FloatArray        # B97(b)/(c): actual-vs-expected claims, P&L memo
    expense_experience: FloatArray       # B97(b)/(c): actual-vs-expected expenses, P&L memo
    period_months: int = 12
    revenue_basis: str = "time"
    model_points: object | None = None
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """The closing-date ``(ModelPoints, InforceState)`` pair that seeds
        the next period's settle. The PAA has no CSM and no locked-in rate,
        so those state slots carry neutral values; the closing loss component
        is preserved for state-file continuity, though the next settle
        recalculates it under paragraphs 57-58 rather than reading it."""
        from fastcashflow.model_points import InforceState
        mp = self.model_points
        if mp is None or mp.mp_id is None:
            raise ValueError(
                "closing_inputs() needs the source model points with mp_id "
                "(the settle entry stamps them; per-MP chaining joins by id)")
        n_mp = self.lrc_closing.shape[0]
        state = InforceState(
            mp_id=mp.mp_id,
            elapsed_months=np.asarray(mp.elapsed_months, dtype=np.int64),
            count=np.asarray(mp.count, dtype=np.float64),
            prior_csm=np.zeros(n_mp, dtype=np.float64),
            lock_in_rate=0.0,
            prior_count=np.asarray(mp.count, dtype=np.float64),
            prior_loss_component=self.loss_component_closing,
            # carry the closing LIC so the next period -- in particular a
            # pure-LIC-runoff close past the contract boundary -- can run the
            # incurred-claims tail down with no in-force to reconstruct it from.
            prior_lic=self.lic_closing,
        )
        return mp, state


_PAA_RECON_BLOCKS = (
    ("LRC", (
        ("Opening", "lrc_opening", "100(a)", False),
        ("Premiums received", "premiums", "55(a)", False),
        ("Revenue recognised", "revenue", "B126", False),
        ("Experience", "lrc_experience", "55(b)", False),
        ("Closing", "lrc_closing", "100(a)", False),
    )),
    ("Loss component", (
        ("Opening", "loss_component_opening", "57", False),
        ("Recognised", "loss_component_recognised", "58", False),
        ("Reversed", "loss_component_reversed", "58", False),
        ("Closing", "loss_component_closing", "57", False),
    )),
    ("LIC", (
        ("Opening", "lic_opening", "100(c)", False),
        ("Claims incurred", "claims_incurred", "42(a)", False),
        ("Finance", "lic_finance", "42(c)", False),
        ("Claims paid", "claims_paid", "100(c)", False),
        ("Closing", "lic_closing", "100(c)", False),
    )),
    ("Memo (P&L)", (
        ("Claims experience", "claims_experience", "B97(b)", True),
        ("Expense experience", "expense_experience", "B97(b)", True),
    )),
)


@dataclass(frozen=True, slots=True)
class SettlementReconciliation:
    """Portfolio totals of a :class:`SettlementMovement` -- the
    paragraph-55(b) settlement table. Revenue, claims-paid and
    loss-component-reversed rows are stored negative (display convention),
    so opening plus every row of a block equals its closing; the movement
    keeps those lines positive."""

    model: ClassVar[str] = PAA

    period_months: int
    revenue_basis: str
    lrc_opening: float
    premiums: float
    revenue: float
    lrc_experience: float
    lrc_closing: float
    loss_component_opening: float
    loss_component_recognised: float
    loss_component_reversed: float
    loss_component_closing: float
    lic_opening: float
    claims_incurred: float
    lic_finance: float
    claims_paid: float
    lic_closing: float
    claims_experience: float = 0.0
    expense_experience: float = 0.0

    def __str__(self) -> str:
        from fastcashflow._display import _format_settlement_reconciliation
        return _format_settlement_reconciliation(
            self, "PAA settlement reconciliation", _PAA_RECON_BLOCKS)


_PAA_SETTLEMENT_LINES = (
    "lrc_opening", "premiums", "revenue", "lrc_experience", "lrc_closing",
    "loss_component_opening", "loss_component_recognised",
    "loss_component_reversed", "loss_component_closing",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
    "claims_experience", "expense_experience",
)


@dataclass(frozen=True, slots=True)
class SettlementAggregate:
    """Portfolio totals of the paragraph-55(b) PAA settlement movement.

    What :func:`fastcashflow.paa.settle_aggregate` returns: every line of
    :class:`SettlementMovement` summed over the model-point axis,
    movement-positive (``reconcile`` applies the display negation of the
    revenue / claims-paid / loss-component-reversed rows and reproduces the
    per-MP movement's table). There is no CSM block -- the PAA holds the LRC
    undiscounted and carries no CSM -- but the LIC carries a finance line (the
    discount unwind on incurred claims). :meth:`closing_inputs` raises --
    chaining needs the per-MP balances.
    """

    model: ClassVar[str] = PAA

    period_months: int
    revenue_basis: str
    lrc_opening: float
    premiums: float
    revenue: float
    lrc_experience: float
    lrc_closing: float
    loss_component_opening: float
    loss_component_recognised: float
    loss_component_reversed: float
    loss_component_closing: float
    lic_opening: float
    claims_incurred: float
    lic_finance: float
    claims_paid: float
    lic_closing: float
    claims_experience: float = 0.0
    expense_experience: float = 0.0
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """Always raises -- see the class docstring."""
        from fastcashflow._measurement.basis import _AGGREGATE_NO_CHAIN
        raise ValueError(_AGGREGATE_NO_CHAIN)
