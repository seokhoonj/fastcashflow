"""IFRS 17 reinsurance-held -- result and treaty types.

The reinsurance model owns its measurement result types and treaty value types
here (the headline :class:`Measurement`, the roll-forward / settlement result
types, the :class:`Report`, the :class:`Treaty` protocol and :class:`QuotaShare`).
These are pure data containers (no projection logic); the measurement and
settlement engine that produces them lives in
:mod:`fastcashflow._measurement.reinsurance.engine`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, TYPE_CHECKING

import numpy as np

from fastcashflow._measurement.model import REINSURANCE
from fastcashflow._typing import FloatArray, IntArray
from fastcashflow._measurement.basis import (
    MEASUREMENT_BASIS_INCEPTION, MEASUREMENT_BASIS_SETTLEMENT_CARRY)

if TYPE_CHECKING:
    from fastcashflow.model_points import ModelPoints
    from fastcashflow.projection import Cashflows


@dataclass(frozen=True, slots=True, eq=False)
class Measurement:
    """Measurement of a reinsurance contract held.

    Headline ``bel``, ``ra`` and ``csm`` are ``(n_mp,)`` inception figures --
    ``bel`` is the present value of reinsurance premiums less recoveries (a
    net cost when positive), ``ra`` is the risk transferred, ``csm`` is the
    inception net cost or gain (may be negative). The ``bel`` symbol is shared
    with the GMM result for a uniform surface, but for reinsurance held it is the
    present value of fulfilment cash flows of a reinsurance ASSET (IFRS 17 para 63),
    not a liability -- a negative ``bel`` is a net reinsurance asset. The
    trajectory fields are
    populated only on the full path; ``csm_path`` reconciles as
    ``csm_path[:, t+1] = csm_path[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    """

    model: ClassVar[str] = REINSURANCE

    # headline -- always present, shape (n_mp,)
    bel: FloatArray            # PV(reinsurance premiums) - PV(recoveries)
    ra: FloatArray             # risk transferred to the reinsurer
    csm: FloatArray            # inception net cost/gain (after any 66A loss recovery)
    loss_recovery_component: FloatArray | None = None  # (n_mp,) 66A/66B: underlying loss x recovery %
    # trajectory -- full only (None on the headline-only path)
    bel_path: FloatArray | None = None         # (n_mp, n_time+1)
    ra_path: FloatArray | None = None          # (n_mp, n_time+1)
    csm_path: FloatArray | None = None         # (n_mp, n_time+1) -- net cost/gain trajectory
    csm_accretion: FloatArray | None = None    # (n_mp, n_time)
    csm_release: FloatArray | None = None      # (n_mp, n_time)
    recovery: FloatArray | None = None         # (n_mp, n_time) -- recoveries received
    reinsurance_premium: FloatArray | None = None    # (n_mp, n_time) -- reinsurance premiums paid
    cashflows: "Cashflows | None" = None
    discount_factor_bom: FloatArray | None = None     # (n_time+1,) -- for grouped CSM re-derivation
    model_points: "ModelPoints | None" = None  # stamped by measure, for group axes
    group_labels: "np.ndarray | None" = None   # per-group label on a grouped result
    group_sizes: IntArray | None = None     # model points per group, aligned with labels
    # Time basis (see _measurement.basis). NOTE the in-force anchors differ by
    # field: bel_path/ra_path stay inception-anchored while csm_path is
    # prior_t-anchored (column 0 = the opening date) -- one more reason the
    # inception-axis consumers must reject 'settlement_carry'.
    measurement_basis: str = MEASUREMENT_BASIS_INCEPTION

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr(f"{self.model}.Measurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str(f"{self.model}.Measurement", self._columns())


@dataclass(frozen=True, slots=True)
class PeriodMovement:
    """One reporting period's movement of a reinsurance-held asset/liability.

    The period covers months ``[month_start, month_end)``. Every array is
    ``(n_mp,)`` and each block reconciles exactly::

        bel_opening + bel_interest  - bel_release  == bel_closing
        ra_opening  + ra_interest   - ra_release   == ra_closing
        csm_opening + csm_accretion - csm_release  == csm_closing

    ``bel`` is the present value of reinsurance premiums less recoveries (a net
    cost when positive); ``csm`` is the net cost / gain of the cover and may be
    negative. ``*_interest`` / ``csm_accretion`` is the unwind at the discount
    rate; ``*_release`` is the expected run-off over the period. There is no
    loss component (Sec. 65).
    """

    model: ClassVar[str] = REINSURANCE

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
    """An IFRS 17 reconciliation of a reinsurance-held asset/liability.

    Portfolio totals for one reporting period -- the BEL, RA and CSM each
    reconciled from opening to closing. ``*_finance`` is the unwind at the
    discount rate; ``*_release`` is the run-off, shown negative -- so opening
    plus every row equals closing. There is no loss component (Sec. 65).
    """

    model: ClassVar[str] = REINSURANCE

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
            f"Reinsurance reconciliation -- months "
            f"{self.month_start + 1}-{self.month_end}",
            f"{'':16}{'BEL':>18}{'RA':>18}{'CSM':>18}",
        ]
        for name, bel, ra, csm in rows:
            lines.append(f"{name:16}{bel:>18,.0f}{ra:>18,.0f}{csm:>18,.0f}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True, eq=False)
class SettlementMovement:
    """One period's IFRS 17 paragraph-66 settlement movement of a reinsurance
    contract held.

    The reinsurance counterpart of :class:`GMMSettlementMovement`. The BEL / RA
    blocks and the CSM accretion / future-service unlocking / finance wedge /
    B119 release are identical to the GMM settlement, with ONE modification:
    a reinsurance contract held cannot be onerous (paragraph 65), so the CSM is
    NOT floored and there is NO loss component. The closing CSM is simply::

        csm_closing == csm_opening + csm_accretion + csm_experience_unlocking
                       - csm_release

    and may be negative throughout -- a net cost of cover, deferred and
    amortised. The three-term cross identity still holds (the future-service
    change is measured at the B72(c) locked-in rate, the wedge to the
    current-rate BEL block is insurance finance income/expense)::

        csm_experience_unlocking + finance_wedge
            == -(bel_experience + ra_experience)

    ``loss_recovery_opening`` / ``loss_recovery_recognised`` /
    ``loss_recovery_reversed`` / ``loss_recovery_closing`` are the
    loss-recovery component (paragraphs 66A-66B), present when the cover is held
    over an ONEROUS underlying group: a separate tracked balance on the asset
    for remaining coverage, re-derived each period as the underlying group's
    loss component x the claim recovery % (B95B / B119D) and amortised in
    lock-step with the underlying loss component (B119F, paragraphs 50-52) --
    its change is a recovery recognised / reversed in P&L, excluded from the
    premium allocation. It does NOT adjust the CSM here (the 66A CSM effect is a
    one-time inception event in ``measure``: csm_after = csm0 -
    loss_recovery). Identity::

        loss_recovery_closing == loss_recovery_opening
            + loss_recovery_recognised - loss_recovery_reversed

    Zero unless ``underlying_loss_opening`` / ``underlying_loss_closing`` are
    supplied (byte-identical to a book with no onerous underlying).
    """

    model: ClassVar[str] = REINSURANCE

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
    csm_accretion: FloatArray            # 66(b)/B72(b): locked-in, direct compounding
    csm_experience_unlocking: FloatArray  # 66(c): future-service change, no floor
    finance_wedge: FloatArray            # B97(a): current-vs-locked-in gap, P&L
    csm_release: FloatArray              # 66(e)/B119: single period-end release
    csm_closing: FloatArray
    loss_recovery_opening: FloatArray      # 66B/B119F: underlying loss x recovery %
    loss_recovery_recognised: FloatArray   # more underlying loss -> more recovery
    loss_recovery_reversed: FloatArray     # underlying loss amortises -> recovery reverses (P&L)
    loss_recovery_closing: FloatArray
    coverage_units_provided: FloatArray
    coverage_units_future: FloatArray
    period_months: int = 12
    lock_in_rate: float = 0.0
    model_points: object | None = None
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """The closing-date ``(ModelPoints, InforceState)`` pair that seeds the
        next period's settle: ``prior_csm`` is this period's closing CSM (which
        may be negative -- there is no loss component) and ``prior_count`` the
        closing count. The caller advances ``elapsed_months`` / ``count`` to the
        next observation date before the next call."""
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
        )
        return mp, state


_REINSURANCE_RECON_BLOCKS = (
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
        ("Accretion", "csm_accretion", "66(b)/B72(b)", False),
        ("Experience unlocking", "csm_experience_unlocking", "66(c)/B96", False),
        ("Release for service", "csm_release", "66(e)/B119", False),
        ("Closing", "csm_closing", "101(c)", False),
    )),
    ("Loss-recovery component", (
        ("Opening", "loss_recovery_opening", "66B", False),
        ("Recognised", "loss_recovery_recognised", "66A", False),
        ("Reversed", "loss_recovery_reversed", "66B", False),
        ("Closing", "loss_recovery_closing", "66B", False),
    )),
    ("Memo (P&L)", (
        ("Finance wedge", "finance_wedge", "B97(a)", True),
    )),
)


@dataclass(frozen=True, slots=True)
class SettlementReconciliation:
    """Portfolio totals of a :class:`SettlementMovement` -- the
    paragraph-66 settlement table. Release rows are stored negative (display
    convention); ``finance_wedge`` keeps the movement sign (a P&L line outside
    the CSM block). There is no loss-component row -- a reinsurance contract
    held cannot be onerous."""

    model: ClassVar[str] = REINSURANCE

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
    finance_wedge: float
    csm_release: float
    csm_closing: float
    loss_recovery_opening: float = 0.0
    loss_recovery_recognised: float = 0.0
    loss_recovery_reversed: float = 0.0
    loss_recovery_closing: float = 0.0

    def __str__(self) -> str:
        from fastcashflow._display import _format_settlement_reconciliation
        return _format_settlement_reconciliation(
            self, "Reinsurance settlement reconciliation",
            _REINSURANCE_RECON_BLOCKS)


_REINSURANCE_SETTLEMENT_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_experience_unlocking",
    "finance_wedge", "csm_release", "csm_closing",
    "loss_recovery_opening", "loss_recovery_recognised",
    "loss_recovery_reversed", "loss_recovery_closing",
    "coverage_units_provided", "coverage_units_future",
)


@dataclass(frozen=True, slots=True)
class SettlementAggregate:
    """Portfolio totals of the paragraph-66 reinsurance settlement movement.

    What :func:`fastcashflow.reinsurance.settle_aggregate` returns: every line
    of :class:`SettlementMovement` summed over the model-point axis,
    movement-positive (``reconcile`` applies the display negation and
    reproduces the per-MP movement's table). There is no loss-component line --
    a reinsurance contract held cannot be onerous. :meth:`closing_inputs`
    raises -- chaining needs the per-MP balances.
    """

    model: ClassVar[str] = REINSURANCE

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
    finance_wedge: float
    csm_release: float
    csm_closing: float
    coverage_units_provided: float
    coverage_units_future: float
    loss_recovery_opening: float = 0.0
    loss_recovery_recognised: float = 0.0
    loss_recovery_reversed: float = 0.0
    loss_recovery_closing: float = 0.0
    measurement_basis: str = "settlement"

    def closing_inputs(self):
        """Always raises -- see the class docstring."""
        from fastcashflow._measurement.basis import _AGGREGATE_NO_CHAIN
        raise ValueError(_AGGREGATE_NO_CHAIN)


_REINSURANCE_PERIOD_LINES = (
    "reinsurance_premium_allocated", "amounts_recovered",
    "reinsurance_service_result", "ra_release", "reinsurance_finance_expense",
    "bel_finance_expense", "ra_finance_expense", "csm_finance_expense",
    "csm_accretion", "csm_release",
)


@dataclass(frozen=True, slots=True)
class Report:
    """IFRS 17 reporting figures for a reinsurance contract held, period by period.

    Reinsurance held is the *mirror* of an issued contract (IFRS 17 paragraph
    82): the cedant pays reinsurance premiums (an outflow) and receives
    recoveries of incurred claims (an inflow). Paragraph 86 lets the entity
    present the premiums paid net against the amounts recovered, or separately,
    so this report exposes the disaggregated components and leaves net-vs-gross
    a presentation choice -- ``net_reinsurance_result`` (a property) is the
    paragraph-86 net, not a stored field.

    Each flow array is shaped ``(n_mp, n_time)`` -- one row per model point, one
    column per month. ``reinsurance_premium_allocated`` (the systematic
    allocation of premiums paid, the cost side) and ``amounts_recovered``
    (recoveries of incurred claims, the income side) are both positive, matching
    the measurement's outflow-positive premium and inflow-positive recovery.
    ``reinsurance_service_result`` is the analog of the issuer service result --
    the release of the risk transferred plus the release of the CSM
    (``ra_release + csm_release``, IFRS 17 paragraphs 82 + B119), *not* the gross
    recovery-less-premium netting (that is ``net_reinsurance_result``).
    ``ra_release`` is the period release of the risk transferred (paragraph 64)
    excluding interest -- the same revenue-earned form as the issuer
    ``_report_gmm`` (the RA interest is in the finance line). The report (a P&L
    view) and the :class:`~fastcashflow.reinsurance.Reconciliation` (a liability
    roll-forward) decompose the same opening->closing transition differently, so
    ``ra_release`` here is the revenue-earned amount, not the reconciliation's
    movement residual; the finance lines and the CSM release do tie out.

    ``reinsurance_finance_expense`` is the interest unwind on the BEL and RA
    plus the CSM accretion at the locked-in rate, disaggregated by source (IFRS
    17 B130-B136) into ``bel_finance_expense`` (finance on the estimates of
    reinsurance cash flows), ``ra_finance_expense`` (finance on the risk
    transferred) and ``csm_finance_expense`` (the CSM interest, B72). The three
    sum to ``reinsurance_finance_expense`` up to floating-point rounding (the
    aggregate is kept as its own expression, so the parts may differ from it by
    a rounding step rather than re-deriving it).

    The CSM analysis of change reconciles as
    ``csm_opening + csm_accretion - csm_release = csm_closing``. There is no
    loss component (IFRS 17 Sec. 65): the CSM is the net cost or gain of the
    cover and may be negative -- a net cost is deferred and amortised, with no
    floor -- so the trajectory carries any negative value through as-is.
    """

    model: ClassVar[str] = REINSURANCE

    reinsurance_premium_allocated: FloatArray   # systematic allocation of premiums paid (cost side)
    amounts_recovered: FloatArray               # recoveries of incurred claims (income side)
    reinsurance_service_result: FloatArray      # ra_release + csm_release (paragraphs 82 + B119)
    ra_release: FloatArray                      # period unwind of the risk transferred (paragraph 64)
    reinsurance_finance_expense: FloatArray     # interest on BEL + RA + CSM accretion
    bel_finance_expense: FloatArray   # B130-B136: finance on the reinsurance FCF estimates
    ra_finance_expense: FloatArray    # B130-B136: finance on the risk transferred
    csm_finance_expense: FloatArray   # B130-B136: CSM interest at the locked-in rate (B72)
    csm_opening: FloatArray
    csm_accretion: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray

    @property
    def net_reinsurance_result(self) -> FloatArray:
        """IFRS 17 paragraph 86 net presentation: recoveries less premiums paid.

        ``amounts_recovered - reinsurance_premium_allocated`` -- positive when
        recoveries exceed the premiums allocated to the period. Paragraph 86
        permits this net presentation or the two disaggregated line items;
        the property supports the net choice without baking it into the report.
        This is *not* the service result (which is ``ra_release + csm_release``).
        """
        return self.amounts_recovered - self.reinsurance_premium_allocated

    def annual(self) -> dict[str, FloatArray]:
        """Portfolio totals aggregated to policy years.

        Each per-period line item is summed across model points and then
        across the twelve months of each policy year.
        """
        from fastcashflow.report import _to_years
        return {
            name: _to_years(getattr(self, name).sum(axis=0))
            for name in (
                "reinsurance_premium_allocated", "amounts_recovered",
                "reinsurance_service_result", "reinsurance_finance_expense",
                "csm_accretion", "csm_release",
            )
        }

    def by_period(self, period_months: int = 12, *, basis: str = "elapsed",
                  inception_month=None) -> dict[str, FloatArray]:
        """Portfolio totals bucketed into reporting periods of ``period_months``.

        The reinsurance-held counterpart of :meth:`Report.by_period`: premiums
        paid, amounts recovered, the service result, the RA release, and the
        finance expense with its B130-B136 split, summed across model points
        into each reporting period. There is no loss component (IFRS 17 Sec. 65).
        ``basis`` and ``inception_month`` behave as in :meth:`Report.by_period`.
        """
        from fastcashflow.report import _by_period
        return _by_period(self, _REINSURANCE_PERIOD_LINES, period_months, basis,
                          inception_month, None)

    def __str__(self) -> str:
        annual = self.annual()
        n_years = len(annual["reinsurance_premium_allocated"])
        shown = min(n_years, 5)
        rows = (
            ("Reinsurance premium", annual["reinsurance_premium_allocated"]),
            ("Amounts recovered",   annual["amounts_recovered"]),
            ("Net result",          annual["amounts_recovered"]
                                    - annual["reinsurance_premium_allocated"]),
            ("Service result",      annual["reinsurance_service_result"]),
            ("Finance expense",     annual["reinsurance_finance_expense"]),
            ("CSM accretion",       annual["csm_accretion"]),
            ("CSM release",         annual["csm_release"]),
        )
        title = "IFRS 17 reinsurance-held report -- annual portfolio totals"
        if n_years > shown:
            title += f" (first {shown} of {n_years} years)"
        header = f"{'':20}" + "".join(
            f"{f'Year {y + 1}':>14}" for y in range(shown)
        )
        lines = [title, header]
        for name, series in rows:
            lines.append(
                f"{name:20}"
                + "".join(f"{series[y]:>14,.0f}" for y in range(shown))
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True, eq=False)
class Aggregate:
    """Portfolio-aggregate reinsurance-held trajectories -- the scalable view.

    BEL / RA / CSM are additive across contracts, so a large ceded book's
    reinsurance asset/liability run-off is its per-model-point trajectories
    summed over the model-point axis. Holds the scalar inception totals plus the
    ``(n_time+1,)`` aggregate ``bel_path`` / ``ra_path`` / ``csm_path`` (matching
    the GMM / VFA aggregates) and the ``(n_time,)`` aggregate ``recovery`` /
    ``reinsurance_premium``. There is no loss component (Sec. 65). What
    :func:`~fastcashflow.reinsurance.measure_aggregate` returns, computed in
    bounded memory.
    """

    model: ClassVar[str] = REINSURANCE

    bel: float                      # portfolio inception BEL total
    ra: float                       # portfolio inception RA total
    csm: float                      # portfolio inception CSM total
    bel_path: FloatArray            # (n_time+1,) -- aggregate BEL trajectory
    ra_path: FloatArray             # (n_time+1,) -- aggregate RA trajectory
    csm_path: FloatArray            # (n_time+1,) -- aggregate CSM trajectory
    recovery: FloatArray            # (n_time,)   -- aggregate recoveries
    reinsurance_premium: FloatArray  # (n_time,)  -- aggregate reinsurance premiums


@dataclass(frozen=True, slots=True, eq=False)
class InforceAggregate:
    """Portfolio-aggregate reinsurance-held in-force carry -- the scale bridge.

    A headline-only total: the period-close BEL / RA / CSM of a ceded book,
    summed over the model-point axis from :func:`measure_inforce`.
    There is no loss component (Sec. 65). ``measurement_basis`` is
    ``'settlement_carry'`` -- this is a carry bridge, not a settlement: the
    reinsurance leaf has no ``settle`` yet, so the prior CSM is rolled forward
    (Sec. 44) without the Sec. 66 unlocking / loss-recovery component. It is
    deprecated once ``reinsurance.settle`` lands. A total cannot be chained.
    """

    model: ClassVar[str] = REINSURANCE

    bel: float
    ra: float
    csm: float
    period_months: int
    measurement_basis: str = MEASUREMENT_BASIS_SETTLEMENT_CARRY

    def closing_inputs(self):
        raise ValueError(
            "a InforceAggregate is a portfolio total, not a per-MP "
            "chaining citizen; carry the per-MP reinsurance.measure_inforce "
            "(or reinsurance.settle once available) to roll a period forward")


class Treaty(Protocol):
    """How a reinsurance treaty cedes the direct cash flows.

    ``cede`` receives the direct portfolio's projected :class:`Cashflows` and
    returns ``(ceded_mortality_cf, ceded_morbidity_cf, reinsurance_premium_cf)`` --
    each ``(n_mp, n_time)``. The two ceded-claim streams are kept split by
    risk so the risk adjustment can weight them by the right cv; their sum is
    the recovery. A new treaty type (excess-of-loss, surplus, ...) implements
    this one method.
    """

    def cede(self, proj: Cashflows) -> tuple[FloatArray, FloatArray, FloatArray]:
        ...


@dataclass(frozen=True, slots=True)
class QuotaShare:
    """Proportional reinsurance -- cede a fixed fraction of claims and premiums.

    ``cession`` (in ``[0, 1]``) is the ceded fraction: the cedant recovers
    that fraction of its claims and pays the same fraction of its premiums as
    reinsurance premium.
    """

    cession: float

    def __post_init__(self) -> None:
        # Validate at construction, not deep in cede(): a non-numeric, NaN or
        # out-of-range cession otherwise surfaces late or as a cryptic error.
        c = float(self.cession)  # ValueError for a non-numeric cession
        if not np.isfinite(c):
            raise ValueError(f"cession must be finite, got {self.cession!r}")
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"cession must be in [0, 1], got {self.cession!r}")

    def cede(self, proj: Cashflows) -> tuple[FloatArray, FloatArray, FloatArray]:
        if not 0.0 <= self.cession <= 1.0:
            raise ValueError(f"cession must be in [0, 1], got {self.cession}")
        return (self.cession * proj.mortality_cf,
                self.cession * proj.morbidity_cf,
                self.cession * proj.premium_cf)


