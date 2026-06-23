"""IFRS 17 Premium Allocation Approach (PAA) -- the simplified measurement.

The PAA is the simplified model the standard permits for short-coverage
contracts -- IFRS 17 paragraphs 53-59 (eligibility, Sec. 53; the liability
for remaining coverage, Sec. 55; insurance revenue, Sec. B126). Instead of
the GMM's BEL / RA / CSM, the Liability for Remaining Coverage (LRC) is
measured like an unearned premium: premiums build it up, insurance revenue
draws it down as coverage is provided. There is no CSM -- profit emerges
as revenue is earned.

Scope and simplifications, each with the standard's basis:

* Acquisition cash flows are expensed as incurred -- the Sec. 59(a) option,
  available when the coverage period is one year or less -- so they are not
  held in the LRC.
* The LRC is held undiscounted: Sec. 56 does not require a financing
  adjustment when the time between providing service and the related
  premium due date is one year or less.
* Insurance revenue is allocated by ``revenue_basis``: Sec. B126(a)
  (passage of time -- premium earned straight-line over the coverage
  period, the default) or Sec. B126(b) (the expected timing of incurred
  claims and expenses).
* The onerous test (Sec. 57-58) is applied at inception. The loss is
  ``max(0, fulfilment cash flows for remaining coverage - LRC)``, which at
  inception equals ``max(0, the GMM fulfilment cash flows)``. It is
  reported separately rather than folded into the LRC carrying amount.
* The Liability for Incurred Claims runs off a claims settlement pattern;
  with no pattern set, claims settle when incurred and it is zero. In the
  settlement measurement it is measured at fulfilment cash flows -- the
  discounted PV of the unpaid run-off plus the risk adjustment (Sec. 40(b)
  / 42(c) / 37), like the GMM LIC. (Sec. 59(b) permits omitting the
  discounting when claims are paid within a year of being incurred;
  discounting is also compliant and kept uniform with the GMM block.)
"""
from __future__ import annotations

from typing import ClassVar

import math
from dataclasses import dataclass, replace

import numpy as np

from fastcashflow._measurement_model import PAA
from fastcashflow._typing import FloatArray, IntArray
from fastcashflow._measurement_basis import (
    MEASUREMENT_BASIS_INCEPTION,
    MEASUREMENT_BASIS_SETTLEMENT_CARRY,
    _inforce_marker_columns,
)
from fastcashflow.basis import Basis, _single_basis
from fastcashflow.io import (
    write_measurement, _write_measurement_columns, _stream_policies_coverages)
from fastcashflow.curves import discount_monthly_curve
from fastcashflow.numerics import (
    _carry_lic_residual, _risk_adjustment, _rollforward_kernel,
    _norm_ppf, _settlement_factor, _settlement_lic, _settlement_lic_discounted)
from fastcashflow.model_points import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows
# In-force helpers shared with the GMM path (engine does not import _paa, and
# io imports engine lazily, so this top-level import is cycle-free).
from fastcashflow.engine import _reconcile_state, _inforce_rescale


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
    model_points: "ModelPoints | None" = None  # stamped by measure_paa, for group axes
    group_labels: "np.ndarray | None" = None   # per-group label on a grouped result
    group_sizes: IntArray | None = None     # model points per group, aligned with labels
    # Time basis of the result (see _measurement_basis): the in-force LRC is an
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
    ``measure_paa(full=True)`` would OOM. Not an IFRS group remeasurement and not
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


@write_measurement.register
def _(measurement: Measurement, path, *, ids=None):
    cols = {"lrc": measurement.lrc,
            "loss_component": measurement.loss_component}
    # In-force output gets marker columns (see _measurement_basis).
    cols.update(_inforce_marker_columns(measurement, measurement.lrc.shape[0]))
    _write_measurement_columns(cols, path, ids)


def _scatter_paa_headline(n_mp, results):
    """Scatter per-chunk headline-only Measurements into one ``(n_mp,)`` result.

    ``results`` is ``[(idx, Measurement)]`` from ``measure_paa(..., full=False)``
    over row-blocks; only the headline ``lrc`` / ``loss_component`` / ``fcf`` are
    laid back, the trajectory fields staying ``None``. The portfolio orchestrator
    uses this on its ``full=False`` path so a chunked PAA partition costs
    ``O(n_mp)`` retained, not ``O(n_mp x n_time)``.
    """
    lrc = np.empty(n_mp)
    loss_component = np.empty(n_mp)
    fcf = np.empty(n_mp)
    for idx, m in results:
        lrc[idx] = m.lrc
        loss_component[idx] = m.loss_component
        fcf[idx] = m.fcf
    return Measurement(lrc=lrc, loss_component=loss_component, fcf=fcf)


def _require_full_paa(measurement, entry: str) -> None:
    """Raise if a headline-only PAA measurement reaches a path needing the
    trajectory fields. PAA has no ``bel_path``, so the shared ``_require_full``
    (which checks ``bel_path``) cannot serve it -- this checks ``lrc_path``."""
    if measurement.lrc_path is None:
        raise ValueError(
            f"{entry} requires a full=True PAA measurement; the trajectory "
            f"fields are None on the full=False headline path. Call "
            f"measure_paa(..., full=True).")


def _stitch_paa_measurements(n_mp, sub_results):
    """Scatter per-segment Measurements into one ``(n_mp, ...)`` result.

    ``sub_results`` is ``[(idx, Measurement)]`` -- each segment's headline
    and trajectories are laid into the portfolio arrays at its rows and
    zero-padded on the right to the portfolio's longest horizon (a contract
    carries no LRC past its coverage period). Unlike the GMM stitch the PAA
    holds the LRC undiscounted, so there is no per-MP discount curve to lay --
    the scatter is a pure ragged zero-pad. The mixed-portfolio orchestrator
    (``fcf.portfolio.measure``) uses this to combine a PAA partition that spans
    several routing segments into one ``Measurement``.
    """
    n_time = max(m.lrc_path.shape[1] - 1 for _, m in sub_results)

    lrc = np.empty(n_mp)
    loss_component = np.empty(n_mp)
    fcf = np.empty(n_mp)
    lrc_path = np.zeros((n_mp, n_time + 1))
    revenue = np.zeros((n_mp, n_time))
    service_expense = np.zeros((n_mp, n_time))
    lic_path = np.zeros((n_mp, n_time + 1))

    cf_2d = ("inforce", "deaths", "premium_cf", "mortality_cf", "morbidity_cf",
             "expense_cf", "annuity_cf", "disability_cf", "surrender_cf")
    cf_arrays = {name: np.zeros((n_mp, n_time)) for name in cf_2d}
    maturity_cf = np.zeros(n_mp)
    maturity_survivors = np.zeros(n_mp)

    for idx, m in sub_results:
        t = m.lrc_path.shape[1] - 1
        lrc[idx] = m.lrc
        loss_component[idx] = m.loss_component
        fcf[idx] = m.fcf
        lrc_path[idx, :t + 1] = m.lrc_path
        revenue[idx, :t] = m.revenue
        service_expense[idx, :t] = m.service_expense
        lic_path[idx, :t + 1] = m.lic_path
        _carry_lic_residual(lic_path, idx, t, n_time, m.lic_path)
        cf = m.cashflows
        for name in cf_2d:
            arr = getattr(cf, name)
            cf_arrays[name][idx, :arr.shape[1]] = arr
        maturity_cf[idx] = cf.maturity_cf
        maturity_survivors[idx] = cf.maturity_survivors

    cashflows = type(sub_results[0][1].cashflows)(
        maturity_cf=maturity_cf, maturity_survivors=maturity_survivors,
        **cf_arrays,
    )
    return Measurement(
        lrc=lrc, loss_component=loss_component, fcf=fcf,
        lrc_path=lrc_path, revenue=revenue, service_expense=service_expense,
        lic_path=lic_path, cashflows=cashflows,
    )


def measure_paa(
    model_points: ModelPoints,
    basis: Basis,
    *,
    revenue_basis: str = "time",
    full: bool = True,
) -> Measurement:
    """Measure a portfolio under the Premium Allocation Approach.

    The LRC rolls forward as ``LRC[t+1] = LRC[t] + premium[t] - revenue[t]``
    from ``LRC[0] = 0`` -- premiums received build it up, insurance revenue
    releases it. A single-premium contract gives the textbook pro-rata
    unearned premium reserve.

    ``revenue_basis`` selects the Sec. B126 allocation of insurance revenue,
    which always sums to the total premium:

    * ``"time"``   -- B126(a), passage of time: the premium earned
      straight-line over the coverage period (the default).
    * ``"claims"`` -- B126(b), the expected timing of incurred claims and
      expenses; for when the release of risk differs significantly from the
      passage of time. A policy with no service expense has no such pattern
      and falls back to ``"time"``.

    The onerous test reuses the GMM fulfilment cash flows: a contract whose
    inception fulfilment cash flows are a net outflow carries that outflow
    as a loss component.

    ``full=True`` (default) returns the LRC / revenue / LIC trajectories;
    ``full=False`` fills only the headline ``lrc`` / ``loss_component`` / ``fcf``
    and leaves the trajectory and cash-flow fields ``None``. The headline path
    skips the LRC roll, revenue allocation and LIC entirely (only the onerous
    test is needed), so ``revenue_basis`` is immaterial there. It is the
    building block the portfolio orchestrator chunks to bound memory.
    ``basis`` must resolve to a single :class:`Basis`; multi-segment routers are not
    accepted.
    """
    if revenue_basis not in ("time", "claims"):
        raise ValueError(
            f"revenue_basis must be 'time' or 'claims', got {revenue_basis!r}")
    basis = _single_basis(basis, entry="measure_paa")
    proj = project_cashflows(model_points, basis)

    # Onerous test -- the GMM inception fulfilment cash flows. Needed by both
    # paths and independent of the LRC roll, so it comes first; the headline
    # path returns right after it.
    onerous_mortality_cf, onerous_morbidity_cf = proj.mortality_cf, proj.morbidity_cf
    if basis.settlement_pattern is not None:
        # Claims are paid over the settlement pattern, not at incurrence --
        # discount them to their payment dates in the fulfilment cash flows,
        # exactly as the GMM onerous test does (engine._measure_full), so the
        # PAA loss component matches GMM for identical incurred claims. The LIC
        # below stays undiscounted (Sec. 59); only the onerous-test FCF / RA
        # see the settlement discount. With a discount curve a settlement
        # pattern is rejected at Basis construction, so discount_monthly is the
        # scalar in-year reference (Sec. 40 / B71 -- the rate at incurrence).
        factor = _settlement_factor(basis.settlement_pattern, basis.discount_monthly)
        onerous_mortality_cf = onerous_mortality_cf * factor
        onerous_morbidity_cf = onerous_morbidity_cf * factor
    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _rollforward_kernel(
        onerous_mortality_cf, onerous_morbidity_cf, proj.disability_cf, proj.expense_cf,
        proj.premium_cf, proj.annuity_cf, proj.maturity_cf, proj.surrender_cf,
        model_points.contract_boundary_months,
        discount_monthly_curve(basis, proj.n_time),
    )
    # Shared RA helper so PAA honours basis.ra_method (it hardcoded the
    # confidence-level form before, silently ignoring 'cost_of_capital').
    ra = _risk_adjustment(basis, pv_claims, pv_morbidity, pv_disability,
                          pv_survival, discount_monthly_curve(basis, proj.n_time))
    fcf = bel[:, 0] + ra[:, 0]
    loss_component = np.maximum(0.0, fcf)

    if not full:
        # Headline only: the inception opening LRC is 0 (the full path's
        # lrc[:, 0]); the trajectory, revenue and cash flows are dropped so the
        # chunked portfolio path retains O(n_mp) per row, not O(n_mp x n_time).
        return Measurement(
            lrc=np.zeros(model_points.n_mp), loss_component=loss_component,
            fcf=fcf, model_points=model_points)

    # Full path only -- the (n_mp, n_time) revenue / service-expense arrays the
    # headline never needs (kept below the early return so the headline path
    # does not allocate them).
    premium_total = proj.premium_cf.sum(axis=1)          # (n_mp,)
    service_expense = proj.mortality_cf + proj.morbidity_cf + proj.expense_cf

    # Liability for incurred claims -- claims incurred build it up, claims
    # paid (spread over the settlement pattern) run it off. Held undiscounted.
    incurred = proj.mortality_cf + proj.morbidity_cf
    if basis.settlement_pattern is None:
        lic_path = np.zeros((incurred.shape[0], incurred.shape[1] + 1))
    else:
        lic_path = _settlement_lic(incurred, basis.settlement_pattern)

    # Insurance revenue -- total premium allocated across the periods of
    # service (Sec. B126), so total revenue equals total premium.
    # B126(a) straight-line weight -- a flat in-coverage mask. Used for the
    # 'time' basis and as the 'claims' fallback when a contract has no claims
    # pattern (B126(b) -> B126(a)); the fallback used the decaying in-force
    # before, which is neither basis. The mask runs to the CONTRACT BOUNDARY
    # (Sec. 34, where coverage ends), not term_months: a contract with a
    # boundary cut (boundary < term) provides no service past the boundary, so
    # spreading revenue to term over-allocates it past coverage end.
    in_coverage = (np.arange(proj.n_time)[None, :]
                   < model_points.contract_boundary_months[:, None]
                   ).astype(np.float64)
    if revenue_basis == "time":
        weight = in_coverage
    else:                                                # "claims" (validated above)
        weight = service_expense.copy()                  # B126(b)
        empty = weight.sum(axis=1) == 0.0                # no pattern -> B126(a)
        weight[empty] = in_coverage[empty]
    weight_sum = weight.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)   # safe divide; weight=0 -> revenue=0
    revenue = premium_total[:, None] * weight / weight_sum

    # LRC roll-forward -- premiums build it up, revenue releases it.
    lrc_delta = proj.premium_cf - revenue
    n_mp, n_time = lrc_delta.shape
    lrc = np.zeros((n_mp, n_time + 1))
    lrc[:, 1:] = np.cumsum(lrc_delta, axis=1)

    return Measurement(
        lrc=lrc[:, 0],
        loss_component=loss_component,
        fcf=fcf,
        lrc_path=lrc,
        revenue=revenue,
        service_expense=service_expense,
        lic_path=lic_path,
        cashflows=proj,
        model_points=model_points,
    )


def measure_stream(
    input_path,
    output_dir,
    basis: Basis,
    *,
    coverages=None,
    calculation_methods=None,
    chunk_size: int = 20_000_000,
    revenue_basis: str = "time",
    id_column: str | None = None,
    validate_unique_mp_id: bool = True,
) -> int:
    """Stream a PAA valuation through a parquet file, chunk by chunk.

    The PAA counterpart of :func:`~fastcashflow.gmm.measure_stream`: reads the
    policies + coverages parquet in ``chunk_size`` blocks, measures each with
    ``paa.measure(..., full=False)``, and writes per-chunk
    ``part-NNNNN.parquet`` results (lrc / loss_component). Returns the number of
    model points processed. ``basis`` is a single :class:`Basis`.

    Marginal benefit note: streaming is for portfolios too large to hold in
    memory (a GMM book of 1e8 rows). PAA books -- short-duration, often grouped
    -- are typically small, so :func:`measure` or :func:`measure_aggregate` is
    usually enough; this exists for API symmetry with the other models.
    """
    basis = _single_basis(basis, entry="paa.measure_stream")
    return _stream_policies_coverages(
        input_path, output_dir, coverages=coverages,
        calculation_methods=calculation_methods, chunk_size=chunk_size,
        id_column=id_column, validate_unique_mp_id=validate_unique_mp_id,
        measure_fn=lambda mp: measure_paa(mp, basis, revenue_basis=revenue_basis,
                                          full=False),
    )


def measure_aggregate(
    model_points: ModelPoints,
    basis: Basis,
    *,
    revenue_basis: str = "time",
    chunk_size: int = 200_000,
) -> Aggregate:
    """Portfolio-aggregate PAA measurement in bounded memory.

    The PAA analogue of :func:`fastcashflow.gmm.measure_aggregate`: the LRC,
    revenue, service expense and LIC are additive across contracts, so the
    portfolio's run-off is the per-model-point trajectories summed over the
    model-point axis. Runs ``measure_paa(..., full=True)`` over row-blocks of
    ``chunk_size`` model points and accumulates only the ``(n_time+1,)`` /
    ``(n_time,)`` sums, so peak memory is ``O(chunk_size x n_time)`` regardless
    of ``n_mp`` (the PAA has no fused kernel -- a block still materialises dense
    transients, so chunking is the memory lever).

    Returns a :class:`Aggregate` (scalar LRC / loss-component totals + the
    aggregate ``lrc_path`` / ``revenue`` / ``service_expense`` / ``lic_path``). It is
    a scalable sum of the measured model-point results -- not a group
    remeasurement; the onerous loss is each contract's floored loss summed, not a
    group-level re-floor. ``basis`` is a single :class:`Basis` (mixed / routed
    portfolios go through :func:`fastcashflow.portfolio.measure_aggregate`).
    """
    if chunk_size < 1:
        # Guard before the chunk loop: chunk_size <= 0 would skip every block and
        # return zero aggregates (silently wrong) instead of measuring anything.
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    n_mp = model_points.n_mp
    # A chunk projects only to its own boundary.max(); its (shorter) aggregate
    # path adds into the leading slice of the global one -- a contract carries
    # nothing past its coverage period.
    n_time = int(np.asarray(model_points.contract_boundary_months).max())
    lrc_path = np.zeros(n_time + 1)
    revenue = np.zeros(n_time)
    service_expense = np.zeros(n_time)
    lic_path = np.zeros(n_time + 1)
    lrc = loss = 0.0
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        m = measure_paa(model_points.subset(idx), basis,
                        revenue_basis=revenue_basis, full=True)
        nt1 = m.lrc_path.shape[1]
        nt = m.revenue.shape[1]
        lrc_path[:nt1] += m.lrc_path.sum(axis=0)
        revenue[:nt] += m.revenue.sum(axis=0)
        service_expense[:nt] += m.service_expense.sum(axis=0)
        lic_path[:nt1] += m.lic_path.sum(axis=0)
        lrc += float(m.lrc.sum())
        loss += float(m.loss_component.sum())
    return Aggregate(
        lrc=lrc, loss_component=loss, lrc_path=lrc_path, revenue=revenue,
        service_expense=service_expense, lic_path=lic_path)


def measure_inforce(
    model_points: ModelPoints,
    state: "InforceState",
    basis: Basis,
    *,
    revenue_basis: str = "time",
    full: bool = True,
) -> Measurement:
    """In-force diagnostic / runoff valuation of a PAA book at a single date.

    The PAA has no CSM, so there is no prior-CSM carry-forward (IFRS 17 Sec. 44
    is the CSM roll, which the PAA does not have): the in-force Liability for
    Remaining Coverage is the unearned premium still to be earned at each
    contract's ``elapsed_months`` duration. The projection runs from inception
    (:func:`measure_paa`) and the LRC trajectory is sliced at ``elapsed_months``
    per model point, re-based by ``count / inforce[elapsed]`` so the as-of
    in-force equals the input ``count`` -- exact for the LRC, which is linear in
    the in-force.

    ``state`` (an :class:`~fastcashflow.InforceState`) supplies the period-close
    ``elapsed_months`` / ``count``, reconciled onto ``model_points`` by
    :func:`~fastcashflow.apply_inforce_state`; its ``prior_csm`` /
    ``lock_in_rate`` are ignored -- the PAA has no CSM. The subsequent onerous
    re-test on remaining coverage (Sec. 57-58) belongs to the PAA period-close
    settlement (a later phase), so ``loss_component`` is zero and ``fcf`` is
    ``None`` here -- this is a diagnostic / runoff view, like
    ``gmm.measure_inforce``.

    ``full=True`` (default) keeps the inception-to-horizon LRC / revenue /
    service-expense / LIC trajectories for movement analysis, with the headline
    ``lrc`` the as-of valuation-date value; ``full=False`` returns just the
    headline ``lrc``.

    Limitations (v1): because ``fcf`` is ``None`` (the onerous re-test is
    deferred), :func:`~fastcashflow.group_of_contracts` on a PAA in-force result
    raises -- the re-floor has no fulfilment-cash-flow input.
    """
    basis = _single_basis(basis, entry="paa.measure_inforce")
    # Reorder state to model-points order and reject a stale snapshot whose
    # elapsed_months / count disagree with the state (same guard as the GMM
    # path). PAA ignores prior_csm / lock_in_rate -- there is no CSM.
    _reconcile_state(model_points, state)
    m = measure_paa(model_points, basis, revenue_basis=revenue_basis, full=True)
    n_mp = m.lrc.shape[0]
    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    # The LRC trajectory and in-force only extend to t = contract_boundary_months
    # (Sec. 34; == term when no boundary cut). At or beyond the boundary there is
    # no remaining coverage, and _inforce_rescale's inforce[rows, em] would read a
    # stale zero or index out of bounds. boundary is backfilled to term in
    # ModelPoints.__post_init__ (never None, <= term).
    boundary = np.asarray(model_points.contract_boundary_months, dtype=np.int64)
    runoff = em >= boundary
    if np.any(runoff):
        bad = int(np.argmax(runoff))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em[bad])} >= "
            f"contract_boundary_months[{bad}]={int(boundary[bad])} (the Sec. 34 "
            "horizon; equal to term_months when no boundary cut); the contract "
            "has no remaining coverage at the valuation date. paa.measure_inforce "
            "needs an as-of date strictly before the contract boundary.")
    rows = np.arange(n_mp)
    # Re-base the inception-run LRC to the valuation date (see _inforce_rescale):
    # exact for the LRC, which is linear in the in-force.
    rescale = _inforce_rescale(m.cashflows.inforce, model_points, em, rows)
    lrc = m.lrc_path[rows, em] * rescale
    # Subsequent onerous recognition is deferred (see docstring): zero loss, no
    # fcf re-test -- roll_forward / a later phase performs the unlocking.
    loss = np.zeros(n_mp, dtype=np.float64)
    if not full:
        return Measurement(
            lrc=lrc, loss_component=loss, model_points=model_points,
            measurement_basis=MEASUREMENT_BASIS_SETTLEMENT_CARRY)
    # Trajectory fields keep the full inception-to-horizon paths (as the GMM
    # in-force does); only the headline lrc is the as-of, re-based slice.
    return Measurement(
        lrc=lrc, loss_component=loss, fcf=None,
        lrc_path=m.lrc_path, revenue=m.revenue,
        service_expense=m.service_expense, lic_path=m.lic_path,
        cashflows=m.cashflows, model_points=model_points,
        measurement_basis=MEASUREMENT_BASIS_SETTLEMENT_CARRY)


def settle(
    model_points: ModelPoints,
    state: "InforceState",
    basis: Basis,
    *,
    revenue_basis: str = "time",
    period_months: int | None = None,
) -> "PAASettlementMovement":
    """Paragraph-55(b) period-close settlement of a PAA in-force book.

    The opening -> closing movement over one reporting period: the Liability
    for Remaining Coverage rolled per Sec. 55(b), the loss component
    recalculated per Sec. 57-58 at each date (no balance tracking), and the
    Liability for Incurred Claims rolled on the expected basis -- including
    settlement-pattern books, unlike ``gmm.settle`` / ``vfa.settle``.

    The PAA's structural simplification: the opening balances are
    RECONSTRUCTED from a unit projection rather than carried. The GMM CSM is
    history-dependent (accumulated unlocking) and must arrive in
    ``state.prior_csm``; the PAA LRC is the mechanical Sec. 55(b) roll, so
    with expected within-period cash flows (the v1 cut) the unit projection
    rebuilds every opening figure exactly. ``state.prior_count`` is therefore
    the only required prior-date input; ``prior_csm`` / ``lock_in_rate`` /
    ``prior_loss_component`` are ignored (``prior_loss_component`` is echoed
    by ``closing_inputs()`` for state-file continuity only). Chaining
    telescopes without any carried state -- the next period's reconstructed
    opening equals this period's closing as an identity, on- or off-track.

    One unit projection (``count = 1``), two scales:
    ``k_exp = prior_count / unit_inforce[em_open]`` (the expected leg --
    opening LRC, premiums, revenue, the whole LIC block) and
    ``k_obs = count / unit_inforce[em_close]`` (the observation -- closing
    LRC). Their gap is the ``lrc_experience`` line. The LIC block stays
    entirely at ``k_exp``: incurred claims are past events, not in-force, so
    re-scaling them by the closing count would be meaningless.

    Sec. 55(b)(i) premium experience (the PAA counterpart of the GMM B96(a)):
    when ``state.actual_premium`` is given, the ``premiums`` line is the actual
    premium received over the period rather than the expected. PAA has no CSM,
    so the difference sits in the LRC (the unearned premium, Sec. 55(a)) and is
    earned as revenue over the remaining coverage -- there is no future/current
    split. The higher LRC also feeds the Sec. 57-58 onerous re-test. Absent
    the input the premiums stay expected (byte-identical).

    Sec. 55(b) items with no movement line are documented engine cuts, not
    omissions: acquisition cash flows and their amortisation are zero under
    the Sec. 59(a) expensed-as-incurred option, the financing adjustment is
    zero because the LRC is held undiscounted (Sec. 56), and investment
    components are out of scope for the short-coverage book (v1). For the
    Sec. 100(c) incurred-claims table the LIC lines supply the cash-flow
    column; the risk-adjustment column is structurally zero (the PAA LIC
    carries no risk adjustment, Sec. 59(b)).

    A pure-LIC-runoff close (opening date at or past the contract boundary,
    only the claims tail of already-incurred claims left) is supported: coverage
    has ended so every in-force-scaled line is zero, and the carried
    ``state.prior_lic`` seeds the LIC run-off over an extended horizon (claims
    incurred in the last coverage months still settle). A final settlement
    (``em_close >= boundary``) is allowed with a zero closing count: the LRC
    releases in full through revenue while the LIC tail stays outstanding, and
    ``closing_inputs()`` carries that tail forward as the next period's
    ``prior_lic``.

    v1 scope (documented cuts, mirroring the gmm / vfa settles): within-period
    cash flows are as expected -- the observed input is the closing count only
    (actual premiums received are the Sec. 55(b)(i) input verbatim, so they and
    actual claims paid are the v1.1 priority); no subsequent Sec. 53 eligibility
    re-test; no OCI (the Sec. 56 undiscounted LRC has no finance line at all).
    """
    if revenue_basis not in ("time", "claims"):
        raise ValueError(
            f"revenue_basis must be 'time' or 'claims', got {revenue_basis!r}")
    basis = _single_basis(basis, entry="paa.settle")
    state = _reconcile_state(model_points, state)
    if state.prior_count is None:
        raise ValueError(
            "paa.settle needs state.prior_count -- the in-force count at the "
            "opening date (the expected leg's scale for the Sec. 55(b) roll)."
        )
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period}")

    n_mp = model_points.n_mp
    em_close = np.asarray(model_points.elapsed_months, dtype=np.int64)
    em_open = em_close - period
    if np.any(em_open < 0):
        bad = int(np.argmin(em_open))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em_close[bad])} < "
            f"period_months={period}; the opening date precedes inception, "
            "which has no balances to settle from."
        )

    boundary = np.asarray(model_points.contract_boundary_months,
                          dtype=np.int64)
    # A pure-LIC-runoff period opens at or past the contract boundary: coverage
    # has ended (the LRC is fully released) and only the claims tail of
    # already-incurred claims remains. There is no in-force to scale the LIC by,
    # so the carried closing LIC (state.prior_lic) seeds the run-off (the LIC
    # block below scales the extended run-off trajectory by prior_lic instead of
    # the in-force k_exp). paa.settle's closing_inputs() carries prior_lic.
    runoff = em_open >= boundary
    if np.any(runoff) and state.prior_lic is None:
        bad = int(np.argmax(runoff))
        raise ValueError(
            f"elapsed_months[{bad}]={int(em_close[bad])} - "
            f"period_months={period} is at or past the contract boundary "
            f"({int(boundary[bad])}); a pure-LIC-runoff settlement needs "
            "state.prior_lic (the carried closing liability for incurred "
            "claims) -- there is no in-force left to reconstruct the liability "
            "from. paa.settle's closing_inputs() carries it forward."
        )

    count = np.asarray(model_points.count, dtype=np.float64)
    final = em_close >= boundary
    if np.any(final & (count > 0.0)):
        bad = int(np.argmax(final & (count > 0.0)))
        raise ValueError(
            f"row {bad} closes at or past the contract boundary with "
            f"count={count[bad]}; a final settlement needs a zero closing "
            "snapshot."
        )

    # One unit projection, two scales (contract Sec. 2): seed count=1 so the
    # k factors carry the actual book scale.
    unit_mp = replace(model_points, count=np.ones(n_mp, dtype=np.float64))
    unit = measure_paa(unit_mp, basis, revenue_basis=revenue_basis, full=True)
    cf = unit.cashflows
    rows = np.arange(n_mp)
    n_time = cf.inforce.shape[1]
    cap = np.minimum(em_close, boundary)
    # Runoff MPs open at or past the boundary -- clamp their index into the
    # coverage-period trajectories (their k_exp is forced to zero just below, so
    # every in-force-scaled line auto-zeros; only the LIC, scaled by prior_lic
    # over the extended run-off, is non-zero for them).
    em_open_idx = np.minimum(em_open, n_time - 1)

    surv_open = cf.inforce[rows, em_open_idx]
    k_exp = (np.asarray(state.prior_count, dtype=np.float64)
             / np.where(surv_open > 0.0, surv_open, 1.0))
    # No in-force expected leg past the boundary: zero k_exp so the LRC, revenue,
    # premiums, loss-component and claims_incurred lines are all zero for runoff
    # MPs, regardless of the (meaningless) prior_count at a past-coverage date.
    k_exp = np.where(runoff, 0.0, k_exp)

    close_idx = np.minimum(cap, n_time - 1)
    surv_close = np.where(final, 0.0, cf.inforce[rows, close_idx])
    dead_unit = (~final) & (surv_close <= 0.0) & (count > 0.0)
    if np.any(dead_unit):
        bad = int(np.argmax(dead_unit))
        raise ValueError(
            f"row {bad}: the projection has no survivors at the closing date "
            f"but the observed count is {count[bad]}; reconcile the snapshot."
        )
    k_obs = np.where(surv_close > 0.0,
                     count / np.where(surv_close > 0.0, surv_close, 1.0), 0.0)

    # Within-period sums over [em_open, cap) -- expected (k_exp) scale.
    cols = em_open[:, None] + np.arange(period)[None, :]
    col_ok = cols < cap[:, None]
    cols_safe = np.where(col_ok, cols, n_time - 1)

    premiums = k_exp * (cf.premium_cf[rows[:, None], cols_safe]
                        * col_ok).sum(axis=1)
    revenue = k_exp * (unit.revenue[rows[:, None], cols_safe]
                       * col_ok).sum(axis=1)

    lrc_opening    = k_exp * unit.lrc_path[rows, em_open_idx]
    lrc_closing    = k_obs * unit.lrc_path[rows, cap]
    lrc_experience = (k_obs - k_exp) * unit.lrc_path[rows, cap]

    # Sec. 55(b)(i) premium experience (the PAA counterpart of the GMM B96(a)):
    # the premiums line is the ACTUAL premium received over the period. PAA has
    # no CSM, so the difference from expected sits in the LRC (the unearned
    # premium, Sec. 55(a)) and is earned as revenue over the remaining coverage
    # -- there is no future/current split. The higher LRC also feeds the
    # Sec. 57-58 onerous re-test below (more premium -> a higher LRC -> less
    # onerous). Absent state.actual_premium => premiums stay expected
    # (byte-identical). The block identity lrc_closing == lrc_opening +
    # premiums - revenue + lrc_experience is preserved.
    if state.actual_premium is not None:
        actual_premium = np.asarray(state.actual_premium, dtype=np.float64)
        premium_experience = actual_premium - premiums
        premiums = actual_premium
        lrc_closing = lrc_closing + premium_experience

    # Sec. 57-58 re-test: the fulfilment cash flows for remaining coverage at
    # each date, with the settlement discount on claims exactly as the
    # inception onerous test applies it (measure_paa above).
    onerous_mortality_cf, onerous_morbidity_cf = cf.mortality_cf, cf.morbidity_cf
    if basis.settlement_pattern is not None:
        factor = _settlement_factor(basis.settlement_pattern,
                                    basis.discount_monthly)
        onerous_mortality_cf = onerous_mortality_cf * factor
        onerous_morbidity_cf = onerous_morbidity_cf * factor
    discount_monthly = discount_monthly_curve(basis, n_time)
    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = (
        _rollforward_kernel(
            onerous_mortality_cf, onerous_morbidity_cf, cf.disability_cf,
            cf.expense_cf, cf.premium_cf, cf.annuity_cf, cf.maturity_cf,
            cf.surrender_cf, boundary, discount_monthly))
    ra = _risk_adjustment(basis, pv_claims, pv_morbidity, pv_disability,
                          pv_survival, discount_monthly)

    fcf_open  = k_exp * (bel[rows, em_open_idx] + ra[rows, em_open_idx])
    fcf_close = k_obs * (bel[rows, cap] + ra[rows, cap])
    loss_component_opening = np.maximum(0.0, fcf_open - lrc_opening)
    loss_component_closing = np.maximum(0.0, fcf_close - lrc_closing)
    loss_component_recognised = np.maximum(
        0.0, loss_component_closing - loss_component_opening)
    loss_component_reversed = np.maximum(
        0.0, loss_component_opening - loss_component_closing)

    # LIC block (paragraphs 40(b) / 42(c) / 37): the liability for incurred
    # claims measured at fulfilment cash flows -- the discounted PV of the unpaid
    # run-off plus the risk adjustment. paragraph 59(b) PERMITS omitting the
    # discounting for <=1yr PAA claims, but discounting is also compliant and
    # keeps the LIC uniform with the GMM block (and 59(b) never exempts the RA).
    # claims_incurred and claims_paid stay NOMINAL; lic_finance is the
    # reconciling residual (the 42(c) discount unwind + discounting/RA effect).
    #
    # One formula spans in-coverage and pure-runoff MPs via a unified SCALE: an
    # in-coverage period reconstructs the run-off from the projection at the
    # in-force scale k_exp; a runoff period (em_open >= boundary, k_exp forced to
    # zero above) has no in-force, so the carried prior_lic implies the scale and
    # the run-off extends past the boundary -- claims incurred in the last
    # coverage months still settle (the run-off trajectory is padded by
    # pattern_length - 1 months). For an in-coverage MP the extended trajectory
    # equals the in-coverage one within the boundary, so this is byte-identical.
    incurred = cf.mortality_cf + cf.morbidity_cf
    claims_incurred = k_exp * (incurred[rows[:, None], cols_safe]
                               * col_ok).sum(axis=1)
    if basis.settlement_pattern is not None:
        pattern = np.asarray(basis.settlement_pattern, dtype=np.float64)
        pad = pattern.size - 1                 # claims in the last coverage months
        claim_ext = np.concatenate([cf.mortality_cf, np.zeros((n_mp, pad))], axis=1)
        morb_ext = np.concatenate([cf.morbidity_cf, np.zeros((n_mp, pad))], axis=1)
        r_lic = basis.discount_monthly
        lic_death = _settlement_lic_discounted(claim_ext, pattern, r_lic)
        lic_morb = _settlement_lic_discounted(morb_ext, pattern, r_lic)
        z = _norm_ppf(basis.ra_confidence)
        lic_ra = z * (basis.mortality_cv * lic_death
                      + basis.morbidity_cv * lic_morb)
        lic_fcf_ext = lic_death + lic_morb + lic_ra
        lic_undisc_ext = _settlement_lic(claim_ext + morb_ext, pattern)
        n_ext = lic_fcf_ext.shape[1]
        # in-coverage closing caps at the boundary (the retained tail); runoff
        # opens / closes in the extended range and runs the carried tail down.
        open_idx = np.where(runoff, np.minimum(em_open, n_ext - 1), em_open)
        close_idx = np.where(runoff, np.minimum(em_close, n_ext - 1), cap)
        fcf_open_unit = lic_fcf_ext[rows, open_idx]
        if state.prior_lic is not None:
            prior_lic = np.asarray(state.prior_lic, dtype=np.float64)
            # An inconsistent carry: a positive prior_lic with no remaining
            # run-off schedule at the opening date (the unit run-off is already
            # exhausted there, so the carried tail and the settlement pattern
            # disagree). Reject rather than silently zero the carried balance --
            # correct closing_inputs() chaining never reaches this (prior_lic is
            # zero once the tail has fully run off).
            stuck = runoff & (prior_lic > 1e-9) & (fcf_open_unit <= 1e-12)
            if np.any(stuck):
                bad = int(np.argmax(stuck))
                raise ValueError(
                    f"row {bad}: state.prior_lic={float(prior_lic[bad]):.6g} > 0 "
                    "but the claims run-off schedule is already exhausted at the "
                    f"opening date (elapsed_months - period_months = "
                    f"{int(em_open[bad])}); the carried liability and the "
                    "settlement pattern disagree -- check the opening date or "
                    "prior_lic."
                )
            scale = np.where(
                runoff,
                prior_lic / np.where(fcf_open_unit > 0.0, fcf_open_unit, 1.0),
                k_exp)
        else:
            scale = k_exp
        lic_opening = scale * fcf_open_unit
        lic_closing = scale * lic_fcf_ext[rows, close_idx]
        claims_paid = (scale * lic_undisc_ext[rows, open_idx] + claims_incurred
                       - scale * lic_undisc_ext[rows, close_idx])
    else:
        # no settlement pattern: claims pay as incurred, so there is no LIC and
        # no run-off (the carried prior_lic, if any, is zero).
        lic_opening = k_exp * unit.lic_path[rows, em_open_idx]
        lic_closing = k_exp * unit.lic_path[rows, cap]
        claims_paid = lic_opening + claims_incurred - lic_closing
    lic_finance = lic_closing - lic_opening - claims_incurred + claims_paid

    # B97(b)/(c) within-period claims and expense experience (the gmm.settle /
    # vfa.settle mirror): the actual claims / expenses incurred over the period
    # less the expected, recognised in the insurance service result (P&L memos
    # -- not a balance recursion). Expected claims are claims_incurred (above);
    # expected expenses are the period expense run at the expected scale. Absent
    # state.actual_claims / state.actual_expenses => zero (byte-identical).
    exp_expenses = k_exp * (cf.expense_cf[rows[:, None], cols_safe]
                            * col_ok).sum(axis=1)
    if state.actual_claims is not None:
        claims_experience = (np.asarray(state.actual_claims, dtype=np.float64)
                             - claims_incurred)
    else:
        claims_experience = np.zeros(n_mp)
    if state.actual_expenses is not None:
        expense_experience = (np.asarray(state.actual_expenses, dtype=np.float64)
                              - exp_expenses)
    else:
        expense_experience = np.zeros(n_mp)

    from fastcashflow.movement import PAASettlementMovement
    return PAASettlementMovement(
        lrc_opening=lrc_opening,
        premiums=premiums,
        revenue=revenue,
        lrc_experience=lrc_experience,
        claims_experience=claims_experience,
        expense_experience=expense_experience,
        lrc_closing=lrc_closing,
        loss_component_opening=loss_component_opening,
        loss_component_recognised=loss_component_recognised,
        loss_component_reversed=loss_component_reversed,
        loss_component_closing=loss_component_closing,
        lic_opening=lic_opening,
        claims_incurred=claims_incurred,
        lic_finance=lic_finance,
        claims_paid=claims_paid,
        lic_closing=lic_closing,
        period_months=period,
        revenue_basis=revenue_basis,
        model_points=model_points,
    )


def settle_aggregate(
    model_points: ModelPoints,
    state: "InforceState",
    basis: Basis,
    *,
    revenue_basis: str = "time",
    period_months: int | None = None,
    chunk_size: int = 200_000,
) -> "PAASettlementAggregate":
    """Portfolio-total paragraph-55(b) PAA settlement in bounded memory.

    The PAA counterpart of :func:`~fastcashflow.gmm.settle_aggregate`: runs
    :func:`settle` over row blocks of ``chunk_size`` model points and
    accumulates only the scalar line totals (every LRC / loss-component / LIC
    line is additive across contracts), combined with ``math.fsum`` so the
    total does not depend on the chunking. ``state`` joins ``model_points`` by
    mp_id once, before chunking. The aggregate cannot be chained --
    ``closing_inputs()`` raises; chain per-MP movements instead.
    """
    from fastcashflow.movement import (
        _PAA_SETTLEMENT_LINES, PAASettlementAggregate)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    period = 12 if period_months is None else int(period_months)
    if period < 1:
        raise ValueError(f"period_months must be >= 1, got {period}")
    state = _reconcile_state(model_points, state)
    n_mp = model_points.n_mp
    parts: dict[str, list[float]] = {n: [] for n in _PAA_SETTLEMENT_LINES}
    for start in range(0, n_mp, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n_mp))
        mv = settle(model_points.subset(idx), state.subset(idx), basis,
                    revenue_basis=revenue_basis, period_months=period)
        for name in _PAA_SETTLEMENT_LINES:
            parts[name].append(float(getattr(mv, name).sum()))
    return PAASettlementAggregate(
        period_months=period, revenue_basis=revenue_basis,
        **{name: math.fsum(vals) for name, vals in parts.items()})


def settle_stream(
    input_path,
    output_dir,
    basis: Basis,
    *,
    coverages=None,
    calculation_methods=None,
    state_path=None,
    revenue_basis: str = "time",
    period_months: int | None = None,
    chunk_size: int = 200_000,
    id_column: str | None = None,
    validate_unique_mp_id: bool = True,
) -> int:
    """Stream a paragraph-55(b) PAA period close through a parquet file.

    The out-of-core variant of :func:`~fastcashflow.paa.settle`: reads the
    in-force book in ``chunk_size`` blocks, settles each, and writes the per-MP
    settlement movements as a parquet dataset (one ``part-NNNNN.parquet`` per
    chunk). Same one-combined-file / two-file (``state_path``) layouts as
    :func:`~fastcashflow.gmm.settle_stream`; the PAA needs only
    ``prior_count`` in the state (the LRC roll reconstructs the rest). Returns
    the number of model points processed.
    """
    from fastcashflow.io import _settle_stream_driver, _coverages_build_mp
    basis = _single_basis(basis, entry="paa.settle_stream")
    build_mp = _coverages_build_mp(coverages, calculation_methods,
                                   entry="paa.settle_stream")
    return _settle_stream_driver(
        input_path, output_dir, state_path=state_path, chunk_size=chunk_size,
        id_column=id_column, validate_unique_mp_id=validate_unique_mp_id,
        build_mp=build_mp,
        settle_fn=lambda mp, st: settle(mp, st, basis,
                                        revenue_basis=revenue_basis,
                                        period_months=period_months),
        entry="paa.settle_stream")
