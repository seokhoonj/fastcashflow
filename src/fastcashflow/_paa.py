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
* The Liability for Incurred Claims (Sec. 59(b)) runs off a claims
  settlement pattern; with no pattern set, claims settle when incurred and
  it is zero. It is held undiscounted -- Sec. 59(b) permits this when
  claims are paid within a year of being incurred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.basis import Basis, _single_basis
from fastcashflow.io import write_measurement, _write_measurement_columns
from fastcashflow.curves import discount_monthly_curve
from fastcashflow.numerics import (
    _carry_lic_residual, _risk_adjustment, _rollforward_kernel, _settlement_lic)
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows
# In-force helpers shared with the GMM path (engine does not import _paa, and
# io imports engine lazily, so this top-level import is cycle-free).
from fastcashflow.engine import _reconcile_state, _inforce_rescale


@dataclass(frozen=True, slots=True, eq=False)
class PAAMeasurement:
    """PAA measurement -- the Liability for Remaining Coverage and the
    underwriting result released from it.

    ``lrc`` is an ``(n_mp, n_time+1)`` trajectory; column 0 is the inception
    LRC. ``revenue`` and ``service_expense`` are ``(n_mp, n_time)`` -- the
    insurance revenue earned and the insurance service expense incurred each
    month. ``service_result`` (a property) is their difference. ``lic`` is
    the ``(n_mp, n_time+1)`` liability for incurred claims -- claims build it
    up as they are incurred and run it off as they are paid.
    """

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
    lic: FloatArray | None = None              # (n_mp, n_time+1) -- liability for incurred claims
    # The terminal column holds the residual of claims whose settlement tail
    # runs past the horizon (stays non-zero by design, not a leak).
    cashflows: "Cashflows | None" = None
    model_points: "ModelPoints | None" = None  # stamped by measure_paa, for group axes
    group_labels: "np.ndarray | None" = None   # per-group label on a grouped result
    group_sizes: IntArray | None = None     # model points per group, aligned with labels

    @property
    def service_result(self) -> FloatArray:
        """Insurance service result -- revenue less service expense."""
        return self.revenue - self.service_expense

    def _columns(self):
        return [("LRC", self.lrc), ("loss", self.loss_component)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr("PAAMeasurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str("PAAMeasurement", self._columns())


@dataclass(frozen=True, slots=True, eq=False)
class PAAAggregate:
    """Portfolio-aggregate PAA view -- a scalable sum of measured model-point
    results, holding no per-model-point row. Inception totals plus the run-off
    trajectories summed over the model-point axis (``lrc`` is the column-0 total).
    Computed in bounded memory, so it works where a per-model-point
    ``measure_paa(full=True)`` would OOM. Not an IFRS group remeasurement and not
    a group re-floor engine: ``loss_component`` is the sum of each contract's
    floored loss, matching the headline -- not a group-level re-floor.
    """

    lrc: float                   # portfolio inception LRC total
    loss_component: float        # portfolio inception loss-component total
    lrc_path: FloatArray         # (n_time+1,) -- aggregate LRC trajectory
    revenue: FloatArray          # (n_time,)   -- aggregate insurance revenue
    service_expense: FloatArray  # (n_time,)   -- aggregate service expense
    lic: FloatArray              # (n_time+1,) -- aggregate liability for incurred claims


@write_measurement.register
def _(measurement: PAAMeasurement, path, *, ids=None):
    _write_measurement_columns(
        {"lrc": measurement.lrc, "loss_component": measurement.loss_component},
        path, ids)


def _scatter_paa_headline(n_mp, results):
    """Scatter per-chunk headline-only PAAMeasurements into one ``(n_mp,)`` result.

    ``results`` is ``[(idx, PAAMeasurement)]`` from ``measure_paa(..., full=False)``
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
    return PAAMeasurement(lrc=lrc, loss_component=loss_component, fcf=fcf)


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
    """Scatter per-segment PAAMeasurements into one ``(n_mp, ...)`` result.

    ``sub_results`` is ``[(idx, PAAMeasurement)]`` -- each segment's headline
    and trajectories are laid into the portfolio arrays at its rows and
    zero-padded on the right to the portfolio's longest horizon (a contract
    carries no LRC past its coverage period). Unlike the GMM stitch the PAA
    holds the LRC undiscounted, so there is no per-MP discount curve to lay --
    the scatter is a pure ragged zero-pad. The mixed-portfolio orchestrator
    (``fcf.portfolio.measure``) uses this to combine a PAA partition that spans
    several routing segments into one ``PAAMeasurement``.
    """
    n_time = max(m.lrc_path.shape[1] - 1 for _, m in sub_results)

    lrc = np.empty(n_mp)
    loss_component = np.empty(n_mp)
    fcf = np.empty(n_mp)
    lrc_path = np.zeros((n_mp, n_time + 1))
    revenue = np.zeros((n_mp, n_time))
    service_expense = np.zeros((n_mp, n_time))
    lic = np.zeros((n_mp, n_time + 1))

    cf_2d = ("inforce", "deaths", "premium_cf", "claim_cf", "morbidity_cf",
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
        lic[idx, :t + 1] = m.lic
        _carry_lic_residual(lic, idx, t, n_time, m.lic)
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
    return PAAMeasurement(
        lrc=lrc, loss_component=loss_component, fcf=fcf,
        lrc_path=lrc_path, revenue=revenue, service_expense=service_expense,
        lic=lic, cashflows=cashflows,
    )


def measure_paa(
    model_points: ModelPoints,
    basis: Basis,
    *,
    revenue_basis: str = "time",
    full: bool = True,
) -> PAAMeasurement:
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
    """
    if revenue_basis not in ("time", "claims"):
        raise ValueError(
            f"revenue_basis must be 'time' or 'claims', got {revenue_basis!r}")
    basis = _single_basis(basis, entry="measure_paa")
    proj = project_cashflows(model_points, basis)

    # Onerous test -- the GMM inception fulfilment cash flows. Needed by both
    # paths and independent of the LRC roll, so it comes first; the headline
    # path returns right after it.
    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _rollforward_kernel(
        proj.claim_cf, proj.morbidity_cf, proj.disability_cf, proj.expense_cf,
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
        return PAAMeasurement(
            lrc=np.zeros(model_points.n_mp), loss_component=loss_component,
            fcf=fcf, model_points=model_points)

    # Full path only -- the (n_mp, n_time) revenue / service-expense arrays the
    # headline never needs (kept below the early return so the headline path
    # does not allocate them).
    premium_total = proj.premium_cf.sum(axis=1)          # (n_mp,)
    service_expense = proj.claim_cf + proj.morbidity_cf + proj.expense_cf

    # Liability for incurred claims -- claims incurred build it up, claims
    # paid (spread over the settlement pattern) run it off. Held undiscounted.
    incurred = proj.claim_cf + proj.morbidity_cf
    if basis.settlement_pattern is None:
        lic = np.zeros((incurred.shape[0], incurred.shape[1] + 1))
    else:
        lic = _settlement_lic(incurred, basis.settlement_pattern)

    # Insurance revenue -- total premium allocated across the periods of
    # service (Sec. B126), so total revenue equals total premium.
    # B126(a) straight-line weight -- a flat in-coverage mask. Used for the
    # 'time' basis and as the 'claims' fallback when a contract has no claims
    # pattern (B126(b) -> B126(a)); the fallback used the decaying in-force
    # before, which is neither basis.
    in_coverage = (np.arange(proj.n_time)[None, :]
                   < model_points.term_months[:, None]).astype(np.float64)
    if revenue_basis == "time":
        weight = in_coverage
    else:                                                # "claims" (validated above)
        weight = service_expense.copy()                  # B126(b)
        empty = weight.sum(axis=1) == 0.0                # no pattern -> B126(a)
        weight[empty] = in_coverage[empty]
    weight_sum = weight.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)   # safe divide; weight=0 → revenue=0
    revenue = premium_total[:, None] * weight / weight_sum

    # LRC roll-forward -- premiums build it up, revenue releases it.
    lrc_delta = proj.premium_cf - revenue
    n_mp, n_time = lrc_delta.shape
    lrc = np.zeros((n_mp, n_time + 1))
    lrc[:, 1:] = np.cumsum(lrc_delta, axis=1)

    return PAAMeasurement(
        lrc=lrc[:, 0],
        loss_component=loss_component,
        fcf=fcf,
        lrc_path=lrc,
        revenue=revenue,
        service_expense=service_expense,
        lic=lic,
        cashflows=proj,
        model_points=model_points,
    )


def measure_aggregate(
    model_points: ModelPoints,
    basis: Basis,
    *,
    revenue_basis: str = "time",
    chunk_size: int = 200_000,
) -> PAAAggregate:
    """Portfolio-aggregate PAA measurement in bounded memory.

    The PAA analogue of :func:`fastcashflow.gmm.measure_aggregate`: the LRC,
    revenue, service expense and LIC are additive across contracts, so the
    portfolio's run-off is the per-model-point trajectories summed over the
    model-point axis. Runs ``measure_paa(..., full=True)`` over row-blocks of
    ``chunk_size`` model points and accumulates only the ``(n_time+1,)`` /
    ``(n_time,)`` sums, so peak memory is ``O(chunk_size x n_time)`` regardless
    of ``n_mp`` (the PAA has no fused kernel -- a block still materialises dense
    transients, so chunking is the memory lever).

    Returns a :class:`PAAAggregate` (scalar LRC / loss-component totals + the
    aggregate ``lrc_path`` / ``revenue`` / ``service_expense`` / ``lic``). It is
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
    lic = np.zeros(n_time + 1)
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
        lic[:nt1] += m.lic.sum(axis=0)
        lrc += float(m.lrc.sum())
        loss += float(m.loss_component.sum())
    return PAAAggregate(
        lrc=lrc, loss_component=loss, lrc_path=lrc_path, revenue=revenue,
        service_expense=service_expense, lic=lic)


def measure_inforce(
    model_points: ModelPoints,
    state: "InforceState",
    basis: Basis,
    *,
    revenue_basis: str = "time",
    full: bool = True,
) -> PAAMeasurement:
    """In-force subsequent measurement of a PAA book at the valuation date.

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
    re-test on remaining coverage (Sec. 57-58) is deferred to
    :func:`~fastcashflow.roll_forward` / a later phase, so ``loss_component`` is
    zero and ``fcf`` is ``None`` here -- the same defer-the-unlocking stance as
    ``gmm.measure_inforce``.

    ``full=True`` (default) keeps the inception-to-horizon LRC / revenue /
    service-expense / LIC trajectories for movement analysis, with the headline
    ``lrc`` the as-of valuation-date value; ``full=False`` returns just the
    headline ``lrc``.

    Limitations (v1): because ``fcf`` is ``None`` (the onerous re-test is
    deferred), :func:`~fastcashflow.group_of_contracts` on a PAA in-force result
    raises -- the re-floor has no fulfilment-cash-flow input. And the in-force
    family is still partial: ``vfa.measure_inforce`` and
    ``portfolio.measure_inforce`` are not yet available (the VFA in-force needs an
    observed account value the current :class:`~fastcashflow.InforceState` does
    not carry).
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
    rescale = _inforce_rescale(m, model_points, em, rows)
    lrc = m.lrc_path[rows, em] * rescale
    # Subsequent onerous recognition is deferred (see docstring): zero loss, no
    # fcf re-test -- roll_forward / a later phase performs the unlocking.
    loss = np.zeros(n_mp, dtype=np.float64)
    if not full:
        return PAAMeasurement(
            lrc=lrc, loss_component=loss, model_points=model_points)
    # Trajectory fields keep the full inception-to-horizon paths (as the GMM
    # in-force does); only the headline lrc is the as-of, re-based slice.
    return PAAMeasurement(
        lrc=lrc, loss_component=loss, fcf=None,
        lrc_path=m.lrc_path, revenue=m.revenue,
        service_expense=m.service_expense, lic=m.lic,
        cashflows=m.cashflows, model_points=model_points)
