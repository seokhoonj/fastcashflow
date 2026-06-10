"""IFRS 17 level of aggregation -- grouping contracts into the unit of account.

IFRS 17 measures insurance contracts not one by one but in *groups* -- the
unit of account (paragraphs 14-24): a portfolio of contracts subject to
similar risks and managed together, divided into annual cohorts (issued no
more than a year apart) and then by profitability (onerous at inception, no
significant possibility of becoming onerous, and the rest).

The grouping is load-bearing for the CSM. The contractual service margin
cannot be negative, and that floor applies to the *group*: contracts within
a group are netted before the floor, contracts in different groups are not.
So a profitable contract's margin absorbs a slightly onerous one's loss
only when they share a group.

``group`` takes a per-model-point measurement and a group assignment and
re-expresses it at the group level -- BEL and RA summed, the CSM and loss
component re-derived on the group aggregate. The result is itself a
measurement, its rows the groups, so it flows on into ``roll_forward``,
``reconcile`` and ``report``.

The group assignment is the user's to make: the portfolio and the annual
cohort are known contract attributes, and a per-model-point measurement's
``loss_component`` flags the contracts that are onerous standalone.
"""
from __future__ import annotations

from functools import singledispatch

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.curves import forward_rates
from fastcashflow._paa import PAAMeasurement
from fastcashflow._reinsurance import ReinsuranceMeasurement
from fastcashflow._vfa import VFAMeasurement, _require_settlement_csm
from fastcashflow.engine import GMMMeasurement, _require_full
from fastcashflow.numerics import _csm_kernel, _csm_roll
from fastcashflow.projection import Cashflows


# In-force floor for the segmented discount-curve check: a month counts as live
# only above this, so a numerical residual past maturity is not read as a live
# month. Legitimate in-force is orders of magnitude larger.
_INFORCE_EPS = 1e-12


class _GroupReducer:
    """Sum rows within each group -- the grouping structure built once, reused.

    ``group()`` sums ~14 arrays (BEL, RA, every cash-flow stream, LIC, ...) over
    the *same* grouping. Building the reduction structure once here and reusing
    it for every :meth:`sum` avoids rebuilding the one-hot / re-sorting per array
    (the dominant cost at portfolio scale). One of two vectorised paths, chosen
    once from the size:

    * **few groups** -- a one-hot ``(n_groups, n) @ arr`` matrix multiply (a
      single BLAS call, no per-element scatter). Skipped when the one-hot would
      be large (``n_groups x n`` elements).
    * **many groups** -- sort once and reduce contiguous runs
      (``np.add.reduceat``), so the one-hot is never materialised.

    Empty groups stay zero. Sums run in group / sorted order rather than input
    order, so the result matches an unbuffered scatter-add to round-off.
    """

    def __init__(self, inverse: IntArray, n_groups: int):
        self.inverse = inverse
        self.n_groups = n_groups
        self.n = inverse.shape[0]
        # The number of model points per group -- also the grouping counts the
        # reduceat path needs, so compute the single bincount here.
        self.sizes = np.bincount(inverse, minlength=n_groups)
        if self.n and n_groups * self.n <= 20_000_000:
            self._onehot = (
                np.arange(n_groups)[:, None] == inverse[None, :]
            ).astype(np.float64)
            self._order = self._starts = self._nonempty = None
        else:
            self._onehot = None
            self._nonempty = np.nonzero(self.sizes)[0]
            self._order = np.argsort(inverse, kind="stable")
            self._starts = np.concatenate(([0], np.cumsum(self.sizes)[:-1]))

    def sum(self, arr: FloatArray) -> FloatArray:
        """Sum the rows of ``arr`` within each group -- shape ``(n_groups, ...)``."""
        if self.n == 0:
            return np.zeros((self.n_groups, *arr.shape[1:]), dtype=np.float64)
        if self._onehot is not None:
            return self._onehot @ arr
        result = np.zeros((self.n_groups, *arr.shape[1:]), dtype=np.float64)
        result[self._nonempty] = np.add.reduceat(
            arr[self._order], self._starts[self._nonempty], axis=0
        )
        return result


def _join_keys(cols, names=None) -> np.ndarray:
    """Composite ``'|'``-joined label per row, rejecting ``'|'`` in any value.

    The separator must round-trip, so a value carrying ``'|'`` would collide two
    distinct axis tuples onto one label and silently merge groups -- the same
    guard the segmented routing applies. Each axis is converted to a string
    column once (``astype(str)``) and joined vectorised with ``np.char.add``; the
    ``'|'`` guard runs only on string-like axes, since a numeric axis (e.g. an
    integer ``issue_year``) can never carry the separator.
    """
    str_cols = []
    for i, col in enumerate(cols):
        col = np.asarray(col)
        s = col.astype(str)
        if col.dtype.kind in "OUS":      # object / unicode / bytes -- can carry '|'
            bad = sorted(set(s[np.char.find(s, "|") >= 0].tolist()))
            if bad:
                where = f" in axis {names[i]!r}" if names else ""
                raise ValueError(
                    f"group key value(s) {bad}{where} contain the '|' character, "
                    "which grouping uses as the key separator -- rename the value "
                    "or change the separator upstream."
                )
        str_cols.append(s)
    out = str_cols[0]
    for s in str_cols[1:]:
        out = np.char.add(np.char.add(out, "|"), s)
    return out.astype(object)


def _resolve_group_ids(measurement: GMMMeasurement, by) -> np.ndarray:
    """Build the per-MP group label array from ``by``.

    ``by`` is one of: a single axis **name**; a **list** of axis names and/or
    precomputed ``(n_mp,)`` label arrays (joined into one composite label); or a
    single precomputed label **array**. Names are resolved via
    :meth:`ModelPoints.axis` against the model points
    :func:`~fastcashflow.gmm.measure` stamped on the measurement.
    """
    def axis(name: str) -> np.ndarray:
        mp = measurement.model_points
        if mp is None:
            raise ValueError(
                f"group(by={name!r}) needs the model points to resolve the name "
                "-- use a measurement returned by measure() (which stamps them), "
                "or pass a precomputed label array instead of a name."
            )
        return np.asarray(mp.axis(name))

    if isinstance(by, str):
        return axis(by)
    if isinstance(by, (list, tuple)):
        cols = [axis(b) if isinstance(b, str) else np.asarray(b) for b in by]
        names = [b if isinstance(b, str) else None for b in by]
        if len(cols) == 1:
            return cols[0]
        return _join_keys(cols, names)
    return np.asarray(by)


def _group_plan(measurement, by, n_mp: int):
    """Resolve ``by`` to per-MP labels, then build the shared reduction plan.

    Returns ``(labels, reducer)`` where ``labels`` is the ascending-order
    per-group composite label and ``reducer`` is the :class:`_GroupReducer`
    every per-field sum in this ``group`` call reuses.
    """
    group_ids = _resolve_group_ids(measurement, by)
    if group_ids.shape != (n_mp,):
        raise ValueError(f"group ids must have one entry per model point ({n_mp})")
    labels, inverse = np.unique(group_ids, return_inverse=True)
    # numpy >= 2.0 can return a 2-D inverse for n-D input; flatten to (n_mp,).
    return labels, _GroupReducer(inverse.reshape(-1), labels.shape[0])


def _sum_cashflows(cf: Cashflows, reducer: _GroupReducer) -> Cashflows:
    """Sum every cash-flow stream within each group (all streams are additive)."""
    return Cashflows(
        inforce=reducer.sum(cf.inforce),
        deaths=reducer.sum(cf.deaths),
        premium_cf=reducer.sum(cf.premium_cf),
        claim_cf=reducer.sum(cf.claim_cf),
        morbidity_cf=reducer.sum(cf.morbidity_cf),
        expense_cf=reducer.sum(cf.expense_cf),
        annuity_cf=reducer.sum(cf.annuity_cf),
        disability_cf=reducer.sum(cf.disability_cf),
        maturity_cf=reducer.sum(cf.maturity_cf),
        maturity_survivors=reducer.sum(cf.maturity_survivors),
        surrender_cf=reducer.sum(cf.surrender_cf),
    )


@singledispatch
def group(measurement, by):
    """Aggregate a per-model-point measurement to any axis.

    A general aggregation primitive -- not IFRS 17-specific. ``by`` is one of:

    * a single **axis name** (e.g. ``"product"``);
    * a **list** of axis names and/or precomputed ``(n_mp,)`` label arrays
      (e.g. ``["product", "issue_year"]``, or
      ``["product", onerous_array]``), joined into one composite label;
    * a single precomputed ``(n_mp,)`` **array** of group labels.

    Names are resolved per model point via :meth:`ModelPoints.axis` against the
    model points the measure stamped on the result, so no re-passing is needed;
    a computed axis with no source column (e.g. an onerous flag from
    ``loss_component``) is passed as an array instead -- an ``np.ndarray``, since
    a Python *list* is read as a list of axes, not a single label vector.

    BEL and RA are summed within each group; the CSM and the loss component are
    re-derived on the group aggregate, so the ``max(0, ...)`` floor nets the
    contracts within a group but not across groups. The IFRS 17 unit of account
    (portfolio x annual cohort x profitability) is one choice of axes --
    :func:`group_of_contracts` is the preset for it; management-accounting,
    profitability and validation views are other choices of ``by``.

    Dispatches on the measurement type (``GMMMeasurement``, ``VFAMeasurement``,
    ``ReinsuranceMeasurement``, ``PAAMeasurement``). A
    :class:`~fastcashflow.portfolio.PortfolioMeasurement` (the mixed-model
    container) is also accepted: each model slot is grouped on its own native
    measurement and a :class:`~fastcashflow.portfolio.PortfolioGroups` is
    returned (a precomputed array ``by`` is subset to each slot's rows).
    Returns a measurement of the same type whose rows are the groups, in
    ascending label order -- usable in turn by
    :func:`~fastcashflow.roll_forward`, :func:`~fastcashflow.reconcile` and
    :func:`~fastcashflow.report`. Its ``group_labels`` attribute carries the
    composite label of each row, so a caller can map a group back to its key
    (e.g. ``"|"``-split a :func:`group_of_contracts` label into portfolio /
    cohort / profitability) without rebuilding the keys; ``group_sizes`` carries
    the number of model points in each group (model-point rows, not the policy
    count -- they differ when a model point's ``count`` stands for several
    policies).
    """
    raise TypeError(
        f"group is not implemented for {type(measurement).__name__}; supported: "
        "GMMMeasurement, VFAMeasurement, ReinsuranceMeasurement, PAAMeasurement."
    )


def _per_group_bom(bom, inforce, reducer, labels):
    """Per-group representative discount curve for a segmented (2-D) measurement.

    Each model point discounts on its own curve, and a group must sit in one
    curve. The curves are padded to the portfolio's longest horizon -- a flat
    tail past each contract's maturity -- so two contracts on the *same* curve
    with different terms have different tails. Compare each row only over its
    live horizon (where it is still in force; the padded tail discounts zero
    in-force and never reaches the CSM), and represent the group by its
    longest-horizon row so the discounting is correct for every contract's whole
    term. Raises if a group mixes genuinely different curves.

    Returns ``(out_bom, reps)`` -- the ``(n_groups, n_time+1)`` per-group curve
    and the representative row index per group, so a caller with a companion
    per-MP curve (e.g. GMM's ``discount_mid``) can index it the same way. Shared
    by the GMM and VFA grouping, both of which now see 2-D curves from the
    portfolio orchestrator's segment stitch.
    """
    cols = np.arange(bom.shape[1])
    # Live = still in force. A small floor (not exact > 0) so a numerical
    # residual past maturity is not read as a live month, which would extend the
    # compared horizon into the padded tail and falsely reject the group.
    # Legitimate in-force is orders of magnitude above this floor.
    live = np.where(inforce > _INFORCE_EPS,
                    np.arange(inforce.shape[1])[None, :], -1).max(axis=1)
    out_bom = np.empty((reducer.n_groups, bom.shape[1]))
    reps = np.empty(reducer.n_groups, dtype=np.int64)
    # group -> its row indices, from a single sort rather than a full
    # ``inverse == g`` scan per group (which would be O(n_groups x n_mp)).
    group_rows = np.split(np.argsort(reducer.inverse, kind="stable"),
                          np.cumsum(reducer.sizes)[:-1])
    for g in range(reducer.n_groups):
        rows = group_rows[g]
        rep = rows[np.argmax(live[rows])]
        livemask = cols[None, :] < (live[rows] + 2)[:, None]
        if not np.allclose(np.where(livemask, bom[rows] - bom[rep], 0.0), 0.0):
            raise ValueError(
                f"group {labels[g]!r} mixes model points with different "
                "discount curves -- a group must sit in one portfolio "
                "(basis). Split it by basis before grouping."
            )
        out_bom[g] = bom[rep]
        reps[g] = rep
    return out_bom, reps


def _finalise_gmm_group(bel, ra, grouped_cf, lic, out_bom, out_mid,
                        labels, sizes) -> GMMMeasurement:
    """Build a grouped GMMMeasurement from already-summed group aggregates.

    The tail shared by the in-memory :func:`group` and the chunked per-group
    aggregate (``fcf.portfolio.measure_group_of_contracts``): given the within-group sums of
    BEL / RA / cash flows / LIC and the per-group representative discount curve,
    re-derive the CSM and loss component on the group aggregate -- the
    ``max(0, ...)`` floor applies to the group, not the contract. ``bel`` / ``ra``
    are ``(n_groups, n_time+1)`` trajectories. ``out_bom`` may be 1-D (a single
    basis) or 2-D per-group (segmented / chunked); ``_csm_roll`` dispatches on its
    ndim. Sharing this function is what makes the chunked aggregate reproduce the
    in-memory grouping byte for byte.
    """
    fcf0 = bel[:, 0] + ra[:, 0]
    csm0 = np.maximum(0.0, -fcf0)
    loss_component = np.maximum(0.0, fcf0)
    monthly_rate = forward_rates(out_bom)
    csm, csm_accretion, csm_release = _csm_roll(
        csm0, np.ascontiguousarray(grouped_cf.inforce), monthly_rate
    )
    return GMMMeasurement(
        bel=bel[:, 0],
        ra=ra[:, 0],
        csm=csm[:, 0],
        loss_component=loss_component,
        bel_path=bel,
        ra_path=ra,
        csm_path=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        lic=lic,
        cashflows=grouped_cf,
        discount_bom=out_bom,
        discount_mid=out_mid,
        group_labels=labels,
        group_sizes=sizes,
    )


@group.register
def _(measurement: GMMMeasurement, by) -> GMMMeasurement:
    _require_full(measurement, "group()")
    labels, reducer = _group_plan(measurement, by, measurement.bel_path.shape[0])
    bel = reducer.sum(measurement.bel_path)
    ra = reducer.sum(measurement.ra_path)
    grouped_cf = _sum_cashflows(measurement.cashflows, reducer)
    lic = reducer.sum(measurement.lic)

    # The discount curve is per-group: a segmented result carries a 2-D per-MP
    # curve, so reconcile each group to one curve; a single basis a 1-D one.
    bom = measurement.discount_bom
    if bom.ndim == 2:
        out_bom, reps = _per_group_bom(
            bom, measurement.cashflows.inforce, reducer, labels)
        out_mid = measurement.discount_mid[reps]
    else:
        out_bom, out_mid = bom, measurement.discount_mid
    return _finalise_gmm_group(
        bel, ra, grouped_cf, lic, out_bom, out_mid, labels, reducer.sizes)


def _finalise_vfa_group(bel, ra, grouped_cf, lic, time_value, variable_fee,
                        out_bom, labels, sizes) -> VFAMeasurement:
    """Build a grouped VFAMeasurement from already-summed group aggregates.

    The VFA analogue of :func:`_finalise_gmm_group`, shared by :func:`group` and
    the chunked per-group aggregate. The inception fulfilment cash flows fold in
    the guarantee time value, and the CSM and loss component are re-derived on the
    group aggregate -- the ``max(0, ...)`` floor applies to the group, not the
    contract (so a grouped CSM differs from a sum of per-contract floors when the
    group mixes profitable and onerous contracts). The CSM accretes at the
    underlying-items return (``out_bom``), released by coverage units.
    ``account_value`` is a per-policy level, not a group quantity, so it does not
    carry to the grouped result.
    """
    fcf0 = bel[:, 0] + ra[:, 0] + time_value
    csm0 = np.maximum(0.0, -fcf0)
    loss_component = np.maximum(0.0, fcf0)
    monthly_rate = forward_rates(out_bom)
    csm, csm_accretion, csm_release = _csm_roll(
        csm0, np.ascontiguousarray(grouped_cf.inforce), monthly_rate
    )
    return VFAMeasurement(
        bel=bel[:, 0],
        ra=ra[:, 0],
        csm=csm[:, 0],
        variable_fee=variable_fee,
        time_value=time_value,
        loss_component=loss_component,
        bel_path=bel,
        ra_path=ra,
        csm_path=csm,
        account_value_path=None,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        lic=lic,
        cashflows=grouped_cf,
        discount_bom=out_bom,
        model_points=None,
        group_labels=labels,
        group_sizes=sizes,
    )


@group.register
def _(measurement: VFAMeasurement, by) -> VFAMeasurement:
    _require_settlement_csm(measurement, "group")
    if measurement.bel_path is None:
        raise ValueError(
            "group() requires a full measurement; the trajectory fields are "
            "None. Re-run vfa.measure()."
        )
    labels, reducer = _group_plan(measurement, by, measurement.bel_path.shape[0])
    bel = reducer.sum(measurement.bel_path)
    ra = reducer.sum(measurement.ra_path)
    grouped_cf = _sum_cashflows(measurement.cashflows, reducer)
    lic = reducer.sum(measurement.lic)
    # variable_fee (PV of the fee) and time_value (a cost) are per-MP amounts --
    # additive.
    time_value = reducer.sum(measurement.time_value)
    variable_fee = reducer.sum(measurement.variable_fee)

    bom = measurement.discount_bom
    if bom.ndim == 2:
        # Portfolio-stitched: each segment discounts at its own underlying-items
        # return, so a group must sit in one curve (the same reconciliation the
        # GMM grouping does).
        out_bom, _ = _per_group_bom(
            bom, measurement.cashflows.inforce, reducer, labels)
    else:
        out_bom = bom
    return _finalise_vfa_group(
        bel, ra, grouped_cf, lic, time_value, variable_fee, out_bom,
        labels, reducer.sizes)


@group.register
def _(measurement: ReinsuranceMeasurement, by) -> ReinsuranceMeasurement:
    if measurement.cashflows is None or measurement.discount_bom is None:
        raise ValueError(
            "group() requires a full reinsurance measurement (cash flows and "
            "discount curve). Re-run reinsurance.measure()."
        )
    labels, reducer = _group_plan(measurement, by, measurement.bel.shape[0])
    bel = reducer.sum(measurement.bel)
    ra = reducer.sum(measurement.ra)
    recovery = reducer.sum(measurement.recovery)
    reinsurance_premium = reducer.sum(measurement.reinsurance_premium)
    grouped_cf = _sum_cashflows(measurement.cashflows, reducer)
    # Reinsurance held has no loss component and no floor (paragraph 65): the
    # CSM is the net cost or gain, csm0 = -(BEL - RA). That is linear, so the
    # grouped CSM equals the sum of the per-contract CSMs; only the accretion /
    # release trajectory changes, re-derived at the single discount curve and
    # released by the grouped coverage units.
    csm0 = -(bel - ra)
    bom = measurement.discount_bom
    monthly_rate = forward_rates(bom)
    csm, csm_accretion, csm_release = _csm_kernel(
        csm0, np.ascontiguousarray(grouped_cf.inforce), monthly_rate
    )
    return ReinsuranceMeasurement(
        bel=bel,
        ra=ra,
        csm=csm[:, 0],
        bel_path=reducer.sum(measurement.bel_path),
        ra_path=reducer.sum(measurement.ra_path),
        csm_path=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        recovery=recovery,
        reinsurance_premium=reinsurance_premium,
        cashflows=grouped_cf,
        discount_bom=bom,
        model_points=None,
        group_labels=labels,
        group_sizes=reducer.sizes,
    )


def _finalise_paa_group(lrc_path, revenue, service_expense, lic, fcf,
                        grouped_cf, labels, sizes) -> PAAMeasurement:
    """Build a grouped PAAMeasurement from already-summed group aggregates.

    The PAA analogue of :func:`_finalise_gmm_group`, shared by :func:`group` and
    the chunked per-group aggregate. The LRC, revenue, service expense and LIC are
    undiscounted and additive -- there is no CSM (paragraphs 53-59). The only
    non-linear part is the onerous loss (paragraph 57): ``loss_component =
    max(0, fcf)`` on the group's aggregate fulfilment cash flows, so a profitable
    contract nets a marginally onerous one within the group.
    """
    loss_component = np.maximum(0.0, fcf)
    return PAAMeasurement(
        lrc=lrc_path[:, 0],
        loss_component=loss_component,
        fcf=fcf,
        lrc_path=lrc_path,
        revenue=revenue,
        service_expense=service_expense,
        lic=lic,
        cashflows=grouped_cf,
        model_points=None,
        group_labels=labels,
        group_sizes=sizes,
    )


@group.register
def _(measurement: PAAMeasurement, by) -> PAAMeasurement:
    if measurement.lrc_path is None or measurement.fcf is None:
        raise ValueError(
            "group() requires a full PAA measurement; the trajectory fields are "
            "None. Re-run paa.measure()."
        )
    labels, reducer = _group_plan(measurement, by, measurement.lrc_path.shape[0])
    lrc_path = reducer.sum(measurement.lrc_path)
    revenue = reducer.sum(measurement.revenue)
    service_expense = reducer.sum(measurement.service_expense)
    lic = reducer.sum(measurement.lic)
    grouped_cf = _sum_cashflows(measurement.cashflows, reducer)
    fcf = reducer.sum(measurement.fcf)
    return _finalise_paa_group(
        lrc_path, revenue, service_expense, lic, fcf, grouped_cf,
        labels, reducer.sizes)


@singledispatch
def group_of_contracts(measurement, *, portfolio: str = "product",
                       cohort: str = "issue_year",
                       profitability=None) -> GMMMeasurement:
    """Aggregate a measurement to the IFRS 17 group of insurance contracts.

    The unit of account (paragraphs 14-24) is a portfolio (14) x annual cohort
    (22) x profitability (16). This preset builds that grouping from the model
    points :func:`~fastcashflow.gmm.measure` stamped on the measurement and runs
    :func:`group`, so the CSM floor nets within a group but not across.

    Dispatches on the measurement type; the profitability axis differs by type
    (a new measurement registers with ``@group_of_contracts.register``):

    * ``GMMMeasurement`` / ``VFAMeasurement`` / ``PAAMeasurement`` -- insurance
      contracts issued, direct-participating, and short-coverage (PAA)
      contracts; profitability is the onerous / remaining split (paragraph 16,
      and 57 for the PAA). The per-type re-derivation differs (VFA accretes the
      CSM at the underlying-items return; the PAA has no CSM, only the LRC and
      the onerous loss), handled by :func:`group`'s own dispatch.
    * ``ReinsuranceMeasurement`` -- reinsurance contracts held; profitability is
      the net-gain split (paragraph 61, ``csm > 0``), and there is no loss
      component or floor (paragraph 65), so the grouped CSM is the sum of the
      contract CSMs.
    * :class:`~fastcashflow.portfolio.PortfolioMeasurement` -- the mixed-model
      container; each model slot is grouped on its own native measurement and a
      :class:`~fastcashflow.portfolio.PortfolioGroups` is returned. For a book too
      large to hold the full per-model-point measurement, use the chunked
      :func:`fastcashflow.portfolio.measure_group_of_contracts` instead.

    Arguments (keyword-only):

    * ``portfolio`` -- the column naming the portfolio axis (default
      ``"product"``: paragraph 14's product line). Pass another column name
      to group on a different portfolio definition.
    * ``cohort`` -- the column naming the annual-cohort axis (default
      ``"issue_year"``, derived from ``issue_date``: paragraph 22). Pass another
      column (e.g. ``"issue_quarter"`` carried in the data) for a finer cohort;
      paragraph 22 caps the span at one year, so a cohort may be finer than
      annual but not coarser.
    * ``profitability`` -- the profitability classification. ``None`` (default)
      derives it from the measurement, since it is an output, not a known input
      (paragraph 16 / 47's net-outflow test). Pass a precomputed ``(n_mp,)``
      array for a custom split (e.g. the paragraph-16 three-way split using a
      CSM-vs-RA threshold), or a column name for a locked classification carried
      in the data (paragraph 24: the group is fixed at inception).

    Requires a ``full=True`` measurement.
    """
    raise TypeError(
        "group_of_contracts is not implemented for "
        f"{type(measurement).__name__}; supported: GMMMeasurement, "
        "VFAMeasurement, ReinsuranceMeasurement, PAAMeasurement."
    )


def _portfolio_cohort(measurement, portfolio, cohort):
    """Resolve the portfolio and annual-cohort label arrays (shared by all presets)."""
    mp = measurement.model_points
    if mp is None:
        raise ValueError(
            "group_of_contracts needs the model points -- use a measurement "
            "returned by measure() (which stamps them)."
        )
    portfolio_arr = mp.axis(portfolio)
    # cohort: issue_year (from issue_date) by default. With the default left in
    # place but no issue_date set, fall back to a single cohort -- all new
    # business sits within one year (paragraph 22). An explicit cohort column
    # that is missing is a typo, so let its KeyError propagate.
    if cohort == "issue_year":
        try:
            cohort_arr = mp.axis("issue_year")
        except KeyError:
            cohort_arr = np.zeros(mp.n_mp, dtype=np.int64)
    else:
        cohort_arr = mp.axis(cohort)
    return mp, portfolio_arr, cohort_arr


def _resolve_profitability(mp, profitability, default):
    """profitability override: a column name, a custom array, or ``None`` -> default.

    ``default`` is the engine-derived split (an output, not a known input). A
    string names a stored (locked, paragraph 24) classification; an array is a
    custom split (e.g. the paragraph-16 three-way split).
    """
    if profitability is None:
        return default
    if isinstance(profitability, str):
        return mp.axis(profitability)
    return np.asarray(profitability)


def _group_of_contracts_onerous(measurement, *, portfolio="product",
                                cohort="issue_year", profitability=None):
    """Shared GMM / VFA / PAA preset -- profitability is the onerous split.

    Insurance contracts issued (GMM), direct-participating contracts (VFA) and
    short-coverage contracts (PAA) use the same onerous / remaining
    classification, derived from the measurement's ``loss_component``
    (paragraph 16, and 57 for the PAA); only ``group``'s per-type re-derivation
    differs.
    """
    if isinstance(measurement, VFAMeasurement):
        _require_settlement_csm(measurement, "group_of_contracts")
    mp, portfolio_arr, cohort_arr = _portfolio_cohort(measurement, portfolio, cohort)
    default = np.where(measurement.loss_component > 0.0, "onerous", "remaining")
    prof = _resolve_profitability(mp, profitability, default)
    return group(measurement, [portfolio_arr, cohort_arr, prof])


group_of_contracts.register(GMMMeasurement, _group_of_contracts_onerous)
group_of_contracts.register(VFAMeasurement, _group_of_contracts_onerous)
group_of_contracts.register(PAAMeasurement, _group_of_contracts_onerous)


@group_of_contracts.register
def _(measurement: ReinsuranceMeasurement, *, portfolio: str = "product",
      cohort: str = "issue_year", profitability=None) -> ReinsuranceMeasurement:
    # Reinsurance held replaces the onerous test with a net gain at initial
    # recognition (paragraph 61). The CSM is the net cost (negative) or net gain
    # (positive), so csm > 0 is the net-gain group.
    mp, portfolio_arr, cohort_arr = _portfolio_cohort(measurement, portfolio, cohort)
    default = np.where(measurement.csm > 0.0, "net_gain", "no_net_gain")
    prof = _resolve_profitability(mp, profitability, default)
    return group(measurement, [portfolio_arr, cohort_arr, prof])
