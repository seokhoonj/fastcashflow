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
from fastcashflow._reinsurance import ReinsuranceMeasurement
from fastcashflow._vfa import VFAMeasurement
from fastcashflow.engine import GMMMeasurement
from fastcashflow.numerics import _csm_kernel, _csm_roll
from fastcashflow.projection import Cashflows


# In-force floor for the segmented discount-curve check: a month counts as live
# only above this, so a numerical residual past maturity is not read as a live
# month. Legitimate in-force is orders of magnitude larger.
_INFORCE_EPS = 1e-12


def _sum_by_group(arr: FloatArray, inverse: IntArray, n_groups: int) -> FloatArray:
    """Sum the rows of ``arr`` within each group.

    Two vectorised paths instead of the slow unbuffered ``np.add.at`` scatter
    that dominated ``group()`` (~14s for 200k model points):

    * **few groups** -- a one-hot ``(n_groups, n) @ arr`` matrix multiply, a
      single BLAS call with no per-element scatter (~10x faster). Skipped when
      the one-hot would be large (``n_groups x n`` elements).
    * **many groups** -- sort once and reduce contiguous runs
      (``np.add.reduceat``), so the one-hot is never materialised.

    Empty groups stay zero. Sums run in group / sorted order rather than input
    order, so the result matches the scatter-add to floating-point round-off.
    """
    n = arr.shape[0]
    if n == 0:
        return np.zeros((n_groups, *arr.shape[1:]), dtype=np.float64)
    if n_groups * n <= 20_000_000:
        onehot = (np.arange(n_groups)[:, None] == inverse[None, :]).astype(np.float64)
        return onehot @ arr
    result = np.zeros((n_groups, *arr.shape[1:]), dtype=np.float64)
    counts = np.bincount(inverse, minlength=n_groups)
    nonempty = np.nonzero(counts)[0]
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    result[nonempty] = np.add.reduceat(arr[order], starts[nonempty], axis=0)
    return result


def _join_keys(cols, names=None) -> np.ndarray:
    """Composite ``'|'``-joined label per row, rejecting ``'|'`` in any value.

    The separator must round-trip, so a value carrying ``'|'`` would collide two
    distinct axis tuples onto one label and silently merge groups -- the same
    guard the segmented routing applies. ``str`` keeps non-string axes (e.g. an
    integer ``issue_year``); ``zip(*cols)`` is the fast per-row path.
    """
    for i, col in enumerate(cols):
        bad = sorted({str(v) for v in col if "|" in str(v)})
        if bad:
            where = f" in axis {names[i]!r}" if names else ""
            raise ValueError(
                f"group key value(s) {bad}{where} contain the '|' character, "
                "which grouping uses as the key separator -- rename the value "
                "or change the separator upstream."
            )
    return np.array(["|".join(map(str, row)) for row in zip(*cols)], dtype=object)


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


@singledispatch
def group(measurement, by):
    """Aggregate a per-model-point measurement to any axis.

    A general aggregation primitive -- not IFRS 17-specific. ``by`` is one of:

    * a single **axis name** (e.g. ``"product_code"``);
    * a **list** of axis names and/or precomputed ``(n_mp,)`` label arrays
      (e.g. ``["product_code", "issue_year"]``, or
      ``["product_code", onerous_array]``), joined into one composite label;
    * a single precomputed ``(n_mp,)`` **array** of group labels.

    Names are resolved per model point via :meth:`ModelPoints.axis` against the
    model points the measure stamped on the result, so no re-passing is needed;
    a computed axis with no source column (e.g. an onerous flag from
    ``loss_component``) is passed as an array instead.

    BEL and RA are summed within each group; the CSM and the loss component are
    re-derived on the group aggregate, so the ``max(0, ...)`` floor nets the
    contracts within a group but not across groups. The IFRS 17 unit of account
    (portfolio x annual cohort x profitability) is one choice of axes --
    :func:`group_of_contracts` is the preset for it; management-accounting,
    profitability and validation views are other choices of ``by``.

    Dispatches on the measurement type (``GMMMeasurement``, ``VFAMeasurement``,
    ``ReinsuranceMeasurement``).
    Returns a measurement of the same type whose rows are the groups, in
    ascending label order -- usable in turn by
    :func:`~fastcashflow.roll_forward`, :func:`~fastcashflow.reconcile` and
    :func:`~fastcashflow.report`.
    """
    raise TypeError(
        f"group is not implemented for {type(measurement).__name__}; "
        "supported: GMMMeasurement, VFAMeasurement, ReinsuranceMeasurement."
    )


@group.register
def _(measurement: GMMMeasurement, by) -> GMMMeasurement:
    if measurement.bel_path is None:
        raise ValueError(
            "group() requires a full=True measurement; the trajectory fields "
            "are None on the full=False fast path. Call measure(..., full=True)."
        )
    group_ids = _resolve_group_ids(measurement, by)
    n_mp = measurement.bel_path.shape[0]
    if group_ids.shape != (n_mp,):
        raise ValueError(
            f"group ids must have one entry per model point ({n_mp})"
        )
    labels, inverse = np.unique(group_ids, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_groups = labels.shape[0]

    bel = _sum_by_group(measurement.bel_path, inverse, n_groups)
    ra = _sum_by_group(measurement.ra_path, inverse, n_groups)
    cf = measurement.cashflows
    grouped_cf = Cashflows(
        inforce=_sum_by_group(cf.inforce, inverse, n_groups),
        deaths=_sum_by_group(cf.deaths, inverse, n_groups),
        premium_cf=_sum_by_group(cf.premium_cf, inverse, n_groups),
        claim_cf=_sum_by_group(cf.claim_cf, inverse, n_groups),
        morbidity_cf=_sum_by_group(cf.morbidity_cf, inverse, n_groups),
        expense_cf=_sum_by_group(cf.expense_cf, inverse, n_groups),
        annuity_cf=_sum_by_group(cf.annuity_cf, inverse, n_groups),
        disability_cf=_sum_by_group(cf.disability_cf, inverse, n_groups),
        maturity_cf=_sum_by_group(cf.maturity_cf, inverse, n_groups),
        maturity_survivors=_sum_by_group(cf.maturity_survivors, inverse, n_groups),
        surrender_cf=_sum_by_group(cf.surrender_cf, inverse, n_groups),
    )

    # The CSM and the loss component are re-derived on the group aggregate --
    # the max(0, ...) floor applies to the group, not the contract.
    fcf0 = bel[:, 0] + ra[:, 0]
    csm0 = np.maximum(0.0, -fcf0)
    loss_component = np.maximum(0.0, fcf0)
    bom = measurement.discount_bom
    if bom.ndim == 2:
        # Segmented: each model point discounts on its own curve, and a group
        # must sit in one curve. But the curves are padded to the portfolio's
        # longest horizon -- a flat tail past each contract's maturity -- so two
        # contracts on the *same* curve with different terms have different
        # tails. Compare each row only over its live horizon (where it is still
        # in force; the padded tail discounts zero in-force and never reaches the
        # CSM), and represent the group by its longest-horizon curve so the
        # discounting is correct for every contract's whole term.
        cols = np.arange(bom.shape[1])
        inforce = measurement.cashflows.inforce
        # Live = still in force. A small floor (not exact > 0) so a numerical
        # residual past maturity is not read as a live month, which would extend
        # the compared horizon into the padded tail and falsely reject the group.
        # Legitimate in-force is orders of magnitude above this floor.
        live = np.where(inforce > _INFORCE_EPS,
                        np.arange(inforce.shape[1])[None, :], -1).max(axis=1)
        out_bom = np.empty((n_groups, bom.shape[1]))
        out_mid = np.empty((n_groups, measurement.discount_mid.shape[1]))
        for g in range(n_groups):
            rows = np.nonzero(inverse == g)[0]
            rep = rows[np.argmax(live[rows])]
            livemask = cols[None, :] < (live[rows] + 2)[:, None]
            if not np.allclose(np.where(livemask, bom[rows] - bom[rep], 0.0), 0.0):
                raise ValueError(
                    f"group {labels[g]!r} mixes model points with different "
                    "discount curves -- a group must sit in one portfolio "
                    "(basis). Split it by basis before grouping."
                )
            out_bom[g] = bom[rep]
            out_mid[g] = measurement.discount_mid[rep]
        monthly_rate = out_bom[:, :-1] / out_bom[:, 1:] - 1.0
    else:
        out_bom, out_mid = bom, measurement.discount_mid
        monthly_rate = bom[:-1] / bom[1:] - 1.0
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
        lic=_sum_by_group(measurement.lic, inverse, n_groups),
        cashflows=grouped_cf,
        discount_bom=out_bom,
        discount_mid=out_mid,
    )


@group.register
def _(measurement: VFAMeasurement, by) -> VFAMeasurement:
    if measurement.bel_path is None:
        raise ValueError(
            "group() requires a full measurement; the trajectory fields are "
            "None. Re-run vfa.measure()."
        )
    group_ids = _resolve_group_ids(measurement, by)
    n_mp = measurement.bel_path.shape[0]
    if group_ids.shape != (n_mp,):
        raise ValueError(f"group ids must have one entry per model point ({n_mp})")
    labels, inverse = np.unique(group_ids, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_groups = labels.shape[0]

    bel = _sum_by_group(measurement.bel_path, inverse, n_groups)
    ra = _sum_by_group(measurement.ra_path, inverse, n_groups)
    cf = measurement.cashflows
    grouped_cf = Cashflows(
        inforce=_sum_by_group(cf.inforce, inverse, n_groups),
        deaths=_sum_by_group(cf.deaths, inverse, n_groups),
        premium_cf=_sum_by_group(cf.premium_cf, inverse, n_groups),
        claim_cf=_sum_by_group(cf.claim_cf, inverse, n_groups),
        morbidity_cf=_sum_by_group(cf.morbidity_cf, inverse, n_groups),
        expense_cf=_sum_by_group(cf.expense_cf, inverse, n_groups),
        annuity_cf=_sum_by_group(cf.annuity_cf, inverse, n_groups),
        disability_cf=_sum_by_group(cf.disability_cf, inverse, n_groups),
        maturity_cf=_sum_by_group(cf.maturity_cf, inverse, n_groups),
        maturity_survivors=_sum_by_group(cf.maturity_survivors, inverse, n_groups),
        surrender_cf=_sum_by_group(cf.surrender_cf, inverse, n_groups),
    )
    # variable_fee (PV of the fee) and time_value (a cost) are per-MP amounts --
    # additive. account_value is a per-policy level, not a group quantity (the
    # group's fund would be sum(inforce x av), a different field), so it does not
    # carry to the grouped result.
    time_value = _sum_by_group(measurement.time_value, inverse, n_groups)
    variable_fee = _sum_by_group(measurement.variable_fee, inverse, n_groups)

    # The CSM and loss component are re-derived on the group aggregate. The VFA
    # inception fulfilment cash flows fold in the guarantee time value, and the
    # CSM accretes at the underlying-items return (the single VFA curve, so no
    # per-MP curve to reconcile), released by coverage units.
    fcf0 = bel[:, 0] + ra[:, 0] + time_value
    csm0 = np.maximum(0.0, -fcf0)
    loss_component = np.maximum(0.0, fcf0)
    bom = measurement.discount_bom
    monthly_rate = bom[:-1] / bom[1:] - 1.0
    csm, csm_accretion, csm_release = _csm_kernel(
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
        lic=_sum_by_group(measurement.lic, inverse, n_groups),
        cashflows=grouped_cf,
        discount_bom=bom,
        model_points=None,
    )


@group.register
def _(measurement: ReinsuranceMeasurement, by) -> ReinsuranceMeasurement:
    if measurement.cashflows is None or measurement.discount_bom is None:
        raise ValueError(
            "group() requires a full reinsurance measurement (cash flows and "
            "discount curve). Re-run reinsurance.measure()."
        )
    group_ids = _resolve_group_ids(measurement, by)
    n_mp = measurement.bel.shape[0]
    if group_ids.shape != (n_mp,):
        raise ValueError(f"group ids must have one entry per model point ({n_mp})")
    labels, inverse = np.unique(group_ids, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_groups = labels.shape[0]

    bel = _sum_by_group(measurement.bel, inverse, n_groups)
    ra = _sum_by_group(measurement.ra, inverse, n_groups)
    recovery = _sum_by_group(measurement.recovery, inverse, n_groups)
    reinsurance_premium = _sum_by_group(
        measurement.reinsurance_premium, inverse, n_groups
    )
    cf = measurement.cashflows
    grouped_cf = Cashflows(
        inforce=_sum_by_group(cf.inforce, inverse, n_groups),
        deaths=_sum_by_group(cf.deaths, inverse, n_groups),
        premium_cf=_sum_by_group(cf.premium_cf, inverse, n_groups),
        claim_cf=_sum_by_group(cf.claim_cf, inverse, n_groups),
        morbidity_cf=_sum_by_group(cf.morbidity_cf, inverse, n_groups),
        expense_cf=_sum_by_group(cf.expense_cf, inverse, n_groups),
        annuity_cf=_sum_by_group(cf.annuity_cf, inverse, n_groups),
        disability_cf=_sum_by_group(cf.disability_cf, inverse, n_groups),
        maturity_cf=_sum_by_group(cf.maturity_cf, inverse, n_groups),
        maturity_survivors=_sum_by_group(cf.maturity_survivors, inverse, n_groups),
        surrender_cf=_sum_by_group(cf.surrender_cf, inverse, n_groups),
    )
    # Reinsurance held has no loss component and no floor (paragraph 65): the
    # CSM is the net cost or gain, csm0 = -(BEL - RA). That is linear, so the
    # grouped CSM equals the sum of the per-contract CSMs; only the accretion /
    # release trajectory changes, re-derived at the single discount curve and
    # released by the grouped coverage units.
    csm0 = -(bel - ra)
    bom = measurement.discount_bom
    monthly_rate = bom[:-1] / bom[1:] - 1.0
    csm, csm_accretion, csm_release = _csm_kernel(
        csm0, np.ascontiguousarray(grouped_cf.inforce), monthly_rate
    )
    return ReinsuranceMeasurement(
        bel=bel,
        ra=ra,
        csm=csm[:, 0],
        csm_path=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        recovery=recovery,
        reinsurance_premium=reinsurance_premium,
        cashflows=grouped_cf,
        discount_bom=bom,
        model_points=None,
    )


@singledispatch
def group_of_contracts(measurement, *, portfolio: str = "product_code",
                       cohort: str = "issue_year",
                       profitability=None) -> GMMMeasurement:
    """Aggregate a measurement to the IFRS 17 group of insurance contracts.

    The unit of account (paragraphs 14-24) is a portfolio (14) x annual cohort
    (22) x profitability (16). This preset builds that grouping from the model
    points :func:`~fastcashflow.gmm.measure` stamped on the measurement and runs
    :func:`group`, so the CSM floor nets within a group but not across.

    Dispatches on the measurement type; the profitability axis differs by type
    (a new measurement registers with ``@group_of_contracts.register``):

    * ``GMMMeasurement`` / ``VFAMeasurement`` -- insurance contracts issued and
      direct-participating contracts; profitability is the onerous / remaining
      split (paragraph 16). The CSM re-derivation differs (VFA accretes at the
      underlying-items return), handled by :func:`group`'s own dispatch.
    * ``ReinsuranceMeasurement`` -- reinsurance contracts held; profitability is
      the net-gain split (paragraph 61, ``csm > 0``), and there is no loss
      component or floor (paragraph 65), so the grouped CSM is the sum of the
      contract CSMs.

    Arguments (keyword-only):

    * ``portfolio`` -- the column naming the portfolio axis (default
      ``"product_code"``: paragraph 14's product line). Pass another column name
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
        "VFAMeasurement, ReinsuranceMeasurement."
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
            cohort_arr = np.zeros(measurement.bel.shape[0], dtype=np.int64)
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


def _group_of_contracts_onerous(measurement, *, portfolio="product_code",
                                cohort="issue_year", profitability=None):
    """Shared GMM / VFA preset -- profitability is the paragraph-16 onerous split.

    Insurance contracts issued (GMM) and direct-participating contracts (VFA)
    use the same onerous / remaining classification, derived from the
    measurement's ``loss_component``; only ``group``'s per-type CSM
    re-derivation differs.
    """
    mp, portfolio_arr, cohort_arr = _portfolio_cohort(measurement, portfolio, cohort)
    default = np.where(measurement.loss_component > 0.0, "onerous", "remaining")
    prof = _resolve_profitability(mp, profitability, default)
    return group(measurement, [portfolio_arr, cohort_arr, prof])


group_of_contracts.register(GMMMeasurement, _group_of_contracts_onerous)
group_of_contracts.register(VFAMeasurement, _group_of_contracts_onerous)


@group_of_contracts.register
def _(measurement: ReinsuranceMeasurement, *, portfolio: str = "product_code",
      cohort: str = "issue_year", profitability=None) -> ReinsuranceMeasurement:
    # Reinsurance held replaces the onerous test with a net gain at initial
    # recognition (paragraph 61). The CSM is the net cost (negative) or net gain
    # (positive), so csm > 0 is the net-gain group.
    mp, portfolio_arr, cohort_arr = _portfolio_cohort(measurement, portfolio, cohort)
    default = np.where(measurement.csm > 0.0, "net_gain", "no_net_gain")
    prof = _resolve_profitability(mp, profitability, default)
    return group(measurement, [portfolio_arr, cohort_arr, prof])
