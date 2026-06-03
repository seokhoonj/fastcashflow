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

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.engine import GMMMeasurement
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.numerics import _csm_roll
from fastcashflow.projection import Cashflows


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


def _resolve_group_ids(measurement: GMMMeasurement, by,
                       model_points: ModelPoints | None) -> np.ndarray:
    """Build the per-MP group label array from ``by`` -- axis names or an array."""
    if isinstance(by, (list, tuple)) and all(isinstance(b, str) for b in by):
        mp = model_points if model_points is not None else measurement.model_points
        if mp is None:
            raise ValueError(
                "group(by=[axis names]) needs the model points to resolve the "
                "names -- pass model_points=, or use a measurement returned by "
                "measure() (which stamps them)."
            )
        cols = [np.asarray(mp.axis(name), dtype=object) for name in by]
        return _join_keys(cols, by)
    return np.asarray(by)


def group(measurement: GMMMeasurement, by, *,
          model_points: "ModelPoints | None" = None) -> GMMMeasurement:
    """Aggregate a per-model-point GMM measurement to any axis.

    A general aggregation primitive -- not IFRS 17-specific. ``by`` is either:

    * a list of **axis names** (e.g. ``["product_code", "channel"]``,
      ``["product_code", "issue_year", "profitability_group"]``), resolved per
      model point via :meth:`ModelPoints.axis` against ``model_points`` -- or
      the model points :func:`~fastcashflow.gmm.measure` stamped on the
      measurement, so no re-passing is needed; or
    * a precomputed ``(n_mp,)`` array of group labels.

    BEL and RA are summed within each group; the CSM and the loss component are
    re-derived on the group aggregate, so the ``max(0, ...)`` floor nets the
    contracts within a group but not across groups. The IFRS 17 unit of account
    (portfolio x annual cohort x profitability) is one choice of axes --
    :func:`group_into_gic` is the preset for it; management-accounting,
    profitability and validation views are other choices of ``by``.

    Returns a measurement whose rows are the groups, in ascending label order --
    usable in turn by :func:`~fastcashflow.roll_forward`,
    :func:`~fastcashflow.reconcile` and :func:`~fastcashflow.report`.
    """
    if measurement.bel_path is None:
        raise ValueError(
            "group() requires a full=True measurement; the trajectory fields "
            "are None on the full=False fast path. Call measure(..., full=True)."
        )
    group_ids = _resolve_group_ids(measurement, by, model_points)
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
        # Segmented: each group must sit in one portfolio (basis), so its model
        # points share a discount curve. Take each group's curve (verifying it
        # is uniform) and re-derive the group CSM at that rate.
        out_bom = np.empty((n_groups, bom.shape[1]))
        out_mid = np.empty((n_groups, measurement.discount_mid.shape[1]))
        for g in range(n_groups):
            rows = np.nonzero(inverse == g)[0]
            if not np.allclose(bom[rows], bom[rows[0]]):
                raise ValueError(
                    f"group {labels[g]!r} mixes model points with different "
                    "discount curves -- a group must sit in one portfolio "
                    "(basis). Split it by basis before grouping."
                )
            out_bom[g] = bom[rows[0]]
            out_mid[g] = measurement.discount_mid[rows[0]]
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


def assign_gic(portfolio: np.ndarray, cohort: np.ndarray,
               profitability: np.ndarray) -> np.ndarray:
    """Build IFRS 17 group-of-insurance-contracts (GIC) labels from three axes.

    The IFRS 17 unit of account (paragraphs 14-24) is a portfolio (14) x annual
    cohort (22) x profitability group (16). This combines the three
    per-model-point label arrays into one ``(n_mp,)`` composite label,
    ``"{portfolio}|{cohort}|{profitability}"``, ready for :func:`group`.

    The three axes are the entity's to determine -- fastcashflow does not set
    the IFRS 17 grouping policy, it only aggregates on the labels given:

    * ``portfolio`` -- usually the product line (paragraph 14: contracts in the
      same product line generally share a portfolio when managed together).
    * ``cohort`` -- the year of issue (paragraph 22: contracts issued more than
      a year apart cannot share a group).
    * ``profitability`` -- the company's profitability classification. The
      labels are free: a two-way ``"onerous"`` / ``"remaining"`` split is the
      common starting point; the paragraph 16 three-way split (adding
      ``"no_significant_possibility"``) is the same call with more labels. A
      quick onerous split needs no extra input --
      ``np.where(measurement.loss_component > 0, "onerous", "remaining")``,
      since the per-model-point ``loss_component`` already flags the contracts
      that are onerous standalone.
    """
    portfolio = np.asarray(portfolio, dtype=object)
    cohort = np.asarray(cohort, dtype=object)
    profitability = np.asarray(profitability, dtype=object)
    if not (portfolio.ndim == cohort.ndim == profitability.ndim == 1
            and portfolio.shape == cohort.shape == profitability.shape):
        raise ValueError(
            "portfolio, cohort and profitability must be 1-D arrays of the "
            f"same length; got {portfolio.shape}, {cohort.shape}, "
            f"{profitability.shape}"
        )
    return _join_keys((portfolio, cohort, profitability),
                      ("portfolio", "cohort", "profitability"))


def group_into_gic(measurement: GMMMeasurement,
                   model_points: ModelPoints | None = None, *,
                   portfolio: str = "portfolio_id", cohort: str = "issue_date",
                   profitability: str = "profitability_group") -> GMMMeasurement:
    """IFRS 17 grouping preset over :func:`group`.

    Builds the IFRS 17 group of insurance contracts (GIC) -- portfolio x annual
    cohort x profitability (paragraphs 14/22/16) -- from the model points'
    source attributes and re-expresses the measurement at the group level, so
    the CSM floor nets within a GIC but not across. The model points are taken
    from ``model_points`` or, if omitted, the ones :func:`~fastcashflow.gmm.measure`
    stamped on the measurement.

    Each axis is a column name with an IFRS 17 default rule:

    * ``portfolio`` -- the ``portfolio_id`` column if the model points carry it,
      else ``product_code`` (paragraph 14: the product line is the usual
      portfolio).
    * ``cohort`` -- the issue year, derived from ``issue_date`` (paragraph 22);
      a single cohort (all new business) when no ``issue_date`` is set.
    * ``profitability`` -- the ``profitability_group`` column if present, else a
      two-way onerous / remaining split from the measurement's own
      ``loss_component`` (paragraph 47's net-outflow test, at the individual
      contract level of paragraph 17). Labels are free, so a paragraph-16
      three-way split is just a column with three values.

    Requires a ``full=True`` measurement. Each GIC must sit in one portfolio
    (one discount curve), which :func:`group` enforces.
    """
    mp = model_points if model_points is not None else measurement.model_points
    if mp is None:
        raise ValueError(
            "group_into_gic needs the model points -- pass model_points=, or use "
            "a measurement returned by measure() (which stamps them)."
        )
    n = measurement.bel.shape[0]
    # The IFRS 17 default rules fire only when the axis is left at its default
    # column name -- an explicit (non-default) name that is missing is a typo,
    # so let its KeyError propagate rather than silently falling back.
    # portfolio: portfolio_id if the model points carry it, else product_code.
    if portfolio == "portfolio_id":
        try:
            portfolio_arr = mp.axis("portfolio_id")
        except KeyError:
            portfolio_arr = mp.axis("product_code")
    else:
        portfolio_arr = mp.axis(portfolio)
    # cohort: issue_year from issue_date if available, else a single cohort.
    if cohort == "issue_date":
        try:
            cohort_arr = mp.axis("issue_year")
        except KeyError:
            cohort_arr = np.zeros(n, dtype=np.int64)
    else:
        cohort_arr = mp.axis(cohort)
    # profitability: the named column if present, else derive from loss_component.
    if profitability == "profitability_group":
        try:
            profitability_arr = mp.axis("profitability_group")
        except KeyError:
            profitability_arr = np.where(
                measurement.loss_component > 0.0, "onerous", "remaining"
            )
    else:
        profitability_arr = mp.axis(profitability)
    return group(
        measurement, assign_gic(portfolio_arr, cohort_arr, profitability_arr)
    )
