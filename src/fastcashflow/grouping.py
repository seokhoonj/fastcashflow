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
from fastcashflow.numerics import _csm_roll
from fastcashflow.projection import Cashflows


def _sum_by_group(arr: FloatArray, inverse: IntArray, n_groups: int) -> FloatArray:
    """Sum the rows of ``arr`` within each group."""
    result = np.zeros((n_groups, *arr.shape[1:]), dtype=np.float64)
    np.add.at(result, inverse, arr)
    return result


def group(measurement: GMMMeasurement, group_ids: FloatArray) -> GMMMeasurement:
    """Aggregate a per-model-point GMM measurement to IFRS 17 groups.

    ``group_ids`` assigns each model point to a group -- the IFRS 17 unit of
    account (portfolio x annual cohort x profitability bucket). BEL and RA
    are summed within each group; the CSM and the loss component are
    re-derived on the group aggregate, so the ``max(0, ...)`` floor nets the
    contracts within a group but not across groups.

    Returns a measurement whose rows are the groups, in ascending order of
    group id -- usable in turn by :func:`~fastcashflow.roll_forward`,
    :func:`~fastcashflow.reconcile` and :func:`~fastcashflow.report`.
    """
    if measurement.bel_path is None:
        raise ValueError(
            "group() requires a full=True measurement; the trajectory fields "
            "are None on the full=False fast path. Call measure(..., full=True)."
        )
    group_ids = np.asarray(group_ids)
    n_mp = measurement.bel_path.shape[0]
    if group_ids.shape != (n_mp,):
        raise ValueError(
            f"group_ids must have one entry per model point ({n_mp})"
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
