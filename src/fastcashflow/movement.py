"""Period-close roll-forward -- the IFRS 17 analysis of change.

A reporting period's movement bridges the opening insurance contract
liability to the closing one and decomposes the change into its drivers --
the analysis of change (AoC). This is the step from a measurement
calculator towards a reporting engine.

``roll_forward`` slices a GMM :class:`~fastcashflow.Measurement` into
reporting periods, reconciling each period's opening and closing BEL, RA
and CSM. It models all three drivers of the movement:

* the expected unwind -- interest accretion at the locked-in rate, and the
  expected release of cash flows and of the CSM;
* an assumption revision -- a change in the estimate of future cash flows;
* in-force experience -- the actual in-force at the period end differing
  from what was projected.

The latter two both relate to future service, so each adjusts the CSM
(floored at zero; any excess falls into the loss component) rather than
profit or loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.engine import Measurement
from fastcashflow.gmm import _csm_kernel


@dataclass(frozen=True, slots=True)
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


def roll_forward(
    measurement: Measurement,
    period_months: int = 12,
    *,
    revised: Measurement | None = None,
    revised_at: int | None = None,
    actual_inforce: FloatArray | None = None,
    experience_at: int | None = None,
) -> list[PeriodMovement]:
    """Slice a GMM measurement into reporting-period movements.

    Returns one :class:`PeriodMovement` per reporting period of
    ``period_months`` months, reconciling the opening and closing BEL, RA
    and CSM. Consecutive periods chain, and a horizon that is not a whole
    number of periods gives a shorter final period.

    An assumption revision is recognised by passing ``revised`` -- a second
    measurement of the same book under updated assumptions -- and
    ``revised_at``, the month it takes effect. In-force experience is
    recognised by passing ``actual_inforce`` -- the ``(n_mp,)`` in-force
    actually remaining at the period end -- and ``experience_at``, that
    month. The revision month and the experience month must be positive
    multiples of ``period_months``. Either change adjusts the CSM by the
    resulting change in fulfilment cash flows (floored at zero, any excess
    falling into the loss component); v1 recognises one or the other, not
    both in a single call.
    """
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    n_time = measurement.bel.shape[1] - 1
    n_mp = measurement.bel.shape[0]
    if (revised is None) != (revised_at is None):
        raise ValueError("pass revised and revised_at together, or neither")
    if (actual_inforce is None) != (experience_at is None):
        raise ValueError("pass actual_inforce and experience_at together, or neither")
    if revised is not None and actual_inforce is not None:
        raise ValueError(
            "v1 recognises an assumption revision or in-force experience, "
            "not both in a single call"
        )

    discount_start = measurement.discount_start
    monthly_rate = discount_start[:-1] / discount_start[1:] - 1.0   # (n_time,)
    zero = np.zeros(n_mp)

    bel, ra, csm = measurement.bel, measurement.ra, measurement.csm
    csm_accretion = measurement.csm_accretion
    csm_release = measurement.csm_release
    change_at: int | None = None
    change_kind = ""
    post_bel = post_ra = csm_after = None
    loss = zero

    if revised is not None:
        if revised.bel.shape != measurement.bel.shape:
            raise ValueError("revised must measure the same book as measurement")
        change_at, change_kind = revised_at, "assumption"
        post_bel, post_ra = revised.bel, revised.ra
        post_inforce = revised.cashflows.inforce
    elif actual_inforce is not None:
        actual_inforce = np.asarray(actual_inforce, dtype=np.float64)
        if actual_inforce.shape != (n_mp,):
            raise ValueError(f"actual_inforce must have shape ({n_mp},)")
        change_at, change_kind = experience_at, "experience"
        expected = measurement.cashflows.inforce[:, experience_at]
        safe = np.where(expected > 1e-12, expected, 1.0)
        # In-force experience scales the remaining contract: the future
        # projection uses the same assumptions, so the closing FCF scales
        # linearly with the in-force actually remaining.
        ratio = np.where(expected > 1e-12, actual_inforce / safe, 1.0)
        post_bel = measurement.bel * ratio[:, None]
        post_ra = measurement.ra * ratio[:, None]
        post_inforce = measurement.cashflows.inforce

    if change_at is not None:
        k = change_at
        if k % period_months != 0 or not 0 < k < n_time:
            raise ValueError(
                "the change month must be a positive multiple of "
                f"period_months below the horizon ({n_time}), got {k}"
            )
        rate = 1.0 / discount_start[1] - 1.0
        delta_fcf = ((post_bel[:, k] + post_ra[:, k])
                     - (measurement.bel[:, k] + measurement.ra[:, k]))
        csm_before = measurement.csm[:, k]
        csm_after = np.maximum(0.0, csm_before - delta_fcf)
        loss = np.maximum(0.0, delta_fcf - csm_before)
        re_csm, re_acc, re_rel = _csm_kernel(
            csm_after, np.ascontiguousarray(post_inforce[:, k:]), rate
        )
        bel = np.concatenate([measurement.bel[:, :k + 1], post_bel[:, k + 1:]],
                             axis=1)
        ra = np.concatenate([measurement.ra[:, :k + 1], post_ra[:, k + 1:]],
                            axis=1)
        csm = np.concatenate([measurement.csm[:, :k + 1], re_csm[:, 1:]], axis=1)
        csm_accretion = np.concatenate(
            [measurement.csm_accretion[:, :k], re_acc], axis=1)
        csm_release = np.concatenate(
            [measurement.csm_release[:, :k], re_rel], axis=1)

    movements: list[PeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_open, ra_open, csm_open = bel[:, a], ra[:, a], csm[:, a]
        bel_ac = bel_ex = ra_ac = ra_ex = csm_ac = csm_ex = loss_line = zero
        bel_traj, ra_traj = bel, ra
        if change_at is not None and a == change_at:
            d_bel = post_bel[:, a] - bel_open
            d_ra = post_ra[:, a] - ra_open
            d_csm = csm_after - csm_open
            if change_kind == "assumption":
                bel_ac, ra_ac, csm_ac = d_bel, d_ra, d_csm
            else:
                bel_ex, ra_ex, csm_ex = d_bel, d_ra, d_csm
            loss_line = loss
            bel_traj, ra_traj = post_bel, post_ra
        bel_interest = (bel_traj[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        ra_interest = (ra_traj[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        movements.append(PeriodMovement(
            month_start=a,
            month_end=b,
            bel_opening=bel_open,
            bel_assumption_change=bel_ac,
            bel_experience=bel_ex,
            bel_interest=bel_interest,
            bel_release=bel_open + bel_ac + bel_ex + bel_interest - bel[:, b],
            bel_closing=bel[:, b],
            ra_opening=ra_open,
            ra_assumption_change=ra_ac,
            ra_experience=ra_ex,
            ra_interest=ra_interest,
            ra_release=ra_open + ra_ac + ra_ex + ra_interest - ra[:, b],
            ra_closing=ra[:, b],
            csm_opening=csm_open,
            csm_assumption_change=csm_ac,
            csm_experience=csm_ex,
            csm_accretion=csm_accretion[:, a:b].sum(axis=1),
            csm_release=csm_release[:, a:b].sum(axis=1),
            csm_closing=csm[:, b],
            loss_component_recognised=loss_line,
        ))
    return movements
