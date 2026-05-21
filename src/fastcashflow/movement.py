"""Period-close roll-forward -- the IFRS 17 analysis of change.

A reporting period's movement bridges the opening insurance contract
liability to the closing one and decomposes the change into its drivers --
the analysis of change (AoC). This is the step from a measurement
calculator towards a reporting engine.

``roll_forward`` slices a GMM :class:`~fastcashflow.Measurement` into
reporting periods, reconciling each period's opening and closing BEL, RA
and CSM. Two of the three movement drivers are modelled:

* the expected unwind -- interest accretion at the locked-in rate, and the
  expected release of cash flows and of the CSM;
* an assumption revision -- a change in the estimate of future cash flows.
  Relating to future service, it adjusts the CSM (floored at zero; any
  excess falls into the loss component) rather than profit or loss.

The third driver, experience variance (actual cash flows and in-force
differing from expected), is added on top of this structure in a later
phase.
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

        bel_opening + bel_assumption_change + bel_interest  - bel_release  == bel_closing
        ra_opening  + ra_assumption_change  + ra_interest   - ra_release   == ra_closing
        csm_opening + csm_assumption_change + csm_accretion - csm_release  == csm_closing

    ``*_interest`` / ``csm_accretion`` is the unwind of discount at the
    locked-in rate; ``*_release`` is the expected run-off over the period.
    ``*_assumption_change`` is the effect of revised assumptions -- non-zero
    only in the revision period -- a change in fulfilment cash flows for
    future service that adjusts the CSM rather than profit or loss.
    ``loss_from_assumption_change`` is the part of an unfavourable revision
    beyond the CSM, which falls into the loss component.
    """

    month_start: int
    month_end: int
    bel_opening: FloatArray
    bel_assumption_change: FloatArray
    bel_interest: FloatArray
    bel_release: FloatArray
    bel_closing: FloatArray
    ra_opening: FloatArray
    ra_assumption_change: FloatArray
    ra_interest: FloatArray
    ra_release: FloatArray
    ra_closing: FloatArray
    csm_opening: FloatArray
    csm_assumption_change: FloatArray
    csm_accretion: FloatArray
    csm_release: FloatArray
    csm_closing: FloatArray
    loss_from_assumption_change: FloatArray


def roll_forward(
    measurement: Measurement,
    period_months: int = 12,
    *,
    revised: Measurement | None = None,
    revised_at: int | None = None,
) -> list[PeriodMovement]:
    """Slice a GMM measurement into reporting-period movements.

    Returns one :class:`PeriodMovement` per reporting period of
    ``period_months`` months, reconciling the opening and closing BEL, RA
    and CSM. Consecutive periods chain, and a horizon that is not a whole
    number of periods gives a shorter final period.

    To recognise an assumption revision, pass ``revised`` -- a second
    measurement of the same book under the updated assumptions -- and
    ``revised_at``, the month the revision takes effect (a positive multiple
    of ``period_months``). The change in fulfilment cash flows adjusts the
    CSM (floored at zero; any excess falls into the loss component), and the
    periods from then on follow the revised projection.
    """
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    n_time = measurement.bel.shape[1] - 1
    if (revised is None) != (revised_at is None):
        raise ValueError("pass revised and revised_at together, or neither")
    if revised is not None:
        if revised.bel.shape != measurement.bel.shape:
            raise ValueError("revised must measure the same book as measurement")
        if revised_at % period_months != 0 or not 0 < revised_at < n_time:
            raise ValueError(
                "revised_at must be a positive multiple of period_months "
                f"below the horizon ({n_time}), got {revised_at}"
            )

    discount_start = measurement.discount_start
    monthly_rate = discount_start[:-1] / discount_start[1:] - 1.0   # (n_time,)
    n_mp = measurement.bel.shape[0]
    zero = np.zeros(n_mp)

    if revised is None:
        bel, ra, csm = measurement.bel, measurement.ra, measurement.csm
        csm_accretion = measurement.csm_accretion
        csm_release = measurement.csm_release
        csm_after = None
    else:
        k = revised_at
        rate = 1.0 / discount_start[1] - 1.0
        # Change in fulfilment cash flows from the revision -- it relates to
        # future service, so it adjusts the CSM (floored at zero).
        delta_fcf = ((revised.bel[:, k] + revised.ra[:, k])
                     - (measurement.bel[:, k] + measurement.ra[:, k]))
        csm_before = measurement.csm[:, k]
        csm_after = np.maximum(0.0, csm_before - delta_fcf)
        loss_from_change = np.maximum(0.0, delta_fcf - csm_before)
        # Re-roll the CSM from the revision over the revised coverage units.
        re_csm, re_acc, re_rel = _csm_kernel(
            csm_after,
            np.ascontiguousarray(revised.cashflows.inforce[:, k:]),
            rate,
        )
        # Spliced trajectories: original up to the revision month (so the
        # pre-revision period still closes on the old basis), revised after.
        bel = np.concatenate([measurement.bel[:, :k + 1], revised.bel[:, k + 1:]],
                             axis=1)
        ra = np.concatenate([measurement.ra[:, :k + 1], revised.ra[:, k + 1:]],
                            axis=1)
        csm = np.concatenate([measurement.csm[:, :k + 1], re_csm[:, 1:]], axis=1)
        csm_accretion = np.concatenate(
            [measurement.csm_accretion[:, :k], re_acc], axis=1)
        csm_release = np.concatenate(
            [measurement.csm_release[:, :k], re_rel], axis=1)

    movements: list[PeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        is_rev = revised is not None and a == revised_at
        if is_rev:
            # The revision period opens on the old basis; the assumption
            # change is a movement line, then the period unwinds revised.
            bel_open, ra_open, csm_open = bel[:, a], ra[:, a], csm[:, a]
            bel_ac = revised.bel[:, a] - bel_open
            ra_ac = revised.ra[:, a] - ra_open
            csm_ac = csm_after - csm_open
            loss_ac = loss_from_change
            bel_traj, ra_traj = revised.bel, revised.ra
        else:
            bel_open, ra_open, csm_open = bel[:, a], ra[:, a], csm[:, a]
            bel_ac = ra_ac = csm_ac = loss_ac = zero
            bel_traj, ra_traj = bel, ra
        bel_interest = (bel_traj[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        ra_interest = (ra_traj[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        movements.append(PeriodMovement(
            month_start=a,
            month_end=b,
            bel_opening=bel_open,
            bel_assumption_change=bel_ac,
            bel_interest=bel_interest,
            bel_release=bel_open + bel_ac + bel_interest - bel[:, b],
            bel_closing=bel[:, b],
            ra_opening=ra_open,
            ra_assumption_change=ra_ac,
            ra_interest=ra_interest,
            ra_release=ra_open + ra_ac + ra_interest - ra[:, b],
            ra_closing=ra[:, b],
            csm_opening=csm_open,
            csm_assumption_change=csm_ac,
            csm_accretion=csm_accretion[:, a:b].sum(axis=1),
            csm_release=csm_release[:, a:b].sum(axis=1),
            csm_closing=csm[:, b],
            loss_from_assumption_change=loss_ac,
        ))
    return movements
