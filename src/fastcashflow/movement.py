"""Period-close roll-forward -- the IFRS 17 analysis of change.

A reporting period's movement bridges the opening insurance contract
liability to the closing one and decomposes the change into its drivers --
the analysis of change (AoC). This is the step from a measurement
calculator towards a reporting engine.

``roll_forward`` takes a GMM :class:`~fastcashflow.Measurement` -- a
projection from one assumption set -- and slices its trajectory into
reporting periods, reconciling each period's opening and closing BEL, RA
and CSM on the *expected* basis: interest accretion at the locked-in rate
and the expected release of cash flows and of the CSM.

This is the expected-basis skeleton. Experience variance (actual cash
flows and in-force differing from expected) and the effect of assumption
changes -- the other two drivers of the movement -- are added on top of
this structure in later phases.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastcashflow._typing import FloatArray
from fastcashflow.engine import Measurement


@dataclass(frozen=True, slots=True)
class PeriodMovement:
    """One reporting period's analysis of change, on the expected basis.

    The period covers months ``[month_start, month_end)``. Every balance and
    movement array is ``(n_mp,)``, and each block reconciles exactly::

        bel_opening + bel_interest  - bel_release   == bel_closing
        ra_opening  + ra_interest   - ra_release    == ra_closing
        csm_opening + csm_accretion - csm_release   == csm_closing

    ``*_interest`` and ``csm_accretion`` are the unwind of discount at the
    locked-in rate; ``*_release`` is the expected run-off over the period --
    cash flows for the BEL and RA, services provided for the CSM.
    """

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


def roll_forward(
    measurement: Measurement, period_months: int = 12
) -> list[PeriodMovement]:
    """Slice a GMM measurement into reporting-period movements.

    Returns one :class:`PeriodMovement` per reporting period of
    ``period_months`` months, reconciling the opening and closing BEL, RA
    and CSM on the expected basis. Consecutive periods chain -- each
    period's closing balances are the next period's opening balances -- and
    a horizon that is not a whole number of periods gives a shorter final
    period.
    """
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")

    bel, ra, csm = measurement.bel, measurement.ra, measurement.csm
    csm_accretion = measurement.csm_accretion
    csm_release = measurement.csm_release
    n_time = bel.shape[1] - 1
    discount_start = measurement.discount_start
    monthly_rate = discount_start[:-1] / discount_start[1:] - 1.0   # (n_time,)

    movements: list[PeriodMovement] = []
    for a in range(0, n_time, period_months):
        b = min(a + period_months, n_time)
        bel_interest = (bel[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        ra_interest = (ra[:, a:b] * monthly_rate[a:b]).sum(axis=1)
        movements.append(PeriodMovement(
            month_start=a,
            month_end=b,
            bel_opening=bel[:, a],
            bel_interest=bel_interest,
            bel_release=bel[:, a] + bel_interest - bel[:, b],
            bel_closing=bel[:, b],
            ra_opening=ra[:, a],
            ra_interest=ra_interest,
            ra_release=ra[:, a] + ra_interest - ra[:, b],
            ra_closing=ra[:, b],
            csm_opening=csm[:, a],
            csm_accretion=csm_accretion[:, a:b].sum(axis=1),
            csm_release=csm_release[:, a:b].sum(axis=1),
            csm_closing=csm[:, b],
        ))
    return movements
