"""IFRS 17 transition -- the fair value approach.

On first applying IFRS 17 an entity must measure its in-force contracts.
Where the full retrospective approach -- measuring as if the standard had
always applied, which is what ``measure`` does from inception -- is
impracticable, IFRS 17 permits the fair value approach (paragraphs
C20-C24): the CSM at the transition date is the fair value of the
contracts less their fulfilment cash flows.

``transition`` takes a measurement of the in-force book at the transition
date and a supplied fair value, and re-sets the CSM on that basis. Contract
modification and derecognition are not separate machinery here: a
modification that does not derecognise the contract (paragraph 73) is a
change in fulfilment cash flows for future service -- the assumption
revision of ``roll_forward`` -- and derecognition through lapse or
surrender is the in-force experience of ``roll_forward``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.engine import GMMMeasurement
from fastcashflow.numerics import _csm_roll


def transition(measurement: GMMMeasurement, fair_value: FloatArray) -> GMMMeasurement:
    """Re-set the CSM on the IFRS 17 fair value transition basis.

    ``measurement`` is a measurement of the in-force book at the transition
    date -- its inception column being that date. ``fair_value`` is the
    ``(n_mp,)`` fair value of each contract or group. The CSM becomes
    ``max(0, fair_value - fulfilment cash flows)``, any excess of the
    fulfilment cash flows over the fair value falling into the loss
    component, and is rolled forward from there.

    Returns a measurement with the re-set CSM and loss component; the BEL
    and RA are unchanged. It flows on into :func:`~fastcashflow.roll_forward`,
    :func:`~fastcashflow.reconcile` and :func:`~fastcashflow.report`.
    """
    if measurement.bel_path is None:
        raise ValueError(
            "transition() requires a full=True measurement; the trajectory "
            "fields are None on the full=False fast path. Call measure(..., full=True)."
        )
    fair_value = np.asarray(fair_value, dtype=np.float64)
    n_mp = measurement.bel_path.shape[0]
    if fair_value.shape != (n_mp,):
        raise ValueError(f"fair_value must have one entry per row ({n_mp})")

    fcf0 = measurement.bel_path[:, 0] + measurement.ra_path[:, 0]
    csm0 = np.maximum(0.0, fair_value - fcf0)
    loss_component = np.maximum(0.0, fcf0 - fair_value)
    # Per-month rate curve implied by the measurement's discount factors --
    # ratio of consecutive start-of-month factors. Carries the locked-in curve
    # even if it is non-flat. The last axis is time, so this is (n_time,) for a
    # single basis or (n_mp, n_time) for a segmented (per-row-curve) measurement.
    monthly_rate = (measurement.discount_bom[..., :-1]
                    / measurement.discount_bom[..., 1:]) - 1.0
    csm, csm_accretion, csm_release = _csm_roll(
        csm0, np.ascontiguousarray(measurement.cashflows.inforce), monthly_rate
    )
    return replace(
        measurement,
        csm=csm[:, 0],
        csm_path=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        loss_component=loss_component,
    )
