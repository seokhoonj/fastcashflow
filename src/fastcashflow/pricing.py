"""Premium solving -- pricing on the level-premium term product.

Fulfilment cash flows are linear in the premium: claims, expenses and the
in-force run-off do not depend on it, so ``FCF = A - premium * B``. Two
valuations pin down ``A`` and ``B``, and the premium that meets a
profitability target then has a closed form -- no iteration.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis
from fastcashflow.engine import measure
from fastcashflow.modelpoints import ModelPoints


def _with_premium(model_points: ModelPoints, premium: float) -> ModelPoints:
    """A copy of ``model_points`` with every level premium set to ``premium``.

    Every other field -- including the payment frequency -- is carried over
    unchanged, so the two valuations that pin down the premium see the same
    contract bar the premium itself.
    """
    return replace(
        model_points, premium=np.full(model_points.n_mp, premium)
    )


def solve_premium(
    model_points: ModelPoints,
    basis: Basis,
    *,
    break_even: bool = False,
    margin: float | None = None,
    csm: float | None = None,
) -> FloatArray:
    """Solve the level premium that meets a profitability target.

    Exactly one target must be given:

    * ``break_even`` -- the lowest non-onerous premium (FCF = 0, zero CSM).
    * ``margin``     -- a profit margin, ``CSM / PV(premiums) = margin``
      (e.g. ``0.10`` for 10%); must satisfy ``0 <= margin < 1``.
    * ``csm``        -- an absolute target CSM (profit) per model point.

    Every product field of ``model_points`` is used as given -- only
    ``premium`` is ignored, since it is the unknown being solved for.
    Returns the solved premium per model point, shape ``(n_mp,)``.
    """
    chosen = (break_even, margin is not None, csm is not None)
    if sum(chosen) != 1:
        raise ValueError(
            "specify exactly one target: break_even, margin or csm"
        )
    if margin is not None and not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must be in [0, 1), got {margin}")

    # FCF is linear in the premium -- FCF = A - premium * B -- so two
    # valuations (premium 0 and 1) pin the line down exactly.
    at_zero = measure(_with_premium(model_points, 0.0), basis, full=False)
    at_one = measure(_with_premium(model_points, 1.0), basis, full=False)
    a = at_zero.bel + at_zero.ra
    b = a - (at_one.bel + at_one.ra)

    zero_sens = np.abs(b) < 1e-12
    if np.any(zero_sens):
        raise ValueError(
            "solve_premium: FCF is insensitive to the premium for "
            f"{int(zero_sens.sum())} model point(s) -- cannot solve. "
            "Check that premium enters the cash flows (non-zero "
            "premium term and payment frequency)."
        )

    if break_even:
        return a / b
    if margin is not None:
        return a / (b * (1.0 - margin))
    return (csm + a) / b
