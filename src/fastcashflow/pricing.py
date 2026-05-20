"""Premium solving -- pricing on the level-premium term product.

Fulfilment cash flows are linear in the premium: claims, expenses and the
in-force run-off do not depend on it, so ``FCF = A - premium * B``. Two
valuations pin down ``A`` and ``B``, and the premium that meets a
profitability target then has a closed form -- no iteration.
"""
from __future__ import annotations

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.engine import value
from fastcashflow.modelpoint import ModelPointSet


def _with_premium(mps: ModelPointSet, premium: float) -> ModelPointSet:
    """A copy of ``mps`` with every monthly premium set to ``premium``."""
    return ModelPointSet(
        issue_age=mps.issue_age,
        monthly_premium=np.full(mps.n_mp, premium),
        term_months=mps.term_months,
        maturity_benefit=mps.maturity_benefit,
        annuity_payment=mps.annuity_payment,
        single_premium=mps.single_premium,
        cov_kind=mps.cov_kind,
        cov_amount=mps.cov_amount,
        cov_offset=mps.cov_offset,
    )


def solve_premium(
    mps: ModelPointSet,
    asmp: Assumptions,
    *,
    break_even: bool = False,
    margin: float | None = None,
    csm: float | None = None,
) -> FloatArray:
    """Solve the level monthly premium that meets a profitability target.

    Exactly one target must be given:

    * ``break_even`` -- the lowest non-onerous premium (FCF = 0, zero CSM).
    * ``margin``     -- a profit margin, ``CSM / PV(premiums) = margin``
      (e.g. ``0.10`` for 10%); must satisfy ``0 <= margin < 1``.
    * ``csm``        -- an absolute target CSM (profit) per model point.

    Every product field of ``mps`` is used as given -- only ``monthly_premium``
    is ignored, since it is the unknown being solved for. (A fixed
    ``single_premium``, if any, stays as given: the level premium is solved
    on top of it.)
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
    at_zero = value(_with_premium(mps, 0.0), asmp)
    at_one = value(_with_premium(mps, 1.0), asmp)
    a = at_zero.bel + at_zero.ra
    b = a - (at_one.bel + at_one.ra)

    if break_even:
        return a / b
    if margin is not None:
        return a / (b * (1.0 - margin))
    return (csm + a) / b
