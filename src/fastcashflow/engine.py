"""Engine entry point -- ties projection and GMM measurement together."""
from __future__ import annotations

from dataclasses import dataclass

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions
from fastcashflow.gmm import compute_bel, compute_csm, compute_ra, discount_factors
from fastcashflow.modelpoint import ModelPointSet
from fastcashflow.projection import CashflowProjection, project_cashflows


@dataclass(frozen=True, slots=True)
class GMMResult:
    """Phase 0 GMM measurement result.

    Per-model-point arrays have shape ``(n_mp,)`` unless stated otherwise.
    """

    bel: FloatArray              # Best Estimate of Liability
    ra: FloatArray               # Risk Adjustment
    csm0: FloatArray             # CSM at initial recognition
    loss_component: FloatArray   # loss component at inception (onerous contracts)
    csm: FloatArray              # (n_mp, n_time+1) -- CSM trajectory
    csm_release: FloatArray      # (n_mp, n_time)   -- CSM released each month
    projection: CashflowProjection
    discount: FloatArray         # (n_time,) -- monthly discount factors


def run(mps: ModelPointSet, asmp: Assumptions) -> GMMResult:
    """Run the Phase 0 GMM projection for a model point set.

    Parameters
    ----------
    mps  : the contracts to project.
    asmp : the deterministic assumption set.
    """
    proj = project_cashflows(mps, asmp)
    discount = discount_factors(asmp, proj.n_time)

    bel = compute_bel(proj, discount)
    ra = compute_ra(proj, discount, asmp.ra_confidence, asmp.claims_cv)
    csm = compute_csm(bel, ra, proj, asmp)

    return GMMResult(
        bel=bel,
        ra=ra,
        csm0=csm.csm[:, 0],
        loss_component=csm.loss_component,
        csm=csm.csm,
        csm_release=csm.release,
        projection=proj,
        discount=discount,
    )
