"""IFRS 17 reinsurance contracts held -- a quota-share treaty.

A cedant buys reinsurance to transfer risk. This module measures a
proportional (quota-share) reinsurance contract held over a direct
portfolio: the cedant cedes a fixed fraction of its claims and pays the
same fraction of its premiums to the reinsurer.

IFRS 17 measures reinsurance held with the general model but with two
modifications (paragraphs 60-70):

* The risk adjustment is the amount of risk *transferred* to the reinsurer
  (paragraph 64) -- here, the margin on the ceded claims.
* There is no unearned profit; the CSM is instead the net cost or net gain
  of buying the cover (paragraph 65). So the CSM may be negative -- a net
  cost is deferred and amortised, not expensed -- and there is no loss
  component.

v1 scope: a single quota-share cession rate over the portfolio, with no
ceding commission. The reinsurer's non-performance risk and the
loss-recovery component (for onerous underlying contracts) are left for
later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.basis import Basis, _single_basis
from fastcashflow.curves import discount_factors, discount_monthly_curve
from fastcashflow.numerics import _csm_kernel, _norm_ppf
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows


@dataclass(frozen=True, slots=True)
class ReinsuranceMeasurement:
    """Measurement of a reinsurance contract held.

    Headline ``bel``, ``ra`` and ``csm`` are ``(n_mp,)`` inception figures --
    ``bel`` is the present value of reinsurance premiums less recoveries (a
    net cost when positive), ``ra`` is the risk transferred, ``csm`` is the
    inception net cost or gain (may be negative). The trajectory fields are
    populated only on the full path; ``csm_path`` reconciles as
    ``csm_path[:, t+1] = csm_path[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    """

    # headline -- always present, shape (n_mp,)
    bel: FloatArray            # PV(reinsurance premiums) - PV(recoveries)
    ra: FloatArray             # risk transferred to the reinsurer
    csm: FloatArray            # inception net cost/gain
    # trajectory -- full only (None on the headline-only path)
    csm_path: FloatArray | None = None         # (n_mp, n_time+1) -- net cost/gain trajectory
    csm_accretion: FloatArray | None = None    # (n_mp, n_time)
    csm_release: FloatArray | None = None      # (n_mp, n_time)
    recovery: FloatArray | None = None         # (n_mp, n_time) -- recoveries received
    reinsurance_premium: FloatArray | None = None    # (n_mp, n_time) -- reinsurance premiums paid
    cashflows: "Cashflows | None" = None
    discount_bom: FloatArray | None = None     # (n_time+1,) -- for grouped CSM re-derivation
    model_points: "ModelPoints | None" = None  # stamped by measure_reinsurance, for group axes
    group_labels: "np.ndarray | None" = None   # per-group label on a grouped result
    group_sizes: IntArray | None = None     # model points per group, aligned with labels


class Treaty(Protocol):
    """How a reinsurance treaty cedes the direct cash flows.

    ``cede`` receives the direct portfolio's projected :class:`Cashflows` and
    returns ``(ceded_mortality_cf, ceded_morbidity_cf, reinsurance_premium_cf)`` --
    each ``(n_mp, n_time)``. The two ceded-claim streams are kept split by
    risk so the risk adjustment can weight them by the right cv; their sum is
    the recovery. A new treaty type (excess-of-loss, surplus, ...) implements
    this one method.
    """

    def cede(self, proj: Cashflows) -> tuple[FloatArray, FloatArray, FloatArray]:
        ...


@dataclass(frozen=True, slots=True)
class QuotaShare:
    """Proportional reinsurance -- cede a fixed fraction of claims and premiums.

    ``cession`` (in ``[0, 1]``) is the ceded fraction: the cedant recovers
    that fraction of its claims and pays the same fraction of its premiums as
    reinsurance premium.
    """

    cession: float

    def __post_init__(self) -> None:
        # Validate at construction, not deep in cede(): a non-numeric, NaN or
        # out-of-range cession otherwise surfaces late or as a cryptic error.
        c = float(self.cession)  # ValueError for a non-numeric cession
        if not np.isfinite(c):
            raise ValueError(f"cession must be finite, got {self.cession!r}")
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"cession must be in [0, 1], got {self.cession!r}")

    def cede(self, proj: Cashflows) -> tuple[FloatArray, FloatArray, FloatArray]:
        if not 0.0 <= self.cession <= 1.0:
            raise ValueError(f"cession must be in [0, 1], got {self.cession}")
        return (self.cession * proj.claim_cf,
                self.cession * proj.morbidity_cf,
                self.cession * proj.premium_cf)


def measure_reinsurance(
    model_points: ModelPoints, basis: Basis, treaty: Treaty
) -> ReinsuranceMeasurement:
    """Measure a reinsurance contract held over a direct portfolio.

    ``treaty`` describes how the cover cedes the direct cash flows -- e.g.
    :class:`QuotaShare(cession=0.5)`. The BEL is the present value of
    reinsurance premiums less recoveries; the RA is the margin on the ceded
    claims (the risk transferred). The CSM is ``-(BEL - RA)`` -- the net cost
    or gain of the cover -- and may be negative; it is accreted and released
    by coverage units like a direct contract's CSM, but with no loss
    component (paragraph 65).
    """
    basis = _single_basis(basis, entry="measure_reinsurance")
    proj = project_cashflows(model_points, basis)
    discount_bom, discount_mid = discount_factors(basis, proj.n_time)

    ceded_mortality, ceded_morbidity, reinsurance_premium = treaty.cede(proj)
    recovery = ceded_mortality + ceded_morbidity

    pv_recovery = (recovery * discount_mid).sum(axis=1)
    pv_reinsurance_premium = (reinsurance_premium * discount_bom[:-1]).sum(axis=1)
    bel = pv_reinsurance_premium - pv_recovery

    # RA -- the risk transferred, i.e. the margin on the ceded claims.
    z = _norm_ppf(basis.ra_confidence)
    pv_ceded_mortality = (ceded_mortality * discount_mid).sum(axis=1)
    pv_ceded_morbidity = (ceded_morbidity * discount_mid).sum(axis=1)
    ra = z * (basis.mortality_cv * pv_ceded_mortality
              + basis.morbidity_cv * pv_ceded_morbidity)

    # CSM -- the net cost or gain of the cover. No loss component: a net cost
    # is a negative CSM, deferred and amortised over the coverage.
    csm0 = -(bel - ra)
    csm, csm_accretion, csm_release = _csm_kernel(
        csm0, proj.inforce,
        discount_monthly_curve(basis, proj.n_time),
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
        cashflows=proj,
        discount_bom=discount_bom,
        model_points=model_points,
    )
