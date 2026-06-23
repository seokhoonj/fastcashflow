"""GMM measurement assembly -- result types and the full-measurement builder.

The GMM model owns its measurement here: the result dataclasses
(:class:`GMMMeasurement`, :class:`CurrentEstimate`, :class:`GMMAggregate`), the
CSM orchestration (:func:`_compute_csm`), and the full-measurement assembler
(:func:`_measure_full`) that values a projection into a GMM result. The
assembler builds on the model-agnostic :func:`~fastcashflow.engine.valued_projection`
bundle and adds the GMM CSM roll, so no other model borrows a GMM container.

The shared valuation kernel (the cash-flow projection, ``valued_projection`` and
the GMM fast ``@njit`` codegen cluster) lives in :mod:`fastcashflow.engine`;
this module is imported back by ``engine`` for the result types and the
assembler, while ``valued_projection`` is imported here at call time to keep the
module load acyclic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow._measurement_basis import (
    MEASUREMENT_BASIS_INCEPTION,
    _inforce_marker_columns,
)
from fastcashflow._measurement_model import GMM
from fastcashflow.io import write_measurement, _write_measurement_columns
from fastcashflow.numerics import _csm_kernel


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, eq=False)
class GMMMeasurement:
    """IFRS 17 GMM measurement: BEL, RA and CSM.

    The headline fields (``bel``, ``ra``, ``csm``, ``loss_component``) are
    ``(n_mp,)`` inception values and are **always** present.

    The trajectory fields are the roll-forward over time and are populated
    only by ``measure(..., full=True)``; on the headline-only fast path
    (``full=False``) they are ``None``. ``bel_path`` / ``ra_path`` /
    ``csm_path`` are the ``(n_mp, n_time+1)`` trajectories whose column 0 is
    the inception value (so ``bel == bel_path[:, 0]`` when full). The CSM
    roll-forward decomposes as
    ``csm_path[:, t+1] = csm_path[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    ``lic_path`` is the liability for incurred claims -- zero unless a claims
    settlement pattern is set, which also discounts claims to their payment
    dates in the BEL.
    """

    model: ClassVar[str] = GMM

    # Headline -- always present, shape (n_mp,)
    bel: FloatArray              # inception Best Estimate of Liability
    ra: FloatArray               # inception Risk Adjustment
    csm: FloatArray              # inception Contractual Service Margin
    loss_component: FloatArray   # loss component at inception (onerous contracts)

    # Trajectory -- full=True only (None on the headline-only fast path)
    bel_path: FloatArray | None = None        # (n_mp, n_time+1) -- BEL trajectory
    ra_path: FloatArray | None = None         # (n_mp, n_time+1) -- RA trajectory
    csm_path: FloatArray | None = None        # (n_mp, n_time+1) -- CSM trajectory
    csm_accretion: FloatArray | None = None   # (n_mp, n_time)   -- CSM interest accreted
    csm_release: FloatArray | None = None     # (n_mp, n_time)   -- CSM released each month
    lic_path: FloatArray | None = None        # (n_mp, n_time+1) -- liability for incurred claims
    # The terminal column holds the residual of claims whose settlement tail
    # runs past the horizon (stays non-zero by design, not a leak).
    cashflows: "Cashflows | None" = None
    # bom = beginning of month, mom = mid of month: discount factors for a flow
    # at the start vs the middle of each month. Shape (n_time+1,) / (n_time,)
    # for a single basis; (n_mp, n_time+1) / (n_mp, n_time) when measured under
    # a per-segment basis dict, where each row discounts on its own curve.
    discount_factor_bom: FloatArray | None = None  # beginning-of-month discount factors
    discount_factor_mid: FloatArray | None = None  # mid-of-month discount factors
    # Source model points, stamped by ``measure`` so ``group(m, by=[...])`` can
    # resolve axis names without re-passing them. A reference, not a copy; None
    # on a grouped result (its rows are groups, not model points).
    model_points: "ModelPoints | None" = None
    # The per-group composite label (one per row) on a result returned by
    # ``group`` / ``group_of_contracts``; None on a per-model-point measurement.
    group_labels: "np.ndarray | None" = None
    # The number of model points in each group, aligned with ``group_labels``.
    group_sizes: IntArray | None = None
    # Time basis of the result (see _measurement_basis): 'inception' for
    # new-business measure(); in-force results re-base the headline to the
    # valuation date while the trajectories stay on the inception axis, so
    # inception-axis consumers (group / roll_forward / report / transition /
    # plot_*) reject anything else via _require_inception.
    measurement_basis: str = MEASUREMENT_BASIS_INCEPTION

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm),
                ("loss", self.loss_component)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr("GMMMeasurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str("GMMMeasurement", self._columns())

    def estimate_at(self, month: int) -> "CurrentEstimate":
        """The current estimate (BEL / RA / CSM / LIC) at a future ``month``.

        This is the deterministic nested-projection view (IFRS 17 Sec. 40): the
        cohort liability the entity would carry at month ``t`` if the central
        best-estimate scenario unfolds to it. It is column ``t`` of the
        trajectories -- so ``estimate_at(0).bel`` equals the inception headline
        ``bel`` -- and equals a fresh in-force measurement at that month,
        ``gmm.measure_inforce(elapsed=t)`` carrying the deterministic survivor
        count (reading the trajectory and re-projecting agree; the tests assert
        this for every ``t``). The returned object's ``per_survivor`` re-bases
        every figure to one surviving policy.

        Requires a ``full=True`` measurement (the trajectory paths); the fast
        path carries only the inception headline. GMM only for now -- VFA / PAA
        carry the same ``*_path`` shape and can gain this later.
        """
        if self.bel_path is None or self.cashflows is None:
            raise ValueError(
                "estimate_at requires a full=True measurement (trajectory "
                "paths); the fast path (full=False) carries only the headline."
            )
        n_time = self.bel_path.shape[1] - 1
        t = int(month)
        if not 0 <= t <= n_time:
            raise ValueError(
                f"month must be in [0, {n_time}] (the projection horizon); "
                f"got {month}."
            )
        # Survivors at the start of month t. The terminal column t == n_time has
        # no in-force column (inforce is (n_mp, n_time)), so the count there is
        # the maturity exit count -- the in-force that reached term.
        if t < n_time:
            inforce = self.cashflows.inforce[:, t]
        else:
            inforce = self.cashflows.maturity_survivors
        lic = (self.lic_path[:, t] if self.lic_path is not None
               else np.zeros_like(self.bel_path[:, t]))
        return CurrentEstimate(
            month=t,
            bel=self.bel_path[:, t],
            ra=self.ra_path[:, t],
            csm=self.csm_path[:, t],
            lic=lic,
            inforce=np.asarray(inforce, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True, eq=False)
class CurrentEstimate:
    """The GMM current estimate at one future month (IFRS 17 Sec. 40).

    Returned by :meth:`GMMMeasurement.estimate_at`. The fields are the cohort
    BEL / RA / CSM / LIC at ``month`` -- the liability the entity would carry at
    that date if the central scenario runs to it -- each shape ``(n_mp,)``.
    ``inforce`` is the deterministic survivor count at ``month``. Derived views
    are properties: ``fcf`` = BEL + RA, ``lrc`` = FCF + CSM (the GMM carrying
    amount), and ``per_survivor`` re-bases every money figure to one surviving
    policy (money / ``inforce``).
    """

    model: ClassVar[str] = GMM

    month: int
    bel: FloatArray          # (n_mp,) cohort BEL at `month`
    ra: FloatArray           # (n_mp,) cohort RA at `month`
    csm: FloatArray          # (n_mp,) cohort CSM at `month`
    lic: FloatArray          # (n_mp,) cohort LIC at `month`
    inforce: FloatArray      # (n_mp,) deterministic survivors at `month`

    @property
    def fcf(self) -> FloatArray:
        """Fulfilment cash flows = BEL + RA (IFRS 17 Sec. 32, 37)."""
        return self.bel + self.ra

    @property
    def lrc(self) -> FloatArray:
        """Liability for remaining coverage = FCF + CSM (the GMM carrying amount)."""
        return self.bel + self.ra + self.csm

    @property
    def per_survivor(self) -> "CurrentEstimate":
        """The same estimate re-based to one surviving policy (money / inforce).

        Where ``inforce`` is zero (a fully run-off cohort) the money figures are
        already zero, so the per-policy figure is a zero-safe 0.0.
        """
        inf = np.where(self.inforce > 0.0, self.inforce, 1.0)
        return CurrentEstimate(
            month=self.month,
            bel=self.bel / inf,
            ra=self.ra / inf,
            csm=self.csm / inf,
            lic=self.lic / inf,
            inforce=np.ones_like(self.inforce),
        )

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm),
                ("LIC", self.lic)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr(f"CurrentEstimate(month={self.month})",
                                self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str(f"CurrentEstimate(month={self.month})",
                               self._columns())


@dataclass(frozen=True, slots=True, eq=False)
class GMMAggregate:
    """Portfolio-aggregate GMM trajectories -- the scalable ``full=True`` view.

    BEL / RA / CSM are additive across contracts, so a large book's liability
    run-off is its per-model-point trajectories *summed over the model-point
    axis*. This holds only that sum: the scalar inception totals plus the
    ``(n_time+1,)`` aggregate ``bel_path`` / ``ra_path`` / ``csm_path`` (column 0
    is inception, so ``bel == bel_path[0]``). It is what
    :func:`~fastcashflow.gmm.measure_aggregate` returns, computed in bounded
    memory so it works where a per-model-point ``measure(full=True)`` would OOM.
    """

    model: ClassVar[str] = GMM

    bel: float                   # portfolio inception BEL total
    ra: float                    # portfolio inception RA total
    csm: float                   # portfolio inception CSM total
    loss_component: float        # portfolio inception loss-component total
    bel_path: FloatArray         # (n_time+1,) -- aggregate BEL trajectory
    ra_path: FloatArray          # (n_time+1,) -- aggregate RA trajectory
    csm_path: FloatArray         # (n_time+1,) -- aggregate CSM trajectory


@write_measurement.register
def _(measurement: GMMMeasurement, path, *, ids=None):
    cols = {"bel": measurement.bel, "ra": measurement.ra,
            "csm": measurement.csm,
            "loss_component": measurement.loss_component}
    # In-force output gets marker columns so it stays distinguishable from
    # new-business output at the file boundary; inception output is unchanged.
    cols.update(_inforce_marker_columns(measurement, measurement.bel.shape[0]))
    _write_measurement_columns(cols, path, ids)


# ---------------------------------------------------------------------------
# CSM and the full-measurement assembler
# ---------------------------------------------------------------------------

def _compute_csm(bel0, ra0, inforce, discount_monthly, discount_units=False):
    """CSM at initial recognition (Sec. 38) and deterministic roll-forward (Sec. 44).

    Pure-array orchestration: fulfilment cash flows ``FCF = BEL + RA``,
    initial CSM = ``max(0, -FCF)``, loss component = ``max(0, FCF)``, then
    the CSM is rolled forward in :func:`_csm_kernel` (interest accretion at
    the locked-in monthly rate, release proportional to coverage units --
    in-force here).

    ``inforce`` is ``(n_mp, n_time)`` (the coverage-unit series), ``bel0`` /
    ``ra0`` are ``(n_mp,)``. Returns
    ``(csm, accretion, release, loss_component)``.
    """
    fcf = bel0 + ra0
    csm0 = np.maximum(0.0, -fcf)
    loss_component = np.maximum(0.0, fcf)
    csm, accretion, release = _csm_kernel(csm0, inforce, discount_monthly,
                                          discount_units)
    return csm, accretion, release, loss_component


def _measure_full(model_points: "ModelPoints", basis: "Basis", *,
                  discount_monthly: FloatArray | None = None,
                  lapse_scale: FloatArray | None = None) -> GMMMeasurement:
    """Full GMM measurement: BEL, RA and CSM rolled forward over time.

    The shared neutral bundle from :func:`~fastcashflow.engine.valued_projection`
    plus the GMM CSM roll (:func:`_compute_csm`), assembled into a
    :class:`GMMMeasurement` that carries both the ``(n_mp,)`` inception headline
    (column 0 of each trajectory) and the ``(n_mp, n_time+1)`` ``*_path``
    trajectories. Reached by ``measure(..., full=True)``. ``discount_monthly`` /
    ``lapse_scale`` are forwarded to :func:`~fastcashflow.engine.valued_projection`
    (see there for the override semantics).
    """
    from fastcashflow.engine import valued_projection
    vp = valued_projection(model_points, basis,
                           discount_monthly=discount_monthly,
                           lapse_scale=lapse_scale)
    csm, csm_accretion, csm_release, loss_component = _compute_csm(
        vp.bel, vp.ra, vp.cashflows.inforce, vp.discount_monthly,
        basis.coverage_unit_discount,
    )

    return GMMMeasurement(
        bel=vp.bel,
        ra=vp.ra,
        csm=csm[:, 0],
        loss_component=loss_component,
        bel_path=vp.bel_path,
        ra_path=vp.ra_path,
        csm_path=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        lic_path=vp.lic_path,
        cashflows=vp.cashflows,
        discount_factor_bom=vp.discount_factor_bom,
        discount_factor_mid=vp.discount_factor_mid,
    )
