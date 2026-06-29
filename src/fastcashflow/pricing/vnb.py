"""Value of new business (VNB) -- the new business value net of the cost of capital.

VNB is the MCEV value-of-new-business metric (CFO Forum). A thin assembly over the
profit-testing layer: :func:`fastcashflow.pricing.csm_plus_ra` is pre-tax and
*pre-required-capital* -- the present value of future profit with no charge for the
capital the entity must hold behind the contract. This module adds that charge:

    VNB = PVFP - CoC - TVOG

* ``PVFP`` -- present value of future shareholder profit, the present value of a
  profit signature (the IFRS 17 :func:`~fastcashflow.pricing.signature` or the
  traditional :func:`~fastcashflow.pricing.statutory_profit_signature`).
* ``CoC``  -- the frictional cost of holding required capital: a spread charged on
  the required-capital trajectory and present-valued. This is the same arithmetic
  the engine's cost-of-capital risk adjustment uses (a spread times the
  capital held over the run-off); here the capital is supplied by the caller.
* ``TVOG`` -- the time value of options and guarantees (e.g. the interest-rate
  guarantee cost from :func:`~fastcashflow.pricing.interest_tvog`).

This is a traditional single-rate VNB (v1): one ``reference_rate`` discounts both
the profit and the capital-cost streams, with a ``frictional_spread`` charged on
the capital. A reference-rate / CRNHR (MCEV-style) decomposition, real regulatory
required capital, and tax are deferred follow-ups -- v1 keeps the required capital
caller-supplied and transparent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.curves import discount_factors_from_curve
from fastcashflow.pricing.profit import ProfitSignature


@dataclass(frozen=True, slots=True, eq=False)
class VNB:
    """The value of new business, split into its components (portfolio total).

    ``pvfp`` is the present value of future shareholder profit; ``cost_of_capital``
    the frictional cost of holding the required capital; ``tvog`` the time value of
    options and guarantees. :attr:`value` is the value of new business
    ``pvfp - cost_of_capital - tvog`` (positive = value-creating).
    """

    pvfp: float
    cost_of_capital: float
    tvog: float

    @property
    def value(self) -> float:
        """Value of new business: ``pvfp - cost_of_capital - tvog``."""
        return self.pvfp - self.cost_of_capital - self.tvog


def vnb(
    profit_signature: ProfitSignature,
    *,
    reference_rate: float,
    discount_monthly: FloatArray | None = None,
    required_capital: FloatArray | float | None = None,
    reserve: FloatArray | None = None,
    frictional_spread: float = 0.0,
    tvog: float = 0.0,
) -> VNB:
    """Value of new business from a profit signature and a cost of capital.

    ``VNB = PVFP - CoC - TVOG`` (portfolio total). The function is basis-agnostic:
    pass the IFRS 17 :func:`~fastcashflow.pricing.signature` or the traditional
    :func:`~fastcashflow.pricing.statutory_profit_signature` as ``profit_signature``
    -- only its present value is used.

    Parameters
    ----------
    reference_rate
        The annual rate discounting both the profit stream (via
        :meth:`ProfitSignature.present_value`) and the capital-cost stream.
    discount_monthly
        ``(n_time,)`` per-month rate curve for the cost-of-capital present value
        (e.g. :func:`fastcashflow.curves.discount_monthly_curve`). Required when a
        non-zero capital charge is requested; otherwise the CoC is zero.
    required_capital
        The required-capital trajectory ``RC_t``, portfolio total. Either an
        explicit ``(n_time,)`` / ``(n_time+1,)`` array (the capital held at the start
        of each month -- e.g. the confidence-level ``measurement.ra_path.sum(0)`` as
        a risk-capital proxy, or a regulatory capital path), or a scalar capital
        factor applied to ``reserve``. ``None`` gives a zero capital charge.
    reserve
        ``(n_time+1,)`` reserve path, used only when ``required_capital`` is a scalar
        factor (the capital is ``required_capital * reserve``); e.g.
        ``statutory_reserve(...)[0].sum(0)`` or ``measurement.bel_path.sum(0)``.
    frictional_spread
        The annual spread charged on the required capital (the cost of locking it
        up). Zero gives a zero capital charge.
    tvog
        The time value of options and guarantees to deduct (e.g.
        ``interest_tvog(...).total_value``). Default 0.

    Returns
    -------
    VNB
        ``pvfp``, ``cost_of_capital``, ``tvog`` and the derived ``value``.

    Notes
    -----
    Double counting: pairing the traditional ``statutory_profit_signature`` (whose
    profit carries no risk adjustment) with any required capital is clean. Pairing
    the IFRS 17 ``signature`` (whose profit already includes the risk-adjustment
    release) with the confidence-level RA path as the capital is sound but mixes
    views -- the RA *release* is value flowing into the PVFP, the CoC is the
    frictional drag on holding capital, two distinct quantities -- so the
    traditional pairing is the cleaner default.

    The capital charge ``(frictional_spread / 12) * sum_t RC_t * df_bom(t)`` is the
    same inception value the engine's cost-of-capital risk adjustment produces for
    the same capital path and annual spread (the per-month-charged backward present
    value of the capital), so passing the confidence-level ``ra_path.sum(0)`` with
    ``frictional_spread = cost_of_capital_rate`` reproduces that figure exactly.
    """
    pvfp = profit_signature.present_value(reference_rate)

    if (required_capital is None or discount_monthly is None
            or frictional_spread == 0.0):
        coc = 0.0
    else:
        df_bom = discount_factors_from_curve(
            np.asarray(discount_monthly, dtype=np.float64))[0]   # (n_time+1,)
        if np.ndim(required_capital) == 0:
            if reserve is None:
                raise ValueError(
                    "a scalar required_capital is a capital factor and needs "
                    "reserve= (the (n_time+1,) reserve path it scales)")
            rc = float(required_capital) * np.asarray(reserve, dtype=np.float64)
        else:
            rc = np.asarray(required_capital, dtype=np.float64)
        if rc.ndim != 1:
            raise ValueError(
                "required_capital must be 1-D (portfolio total); sum a per-model-"
                "point capital path over the model-point axis first")
        if rc.shape[0] > df_bom.shape[0]:
            raise ValueError(
                f"required_capital has {rc.shape[0]} entries but the discount "
                f"horizon is {df_bom.shape[0]} (n_time+1); they must align")
        # Capital held over each month, present-valued at the start of that month
        # (begin-of-month factor) and charged the annual spread for one month
        # (the 1/12 time step). Summing RC_t * df_bom(t) over the whole path is
        # the engine's cost-of-capital backward present value -- align to the RC
        # length, do NOT truncate to n_time (that would drop the boundary column).
        coc = float(frictional_spread / 12.0 * np.sum(rc * df_bom[:rc.shape[0]]))

    return VNB(pvfp=float(pvfp), cost_of_capital=coc, tvog=float(tvog))


__all__ = ["VNB", "vnb"]
