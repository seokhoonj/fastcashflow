"""``fastcashflow.core`` -- the cash-flow projection engine.

The layer BELOW the IFRS 17 measurement: project the raw monthly cash flows from
model points + a :class:`~fastcashflow.basis.Basis`, with no accounting on top
(no BEL discounting, no CSM / RA). A non-IFRS 17 user -- pricing, ALM, experience
study, asset-adequacy -- works directly against this engine: project the flows
here, build / apply a discount curve from :mod:`fastcashflow.curves`, and value
them, without touching the measurement layer.

A thin re-export facade (the gmm.py / vfa.py pattern); the implementations live
in :mod:`fastcashflow.projection` and :mod:`fastcashflow.gmm._engine`.

    cf = fcf.core.project_cashflows(mp, basis)          # -> fcf.Cashflows
    df_bom, df_mid = fcf.curves.discount_factors(basis, cf.premium_cf.shape[1])
    bel = (cf.mortality_cf * df_mid).sum() - (cf.premium_cf * df_bom).sum()
"""
from fastcashflow._measurement.inforce import inforce_surrender_value
from fastcashflow.projection import project_cashflows

__all__ = ["project_cashflows", "inforce_surrender_value"]
