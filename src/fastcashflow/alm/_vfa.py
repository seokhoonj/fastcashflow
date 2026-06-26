"""VFA (account-value) ALM -- the variable book's interest sensitivity and the
entity general-account net liability ladder, exposed prefix-free under ``fcf.vfa.*``.

The symmetric counterpart of the risk-free ``liability_*`` metrics in
:mod:`fastcashflow.alm`: a variable / universal-life book discounts at the
underlying-items return (``investment_return``), not the risk-free curve, so its
interest sensitivity bumps that return (which moves BOTH the discount and the
account growth, so the two effects partly offset). ``net_liability_cashflows`` is
the entity general-account ladder (guarantee excess + expense - fee), the VFA
counterpart of :func:`fastcashflow.alm.net_liability_cashflows`.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow._duration import DurationResult, _BP
from fastcashflow.basis import Basis
from fastcashflow.model_points import ModelPoints


def net_liability_cashflows(measurement) -> FloatArray:
    """The VFA entity general-account net liability cash flow per month ``(n_time,)``.

    A variable / unit-linked contract's account value is invested in the
    underlying items, so the account-value portion of every benefit is funded by
    the unit fund -- only the GMDB / GMAB excess over the account value lands on
    the entity's own general account (the bonds / equity that an
    :class:`~fastcashflow.assets.Portfolio` represents). Returns the per-month
    net OUTFLOW summed over the portfolio:

        guarantee_excess + expense - variable_fee

    the guarantee top-up and expenses the general account funds, less the income it
    keeps. This is the VFA counterpart of :func:`fastcashflow.alm.net_liability_cashflows` (which
    nets the GROSS benefits of a non-account book); here the gross account-value
    benefit is excluded because the unit fund, not the entity, pays it. Discounting
    at the underlying-items return reproduces the BEL before RA (at a zero return the
    undiscounted sum equals the BEL); the undiscounted ladder is the liquidity
    foundation for the VFA asset-liability gap.

    Two product shapes, both ``full=True``:

    * closed-form VARIABLE-ANNUITY: income is the variable fee, so the net flow is
      ``guarantee_excess + expense - variable_fee``.
    * account-backed UNIVERSAL-LIFE: there is no variable fee; the entity income is
      the bundle of account charges, so the net flow is the guarantee net cost
      (NAR death excess + GMAB maturity excess) plus expense, less the premium load,
      COI, admin and cost-deducting-rider charges and the retained surrender charge.
      The account-value pass-through and credited interest net against the held fund;
      with no crediting guarantee the undiscounted sum reconciles exactly to the UL
      net BEL (the crediting-guarantee intrinsic value is the only residual when the
      floor binds, carried by the BEL itself).

    The reconciliation is the UNDISCOUNTED-sum identity (exact at a zero
    underlying-items return); like :func:`fastcashflow.alm.net_liability_cashflows` this is a monthly
    LIQUIDITY ladder, so each month bundles begin- and mid-month flows into one
    figure -- a non-zero-rate PV is liquidity-grade, not a to-the-cent BEL match."""
    cf = measurement.cashflows
    if cf is None:
        raise ValueError(
            "fcf.vfa.net_liability_cashflows needs a full=True measurement (it carries "
            "the cash flows); the headline-only / aggregate paths do not.")
    if getattr(cf, "account", None) is not None:
        return _ul_net_liability_cashflows(measurement)
    return _va_net_liability_cashflows(measurement)

def _va_net_liability_cashflows(measurement) -> FloatArray:
    """Closed-form variable-annuity entity net liability: ``guarantee_excess +
    expense - variable_fee`` (the fee is the entity's income)."""
    cf = measurement.cashflows
    ge = measurement.guarantee_excess_cf
    fee = measurement.fee_cf
    if ge is None or fee is None:
        raise ValueError(
            "fcf.vfa.net_liability_cashflows needs a full=True closed-form VA "
            "measurement (guarantee_excess_cf / fee_cf); got a headline-only result.")
    return (ge + cf.expense_cf - fee).sum(axis=0)

def _ul_net_liability_cashflows(measurement) -> FloatArray:
    """Universal-life entity net liability on the guarantee-excess basis. The
    account-value pass-through and credited interest telescope against the held
    fund, leaving the guarantee net cost + rider claims + expense less the account
    charge income."""
    cf = measurement.cashflows
    account = cf.account
    n_time = cf.premium_cf.shape[1]
    inforce = np.asarray(cf.inforce, dtype=np.float64)
    deaths = np.asarray(cf.deaths, dtype=np.float64)
    av_mid = np.asarray(account.av_mid, dtype=np.float64)
    av = np.asarray(account.av, dtype=np.float64)
    gmab = np.asarray(measurement.model_points.maturity_benefit, dtype=np.float64)
    term = np.asarray(measurement.model_points.term_months, dtype=np.int64)
    maturity_survivors = np.asarray(cf.maturity_survivors, dtype=np.float64)
    rows = np.arange(term.shape[0])

    # Death entity cost = the death benefit less the account value released on death
    # (deaths * av_mid, which nets against the held fund). For an account death
    # (pays max(av_mid, face)) this is the NAR excess max(0, face - av_mid); written
    # as mortality_cf - deaths*av_mid so any NON-account death claim in mortality_cf
    # is captured too. Morbidity / disability rider claims are pure entity outflows
    # (a cost-deducting rider draws account_charge as income but pays its benefit
    # from the entity), so add them in full.
    mortality_cf = np.asarray(cf.mortality_cf, dtype=np.float64)
    death_entity = (mortality_cf - deaths * av_mid).sum(axis=0)
    rider_claims = (np.asarray(cf.morbidity_cf, dtype=np.float64)
                    + np.asarray(cf.disability_cf, dtype=np.float64)).sum(axis=0)
    # GMAB maturity excess at each policy's term, on the matured (month-end) AV.
    gmab_maturity_excess = np.zeros(n_time, dtype=np.float64)
    np.add.at(gmab_maturity_excess, np.minimum(term, n_time - 1),
              maturity_survivors * np.maximum(
                  0.0, gmab - av[rows, np.minimum(term, n_time)]))
    expense = np.asarray(cf.expense_cf, dtype=np.float64).sum(axis=0)
    premium_load_income = (
        np.asarray(cf.premium_cf, dtype=np.float64)
        - inforce * np.asarray(account.prem_to_av, dtype=np.float64)).sum(axis=0)
    coi_drawn = (inforce * np.asarray(account.coi, dtype=np.float64)).sum(axis=0)
    admin_drawn = (inforce * np.asarray(account.admin_charge, dtype=np.float64)).sum(axis=0)
    account_charge_drawn = (
        inforce * np.asarray(account.account_charge, dtype=np.float64)).sum(axis=0)
    # Retained surrender charge = gross account surrendered less the net paid
    # (surrender_cf already nets the charge). Maturing survivors are removed from
    # the non-maturity exit count at their term - 1 exit column.
    inforce_pad = np.concatenate(
        [inforce, np.zeros((inforce.shape[0], 1), dtype=np.float64)], axis=1)
    non_maturity_exits = (inforce_pad[:, :-1] - inforce_pad[:, 1:]) - deaths
    np.add.at(non_maturity_exits, (rows, np.minimum(term - 1, n_time - 1)),
              -maturity_survivors)
    surrender_charge_retained = (
        non_maturity_exits * av_mid
        - np.asarray(cf.surrender_cf, dtype=np.float64)).sum(axis=0)

    return (death_entity + rider_claims + gmab_maturity_excess + expense
            - premium_load_income - coi_drawn - admin_drawn
            - account_charge_drawn - surrender_charge_retained)

def _bel(model_points: ModelPoints, basis: Basis, investment_return) -> float:
    """VFA portfolio BEL under an underlying-items-return override (fast path)."""
    from fastcashflow._measurement import vfa as _vfa
    m = _vfa.measure(model_points,
                    replace(basis, investment_return=investment_return), full=False)
    return float(m.bel.sum())

def liability_dv01(model_points: ModelPoints, basis: Basis, *,
                       bump: float = _BP) -> float:
    """The VFA liability DV01 -- the decrease in the VFA BEL for a +1bp parallel
    rise in the underlying-items return, by central difference.

    The VFA counterpart of :func:`fastcashflow.alm.liability_dv01`: where the GMM metric bumps the
    risk-free discount curve, this bumps ``basis.investment_return`` (the rate the
    account is credited and the liability discounted at), so the figure nets the
    discount and account-growth responses. Exposed as ``fcf.vfa.liability_dv01``."""
    base = float(np.asarray(basis.investment_return, dtype=np.float64))
    up = _bel(model_points, basis, base + bump)
    dn = _bel(model_points, basis, base - bump)
    return -(up - dn) / (2.0 * bump) * _BP

def liability_duration(model_points: ModelPoints, basis: Basis, *,
                           bump: float = _BP) -> DurationResult:
    """The VFA liability's interest sensitivity -- the VFA counterpart of
    :func:`fastcashflow.alm.liability_duration`, differencing the VFA BEL against the
    underlying-items return (``basis.investment_return``) rather than the
    risk-free curve. ``pv`` is the VFA BEL; ``dv01`` / effective ``modified`` /
    ``convexity`` mirror :func:`fastcashflow.alm.liability_duration` (``modified`` / ``convexity``
    are ``nan`` when ``|pv|`` is negligible). ``macaulay`` is ``nan``.

    A single-rate (flat) sensitivity: the underlying-items return is one scalar,
    not a per-year curve, so there is no key-rate decomposition counterpart.
    Exposed as ``fcf.vfa.liability_duration``."""
    base = float(np.asarray(basis.investment_return, dtype=np.float64))
    pv = _bel(model_points, basis, base)
    dv01 = liability_dv01(model_points, basis, bump=bump)
    if abs(pv) > 1.0:
        modified = dv01 / (abs(pv) * _BP)
        up = _bel(model_points, basis, base + bump)
        dn = _bel(model_points, basis, base - bump)
        convexity = (up + dn - 2.0 * pv) / (abs(pv) * bump * bump)
    else:
        modified = convexity = float("nan")
    return DurationResult(pv=pv, macaulay=float("nan"), modified=modified,
                          dv01=dv01, convexity=convexity)
