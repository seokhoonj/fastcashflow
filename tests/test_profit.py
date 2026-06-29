"""Profit testing -- new business value, profit margin, signature, IRR.

Hand-calc anchors on tiny cases plus the structural identities: the new business
value is the negated BEL (equivalently CSM + RA - loss component), the profit
margin is that over the PV of premiums, and the IFRS 17 profit signature
re-discounts to the CSM+RA.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import pricing


def _flat(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


_CM = {"DEATH": fcf.CalculationMethod.DEATH}


def _basis(**over):
    kw = dict(mortality_annual=_flat(0.01), lapse_annual=_flat(0.0),
              discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.0,
              coverages=(fcf.CoverageRate("DEATH", _flat(0.01)),))
    kw.update(over)
    return fcf.Basis(**kw)


def _mp(premium=200_000.0, term=3, db=100_000_000.0):
    return fcf.ModelPoints.single(issue_age=40, premium=premium, term_months=term,
                                  benefits={"DEATH": db}, calculation_methods=_CM)


# ---------------------------------------------------------------------------
# New business value -- the core identity
# ---------------------------------------------------------------------------

def test_csm_plus_ra_is_negated_bel():
    """CSM+RA = -BEL = CSM + RA - loss component, for any contract."""
    m = fcf.gmm.measure(_mp(term=120), _basis(mortality_cv=0.10, discount_annual=0.03))
    v = pricing.csm_plus_ra(m)
    assert np.allclose(v, -m.bel)
    assert np.allclose(v, m.csm + m.ra - m.loss_component)


def test_csm_plus_ra_handcalc():
    """A 3-month, 1-policy contract, zero discount, no RA: CSM+RA is the by-hand
    PV(premiums) - PV(claims)."""
    q, P, db, term = 0.01, 200_000.0, 100_000_000.0, 3
    m = fcf.gmm.measure(_mp(premium=P, term=term, db=db), _basis())
    mq = 1.0 - (1.0 - q) ** (1.0 / 12.0)
    inforce = (1.0 - mq) ** np.arange(term)
    deaths = inforce * mq
    csm_plus_ra_hand = float(np.sum(inforce * P) - np.sum(deaths * db))
    assert np.isclose(pricing.csm_plus_ra(m)[0], csm_plus_ra_hand, rtol=1e-9)


def test_profit_margin_is_csm_plus_ra_over_pv_premium():
    """Profit margin = CSM+RA / PV(premiums); zero discount makes PV(premiums) the
    in-force-weighted premium sum."""
    q, P, term = 0.01, 200_000.0, 3
    m = fcf.gmm.measure(_mp(premium=P, term=term), _basis())
    mq = 1.0 - (1.0 - q) ** (1.0 / 12.0)
    pv_prem = float(np.sum((1.0 - mq) ** np.arange(term) * P))
    assert np.isclose(pricing.profit_margin(m)[0],
                      pricing.csm_plus_ra(m)[0] / pv_prem, rtol=1e-9)


def test_csm_plus_ra_works_on_fast_path():
    """CSM+RA uses only the headline CSM/RA/loss, so it works without full=True."""
    m = fcf.gmm.measure(_mp(term=120), _basis(mortality_cv=0.1), full=False)
    assert np.all(np.isfinite(pricing.csm_plus_ra(m)))


def test_profit_margin_needs_full():
    """Profit margin needs the premium cash flows -- the fast path has none."""
    m = fcf.gmm.measure(_mp(term=120), _basis(), full=False)
    with pytest.raises(ValueError, match="full=True"):
        pricing.profit_margin(m)


# ---------------------------------------------------------------------------
# Profit signature
# ---------------------------------------------------------------------------

def test_signature_reconciles_to_csm_plus_ra():
    """The IFRS 17 profit signature (per-year insurance service result) discounts
    back, mid-year, to approximately the portfolio CSM+RA."""
    m = fcf.gmm.measure(_mp(premium=200_000.0, term=120),
                        _basis(mortality_cv=0.10, discount_annual=0.03))
    sig = pricing.signature(m, period_months=12)
    assert sig.profit.shape[0] == 10                       # 120 months / 12
    assert np.isclose(sig.present_value(0.03), float(pricing.csm_plus_ra(m).sum()),
                      rtol=0.01)


# ---------------------------------------------------------------------------
# IRR / break-even (pure functions on a shareholder cash-flow stream)
# ---------------------------------------------------------------------------

def test_irr_handcalc():
    assert np.isclose(pricing.irr(np.array([-100.0, 110.0])), 0.10, atol=1e-6)
    assert np.isclose(pricing.irr(np.array([-100.0, 0.0, 121.0])), 0.10, atol=1e-6)


def test_irr_needs_sign_change():
    with pytest.raises(ValueError, match="change sign"):
        pricing.irr(np.array([10.0, 20.0, 30.0]))          # all positive


def test_break_even_year():
    assert pricing.break_even_year(np.array([-100.0, 50.0, 60.0])) == 3
    assert pricing.break_even_year(np.array([-100.0, 50.0, 40.0])) == -1
    assert pricing.break_even_year(np.array([-100.0, 150.0])) == 2


# ---------------------------------------------------------------------------
# Traditional / statutory profit test -- net-level-premium reserve
# ---------------------------------------------------------------------------

from dataclasses import replace


def _endowment():
    return fcf.ModelPoints.single(
        issue_age=40, premium=0.0, term_months=120, benefits={"DEATH": 100_000_000.0},
        maturity_benefit=100_000_000.0, calculation_methods=_CM)


def _stat_basis(rate=0.025):
    return fcf.Basis(mortality_annual=_flat(0.01), lapse_annual=_flat(0.0),
                     discount_annual=rate, ra_confidence=0.75, mortality_cv=0.0,
                     coverages=(fcf.CoverageRate("DEATH", _flat(0.01)),))


def test_statutory_reserve_builds_for_an_endowment():
    """The net-premium reserve (no commutation -- a projection PV) starts ~0 and
    rises toward the maturity sum for an endowment."""
    stat = _stat_basis()
    V, net = pricing.statutory_reserve(_endowment(), stat)
    assert np.isclose(V[0, 0], 0.0, atol=1.0)             # nil at issue
    assert V[0, 60] > V[0, 12] > 0.0                      # rising
    assert net[0] > 0.0


def test_statutory_reserve_is_self_financing():
    """THE anchor: with the pricing basis = the statutory basis and the net
    premium, the reserve exactly funds the benefits -- profit is zero every
    period (to floating-point), so the reserve and the emergence recursion agree."""
    stat = _stat_basis()
    _, net = pricing.statutory_reserve(_endowment(), stat)
    mp_net = replace(_endowment(), premium=net)
    sig = pricing.statutory_profit_signature(mp_net, stat, stat)
    assert np.max(np.abs(sig.profit)) < 1e-3


def test_premium_loading_emerges_as_profit():
    """A gross premium above the net premium releases the loading as profit."""
    stat = _stat_basis()
    _, net = pricing.statutory_reserve(_endowment(), stat)
    sig = pricing.statutory_profit_signature(
        replace(_endowment(), premium=net * 1.10), stat, stat)
    assert sig.total > 0.0 and np.all(sig.profit >= -1e-3)


def test_interest_spread_emerges_as_profit():
    """Earning above the valuation rate on the reserve releases the spread."""
    stat = _stat_basis(rate=0.025)
    _, net = pricing.statutory_reserve(_endowment(), stat)
    mp_net = replace(_endowment(), premium=net)
    flat_sig = pricing.statutory_profit_signature(mp_net, stat, stat)
    spread_sig = pricing.statutory_profit_signature(mp_net, stat, stat, earned_rate=0.04)
    assert spread_sig.total > flat_sig.total                # earning more -> profit


def test_statutory_signature_rejects_router():
    stat = _stat_basis()
    with pytest.raises(NotImplementedError, match="single Basis"):
        pricing.statutory_profit_signature(_endowment(), fcf.samples.basis("gmm"), stat)
