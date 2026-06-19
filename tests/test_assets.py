"""Assets and the solvency balance sheet -- hand-calc anchors."""
import numpy as np

import fastcashflow as fcf
from fastcashflow import assets, alm
from fastcashflow import solvency as sv
from fastcashflow.engine import measure

from conftest import make_death_basis, PATTERNS


def _basis(rate=0.03):
    return make_death_basis(mortality_q=0.001, lapse_q=0.0, discount_annual=rate,
                            mortality_cv=0.0)


def _mp():
    return fcf.ModelPoints.single(40, 50_000.0, 120, benefits={"DEATH": 1e8},
                                  calculation_methods=PATTERNS)


def _parallel_curves(n=10, base=0.03, bp=0.001):
    """A small +/- parallel curve shock (relative form) for clean immunisation."""
    rel = bp / base
    return (sv.shock_curve(np.full(n, rel), up=True),
            sv.shock_curve(np.full(n, -rel), up=False))


def test_portfolio_value_sums_holdings():
    """A bond (priced at the curve) plus equity / cash (carried) sum up."""
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(100.0, 0.05, 10, 1), assets.Equity(50.0), assets.Cash(10.0)))
    v = assets.portfolio_value(p, 0.05)
    assert np.isclose(v, 100.0 + 50.0 + 10.0)          # the bond is at par at 5%
    # a per-year curve gives the same value as the equivalent flat scalar
    assert np.isclose(assets.portfolio_value(p, np.full(10, 0.05)), v)


def test_available_capital_hand_calc():
    """Available capital = portfolio value - (BEL + risk margin)."""
    assert np.isclose(assets.available_capital(160.0, 90.0, 20.0), 50.0)
    # a portfolio short of the liability is insolvent (negative own funds)
    assert assets.available_capital(80.0, 90.0, 20.0) < 0.0


def test_net_interest_scr_formula_and_sign():
    """A rate rise lowers both the asset value and the BEL; the net interest SCR
    is the worst-of own-funds loss over the up / down shocks."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(alm.Bond(2_000_000.0, 0.03, 10, 1),))
    curves = _parallel_curves()
    base_pv = assets.portfolio_value(p, basis.discount_annual)
    base_bel = float(measure(mp, basis, full=False).bel.sum())
    _, b_up = curves[0].apply(mp, basis)
    assert assets.portfolio_value(p, b_up.discount_annual) < base_pv      # asset falls
    assert float(measure(mp, b_up, full=False).bel.sum()) < base_bel      # BEL falls

    def nav(b):
        return (assets.portfolio_value(p, b.discount_annual)
                - float(measure(mp, b, full=False).bel.sum()))
    nav_base = base_pv - base_bel
    expected = max(0.0, max(nav_base - nav(s.apply(mp, basis)[1]) for s in curves))
    assert np.isclose(assets.net_interest_scr(p, mp, basis, interest_curves=curves),
                      expected)


def test_matched_book_net_interest_near_zero():
    """A bond book sized to the liability DV01 immunises the net interest SCR."""
    mp, basis = _mp(), _basis()
    liab_dv01 = alm.liability_dv01(mp, basis)
    per_face = alm.bond_duration(alm.Bond(100.0, 0.03, 10, 1), 0.03).dv01
    face = liab_dv01 / per_face * 100.0
    p = assets.AssetPortfolio(holdings=(alm.Bond(face, 0.03, 10, 1),))
    ni = assets.net_interest_scr(p, mp, basis, interest_curves=_parallel_curves())
    assert ni < abs(face) * 1e-4                 # negligible vs the book (immunised)


def test_unmatched_book_positive_net_interest():
    """An all-cash book leaves the liability's rate move unhedged -> positive SCR."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(assets.Cash(10_000_000.0),))
    assert assets.net_interest_scr(p, mp, basis, interest_curves=_parallel_curves()) > 0.0


def test_equity_scr_by_type():
    """Equity SCR = market value times the regime price-fall shock; unknown type raises."""
    import pytest
    p = assets.AssetPortfolio(holdings=(
        assets.Equity(1_000_000.0, "developed"), assets.Equity(500_000.0, "emerging")))
    assert np.isclose(assets.equity_scr(p, fcf.SOLVENCY2),
                      1_000_000.0 * 0.35 + 500_000.0 * 0.48)
    assert np.isclose(assets.equity_scr(p, fcf.KICS), assets.equity_scr(p, fcf.SOLVENCY2))
    bad = assets.AssetPortfolio(holdings=(assets.Equity(1.0, "exotic"),))
    with pytest.raises(ValueError, match="risk_type"):
        assets.equity_scr(bad, fcf.SOLVENCY2)


def test_property_scr():
    p = assets.AssetPortfolio(holdings=(assets.Property(2_000_000.0),))
    assert np.isclose(assets.property_scr(p, fcf.SOLVENCY2), 2_000_000.0 * 0.25)


def test_market_module_aggregates_sub_risks():
    """The market module is sqrt(c^T R c) over (interest, equity, property)."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        assets.Equity(3_000_000.0, "developed"), assets.Property(1_000_000.0)))
    # K-ICS: no interest curves -> interest component 0
    eq = assets.equity_scr(p, fcf.KICS)
    pr = assets.property_scr(p, fcf.KICS)
    c = np.array([0.0, eq, pr])
    R = np.array([[1.0, 0.25, 0.25], [0.25, 1.0, 0.25], [0.25, 0.25, 1.0]])
    assert np.isclose(assets.market_module_scr(p, mp, basis, regime=fcf.KICS),
                      np.sqrt(c @ R @ c))


def test_assess_solvency_components():
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(2_600_000.0, 0.03, 10, 1), assets.Cash(3_000_000.0)))
    a = assets.assess_solvency(p, mp, basis, regime=fcf.SOLVENCY2)
    assert np.isclose(a.total_scr, a.insurance_scr + a.net_interest_scr)
    assert np.isclose(a.solvency_ratio, a.available_capital / a.total_scr)
    assert np.isclose(a.available_capital, a.portfolio_value - (a.bel + a.risk_margin))
    assert a.net_interest_scr != a.liability_interest_capital   # net replaces liability-only


def test_assess_solvency_kics_fallback():
    """K-ICS carries no interest curves -> net interest falls back to the liability
    figure (no crash, no silently dropped interest risk)."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(assets.Cash(8_000_000.0),))
    a = assets.assess_solvency(p, mp, basis, regime=fcf.KICS)
    assert a.net_interest_scr == a.liability_interest_capital
    assert np.isfinite(a.solvency_ratio)
