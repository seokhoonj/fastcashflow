"""Assets and the solvency balance sheet -- hand-calc anchors."""
import numpy as np
import pytest

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


def test_asset_portfolio_value_sums_holdings():
    """A bond (priced at the curve) plus equity / cash (carried) sum up."""
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(100.0, 0.05, 10, 1), assets.Equity(50.0), assets.Cash(10.0)))
    v = assets.asset_portfolio_value(p, 0.05)
    assert np.isclose(v, 100.0 + 50.0 + 10.0)          # the bond is at par at 5%
    # a per-year curve gives the same value as the equivalent flat scalar
    assert np.isclose(assets.asset_portfolio_value(p, np.full(10, 0.05)), v)


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
    base_pv = assets.asset_portfolio_value(p, basis.discount_annual)
    base_bel = float(measure(mp, basis, full=False).bel.sum())
    _, b_up = curves[0].apply(mp, basis)
    assert assets.asset_portfolio_value(p, b_up.discount_annual) < base_pv      # asset falls
    assert float(measure(mp, b_up, full=False).bel.sum()) < base_bel      # BEL falls

    def nav(b):
        return (assets.asset_portfolio_value(p, b.discount_annual)
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


def _kics_scenarios():
    """A simple K-ICS five-scenario set: up +1pp parallel, flat short-up/long-down."""
    up = np.full(10, 0.01)
    flat = np.concatenate([np.full(3, 0.004), np.full(7, -0.004)])
    return sv.KICSInterest.from_spreads(up=up, down=-up, flat=flat, steep=-flat,
                                        mean_reversion=np.full(10, 0.0))


def test_net_interest_kics_five_scenario_on_nav():
    """K-ICS net interest aggregates the five scenarios on NET asset value by the
    handbook p.205 formula -- recompute the five NAV-decrease amounts by hand."""
    import math
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(alm.Bond(2_000_000.0, 0.03, 10, 1),))
    ki = _kics_scenarios()
    nav_base = (assets.asset_portfolio_value(p, basis.discount_annual)
                - float(measure(mp, basis, full=False).bel.sum()))

    def amt(stress):
        _, b = stress.apply(mp, basis)
        nav = (assets.asset_portfolio_value(p, b.discount_annual)
               - float(measure(mp, b, full=False).bel.sum()))
        return nav_base - nav
    up, down = max(0.0, amt(ki.up)), max(0.0, amt(ki.down))
    flat, steep = max(0.0, amt(ki.flat)), max(0.0, amt(ki.steep))
    mr = amt(ki.mean_reversion)                          # signed
    expected = math.sqrt(max(up, down)**2 + max(flat, steep)**2) + mr
    assert np.isclose(assets.net_interest_kics_scr(p, mp, basis, scenarios=ki), expected)


def test_assess_solvency_kics_interest_enters_market_module():
    """K-ICS interest now flows into the market module (net), not zero as before,
    and not into the insurance module (no double count)."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(assets.Cash(10_000_000.0),))    # unhedged
    ki = _kics_scenarios()
    without = assets.assess_solvency(p, mp, basis, regime=fcf.KICS)
    with_ = assets.assess_solvency(p, mp, basis, regime=fcf.KICS, interest_scenarios=ki)
    assert without.net_interest_scr == 0.0               # K-ICS supplied no curves before
    assert with_.net_interest_scr > 0.0                  # now the five-scenario net amount
    assert with_.market_module_scr > without.market_module_scr
    assert with_.insurance_scr == without.insurance_scr  # interest not in the insurance module


def test_sii_toplevel_three_module_diversifies():
    """Solvency II (Annex IV) aggregates the (life, market, credit) modules at
    all-pairwise 0.25 -- the same values as K-ICS, and below the simple sum."""
    import math
    ins, mkt, cr = 300.0, 400.0, 200.0
    sii = assets.aggregate_required_capital(ins, mkt, cr, regime=fcf.SOLVENCY2)
    kics = assets.aggregate_required_capital(ins, mkt, cr, regime=fcf.KICS)
    c = np.array([ins, mkt, cr])
    R = np.array([[1.0, .25, .25], [.25, 1.0, .25], [.25, .25, 1.0]])
    assert np.isclose(sii, math.sqrt(c @ R @ c))
    assert np.isclose(sii, kics)                     # 3-module values coincide
    assert sii < ins + mkt + cr                      # diversification vs the old simple sum


def test_sii_toplevel_four_module_nonlife_default_half():
    """With a general (non-life) module, Solvency II's non-life<->credit is 0.5
    (vs K-ICS 0.25), so the SII top-level aggregate exceeds the K-ICS one."""
    import math
    ins, gen, mkt, cr = 300.0, 250.0, 400.0, 200.0
    sii = assets.aggregate_required_capital(ins, mkt, cr, regime=fcf.SOLVENCY2,
                                            general_insurance=gen)
    kics = assets.aggregate_required_capital(ins, mkt, cr, regime=fcf.KICS,
                                             general_insurance=gen)
    c = np.array([ins, gen, mkt, cr])
    R = np.array([[1.0, 0.0, .25, .25], [0.0, 1.0, .25, .50],
                  [.25, .25, 1.0, .25], [.25, .50, .25, 1.0]])
    assert np.isclose(sii, math.sqrt(c @ R @ c))
    assert sii > kics                                # 0.5 vs 0.25 on general-credit


def test_equity_scr_by_type():
    """Equity SCR aggregates the per-type amounts at the 0.75 inter-type correlation
    (handbook 4-3); a single type is just its amount; unknown type raises."""
    import pytest
    one = assets.AssetPortfolio(holdings=(assets.Equity(3_000_000.0, "developed"),))
    assert np.isclose(assets.equity_scr(one, fcf.KICS), 3_000_000.0 * 0.35)
    p = assets.AssetPortfolio(holdings=(
        assets.Equity(1_000_000.0, "developed"), assets.Equity(500_000.0, "emerging")))
    dev, emg = 1_000_000.0 * 0.35, 500_000.0 * 0.48
    assert np.isclose(assets.equity_scr(p, fcf.KICS),
                      np.sqrt(dev**2 + emg**2 + 2 * 0.75 * dev * emg))
    assert assets.equity_scr(p, fcf.KICS) < dev + emg            # diversification
    bad = assets.AssetPortfolio(holdings=(assets.Equity(1.0, "exotic"),))
    with pytest.raises(ValueError, match="risk_type"):
        assets.equity_scr(bad, fcf.SOLVENCY2)


def test_equity_subtypes_shocks():
    """K-ICS equity sub-types each carry their handbook 4-3 shock (preferred is
    rating-based, tested separately)."""
    for risk_type, shock in [("infrastructure", 0.20), ("long_term", 0.20),
                             ("other", 0.49)]:
        p = assets.AssetPortfolio(holdings=(assets.Equity(1_000_000.0, risk_type),))
        assert np.isclose(assets.equity_scr(p, fcf.KICS), 1_000_000.0 * shock)


def test_property_scr():
    p = assets.AssetPortfolio(holdings=(assets.Property(2_000_000.0),))
    assert np.isclose(assets.property_scr(p, fcf.SOLVENCY2), 2_000_000.0 * 0.25)


def test_market_module_aggregates_sub_risks():
    """The market module is sqrt(c^T R c) over (interest, equity, property, FX,
    concentration) with the table-19 correlation."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        assets.Equity(3_000_000.0, "developed"), assets.Property(1_000_000.0)))
    # K-ICS: no interest curves -> interest 0; no foreign currency -> FX 0; the
    # property holding does exceed the 6% individual limit -> a concentration charge
    eq = assets.equity_scr(p, fcf.KICS)
    pr = assets.property_scr(p, fcf.KICS)
    fx = assets.fx_scr(p, fcf.KICS, basis.discount_annual)
    conc = assets.concentration_scr(p, fcf.KICS, basis.discount_annual)
    c = np.array([0.0, eq, pr, fx, conc])
    R = assets._MARKET_CORRELATION
    assert np.isclose(assets.market_module_scr(p, mp, basis, regime=fcf.KICS),
                      np.sqrt(c @ R @ c))


def test_operational_scr_kics():
    """K-ICS operational = max(premium x 3.5%, BEL x 0.4%)."""
    mp, basis = _mp(), _basis()
    m = measure(mp, basis, full=True)
    bel = max(0.0, float(m.bel.sum()))
    prem = max(0.0, float(m.cashflows.premium_cf[:, :12].sum()))
    expected = max(prem * 0.035, bel * 0.004)
    assert np.isclose(assets.operational_scr(mp, basis, fcf.KICS), expected)
    assert assets.operational_scr(mp, basis, fcf.KICS) > 0.0


def test_operational_scr_sii_cap():
    """Solvency II operational is capped at 0.3 x BSCR."""
    mp, basis = _mp(), _basis()
    m = measure(mp, basis, full=True)
    bel = max(0.0, float(m.bel.sum()))
    prem = max(0.0, float(m.cashflows.premium_cf[:, :12].sum()))
    op_uncapped = max(prem * 0.04, bel * 0.0045)
    # a tiny BSCR makes the 0.3 x BSCR cap bite
    small = op_uncapped / 10.0
    assert np.isclose(assets.operational_scr(mp, basis, fcf.SOLVENCY2, bscr=small),
                      0.30 * small)
    # a large BSCR leaves it uncapped
    assert np.isclose(assets.operational_scr(mp, basis, fcf.SOLVENCY2, bscr=1e12),
                      op_uncapped)


def test_effective_maturity_hand_calc():
    """Effective maturity = sum(t x CF) / sum(CF). A 10y 3% annual bond:
    sum(t x CF) = 3 x (1+..+10) + 100 x 10 = 1165; sum(CF) = 130; 1165/130."""
    b = alm.Bond(100.0, 0.03, 10, 1)
    assert np.isclose(alm.effective_maturity(b), 1165.0 / 130.0)


def test_credit_bucket_boundaries():
    """Bucket k is k < m <= k+1 (so 0-1 is index 0), capped at the 14+ bucket."""
    assert assets._credit_bucket(0.5) == 0
    assert assets._credit_bucket(1.0) == 0
    assert assets._credit_bucket(1.5) == 1
    assert assets._credit_bucket(10.0) == 9
    assert assets._credit_bucket(14.0) == 13
    assert assets._credit_bucket(20.0) == 14


def test_credit_scr_kics_hand_calc():
    """K-ICS credit SCR = market value x factor[rating row][maturity bucket].
    An AA corporate bond (10y 3%) has effective maturity 8.96 -> bucket 8, and the
    corporate '1-2' row there is 2.0% (handbook table 30)."""
    b = alm.Bond(100.0, 0.03, 10, 1, credit_rating="AA", exposure_class="corporate")
    p = assets.AssetPortfolio(holdings=(b, assets.Cash(1000.0), assets.Equity(500.0)))
    mv = alm.bond_value(b, 0.03)
    assert np.isclose(assets.credit_scr(p, fcf.KICS, 0.03), mv * 0.02)   # only the bond


def test_credit_scr_sii_spread():
    """Solvency II credit (Art 176 spread) = market value times a stress that is
    piecewise-linear in modified duration by credit quality step."""
    b = alm.Bond(100.0, 0.03, 10, 1, credit_rating="A")        # CQS 2
    mod = alm.bond_duration(b, 0.03).modified                  # ~8.53 -> bucket 5-10
    factor = 0.070 + 0.007 * (mod - 5)                         # a + b x (dur - 5)
    p = assets.AssetPortfolio(holdings=(b, assets.Cash(1000.0)))
    assert np.isclose(assets.credit_scr(p, fcf.SOLVENCY2, 0.03),
                      alm.bond_value(b, 0.03) * factor)
    # the piecewise stress at representative points
    assert np.isclose(assets._sii_spread_stress("AAA", 3), 0.009 * 3)        # 0-5
    assert np.isclose(assets._sii_spread_stress("BBB", 12), 0.200 + 0.010 * 2)  # 10-15
    assert np.isclose(assets._sii_spread_stress("BB", 25), 0.466 + 0.005 * 5)   # 20+


def test_credit_scr_rating_and_class():
    """A lower rating and a riskier exposure class both raise the factor."""
    aa = alm.Bond(100.0, 0.03, 10, 1, credit_rating="AA", exposure_class="corporate")
    bb = alm.Bond(100.0, 0.03, 10, 1, credit_rating="BB", exposure_class="corporate")
    sec = alm.Bond(100.0, 0.03, 10, 1, credit_rating="BB", exposure_class="securitisation")
    s_aa = assets.credit_scr(assets.AssetPortfolio(holdings=(aa,)), fcf.KICS, 0.03)
    s_bb = assets.credit_scr(assets.AssetPortfolio(holdings=(bb,)), fcf.KICS, 0.03)
    s_sec = assets.credit_scr(assets.AssetPortfolio(holdings=(sec,)), fcf.KICS, 0.03)
    assert s_bb > s_aa                       # BB (grade 5) charges more than AA (1-2)
    assert s_sec > s_bb                      # securitisation BB charges more than corporate
    bad = alm.Bond(100.0, 0.03, 10, 1, exposure_class="exotic")
    with pytest.raises(ValueError, match="exposure_class"):
        assets.credit_scr(assets.AssetPortfolio(holdings=(bad,)), fcf.KICS, 0.03)


def test_fx_scr_hand_calc():
    """K-ICS FX SCR = worse of won-up / won-down, aggregated at 0.5 correlation.
    USD 1000 (shock 25% -> 250) and JPY 500 (40% -> 200), both long, so the won-up
    scenario binds: sqrt(250^2 + 200^2 + 2 x 0.5 x 250 x 200)."""
    p = assets.AssetPortfolio(holdings=(
        assets.Cash(1000.0, currency="USD"), assets.Cash(500.0, currency="JPY"),
        assets.Cash(2000.0)))                          # 2000 won: no FX
    expected = np.sqrt(250.0**2 + 200.0**2 + 2 * 0.5 * 250.0 * 200.0)
    assert np.isclose(assets.fx_scr(p, fcf.KICS, 0.03), expected)


def test_fx_scr_sii_flat_25():
    """Solvency II currency risk (Art 188) is a flat 25% per foreign currency,
    summed (no diversification); any currency applies (no table lookup)."""
    p = assets.AssetPortfolio(holdings=(
        assets.Cash(1000.0, currency="USD"), assets.Cash(500.0, currency="EUR"),
        assets.Cash(2000.0)))
    assert np.isclose(assets.fx_scr(p, fcf.SOLVENCY2, 0.03), 0.25 * 1000 + 0.25 * 500)
    # an unlisted currency is fine under Solvency II (no table)
    q = assets.AssetPortfolio(holdings=(assets.Cash(800.0, currency="XYZ"),))
    assert np.isclose(assets.fx_scr(q, fcf.SOLVENCY2, 0.03), 0.25 * 800)


def test_fx_scr_short_position_up_scenario():
    """A net-short currency (a foreign liability) loses under the won-down scenario:
    a short EUR 800 -> 35% x 800."""
    p = assets.AssetPortfolio(holdings=(assets.Cash(-800.0, currency="EUR"),))
    assert np.isclose(assets.fx_scr(p, fcf.KICS, 0.03), 0.35 * 800.0)
    bad = assets.AssetPortfolio(holdings=(assets.Cash(1.0, currency="XXX"),))
    with pytest.raises(ValueError, match="currency"):
        assets.fx_scr(bad, fcf.KICS, 0.03)


def test_market_module_includes_fx_negative_correlation():
    """FX enters the market module; equity <-> FX is NEGATIVE 0.25, so an equity +
    FX book diversifies: sqrt(eq^2 + fx^2 - 2 x 0.25 x eq x fx)."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        assets.Equity(1000.0, "developed"), assets.Cash(1000.0, currency="USD")))
    eq = assets.equity_scr(p, fcf.KICS)                # 350
    fx = assets.fx_scr(p, fcf.KICS, basis.discount_annual)   # 250
    got = assets.market_module_scr(p, mp, basis, regime=fcf.KICS)
    assert np.isclose(got, np.sqrt(eq**2 + fx**2 + 2 * (-0.25) * eq * fx))


def test_concentration_scr_hand_calc():
    """K-ICS concentration = sqrt(counterparty^2 + property^2). Total assets 10m:
    an AA issuer (band 1-2: limit 4%, factor 15%) with 1m exposure charges
    (1m - 400k) x 15% = 90k; a 1m property charges (1m - 600k) x 20% = 80k."""
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(1_000_000.0, 0.03, 5, 1, credit_rating="AA", issuer="KB"),
        assets.Property(1_000_000.0), assets.Cash(8_000_000.0)))
    cp = (1_000_000.0 - 10_000_000.0 * 0.04) * 0.15
    pr = (1_000_000.0 - 10_000_000.0 * 0.06) * 0.20
    got = assets.concentration_scr(p, fcf.KICS, 0.03, total_assets=10_000_000.0)
    assert np.isclose(got, np.sqrt(cp**2 + pr**2))


def test_concentration_scr_untagged_is_zero():
    """A book with no tagged issuers and no property has no concentration charge,
    under both regimes."""
    p = assets.AssetPortfolio(holdings=(alm.Bond(1e6, 0.03, 5, 1), assets.Cash(1e6)))
    assert assets.concentration_scr(p, fcf.KICS, 0.03) == 0.0
    assert assets.concentration_scr(p, fcf.SOLVENCY2, 0.03) == 0.0


def test_concentration_scr_sii_excess():
    """Solvency II concentration (Art 184-187): single-name excess
    max(0, exposure - CT(CQS) x assets) x g(CQS). A BBB issuer is CQS 3 (CT 1.5%,
    g 27%); an AA issuer is CQS 1 (CT 3%, g 12%)."""
    bbb = assets.AssetPortfolio(holdings=(
        alm.Bond(1e6, 0.03, 5, 1, credit_rating="BBB", issuer="X"),))
    assert np.isclose(assets.concentration_scr(bbb, fcf.SOLVENCY2, 0.03, total_assets=10e6),
                      max(0.0, 1e6 - 10e6 * 0.015) * 0.27)
    aa = assets.AssetPortfolio(holdings=(
        alm.Bond(1e6, 0.03, 5, 1, credit_rating="AA", issuer="Y"),))
    assert np.isclose(assets.concentration_scr(aa, fcf.SOLVENCY2, 0.03, total_assets=10e6),
                      max(0.0, 1e6 - 10e6 * 0.03) * 0.12)


def test_concentration_band_by_rating():
    """Lower-rated issuers fall in a tighter band (lower limit, higher factor)."""
    aa = assets.AssetPortfolio(holdings=(
        alm.Bond(1e6, 0.03, 5, 1, credit_rating="AA", issuer="X"),))
    bb = assets.AssetPortfolio(holdings=(
        alm.Bond(1e6, 0.03, 5, 1, credit_rating="BB", issuer="X"),))
    s_aa = assets.concentration_scr(aa, fcf.KICS, 0.03, total_assets=10e6)  # 4%/15%
    s_bb = assets.concentration_scr(bb, fcf.KICS, 0.03, total_assets=10e6)  # 1.5%/50%
    assert np.isclose(s_aa, (1e6 - 10e6 * 0.04) * 0.15)
    assert np.isclose(s_bb, (1e6 - 10e6 * 0.015) * 0.50)
    assert s_bb > s_aa


def test_concentration_property_whole_book_limit():
    """Property concentration takes the worse of the individual (6%) and whole-book
    (25%) limits. Two 2m properties on 10m assets: whole-book excess (4m - 2.5m)."""
    p = assets.AssetPortfolio(holdings=(assets.Property(2e6), assets.Property(2e6)))
    got = assets.concentration_scr(p, fcf.KICS, 0.03, total_assets=10e6)
    individual = np.sqrt(2 * ((2e6 - 10e6 * 0.06) * 0.20) ** 2)
    whole = (4e6 - 10e6 * 0.25) * 0.20
    assert np.isclose(got, max(individual, whole))


def test_assess_solvency_components():
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(2_600_000.0, 0.03, 10, 1), assets.Cash(3_000_000.0)))
    a = assets.assess_solvency(p, mp, basis, regime=fcf.SOLVENCY2)
    # no equity/property -> the market module is just the net interest SCR; the SII
    # top-level BSCR aggregates insurance + market + Art-176 credit at all-pairwise
    # 0.25 (Annex IV); the total adds operational on top
    assert np.isclose(a.market_module_scr, a.net_interest_scr)
    assert a.credit_scr > 0.0                               # SII Art-176 spread on the bond
    m = np.array([a.insurance_scr, a.market_module_scr, a.credit_scr])
    R = np.array([[1.0, .25, .25], [.25, 1.0, .25], [.25, .25, 1.0]])
    assert np.isclose(a.bscr, np.sqrt(m @ R @ m))
    assert np.isclose(a.basic_required_capital, a.bscr + a.operational_scr)
    assert a.tax_adjustment == 0.0                          # no tax relief by default
    assert np.isclose(a.total_scr, a.basic_required_capital - a.tax_adjustment)
    assert np.isclose(a.solvency_ratio, a.available_capital / a.total_scr)
    assert np.isclose(a.available_capital, a.asset_portfolio_value - (a.bel + a.risk_margin))


def test_tax_adjustment_loss_absorption():
    """K-ICS chapter 7: the tax adjustment = min(basic x tax_rate, recoverability
    limit) is subtracted from the basic required capital, raising the ratio."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(2_600_000.0, 0.03, 10, 1), assets.Cash(3_000_000.0)))
    base = assets.assess_solvency(p, mp, basis, regime=fcf.KICS)
    taxed = assets.assess_solvency(p, mp, basis, regime=fcf.KICS, tax_rate=0.22)
    assert np.isclose(taxed.tax_adjustment, base.basic_required_capital * 0.22)
    assert np.isclose(taxed.total_scr, base.basic_required_capital * (1 - 0.22))
    assert taxed.solvency_ratio > base.solvency_ratio       # tax relief lowers the SCR
    # the recoverability limit caps the relief
    cap = base.basic_required_capital * 0.22 / 3
    capped = assets.assess_solvency(p, mp, basis, regime=fcf.KICS, tax_rate=0.22,
                                    tax_recoverability_limit=cap)
    assert np.isclose(capped.tax_adjustment, cap)


def test_assess_solvency_kics_no_curves():
    """K-ICS supplies no interest curves -> the net interest component is 0;
    an all-cash book then has total SCR == the insurance SCR (no market risk)."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(assets.Cash(8_000_000.0),))
    a = assets.assess_solvency(p, mp, basis, regime=fcf.KICS)
    assert a.net_interest_scr == 0.0
    assert np.isclose(a.bscr, a.insurance_scr)              # all-cash -> no market risk
    assert np.isclose(a.total_scr, a.bscr + a.operational_scr)
    assert np.isfinite(a.solvency_ratio)


def test_top_level_aggregation_kics_vs_sii():
    """K-ICS (table 3) and Solvency II (Annex IV) both aggregate insurance, market
    and credit at all-pairwise 0.25 -- the 3-module correlation values coincide, so
    each BSCR is the sqrt aggregation of its own module amounts."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(2_000_000.0, 0.03, 10, 1), assets.Equity(3_000_000.0, "developed")))
    k = assets.assess_solvency(p, mp, basis, regime=fcf.KICS)
    s = assets.assess_solvency(p, mp, basis, regime=fcf.SOLVENCY2)
    R = np.array([[1.0, 0.25, 0.25], [0.25, 1.0, 0.25], [0.25, 0.25, 1.0]])
    # SII: sqrt aggregation at 0.25 (Annex IV), below the simple module sum
    assert s.credit_scr > 0.0
    m_s = np.array([s.insurance_scr, s.market_module_scr, s.credit_scr])
    assert np.isclose(s.bscr, np.sqrt(m_s @ R @ m_s))
    assert s.bscr < s.insurance_scr + s.market_module_scr + s.credit_scr     # diversifies
    # K-ICS: the bond now carries a credit charge -> same 3-module sqrt aggregation
    assert k.credit_scr > 0.0
    m = np.array([k.insurance_scr, k.market_module_scr, k.credit_scr])
    assert np.isclose(k.bscr, np.sqrt(m @ R @ m))                            # K-ICS sqrt
    assert np.isclose(k.total_scr, k.bscr + k.operational_scr)               # + operational


def test_equity_now_charges_scr():
    """Equity now raises BOTH available capital and the SCR (the v1 overstatement
    where it lifted only the numerator is fixed)."""
    mp, basis = _mp(), _basis()
    base = assets.AssetPortfolio(holdings=(
        alm.Bond(2_000_000.0, 0.03, 10, 1), assets.Cash(5_000_000.0)))
    with_eq = assets.AssetPortfolio(holdings=base.holdings + (assets.Equity(3_000_000.0),))
    a0 = assets.assess_solvency(base, mp, basis, regime=fcf.SOLVENCY2)
    a1 = assets.assess_solvency(with_eq, mp, basis, regime=fcf.SOLVENCY2)
    assert a1.available_capital > a0.available_capital
    assert a1.total_scr > a0.total_scr                       # equity now charges market SCR
    assert np.isclose(a1.equity_scr, 3_000_000.0 * 0.35)


def test_general_insurance_fourth_module():
    """A caller-supplied general (P&C) SCR enters the BSCR as a 4th top-level module
    (table 3: life-vs-general 0, else 0.25), matching disclosed K-ICS structure."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(2_000_000.0, 0.03, 10, 1, credit_rating="A"), assets.Cash(3_000_000.0)))
    gen = 800_000.0
    a = assets.assess_solvency(p, mp, basis, regime=fcf.KICS, general_insurance_scr=gen)
    m = np.array([a.insurance_scr, gen, a.market_module_scr, a.credit_scr])
    R = np.array([[1, 0, .25, .25], [0, 1, .25, .25],
                  [.25, .25, 1, .25], [.25, .25, .25, 1]])
    assert np.isclose(a.general_insurance_scr, gen)
    assert np.isclose(a.bscr, np.sqrt(m @ R @ m))
    # zero general -> the 3-module aggregation (unchanged)
    a0 = assets.assess_solvency(p, mp, basis, regime=fcf.KICS)
    assert a0.general_insurance_scr == 0.0
    assert a.bscr > a0.bscr                                  # adding a module raises BSCR


def test_general_insurance_disclosed_reproduction():
    """The 4-module table-3 aggregation reproduces a disclosed K-ICS basic required
    capital (Hanwha Life FY2025, KRW thousands): sqrt over the 4 modules + operational."""
    life, pc, mkt, cr = 10_654_450_301, 542_312_823, 7_191_902_855, 2_495_852_957
    op = 1_648_722_264
    m = np.array([life, pc, mkt, cr], dtype=float)
    R = np.array([[1, 0, .25, .25], [0, 1, .25, .25],
                  [.25, .25, 1, .25], [.25, .25, .25, 1]])
    basic = np.sqrt(m @ R @ m) + op
    assert np.isclose(basic, 16_977_612_719, rtol=0, atol=2)   # disclosed, to the won


def test_aggregate_required_capital_reproduces_disclosures():
    """The public top-level aggregation reproduces disclosed K-ICS basic required
    capital from the published module amounts. A pure-life book (general = 0) and a
    life + P&C book (general module added) both match the disclosure to the won."""
    # life insurer, general insurance = 0 (KRW millions)
    basic = assets.aggregate_required_capital(
        11_628_115, 34_552_189, 4_166_014, regime=fcf.KICS, operational=1_083_844)
    assert np.isclose(basic, 41_624_006, rtol=0, atol=2)
    # life + general insurance (KRW thousands)
    basic_g = assets.aggregate_required_capital(
        10_654_450_301, 7_191_902_855, 2_495_852_957, regime=fcf.KICS,
        operational=1_648_722_264, general_insurance=542_312_823)
    assert np.isclose(basic_g, 16_977_612_719, rtol=0, atol=2)
    # the diversification effect = simple sum - aggregate (ex operational)
    agg = assets.aggregate_required_capital(11_628_115, 34_552_189, 4_166_014,
                                            regime=fcf.KICS)
    div = (11_628_115 + 34_552_189 + 4_166_014) - agg
    assert np.isclose(div, 9_806_156, rtol=0, atol=2)
    # Solvency II aggregates the modules at the Annex IV 0.25 correlation (+ operational)
    s = assets.aggregate_required_capital(100.0, 200.0, 50.0, regime=fcf.SOLVENCY2,
                                          operational=10.0)
    c = np.array([100.0, 200.0, 50.0])
    R = np.array([[1.0, .25, .25], [.25, 1.0, .25], [.25, .25, 1.0]])
    assert np.isclose(s, np.sqrt(c @ R @ c) + 10.0)


def test_preferred_equity_by_rating_table20():
    """Preferred equity (table 20) is charged by the issue's K-ICS grade: 1-2 grade
    4%, 3 grade 6%, 4 grade 11%, 5 grade 21%, 6+ grade 35%, unrated 35%."""
    for rating, shock in [("AA", 0.04), ("A", 0.06), ("BBB", 0.11), ("BB", 0.21),
                          ("B", 0.35), ("unrated", 0.35)]:
        p = assets.AssetPortfolio(holdings=(
            assets.Equity(1_000_000.0, "preferred", credit_rating=rating),))
        assert np.isclose(assets.equity_scr(p, fcf.KICS), 1_000_000.0 * shock)


# ---------------------------------------------------------------------------
# Dynamic asset engine -- Phase A: asset cash-flow projection
# ---------------------------------------------------------------------------

def test_asset_portfolio_cashflows_places_coupons_and_redemption():
    """Each bond's coupons and final redemption land on the monthly grid at
    round(time_years * 12); equity/property/cash carry no scheduled cash flow."""
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=1000, coupon_rate=0.05, maturity_years=3, frequency=1),
        alm.Bond(face=2000, coupon_rate=0.04, maturity_years=2, frequency=2),
        assets.Equity(market_value=5000),               # no scheduled CF
    ))
    cf = assets.asset_portfolio_cashflows(pf, 36)
    assert cf.shape == (37,)
    # bond1: 50 at 12/24/36 (+1000 at 36); bond2: 40 at 6/12/18/24 (+2000 at 24)
    assert np.isclose(cf[6], 40.0)
    assert np.isclose(cf[12], 50.0 + 40.0)
    assert np.isclose(cf[18], 40.0)
    assert np.isclose(cf[24], 50.0 + 2040.0)
    assert np.isclose(cf[36], 1050.0)
    assert np.isclose(cf.sum(), 40 + 90 + 40 + 2090 + 1050)
    # equity adds nothing
    assert np.isclose(cf.sum(), 3310.0)


def test_asset_portfolio_cashflows_drops_flows_beyond_horizon():
    """Cash flows past n_months are dropped (the horizon truncates the bond)."""
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=1000, coupon_rate=0.05, maturity_years=5, frequency=1),))
    cf = assets.asset_portfolio_cashflows(pf, 24)          # only years 1-2 fit
    assert cf.shape == (25,)
    assert np.isclose(cf[12], 50.0)
    assert np.isclose(cf[24], 50.0)                      # year-2 coupon, no redemption yet
    assert np.isclose(cf.sum(), 100.0)                  # years 3-5 (incl. 1000 face) dropped


def test_asset_portfolio_cashflows_rejects_bad_horizon():
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=1000, coupon_rate=0.05, maturity_years=3),))
    with pytest.raises(ValueError, match="n_months must be positive"):
        assets.asset_portfolio_cashflows(pf, 0)


def test_asset_value_path_runoff():
    """The held portfolio amortises as the bond runs off: it starts at the t=0
    portfolio value, a par bond stays at par on its coupon dates, and once the bond
    redeems only the flat (equity) holding remains."""
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(1000.0, 0.05, 3, 1), assets.Equity(500.0)))
    path = assets.asset_value_path(pf, 36, 0.05)
    assert path.shape == (37,)
    assert np.isclose(path[0], assets.asset_portfolio_value(pf, 0.05))  # t=0 == MV
    assert np.isclose(path[0], 1500.0)                 # par bond 1000 + equity 500
    assert np.isclose(path[12], 1500.0)                # par on the coupon date
    assert np.isclose(path[24], 1500.0)
    assert np.isclose(path[36], 500.0)                 # bond redeemed -> equity only
    assert path[6] > 1500.0                            # mid-period: carries accrued coupon
    assert np.all(path >= 0.0)


def test_asset_value_path_matches_mv_at_zero():
    """At month 0 the run-off value equals asset_portfolio_value for a mixed book."""
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(2000.0, 0.04, 5, 2), alm.Bond(1000.0, 0.06, 8, 1),
        assets.Equity(3000.0), assets.Cash(500.0)))
    path = assets.asset_value_path(pf, 120, 0.03)
    assert np.isclose(path[0], assets.asset_portfolio_value(pf, 0.03))
    assert np.isclose(path[-1], 3500.0)                # both bonds matured -> equity + cash


def test_asset_value_path_rejects_bad_horizon():
    pf = assets.AssetPortfolio(holdings=(alm.Bond(1000.0, 0.05, 3, 1),))
    with pytest.raises(ValueError, match="n_months must be positive"):
        assets.asset_value_path(pf, 0, 0.05)


def test_cashflow_gap_nets_asset_against_liability():
    """The gap folds the liability begin- and mid-month flows into one outflow per
    month and nets the projected asset cash flows against it on the shared grid."""
    mp, basis = _mp(), _basis()
    m = measure(mp, basis, full=True)
    flow_bom, flow_mid = alm.net_liability_cashflows(m)
    n_time = flow_mid.shape[0]
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=1000, coupon_rate=0.05, maturity_years=5, frequency=1),
        assets.Equity(market_value=5000),))                # no scheduled CF
    gap = assets.cashflow_gap(pf, m)
    assert gap.asset_cf.shape == (n_time + 1,)
    assert gap.liability_cf.shape == (n_time + 1,)
    expected_liab = flow_bom.copy(); expected_liab[:n_time] += flow_mid
    assert np.allclose(gap.liability_cf, expected_liab)
    assert np.allclose(gap.asset_cf, assets.asset_portfolio_cashflows(pf, n_time))
    assert np.allclose(gap.net_cf, gap.asset_cf - gap.liability_cf)
    assert np.allclose(gap.cumulative_net, np.cumsum(gap.net_cf))


def test_cashflow_gap_no_assets_is_negative_liability():
    """With no scheduled asset cash flows the net ladder is exactly the negative
    liability outflow, and the running total ends at minus the total paid."""
    mp, basis = _mp(), _basis()
    m = measure(mp, basis, full=True)
    pf = assets.AssetPortfolio(holdings=(assets.Equity(market_value=1000),))
    gap = assets.cashflow_gap(pf, m)
    assert np.allclose(gap.asset_cf, 0.0)
    assert np.allclose(gap.net_cf, -gap.liability_cf)
    assert np.isclose(gap.cumulative_net[-1], -gap.liability_cf.sum())


def test_cashflow_gap_needs_full():
    """A headline-only measurement carries no cash flows -- rejected upstream."""
    mp, basis = _mp(), _basis()
    pf = assets.AssetPortfolio(holdings=(assets.Cash(market_value=1.0),))
    with pytest.raises(ValueError, match="full=True"):
        assets.cashflow_gap(pf, measure(mp, basis, full=False))


def _gap(asset_cf, liability_cf):
    return assets.CashflowGap(asset_cf=np.asarray(asset_cf, float),
                              liability_cf=np.asarray(liability_cf, float))


def test_reinvest_hand_calc():
    """Surplus carried from a prior month earns one month of the new-money rate,
    then the month's net cash lands (no return in its arrival month)."""
    gap = _gap([0, 100, 0, 0], [0, 0, 50, 0])           # net_cf = [0, 100, -50, 0]
    r = assets.reinvest(gap, reinvest_rate=0.12)
    f = 1.12 ** (1.0 / 12.0)
    assert np.isclose(r.balance[0], 0.0)
    assert np.isclose(r.balance[1], 100.0)              # month-1 cash, not yet earning
    assert np.isclose(r.balance[2], 100.0 * f - 50.0)   # month-1 surplus grew one month
    assert np.isclose(r.balance[3], (100.0 * f - 50.0) * f)
    assert np.isclose(r.interest[2], 100.0 * (f - 1.0))
    assert np.isclose(r.closing_balance, r.balance[3])


def test_reinvest_balance_reconciles():
    """balance[m] == balance[m-1] + interest[m] + net_cf[m] on a real gap."""
    mp, basis = _mp(), _basis()
    gap = assets.cashflow_gap(
        assets.AssetPortfolio(holdings=(
            alm.Bond(face=1e6, coupon_rate=0.04, maturity_years=10, frequency=1),)),
        measure(mp, basis, full=True))
    r = assets.reinvest(gap, reinvest_rate=0.03, funding_rate=0.05)
    step = r.balance[1:] - r.balance[:-1]
    assert np.allclose(step, r.interest[1:] + r.net_cf[1:])


def test_reinvest_zero_rate_is_cumulative_net():
    """At a zero rate the roll-forward is exactly the gap's cumulative_net."""
    mp, basis = _mp(), _basis()
    gap = assets.cashflow_gap(
        assets.AssetPortfolio(holdings=(assets.Equity(market_value=1.0),)),
        measure(mp, basis, full=True))
    r = assets.reinvest(gap, reinvest_rate=0.0)
    assert np.allclose(r.balance, gap.cumulative_net)


def test_reinvest_funding_spread_charges_more():
    """A shortfall accrues at the funding rate; a higher funding rate deepens the
    deficit, and funding_rate=None falls back to the reinvest rate (symmetric)."""
    gap = _gap([0, 0, 0], [0, 100, 0])                  # net_cf = [0, -100, 0]
    cheap = assets.reinvest(gap, reinvest_rate=0.03, funding_rate=0.05)
    dear = assets.reinvest(gap, reinvest_rate=0.03, funding_rate=0.20)
    assert dear.closing_balance < cheap.closing_balance < 0.0
    ff = 1.20 ** (1.0 / 12.0)
    assert np.isclose(dear.balance[2], -100.0 * ff)     # funded balance grows by cost
    sym = assets.reinvest(gap, reinvest_rate=0.03)
    same = assets.reinvest(gap, reinvest_rate=0.03, funding_rate=0.03)
    assert np.isclose(sym.closing_balance, same.closing_balance)


def test_reinvest_opening_balance_and_rate_path():
    """opening_balance compounds at the rate; a constant rate path equals the
    scalar; the inception (month-0) net flow seeds the balance."""
    gap = _gap([0, 0, 0, 0], [0, 0, 0, 0])              # net_cf all zero
    r = assets.reinvest(gap, reinvest_rate=0.12, opening_balance=1000.0)
    assert np.isclose(r.balance[3], 1000.0 * 1.12 ** (3.0 / 12.0))
    path = assets.reinvest(gap, reinvest_rate=np.full(3, 0.12), opening_balance=1000.0)
    assert np.allclose(path.balance, r.balance)
    seed = _gap([0, 0, 0], [-200, 0, 0])                # month-0 premium inflow (net -(-200)=+200)
    assert np.isclose(assets.reinvest(seed, reinvest_rate=0.0).balance[0], 200.0)


def test_liquidate_hand_calc():
    """A shortfall the carried surplus cannot cover is met by a forced sale; the
    sale crystallises haircut * shortfall and floors the account at zero."""
    gap = _gap([0, 100, 0, 0], [0, 0, 300, 0])          # net_cf = [0, 100, -300, 0]
    r = assets.liquidate(gap, haircut=0.1)              # reinvest_rate 0 -> isolate
    assert np.allclose(r.balance, [0, 100, 0, 0])       # month-2 shortfall floored to 0
    assert np.allclose(r.forced_sale, [0, 0, 200, 0])   # 300 needed, 100 surplus -> sell 200
    assert np.allclose(r.realized_loss, [0, 0, 20, 0])  # 200 * 0.1
    assert np.isclose(r.total_realized_loss, 20.0)


def test_liquidate_surplus_earns_before_sale():
    """The carried surplus earns one month of the reinvest rate before the shortfall
    is netted, so a higher rate shrinks the forced sale."""
    gap = _gap([0, 100, 0], [0, 0, 300])               # net_cf = [0, 100, -300]
    f = 1.12 ** (1.0 / 12.0)
    r = assets.liquidate(gap, haircut=0.2, reinvest_rate=0.12)
    assert np.isclose(r.forced_sale[2], 300.0 - 100.0 * f)
    assert np.isclose(r.realized_loss[2], (300.0 - 100.0 * f) * 0.2)


def test_liquidate_no_shortfall_no_loss():
    """A gap that never runs short triggers no sale and no realized loss."""
    mp, basis = _mp(), _basis()
    gap = assets.cashflow_gap(
        assets.AssetPortfolio(holdings=(
            alm.Bond(face=1e7, coupon_rate=0.05, maturity_years=10, frequency=1),)),
        measure(mp, basis, full=True))
    r = assets.liquidate(gap, haircut=0.15, opening_balance=1e7)
    assert np.allclose(r.forced_sale, 0.0)
    assert r.total_realized_loss == 0.0
    assert np.all(r.balance >= 0.0)


def test_liquidate_deeper_haircut_costs_more():
    """The same shortfall under a wider stress (deeper haircut) realizes more loss."""
    gap = _gap([0, 0, 0], [0, 500, 0])                 # net_cf = [0, -500, 0]
    cheap = assets.liquidate(gap, haircut=0.05)
    dear = assets.liquidate(gap, haircut=0.25)
    assert dear.total_realized_loss > cheap.total_realized_loss > 0.0
    assert np.isclose(dear.total_realized_loss / cheap.total_realized_loss, 0.25 / 0.05)


def test_liquidate_caps_forced_sale_at_asset_stock():
    """A forced sale cannot exceed the asset stock; the uncovered shortfall is
    unfunded (insolvency)."""
    gap = _gap([0, 0, 0], [0, 0, 300])                 # net_cf = [0, 0, -300]
    avail = np.array([1000.0, 1000.0, 220.0])          # 220 fair value at month 2
    r = assets.liquidate(gap, haircut=0.1, available_assets=avail)
    # 220 of fair value at a 10% haircut nets 220 / 1.1 = 200 cash (20 loss = the
    # remaining 20 of stock destroyed), leaving 300 - 200 = 100 unfunded.
    assert np.allclose(r.forced_sale, [0, 0, 200])
    assert np.allclose(r.realized_loss, [0, 0, 20])    # 200 * 0.1
    assert np.allclose(r.unfunded, [0, 0, 100])
    assert np.isclose(r.total_unfunded, 100.0)


def test_liquidate_ample_assets_matches_uncapped():
    """A non-binding cap reproduces the uncapped roll-forward, with no unfunded."""
    gap = _gap([0, 100, 0], [0, 0, 300])               # net_cf = [0, 100, -300]
    capped = assets.liquidate(gap, haircut=0.2, available_assets=np.full(3, 1e9))
    uncapped = assets.liquidate(gap, haircut=0.2)
    assert np.allclose(capped.forced_sale, uncapped.forced_sale)
    assert np.allclose(capped.unfunded, 0.0)
    assert np.isclose(capped.total_realized_loss, uncapped.total_realized_loss)


def test_liquidate_stock_depletes_across_months():
    """Sales accumulate against the stock: an earlier sale shrinks what a later
    shortfall can raise."""
    gap = _gap([0, 0, 0, 0], [0, 150, 0, 150])         # net_cf = [0, -150, 0, -150]
    avail = np.array([300.0, 300.0, 300.0, 250.0])     # 250 left at month 3
    r = assets.liquidate(gap, haircut=0.0, available_assets=avail)
    assert np.allclose(r.forced_sale, [0, 150, 0, 100])   # month 3 capped: 250 - 150 sold
    assert np.allclose(r.unfunded, [0, 0, 0, 50])         # 150 needed - 100 raised


def test_interaction_loss_decomposes():
    """total_loss = revaluation_loss + forced_sale_loss, and each leg reproduces its
    standalone build (the coupled-stress NAV revaluation and the stressed-gap
    liquidation)."""
    from fastcashflow import solvency as sv
    mp, basis = _mp(), _basis()
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=2e6, coupon_rate=0.03, maturity_years=12, frequency=1),))
    shift, sens, hc = 0.01, 8.0, 0.1
    res = assets.interaction_loss(pf, mp, basis, shift=shift,
                                  lapse_sensitivity=sens, haircut=hc)
    # base NAV reproduces asset_portfolio_value - BEL
    base_nav = (assets.asset_portfolio_value(pf, basis.discount_annual)
                - float(measure(mp, basis, full=False).bel.sum()))
    assert np.isclose(res.base_nav, base_nav)
    # stressed NAV reproduces the coupled-stress re-measure
    mp_s, basis_s = sv.interest_with_dynamic_lapse(shift, sens).apply(mp, basis)
    stressed_nav = (assets.asset_portfolio_value(pf, basis_s.discount_annual)
                    - float(measure(mp_s, basis_s, full=False).bel.sum()))
    assert np.isclose(res.stressed_nav, stressed_nav)
    # forced-sale leg reproduces liquidate on the stressed gap
    gap_s = assets.cashflow_gap(pf, measure(mp_s, basis_s, full=True))
    liq = assets.liquidate(gap_s, haircut=hc)
    assert np.isclose(res.forced_sale_loss, liq.total_realized_loss)
    assert np.isclose(res.total_loss, res.revaluation_loss + res.forced_sale_loss)


def test_interaction_loss_zero_haircut_drops_friction():
    """A zero haircut removes the forced-sale friction, leaving only the
    mark-to-market revaluation."""
    mp, basis = _mp(), _basis()
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=1e6, coupon_rate=0.04, maturity_years=10, frequency=1),))
    res = assets.interaction_loss(pf, mp, basis, shift=0.01,
                                  lapse_sensitivity=8.0, haircut=0.0)
    assert res.forced_sale_loss == 0.0
    assert np.isclose(res.total_loss, res.revaluation_loss)


def _solvency_portfolio():
    return assets.AssetPortfolio(holdings=(
        alm.Bond(2_600_000.0, 0.03, 10, 1), assets.Cash(3_000_000.0)))


def test_dynamic_solvency_decomposes():
    """The dynamic ratio takes the coupled-stress interaction loss off the static
    available capital and divides by the unchanged required capital; the parts
    reproduce their standalone builds."""
    mp, basis = _mp(), _basis()
    pf = _solvency_portfolio()
    d = assets.dynamic_solvency(pf, mp, basis, regime=fcf.SOLVENCY2,
                                shift=0.01, lapse_sensitivity=8.0, haircut=0.1)
    static = assets.assess_solvency(pf, mp, basis, regime=fcf.SOLVENCY2)
    interaction = assets.interaction_loss(pf, mp, basis, shift=0.01,
                                          lapse_sensitivity=8.0, haircut=0.1)
    assert np.isclose(d.static.solvency_ratio, static.solvency_ratio)
    assert np.isclose(d.interaction.total_loss, interaction.total_loss)
    assert np.isclose(d.stressed_available_capital,
                      static.available_capital - interaction.total_loss)
    assert np.isclose(d.stressed_ratio, d.stressed_available_capital / static.total_scr)
    # the liquidation trajectory is exposed for the surplus / forced-sale path
    assert np.isclose(d.liquidation.total_realized_loss, d.interaction.forced_sale_loss)


def test_dynamic_solvency_zero_scenario_is_static():
    """A null scenario (no shift, no haircut) leaves the ratio at the static value."""
    mp, basis = _mp(), _basis()
    pf = _solvency_portfolio()
    d = assets.dynamic_solvency(pf, mp, basis, regime=fcf.KICS,
                                shift=0.0, lapse_sensitivity=8.0, haircut=0.0)
    assert d.interaction.total_loss == 0.0
    assert np.isclose(d.stressed_ratio, d.static.solvency_ratio)
    assert np.isclose(d.stressed_available_capital, d.static.available_capital)


def test_dynamic_solvency_deeper_haircut_lowers_ratio():
    """A deeper liquidation haircut crystallises more forced-sale loss, lowering the
    stressed coverage ratio (the static ratio is unchanged)."""
    mp, basis = _mp(), _basis()
    pf = _solvency_portfolio()
    shallow = assets.dynamic_solvency(pf, mp, basis, regime=fcf.SOLVENCY2,
                                      shift=0.01, lapse_sensitivity=8.0, haircut=0.05)
    deep = assets.dynamic_solvency(pf, mp, basis, regime=fcf.SOLVENCY2,
                                   shift=0.01, lapse_sensitivity=8.0, haircut=0.30)
    assert deep.interaction.forced_sale_loss > shallow.interaction.forced_sale_loss
    assert deep.stressed_ratio < shallow.stressed_ratio
    assert np.isclose(deep.static.solvency_ratio, shallow.static.solvency_ratio)


def test_dynamic_solvency_passes_through_assess_kwargs():
    """Extra kwargs (e.g. tax_rate) reach assess_solvency unchanged."""
    mp, basis = _mp(), _basis()
    pf = _solvency_portfolio()
    d = assets.dynamic_solvency(pf, mp, basis, regime=fcf.KICS, shift=0.0,
                                lapse_sensitivity=0.0, haircut=0.0, tax_rate=0.22)
    taxed = assets.assess_solvency(pf, mp, basis, regime=fcf.KICS, tax_rate=0.22)
    assert np.isclose(d.static.tax_adjustment, taxed.tax_adjustment)
    assert d.static.tax_adjustment > 0.0


def test_dynamic_solvency_report_renders():
    """report(dynamic_solvency(...)) yields an ASCII DynamicSolvencyReport with the
    static / scenario / after-scenario blocks, and ties to the result."""
    mp = _mp()
    basis = make_death_basis(mortality_q=0.001, lapse_q=0.03, discount_annual=0.03,
                             mortality_cv=0.0)
    pf = _solvency_portfolio()
    d = assets.dynamic_solvency(pf, mp, basis, regime=fcf.SOLVENCY2,
                                shift=0.01, lapse_sensitivity=8.0, haircut=0.1)
    rep = fcf.report(d)
    assert isinstance(rep, fcf.DynamicSolvencyReport)
    text = str(rep)
    assert all(ord(c) < 128 for c in text)                 # ASCII only (global surface)
    for label in ("Dynamic solvency", "Solvency ratio", "Total interaction loss",
                  "Stressed solvency ratio"):
        assert label in text
    assert f"{d.stressed_available_capital:,.0f}" in text  # the figure ties to the result
