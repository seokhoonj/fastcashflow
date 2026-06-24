"""Assets and the solvency balance sheet -- hand-calc anchors."""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import assets, alm
from fastcashflow import _solvency_assessment as sa
import fastcashflow._vfa_solvency as vs
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
    assert np.isclose(sa.net_interest_scr(p, mp, basis, interest_curves=curves),
                      expected)


def test_matched_book_net_interest_near_zero():
    """A bond book sized to the liability DV01 immunises the net interest SCR."""
    mp, basis = _mp(), _basis()
    liab_dv01 = alm.liability_dv01(mp, basis)
    per_face = alm.bond_duration(alm.Bond(100.0, 0.03, 10, 1), 0.03).dv01
    face = liab_dv01 / per_face * 100.0
    p = assets.AssetPortfolio(holdings=(alm.Bond(face, 0.03, 10, 1),))
    ni = sa.net_interest_scr(p, mp, basis, interest_curves=_parallel_curves())
    assert ni < abs(face) * 1e-4                 # negligible vs the book (immunised)


def test_unmatched_book_positive_net_interest():
    """An all-cash book leaves the liability's rate move unhedged -> positive SCR."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(assets.Cash(10_000_000.0),))
    assert sa.net_interest_scr(p, mp, basis, interest_curves=_parallel_curves()) > 0.0


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
    assert np.isclose(sa.net_interest_kics_scr(p, mp, basis, scenarios=ki), expected)


def test_assess_kics_interest_enters_market_module():
    """K-ICS interest now flows into the market module (net), not zero as before,
    and not into the insurance module (no double count)."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(assets.Cash(10_000_000.0),))    # unhedged
    ki = _kics_scenarios()
    without = sa.assess(p, mp, basis, regime=fcf.solvency.KICS)
    with_ = sa.assess(p, mp, basis, regime=fcf.solvency.KICS, interest_scenarios=ki)
    assert without.net_interest_scr == 0.0               # K-ICS supplied no curves before
    assert with_.net_interest_scr > 0.0                  # now the five-scenario net amount
    assert with_.market_scr > without.market_scr
    assert with_.insurance_scr == without.insurance_scr  # interest not in the insurance module


def test_sii_toplevel_three_module_diversifies():
    """Solvency II (Annex IV) aggregates the (life, market, credit) modules at
    all-pairwise 0.25 -- the same values as K-ICS, and below the simple sum."""
    import math
    ins, mkt, cr = 300.0, 400.0, 200.0
    sii = sa.basic_scr(ins, mkt, cr, regime=fcf.solvency.SII)
    kics = sa.basic_scr(ins, mkt, cr, regime=fcf.solvency.KICS)
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
    sii = sa.basic_scr(ins, mkt, cr, regime=fcf.solvency.SII,
                                            general_insurance=gen)
    kics = sa.basic_scr(ins, mkt, cr, regime=fcf.solvency.KICS,
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
    assert np.isclose(sa.equity_scr(one, fcf.solvency.KICS), 3_000_000.0 * 0.35)
    p = assets.AssetPortfolio(holdings=(
        assets.Equity(1_000_000.0, "developed"), assets.Equity(500_000.0, "emerging")))
    dev, emg = 1_000_000.0 * 0.35, 500_000.0 * 0.48
    assert np.isclose(sa.equity_scr(p, fcf.solvency.KICS),
                      np.sqrt(dev**2 + emg**2 + 2 * 0.75 * dev * emg))
    assert sa.equity_scr(p, fcf.solvency.KICS) < dev + emg            # diversification
    bad = assets.AssetPortfolio(holdings=(assets.Equity(1.0, "exotic"),))
    with pytest.raises(ValueError, match="risk_type"):
        sa.equity_scr(bad, fcf.solvency.SII)


def test_equity_subtypes_shocks():
    """K-ICS equity sub-types each carry their handbook 4-3 shock (preferred is
    rating-based, tested separately)."""
    for risk_type, shock in [("infrastructure", 0.20), ("long_term", 0.20),
                             ("other", 0.49)]:
        p = assets.AssetPortfolio(holdings=(assets.Equity(1_000_000.0, risk_type),))
        assert np.isclose(sa.equity_scr(p, fcf.solvency.KICS), 1_000_000.0 * shock)


def test_property_scr():
    p = assets.AssetPortfolio(holdings=(assets.Property(2_000_000.0),))
    assert np.isclose(sa.property_scr(p, fcf.solvency.SII), 2_000_000.0 * 0.25)


def test_market_module_aggregates_sub_risks():
    """The market module is sqrt(c^T R c) over (interest, equity, property, FX,
    concentration) with the table-19 correlation."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        assets.Equity(3_000_000.0, "developed"), assets.Property(1_000_000.0)))
    # K-ICS: no interest curves -> interest 0; no foreign currency -> FX 0; the
    # property holding does exceed the 6% individual limit -> a concentration charge
    eq = sa.equity_scr(p, fcf.solvency.KICS)
    pr = sa.property_scr(p, fcf.solvency.KICS)
    fx = sa.fx_scr(p, fcf.solvency.KICS, basis.discount_annual)
    conc = sa.concentration_scr(p, fcf.solvency.KICS, basis.discount_annual)
    c = np.array([0.0, eq, pr, fx, conc])
    R = sa._MARKET_CORRELATION
    assert np.isclose(sa.market_scr(p, mp, basis, regime=fcf.solvency.KICS),
                      np.sqrt(c @ R @ c))


def test_operational_scr_kics():
    """K-ICS operational = max(premium x 3.5%, BEL x 0.4%)."""
    mp, basis = _mp(), _basis()
    m = measure(mp, basis, full=True)
    bel = max(0.0, float(m.bel.sum()))
    prem = max(0.0, float(m.cashflows.premium_cf[:, :12].sum()))
    expected = max(prem * 0.035, bel * 0.004)
    assert np.isclose(sa.operational_scr(mp, basis, fcf.solvency.KICS), expected)
    assert sa.operational_scr(mp, basis, fcf.solvency.KICS) > 0.0


def test_operational_scr_sii_cap():
    """Solvency II operational is capped at 0.3 x BSCR."""
    mp, basis = _mp(), _basis()
    m = measure(mp, basis, full=True)
    bel = max(0.0, float(m.bel.sum()))
    prem = max(0.0, float(m.cashflows.premium_cf[:, :12].sum()))
    op_uncapped = max(prem * 0.04, bel * 0.0045)
    # a tiny BSCR makes the 0.3 x BSCR cap bite
    small = op_uncapped / 10.0
    assert np.isclose(sa.operational_scr(mp, basis, fcf.solvency.SII, bscr=small),
                      0.30 * small)
    # a large BSCR leaves it uncapped
    assert np.isclose(sa.operational_scr(mp, basis, fcf.solvency.SII, bscr=1e12),
                      op_uncapped)


def test_effective_maturity_hand_calc():
    """Effective maturity = sum(t x CF) / sum(CF). A 10y 3% annual bond:
    sum(t x CF) = 3 x (1+..+10) + 100 x 10 = 1165; sum(CF) = 130; 1165/130."""
    b = alm.Bond(100.0, 0.03, 10, 1)
    assert np.isclose(alm.effective_maturity(b), 1165.0 / 130.0)


def test_credit_bucket_boundaries():
    """Bucket k is k < m <= k+1 (so 0-1 is index 0), capped at the 14+ bucket."""
    assert sa._credit_bucket(0.5) == 0
    assert sa._credit_bucket(1.0) == 0
    assert sa._credit_bucket(1.5) == 1
    assert sa._credit_bucket(10.0) == 9
    assert sa._credit_bucket(14.0) == 13
    assert sa._credit_bucket(20.0) == 14


def test_credit_scr_kics_hand_calc():
    """K-ICS credit SCR = market value x factor[rating row][maturity bucket].
    An AA corporate bond (10y 3%) has effective maturity 8.96 -> bucket 8, and the
    corporate '1-2' row there is 2.0% (handbook table 30)."""
    b = alm.Bond(100.0, 0.03, 10, 1, credit_rating="AA", exposure_class="corporate")
    p = assets.AssetPortfolio(holdings=(b, assets.Cash(1000.0), assets.Equity(500.0)))
    mv = alm.bond_value(b, 0.03)
    assert np.isclose(sa.credit_scr(p, fcf.solvency.KICS, 0.03), mv * 0.02)   # only the bond


def test_credit_scr_sii_spread():
    """Solvency II credit (Art 176 spread) = market value times a stress that is
    piecewise-linear in modified duration by credit quality step."""
    b = alm.Bond(100.0, 0.03, 10, 1, credit_rating="A")        # CQS 2
    mod = alm.bond_duration(b, 0.03).modified                  # ~8.53 -> bucket 5-10
    factor = 0.070 + 0.007 * (mod - 5)                         # a + b x (dur - 5)
    p = assets.AssetPortfolio(holdings=(b, assets.Cash(1000.0)))
    assert np.isclose(sa.credit_scr(p, fcf.solvency.SII, 0.03),
                      alm.bond_value(b, 0.03) * factor)
    # the piecewise stress at representative points
    assert np.isclose(sa._sii_spread_stress("AAA", 3), 0.009 * 3)        # 0-5
    assert np.isclose(sa._sii_spread_stress("BBB", 12), 0.200 + 0.010 * 2)  # 10-15
    assert np.isclose(sa._sii_spread_stress("BB", 25), 0.466 + 0.005 * 5)   # 20+


def test_credit_scr_rating_and_class():
    """A lower rating and a riskier exposure class both raise the factor."""
    aa = alm.Bond(100.0, 0.03, 10, 1, credit_rating="AA", exposure_class="corporate")
    bb = alm.Bond(100.0, 0.03, 10, 1, credit_rating="BB", exposure_class="corporate")
    sec = alm.Bond(100.0, 0.03, 10, 1, credit_rating="BB", exposure_class="securitisation")
    s_aa = sa.credit_scr(assets.AssetPortfolio(holdings=(aa,)), fcf.solvency.KICS, 0.03)
    s_bb = sa.credit_scr(assets.AssetPortfolio(holdings=(bb,)), fcf.solvency.KICS, 0.03)
    s_sec = sa.credit_scr(assets.AssetPortfolio(holdings=(sec,)), fcf.solvency.KICS, 0.03)
    assert s_bb > s_aa                       # BB (grade 5) charges more than AA (1-2)
    assert s_sec > s_bb                      # securitisation BB charges more than corporate
    bad = alm.Bond(100.0, 0.03, 10, 1, exposure_class="exotic")
    with pytest.raises(ValueError, match="exposure_class"):
        sa.credit_scr(assets.AssetPortfolio(holdings=(bad,)), fcf.solvency.KICS, 0.03)


def test_fx_scr_hand_calc():
    """K-ICS FX SCR = worse of won-up / won-down, aggregated at 0.5 correlation.
    USD 1000 (shock 25% -> 250) and JPY 500 (40% -> 200), both long, so the won-up
    scenario binds: sqrt(250^2 + 200^2 + 2 x 0.5 x 250 x 200)."""
    p = assets.AssetPortfolio(holdings=(
        assets.Cash(1000.0, currency="USD"), assets.Cash(500.0, currency="JPY"),
        assets.Cash(2000.0)))                          # 2000 won: no FX
    expected = np.sqrt(250.0**2 + 200.0**2 + 2 * 0.5 * 250.0 * 200.0)
    assert np.isclose(sa.fx_scr(p, fcf.solvency.KICS, 0.03), expected)


def test_fx_scr_sii_flat_25():
    """Solvency II currency risk (Art 188) is a flat 25% per foreign currency,
    summed (no diversification); any currency applies (no table lookup)."""
    p = assets.AssetPortfolio(holdings=(
        assets.Cash(1000.0, currency="USD"), assets.Cash(500.0, currency="EUR"),
        assets.Cash(2000.0)))
    assert np.isclose(sa.fx_scr(p, fcf.solvency.SII, 0.03), 0.25 * 1000 + 0.25 * 500)
    # an unlisted currency is fine under Solvency II (no table)
    q = assets.AssetPortfolio(holdings=(assets.Cash(800.0, currency="XYZ"),))
    assert np.isclose(sa.fx_scr(q, fcf.solvency.SII, 0.03), 0.25 * 800)


def test_fx_scr_short_position_up_scenario():
    """A net-short currency (a foreign liability) loses under the won-down scenario:
    a short EUR 800 -> 35% x 800."""
    p = assets.AssetPortfolio(holdings=(assets.Cash(-800.0, currency="EUR"),))
    assert np.isclose(sa.fx_scr(p, fcf.solvency.KICS, 0.03), 0.35 * 800.0)
    bad = assets.AssetPortfolio(holdings=(assets.Cash(1.0, currency="XXX"),))
    with pytest.raises(ValueError, match="currency"):
        sa.fx_scr(bad, fcf.solvency.KICS, 0.03)


def test_market_module_includes_fx_negative_correlation():
    """FX enters the market module; equity <-> FX is NEGATIVE 0.25, so an equity +
    FX book diversifies: sqrt(eq^2 + fx^2 - 2 x 0.25 x eq x fx)."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        assets.Equity(1000.0, "developed"), assets.Cash(1000.0, currency="USD")))
    eq = sa.equity_scr(p, fcf.solvency.KICS)                # 350
    fx = sa.fx_scr(p, fcf.solvency.KICS, basis.discount_annual)   # 250
    got = sa.market_scr(p, mp, basis, regime=fcf.solvency.KICS)
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
    got = sa.concentration_scr(p, fcf.solvency.KICS, 0.03, total_assets=10_000_000.0)
    assert np.isclose(got, np.sqrt(cp**2 + pr**2))


def test_concentration_scr_untagged_is_zero():
    """A book with no tagged issuers and no property has no concentration charge,
    under both regimes."""
    p = assets.AssetPortfolio(holdings=(alm.Bond(1e6, 0.03, 5, 1), assets.Cash(1e6)))
    assert sa.concentration_scr(p, fcf.solvency.KICS, 0.03) == 0.0
    assert sa.concentration_scr(p, fcf.solvency.SII, 0.03) == 0.0


def test_concentration_scr_sii_excess():
    """Solvency II concentration (Art 184-187): single-name excess
    max(0, exposure - CT(CQS) x assets) x g(CQS). A BBB issuer is CQS 3 (CT 1.5%,
    g 27%); an AA issuer is CQS 1 (CT 3%, g 12%)."""
    bbb = assets.AssetPortfolio(holdings=(
        alm.Bond(1e6, 0.03, 5, 1, credit_rating="BBB", issuer="X"),))
    assert np.isclose(sa.concentration_scr(bbb, fcf.solvency.SII, 0.03, total_assets=10e6),
                      max(0.0, 1e6 - 10e6 * 0.015) * 0.27)
    aa = assets.AssetPortfolio(holdings=(
        alm.Bond(1e6, 0.03, 5, 1, credit_rating="AA", issuer="Y"),))
    assert np.isclose(sa.concentration_scr(aa, fcf.solvency.SII, 0.03, total_assets=10e6),
                      max(0.0, 1e6 - 10e6 * 0.03) * 0.12)


def test_concentration_band_by_rating():
    """Lower-rated issuers fall in a tighter band (lower limit, higher factor)."""
    aa = assets.AssetPortfolio(holdings=(
        alm.Bond(1e6, 0.03, 5, 1, credit_rating="AA", issuer="X"),))
    bb = assets.AssetPortfolio(holdings=(
        alm.Bond(1e6, 0.03, 5, 1, credit_rating="BB", issuer="X"),))
    s_aa = sa.concentration_scr(aa, fcf.solvency.KICS, 0.03, total_assets=10e6)  # 4%/15%
    s_bb = sa.concentration_scr(bb, fcf.solvency.KICS, 0.03, total_assets=10e6)  # 1.5%/50%
    assert np.isclose(s_aa, (1e6 - 10e6 * 0.04) * 0.15)
    assert np.isclose(s_bb, (1e6 - 10e6 * 0.015) * 0.50)
    assert s_bb > s_aa


def test_concentration_property_whole_book_limit():
    """Property concentration takes the worse of the individual (6%) and whole-book
    (25%) limits. Two 2m properties on 10m assets: whole-book excess (4m - 2.5m)."""
    p = assets.AssetPortfolio(holdings=(assets.Property(2e6), assets.Property(2e6)))
    got = sa.concentration_scr(p, fcf.solvency.KICS, 0.03, total_assets=10e6)
    individual = np.sqrt(2 * ((2e6 - 10e6 * 0.06) * 0.20) ** 2)
    whole = (4e6 - 10e6 * 0.25) * 0.20
    assert np.isclose(got, max(individual, whole))


def test_assess_components():
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(2_600_000.0, 0.03, 10, 1), assets.Cash(3_000_000.0)))
    a = sa.assess(p, mp, basis, regime=fcf.solvency.SII)
    # no equity/property -> the market module is just the net interest SCR; the SII
    # top-level BSCR aggregates insurance + market + Art-176 credit at all-pairwise
    # 0.25 (Annex IV); the total adds operational on top
    assert np.isclose(a.market_scr, a.net_interest_scr)
    assert a.credit_scr > 0.0                               # SII Art-176 spread on the bond
    m = np.array([a.insurance_scr, a.market_scr, a.credit_scr])
    R = np.array([[1.0, .25, .25], [.25, 1.0, .25], [.25, .25, 1.0]])
    assert np.isclose(a.basic_scr, np.sqrt(m @ R @ m))
    assert np.isclose(a.basic_required_capital, a.basic_scr + a.operational_scr)
    assert a.tax_adjustment == 0.0                          # no tax relief by default
    assert np.isclose(a.total_scr, a.basic_required_capital - a.tax_adjustment)
    assert np.isclose(a.ratio, a.available_capital / a.total_scr)
    assert np.isclose(a.available_capital, a.asset_portfolio_value - (a.bel + a.risk_margin))


def test_tax_adjustment_loss_absorption():
    """K-ICS chapter 7: the tax adjustment = min(basic x tax_rate, recoverability
    limit) is subtracted from the basic required capital, raising the ratio."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(2_600_000.0, 0.03, 10, 1), assets.Cash(3_000_000.0)))
    base = sa.assess(p, mp, basis, regime=fcf.solvency.KICS)
    taxed = sa.assess(p, mp, basis, regime=fcf.solvency.KICS, tax_rate=0.22)
    assert np.isclose(taxed.tax_adjustment, base.basic_required_capital * 0.22)
    assert np.isclose(taxed.total_scr, base.basic_required_capital * (1 - 0.22))
    assert taxed.ratio > base.ratio       # tax relief lowers the SCR
    # the recoverability limit caps the relief
    cap = base.basic_required_capital * 0.22 / 3
    capped = sa.assess(p, mp, basis, regime=fcf.solvency.KICS, tax_rate=0.22,
                                    tax_recoverability_limit=cap)
    assert np.isclose(capped.tax_adjustment, cap)


def test_assess_kics_no_curves():
    """K-ICS supplies no interest curves -> the net interest component is 0;
    an all-cash book then has total SCR == the insurance SCR (no market risk)."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(assets.Cash(8_000_000.0),))
    a = sa.assess(p, mp, basis, regime=fcf.solvency.KICS)
    assert a.net_interest_scr == 0.0
    assert np.isclose(a.basic_scr, a.insurance_scr)              # all-cash -> no market risk
    assert np.isclose(a.total_scr, a.basic_scr + a.operational_scr)
    assert np.isfinite(a.ratio)


def test_top_level_aggregation_kics_vs_sii():
    """K-ICS (table 3) and Solvency II (Annex IV) both aggregate insurance, market
    and credit at all-pairwise 0.25 -- the 3-module correlation values coincide, so
    each BSCR is the sqrt aggregation of its own module amounts."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(2_000_000.0, 0.03, 10, 1), assets.Equity(3_000_000.0, "developed")))
    k = sa.assess(p, mp, basis, regime=fcf.solvency.KICS)
    s = sa.assess(p, mp, basis, regime=fcf.solvency.SII)
    R = np.array([[1.0, 0.25, 0.25], [0.25, 1.0, 0.25], [0.25, 0.25, 1.0]])
    # SII: sqrt aggregation at 0.25 (Annex IV), below the simple module sum
    assert s.credit_scr > 0.0
    m_s = np.array([s.insurance_scr, s.market_scr, s.credit_scr])
    assert np.isclose(s.basic_scr, np.sqrt(m_s @ R @ m_s))
    assert s.basic_scr < s.insurance_scr + s.market_scr + s.credit_scr     # diversifies
    # K-ICS: the bond now carries a credit charge -> same 3-module sqrt aggregation
    assert k.credit_scr > 0.0
    m = np.array([k.insurance_scr, k.market_scr, k.credit_scr])
    assert np.isclose(k.basic_scr, np.sqrt(m @ R @ m))                            # K-ICS sqrt
    assert np.isclose(k.total_scr, k.basic_scr + k.operational_scr)               # + operational


def test_equity_now_charges_scr():
    """Equity now raises BOTH available capital and the SCR (the v1 overstatement
    where it lifted only the numerator is fixed)."""
    mp, basis = _mp(), _basis()
    base = assets.AssetPortfolio(holdings=(
        alm.Bond(2_000_000.0, 0.03, 10, 1), assets.Cash(5_000_000.0)))
    with_eq = assets.AssetPortfolio(holdings=base.holdings + (assets.Equity(3_000_000.0),))
    a0 = sa.assess(base, mp, basis, regime=fcf.solvency.SII)
    a1 = sa.assess(with_eq, mp, basis, regime=fcf.solvency.SII)
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
    a = sa.assess(p, mp, basis, regime=fcf.solvency.KICS, general_insurance_scr=gen)
    m = np.array([a.insurance_scr, gen, a.market_scr, a.credit_scr])
    R = np.array([[1, 0, .25, .25], [0, 1, .25, .25],
                  [.25, .25, 1, .25], [.25, .25, .25, 1]])
    assert np.isclose(a.general_insurance_scr, gen)
    assert np.isclose(a.basic_scr, np.sqrt(m @ R @ m))
    # zero general -> the 3-module aggregation (unchanged)
    a0 = sa.assess(p, mp, basis, regime=fcf.solvency.KICS)
    assert a0.general_insurance_scr == 0.0
    assert a.basic_scr > a0.basic_scr                                  # adding a module raises BSCR


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


def test_basic_scr_reproduces_disclosures():
    """The public top-level aggregation reproduces disclosed K-ICS basic required
    capital from the published module amounts. A pure-life book (general = 0) and a
    life + P&C book (general module added) both match the disclosure to the won."""
    # life insurer, general insurance = 0 (KRW millions)
    basic = sa.basic_scr(
        11_628_115, 34_552_189, 4_166_014, regime=fcf.solvency.KICS, operational=1_083_844)
    assert np.isclose(basic, 41_624_006, rtol=0, atol=2)
    # life + general insurance (KRW thousands)
    basic_g = sa.basic_scr(
        10_654_450_301, 7_191_902_855, 2_495_852_957, regime=fcf.solvency.KICS,
        operational=1_648_722_264, general_insurance=542_312_823)
    assert np.isclose(basic_g, 16_977_612_719, rtol=0, atol=2)
    # the diversification effect = simple sum - aggregate (ex operational)
    agg = sa.basic_scr(11_628_115, 34_552_189, 4_166_014,
                                            regime=fcf.solvency.KICS)
    div = (11_628_115 + 34_552_189 + 4_166_014) - agg
    assert np.isclose(div, 9_806_156, rtol=0, atol=2)
    # Solvency II aggregates the modules at the Annex IV 0.25 correlation (+ operational)
    s = sa.basic_scr(100.0, 200.0, 50.0, regime=fcf.solvency.SII,
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
        assert np.isclose(sa.equity_scr(p, fcf.solvency.KICS), 1_000_000.0 * shock)


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


def test_vfa_cashflow_gap_uses_the_guarantee_excess_basis():
    """The VFA gap nets assets against the entity guarantee-excess liability (not
    the gross account-value benefit) and -- unlike cashflow_gap -- accepts an
    account-value book."""
    basis = make_death_basis(mortality_q=0.002, lapse_q=0.004, discount_annual=0.03,
                             ra_confidence=0.75, investment_return=0.0, fund_fee=0.02)
    mp = fcf.ModelPoints.single(40, 0.0, 60, account_value=1000.0,
                                minimum_accumulation_benefit=1200.0,
                                calculation_methods=PATTERNS)
    m = fcf.vfa.measure(mp, basis)
    net = alm.vfa_net_liability_cashflows(m)
    n_time = net.shape[0]
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=100, coupon_rate=0.05, maturity_years=5, frequency=1),
        assets.Equity(market_value=500),))                  # no scheduled CF
    gap = assets.vfa_cashflow_gap(pf, m)
    assert gap.asset_cf.shape == (n_time + 1,)
    assert gap.liability_cf.shape == (n_time + 1,)
    expected_liab = np.zeros(n_time + 1); expected_liab[:n_time] = net
    assert np.allclose(gap.liability_cf, expected_liab)
    assert np.allclose(gap.asset_cf, assets.asset_portfolio_cashflows(pf, n_time))
    assert np.allclose(gap.net_cf, gap.asset_cf - gap.liability_cf)
    # The gross account-value benefit is excluded -- the liability is far smaller
    # than the total benefit the unit fund pays.
    assert abs(gap.liability_cf.sum()) < float(m.benefit_cf.sum())


def _vfa_book():
    basis = make_death_basis(mortality_q=0.002, lapse_q=0.004, discount_annual=0.03,
                             ra_confidence=0.75, investment_return=0.04, fund_fee=0.02)
    mp = fcf.ModelPoints.single(40, 0.0, 60, account_value=1000.0,
                                minimum_accumulation_benefit=1200.0,
                                calculation_methods=PATTERNS)
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=200, coupon_rate=0.04, maturity_years=5, frequency=1),))
    return pf, mp, basis


def test_vfa_interaction_loss_null_scenario_is_zero():
    """No shock, no forced sale, static lapse -> base and stressed NAV coincide and
    the total loss is exactly zero."""
    pf, mp, basis = _vfa_book()
    r = fcf.vfa.interaction_loss(pf, mp, basis, return_shock=0.0,
                                    lapse_sensitivity=0.0, haircut=0.0)
    assert np.isclose(r.base_nav, r.stressed_nav)
    assert np.isclose(r.revaluation_loss, 0.0)
    assert np.isclose(r.forced_sale_loss, 0.0)
    assert np.isclose(r.total_loss, 0.0)


def test_vfa_interaction_loss_av_drop_lifts_the_guarantee_cost():
    """An account-value drop pushes the GMAB in-the-money, so the guarantee cost
    (VFA net BEL) rises and the NAV falls -- a positive revaluation loss. The
    moneyness lapse holds more policies on the now-valuable guarantee, deepening
    the loss beyond the static-lapse case. total = revaluation + forced sale."""
    pf, mp, basis = _vfa_book()
    static = fcf.vfa.interaction_loss(pf, mp, basis, return_shock=-0.30,
                                         lapse_sensitivity=0.0, haircut=0.0)
    dyn = fcf.vfa.interaction_loss(pf, mp, basis, return_shock=-0.30,
                                      lapse_sensitivity=0.8, haircut=0.10)
    assert static.revaluation_loss > 0.0                       # AV drop -> ITM -> BEL up
    assert dyn.revaluation_loss > static.revaluation_loss      # moneyness lapse deepens it
    assert dyn.forced_sale_loss >= 0.0
    assert np.isclose(dyn.total_loss, dyn.revaluation_loss + dyn.forced_sale_loss)
    # The revaluation loss reconciles to an independent NAV recompute.
    base_nav = vs._portfolio_nav_vfa(pf, mp, basis)
    assert np.isclose(static.base_nav, base_nav)


def test_assess_dynamic_vfa_overlays_the_interaction_on_a_supplied_static():
    """The VFA dynamic solvency takes the scenario interaction loss off a supplied
    static available capital and divides by the unchanged required capital. A null
    scenario leaves the ratio static; a shock lowers it by total_loss / total_scr."""
    pf, mp, basis = _vfa_book()
    static_ac, total_scr = 1000.0, 400.0

    null = fcf.vfa.assess_dynamic(
        pf, mp, basis, static_available_capital=static_ac, total_scr=total_scr,
        return_shock=0.0, lapse_sensitivity=0.0, haircut=0.0)
    assert np.isclose(null.interaction.total_loss, 0.0)
    assert np.isclose(null.stressed_available_capital, static_ac)
    assert np.isclose(null.stressed_ratio, static_ac / total_scr)

    dyn = fcf.vfa.assess_dynamic(
        pf, mp, basis, static_available_capital=static_ac, total_scr=total_scr,
        return_shock=-0.30, lapse_sensitivity=0.8, haircut=0.10)
    loss = fcf.vfa.interaction_loss(pf, mp, basis, return_shock=-0.30,
                                       lapse_sensitivity=0.8, haircut=0.10).total_loss
    assert np.isclose(dyn.interaction.total_loss, loss)
    assert np.isclose(dyn.stressed_available_capital, static_ac - loss)
    assert np.isclose(dyn.stressed_ratio, (static_ac - loss) / total_scr)
    assert dyn.stressed_ratio < null.stressed_ratio          # the shock erodes the ratio

    # A risk-free book (zero required capital) gives an unbounded ratio.
    inf = fcf.vfa.assess_dynamic(
        pf, mp, basis, static_available_capital=static_ac, total_scr=0.0,
        return_shock=0.0, lapse_sensitivity=0.0, haircut=0.0)
    assert inf.stressed_ratio == float("inf")


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
    res = sa.interaction_loss(pf, mp, basis, shift=shift,
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
    res = sa.interaction_loss(pf, mp, basis, shift=0.01,
                                  lapse_sensitivity=8.0, haircut=0.0)
    assert res.forced_sale_loss == 0.0
    assert np.isclose(res.total_loss, res.revaluation_loss)


def _solvency_portfolio():
    return assets.AssetPortfolio(holdings=(
        alm.Bond(2_600_000.0, 0.03, 10, 1), assets.Cash(3_000_000.0)))


def test_assess_dynamic_decomposes():
    """The dynamic ratio takes the coupled-stress interaction loss off the static
    available capital and divides by the unchanged required capital; the parts
    reproduce their standalone builds."""
    mp, basis = _mp(), _basis()
    pf = _solvency_portfolio()
    d = sa.assess_dynamic(pf, mp, basis, regime=fcf.solvency.SII,
                                shift=0.01, lapse_sensitivity=8.0, haircut=0.1)
    static = sa.assess(pf, mp, basis, regime=fcf.solvency.SII)
    interaction = sa.interaction_loss(pf, mp, basis, shift=0.01,
                                          lapse_sensitivity=8.0, haircut=0.1)
    assert np.isclose(d.static.ratio, static.ratio)
    assert np.isclose(d.interaction.total_loss, interaction.total_loss)
    assert np.isclose(d.stressed_available_capital,
                      static.available_capital - interaction.total_loss)
    assert np.isclose(d.stressed_ratio, d.stressed_available_capital / static.total_scr)
    # the liquidation trajectory is exposed for the surplus / forced-sale path
    assert np.isclose(d.liquidation.total_realized_loss, d.interaction.forced_sale_loss)


def test_assess_dynamic_zero_scenario_is_static():
    """A null scenario (no shift, no haircut) leaves the ratio at the static value."""
    mp, basis = _mp(), _basis()
    pf = _solvency_portfolio()
    d = sa.assess_dynamic(pf, mp, basis, regime=fcf.solvency.KICS,
                                shift=0.0, lapse_sensitivity=8.0, haircut=0.0)
    assert d.interaction.total_loss == 0.0
    assert np.isclose(d.stressed_ratio, d.static.ratio)
    assert np.isclose(d.stressed_available_capital, d.static.available_capital)


def test_assess_dynamic_deeper_haircut_lowers_ratio():
    """A deeper liquidation haircut crystallises more forced-sale loss, lowering the
    stressed coverage ratio (the static ratio is unchanged)."""
    mp, basis = _mp(), _basis()
    pf = _solvency_portfolio()
    shallow = sa.assess_dynamic(pf, mp, basis, regime=fcf.solvency.SII,
                                      shift=0.01, lapse_sensitivity=8.0, haircut=0.05)
    deep = sa.assess_dynamic(pf, mp, basis, regime=fcf.solvency.SII,
                                   shift=0.01, lapse_sensitivity=8.0, haircut=0.30)
    assert deep.interaction.forced_sale_loss > shallow.interaction.forced_sale_loss
    assert deep.stressed_ratio < shallow.stressed_ratio
    assert np.isclose(deep.static.ratio, shallow.static.ratio)


def test_assess_dynamic_passes_through_assess_kwargs():
    """Extra kwargs (e.g. tax_rate) reach assess unchanged."""
    mp, basis = _mp(), _basis()
    pf = _solvency_portfolio()
    d = sa.assess_dynamic(pf, mp, basis, regime=fcf.solvency.KICS, shift=0.0,
                                lapse_sensitivity=0.0, haircut=0.0, tax_rate=0.22)
    taxed = sa.assess(pf, mp, basis, regime=fcf.solvency.KICS, tax_rate=0.22)
    assert np.isclose(d.static.tax_adjustment, taxed.tax_adjustment)
    assert d.static.tax_adjustment > 0.0


def test_assess_dynamic_report_renders():
    """report(assess_dynamic(...)) yields an ASCII DynamicAssessmentReport with the
    static / scenario / after-scenario blocks, and ties to the result."""
    mp = _mp()
    basis = make_death_basis(mortality_q=0.001, lapse_q=0.03, discount_annual=0.03,
                             mortality_cv=0.0)
    pf = _solvency_portfolio()
    d = sa.assess_dynamic(pf, mp, basis, regime=fcf.solvency.SII,
                                shift=0.01, lapse_sensitivity=8.0, haircut=0.1)
    rep = fcf.report(d)
    assert isinstance(rep, fcf.DynamicAssessmentReport)
    text = str(rep)
    assert all(ord(c) < 128 for c in text)                 # ASCII only (global surface)
    for label in ("Dynamic solvency", "Solvency ratio", "Total interaction loss",
                  "Stressed solvency ratio"):
        assert label in text
    assert f"{d.stressed_available_capital:,.0f}" in text  # the figure ties to the result


def test_assess_vfa_assembles_the_static_ratio():
    """The VFA static assessment prices the insurance module on the VFA net BEL,
    adds the guarantee equity sensitivity to the asset-side equity SCR, and forms
    the coverage ratio. The net interest module is zero (v1)."""
    import fastcashflow.solvency as sv
    pf, mp, basis = _vfa_book()
    regime = sv.KICS
    a = fcf.vfa.assess(pf, mp, basis, regime=regime)

    # Insurance module is the VFA life SCR; BEL is the VFA net BEL.
    scr = fcf.vfa.required_capital(mp, basis, regime=regime)
    assert np.isclose(a.insurance_scr, scr.insurance_scr)
    assert np.isclose(a.bel, scr.base_bel)
    assert np.isclose(a.net_interest_scr,
                      fcf.vfa.interest_scr(mp, basis, shift=0.01))   # parallel 100bp
    # Equity = asset equity + guarantee equity (added under one shock).
    eq_shock = sa._market_cal(regime)["equity_shocks"]["developed"]
    assert np.isclose(
        a.equity_scr,
        sa.equity_scr(pf, regime)
        + fcf.vfa.equity_scr(mp, basis, equity_shock=eq_shock))
    assert a.equity_scr > 0.0                                 # the guarantee bites
    # Available capital = assets - (VFA BEL + risk margin); ratio = ac / total_scr.
    assert np.isclose(a.available_capital,
                      a.asset_portfolio_value - a.bel - a.risk_margin)
    assert np.isclose(a.ratio, a.available_capital / a.total_scr)


def test_assess_dynamic_vfa_computes_its_own_static_from_a_regime():
    """Passing regime= makes assess_dynamic_vfa compute the static assessment via
    assess_vfa, then overlay the scenario interaction on it."""
    import fastcashflow.solvency as sv
    pf, mp, basis = _vfa_book()
    static = fcf.vfa.assess(pf, mp, basis, regime=sv.KICS)

    d = fcf.vfa.assess_dynamic(
        pf, mp, basis, regime=sv.KICS, return_shock=-0.30,
        lapse_sensitivity=0.8, haircut=0.10)
    assert d.static is not None
    assert np.isclose(d.static_available_capital, static.available_capital)
    assert np.isclose(d.total_scr, static.total_scr)
    assert np.isclose(d.stressed_available_capital,
                      static.available_capital - d.interaction.total_loss)
    assert d.stressed_ratio < static.ratio          # the shock erodes it

    # Neither a regime nor a supplied static position -> a clear error.
    with pytest.raises(ValueError, match="regime="):
        fcf.vfa.assess_dynamic(pf, mp, basis, return_shock=0.0,
                                    lapse_sensitivity=0.0, haircut=0.0)


def _ul_book():
    """A universal-life (account-backed) VFA book + a backing bond portfolio."""
    from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
    coi = 0.0015
    basis = Basis(
        mortality_annual=0.004, lapse_annual=0.03, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1, investment_return=0.06,
        coi_annual=coi, premium_load=0.08,
        coverages=(CoverageRate("DEATH", coi, funds_from_account=True,
                                pays_account_balance=True),))
    mp = ModelPoints(
        issue_age=np.array([40.0]), premium=np.array([500_000.0]),
        term_months=np.array([60]), account_value=np.array([1_000_000.0]),
        minimum_death_benefit=np.array([80_000_000.0]),
        minimum_accumulation_benefit=np.array([40_000_000.0]),
        minimum_crediting_rate=np.array([0.0]), sex=np.array([0]),
        benefits={"DEATH": np.array([80_000_000.0])},
        calculation_methods={"DEATH": CalculationMethod.DEATH})
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=2_000_000, coupon_rate=0.04, maturity_years=5, frequency=1),))
    return pf, mp, basis


def test_assess_vfa_supports_account_backed_ul():
    """The VFA static assessment works on a universal-life account book: the SCR
    modules re-measure the UL net BEL via measure_vfa, so no special path is needed.
    The interest and equity guarantee modules are positive (the GMAB bites)."""
    import fastcashflow.solvency as sv
    pf, mp, basis = _ul_book()
    a = fcf.vfa.assess(pf, mp, basis, regime=sv.KICS)
    scr = fcf.vfa.required_capital(mp, basis, regime=sv.KICS)
    assert np.isclose(a.insurance_scr, scr.insurance_scr)
    assert np.isclose(a.bel, scr.base_bel)
    assert a.net_interest_scr > 0.0 and a.equity_scr > 0.0
    assert np.isfinite(a.ratio)


def test_ul_vfa_net_liability_cashflows_reconciles_to_bel():
    """HARD GATE: the UL entity net-liability cash flow reconciles to the UL net
    BEL. With no crediting guarantee and a zero underlying-items return, credit ==
    discount == 0, so the account-value pass-through and credited interest net
    exactly against the held fund and the undiscounted entity_net sum == net BEL."""
    from fastcashflow import (Basis, CalculationMethod, CoverageRate, ModelPoints,
                              NO_GUARANTEE_RATE)
    coi = 0.0015
    basis = Basis(
        mortality_annual=0.006, lapse_annual=0.03, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1, investment_return=0.0,
        coi_annual=coi, premium_load=0.08,
        expense_items=(fcf.ExpenseItem("maint", "gamma_fixed", 300.0),),
        coverages=(CoverageRate("DEATH", coi, funds_from_account=True,
                                pays_account_balance=True),))
    mp = ModelPoints(   # mixed terms, real GMDB / GMAB, load + admin + COI exercised
        issue_age=np.array([40.0, 55.0]), premium=np.array([500_000.0, 300_000.0]),
        term_months=np.array([60, 36]),
        account_value=np.array([1_000_000.0, 500_000.0]),
        minimum_death_benefit=np.array([5_000_000.0, 3_000_000.0]),
        maturity_benefit=np.array([1_100_000.0, 520_000.0]),
        minimum_crediting_rate=np.array([NO_GUARANTEE_RATE, NO_GUARANTEE_RATE]),
        sex=np.array([0, 1]),
        benefits={"DEATH": np.array([5_000_000.0, 3_000_000.0])},
        calculation_methods={"DEATH": CalculationMethod.DEATH})
    m = fcf.vfa.measure(mp, basis, full=True)
    net = alm.vfa_net_liability_cashflows(m)
    assert net.shape == (m.cashflows.inforce.shape[1],)
    assert np.isclose(net.sum(), float(m.bel.sum()))         # reconciles to the UL net BEL


def test_ul_vfa_net_liability_reconciles_with_cost_deducting_rider():
    """HARD GATE (rider case): with a cost-deducting rider -- its charge drawn from
    the account (entity income) AND its benefit paid as a morbidity claim (entity
    outflow) -- the UL entity net liability still reconciles to the net BEL. Pins
    that the rider claim is added back, not just the charge subtracted."""
    from fastcashflow import (Basis, CalculationMethod, CoverageRate, ModelPoints,
                              NO_GUARANTEE_RATE)
    basis = Basis(
        mortality_annual=0.006, lapse_annual=0.02, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1, investment_return=0.0,
        coi_annual=0.0015, premium_load=0.08,
        expense_items=(fcf.ExpenseItem("maint", "gamma_fixed", 300.0),),
        coverages=(CoverageRate("DEATH", 0.0015, funds_from_account=True,
                                pays_account_balance=True),
                   CoverageRate("CANCER", 0.004, funds_from_account=True,
                                pays_account_balance=False)))
    mp = ModelPoints(
        issue_age=np.array([45.0, 50.0]), premium=np.array([400_000.0, 300_000.0]),
        term_months=np.array([48, 36]),
        account_value=np.array([800_000.0, 500_000.0]),
        minimum_death_benefit=np.array([4_000_000.0, 3_000_000.0]),
        maturity_benefit=np.array([900_000.0, 520_000.0]),
        minimum_crediting_rate=np.array([NO_GUARANTEE_RATE, NO_GUARANTEE_RATE]),
        sex=np.array([0, 1]),
        benefits={"DEATH": np.array([4_000_000.0, 3_000_000.0]),
                  "CANCER": np.array([200_000.0, 150_000.0])},
        calculation_methods={"DEATH": CalculationMethod.DEATH,
                             "CANCER": CalculationMethod.MORBIDITY})
    m = fcf.vfa.measure(mp, basis, full=True)
    assert m.cashflows.account.account_charge.sum() > 0.0    # rider charge drawn
    assert m.cashflows.morbidity_cf.sum() > 0.0              # rider benefit paid
    assert np.isclose(alm.vfa_net_liability_cashflows(m).sum(), float(m.bel.sum()))


def test_vfa_gap_and_interaction_support_account_backed_ul():
    """The VFA asset-liability gap / interaction now work on a UL book (the account
    charge flows let the entity net liability be built on the guarantee-excess
    basis), alongside the static assessment."""
    import fastcashflow.solvency as sv
    pf, mp, basis = _ul_book()
    m = fcf.vfa.measure(mp, basis, full=True)
    gap = assets.vfa_cashflow_gap(pf, m)
    assert np.all(np.isfinite(gap.net_cf))
    il = fcf.vfa.interaction_loss(pf, mp, basis, return_shock=-0.3,
                                     lapse_sensitivity=0.0, haircut=0.0)
    assert np.isfinite(il.total_loss)
    ds = fcf.vfa.assess_dynamic(pf, mp, basis, regime=sv.KICS,
                                     return_shock=-0.3, lapse_sensitivity=0.0,
                                     haircut=0.0)
    assert np.isfinite(ds.stressed_ratio)


def _vfa_guarantee_book():
    """A VFA book with a costly guarantee + a backing bond portfolio + ESG returns."""
    from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints, esg
    basis = Basis(mortality_annual=0.004, lapse_annual=0.02, discount_annual=0.03,
                  ra_confidence=0.75, mortality_cv=0.1, investment_return=0.05,
                  fund_fee=0.015, coverages=(CoverageRate("DEATH", 0.004),))
    mp = ModelPoints.single(45, 0.0, 60, account_value=1e7,
                            minimum_accumulation_benefit=1.2e7,
                            minimum_crediting_rate=0.02,
                            benefits={"DEATH": 0.0},
                            calculation_methods={"DEATH": CalculationMethod.DEATH})
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=1.5e7, coupon_rate=0.04, maturity_years=5, frequency=1),))
    es = esg.simulate(np.array([1., 2., 3., 5., 10., 20.]),
                      np.array([.031, .0355, .0368, .039, .0408, .041]),
                      ufr=0.0405, alpha=0.10, mean_reversion=0.10, rate_vol=0.01,
                      equity_vol=0.15, correlation=-0.2, n_scenarios=300,
                      n_time=60, seed=7)
    return pf, mp, basis, es


def test_assess_stochastic_vfa_reconciles_and_tails():
    """The stochastic coverage ratio distributes over ESG scenarios: it accepts the
    EconomicScenarios object, its mean reconciles to the static ratio less the
    guarantee time-value drag, and cte(5) <= p5 <= mean (the tail ordering)."""
    import fastcashflow.solvency as sv
    pf, mp, basis, es = _vfa_guarantee_book()
    ss = fcf.vfa.assess_stochastic(pf, mp, basis, es, regime=sv.KICS)   # ESG object
    assert ss.ratio.shape == (300,)

    dist = fcf.vfa.stochastic(mp, basis, es.returns)
    tv_drag = dist.bel.mean() - ss.static.bel
    expected_mean = (ss.static.available_capital - tv_drag) / ss.static.total_scr
    assert np.isclose(ss.mean()["ratio"], expected_mean)
    assert ss.cte(5) <= ss.percentile(5)["ratio"] <= ss.mean()["ratio"]
    # The stochastic guarantee tail is worse than the prescribed-SCR t=0 ratio.
    assert ss.mean()["ratio"] < ss.static.ratio


def test_assess_stochastic_vfa_accepts_a_raw_array_and_validates_cte():
    import fastcashflow.solvency as sv
    pf, mp, basis, es = _vfa_guarantee_book()
    ss = fcf.vfa.assess_stochastic(pf, mp, basis, es.returns, regime=sv.KICS)  # raw array
    assert np.isfinite(ss.mean()["ratio"])
    with pytest.raises(ValueError, match="q must be"):
        ss.cte(0.0)


# ---------------------------------------------------------------------------
# assess_stochastic_gmm -- the GMM coverage-ratio distribution over rate
# scenarios (the GMM counterpart of assess_stochastic_vfa).
# ---------------------------------------------------------------------------

def _gmm_solvency_book():
    """A GMM death book with a backing bond portfolio, flat base curve."""
    from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
    basis = Basis(mortality_annual=0.004, lapse_annual=0.02, discount_annual=0.03,
                  ra_confidence=0.75, mortality_cv=0.1,
                  coverages=(CoverageRate("DEATH", 0.004),))
    mp = ModelPoints.single(45, 50_000.0, 60, benefits={"DEATH": 1e7},
                            calculation_methods={"DEATH": CalculationMethod.DEATH})
    pf = assets.AssetPortfolio(holdings=(
        alm.Bond(face=1.5e7, coupon_rate=0.04, maturity_years=5, frequency=1),))
    return pf, mp, basis


def test_assess_stochastic_gmm_reconciles_at_base_curve():
    """A single scenario equal to the basis's own flat discount rate reproduces
    the static coverage ratio exactly -- the null-scenario anchor."""
    import fastcashflow.solvency as sv
    pf, mp, basis = _gmm_solvency_book()
    static = fcf.solvency.assess(pf, mp, basis, regime=sv.SII)
    ss = fcf.solvency.assess_stochastic(pf, mp, basis, np.array([0.03]), regime=sv.SII)
    assert ss.ratio.shape == (1,)
    assert np.isclose(ss.available_capital[0], static.available_capital)
    assert np.isclose(ss.ratio[0], static.ratio)


def test_assess_stochastic_gmm_distributes_over_rate_scenarios():
    """The available capital / ratio is assets less the per-scenario BEL and risk
    margin, over the prescribed SCR -- and the tail orders cte <= percentile."""
    import fastcashflow.solvency as sv
    pf, mp, basis = _gmm_solvency_book()
    scen = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    ss = fcf.solvency.assess_stochastic(pf, mp, basis, scen, regime=sv.SII)
    assert ss.ratio.shape == (5,)

    dist = fcf.gmm.stochastic(mp, basis, scen)
    expected_ac = ss.static.asset_portfolio_value - (dist.bel + ss.static.risk_margin)
    assert np.allclose(ss.available_capital, expected_ac)
    assert np.allclose(ss.ratio, expected_ac / ss.static.total_scr)
    # available capital is the assets less the per-scenario BEL: the worst-BEL
    # scenario carries the least capital (assets / risk margin held fixed)
    assert np.argmax(dist.bel) == np.argmin(ss.available_capital)
    assert ss.cte(40) <= ss.percentile(40)["ratio"]


def test_assess_stochastic_gmm_accepts_esg_object():
    """It accepts an EconomicScenarios (its .rates feed the GMM distribution),
    like its VFA counterpart accepts .returns."""
    from fastcashflow import esg
    import fastcashflow.solvency as sv
    pf, mp, basis = _gmm_solvency_book()
    es = esg.simulate(np.array([1., 2., 3., 5., 10., 20.]),
                      np.array([.031, .0355, .0368, .039, .0408, .041]),
                      ufr=0.0405, alpha=0.10, mean_reversion=0.10, rate_vol=0.01,
                      equity_vol=0.15, correlation=-0.2, n_scenarios=200,
                      n_time=60, seed=7)
    ss = fcf.solvency.assess_stochastic(pf, mp, basis, es, regime=sv.SII)   # ESG -> .rates
    assert ss.ratio.shape == (200,)
    assert np.isfinite(ss.mean()["ratio"])
    assert ss.cte(5) <= ss.percentile(5)["ratio"] <= ss.mean()["ratio"]


# ---------------------------------------------------------------------------
# asset_value_by_scenario -- the co-moving asset leg (P1): revalue the portfolio
# per rate scenario, the bond discounting matching the liability's at the year grid.
# ---------------------------------------------------------------------------

def _bond_heavy_portfolio():
    return assets.AssetPortfolio(holdings=(
        alm.Bond(1.5e7, 0.04, 10, 1), assets.Equity(5e5), assets.Cash(1e6)))


def test_asset_value_by_scenario_flat_is_exact():
    """A 1-D scenario is one flat rate per scenario -- each value is the portfolio
    priced at that flat rate; a lower rate gives a higher value (bond duration)."""
    pf = _bond_heavy_portfolio()
    flat = np.array([0.01, 0.03, 0.05])
    av = assets.asset_value_by_scenario(pf, flat)
    exp = np.array([assets.asset_portfolio_value(pf, r) for r in flat])
    assert np.allclose(av, exp)
    assert av[0] > av[1] > av[2]                        # lower rate -> higher bond value


def test_asset_value_by_scenario_curve_matches_liability_df_at_year_grid():
    """The annual-forward bootstrap reproduces the liability's cumulative discount
    factor at the year grid exactly -- so the co-moving asset and the stochastic
    liability discount on the same curve (no asset-liability discounting drift)."""
    from fastcashflow.alm import _annual_df
    from fastcashflow.assets import _annual_forward_curve
    rng = np.random.default_rng(0)
    n_time = 60
    rs = 0.03 + 0.01 * rng.standard_normal((4, n_time))
    monthly = (1.0 + rs) ** (1.0 / 12.0) - 1.0
    dfm = np.concatenate([np.ones((4, 1)),
                          np.cumprod(1.0 / (1.0 + monthly), axis=1)], axis=1)
    c = _annual_forward_curve(rs)
    for s in range(4):
        for j in range(1, c.shape[1] + 1):
            asset_df = float(_annual_df(np.array([float(j)]), c[s])[0])
            assert np.isclose(dfm[s, 12 * j], asset_df, rtol=1e-12), (s, j)


def test_asset_value_by_scenario_flat_per_year_bootstrap_is_the_annual_rate():
    """A path flat within each policy year bootstraps to that year's annual rate."""
    from fastcashflow.assets import _annual_forward_curve
    rates = [0.02, 0.04, 0.06, 0.06, 0.06]
    path = np.repeat(np.array([rates]), 12, axis=1).reshape(1, 60)
    assert np.allclose(_annual_forward_curve(path)[0], rates)


def test_asset_value_by_scenario_rejects_3d():
    with pytest.raises(ValueError, match="1-D .* or 2-D"):
        assets.asset_value_by_scenario(_bond_heavy_portfolio(), np.zeros((2, 3, 4)))


# ---------------------------------------------------------------------------
# assess_stochastic_gmm co_moving_assets (P2): the asset value moves with the
# rate scenario, so the ratio reflects the asset-liability duration gap.
# ---------------------------------------------------------------------------

def test_assess_stochastic_gmm_co_moving_off_is_unchanged():
    """co_moving_assets=False (the default) holds the asset value fixed -- the
    liability-only distribution, identical to the prior behaviour."""
    import fastcashflow.solvency as sv
    pf, mp, basis = _gmm_solvency_book()
    scen = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    ss = fcf.solvency.assess_stochastic(pf, mp, basis, scen, regime=sv.SII)
    expected_ac = ss.static.asset_portfolio_value - (
        fcf.gmm.stochastic(mp, basis, scen).bel + ss.static.risk_margin)
    assert np.allclose(ss.available_capital, expected_ac)   # asset value fixed


def test_assess_stochastic_gmm_co_moving_formula_and_anchor():
    """co_moving_assets=True revalues the assets per scenario; a flat scenario equal
    to the base discount reproduces the static ratio EXACTLY on both legs."""
    import fastcashflow.solvency as sv
    pf, mp, basis = _gmm_solvency_book()
    scen = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    ss = fcf.solvency.assess_stochastic(pf, mp, basis, scen, regime=sv.SII,
                                     co_moving_assets=True)
    dist = fcf.gmm.stochastic(mp, basis, scen)
    asset_val = fcf.assets.asset_value_by_scenario(pf, scen)
    assert np.allclose(ss.available_capital, asset_val - (dist.bel + ss.static.risk_margin))

    base = fcf.solvency.assess_stochastic(pf, mp, basis, np.array([0.03]),
                                       regime=sv.SII, co_moving_assets=True)
    assert np.isclose(base.available_capital[0], ss.static.available_capital)
    assert np.isclose(base.ratio[0], ss.static.ratio)


def test_assess_stochastic_gmm_co_moving_differs_from_fixed():
    """With co-moving assets the available capital differs from the fixed-asset run
    away from the base curve -- the bond revaluation is the duration-gap leg."""
    import fastcashflow.solvency as sv
    pf, mp, basis = _gmm_solvency_book()
    scen = np.array([0.01, 0.05])
    fixed = fcf.solvency.assess_stochastic(pf, mp, basis, scen, regime=sv.SII)
    moving = fcf.solvency.assess_stochastic(pf, mp, basis, scen, regime=sv.SII,
                                         co_moving_assets=True)
    # the low-rate scenario lifts the bond value above its base level (co-moving),
    # so its available capital exceeds the fixed-asset figure
    assert moving.available_capital[0] > fixed.available_capital[0]
    assert not np.allclose(moving.available_capital, fixed.available_capital)


# ---------------------------------------------------------------------------
# assess_stochastic_vfa co_moving_assets (P3): the entity bonds co-move with the
# scenario RATE path (the joint ESG), not the fund return.
# ---------------------------------------------------------------------------

def test_assess_stochastic_vfa_co_moving_off_is_unchanged():
    """co_moving_assets=False (the default) holds the asset value fixed."""
    import fastcashflow.solvency as sv
    pf, mp, basis, es = _vfa_guarantee_book()
    ss = fcf.vfa.assess_stochastic(pf, mp, basis, es, regime=sv.KICS)
    expected_ac = ss.static.asset_portfolio_value - (
        fcf.vfa.stochastic(mp, basis, es.returns).bel + ss.static.risk_margin)
    assert np.allclose(ss.available_capital, expected_ac)


def test_assess_stochastic_vfa_co_moving_requires_esg():
    """The bonds co-move with rates, not the fund return, so a raw returns array
    (no rate path) is rejected when co_moving_assets=True."""
    import fastcashflow.solvency as sv
    pf, mp, basis, es = _vfa_guarantee_book()
    with pytest.raises(ValueError, match="needs an EconomicScenarios"):
        fcf.vfa.assess_stochastic(pf, mp, basis, es.returns, regime=sv.KICS,
                                    co_moving_assets=True)


def test_assess_stochastic_vfa_co_moving_uses_the_rate_path():
    """co_moving_assets=True revalues the bonds on the ESG rate path (its .rates),
    while the fund returns still drive the guarantee liability."""
    import fastcashflow.solvency as sv
    pf, mp, basis, es = _vfa_guarantee_book()
    ss = fcf.vfa.assess_stochastic(pf, mp, basis, es, regime=sv.KICS,
                                     co_moving_assets=True)
    dist = fcf.vfa.stochastic(mp, basis, es.returns)
    asset_val = fcf.assets.asset_value_by_scenario(pf, es.rates)
    assert np.allclose(ss.available_capital, asset_val - (dist.bel + ss.static.risk_margin))
    # the asset value now varies across scenarios (vs a single fixed level)
    fixed = fcf.vfa.assess_stochastic(pf, mp, basis, es, regime=sv.KICS)
    assert not np.allclose(ss.available_capital, fixed.available_capital)
