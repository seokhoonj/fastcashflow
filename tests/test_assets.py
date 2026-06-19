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


def test_credit_scr_sii_deferred():
    """Solvency II credit (spread / counterparty) is a separate framework -> 0."""
    b = alm.Bond(100.0, 0.03, 10, 1, credit_rating="BBB")
    p = assets.AssetPortfolio(holdings=(b,))
    assert assets.credit_scr(p, fcf.SOLVENCY2, 0.03) == 0.0


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


def test_fx_scr_sii_deferred():
    """Solvency II currency risk is a separate calibration -> 0."""
    p = assets.AssetPortfolio(holdings=(assets.Cash(1000.0, currency="USD"),))
    assert assets.fx_scr(p, fcf.SOLVENCY2, 0.03) == 0.0


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
    """A book with no tagged issuers and no property has no concentration charge."""
    p = assets.AssetPortfolio(holdings=(alm.Bond(1e6, 0.03, 5, 1), assets.Cash(1e6)))
    assert assets.concentration_scr(p, fcf.KICS, 0.03) == 0.0
    # Solvency II concentration is deferred
    tagged = assets.AssetPortfolio(holdings=(
        alm.Bond(1e6, 0.03, 5, 1, issuer="X"),))
    assert assets.concentration_scr(tagged, fcf.SOLVENCY2, 0.03) == 0.0


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
    # no equity/property -> the market module is just the net interest SCR, and the
    # SII top-level BSCR is a simple sum; SII credit is deferred (0); the total
    # adds operational on top
    assert np.isclose(a.market_module_scr, a.net_interest_scr)
    assert a.credit_scr == 0.0                              # SII credit deferred
    assert np.isclose(a.bscr, a.insurance_scr + a.net_interest_scr + a.credit_scr)
    assert np.isclose(a.basic_required_capital, a.bscr + a.operational_scr)
    assert a.tax_adjustment == 0.0                          # no tax relief by default
    assert np.isclose(a.total_scr, a.basic_required_capital - a.tax_adjustment)
    assert np.isclose(a.solvency_ratio, a.available_capital / a.total_scr)
    assert np.isclose(a.available_capital, a.portfolio_value - (a.bel + a.risk_margin))


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
    """K-ICS aggregates insurance, market and credit with the table-3 correlation
    (all 0.25); Solvency II falls back to a simple sum (top-level matrix not
    extracted, and SII credit deferred)."""
    mp, basis = _mp(), _basis()
    p = assets.AssetPortfolio(holdings=(
        alm.Bond(2_000_000.0, 0.03, 10, 1), assets.Equity(3_000_000.0, "developed")))
    k = assets.assess_solvency(p, mp, basis, regime=fcf.KICS)
    s = assets.assess_solvency(p, mp, basis, regime=fcf.SOLVENCY2)
    # SII: simple sum, credit deferred (0)
    assert s.credit_scr == 0.0
    assert np.isclose(s.bscr, s.insurance_scr + s.market_module_scr + s.credit_scr)
    # K-ICS: the bond now carries a credit charge -> 3-module sqrt aggregation
    assert k.credit_scr > 0.0
    m = np.array([k.insurance_scr, k.market_module_scr, k.credit_scr])
    R = np.array([[1.0, 0.25, 0.25], [0.25, 1.0, 0.25], [0.25, 0.25, 1.0]])
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
    # Solvency II sums the modules (no top-level diversification)
    s = assets.aggregate_required_capital(100.0, 200.0, 50.0, regime=fcf.SOLVENCY2,
                                          operational=10.0)
    assert np.isclose(s, 360.0)


def test_preferred_equity_by_rating_table20():
    """Preferred equity (table 20) is charged by the issue's K-ICS grade: 1-2 grade
    4%, 3 grade 6%, 4 grade 11%, 5 grade 21%, 6+ grade 35%, unrated 35%."""
    for rating, shock in [("AA", 0.04), ("A", 0.06), ("BBB", 0.11), ("BB", 0.21),
                          ("B", 0.35), ("unrated", 0.35)]:
        p = assets.AssetPortfolio(holdings=(
            assets.Equity(1_000_000.0, "preferred", credit_rating=rating),))
        assert np.isclose(assets.equity_scr(p, fcf.KICS), 1_000_000.0 * shock)
