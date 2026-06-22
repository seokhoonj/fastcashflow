"""Solvency-capital (SCR) engine -- mechanics hand-calc anchors (regime-agnostic).

These pin the engine itself: the correlation aggregation, the shock -> re-measure
-> max(Delta BEL, 0) capital, the two-sited mortality shock, the worst-of
selection, the mass-lapse count haircut, and the two risk-margin methods. The
regime calibrations (K-ICS, Solvency II) and their primary-source numbers are
checked separately.
"""
import math

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import solvency as sv
from fastcashflow.coverage import CalculationMethod
from fastcashflow.engine import measure
from fastcashflow.numerics import _cost_of_capital_ra
from fastcashflow.curves import discount_monthly_curve
from fastcashflow.state_model import StateModel, State, Transition

from conftest import make_death_basis, PATTERNS, annual_from_monthly


def _basis(**over):
    kw = dict(mortality_q=0.002, lapse_q=0.004, discount_annual=0.03, mortality_cv=0.10)
    kw.update(over)
    return make_death_basis(**kw)


def _mp(term=120):
    return fcf.ModelPoints.single(40, 60_000.0, term, benefits={"DEATH": 1e8},
                                  calculation_methods=PATTERNS)


def _dummy(corr=0.0, margin="percentile"):
    return sv.RegimeSpec(
        name="dummy",
        sub_risks=(
            sv.SubRisk("mortality", (sv.scale_mortality(1.125),), "single"),
            sv.SubRisk("lapse", (sv.scale_lapse(1.35), sv.scale_lapse(0.65),
                                 sv.mass_lapse(0.30)), "worst_of"),
        ),
        correlation=np.array([[1.0, corr], [corr, 1.0]]),
        risk_margin_method=margin, risk_margin_factor=0.40, risk_margin_coc_rate=0.06,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_two_risk_quadratic_form():
    spec = _dummy(corr=0.25)
    cap = {"mortality": 300.0, "lapse": 400.0}
    expected = math.sqrt(300.0**2 + 400.0**2 + 2 * 0.25 * 300.0 * 400.0)
    assert math.isclose(sv.aggregate(cap, spec), expected, rel_tol=1e-12)


def test_aggregate_zero_correlation_is_pythagorean():
    spec = _dummy(corr=0.0)
    cap = {"mortality": 30.0, "lapse": 40.0}
    assert math.isclose(sv.aggregate(cap, spec), 50.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Shock -> re-measure -> capital
# ---------------------------------------------------------------------------

def test_mortality_capital_is_delta_bel():
    mp, basis = _mp(), _basis()
    base = measure(mp, basis, full=False).bel.sum()
    _, shocked = sv.scale_mortality(1.125).apply(mp, basis)
    shocked_bel = measure(mp, shocked, full=False).bel.sum()
    expected = max(float(shocked_bel - base), 0.0)

    res = sv.required_capital(mp, basis, regime=_dummy())
    assert np.isclose(res.sub_risk_capital["mortality"], expected)
    assert expected > 0.0                      # a mortality-up stress raises the liability


def test_mortality_shock_is_two_sited():
    """The stress must scale BOTH the in-force decrement and the DEATH claim rate."""
    mp, basis = _mp(), _basis()
    _, shocked = sv.scale_mortality(1.125).apply(mp, basis)
    grid = (np.array([0]), np.array([40]), np.array([24]), np.array([0]), np.array([24]))
    # decrement
    assert np.allclose(shocked.mortality_annual(*grid),
                       basis.mortality_annual(*grid) * 1.125)
    # DEATH coverage claim rate
    base_cov = {r.code: r for r in basis.coverages}
    shock_cov = {r.code: r for r in shocked.coverages}
    assert np.allclose(shock_cov["DEATH"].rate(*grid),
                       base_cov["DEATH"].rate(*grid) * 1.125)


def test_longevity_shock_is_decrement_only():
    mp, basis = _mp(), _basis()
    _, shocked = sv.scale_longevity(0.825).apply(mp, basis)
    grid = (np.array([0]), np.array([40]), np.array([24]), np.array([0]), np.array([24]))
    assert np.allclose(shocked.mortality_annual(*grid),
                       basis.mortality_annual(*grid) * 0.825)
    base_cov = {r.code: r for r in basis.coverages}
    shock_cov = {r.code: r for r in shocked.coverages}
    assert np.allclose(shock_cov["DEATH"].rate(*grid), base_cov["DEATH"].rate(*grid))


def test_mass_lapse_is_count_haircut_when_no_surrender_value():
    """With no surrender_value_curve the mass lapse is the pure count haircut:
    max(BEL(count*0.70) - BEL(count), 0). The t=0 surrender-value add-on is zero
    (the basis prices no surrender value), so only the lost-business term bites."""
    from dataclasses import replace
    from fastcashflow.engine import inforce_surrender_value
    mp, basis = _mp(), _basis()
    assert np.allclose(inforce_surrender_value(mp, basis), 0.0)   # no curve -> no add-on
    base = measure(mp, basis, full=False).bel.sum()
    n = mp.n_mp
    count = np.ones(n) if mp.count is None else np.asarray(mp.count, float)
    mp_cut = replace(mp, count=count * 0.70)
    expected = max(float(measure(mp_cut, basis, full=False).bel.sum() - base), 0.0)
    single = sv.RegimeSpec(
        name="ml", sub_risks=(sv.SubRisk("lapse", (sv.mass_lapse(0.30),), "single"),),
        correlation=np.array([[1.0]]), risk_margin_method="percentile")
    assert np.isclose(
        sv.required_capital(mp, basis, regime=single).sub_risk_capital["lapse"], expected)


def test_mass_lapse_adds_t0_surrender_value():
    """With a surrender_value_curve the mass lapse adds the t=0 outflow: capital
    = max(count-haircut delta + fraction * sum(count * surrender_value), 0). The
    leaving fraction is paid its valuation-date surrender value -- the strain the
    count haircut (future cash flows only) cannot carry."""
    from dataclasses import replace
    from fastcashflow.engine import inforce_surrender_value
    f = 0.30
    amount = np.full(_mp().term_months[0] + 1, 5_000.0)   # flat per-policy SV
    basis = _basis(surrender_value_curve=amount,
                   surrender_value_basis="amount_per_policy")
    mp = _mp()
    base = float(measure(mp, basis, full=False).bel.sum())
    count = np.asarray(mp.count, float) if mp.count is not None else np.ones(mp.n_mp)
    haircut = float(measure(replace(mp, count=count * (1.0 - f)), basis,
                            full=False).bel.sum()) - base
    addon = f * float(inforce_surrender_value(mp, basis).sum())
    assert addon > 0.0                                   # the curve makes it bite
    expected = max(0.0, haircut + addon)
    single = sv.RegimeSpec(
        name="ml", sub_risks=(sv.SubRisk("lapse", (sv.mass_lapse(f),), "single"),),
        correlation=np.array([[1.0]]), risk_margin_method="percentile")
    assert np.isclose(
        sv.required_capital(mp, basis, regime=single).sub_risk_capital["lapse"], expected)


def test_worst_of_selects_largest():
    mp, basis = _mp(), _basis()
    base = measure(mp, basis, full=False).bel.sum()
    deltas = []
    for v in (sv.scale_lapse(1.35), sv.scale_lapse(0.65), sv.mass_lapse(0.30)):
        mp2, b2 = v.apply(mp, basis)
        deltas.append(float(measure(mp2, b2, full=False).bel.sum() - base))
    expected = max(0.0, max(deltas))
    res = sv.required_capital(mp, basis, regime=_dummy())
    assert np.isclose(res.sub_risk_capital["lapse"], expected)


# ---------------------------------------------------------------------------
# Risk margin
# ---------------------------------------------------------------------------

def test_risk_margin_percentile():
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=_dummy(margin="percentile"))
    assert np.isclose(res.risk_margin, res.insurance_scr * 0.40)
    assert res.scr_path is None


def test_risk_margin_cost_of_capital_reproduces_engine():
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=_dummy(margin="cost_of_capital"))
    assert res.scr_path is not None
    n_time = res.scr_path.shape[0] - 1
    dm = discount_monthly_curve(basis, n_time)
    expected = float(_cost_of_capital_ra(res.scr_path.reshape(1, -1), dm, 0.06)[0, 0])
    assert np.isclose(res.risk_margin, expected)
    assert res.risk_margin > 0.0


def test_regime_rejects_bad_correlation():
    sr = (sv.SubRisk("a", (sv.scale_lapse(1.1),)), sv.SubRisk("b", (sv.scale_lapse(1.2),)))
    with pytest.raises(ValueError, match="symmetric"):
        sv.RegimeSpec(name="x", sub_risks=sr,
                      correlation=np.array([[1.0, 0.3], [0.1, 1.0]]))
    with pytest.raises(ValueError, match="diagonal"):
        sv.RegimeSpec(name="x", sub_risks=sr,
                      correlation=np.array([[1.0, 0.3], [0.3, 0.9]]))
    with pytest.raises(ValueError, match="to match"):
        sv.RegimeSpec(name="x", sub_risks=sr[:1],
                      correlation=np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_mortality_requires_classified_coverages():
    """A mortality stress must not silently skip the DEATH claim when the coverage
    cannot be classified (Codex gate A finding)."""
    from dataclasses import replace
    mp, basis = _mp(), _basis()
    mp_noclass = replace(mp, calculation_methods=None)
    if mp_noclass.calculation_methods is None:        # only if MP keeps it None
        with pytest.raises(ValueError, match="calculation_methods"):
            sv.scale_mortality(1.125).apply(mp_noclass, basis)


def test_total_is_insurance_plus_interest():
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=_dummy())
    assert np.isclose(res.total_scr, res.insurance_scr + res.interest_capital)
    assert res.interest_capital == 0.0          # dummy has no interest curves


# ---------------------------------------------------------------------------
# K-ICS interest-rate risk -- five-scenario aggregation (handbook 4-2, p.205)
#   capital = sqrt( max(up, down)^2 + max(flat, steep)^2 ) + mean_reversion
# ---------------------------------------------------------------------------

def _flat_spreads(up, down, flat, steep, mr):
    """A KICSInterest whose five scenarios are flat (single-maturity) spreads."""
    return sv.KICSInterest.from_spreads(
        up=[up], down=[down], flat=[flat], steep=[steep], mean_reversion=[mr])


def test_kics_interest_aggregation_formula():
    # Drive the formula with a stub delta keyed by scenario name: level pair
    # max(30, 50) = 50, twist pair max(40, 20) = 40, mean reversion -10 (signed).
    ki = _flat_spreads(0.01, -0.01, 0.005, -0.005, 0.0)
    amounts = {"interest up": 30.0, "interest down": 50.0, "interest flat": 40.0,
               "interest steep": 20.0, "interest mean reversion": -10.0}
    cap, comp = ki.capital(lambda s: amounts[s.name])
    assert cap == pytest.approx(math.sqrt(50.0**2 + 40.0**2) - 10.0)   # 64.0312... - 10
    assert comp["interest_down"] == 50.0 and comp["interest_flat"] == 40.0
    assert comp["interest_mean_reversion"] == -10.0                    # signed, kept


def test_kics_interest_directional_amounts_are_floored_mean_reversion_signed():
    # A scenario whose NAV change is a GAIN (negative Delta BEL) floors to 0 for the
    # four directional legs, but mean reversion keeps its sign (can lower the charge).
    ki = _flat_spreads(0.01, -0.01, 0.005, -0.005, 0.0)
    amounts = {"interest up": -5.0, "interest down": 80.0, "interest flat": -3.0,
               "interest steep": 0.0, "interest mean reversion": -12.0}
    cap, comp = ki.capital(lambda s: amounts[s.name])
    assert comp["interest_up"] == 0.0 and comp["interest_flat"] == 0.0   # floored
    assert cap == pytest.approx(math.sqrt(80.0**2 + 0.0**2) - 12.0)      # 80 - 12 = 68


def test_shock_spread_is_additive_and_held_flat():
    # base 0.03 flat; a two-entry spread [0.004, 0.006] then held flat at 0.006.
    mp, basis = _mp(), _basis()
    _, b = sv.shock_spread([0.004, 0.006], name="x").apply(mp, basis)
    curve = np.asarray(b.discount_annual, float)
    assert curve[0] == pytest.approx(0.034) and curve[1] == pytest.approx(0.036)
    assert np.allclose(curve[2:], 0.036)        # held flat past the last entry


def test_shock_spread_continuous_compounding():
    # A continuous shock spread meets the annual curve as (1+base)*exp(spread)-1,
    # NOT base+spread -- the form the FSS-published K-ICS shocks take.
    mp, basis = _mp(), _basis()                  # base 0.03 flat
    _, b = sv.shock_spread([0.01], name="x", compounding="continuous").apply(mp, basis)
    curve = np.asarray(b.discount_annual, float)
    assert curve[0] == pytest.approx((1.0 + 0.03) * np.exp(0.01) - 1.0)
    assert curve[0] > 0.03 + 0.01                # exceeds the naive additive 0.04
    with pytest.raises(ValueError):
        sv.shock_spread([0.01], name="x", compounding="discrete")


def test_kics_interest_capital_end_to_end():
    # An up shock raises the discount rate -> lowers PV(claims) of a death book ->
    # a GAIN (Delta BEL < 0), floored to 0; a down shock is the binding loss. Wire
    # through required_capital and reproduce the aggregation from the components.
    mp, basis = _mp(), _basis()
    up, down = [0.010], [-0.010]                 # +/-1pp parallel, flat past year 1
    flat, steep = [0.004], [-0.004]
    ki = sv.KICSInterest.from_spreads(up=up, down=down, flat=flat, steep=steep,
                                      mean_reversion=[0.0])
    res = sv.required_capital(mp, basis, regime=sv.KICS, interest_scenarios=ki)
    c = res.sub_risk_capital
    expected = math.sqrt(max(c["interest_up"], c["interest_down"])**2
                         + max(c["interest_flat"], c["interest_steep"])**2) \
        + c["interest_mean_reversion"]
    assert res.interest_capital == pytest.approx(expected)
    assert res.interest_capital > 0.0                       # down shock binds
    assert c["interest_up"] == 0.0                          # up shock is a gain -> floored
    assert np.isclose(res.total_scr, res.insurance_scr + res.interest_capital)


def test_kics_from_spreads_passes_compounding_through():
    # from_spreads(compounding="continuous") applies every scenario in continuous
    # space, so the up curve is (1+base)*exp(spread)-1 -- the FSS data path.
    mp, basis = _mp(), _basis()                  # base 0.03 flat
    ki = sv.KICSInterest.from_spreads(up=[0.01], down=[-0.01], flat=[0.004],
                                      steep=[-0.004], mean_reversion=[0.0],
                                      compounding="continuous")
    _, b = ki.up.apply(mp, basis)
    assert np.asarray(b.discount_annual, float)[0] == pytest.approx(
        (1.0 + 0.03) * np.exp(0.01) - 1.0)


# ---------------------------------------------------------------------------
# K-ICS calibration (primary-source numbers; Codex gate B cross-checks the source)
# ---------------------------------------------------------------------------

def test_kics_parameters():
    spec = sv.KICS
    names = [sr.name for sr in spec.sub_risks]
    assert names == ["mortality", "longevity", "morbidity", "lapse", "expense"]
    # mortality factor 1.125, longevity 0.825 (evaluate the wrapped rate)
    _, b_m = spec.sub_risks[0].variants[0].apply(*( _mp(), _basis()))
    _, b_l = spec.sub_risks[1].variants[0].apply(*( _mp(), _basis()))
    grid = (np.array([0]), np.array([40]), np.array([24]), np.array([0]), np.array([24]))
    base = _basis()
    assert np.allclose(b_m.mortality_annual(*grid), base.mortality_annual(*grid) * 1.125)
    assert np.allclose(b_l.mortality_annual(*grid), base.mortality_annual(*grid) * 0.825)
    # lapse worst-of has three variants incl. mass lapse
    assert spec.sub_risks[3].combine == "worst_of"
    assert len(spec.sub_risks[3].variants) == 3
    # correlation: symmetric, unit diagonal, named cells from the source table
    R = np.asarray(spec.correlation)
    assert np.allclose(R, R.T) and np.allclose(np.diag(R), 1.0)
    assert R[0, 1] == -0.25       # mortality x longevity
    assert R[2, 4] == 0.50        # morbidity x expense
    assert R[0, 3] == 0.00        # mortality x lapse
    # percentile risk margin factor
    assert spec.risk_margin_method == "percentile" and spec.risk_margin_factor == 0.40


def test_kics_runs_end_to_end():
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=sv.KICS)
    assert set(res.sub_risk_capital) == {"mortality", "longevity", "morbidity", "lapse", "expense"}
    assert res.insurance_scr > 0.0
    assert all(c >= 0.0 for c in res.sub_risk_capital.values())
    assert np.isclose(res.risk_margin, res.insurance_scr * 0.40)
    assert res.regime == "K-ICS"


# ---------------------------------------------------------------------------
# Solvency II calibration (primary-source numbers; Codex gate C cross-checks)
# ---------------------------------------------------------------------------

def test_sii_parameters():
    spec = sv.SII
    names = [sr.name for sr in spec.sub_risks]
    assert names == ["mortality", "longevity", "disability", "expense", "revision",
                     "lapse", "catastrophe"]
    grid = (np.array([0]), np.array([40]), np.array([24]), np.array([0]), np.array([24]))
    base = _basis()
    _, b_m = spec.sub_risks[0].variants[0].apply(_mp(), _basis())
    _, b_l = spec.sub_risks[1].variants[0].apply(_mp(), _basis())
    assert np.allclose(b_m.mortality_annual(*grid), base.mortality_annual(*grid) * 1.15)
    assert np.allclose(b_l.mortality_annual(*grid), base.mortality_annual(*grid) * 0.80)
    # lapse worst-of (+/-50%, mass 40%)
    assert spec.sub_risks[5].combine == "worst_of" and len(spec.sub_risks[5].variants) == 3
    # correlation cells from Annex IV point 3
    R = np.asarray(spec.correlation)
    assert np.allclose(R, R.T) and np.allclose(np.diag(R), 1.0)
    assert R[0, 1] == -0.25       # mortality x longevity
    assert R[2, 3] == 0.50        # disability x expense
    assert R[3, 5] == 0.50        # expense x lapse
    assert R[0, 4] == 0.00        # mortality x revision
    assert R[0, 6] == 0.25 and R[1, 6] == 0.00 and R[5, 6] == 0.25   # Art 136 cat row
    # cost-of-capital risk margin, EIOPA interest curves present
    assert spec.risk_margin_method == "cost_of_capital" and spec.risk_margin_coc_rate == 0.06
    assert spec.interest_curves is not None and len(spec.interest_curves) == 2


def test_sii_runs_end_to_end():
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=sv.SII)
    assert set(res.sub_risk_capital) == {
        "mortality", "longevity", "disability", "expense", "revision", "lapse",
        "catastrophe"}
    assert res.insurance_scr > 0.0
    assert res.interest_capital >= 0.0
    assert res.scr_path is not None             # cost-of-capital margin builds a path
    assert res.risk_margin > 0.0
    assert res.regime == "Solvency II"


def test_cost_of_capital_run_off_excludes_interest():
    """The cost-of-capital risk margin covers non-hedgeable (insurance) risk, so
    its capital run-off starts at the insurance SCR, not the total (interest-rate
    risk is excluded) -- Codex gate D finding."""
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=sv.SII)
    assert res.interest_capital > 0.0           # interest is material here
    assert np.isclose(res.scr_path[0], res.insurance_scr)
    assert np.all(res.scr_path >= 0.0)


def test_solvency_ratio():
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=sv.SII)
    assert np.isclose(sv.solvency_ratio(res, 20_000_000.0),
                      20_000_000.0 / res.total_scr)


def test_catastrophe_scr_hand_calc():
    """K-ICS catastrophe (handbook 2-8): pandemic = death SA x 0.1%; large accident
    = sum of zone-exposure x max(SA x shock - prior claims, 0); total = sqrt of the
    two (correlation 0)."""
    assert np.isclose(sv.catastrophe_scr(pandemic_death=1e12), 1e12 * 0.001)
    ad = sv.catastrophe_scr(accident_death=1e12)
    assert np.isclose(ad, 0.0000711 * 1e12 * 0.15 + 0.0003733 * 1e12 * 0.015)
    full = sv.catastrophe_scr(pandemic_death=1e12, accident_death=1e12,
                              disability=5e11, property=2e11)
    pan = 1e12 * 0.001
    large = (0.0000711 * 1e12 * 0.15 + 0.0003733 * 1e12 * 0.015
             + 0.0000711 * 5e11 * 0.20 + 0.0003733 * 5e11 * 0.10
             + 0.0000711 * 2e11 * 1.00 + 0.0002133 * 2e11 * 0.25
             + 0.0000160 * 2e11 * 0.10)
    assert np.isclose(full, np.sqrt(pan**2 + large**2))


def test_catastrophe_prior_year_claims_offset():
    """Prior-year claims net the large-accident exposure, floored at zero."""
    off = sv.catastrophe_scr(accident_death=1e12, prior_year_claims={"death": 1e11})
    # 1e12 x 1.5% = 1.5e10 < 1e11 -> second term floored to 0
    assert np.isclose(off, 0.0000711 * max(1e12 * 0.15 - 1e11, 0.0))


def test_required_capital_folds_catastrophe_table6():
    """Catastrophe folds into the insurance module via the table-6 correlation; the
    risk margin EXCLUDES it (handbook: insurance amount ex-catastrophe)."""
    mp, basis = _mp(), _basis()
    base = sv.required_capital(mp, basis, regime=sv.KICS)
    cat = 200_000.0
    got = sv.required_capital(mp, basis, regime=sv.KICS, catastrophe=cat)
    names = ["mortality", "longevity", "morbidity", "lapse", "expense"]
    c = np.array([base.sub_risk_capital[n] for n in names] + [cat])
    R = np.eye(6)
    R[:5, :5] = sv._KICS_CORRELATION
    cc = np.array([0.25, 0.0, 0.25, 0.25, 0.25])
    R[:5, 5] = R[5, :5] = cc
    assert np.isclose(got.insurance_scr, np.sqrt(c @ R @ c))
    assert got.insurance_scr > base.insurance_scr
    assert np.isclose(got.risk_margin, base.risk_margin)        # margin ex-catastrophe
    assert np.isclose(got.sub_risk_capital["catastrophe"], cat)


def test_catastrophe_ignored_when_regime_has_no_correlation():
    """Solvency II has no catastrophe_correlation -> a passed catastrophe is not
    folded (it would be a separate shock sub-risk there, deferred)."""
    mp, basis = _mp(), _basis()
    base = sv.required_capital(mp, basis, regime=sv.SII)
    got = sv.required_capital(mp, basis, regime=sv.SII, catastrophe=200_000.0)
    assert np.isclose(got.insurance_scr, base.insurance_scr)


def _mp_with_property():
    """A death + property-coverage policy (the property coverage runs as a
    MORBIDITY indemnity but is its own catastrophe/property category by code)."""
    return fcf.ModelPoints.single(
        45, 80_000.0, 240, benefits={"DEATH": 5e7, "PROP": 3e7},
        calculation_methods={"DEATH": CalculationMethod.DEATH,
                             "PROP": CalculationMethod.MORBIDITY})


def _basis_with_property():
    return fcf.Basis(mortality_annual=0.005, lapse_annual=0.02, discount_annual=0.03,
                     ra_confidence=0.75, mortality_cv=0.10, morbidity_cv=0.10,
                     coverages=(fcf.CoverageRate("DEATH", 0.005),
                                fcf.CoverageRate("PROP", 0.02)))


def test_property_subrisk_shock_and_fold():
    """Long-term property/other = +16% rate shock on the named codes, folded into
    the insurance module via the table-6 property row (vs base = 0,0,0,0,0.5)."""
    mp, basis = _mp_with_property(), _basis_with_property()
    base = sv.required_capital(mp, basis, regime=sv.KICS)
    got = sv.required_capital(mp, basis, regime=sv.KICS, property_codes=("PROP",))
    # property capital = max(0, delta BEL under a +16% PROP-rate shock)
    shocked = sv.scale_coverage_codes(("PROP",), 1.16).apply(mp, basis)
    from fastcashflow.engine import measure as _measure
    dbel = (float(_measure(*shocked, full=False).bel.sum())
            - float(_measure(mp, basis, full=False).bel.sum()))
    assert np.isclose(got.sub_risk_capital["property"], max(0.0, dbel))
    names = ["mortality", "longevity", "morbidity", "lapse", "expense"]
    c = np.array([got.sub_risk_capital[n] for n in names]
                 + [got.sub_risk_capital["property"]])
    R = np.eye(6)
    R[:5, :5] = sv._KICS_CORRELATION
    R[:5, 5] = R[5, :5] = [0.0, 0.0, 0.0, 0.0, 0.5]
    assert np.isclose(got.insurance_scr, np.sqrt(c @ R @ c))
    assert got.insurance_scr > base.insurance_scr


def test_property_and_catastrophe_cross_correlation():
    """Property and catastrophe both fold in, cross-correlated 0.25 (table 6); the
    margin includes property but excludes catastrophe."""
    mp, basis = _mp_with_property(), _basis_with_property()
    prop = sv.required_capital(mp, basis, regime=sv.KICS, property_codes=("PROP",))
    both = sv.required_capital(mp, basis, regime=sv.KICS, property_codes=("PROP",),
                               catastrophe=300_000.0)
    cap = both.sub_risk_capital
    names = ["mortality", "longevity", "morbidity", "lapse", "expense"]
    c = np.array([cap[n] for n in names] + [cap["property"], cap["catastrophe"]])
    R = np.eye(7)
    R[:5, :5] = sv._KICS_CORRELATION
    R[:5, 5] = R[5, :5] = [0.0, 0.0, 0.0, 0.0, 0.5]
    R[:5, 6] = R[6, :5] = [0.25, 0.0, 0.25, 0.25, 0.25]
    R[5, 6] = R[6, 5] = 0.25
    assert np.isclose(both.insurance_scr, np.sqrt(c @ R @ c))
    assert np.isclose(both.risk_margin, prop.risk_margin)   # margin: +property, -catastrophe


def test_property_ignored_for_solvency2():
    """Solvency II has no property_correlation -> property_codes are ignored."""
    mp, basis = _mp_with_property(), _basis_with_property()
    base = sv.required_capital(mp, basis, regime=sv.SII)
    got = sv.required_capital(mp, basis, regime=sv.SII, property_codes=("PROP",))
    assert np.isclose(got.insurance_scr, base.insurance_scr)
    assert "property" not in got.sub_risk_capital


# ---------------------------------------------------------------------------
# Solvency II life catastrophe (Delegated Regulation Art 143 / Art 136 row)
# ---------------------------------------------------------------------------

def test_sii_catastrophe_adds_first_year_mortality_only():
    """Art 143: +0.15pp absolute on the mortality rate over the next 12 months
    (duration 0) only; later policy years are unchanged."""
    base = _basis()
    _, b = sv.catastrophe_mortality(0.0015).apply(_mp(), base)
    g0 = (np.array([0]), np.array([40]), np.array([0]), np.array([0]), np.array([0]))
    g5 = (np.array([0]), np.array([40]), np.array([5]), np.array([0]), np.array([0]))
    assert np.allclose(b.mortality_annual(*g0), base.mortality_annual(*g0) + 0.0015)
    assert np.allclose(b.mortality_annual(*g5), base.mortality_annual(*g5))   # unchanged


def test_sii_catastrophe_is_seventh_sub_risk():
    """Catastrophe is the 7th Solvency II life sub-risk; the correlation row matches
    Article 136 (cat vs mortality/longevity/disability/expense/revision/lapse =
    0.25/0/0.25/0.25/0/0.25) and a death book carries a positive cat capital."""
    spec = sv.SII
    assert [sr.name for sr in spec.sub_risks][-1] == "catastrophe"
    assert spec.correlation.shape == (7, 7)
    np.testing.assert_allclose(spec.correlation[6],
                               [0.25, 0.0, 0.25, 0.25, 0.0, 0.25, 1.0])
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=sv.SII)
    assert res.sub_risk_capital["catastrophe"] > 0.0         # death book -> cat is a loss


# ---------------------------------------------------------------------------
# Dynamic lapse -- the lapse<->rate coupling (the dynamic asset engine, Phase D)
# ---------------------------------------------------------------------------

def test_dynamic_lapse_multiplier_hand_calc():
    """1 + sensitivity*rate_shock, floored at 0."""
    assert np.isclose(sv.dynamic_lapse_multiplier(0.01, 5.0), 1.05)
    assert np.isclose(sv.dynamic_lapse_multiplier(-0.01, 5.0), 0.95)   # rate fall -> less lapse
    assert sv.dynamic_lapse_multiplier(0.01, 0.0) == 1.0               # no coupling
    assert sv.dynamic_lapse_multiplier(-0.5, 5.0) == 0.0               # floored (1 - 2.5 < 0)


def test_interest_with_dynamic_lapse_couples_curve_and_lapse():
    """The coupled stress shifts the curve by +shift and scales lapse by the
    dynamic multiplier (clamped to 1.0 like every rate)."""
    mp, basis = _mp(), _basis()
    shift = 0.01
    _, shocked = sv.interest_with_dynamic_lapse(shift, 5.0).apply(mp, basis)
    grid = (np.array([0]), np.array([40]), np.array([24]), np.array([0]), np.array([24]))
    assert np.allclose(shocked.lapse_annual(*grid),
                       np.minimum(basis.lapse_annual(*grid) * 1.05, 1.0))
    assert np.allclose(np.asarray(shocked.discount_annual),
                       np.asarray(basis.discount_annual, float) + shift)


def test_dynamic_lapse_zero_sensitivity_is_pure_rate():
    """sensitivity=0 leaves lapse untouched -- the coupled stress is then exactly a
    parallel rate shift (the same measured BEL)."""
    from dataclasses import replace
    mp, basis = _mp(), _basis()
    shift = 0.01
    _, coupled = sv.interest_with_dynamic_lapse(shift, 0.0).apply(mp, basis)
    pure = replace(basis, discount_annual=np.asarray(basis.discount_annual, float) + shift)
    assert np.isclose(measure(mp, coupled, full=False).bel.sum(),
                      measure(mp, pure, full=False).bel.sum())


def test_dynamic_lapse_moves_bel_vs_uncoupled():
    """With a positive lapse and sensitivity the coupling moves BEL away from the
    rate-only stress -- the lapse<->rate correlation has a measurable effect."""
    mp, basis = _mp(), _basis()
    shift = 0.01
    uncoupled = measure(mp, sv.interest_with_dynamic_lapse(shift, 0.0).apply(mp, basis)[1],
                        full=False).bel.sum()
    coupled = measure(mp, sv.interest_with_dynamic_lapse(shift, 8.0).apply(mp, basis)[1],
                      full=False).bel.sum()
    assert not np.isclose(coupled, uncoupled)


# ---------------------------------------------------------------------------
# VFA (variable) life sub-risk capital -- required_capital(measure_fn=measure_vfa)
# ---------------------------------------------------------------------------

def _vfa_life_regime():
    return sv.RegimeSpec(
        name="vfa-life",
        sub_risks=(
            sv.SubRisk("mortality", (sv.scale_mortality(1.15),), "single"),
            sv.SubRisk("lapse", (sv.scale_lapse(1.5), sv.scale_lapse(0.5)),
                       "worst_of"),
            sv.SubRisk("expense", (sv.scale_expense(1.10, 0.01),), "single"),
        ),
        correlation=np.eye(3),
        risk_margin_method="percentile", risk_margin_factor=0.0,
    )


def test_vfa_required_capital_routes_through_the_vfa_measure():
    """vfa_required_capital prices the regime's life sub-risks on the VFA NET BEL:
    base_bel is the VFA measure, and the lapse sub-risk reconciles to the worst-of
    the up/down re-measures -- lapse-down holds policies on the valuable GMAB, so
    the lapse capital is positive."""
    basis = make_death_basis(
        mortality_q=0.002, lapse_q=0.004, discount_annual=0.03, ra_confidence=0.75,
        expense_cv=0.10, investment_return=0.0, fund_fee=0.02,
        expense_items=(fcf.ExpenseItem("maintenance", "gamma_fixed", 5.0),))
    mp = fcf.ModelPoints.single(40, 0.0, 120, account_value=1000.0,
                                minimum_accumulation_benefit=1300.0,
                                calculation_methods=PATTERNS)
    regime = _vfa_life_regime()
    scr = sv.vfa_required_capital(mp, basis, regime=regime)

    base = float(fcf.vfa.measure(mp, basis, full=False).bel.sum())
    assert np.isclose(scr.base_bel, base)            # routed through measure_vfa

    lapse_sr = next(s for s in regime.sub_risks if s.name == "lapse")
    deltas = [float(fcf.vfa.measure(*v.apply(mp, basis), full=False).bel.sum()) - base
              for v in lapse_sr.variants]
    assert np.isclose(scr.sub_risk_capital["lapse"], max(0.0, max(deltas)))
    assert scr.sub_risk_capital["lapse"] > 0.0       # lapse-down holds the GMAB
    assert scr.total_scr >= scr.sub_risk_capital["lapse"]   # module >= one sub-risk


def test_required_capital_measure_fn_default_is_unchanged():
    """The measure_fn refactor leaves the GMM path byte-identical (default arg)."""
    mp, basis = _mp(), _basis()
    regime = _dummy(corr=0.25)
    a = sv.required_capital(mp, basis, regime=regime)
    b = sv.required_capital(mp, basis, regime=regime, measure_fn=measure)
    assert np.isclose(a.total_scr, b.total_scr)
    assert np.isclose(a.base_bel, b.base_bel)


def test_vfa_equity_scr_hand_calc():
    """The VFA net BEL's equity capital is the rise in the net BEL when the account
    value is shocked down -- both the GMAB going in-the-money (guarantee cost up) and
    the variable fee income falling. Reconciles to the re-measure, is monotone in the
    shock, and -- with no fee and an out-of-the-money guarantee -- is zero."""
    basis = make_death_basis(mortality_q=0.002, lapse_q=0.004, discount_annual=0.03,
                             ra_confidence=0.75, investment_return=0.0, fund_fee=0.02)
    mp = fcf.ModelPoints.single(40, 0.0, 120, account_value=1000.0,
                                minimum_accumulation_benefit=1100.0,
                                calculation_methods=PATTERNS)
    base = float(fcf.vfa.measure(mp, basis, full=False).bel.sum())

    cap = sv.vfa_equity_scr(mp, basis, equity_shock=0.39)
    from dataclasses import replace
    mp_s = replace(mp, account_value=mp.account_value * (1 - 0.39))
    expected = max(0.0, float(fcf.vfa.measure(mp_s, basis, full=False).bel.sum()) - base)
    assert np.isclose(cap, expected)
    assert cap > 0.0                                         # AV drop -> ITM + less fee
    assert np.isclose(sv.vfa_equity_scr(mp, basis, equity_shock=0.0), 0.0)
    # Monotone in the shock.
    assert sv.vfa_equity_scr(mp, basis, equity_shock=0.50) > cap

    # No fee and an out-of-the-money guarantee (still OTM after the shock): neither
    # leg moves the net BEL, so the equity capital is zero.
    nofee = make_death_basis(mortality_q=0.002, lapse_q=0.004, discount_annual=0.03,
                             ra_confidence=0.75, investment_return=0.0, fund_fee=0.0)
    otm = fcf.ModelPoints.single(40, 0.0, 120, account_value=1000.0,
                                 minimum_accumulation_benefit=300.0,
                                 calculation_methods=PATTERNS)
    assert np.isclose(sv.vfa_equity_scr(otm, nofee, equity_shock=0.39), 0.0)


def test_vfa_interest_scr_hand_calc():
    """The VFA interest capital is the worst of re-measuring the net BEL under a
    parallel +/- shift to the underlying-items return. A return fall pushes the
    guarantee in-the-money (the binding direction), so the capital reconciles to the
    down-shift re-measure, is positive, and grows with the shift."""
    basis = make_death_basis(mortality_q=0.002, lapse_q=0.004, discount_annual=0.03,
                             ra_confidence=0.75, investment_return=0.04, fund_fee=0.02)
    mp = fcf.ModelPoints.single(40, 0.0, 120, account_value=1000.0,
                                minimum_accumulation_benefit=1200.0,
                                calculation_methods=PATTERNS)
    from dataclasses import replace
    base = float(fcf.vfa.measure(mp, basis, full=False).bel.sum())
    up = float(fcf.vfa.measure(mp, replace(basis, investment_return=0.05),
                               full=False).bel.sum())
    dn = float(fcf.vfa.measure(mp, replace(basis, investment_return=0.03),
                               full=False).bel.sum())
    cap = sv.vfa_interest_scr(mp, basis, shift=0.01)
    assert np.isclose(cap, max(0.0, up - base, dn - base))
    assert np.isclose(cap, dn - base)                        # the fall binds
    assert cap > 0.0
    assert sv.vfa_interest_scr(mp, basis, shift=0.02) > cap  # grows with the shift


# ---------------------------------------------------------------------------
# scale_state_rate -- a named state-machine transition-rate shock (Phase 1 of
# the Solvency II disability sub-risk: Art. 153 recovery / inception shocks).
# ---------------------------------------------------------------------------

def _di_recovery_model() -> StateModel:
    """A two-state disability-income model: active -> disabled (inception) and
    disabled -> active (recovery)."""
    return StateModel(states=(
        State("active", pays_premium=True, transitions=(
            Transition("mortality"),
            Transition("waiver_incidence", to="disabled"),
            Transition("lapse"))),
        State("disabled", pays_periodic_benefit=True, sojourn_tracking_months=12,
              transitions=(
                  Transition("mortality"),
                  Transition("disability_recovery", to="active",
                             sojourn_dependent=True))),
    ), seating=(0, 1, 1))


def _di_recovery_basis(recovery_monthly: float) -> fcf.Basis:
    return fcf.Basis(
        mortality_annual=lambda s, a, d: np.full(d.shape, annual_from_monthly(0.001)),
        lapse_annual=lambda s, a, d: np.full(d.shape, 0.0),
        waiver_incidence_annual=lambda s, a, d: np.full(d.shape, 0.0),
        disability_recovery_annual=lambda s, a, p, sd: np.full(
            sd.shape, annual_from_monthly(recovery_monthly), dtype=float),
        discount_annual=0.0, ra_confidence=0.5, mortality_cv=0.0, disability_cv=0.0,
        state_model=_di_recovery_model(),
        coverages=(fcf.CoverageRate(
            "DEATH", lambda s, a, d: np.full(d.shape, annual_from_monthly(0.001))),),
    )


def _di_seated_disabled(term_months: int) -> fcf.ModelPoints:
    return fcf.ModelPoints(
        issue_age=np.array([45], dtype=np.int64),
        benefits={"DEATH": np.array([0.0])},
        premium=np.array([0.0]),
        term_months=np.array([term_months], dtype=np.int64),
        disability_income=np.array([1_000_000.0]),
        state=np.array([1], dtype=np.int64),       # seated on disabled
        calculation_methods=PATTERNS,
    )


def test_scale_state_rate_recovery_hand_calc():
    """A -20% recovery shock on a 2-month seated-disabled DI contract. Monthly
    mortality q = 0.001, base monthly recovery r = 0.05, disability income
    DI = 1,000,000, no discount.

      t = 0: disabled occ = 1.0  -> income = 1.0 * DI
      step 0->1 (ordered decrements: mortality then recovery):
             disabled survivor = (1 - q)(1 - r)
      t = 1: income = (1 - q)(1 - r) * DI

      BEL = DI * [1 + (1 - q)(1 - r)]

    The shock scales the ANNUAL recovery by 0.80; the engine's monthly recovery
    becomes r' = 1 - (1 - 0.80 * A_r)^(1/12), where A_r is the base annual rate.
    Lower recovery -> the disabled stay on claim -> BEL rises."""
    q, r0 = 0.001, 0.05
    DI = 1_000_000.0
    mp = _di_seated_disabled(2)
    basis = _di_recovery_basis(r0)

    base_bel = DI * (1.0 + (1.0 - q) * (1.0 - r0))
    assert np.isclose(measure(mp, basis, full=False).bel[0], base_bel), \
        measure(mp, basis, full=False).bel[0]

    _, shocked = sv.scale_state_rate("disability_recovery", 0.80).apply(mp, basis)
    a_r = annual_from_monthly(r0)
    r1 = 1.0 - (1.0 - 0.80 * a_r) ** (1.0 / 12.0)        # engine monthly from shocked annual
    shock_bel = DI * (1.0 + (1.0 - q) * (1.0 - r1))
    assert np.isclose(measure(mp, shocked, full=False).bel[0], shock_bel), \
        measure(mp, shocked, full=False).bel[0]

    # recovery down -> liability up; the shock is a positive disability strain
    assert shock_bel > base_bel
    assert np.isclose(shock_bel - base_bel, DI * (1.0 - q) * (r0 - r1))


def test_scale_state_rate_noop_when_rate_absent():
    """A product without the named transition (the field is None) is unchanged --
    the shock is a no-op, so a plain death book's BEL does not move."""
    mp = _mp(term=24)
    basis = _basis()
    assert basis.disability_recovery_annual is None
    _, shocked = sv.scale_state_rate("disability_recovery", 0.80).apply(mp, basis)
    assert np.isclose(measure(mp, shocked, full=False).bel.sum(),
                      measure(mp, basis, full=False).bel.sum())


def test_scale_state_rate_rejects_unknown_rate():
    with pytest.raises(ValueError, match="unknown state-machine rate"):
        sv.scale_state_rate("not_a_rate", 0.8)


# ---------------------------------------------------------------------------
# scale_coverages_first_year -- the +35% year-1 / +25%-thereafter duration split
# (Phase 2 of the Solvency II disability inception shock, Art. 153).
# ---------------------------------------------------------------------------

_MORB = CalculationMethod.MORBIDITY


def _morbidity_mp_basis(rate_monthly: float = 0.01):
    """A single 2-year morbidity contract with a flat claim rate, no decrement
    and no discount, so the claim PV is a clean per-month sum."""
    rate = lambda s, a, d: np.full(np.asarray(d).shape, annual_from_monthly(rate_monthly))
    basis = make_death_basis(mortality_q=0.0, lapse_q=0.0, discount_annual=0.0,
                             mortality_cv=0.0,
                             coverages=(fcf.CoverageRate("INP", rate),))
    mp = fcf.ModelPoints.single(40, 0.0, 24, benefits={"INP": 100_000.0},
                                calculation_methods={"INP": _MORB})
    return mp, basis


def test_scale_coverages_first_year_grid_split():
    """The first policy year (duration == 0) is scaled by the first-year factor,
    later years by the steady factor -- evaluated on the rate grid."""
    mp, basis = _morbidity_mp_basis()
    _, shocked = sv.scale_coverages_first_year({_MORB: 1.35}, {_MORB: 1.25}).apply(mp, basis)
    base_cov = {r.code: r for r in basis.coverages}["INP"]
    shock_cov = {r.code: r for r in shocked.coverages}["INP"]
    for dur, factor in [(0, 1.35), (1, 1.25), (2, 1.25)]:
        g = (np.array([0]), np.array([40]), np.array([dur]), np.array([0]), np.array([0]))
        assert np.allclose(shock_cov.rate(*g), base_cov.rate(*g) * factor), dur


def test_scale_coverages_first_year_clamps_to_one():
    """A base rate high enough that the first-year factor would exceed 1.0 is
    clamped (rates are probabilities)."""
    mp, basis = _morbidity_mp_basis(rate_monthly=0.0)
    high = lambda s, a, d: np.full(np.asarray(d).shape, 0.80)     # 0.80 annual, x1.35 = 1.08
    basis = make_death_basis(mortality_q=0.0, lapse_q=0.0, discount_annual=0.0,
                             mortality_cv=0.0, coverages=(fcf.CoverageRate("INP", high),))
    _, shocked = sv.scale_coverages_first_year({_MORB: 1.35}, {_MORB: 1.25}).apply(mp, basis)
    shock_cov = {r.code: r for r in shocked.coverages}["INP"]
    g0 = (np.array([0]), np.array([40]), np.array([0]), np.array([0]), np.array([0]))
    assert np.allclose(shock_cov.rate(*g0), 1.0)                  # 1.08 clamped to 1.0


def test_scale_coverages_first_year_rejects_mismatched_methods():
    with pytest.raises(ValueError, match="same methods"):
        sv.scale_coverages_first_year({_MORB: 1.35}, {CalculationMethod.DIAGNOSIS: 1.25})


def test_scale_coverages_first_year_bel_exceeds_uniform():
    """The first-year split must add the +35% year-1 bump ON TOP of the steady
    +25%: its BEL exceeds the uniform +25% shock, which exceeds the base."""
    mp, basis = _morbidity_mp_basis()
    base = float(measure(mp, basis, full=False).bel.sum())
    _, split = sv.scale_coverages_first_year({_MORB: 1.35}, {_MORB: 1.25}).apply(mp, basis)
    _, uniform = sv.scale_coverages({_MORB: 1.25}).apply(mp, basis)
    bel_split = float(measure(mp, split, full=False).bel.sum())
    bel_uniform = float(measure(mp, uniform, full=False).bel.sum())
    assert bel_split > bel_uniform > base

    # the year-0 half carries the +35%, the year-1 half the +25% -- check the
    # two-segment composition (no decrement, no discount, so PV is additive).
    m0 = 1.0 - (1.0 - 1.35 * annual_from_monthly(0.01)) ** (1.0 / 12.0)
    m1 = 1.0 - (1.0 - 1.25 * annual_from_monthly(0.01)) ** (1.0 / 12.0)
    expected = 12 * m0 * 100_000.0 + 12 * m1 * 100_000.0
    assert np.isclose(bel_split, expected)


# ---------------------------------------------------------------------------
# SII disability sub-risk (Art 153) -- the composed inception-up / recovery-down
# shock wired into SII (Phase 3).
# ---------------------------------------------------------------------------

def _disability_only_regime():
    """A one-sub-risk regime carrying just the SII disability shock, so
    sub_risk_capital['disability'] is the standalone Art 153 capital."""
    dis = next(sr for sr in sv.SII.sub_risks if sr.name == "disability")
    return sv.RegimeSpec(name="dis", sub_risks=(dis,),
                         correlation=np.array([[1.0]]), risk_margin_method="percentile")


def test_sii_disability_subrisk_includes_recovery():
    """On a DI book whose only disability exposure is the recovery edge (a flat
    death coverage, zero inception), the disability capital is exactly the -20%
    recovery shock's delta BEL -- proving recovery is wired into the sub-risk."""
    mp, basis = _di_seated_disabled(24), _di_recovery_basis(0.05)
    base = float(measure(mp, basis, full=False).bel.sum())
    _, rec = sv.scale_state_rate("disability_recovery", 0.80).apply(mp, basis)
    expected = max(0.0, float(measure(mp, rec, full=False).bel.sum()) - base)
    cap = sv.required_capital(
        mp, basis, regime=_disability_only_regime()).sub_risk_capital["disability"]
    assert expected > 0.0                          # recovery down raises the liability
    assert np.isclose(cap, expected)


def test_sii_disability_subrisk_first_year_exceeds_uniform():
    """On a morbidity book the Art 153 capital exceeds the old uniform +25%
    capital -- the +35% first-year inception bump adds strain."""
    mp, basis = _morbidity_mp_basis()
    base = float(measure(mp, basis, full=False).bel.sum())
    _, uni = sv.scale_coverages(
        {_MORB: 1.25, CalculationMethod.DIAGNOSIS: 1.25}).apply(mp, basis)
    cap_old = max(0.0, float(measure(mp, uni, full=False).bel.sum()) - base)
    cap_new = sv.required_capital(
        mp, basis, regime=_disability_only_regime()).sub_risk_capital["disability"]
    assert cap_old > 0.0
    assert cap_new > cap_old


def test_sii_disability_subrisk_zero_for_death_book():
    """A plain death book has no disability / morbidity exposure, so the Art 153
    shock leaves the BEL unchanged and the disability capital is zero (the
    doc-exec term-life books are unaffected by the recalibration)."""
    mp, basis = _mp(term=24), _basis()
    cap = sv.required_capital(
        mp, basis, regime=_disability_only_regime()).sub_risk_capital["disability"]
    assert cap == 0.0
