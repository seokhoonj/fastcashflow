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

from conftest import make_death_basis, PATTERNS


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


def test_mass_lapse_is_count_haircut():
    """Mass lapse = max(BEL(count*0.70) - BEL(count), 0). The count haircut omits
    the t=0 surrender value paid to the lapsing policies (a documented v1
    simplification that can understate the mass-lapse capital)."""
    from dataclasses import replace
    mp, basis = _mp(), _basis()
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
    spec = sv.SOLVENCY2
    names = [sr.name for sr in spec.sub_risks]
    assert names == ["mortality", "longevity", "disability", "expense", "revision", "lapse"]
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
    # cost-of-capital risk margin, EIOPA interest curves present
    assert spec.risk_margin_method == "cost_of_capital" and spec.risk_margin_coc_rate == 0.06
    assert spec.interest_curves is not None and len(spec.interest_curves) == 2


def test_sii_runs_end_to_end():
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=sv.SOLVENCY2)
    assert set(res.sub_risk_capital) == {
        "mortality", "longevity", "disability", "expense", "revision", "lapse"}
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
    res = sv.required_capital(mp, basis, regime=sv.SOLVENCY2)
    assert res.interest_capital > 0.0           # interest is material here
    assert np.isclose(res.scr_path[0], res.insurance_scr)
    assert np.all(res.scr_path >= 0.0)


def test_solvency_ratio():
    mp, basis = _mp(), _basis()
    res = sv.required_capital(mp, basis, regime=sv.SOLVENCY2)
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
    base = sv.required_capital(mp, basis, regime=sv.SOLVENCY2)
    got = sv.required_capital(mp, basis, regime=sv.SOLVENCY2, catastrophe=200_000.0)
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
    base = sv.required_capital(mp, basis, regime=sv.SOLVENCY2)
    got = sv.required_capital(mp, basis, regime=sv.SOLVENCY2, property_codes=("PROP",))
    assert np.isclose(got.insurance_scr, base.insurance_scr)
    assert "property" not in got.sub_risk_capital
