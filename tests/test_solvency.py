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
