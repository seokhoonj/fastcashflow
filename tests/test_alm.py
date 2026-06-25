"""ALM foundations -- liability duration / DV01 / key-rate, hand-calc anchors."""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import alm, assets, pricing
from fastcashflow._measurement.gmm import measure

from conftest import make_death_basis, PATTERNS


def _basis(rate=0.03):
    return make_death_basis(mortality_q=0.001, lapse_q=0.0, discount_annual=rate,
                            mortality_cv=0.0)


def _mp(premium=50_000.0, term=120):
    return fcf.ModelPoints.single(40, premium, term, benefits={"DEATH": 1e8},
                                  calculation_methods=PATTERNS)


def test_net_cashflows_reconstruct_bel():
    """flow_bom . df_bom + flow_mid . df_mid == BEL (the assembly is exact)."""
    mp, basis = _mp(), _basis()
    m = measure(mp, basis, full=True)
    fb, fm = alm.net_liability_cashflows(m)
    recon = float(fb @ m.discount_factor_bom + fm @ m.discount_factor_mid)
    assert np.isclose(recon, float(m.bel.sum()), rtol=1e-9)


def test_net_cashflows_needs_full():
    mp, basis = _mp(), _basis()
    with pytest.raises(ValueError, match="full=True"):
        alm.net_liability_cashflows(measure(mp, basis, full=False))


def _ul_basis():
    """A minimal universal-life basis -- account-backed DEATH coverage."""
    from fastcashflow import Basis, CoverageRate
    return Basis(
        mortality_annual=lambda s, a, d: np.full(s.shape, 0.001),
        lapse_annual=lambda s, a, d: np.full(s.shape, 0.0),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.1,
        investment_return=0.04,
        coverages=(CoverageRate("DEATH", lambda s, a, d: np.full(s.shape, 0.01),
                                funds_from_account=True, pays_account_balance=True),))


def _ul_mp():
    from fastcashflow import CalculationMethod
    return fcf.ModelPoints.single(40, 0.0, 120, benefits={"DEATH": 1e5},
                                  account_value=1_000.0, minimum_death_benefit=1e5,
                                  count=100.0,
                                  calculation_methods={"DEATH": CalculationMethod.DEATH})


def test_net_cashflows_account_book_redirects_to_vfa():
    """An account-value (UL) book is routed to the entity net-liability ladder --
    the error names vfa_net_liability_cashflows / vfa_cashflow_gap, not a bare
    'unsupported'."""
    m = fcf.vfa.measure(_ul_mp(), _ul_basis())
    with pytest.raises(NotImplementedError, match="vfa_net_liability_cashflows"):
        alm.net_liability_cashflows(m)


def test_liability_interest_metrics_account_book_redirect_to_vfa():
    """The discount-bump interest metrics do not apply to an account book (it
    discounts at the underlying-items return); the error routes to the VFA
    interest sub-risk rather than raising the engine's mixed-book message."""
    mp, basis = _ul_mp(), _ul_basis()
    for fn in (alm.liability_dv01, alm.liability_duration, alm.key_rate_dv01s):
        with pytest.raises(NotImplementedError, match="vfa.liability_duration"):
            fn(mp, basis)


def test_alm_namespaces_mirror_module():
    """ALM is exposed through the model namespaces like measure / trace: the GMM
    metrics under fcf.gmm.*, the VFA metrics under fcf.vfa.* (the symmetric home
    for fcf.vfa.measure), each the same object as its alm.* implementation."""
    assert fcf.gmm.liability_duration is alm.liability_duration
    assert fcf.gmm.liability_dv01 is alm.liability_dv01
    assert fcf.gmm.key_rate_dv01s is alm.key_rate_dv01s
    assert fcf.gmm.net_liability_cashflows is alm.net_liability_cashflows
    assert fcf.vfa.liability_duration is alm.vfa_liability_duration
    assert fcf.vfa.liability_dv01 is alm.vfa_liability_dv01
    assert fcf.vfa.net_liability_cashflows is alm.vfa_net_liability_cashflows


def test_vfa_namespace_exposes_solvency_and_asset_tools():
    """The VFA-specific solvency / asset-liability tools are reachable under the
    fcf.vfa namespace (the only home -- flat fcf.vfa_* aliases were removed)."""
    assert callable(fcf.vfa.required_capital)
    assert callable(fcf.vfa.equity_scr)
    assert callable(fcf.vfa.interest_scr)
    assert callable(fcf.vfa.cashflow_gap)
    assert callable(fcf.vfa.assess)
    assert callable(fcf.vfa.interaction_loss)


def _ul_guaranteed_mp():
    """A UL book with a 6% crediting guarantee above the 4% underlying return --
    the floor binds, so the liability is sensitive to the underlying return (an
    interest gap the VFA duration captures)."""
    from fastcashflow import CalculationMethod
    return fcf.ModelPoints.single(40, 0.0, 120, benefits={"DEATH": 1e5},
                                  account_value=1e4, minimum_death_benefit=1e5,
                                  minimum_crediting_rate=0.06, count=100.0,
                                  calculation_methods={"DEATH": CalculationMethod.DEATH})


def test_vfa_liability_duration_responds_to_crediting_guarantee():
    """fcf.vfa.liability_duration differences the VFA BEL against the underlying-
    items return. With a binding crediting guarantee the BEL moves, so the dv01 is
    non-zero and finite; the duration's dv01 matches vfa_liability_dv01, and the
    modified duration is dv01 / (|pv| * 1bp)."""
    mp, basis = _ul_guaranteed_mp(), _ul_basis()
    dur = fcf.vfa.liability_duration(mp, basis)
    dv01 = fcf.vfa.liability_dv01(mp, basis)
    assert np.isfinite(dur.pv) and abs(dur.dv01) > 1.0          # the guarantee bites
    assert np.isclose(dur.dv01, dv01)                          # the two agree
    assert np.isclose(dur.modified, dur.dv01 / (abs(dur.pv) * 1e-4))
    assert np.isnan(dur.macaulay)                              # mixed-sign stream


def test_liability_dv01_sign_and_finite():
    """A positive-reserve liability falls in value when rates rise, so DV01 > 0."""
    mp, basis = _mp(), _basis()
    assert float(measure(mp, basis, full=False).bel.sum()) > 0.0   # positive BEL
    dv01 = alm.liability_dv01(mp, basis)
    assert dv01 > 0.0 and np.isfinite(dv01)


def test_liability_duration_result():
    mp, basis = _mp(), _basis()
    d = alm.liability_duration(mp, basis)
    assert d.pv > 0.0 and d.dv01 > 0.0
    assert np.isfinite(d.modified) and d.modified > 0.0
    assert np.isnan(d.macaulay)                       # mixed-sign liability stream
    # modified duration ties DV01 and PV: dv01 == modified * |pv| * 1bp
    assert np.isclose(d.dv01, d.modified * abs(d.pv) * 1e-4)


def test_key_rate_dv01s_sum_to_parallel():
    """The per-year key-rate DV01s decompose the parallel DV01."""
    mp, basis = _mp(), _basis()
    krd = alm.key_rate_dv01s(mp, basis)
    assert np.isclose(krd.sum(), alm.liability_dv01(mp, basis), rtol=1e-3)
    assert krd.shape[0] == 10                          # 120-month term -> 10 years


def test_bond_textbook_duration():
    """A 5% annual-coupon 10-year bond at a flat 5% is par; Macaulay / Modified /
    DV01 match an independent closed-form computation."""
    bond = assets.Bond(face=100.0, coupon_rate=0.05, maturity_years=10, frequency=1)
    d = assets.bond_duration(bond, 0.05)
    assert np.isclose(d.pv, 100.0)                    # par
    t = np.arange(1, 11, dtype=float)
    cf = np.full(10, 5.0); cf[-1] += 100.0
    v = 1.05 ** (-t)
    mac = float((t * cf * v).sum() / (cf * v).sum())
    assert np.isclose(d.macaulay, mac)
    assert np.isclose(d.modified, mac / 1.05)
    assert np.isclose(d.dv01, d.modified * d.pv * 1e-4)


def test_bond_effective_matches_analytic():
    """Re-pricing the bond at +/-1bp (effective DV01) matches the analytic DV01."""
    bond = assets.Bond(80.0, 0.04, 7, frequency=2)
    eff = -(assets.bond_value(bond, 0.05 + 1e-4)
            - assets.bond_value(bond, 0.05 - 1e-4)) / (2.0 * 1e-4) * 1e-4
    assert np.isclose(assets.bond_duration(bond, 0.05).dv01, eff, rtol=1e-4)


def test_bond_convexity_matches_effective():
    """The analytic yield convexity equals the second central difference of the
    bond value under a parallel yield shift."""
    bond = assets.Bond(face=100.0, coupon_rate=0.05, maturity_years=10, frequency=1)
    d = assets.bond_duration(bond, 0.04)
    h = 1e-3
    pv = assets.bond_value(bond, 0.04)
    eff = (assets.bond_value(bond, 0.04 + h) + assets.bond_value(bond, 0.04 - h)
           - 2.0 * pv) / (pv * h * h)
    assert np.isclose(d.convexity, eff, rtol=1e-4)
    assert d.convexity > 0.0                          # a bullet bond is convex


def test_bond_second_order_price_move():
    """Duration + convexity predict a finite rate move better than duration alone:
    dPV/PV ~ -modified*dy + 0.5*convexity*dy^2."""
    bond = assets.Bond(face=100.0, coupon_rate=0.05, maturity_years=10, frequency=1)
    d = assets.bond_duration(bond, 0.05)
    dy = 0.01
    actual = assets.bond_value(bond, 0.05 + dy) / d.pv - 1.0
    linear = -d.modified * dy
    quad = -d.modified * dy + 0.5 * d.convexity * dy * dy
    assert abs(quad - actual) < abs(linear - actual)  # convexity tightens the fit


def test_liability_convexity_finite_and_guarded():
    """A positive-BEL liability has a finite effective convexity; at break-even the
    near-zero BEL guards it (and modified) to nan."""
    mp, basis = _mp(), _basis()
    d = alm.liability_duration(mp, basis)
    assert np.isfinite(d.convexity)
    mp0 = fcf.ModelPoints.single(40, 0.0, 120, benefits={"DEATH": 1e8},
                                 calculation_methods=PATTERNS)
    net = pricing.solve_premium(mp0, basis, break_even=True)[0]
    d0 = alm.liability_duration(replace(mp0, premium=np.full(1, net)), basis)
    assert np.isnan(d0.convexity) and np.isnan(d0.modified)


def test_duration_gap_immunises_surplus():
    """duration_gap = D_A - (L/A) D_L; choosing D_A = (L/A) D_L zeroes the gap and
    the surplus DV01."""
    A, L, D_L = 1200.0, 1000.0, 8.0
    D_A = (L / A) * D_L                                # immunising asset duration
    g = alm.duration_gap(D_A, A, D_L, L)
    assert np.isclose(g["leverage"], L / A)
    assert np.isclose(g["duration_gap"], 0.0)
    assert np.isclose(g["surplus_dv01"], 0.0)
    # a longer asset duration opens a positive gap (surplus falls when rates rise)
    assert alm.duration_gap(D_A + 1.0, A, D_L, L)["surplus_dv01"] > 0.0


def test_duration_gap_matches_dv01_gap():
    """surplus_dv01 equals the alm_gap dv01_gap when durations and values are
    mutually consistent (dv01 = modified * value * 1bp)."""
    A, L, D_A, D_L = 1200.0, 1000.0, 6.0, 9.0
    asset_dv01 = D_A * A * 1e-4
    liab_dv01 = D_L * L * 1e-4
    g = alm.duration_gap(D_A, A, D_L, L)
    assert np.isclose(g["surplus_dv01"], alm.alm_gap(asset_dv01, liab_dv01)["dv01_gap"])


def test_matched_book_gap_is_zero():
    """A bond book sized to the liability DV01 immunises the parallel gap."""
    mp, basis = _mp(), _basis()
    liab = alm.liability_dv01(mp, basis)
    per_face = assets.bond_duration(assets.Bond(100.0, 0.03, 10, 1), 0.03).dv01
    matched = assets.Bond(100.0 * liab / per_face, 0.03, 10, 1)
    g = alm.alm_gap(assets.bond_duration(matched, 0.03).dv01, liab)
    assert np.isclose(g["dv01_gap"], 0.0, atol=abs(liab) * 1e-6)


def test_modified_guard_near_zero_bel():
    """At break-even the BEL is ~ 0, so the modified-duration ratio is guarded to
    nan while the DV01 stays finite."""
    mp0 = fcf.ModelPoints.single(40, 0.0, 120, benefits={"DEATH": 1e8},
                                 calculation_methods=PATTERNS)
    basis = _basis()
    net = pricing.solve_premium(mp0, basis, break_even=True)[0]
    mp = replace(mp0, premium=np.full(1, net))
    d = alm.liability_duration(mp, basis)
    assert abs(d.pv) <= 1.0
    assert np.isnan(d.modified) and np.isfinite(d.dv01)
