"""ALM foundations -- liability duration / DV01 / key-rate, hand-calc anchors."""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import alm, pricing
from fastcashflow.engine import measure

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


def test_key_rate_durations_sum_to_parallel():
    """The per-year key-rate DV01s decompose the parallel DV01."""
    mp, basis = _mp(), _basis()
    krd = alm.key_rate_durations(mp, basis)
    assert np.isclose(krd.sum(), alm.liability_dv01(mp, basis), rtol=1e-3)
    assert krd.shape[0] == 10                          # 120-month term -> 10 years


def test_bond_textbook_duration():
    """A 5% annual-coupon 10-year bond at a flat 5% is par; Macaulay / Modified /
    DV01 match an independent closed-form computation."""
    bond = alm.Bond(face=100.0, coupon_rate=0.05, maturity_years=10, frequency=1)
    d = alm.bond_duration(bond, 0.05)
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
    bond = alm.Bond(80.0, 0.04, 7, frequency=2)
    eff = -(alm.bond_value(bond, 0.05 + 1e-4)
            - alm.bond_value(bond, 0.05 - 1e-4)) / (2.0 * 1e-4) * 1e-4
    assert np.isclose(alm.bond_duration(bond, 0.05).dv01, eff, rtol=1e-4)


def test_matched_book_gap_is_zero():
    """A bond book sized to the liability DV01 immunises the parallel gap."""
    mp, basis = _mp(), _basis()
    liab = alm.liability_dv01(mp, basis)
    per_face = alm.bond_duration(alm.Bond(100.0, 0.03, 10, 1), 0.03).dv01
    matched = alm.Bond(100.0 * liab / per_face, 0.03, 10, 1)
    g = alm.alm_gap(alm.bond_duration(matched, 0.03).dv01, liab)
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
