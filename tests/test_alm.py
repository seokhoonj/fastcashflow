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
