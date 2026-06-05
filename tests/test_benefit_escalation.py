"""Escalating benefits (체증형).

Two orthogonal mechanisms:
* ``Basis.annuity_factor_annual`` -- the survival-benefit twin of
  ``premium_factor_annual``: a per-MP year-grid factor on ``annuity_payment``
  for an escalating annuity (체증형 연금, e.g. 5%/yr).
* per-coverage step (``coverage_step_month`` / ``coverage_step_factor``) -- a
  benefit step-up at a duration (체증형 종신 / 간병비), the bidirectional
  partner of the existing reduction rule.

The factor must apply identically across every kernel path, so the key tests are
full==fast parity (Markov and semi-Markov) and fast==gpu.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.basis import Basis, CoverageRate

_NONE = lambda s, a, d: np.full(np.shape(a), 0.0)


# ---------------------------------------------------------------------------
# BE1 -- annuity escalation hand-calc: 5%/yr grows 100 -> 105 -> 110.25 exactly
# ---------------------------------------------------------------------------
def _annuity_mp():
    return fcf.ModelPoints(
        issue_age=np.array([60], dtype=np.int64), premium=np.array([0.0]),
        term_months=np.array([36], dtype=np.int64),
        annuity_payment=np.array([100.0]),
        annuity_frequency_months=np.array([12], dtype=np.int64),
        benefits={0: np.array([0.0])},
        calculation_methods={"ANN": fcf.CalculationMethod.ANNUITY})


def _annuity_basis(af=None):
    return Basis(mortality_annual=_NONE, lapse_annual=_NONE, discount_annual=0.0,
                 ra_confidence=0.75, mortality_cv=0.10,
                 coverages=(CoverageRate("ANN", _NONE),), annuity_factor_annual=af)


def test_annuity_escalation_hand_calc():
    """No decrement, flat 0 discount, annual annuity 100. A 5%/yr escalation
    factor pays 100, 105, 110.25 in years 0, 1, 2 (annuity is an outflow, so a
    higher payout raises the BEL by exactly the extra PV)."""
    af = lambda s, a, d, ic, el: 1.05 ** d
    mp = _annuity_mp()
    m = fcf.gmm.measure(mp, _annuity_basis(af), full=True)
    acf = m.cashflows.annuity_cf[0]
    assert acf[0] == pytest.approx(100.0)
    assert acf[12] == pytest.approx(105.0)
    assert acf[24] == pytest.approx(110.25)
    # default None is the level annuity, bit-identical
    acf0 = fcf.gmm.measure(mp, _annuity_basis(), full=True).cashflows.annuity_cf[0]
    assert acf0[12] == pytest.approx(100.0)
    # escalation raises BEL (annuity outflow up)
    m0 = fcf.gmm.measure(mp, _annuity_basis(), full=True)
    assert m.bel[0] > m0.bel[0]


def test_annuity_escalation_full_matches_fast():
    """full==fast parity with an escalating annuity (Markov)."""
    af = lambda s, a, d, ic, el: 1.05 ** d
    mp = _annuity_mp()
    b = Basis(mortality_annual=lambda s, a, d: np.full(np.shape(a), 0.01),
              lapse_annual=lambda s, a, d: np.full(np.shape(d), 0.0),
              discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
              coverages=(CoverageRate("ANN", _NONE),), annuity_factor_annual=af)
    full = fcf.gmm.measure(mp, b, full=True)
    fast = fcf.gmm.measure(mp, b, full=False)
    assert np.allclose(full.bel, fast.bel, rtol=1e-9)
    assert np.allclose(full.csm, fast.csm, rtol=1e-9)
    assert not np.isclose(full.bel[0],
                          fcf.gmm.measure(mp, _annuity_basis(), full=True).bel[0])
