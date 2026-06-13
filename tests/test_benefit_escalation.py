"""Escalating benefits.

Two orthogonal mechanisms:
* ``Basis.annuity_factor_annual`` -- the survival-benefit twin of
  ``premium_factor_annual``: a per-MP year-grid factor on ``annuity_payment``
  for an escalating annuity (e.g. 5%/yr).
* per-coverage step (``coverage_step_month`` / ``coverage_step_factor``) -- a
  benefit step-up at a duration (escalating whole-life / LTC), the bidirectional
  partner of the existing reduction rule.

The factor must apply identically across every kernel path, so the key tests are
full==fast parity (Markov and semi-Markov). (The GPU path shares the same dense
factor grid but is not separately exercised here -- it needs a CUDA device.)
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


# ---------------------------------------------------------------------------
# BE2 -- per-coverage benefit step / escalation (escalating benefit / LTC)
# ---------------------------------------------------------------------------
_ONE = lambda s, a, d: np.full(np.shape(a), 1.0)


def _care_basis():
    return Basis(mortality_annual=_NONE, lapse_annual=_NONE, discount_annual=0.0,
                 ra_confidence=0.75, mortality_cv=0.10,
                 coverages=(CoverageRate("CARE", _ONE),))


def _care_mp(**kw):
    base = dict(issue_age=np.array([50], dtype=np.int64), premium=np.array([0.0]),
                term_months=np.array([360], dtype=np.int64),
                coverage_index=np.array([0], np.int64),
                coverage_amount=np.array([1.0]),
                coverage_offset=np.array([0, 1], np.int64),
                calculation_methods={"CARE": fcf.CalculationMethod.MORBIDITY})
    base.update(kw)
    return fcf.ModelPoints(**base)


def test_benefit_annual_escalation_hand_calc():
    """Escalating benefit: 10%/yr compounds the benefit -- claim ratios are 1.1, 1.21."""
    mp = _care_mp(coverage_escalation_annual=np.array([0.10]))
    cf = fcf.gmm.measure(mp, _care_basis(), full=True).cashflows.morbidity_cf[0]
    assert cf[12] / cf[0] == pytest.approx(1.10, rel=1e-9)
    assert cf[24] / cf[0] == pytest.approx(1.21, rel=1e-9)


def test_benefit_escalation_cap():
    """A cap holds the escalation at cap x base: 15%/yr capped at 3x reaches 3x
    by year 8 (1.15**8 = 3.06 > 3) and stays there."""
    mp = _care_mp(coverage_escalation_annual=np.array([0.15]),
                  coverage_escalation_cap=np.array([3.0]))
    cf = fcf.gmm.measure(mp, _care_basis(), full=True).cashflows.morbidity_cf[0]
    assert cf[84] / cf[0] == pytest.approx(1.15 ** 7, rel=1e-9)     # year 7, uncapped
    assert cf[96] / cf[0] == pytest.approx(3.0, rel=1e-9)           # year 8, capped
    assert cf[240] / cf[0] == pytest.approx(3.0, rel=1e-9)          # year 20, still capped


def test_benefit_single_step():
    """Escalating LTC, '2x after 20 years': the benefit steps to 2x at month 240 and the
    direction is correct (base early, step UP after -- not the reverse)."""
    mp = _care_mp(coverage_step_month=np.array([240], np.int64),
                  coverage_step_factor=np.array([2.0]))
    cf = fcf.gmm.measure(mp, _care_basis(), full=True).cashflows.morbidity_cf[0]
    assert cf[239] / cf[0] == pytest.approx(1.0, rel=1e-9)
    assert cf[240] / cf[0] == pytest.approx(2.0, rel=1e-9)


def test_escalation_default_is_inert_and_fast_auto_routes():
    """No escalation is bit-identical to today (full==fast unchanged); an
    escalating-benefit book auto-routes from full=False to the full kernel
    (no longer raises) -- byte-identical to full=True."""
    mp = _care_mp()
    f = fcf.gmm.measure(mp, _care_basis(), full=True)
    s = fcf.gmm.measure(mp, _care_basis(), full=False)
    assert np.allclose(f.bel, s.bel) and np.allclose(f.csm, s.csm)
    esc = _care_mp(coverage_escalation_annual=np.array([0.10]))
    fast = fcf.gmm.measure(esc, _care_basis(), full=False)
    full = fcf.gmm.measure(esc, _care_basis(), full=True)
    assert np.allclose(fast.bel, full.bel) and np.allclose(fast.csm, full.csm)


# ---------------------------------------------------------------------------
# BE3 -- the CSR rule arrays are validated at construction (one entry per
# coverage, finite, non-negative). A wrong length silently drops / misreads a
# coverage's rule; a negative month / factor silently mis-times or flips it.
# ---------------------------------------------------------------------------
def test_coverage_rule_arrays_must_align_with_coverages():
    # _care_mp has one coverage; a 2-element rule array is a length mismatch
    with pytest.raises(ValueError, match="coverage_step_factor must align"):
        _care_mp(coverage_step_factor=np.array([2.0, 2.0]))
    with pytest.raises(ValueError, match="coverage_escalation_annual must align"):
        _care_mp(coverage_escalation_annual=np.array([0.1, 0.1, 0.1]))


def test_semi_markov_escalation_only_coverage_is_not_dropped():
    """Regression: a step-/escalation-only coverage on a SEMI-MARKOV contract
    must still be paid. The semi-Markov rule-pass skip guard matched only
    waiting / reduction, so a coverage carrying escalation (but no waiting /
    reduction) was excluded from the main pass AND skipped by the rule pass --
    dropped to zero. Reachable for an escalating care benefit on an LTC model."""
    from fastcashflow import State, Transition, StateModel
    _Z = lambda s, a, d: np.full(np.shape(a), 0.0)
    sm = StateModel(states=(
        State("active", pays_premium=True, transitions=(
            Transition("mortality"), Transition("lapse"))),
        State("disabled", sojourn_tracking_months=12, transitions=(Transition("mortality"),)),
    ), seating=(0, 1))
    basis = Basis(mortality_annual=_Z, lapse_annual=_Z, discount_annual=0.0,
                  ra_confidence=0.75, mortality_cv=0.10, state_model=sm,
                  coverages=(CoverageRate("CARE", _ONE),))
    mp = fcf.ModelPoints(
        issue_age=np.array([50], dtype=np.int64), premium=np.array([0.0]),
        term_months=np.array([36], dtype=np.int64),
        coverage_index=np.array([0], np.int64), coverage_amount=np.array([1000.0]),
        coverage_offset=np.array([0, 1], np.int64),
        coverage_escalation_annual=np.array([0.10]),     # escalation, no waiting / reduction
        state=np.array([1], dtype=np.int64),             # seated in the semi-Markov state
        calculation_methods={"CARE": fcf.CalculationMethod.MORBIDITY})
    cf = fcf.gmm.measure(mp, basis, full=True).cashflows.morbidity_cf[0]
    assert cf[0] == pytest.approx(1000.0)                # not dropped to zero
    assert cf[12] / cf[0] == pytest.approx(1.10, rel=1e-9)
    assert cf[24] / cf[0] == pytest.approx(1.21, rel=1e-9)


def test_coverage_rule_arrays_reject_negative_and_nan():
    with pytest.raises(ValueError, match="coverage_step_month must be >= 0"):
        _care_mp(coverage_step_month=np.array([-1], np.int64))
    with pytest.raises(ValueError, match="coverage_step_factor must be >= 0"):
        _care_mp(coverage_step_factor=np.array([-2.0]))
    with pytest.raises(ValueError, match="coverage_escalation_annual must be finite"):
        _care_mp(coverage_escalation_annual=np.array([np.nan]))
    with pytest.raises(ValueError, match="coverage_escalation_cap must be >= 0"):
        _care_mp(coverage_escalation_cap=np.array([-3.0]))
