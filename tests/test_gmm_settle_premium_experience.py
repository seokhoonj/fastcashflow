"""gmm.settle B96(a)/B97(c) premium experience adjustment (skeleton).

Authoritative skeleton: written before the implementation and activated
unchanged once the feature lands. The anchor facts come from the G-gate
(dev/b96a-premium-experience-gate.md, signed off 2026-06-13) which quotes
IFRS 17 B96(a)/B97(c) and TRG 2018-09 (Agenda Paper 04) verbatim:

* Experience adjustment (Appendix A): the difference between the
  beginning-of-period estimate of premium expected in the period and the
  actual premium cash flow in the period.
* B96(a): a premium experience adjustment that relates to FUTURE service
  adjusts the CSM (measured at the B72(c) locked-in rate).
* B97(c): every OTHER experience adjustment (current / past service) does
  NOT adjust the CSM -- it is recognised immediately in profit or loss, as
  insurance revenue (TRG para 20, B120/B123).
* The split (future vs current/past) is an ENTITY JUDGMENT (TRG para 13, no
  mechanical formula in the standard); fastcashflow exposes it as the
  ``premium_experience_future_fraction`` settle argument, default 0.0 (all
  current/past = the BC233 general rule, and all three TRG worked examples).

Engine wiring (gate part 2, dev/scratch_b96a_gate.py):

* The expected within-period premium is ``k_exp *
  sum(premium_cf[em_open:em_close])`` on the unit projection -- the same
  scale/window the interest line uses.
* ``premium_experience = actual_premium - expected_premium``.
* ``csm_premium_experience    = frac       * premium_experience`` (B96(a),
  into the CSM block, through the SAME paragraph-48/50(b) algebra as the
  count-channel unlocking, BEFORE the floor).
* ``premium_experience_revenue = (1 - frac) * premium_experience`` (B97(c),
  a P&L memo -- in NO balance recursion, mirroring ``finance_wedge``).
* Sign: a premium inflow reduces the BEL, so MORE premium is FAVOURABLE and
  ``csm_premium_experience > 0`` increases the CSM.

Absent ``actual_premium`` every new line is 0 and the movement is
byte-identical to today (the engine.py:974-984 v1 cut is relaxed for the
premium leg only).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, InforceState, ModelPoints)
from fastcashflow.movement import GMMSettlementMovement

settle = getattr(fcf.gmm, "settle", None)
_HAS_PREMIUM_EXPERIENCE = (
    settle is not None
    and "csm_premium_experience" in GMMSettlementMovement.__dataclass_fields__
    and "actual_premium" in InforceState.__dataclass_fields__)
pytestmark = pytest.mark.skipif(
    not _HAS_PREMIUM_EXPERIENCE,
    reason="B96(a) premium experience not implemented yet (v1.1; skeleton "
           "activates unchanged once csm_premium_experience / actual_premium "
           "land)")

CM = {"DEATH": CalculationMethod.DEATH}

# Engine oracle pinned by dev/scratch_b96a_gate.py for the contract below.
EXPECTED_PREMIUM = 1_094_260.860597
PREMIUM_EXPERIENCE_5PCT = 54_713.043030


def _flat(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis(*, discount=0.03):
    return Basis(
        mortality_annual=_flat(0.012), lapse_annual=_flat(0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat(0.012)),),
    )


def _unit(basis, *, term=36):
    unit = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([term]), benefits={0: np.array([1e6])},
        count=np.array([1.0]), calculation_methods=CM,
    )
    return fcf.gmm.measure(unit, basis, full=True)


def _book(basis, *, em_open=12, period=12, scale=1000.0, term=36,
          prior_csm=5_000.0, lc_open=0.0, actual_premium=None):
    em_close = em_open + period
    surv = _unit(basis, term=term).cashflows.inforce[0]
    prior_count = scale * surv[em_open]
    count_close = scale * surv[em_close]
    ids = np.array(["P0"])
    rep = lambda v: np.full(1, v)
    mp = ModelPoints(
        issue_age=rep(40).astype(np.int64), premium=rep(100.0),
        term_months=rep(term).astype(np.int64), benefits={0: rep(1e6)},
        count=rep(count_close), elapsed_months=rep(em_close).astype(np.int64),
        mp_id=ids, product=np.full(1, "A"), calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=rep(em_close).astype(np.int64),
        count=rep(count_close), prior_csm=rep(prior_csm),
        lock_in_rate=basis.discount_annual, prior_count=rep(prior_count),
        prior_loss_component=rep(lc_open) if lc_open else None,
        actual_premium=(None if actual_premium is None
                        else rep(actual_premium)),
    )
    return mp, state


def _expected_premium_oracle(basis, *, em_open=12, period=12, scale=1000.0,
                             term=36):
    """Recompute the expected within-period premium independently."""
    from dataclasses import replace
    from fastcashflow.engine import _measure_full
    unit = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([term]), benefits={0: np.array([1e6])},
        count=np.ones(1), calculation_methods=CM)
    m = _measure_full(unit, basis)
    cf = m.cashflows
    surv_open = cf.inforce[0, em_open]
    prior_count = scale * surv_open
    k_exp = prior_count / surv_open
    cols = em_open + np.arange(period)
    return float(k_exp * cf.premium_cf[0, cols].sum())


# ---------------------------------------------------------------------------
# absent input -- byte-identical to today
# ---------------------------------------------------------------------------

def test_absent_actual_premium_is_byte_identical():
    basis = _basis()
    mp, state = _book(basis)               # actual_premium=None
    mv = settle(mp, state, basis, period_months=12)
    assert np.all(mv.csm_premium_experience == 0.0)
    assert np.all(mv.premium_experience_revenue == 0.0)
    # baseline csm/bel unaffected by the new (inert) lines
    base_walk = (mv.csm_opening + mv.csm_accretion
                 + mv.csm_experience_unlocking + mv.csm_premium_experience
                 - mv.loss_component_reversed + mv.loss_component_recognised
                 - mv.csm_release)
    np.testing.assert_allclose(base_walk, mv.csm_closing, rtol=1e-12)


# ---------------------------------------------------------------------------
# the engine oracle (gate part 2)
# ---------------------------------------------------------------------------

def test_expected_within_period_premium_matches_gate():
    basis = _basis()
    got = _expected_premium_oracle(basis)
    np.testing.assert_allclose(got, EXPECTED_PREMIUM, rtol=1e-9)


def test_premium_experience_is_actual_minus_expected():
    basis = _basis()
    actual = EXPECTED_PREMIUM * 1.05
    mp, state = _book(basis, actual_premium=actual)
    mv = settle(mp, state, basis, period_months=12,
                premium_experience_future_fraction=0.0)
    pe = mv.csm_premium_experience + mv.premium_experience_revenue
    np.testing.assert_allclose(pe[0], PREMIUM_EXPERIENCE_5PCT, rtol=1e-7)
    np.testing.assert_allclose(pe[0], actual - EXPECTED_PREMIUM, rtol=1e-9)


# ---------------------------------------------------------------------------
# B97(c): frac=0 -> all current/past -> insurance revenue, CSM untouched
# (TRG Examples A and B)
# ---------------------------------------------------------------------------

def test_frac_zero_routes_all_to_revenue_not_csm():
    basis = _basis()
    actual = EXPECTED_PREMIUM * 1.05
    mp, state = _book(basis, actual_premium=actual)
    base = settle(*_book(basis), basis, period_months=12)
    mv = settle(mp, state, basis, period_months=12,
                premium_experience_future_fraction=0.0)
    # the whole experience is revenue (P&L)
    np.testing.assert_allclose(
        mv.premium_experience_revenue[0], PREMIUM_EXPERIENCE_5PCT, rtol=1e-7)
    assert np.all(mv.csm_premium_experience == 0.0)
    # P&L memo only: the CSM / BEL balances equal the no-premium baseline
    np.testing.assert_allclose(mv.csm_closing, base.csm_closing, rtol=1e-12)
    np.testing.assert_allclose(mv.bel_closing, base.bel_closing, rtol=1e-12)


# ---------------------------------------------------------------------------
# B96(a): frac=1 -> all future service -> CSM, no revenue
# ---------------------------------------------------------------------------

def test_frac_one_routes_all_to_csm_not_revenue():
    basis = _basis()
    actual = EXPECTED_PREMIUM * 1.05
    mp, state = _book(basis, actual_premium=actual)
    mv = settle(mp, state, basis, period_months=12,
                premium_experience_future_fraction=1.0)
    np.testing.assert_allclose(
        mv.csm_premium_experience[0], PREMIUM_EXPERIENCE_5PCT, rtol=1e-7)
    assert np.all(mv.premium_experience_revenue == 0.0)
    # a favourable premium experience increases the post-adjustment CSM
    base = settle(*_book(basis), basis, period_months=12)
    assert mv.csm_closing[0] > base.csm_closing[0]


def test_split_fraction_partitions_the_experience():
    basis = _basis()
    actual = EXPECTED_PREMIUM * 1.05
    frac = 0.3
    mp, state = _book(basis, actual_premium=actual)
    mv = settle(mp, state, basis, period_months=12,
                premium_experience_future_fraction=frac)
    pe = PREMIUM_EXPERIENCE_5PCT
    np.testing.assert_allclose(
        mv.csm_premium_experience[0], frac * pe, rtol=1e-6)
    np.testing.assert_allclose(
        mv.premium_experience_revenue[0], (1.0 - frac) * pe, rtol=1e-6)


# ---------------------------------------------------------------------------
# reconciliation: the new CSM term is in the closing recursion; the
# count-channel three-term cross identity is UNCHANGED
# ---------------------------------------------------------------------------

def test_csm_closing_recursion_includes_premium_experience():
    basis = _basis()
    actual = EXPECTED_PREMIUM * 1.10
    mp, state = _book(basis, actual_premium=actual)
    mv = settle(mp, state, basis, period_months=12,
                premium_experience_future_fraction=0.5)
    walk = (mv.csm_opening + mv.csm_accretion + mv.csm_experience_unlocking
            + mv.csm_premium_experience
            - mv.loss_component_reversed + mv.loss_component_recognised
            - mv.csm_release)
    np.testing.assert_allclose(walk, mv.csm_closing, rtol=1e-10)


def test_count_channel_cross_identity_unchanged():
    """csm_experience_unlocking + finance_wedge == -(bel_exp + ra_exp);
    the premium experience is a SEPARATE line and must not pollute it."""
    basis = _basis()
    actual = EXPECTED_PREMIUM * 1.05
    mp, state = _book(basis, actual_premium=actual)
    mv = settle(mp, state, basis, period_months=12,
                premium_experience_future_fraction=0.5)
    np.testing.assert_allclose(
        mv.csm_experience_unlocking + mv.finance_wedge,
        -(mv.bel_experience + mv.ra_experience), rtol=1e-10)


# ---------------------------------------------------------------------------
# loss component: a favourable future-service premium experience reverses an
# opening loss component (paragraph 48), same algebra as the count channel
# ---------------------------------------------------------------------------

def test_favourable_premium_experience_reverses_loss_component():
    basis = _basis()
    actual = EXPECTED_PREMIUM * 1.20    # strongly favourable
    mp, state = _book(basis, prior_csm=0.0, lc_open=20_000.0,
                      actual_premium=actual)
    mv = settle(mp, state, basis, period_months=12,
                premium_experience_future_fraction=1.0)
    assert mv.loss_component_reversed[0] > 0.0
    np.testing.assert_allclose(
        mv.loss_component_opening - mv.loss_component_reversed
        + mv.loss_component_recognised, mv.loss_component_closing, rtol=1e-10)


# ---------------------------------------------------------------------------
# fraction validation + aggregate forwarding (Codex review P1 x3)
# ---------------------------------------------------------------------------

def test_nan_fraction_rejected_even_when_premium_absent():
    """A NaN fraction must raise, not silently leak NaN through NaN*0 into the
    closing balances when actual_premium is absent."""
    basis = _basis()
    mp, state = _book(basis)               # actual_premium=None
    with pytest.raises(ValueError, match="finite"):
        settle(mp, state, basis, period_months=12,
               premium_experience_future_fraction=np.nan)


def test_out_of_range_fraction_rejected():
    basis = _basis()
    mp, state = _book(basis, actual_premium=EXPECTED_PREMIUM)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"\[0, 1\]|finite"):
            settle(mp, state, basis, period_months=12,
                   premium_experience_future_fraction=bad)


def test_wrong_shape_fraction_rejected():
    basis = _basis()
    mp, state = _book(basis, actual_premium=EXPECTED_PREMIUM)
    with pytest.raises(ValueError, match="scalar or one entry"):
        settle(mp, state, basis, period_months=12,
               premium_experience_future_fraction=np.zeros((1, 1)))


def test_aggregate_forwards_fraction_and_matches_per_mp_sum():
    """settle_aggregate must forward the fraction so the aggregate reproduces
    the per-MP settle sum (else it routes the whole experience to revenue)."""
    basis = _basis()
    actual = EXPECTED_PREMIUM * 1.05
    mp, state = _book(basis, actual_premium=actual)
    frac = 0.4
    mv = settle(mp, state, basis, period_months=12,
                premium_experience_future_fraction=frac)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12,
                                   premium_experience_future_fraction=frac)
    np.testing.assert_allclose(
        agg.csm_premium_experience,
        float(mv.csm_premium_experience.sum()), rtol=1e-9)
    np.testing.assert_allclose(
        agg.premium_experience_revenue,
        float(mv.premium_experience_revenue.sum()), rtol=1e-9)
    np.testing.assert_allclose(
        agg.csm_closing, float(mv.csm_closing.sum()), rtol=1e-9)
    assert abs(agg.csm_premium_experience) > 0.0
