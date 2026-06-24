"""gmm.settle -- the IFRS 17 paragraph-44 settlement movement (skeleton).

Authoritative skeleton (P-5c pattern): written before the implementation and
activated unchanged by it. The anchor facts, from dev/gmm-settle-contract.md
(O-6(ii) signed off 2026-06-12) and the G1 gate hand-calcs:

* GMM has no account value, so the expected and observed legs share ONE unit
  projection and differ only by scale: k_exp = prior_count /
  unit_inforce[em_open], k_obs = count / unit_inforce[em_close].
* CSM accretion and the paragraph-44(c) unlocking are measured at the
  LOCKED-IN rate (B72(b)/(c), "determined on initial recognition" -- AASB 17
  verbatim, g1-verbatim.md); the BEL / RA blocks are current-rate (B72(a)).
  The difference is a named ``finance_wedge`` line (B97(a), outside the CSM
  block) -- the VFA two-term cross-tie does NOT carry over; the GMM tie is
  three-term:

      csm_experience_unlocking + finance_wedge
          == -(bel_experience + ra_experience)

* With on-track experience the single period-end B119 release (em_open
  denominator) telescopes to the monthly carry of ``gmm.measure_inforce``
  exactly, and the closing BEL / RA equal the carry headline (the F2
  identity, machine precision in g1-handcalc.md).
* The loss-component algebra reuses the paragraph-48/50(b) conservation
  identity (the full six-case scalar sign grid lives with the shared
  algebra helper -- extend the vfa grid when it is hoisted; this file pins
  the behavioural cases through the public entry).

Sign of a count shock on this outflow-heavy test book (claims >> premiums,
unit BEL > 0): MORE survivors than expected -> more future net outflow ->
bel_experience > 0 -> x < 0, UNFAVOURABLE; fewer survivors -> FAVOURABLE.
"""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, InforceState, ModelPoints)

settle = getattr(fcf.gmm, "settle", None)
pytestmark = pytest.mark.skipif(
    settle is None,
    reason="gmm.settle not implemented yet (redesign step 1; skeleton "
           "activates unchanged once it lands)")

CM = {"DEATH": CalculationMethod.DEATH}


def _flat(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis(*, discount=0.03, surrender=False):
    kw = {}
    if surrender:
        kw.update(surrender_value_curve=np.full(36, 30_000.0),
                  surrender_value_basis="amount_per_policy")
    return Basis(
        mortality_annual=_flat(0.012), lapse_annual=_flat(0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat(0.012)),), **kw,
    )


def _unit(basis, *, term=36, age=40, prem=100.0):
    unit = ModelPoints(
        issue_age=np.array([age]), premium=np.array([prem]),
        term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1.0]), calculation_methods=CM,
    )
    return fcf.gmm.measure(unit, basis, full=True)


def _book(basis, *, em_open=12, period=12, scale=1000.0, term=36,
          prior_csm=5_000.0, lc_open=0.0, count_factor=1.0, lock_in=None,
          n=1):
    """An in-force book seated at em_open + period.

    On-track by default (the closing count is the prior count decremented by
    the engine's own survival); ``count_factor`` scales the closing count
    off-track (>1 unfavourable, <1 favourable on this outflow-heavy book).
    """
    em_close = em_open + period
    surv = _unit(basis, term=term).cashflows.inforce[0]
    prior_count = scale * surv[em_open]
    surv_close = surv[em_close] if em_close < surv.shape[0] else 0.0
    count_close = scale * surv_close * count_factor
    ids = np.array([f"P{i}" for i in range(n)])
    rep = lambda v: np.full(n, v)
    mp = ModelPoints(
        issue_age=rep(40).astype(np.int64), premium=rep(100.0),
        term_months=rep(term).astype(np.int64), benefits={"DEATH": rep(1e6)},
        count=rep(count_close), elapsed_months=rep(em_close).astype(np.int64),
        mp_id=ids, product=np.full(n, "A"), calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=rep(em_close).astype(np.int64),
        count=rep(count_close), prior_csm=rep(prior_csm),
        lock_in_rate=(basis.discount_annual if lock_in is None else lock_in),
        prior_count=rep(prior_count),
        prior_loss_component=rep(lc_open) if lc_open else None,
    )
    return mp, state


def _csm_walk(mv):
    return (mv.csm_opening + mv.csm_accretion + mv.csm_experience_unlocking
            - mv.loss_component_reversed + mv.loss_component_recognised
            - mv.csm_release)


# ---------------------------------------------------------------------------
# block identities -- every movement reconciles by construction
# ---------------------------------------------------------------------------

def test_settle_blocks_reconcile():
    mp, state = _book(_basis())
    mv = settle(mp, state, _basis(), period_months=12)
    np.testing.assert_allclose(
        mv.bel_opening + mv.bel_interest - mv.bel_release + mv.bel_experience,
        mv.bel_closing, rtol=1e-10)
    np.testing.assert_allclose(
        mv.ra_opening + mv.ra_interest - mv.ra_release + mv.ra_experience,
        mv.ra_closing, rtol=1e-10)
    np.testing.assert_allclose(_csm_walk(mv), mv.csm_closing, rtol=1e-10)
    np.testing.assert_allclose(
        mv.loss_component_opening - mv.loss_component_reversed
        + mv.loss_component_recognised, mv.loss_component_closing, rtol=1e-10)
    assert mv.measurement_basis == "settlement"
    assert mv.period_months == 12


def test_settle_universal_life_reconciles_and_matches_measure_inforce():
    """A universal-life account book settles: the locked-in BEL is netted of the
    account fund and the incurred-claims line reads the net amount at risk (the
    account-value part of the death benefit is the policyholder's deposit, an
    investment component, B121). On-track, the movement reconciles, the
    experience lines vanish, and the closing BEL lands exactly on a fresh
    measure_inforce of the same account book at the closing date.
    """
    basis = fcf.samples.basis("ul")
    one = fcf.samples.model_points("ul").subset([0])
    em_open, period = 12, 12
    em_close = em_open + period
    mu = fcf.gmm.measure(replace(one, count=np.ones(1)), basis, full=True)
    surv = mu.cashflows.inforce[0]
    scale = 1000.0
    ids = np.array(["U0"])
    rep = lambda v: np.full(1, v)
    mp = replace(one, count=rep(scale * surv[em_close]),
                 elapsed_months=rep(em_close).astype(np.int64),
                 mp_id=ids, product=np.full(1, "A"))
    state = InforceState(
        mp_id=ids, elapsed_months=rep(em_close).astype(np.int64),
        count=rep(scale * surv[em_close]),
        prior_csm=rep(float(mu.csm[0]) * scale),
        lock_in_rate=basis.discount_annual,
        prior_count=rep(scale * surv[em_open]))
    assert mu.cashflows.account is not None          # it really is an account book
    mv = settle(mp, state, basis, period_months=period)
    np.testing.assert_allclose(
        mv.bel_opening + mv.bel_interest - mv.bel_release + mv.bel_experience,
        mv.bel_closing, rtol=1e-9)
    np.testing.assert_allclose(_csm_walk(mv), mv.csm_closing, rtol=1e-9)
    # on-track: the experience / unlocking lines vanish
    np.testing.assert_allclose(mv.bel_experience, 0.0, atol=1.0)
    np.testing.assert_allclose(mv.csm_experience_unlocking, 0.0, atol=1.0)
    # gold anchor: the settled closing BEL == a fresh measure_inforce of the same
    # account book at the closing date (the fund netting telescopes exactly).
    carry = fcf.gmm.measure_inforce(mp, state, basis, period_months=period,
                                    full=True)
    np.testing.assert_allclose(mv.bel_closing, carry.bel, rtol=1e-6)


def test_settle_annuitizing_universal_life_payout_has_no_phantom_death_claim():
    """In an annuitizing UL's payout phase the account is converted (av_mid == 0)
    and there is no death benefit, so the incurred insurance claim is zero. The
    NAR is cf.mortality_cf - deaths * av_mid (the actual gross less the account
    part), NOT a face-minus-av_mid reconstruction -- which would fabricate a
    full-face claim where the account no longer exists. Regression for the
    annuitizing-payout phantom-claim defect.
    """
    basis = fcf.samples.basis("ul-annuity")
    one = fcf.samples.model_points("ul-annuity").subset([0])
    a = int(np.asarray(one.annuitization_months)[0])
    em_open, period = a + 12, 12          # a window inside the payout phase
    em_close = em_open + period
    mu = fcf.gmm.measure(replace(one, count=np.ones(1)), basis, full=True)
    surv = mu.cashflows.inforce[0]
    # the death benefit is genuinely zero across the payout window
    assert mu.cashflows.mortality_cf[0, em_open:em_close].sum() == 0.0
    scale = 1000.0
    ids = np.array(["U0"])
    rep = lambda v: np.full(1, v)
    mp = replace(one, count=rep(scale * surv[em_close]),
                 elapsed_months=rep(em_close).astype(np.int64),
                 mp_id=ids, product=np.full(1, "A"))
    state = InforceState(
        mp_id=ids, elapsed_months=rep(em_close).astype(np.int64),
        count=rep(scale * surv[em_close]),
        prior_csm=rep(float(mu.csm[0]) * scale),
        lock_in_rate=basis.discount_annual,
        prior_count=rep(scale * surv[em_open]))
    mv = settle(mp, state, basis, period_months=period)
    np.testing.assert_allclose(mv.claims_incurred, 0.0, atol=1e-6)
    np.testing.assert_allclose(
        mv.bel_opening + mv.bel_interest - mv.bel_release + mv.bel_experience,
        mv.bel_closing, rtol=1e-9)


def test_three_term_tie_is_the_gmm_cross_identity():
    """csm_experience_unlocking + finance_wedge == -(bel_exp + ra_exp).
    Off-track book under two rates -- the tie must hold exactly even when
    the wedge is materially nonzero (g1-handcalc.md (1))."""
    basis = _basis(discount=0.05)
    mp, state = _book(basis, lock_in=0.03, count_factor=1.5)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(
        mv.csm_experience_unlocking + mv.finance_wedge,
        -(mv.bel_experience + mv.ra_experience), rtol=1e-10)
    assert not np.allclose(mv.finance_wedge, 0.0)


# ---------------------------------------------------------------------------
# the two-rate mechanics (O-6(ii))
# ---------------------------------------------------------------------------

def test_two_rate_unlocking_matches_first_principles():
    """The G1 gate case in engine convention: current 5%, lock-in 3%, an
    off-track count. The unlocking is the locked-in PV of the future-service
    delta, re-derived here from the unit cash flows with the engine's
    bom/mid timing (premiums at month start, claims and expenses mid-month,
    maturity at the boundary) -- not read back from the implementation."""
    basis = _basis(discount=0.05)
    em_open, period, term = 12, 12, 36
    em_close = em_open + period
    factor = 1.5
    m = _unit(basis, term=term)
    surv = m.cashflows.inforce[0]
    k_exp = (1000.0 * surv[em_open]) / surv[em_open]      # = 1000 (unit leg)
    k_obs = (1000.0 * surv[em_close] * factor) / surv[em_close]
    dk = k_obs - k_exp

    cf = m.cashflows
    t = np.arange(term)
    bom_l = (1.0 + 0.03) ** (-t / 12.0)                   # locked-in factors
    mid_l = (1.0 + 0.03) ** (-(t + 0.5) / 12.0)
    bom_l_T = (1.0 + 0.03) ** (-term / 12.0)
    # engine timing (numerics._roll_forward_kernel): premiums and annuities at
    # the month start, claims / morbidity / disability / expenses / surrender
    # mid-month, the maturity benefit at the boundary
    out_mid = (cf.mortality_cf + cf.morbidity_cf + cf.expense_cf
               + cf.disability_cf + cf.surrender_cf)[0]
    pv_lock = ((out_mid[em_close:] * mid_l[em_close:]).sum()
               + ((cf.annuity_cf - cf.premium_cf)[0, em_close:]
                  * bom_l[em_close:]).sum()
               + cf.maturity_cf[0] * bom_l_T) / bom_l[em_close]
    # dk carries the per-survivor normalisation (k = count / inforce), so the
    # cohort-level locked PV multiplies dk directly -- the exact locked-in
    # analogue of bel_experience = dk x bel_path[em_close]. The RA change
    # has no rate prescription (B96(d), g1-verbatim.md) and enters the CSM
    # at its current measure: dk x ra_path[em_close].
    expected_unlocking = -(dk * pv_lock + dk * m.ra_path[0, em_close])

    mp, state = _book(basis, em_open=em_open, period=period, term=term,
                      lock_in=0.03, count_factor=factor)
    mv = settle(mp, state, basis, period_months=period)
    np.testing.assert_allclose(mv.csm_experience_unlocking[0],
                               expected_unlocking, rtol=1e-9)
    # the wedge is the current-vs-locked-in measurement gap of the SAME
    # delta; with current 5% > lock-in 3% on a net-outflow tail it is
    # strictly nonzero (g1-handcalc.md: ~1% of the unlocking)
    assert abs(float(mv.finance_wedge[0])) > 0.0


def test_wedge_is_zero_when_the_flat_basis_equals_the_lock_in():
    """The O-6(i) degeneration pin: flat current == locked-in -> wedge == 0
    (explicit atol -- the wedge is a difference of two PV passes)."""
    basis = _basis(discount=0.03)
    mp, state = _book(basis, lock_in=0.03, count_factor=0.5)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.finance_wedge, 0.0, atol=1e-8)


def test_csm_accretion_is_direct_compounding_at_the_lock_in_rate():
    """44(b)/B72(b): the accretion line is prior_csm x ((1+lock_in)^(p/12)-1)
    -- the locked-in 3%, not the current 5% (g1-handcalc.md (1): 300 vs 500)."""
    basis = _basis(discount=0.05)
    mp, state = _book(basis, lock_in=0.03, prior_csm=10_000.0)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.csm_accretion, 10_000.0 * 0.03, rtol=1e-10)


def test_on_track_experience_lines_are_zero():
    """On-track counts: k_obs == k_exp, so every experience line is zero and
    the wedge with it -- the movement is pure accretion + release."""
    basis = _basis(discount=0.05)
    mp, state = _book(basis, lock_in=0.03)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.bel_experience, 0.0, atol=1e-6)
    np.testing.assert_allclose(mv.csm_experience_unlocking, 0.0, atol=1e-6)
    np.testing.assert_allclose(mv.finance_wedge, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# B119 release -- independently pinned (not via the carry)
# ---------------------------------------------------------------------------

def test_release_is_the_b119_fraction_on_the_post_adjustment_balance():
    """On-track: release == csm_after x (tail[open]-tail[close]) / tail[open],
    the coverage-unit tails read from the engine's own unit in-force --
    independent of measure_inforce (a wrong denominator cannot hide in the
    telescoping test alone)."""
    basis = _basis()
    em_open, period = 12, 12
    em_close = em_open + period
    surv = _unit(basis).cashflows.inforce[0]
    tail = np.concatenate([np.cumsum(surv[::-1])[::-1], [0.0]])
    mp, state = _book(basis, em_open=em_open, period=period,
                      prior_csm=10_000.0)
    mv = settle(mp, state, basis, period_months=period)
    csm_after = (mv.csm_opening + mv.csm_accretion
                 + mv.csm_experience_unlocking - mv.loss_component_reversed
                 + mv.loss_component_recognised)
    frac = (tail[em_open] - tail[em_close]) / tail[em_open]
    np.testing.assert_allclose(mv.csm_release, csm_after * frac, rtol=1e-9)


def test_coverage_unit_lines_are_recorded():
    """B119 inputs for the per-GoC re-aggregation: units provided over the
    period (k_exp scale) and units remaining (k_obs scale), on the movement."""
    basis = _basis()
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=12)
    assert np.all(mv.coverage_units_provided > 0.0)
    assert np.all(mv.coverage_units_future > 0.0)
    frac = mv.coverage_units_provided / (mv.coverage_units_provided
                                         + mv.coverage_units_future)
    csm_after = (mv.csm_opening + mv.csm_accretion
                 + mv.csm_experience_unlocking - mv.loss_component_reversed
                 + mv.loss_component_recognised)
    np.testing.assert_allclose(mv.csm_release, csm_after * frac, rtol=1e-10)


# ---------------------------------------------------------------------------
# telescoping and the F2 identity (the carry cross-checks)
# ---------------------------------------------------------------------------

def test_on_track_settle_equals_the_monthly_carry():
    """Single period-end B119 release == measure_inforce's monthly carry when
    on-track (the telescoping anchor, mirroring test_vfa_settle)."""
    basis = _basis()
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=12)
    carry = fcf.gmm.measure_inforce(mp, state, basis, period_months=12,
                                    full=False)
    np.testing.assert_allclose(mv.csm_closing, carry.csm, rtol=1e-10)


def test_closing_bel_ra_equal_the_carry_headline_f2():
    """The F2 identity (g1-handcalc.md (2)): the settle observed-leg closing
    BEL / RA are the carry headline's re-based slice, exactly -- including a
    surrender book (the identity is shared arithmetic; only the accuracy
    caveat differs by surrender mode)."""
    basis = _basis(surrender=True)
    mp, state = _book(basis, count_factor=0.3)
    mv = settle(mp, state, basis, period_months=12)
    carry = fcf.gmm.measure_inforce(mp, state, basis, period_months=12,
                                    full=False)
    np.testing.assert_allclose(mv.bel_closing, carry.bel, rtol=1e-10)
    np.testing.assert_allclose(mv.ra_closing, carry.ra, rtol=1e-10)


def test_two_six_month_settles_chain_to_one_twelve_month():
    """closing_inputs() returns the closing-date pair whose prior_* fields
    are the closing balances; the caller advances it with the next observed
    snapshot. On-track, 6m x 2 == 12m."""
    basis = _basis()
    surv = _unit(basis).cashflows.inforce[0]
    mp12, state12 = _book(basis, em_open=12, period=12)
    one = settle(mp12, state12, basis, period_months=12)

    mp6, state6 = _book(basis, em_open=12, period=6)
    first = settle(mp6, state6, basis, period_months=6)
    mp_mid, state_mid = first.closing_inputs()
    # the seed pair is at the closing date with the closing balances
    assert isinstance(mp_mid, ModelPoints)
    assert isinstance(state_mid, InforceState)
    np.testing.assert_array_equal(state_mid.elapsed_months, [18])
    np.testing.assert_allclose(state_mid.prior_csm, first.csm_closing)
    np.testing.assert_allclose(state_mid.prior_count, state6.count)
    # advance to the next observation (on-track at month 24)
    count24 = np.array([1000.0 * surv[24]])
    from dataclasses import replace
    mp_next = replace(mp_mid, elapsed_months=np.array([24]), count=count24)
    state_next = InforceState(
        mp_id=state_mid.mp_id, elapsed_months=np.array([24]),
        count=count24, prior_csm=state_mid.prior_csm,
        lock_in_rate=state_mid.lock_in_rate,
        prior_count=state_mid.prior_count,
        prior_loss_component=state_mid.prior_loss_component)
    second = settle(mp_next, state_next, basis, period_months=6)
    np.testing.assert_allclose(second.csm_closing, one.csm_closing,
                               rtol=1e-10)
    np.testing.assert_allclose(second.bel_closing, one.bel_closing,
                               rtol=1e-10)


# ---------------------------------------------------------------------------
# loss component (paragraph 48 / 50(b))
# ---------------------------------------------------------------------------

def test_unfavourable_beyond_the_csm_falls_into_the_loss_component():
    """MORE survivors than expected on this outflow-heavy book is adverse
    (more future net outflow); on a thin CSM the unfavourable change floors
    the CSM at zero and the excess is recognised in the loss component."""
    basis = _basis()
    mp, state = _book(basis, prior_csm=1.0, count_factor=2.0)
    mv = settle(mp, state, basis, period_months=12)
    assert float(mv.csm_experience_unlocking[0]) < 0.0       # adverse, pinned
    np.testing.assert_allclose(mv.csm_closing, 0.0, atol=1e-9)
    assert float(mv.loss_component_recognised[0]) > 0.0
    assert float(mv.loss_component_closing[0]) > 0.0


def test_favourable_change_reverses_the_loss_component_before_the_csm():
    """50(b): a favourable change (FEWER survivors here) goes solely to the
    LC until it is zero; only the excess rebuilds the CSM. With a large
    opening LC the reversal is partial and the CSM stays at zero."""
    basis = _basis()
    mp, state = _book(basis, prior_csm=0.0, lc_open=1e9, count_factor=0.8)
    mv = settle(mp, state, basis, period_months=12)
    assert float(mv.csm_experience_unlocking[0]) > 0.0       # favourable
    assert float(mv.loss_component_reversed[0]) > 0.0
    assert float(mv.loss_component_closing[0]) > 0.0         # LC not used up
    np.testing.assert_allclose(mv.csm_closing, 0.0, atol=1e-9)
    # conservation: reversal capped by the opening LC
    assert float(mv.loss_component_reversed[0]) <= 1e9 + 1e-3


def test_favourable_beyond_the_lc_rebuilds_the_csm():
    basis = _basis()
    mp, state = _book(basis, prior_csm=0.0, lc_open=1.0, count_factor=0.5)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.loss_component_closing, 0.0, atol=1e-9)
    assert float(mv.csm_closing[0]) > 0.0


def test_rejects_a_state_carrying_both_csm_and_loss_component():
    basis = _basis()
    mp, state = _book(basis, prior_csm=5_000.0, lc_open=1_000.0)
    with pytest.raises(ValueError, match="loss_component|prior_csm"):
        settle(mp, state, basis, period_months=12)


# ---------------------------------------------------------------------------
# derecognition / boundaries
# ---------------------------------------------------------------------------

def test_final_settlement_releases_everything():
    """em_close at the boundary with a zero closing snapshot: full B119
    derecognition -- closing CSM, BEL, RA and LC all zero, and the whole
    post-adjustment CSM is in the release line."""
    basis = _basis()
    mp, state = _book(basis, em_open=30, period=6, term=36, count_factor=0.0)
    mv = settle(mp, state, basis, period_months=6)
    np.testing.assert_allclose(mv.csm_closing, 0.0, atol=1e-9)
    np.testing.assert_allclose(mv.bel_closing, 0.0, atol=1e-9)
    np.testing.assert_allclose(mv.ra_closing, 0.0, atol=1e-9)
    np.testing.assert_allclose(mv.loss_component_closing, 0.0, atol=1e-9)
    csm_after = (mv.csm_opening + mv.csm_accretion
                 + mv.csm_experience_unlocking - mv.loss_component_reversed
                 + mv.loss_component_recognised)
    np.testing.assert_allclose(mv.csm_release, csm_after, rtol=1e-9)


def test_final_settlement_closing_past_the_boundary():
    """A long-matured row may close PAST the boundary (term 36, elapsed 42):
    every closing-column read is clamped and zeroed -- full derecognition,
    no IndexError (Codex review P0)."""
    basis = _basis()
    surv = _unit(basis).cashflows.inforce[0]
    ids = np.array(["P0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([36]), benefits={"DEATH": np.array([1e6])},
        count=np.array([0.0]), elapsed_months=np.array([42]), mp_id=ids,
        product=np.array(["A"]), calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([42], dtype=np.int64),
        count=np.array([0.0]), prior_csm=np.array([5_000.0]),
        lock_in_rate=0.03, prior_count=np.array([1000.0 * surv[30]]),
    )
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.csm_closing, 0.0, atol=1e-9)
    np.testing.assert_allclose(mv.bel_closing, 0.0, atol=1e-9)
    np.testing.assert_allclose(mv.loss_component_closing, 0.0, atol=1e-9)


def test_zero_count_before_the_boundary_fully_derecognises():
    """Mid-boundary count=0 (mass surrender): no future coverage units, so
    the release fraction is 1 -- full derecognition without special-casing
    (paragraph 76; the O-3 split routes this to settle, not the carry)."""
    basis = _basis()
    mp, state = _book(basis, em_open=12, period=12, count_factor=0.0)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.csm_closing, 0.0, atol=1e-9)
    np.testing.assert_allclose(mv.bel_closing, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_guards():
    basis = _basis()
    mp, state = _book(basis)

    # settlement_pattern is now accepted -- it carries the liability for
    # incurred claims (paragraphs 40(b)/42); the dedicated LIC behaviour lives
    # in test_gmm_settle_lic.py.
    sp = Basis(
        mortality_annual=_flat(0.012), lapse_annual=_flat(0.05),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        settlement_pattern=np.array([0.5, 0.5]),
        coverages=(CoverageRate("DEATH", _flat(0.012)),),
    )
    sp_mv = settle(mp, state, sp, period_months=12)
    np.testing.assert_allclose(
        sp_mv.lic_opening + sp_mv.claims_incurred + sp_mv.lic_finance
        - sp_mv.claims_paid, sp_mv.lic_closing, rtol=1e-10)

    # prior_count is mandatory (the expected leg's scale)
    bare = InforceState(
        mp_id=state.mp_id, elapsed_months=state.elapsed_months,
        count=state.count, prior_csm=state.prior_csm,
        lock_in_rate=state.lock_in_rate)
    with pytest.raises(ValueError, match="prior_count"):
        settle(mp, bare, basis, period_months=12)

    # the opening date must not precede inception
    with pytest.raises(ValueError, match="period_months|inception"):
        settle(mp, state, basis, period_months=36)


def test_settle_rejects_a_mixed_model_router():
    basis = _basis()
    mp, state = _book(basis)
    router = fcf.BasisRouter({("A", "GA"): basis},
                             measurement_models={("A", "GA"): "PAA"})
    with pytest.raises(ValueError, match="portfolio|PAA"):
        settle(mp, state, router, period_months=12)


# ---------------------------------------------------------------------------
# downstream arms
# ---------------------------------------------------------------------------

def test_reconcile_returns_a_footing_settlement_table():
    basis = _basis()
    mp, state = _book(basis, n=1)
    mv = settle(mp, state, basis, period_months=12)
    table = fcf.reconcile([mv])
    rec = table[0] if isinstance(table, list) else table
    np.testing.assert_allclose(rec.csm_closing, float(mv.csm_closing.sum()),
                               rtol=1e-10)
    # display convention: the release rows are shown negative in the
    # reconciliation only (the movement keeps them positive)
    np.testing.assert_allclose(rec.csm_release,
                               -float(mv.csm_release.sum()), rtol=1e-10)
    np.testing.assert_allclose(rec.finance_wedge,
                               float(mv.finance_wedge.sum()), atol=1e-8)


def test_write_measurement_writes_the_movement_with_markers(tmp_path):
    import polars as pl
    basis = _basis()
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=12)
    out = tmp_path / "settle.parquet"
    fcf.write_measurement(mv, out)
    df = pl.read_parquet(out)
    for col in ("csm_opening", "csm_release", "csm_closing", "finance_wedge",
                "bel_closing", "measurement_basis"):
        assert col in df.columns
    assert df["measurement_basis"].to_list() == ["settlement"]


# ---------------------------------------------------------------------------
# Cohort-aware lock-in rate: a book whose issue cohorts / GoCs locked in
# different inception rates (B72(b)) is settled by partition (was rejected).
# ---------------------------------------------------------------------------

def test_settle_cohort_aware_lock_in_partitions():
    """A mixed locked-in-rate book settles each model point exactly as if its rate
    were settled on its own sub-book -- every per-MP movement line matches."""
    from dataclasses import fields, replace
    basis = _basis()
    mp, state = _book(basis, n=2)
    mixed = fcf.gmm.settle(mp, replace(state, lock_in_rate=np.array([0.03, 0.05])),
                           basis, period_months=12)
    mp0, s0 = _book(basis, n=1)
    mp1, s1 = _book(basis, n=1)
    a0 = fcf.gmm.settle(mp0, replace(s0, lock_in_rate=0.03), basis, period_months=12)
    a1 = fcf.gmm.settle(mp1, replace(s1, lock_in_rate=0.05), basis, period_months=12)
    n_checked = 0
    for f in fields(mixed):
        v = getattr(mixed, f.name)
        if isinstance(v, np.ndarray) and v.shape == (2,):
            n_checked += 1
            # np.ravel so the per-cohort scalar fields (lock_in_rate) compare
            # alongside the per-MP movement lines
            assert np.allclose(v, [np.ravel(getattr(a0, f.name))[0],
                                   np.ravel(getattr(a1, f.name))[0]],
                               equal_nan=True), f.name
    assert n_checked > 10                                  # the whole movement, not one line
    # the locked-in rate genuinely differs the accretion between the two cohorts
    assert not np.isclose(mixed.csm_accretion[0], mixed.csm_accretion[1])
    # lock_in_rate is carried per-row (cohort-aware), seeding the next period's
    # chain with each row's own rate
    assert np.allclose(mixed.lock_in_rate, [0.03, 0.05])


def test_settle_cohort_aware_closing_inputs_chains_per_row():
    """closing_inputs() on a mixed-rate movement carries each row's OWN locked-in
    rate (and the full-book model points) forward, so chaining a second period
    equals chaining each cohort on its own."""
    from dataclasses import replace
    basis = _basis()
    mp, state = _book(basis, em_open=12, period=6, n=2)
    state = replace(state, lock_in_rate=np.array([0.03, 0.05]))
    first = settle(mp, state, basis, period_months=6)
    mp_mid, state_mid = first.closing_inputs()
    # the per-row locked rate and the full-book model points survive the chain
    assert np.allclose(state_mid.lock_in_rate, [0.03, 0.05])
    assert mp_mid.n_mp == 2
    assert state_mid.prior_csm.shape == (2,)
    # oracle: each row's chain equals that cohort settled alone end to end
    for row, rate in ((0, 0.03), (1, 0.05)):
        mp1, s1 = _book(basis, em_open=12, period=6, n=1)
        f1 = settle(mp1, replace(s1, lock_in_rate=rate), basis, period_months=6)
        _, sm1 = f1.closing_inputs()
        assert np.isclose(state_mid.prior_csm[row], sm1.prior_csm[0])
        assert np.isclose(state_mid.lock_in_rate[row], rate)


def test_settle_uniform_lock_in_array_equals_scalar():
    """A uniform lock_in_rate carried as a per-row array gives the scalar result
    (the array is collapsed, not partitioned)."""
    from dataclasses import fields, replace
    basis = _basis()
    mp, state = _book(basis, n=2)
    arr = fcf.gmm.settle(mp, replace(state, lock_in_rate=np.array([0.03, 0.03])),
                         basis, period_months=12)
    sca = fcf.gmm.settle(mp, replace(state, lock_in_rate=0.03), basis, period_months=12)
    for f in fields(arr):
        v = getattr(sca, f.name)
        if isinstance(v, np.ndarray):
            assert np.allclose(getattr(arr, f.name), v, equal_nan=True), f.name


def test_settle_aggregate_cohort_aware_lock_in_equals_per_mp_sum():
    """settle_aggregate is cohort-aware: a mixed locked-in-rate book aggregates to
    the per-MP gmm.settle sum (each chunk partitions by rate, the lines are
    additive), and the chunking is a memory knob, not a numbers knob."""
    from dataclasses import fields, replace
    from fastcashflow._gmm import _GMM_SETTLEMENT_LINES
    basis = _basis()
    mp, state = _book(basis, n=2)
    state = replace(state, lock_in_rate=np.array([0.03, 0.05]))
    mv = fcf.gmm.settle(mp, state, basis, period_months=12)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    for name in _GMM_SETTLEMENT_LINES:
        assert np.isclose(getattr(agg, name), float(getattr(mv, name).sum())), name
    # chunk_size is a memory knob, never a numbers knob (mixed rates split across
    # chunk boundaries still foot to the same total)
    one = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12, chunk_size=1)
    for name in _GMM_SETTLEMENT_LINES:
        assert np.isclose(getattr(agg, name), getattr(one, name)), name
    # the mixed book has no single locked rate -- the echo is NaN, the per-rate
    # detail is in the per-MP movement
    assert np.isnan(agg.lock_in_rate)


def test_settle_aggregate_uniform_lock_in_array_echoes_scalar():
    """A uniform lock_in_rate carried as a per-row array collapses: the aggregate
    echoes the scalar rate (not NaN), matching the scalar-rate aggregate."""
    from dataclasses import replace
    basis = _basis()
    mp, state = _book(basis, n=2)
    arr = fcf.gmm.settle_aggregate(
        mp, replace(state, lock_in_rate=np.array([0.03, 0.03])),
        basis, period_months=12)
    sca = fcf.gmm.settle_aggregate(
        mp, replace(state, lock_in_rate=0.03), basis, period_months=12)
    assert arr.lock_in_rate == 0.03
    from fastcashflow._gmm import _GMM_SETTLEMENT_LINES
    for name in _GMM_SETTLEMENT_LINES:
        assert np.isclose(getattr(arr, name), getattr(sca, name)), name
