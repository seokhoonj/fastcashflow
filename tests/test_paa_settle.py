"""paa.settle -- the IFRS 17 paragraph-55(b) PAA settlement movement (skeleton).

Authoritative skeleton (P-5c pattern): written before the implementation and
activated unchanged by it. The anchor facts, from dev/paa-settle-contract.md
(G3 gate, hand-calcs verified 2026-06-12; Codex GO on the formula section):

* The opening balance is RECONSTRUCTED, not carried: the LRC is the
  mechanical paragraph-55(b) roll and the loss component is recalculated per
  paragraphs 57-58 (no balance tracking), so with expected within-period cash
  flows (the v1 cut) the unit projection rebuilds every opening figure.
  ``state.prior_count`` is the only required prior-date input; ``prior_csm``
  / ``lock_in_rate`` / ``prior_loss_component`` are ignored (the PAA has no
  CSM and holds the LRC undiscounted, paragraph 56).
* One unit projection, two scales: the expected leg at
  ``k_exp = prior_count / unit_inforce[em_open]``, the observation at
  ``k_obs = count / unit_inforce[em_close]``. The experience line is
  ``(k_obs - k_exp) x unit_lrc[em_close]`` -- the LRC is linear in the
  in-force, so chaining telescopes WITHOUT any carried state (stronger than
  the GMM's on-track-only telescoping).
* The LIC block is entirely expected-scale (k_exp): incurred claims are past
  events, not in-force -- re-scaling them by the closing count would be
  meaningless. settlement_pattern books are ACCEPTED (the OPEN-3 decision),
  unlike gmm/vfa.settle which reject them.
* Paragraph-55(b) items zeroed by documented engine cuts: acquisition cash
  flows and their amortisation (paragraph 59(a) expensed-as-incurred), the
  financing adjustment (paragraph 56 undiscounted), investment components (v1).
* No CSM block, no finance_wedge (no two-rate gap without discounting), no
  coverage units (B119 is the CSM release denominator -- PAA has none).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from conftest import PATTERNS, make_death_basis

settle = getattr(fcf.paa, "settle", None)
pytestmark = pytest.mark.skipif(
    settle is None,
    reason="paa.settle not implemented yet (redesign step 5; skeleton "
           "activates unchanged once it lands)")


def _basis(**overrides):
    kw = dict(mortality_q=0.0, lapse_q=0.0, discount_annual=0.0,
              ra_confidence=0.75, mortality_cv=0.10)
    kw.update(overrides)
    return make_death_basis(**kw)


# The two hand-calc books (dev/paa-settle-contract.md Sec. 6): a profitable
# single-premium accident book and a claims-heavy onerous one.
PROFITABLE = dict(premium=120.0, benefit=480.0)
ONEROUS = dict(premium=60.0, benefit=6000.0)


def _book(*, premium, benefit, em_close=6, count=1.0, prior_count=1.0,
          term=12, n=1):
    """A 12-month single-premium book seated at em_close."""
    ids = np.array([f"PA{i}" for i in range(n)])
    rep = lambda v: np.full(n, v)
    mp = ModelPoints(
        issue_age=rep(40).astype(np.int64), premium=rep(premium),
        term_months=rep(term).astype(np.int64),
        premium_term_months=rep(1).astype(np.int64),
        benefits={"DEATH": rep(benefit)}, count=rep(float(count)),
        elapsed_months=rep(em_close).astype(np.int64), mp_id=ids,
        product=np.full(n, "ACC"), calculation_methods=PATTERNS,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=rep(em_close).astype(np.int64),
        count=rep(float(count)), prior_csm=rep(0.0), lock_in_rate=0.0,
        prior_count=rep(float(prior_count)),
    )
    return mp, state


def _unit_inforce(basis, *, premium, benefit, term=12):
    unit = ModelPoints(
        issue_age=np.array([40]), premium=np.array([premium]),
        term_months=np.array([term]), premium_term_months=np.array([1]),
        benefits={"DEATH": np.array([benefit])}, count=np.array([1.0]),
        calculation_methods=PATTERNS,
    )
    return fcf.paa.measure(unit, basis, full=True).cashflows.inforce[0]


def _assert_blocks(mv):
    np.testing.assert_allclose(
        mv.lrc_opening + mv.premiums - mv.revenue + mv.lrc_experience,
        mv.lrc_closing, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        mv.loss_component_opening + mv.loss_component_recognised
        - mv.loss_component_reversed, mv.loss_component_closing,
        rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        mv.lic_opening + mv.claims_incurred + mv.lic_finance - mv.claims_paid,
        mv.lic_closing, rtol=1e-10, atol=1e-12)
    # the recalculation form makes the LC rows mutually exclusive
    assert not np.any((mv.loss_component_recognised > 0)
                      & (mv.loss_component_reversed > 0))


# ---------------------------------------------------------------------------
# paragraph-55(b) hand-calc -- contract Sec. 6 (1), analytic
# ---------------------------------------------------------------------------

def test_55b_roll_hand_calc_on_track():
    """Single premium 120, term 12, no decrements: the LRC is the textbook
    pro-rata UPR. em 3 -> 6: 90 + 0 - 30 + 0 == 60, every line analytic."""
    mp, state = _book(**PROFITABLE)
    mv = settle(mp, state, _basis(), period_months=3)
    np.testing.assert_allclose(mv.lrc_opening, [90.0], rtol=1e-12)
    np.testing.assert_allclose(mv.premiums, [0.0], atol=1e-12)
    np.testing.assert_allclose(mv.revenue, [30.0], rtol=1e-12)
    np.testing.assert_allclose(mv.lrc_experience, [0.0], atol=1e-10)
    np.testing.assert_allclose(mv.lrc_closing, [60.0], rtol=1e-12)
    # no claims on this book: the LC and LIC blocks are zero throughout
    np.testing.assert_allclose(mv.loss_component_closing, [0.0], atol=1e-12)
    np.testing.assert_allclose(mv.lic_closing, [0.0], atol=1e-12)
    _assert_blocks(mv)
    assert mv.measurement_basis == "settlement"
    assert mv.period_months == 3


def test_55b_zeroed_items_are_documented_cuts():
    """paragraph 55(b)(ii)/(iii)/(iv)/(vi) have no movement lines BY DESIGN:
    acquisition cash flows are expensed (paragraph 59(a)), the LRC is undiscounted
    (paragraph 56), investment components are a v1 cut. The movement carries no
    such fields -- and no CSM block at all."""
    mp, state = _book(**PROFITABLE)
    mv = settle(mp, state, _basis(), period_months=3)
    for absent in ("csm_closing", "csm_release", "csm_accretion",
                   "finance_wedge", "coverage_units_provided",
                   "lrc_interest", "acquisition_amortisation"):
        assert not hasattr(mv, absent), absent


# ---------------------------------------------------------------------------
# count experience -- contract Sec. 6 (2), analytic closed form
# ---------------------------------------------------------------------------

def test_lapse_experience_hand_calc():
    """Flat 1%/month lapse, observed off-track (prior 0.96, closing 0.90).
    Single premium makes the unit paths closed-form: unit_lrc[t] = 120 - 10t,
    unit_inforce[t] = 0.99^t -- the pins are independent of the engine."""
    basis = _basis(lapse_q=0.01)
    mp, state = _book(**PROFITABLE, prior_count=0.96, count=0.90)
    mv = settle(mp, state, basis, period_months=3)
    k_exp = 0.96 / 0.99 ** 3
    k_obs = 0.90 / 0.99 ** 6
    np.testing.assert_allclose(mv.lrc_opening, [k_exp * 90.0], rtol=1e-10)
    np.testing.assert_allclose(mv.revenue, [k_exp * 30.0], rtol=1e-10)
    np.testing.assert_allclose(mv.lrc_experience, [(k_obs - k_exp) * 60.0],
                               rtol=1e-10)
    np.testing.assert_allclose(mv.lrc_closing, [k_obs * 60.0], rtol=1e-10)
    # contract Sec. 6 (2) decimals
    np.testing.assert_allclose(mv.lrc_opening, [89.044717], rtol=1e-6)
    np.testing.assert_allclose(mv.lrc_experience, [-2.006651], rtol=1e-5)
    np.testing.assert_allclose(mv.lrc_closing, [57.356493], rtol=1e-6)
    _assert_blocks(mv)


def test_on_track_experience_is_zero():
    """k_obs == k_exp makes the experience line zero by construction."""
    basis = _basis(lapse_q=0.01)
    surv = _unit_inforce(basis, **PROFITABLE)
    mp, state = _book(**PROFITABLE, prior_count=float(surv[3]),
                      count=float(surv[6]))
    mv = settle(mp, state, basis, period_months=3)
    np.testing.assert_allclose(mv.lrc_experience, [0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# paragraph 57-58 loss component -- contract Sec. 6 (3)
# ---------------------------------------------------------------------------

def test_loss_component_recalculation_hand_calc():
    """Claims-heavy book (premium 60, benefit 6000, q=0.002): the LC is
    recalculated as max(0, FCF_remaining - LRC) at each date. Coverage
    run-off shrinks the remaining FCF faster than the LRC, so the period
    REVERSES loss -- the same economics as the inception model's release."""
    basis = _basis(mortality_q=0.002)
    mp, state = _book(**ONEROUS, prior_count=1.0, count=0.97)
    mv = settle(mp, state, basis, period_months=3)
    np.testing.assert_allclose(mv.lrc_opening, [45.271084], rtol=1e-6)
    np.testing.assert_allclose(mv.revenue, [15.090361], rtol=1e-6)
    np.testing.assert_allclose(mv.lrc_experience, [-0.729065], rtol=1e-5)
    np.testing.assert_allclose(mv.lrc_closing, [29.451657], rtol=1e-6)
    np.testing.assert_allclose(mv.loss_component_opening, [69.095421],
                               rtol=1e-6)
    np.testing.assert_allclose(mv.loss_component_reversed, [24.368203],
                               rtol=1e-6)
    np.testing.assert_allclose(mv.loss_component_recognised, [0.0],
                               atol=1e-12)
    np.testing.assert_allclose(mv.loss_component_closing, [44.727218],
                               rtol=1e-6)
    # no settlement pattern: claims settle when incurred, the LIC stays zero
    np.testing.assert_allclose(mv.claims_incurred, [35.928048], rtol=1e-6)
    np.testing.assert_allclose(mv.claims_paid, mv.claims_incurred, rtol=1e-12)
    np.testing.assert_allclose(mv.lic_closing, [0.0], atol=1e-12)
    _assert_blocks(mv)


def test_inception_anchor_matches_measure_paa():
    """em_open == 0 with prior_count == count(0): the opening LC IS the
    inception onerous test of paa.measure."""
    basis = _basis(mortality_q=0.002)
    surv = _unit_inforce(basis, **ONEROUS)
    mp, state = _book(**ONEROUS, em_close=3, prior_count=1.0,
                      count=float(surv[3]))
    mv = settle(mp, state, basis, period_months=3)
    unit = ModelPoints(
        issue_age=np.array([40]), premium=np.array([60.0]),
        term_months=np.array([12]), premium_term_months=np.array([1]),
        benefits={"DEATH": np.array([6000.0])}, count=np.array([1.0]),
        calculation_methods=PATTERNS,
    )
    inception = fcf.paa.measure(unit, basis)
    np.testing.assert_allclose(mv.loss_component_opening,
                               inception.loss_component, rtol=1e-10)


# ---------------------------------------------------------------------------
# LIC block over a settlement pattern -- contract Sec. 6 (4), OPEN-3
# ---------------------------------------------------------------------------

def test_lic_block_hand_calc_settlement_pattern():
    """settlement_pattern [0.6, 0.4] with 3% scalar discount, on-track. The LIC
    is measured at fulfilment cash flows (discounted PV + RA, like the GMM LIC,
    validated against the kernel in test_gmm_settle_lic.py); claims_incurred /
    claims_paid stay nominal. k_exp == 1 here."""
    basis = _basis(mortality_q=0.002, discount_annual=0.03,
                   settlement_pattern=np.array([0.6, 0.4]))
    surv = _unit_inforce(basis, **ONEROUS)
    mp, state = _book(**ONEROUS, prior_count=float(surv[3]),
                      count=float(surv[6]))
    mv = settle(mp, state, basis, period_months=3)
    np.testing.assert_allclose(mv.lic_opening, [5.103281], rtol=1e-6)
    np.testing.assert_allclose(mv.claims_incurred, [35.712911], rtol=1e-6)
    np.testing.assert_allclose(mv.claims_paid, [35.741538], rtol=1e-6)
    np.testing.assert_allclose(mv.lic_finance, [-0.001931], atol=1e-6)
    np.testing.assert_allclose(mv.lic_closing, [5.072722], rtol=1e-6)
    np.testing.assert_allclose(mv.lrc_experience, [0.0], atol=1e-10)
    # the LIC opening is the discounted PV + RA (z x cv-weighted), built from the
    # unit claim run-off via the discounted settlement kernel (k_exp == 1 here)
    from fastcashflow._numerics import _norm_ppf, _settlement_lic_discounted
    pattern = np.array([0.6, 0.4])
    unit = ModelPoints(
        issue_age=np.array([40]), premium=np.array([60.0]),
        term_months=np.array([12]), premium_term_months=np.array([1]),
        benefits={"DEATH": np.array([6000.0])}, count=np.array([1.0]),
        calculation_methods=PATTERNS)
    cf = fcf.paa.measure(unit, basis, full=True).cashflows
    lic_d = _settlement_lic_discounted(cf.mortality_cf, pattern, basis.discount_monthly)
    lic_m = _settlement_lic_discounted(cf.morbidity_cf, pattern, basis.discount_monthly)
    z = _norm_ppf(basis.ra_confidence)
    lic_ra = z * (basis.mortality_cv * lic_d + basis.morbidity_cv * lic_m)
    # k_exp = prior_count / unit_inforce[em_open] = surv[3] / surv[3] = 1
    np.testing.assert_allclose(
        mv.lic_opening[0], (lic_d + lic_m + lic_ra)[0][3], rtol=1e-6)
    # the LC recalculation works under the settlement discount too
    np.testing.assert_allclose(mv.loss_component_opening, [67.323850],
                               rtol=1e-6)
    np.testing.assert_allclose(mv.loss_component_closing, [44.931591],
                               rtol=1e-6)
    _assert_blocks(mv)


# ---------------------------------------------------------------------------
# final settlement and chaining -- contract Sec. 6 (5)/(6)
# ---------------------------------------------------------------------------

def test_final_settlement_releases_lrc_and_keeps_the_lic_tail():
    """em_close == boundary with count == 0: the LRC fully releases through
    revenue, the LC fully reverses -- and the LIC tail STAYS OUTSTANDING
    (claims incurred before the boundary are a liability regardless of the
    coverage period ending; the reason paragraph 100(c) is its own table)."""
    basis = _basis(mortality_q=0.002, discount_annual=0.03,
                   settlement_pattern=np.array([0.6, 0.4]))
    surv = _unit_inforce(basis, **ONEROUS)
    mp, state = _book(**ONEROUS, em_close=12, prior_count=float(surv[9]),
                      count=0.0)
    mv = settle(mp, state, basis, period_months=3)
    np.testing.assert_allclose(mv.lrc_opening, [15.0], rtol=1e-6)
    np.testing.assert_allclose(mv.revenue, [15.0], rtol=1e-6)
    np.testing.assert_allclose(mv.lrc_closing, [0.0], atol=1e-10)
    np.testing.assert_allclose(mv.loss_component_reversed, [22.490884],
                               rtol=1e-6)
    np.testing.assert_allclose(mv.loss_component_closing, [0.0], atol=1e-10)
    np.testing.assert_allclose(mv.lic_closing, [5.012153], rtol=1e-6)
    _assert_blocks(mv)


# ---------------------------------------------------------------------------
# pure-LIC-runoff close (opening date at or past the boundary) -- gap (3)
# ---------------------------------------------------------------------------

def _runoff_state(mp, prior_lic, em_close):
    """A runoff InforceState seated at em_close with a zero closing count and a
    carried prior_lic (no in-force past the boundary)."""
    from dataclasses import replace
    ids = mp.mp_id
    mp_r = replace(mp, elapsed_months=np.array([em_close]), count=np.array([0.0]))
    st_r = InforceState(
        mp_id=ids, elapsed_months=np.array([em_close]), count=np.array([0.0]),
        prior_csm=np.array([0.0]), lock_in_rate=0.0,
        prior_count=np.array([0.0]), prior_lic=np.asarray(prior_lic))
    return mp_r, st_r


def test_pure_lic_runoff_past_the_boundary():
    """A settlement period opening at the contract boundary: coverage has ended,
    only the claims tail of already-incurred claims remains. Every in-force-
    scaled line is zero; the carried prior_lic seeds the LIC run-off."""
    basis = _basis(mortality_q=0.01, discount_annual=0.03,
                   settlement_pattern=np.array([0.6, 0.4]))
    surv = _unit_inforce(basis, **ONEROUS)
    # the final coverage settle (em_close == boundary == 12) leaves the tail
    mp_f, st_f = _book(**ONEROUS, em_close=12, prior_count=float(surv[9]),
                       count=0.0)
    mv_f = settle(mp_f, st_f, basis, period_months=3)
    tail = mv_f.lic_closing
    assert np.all(tail > 0.0)                    # month-11 claims' 0.4 still owed
    # a pure-runoff period: em_open == 12 (boundary), em_close == 13
    mp_r, st_r = _runoff_state(mp_f, tail, em_close=13)
    mv = settle(mp_r, st_r, basis, period_months=1)
    # every in-force-scaled line is zero (coverage ended)
    np.testing.assert_array_equal(mv.lrc_opening, 0.0)
    np.testing.assert_array_equal(mv.lrc_closing, 0.0)
    np.testing.assert_array_equal(mv.revenue, 0.0)
    np.testing.assert_array_equal(mv.premiums, 0.0)
    np.testing.assert_array_equal(mv.loss_component_closing, 0.0)
    np.testing.assert_array_equal(mv.claims_incurred, 0.0)
    # the LIC opens at the carried tail (seam identity) and runs off
    np.testing.assert_allclose(mv.lic_opening, tail, rtol=1e-9)
    assert np.all(mv.claims_paid > 0.0)          # the tail pays down
    assert np.all(mv.lic_closing < mv.lic_opening)
    _assert_blocks(mv)


def test_runoff_telescopes_to_a_full_run_off():
    """Settling the whole run-off month by month pays out exactly the carried
    tail and drives the closing LIC to zero (the pattern is fully settled)."""
    basis = _basis(mortality_q=0.01, discount_annual=0.0,   # undiscounted: paid == tail
                   mortality_cv=0.0, settlement_pattern=np.array([0.5, 0.3, 0.2]))
    surv = _unit_inforce(basis, **ONEROUS)
    mp_f, st_f = _book(**ONEROUS, em_close=12, prior_count=float(surv[9]),
                       count=0.0)
    tail = settle(mp_f, st_f, basis, period_months=3).lic_closing
    total_paid = 0.0
    plic = tail
    for em_close in (13, 14, 15):                # run the 3-month tail off
        mp_r, st_r = _runoff_state(mp_f, plic, em_close=em_close)
        mv = settle(mp_r, st_r, basis, period_months=1)
        total_paid += float(mv.claims_paid[0])
        plic = mv.lic_closing
    np.testing.assert_allclose(plic, 0.0, atol=1e-9)          # fully run off
    np.testing.assert_allclose(total_paid, float(tail[0]), rtol=1e-9)


def test_runoff_without_prior_lic_raises():
    basis = _basis(mortality_q=0.01, settlement_pattern=np.array([0.6, 0.4]))
    mp_f, _ = _book(**ONEROUS, em_close=12, prior_count=1.0, count=0.0)
    from dataclasses import replace
    mp_r = replace(mp_f, elapsed_months=np.array([13]), count=np.array([0.0]))
    bare = InforceState(
        mp_id=mp_r.mp_id, elapsed_months=np.array([13]), count=np.array([0.0]),
        prior_csm=np.array([0.0]), lock_in_rate=0.0, prior_count=np.array([0.0]))
    with pytest.raises(ValueError, match="prior_lic"):
        settle(mp_r, bare, basis, period_months=1)


def test_runoff_with_exhausted_schedule_but_positive_prior_lic_raises():
    """A carried prior_lic > 0 opening past the full run-off (the pattern is
    already exhausted) is an inconsistent input -- rejected, not silently
    zeroed."""
    basis = _basis(mortality_q=0.01, settlement_pattern=np.array([0.6, 0.4]))
    mp_f, _ = _book(**ONEROUS, em_close=12, prior_count=1.0, count=0.0)
    # boundary 12, pattern length 2 -> the tail is fully settled by month 13;
    # opening at month 20 with a positive carried tail is inconsistent
    mp_r, st_r = _runoff_state(mp_f, np.array([5.0]), em_close=21)
    with pytest.raises(ValueError, match="run-off schedule"):
        settle(mp_r, st_r, basis, period_months=1)


def test_closing_inputs_carries_prior_lic():
    basis = _basis(mortality_q=0.01, discount_annual=0.03,
                   settlement_pattern=np.array([0.6, 0.4]))
    surv = _unit_inforce(basis, **ONEROUS)
    mp, state = _book(**ONEROUS, em_close=6, prior_count=float(surv[3]),
                      count=float(surv[6]))
    mv = settle(mp, state, basis, period_months=3)
    _, next_state = mv.closing_inputs()
    np.testing.assert_allclose(next_state.prior_lic, mv.lic_closing, rtol=1e-12)


def _chain(basis, surv, grains):
    """Settle em 0 -> 12 in the given grains through closing_inputs(),
    on-track counts from the unit survival (the boundary step closes the
    book: count = 0). Returns the list of movements."""
    from dataclasses import replace
    movements = []
    em = 0
    mp, state = None, None
    for period in grains:
        em_close = em + period
        count = float(surv[em_close]) if em_close < 12 else 0.0
        if mp is None:
            mp, state = _book(**ONEROUS, em_close=em_close,
                              prior_count=1.0, count=count)
        else:
            arr = np.array([count])
            mp = replace(mp, elapsed_months=np.array([em_close]), count=arr)
            state = InforceState(
                mp_id=state.mp_id, elapsed_months=np.array([em_close]),
                count=arr, prior_csm=state.prior_csm,
                lock_in_rate=state.lock_in_rate,
                prior_count=state.prior_count,
                prior_loss_component=state.prior_loss_component)
        mv = settle(mp, state, basis, period_months=period)
        movements.append(mv)
        mp, state = mv.closing_inputs()
        em = em_close
    return movements


def test_chaining_telescopes_without_carried_state():
    """Contract pin 7: 3m x 4 == 6m x 2 == 12m x 1 over the whole life of
    the book (the last step is the final settlement) -- closing balances AND
    every summed line agree across grains. The reconstruction makes
    opening' == closing an identity, not an on-track special case, and the
    LC run-off is monotone on this book so even the recalculated
    recognised / reversed rows sum across grains."""
    basis = _basis(mortality_q=0.002, discount_annual=0.03,
                   settlement_pattern=np.array([0.6, 0.4]))
    surv = _unit_inforce(basis, **ONEROUS)
    q4 = _chain(basis, surv, [3, 3, 3, 3])
    h2 = _chain(basis, surv, [6, 6])
    y1 = _chain(basis, surv, [12])

    # the joint is an identity: opening' == closing at every seam
    for chain in (q4, h2):
        for a, b in zip(chain, chain[1:]):
            np.testing.assert_allclose(b.lrc_opening, a.lrc_closing,
                                       rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(b.lic_opening, a.lic_closing,
                                       rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(b.loss_component_opening,
                                       a.loss_component_closing,
                                       rtol=1e-10, atol=1e-12)

    def footing(chain):
        summed = {
            line: np.sum([getattr(m, line) for m in chain], axis=0)
            for line in ("premiums", "revenue", "lrc_experience",
                         "claims_incurred", "claims_paid", "lic_finance",
                         "loss_component_recognised",
                         "loss_component_reversed")
        }
        for closing in ("lrc_closing", "lic_closing",
                        "loss_component_closing"):
            summed[closing] = getattr(chain[-1], closing)
        return summed

    f4, f2, f1 = footing(q4), footing(h2), footing(y1)
    for line in f1:
        np.testing.assert_allclose(f4[line], f1[line], rtol=1e-10,
                                   atol=1e-12, err_msg=line)
        np.testing.assert_allclose(f2[line], f1[line], rtol=1e-10,
                                   atol=1e-12, err_msg=line)


def test_settle_equals_measure_inforce_lrc():
    """F2-PAA: the closing LRC equals the diagnostic re-based headline --
    the same arithmetic seen from two surfaces (em_close < boundary only;
    measure_inforce rejects the boundary)."""
    basis = _basis(lapse_q=0.01)
    mp, state = _book(**PROFITABLE, prior_count=0.96, count=0.90)
    mv = settle(mp, state, basis, period_months=3)
    diag = fcf.paa.measure_inforce(mp, state, basis)
    np.testing.assert_allclose(mv.lrc_closing, diag.lrc, rtol=1e-12)


# ---------------------------------------------------------------------------
# input guards and parameters
# ---------------------------------------------------------------------------

def test_missing_prior_count_raises():
    mp, state = _book(**PROFITABLE)
    bare = InforceState(
        mp_id=state.mp_id, elapsed_months=state.elapsed_months,
        count=state.count, prior_csm=state.prior_csm, lock_in_rate=0.0)
    with pytest.raises(ValueError, match="prior_count"):
        settle(mp, bare, _basis(), period_months=3)


def test_opening_before_inception_raises():
    mp, state = _book(**PROFITABLE, em_close=2)
    with pytest.raises(ValueError, match="period_months|elapsed"):
        settle(mp, state, _basis(), period_months=3)


def test_pure_lic_runoff_period_rejected_v1():
    """em_open >= boundary (coverage over, only the claims tail left) is a
    v1 cut -- explicit rejection, not a silent zero movement."""
    basis = _basis(mortality_q=0.002, settlement_pattern=np.array([0.6, 0.4]))
    mp, state = _book(**ONEROUS, em_close=15, count=0.0)
    with pytest.raises(ValueError, match="boundary|remaining coverage"):
        settle(mp, state, basis, period_months=3)


def test_revenue_basis_claims_keeps_the_identities():
    basis = _basis(mortality_q=0.002)
    mp, state = _book(**ONEROUS, prior_count=1.0, count=0.97)
    t = settle(mp, state, basis, period_months=3, revenue_basis="time")
    c = settle(mp, state, basis, period_months=3, revenue_basis="claims")
    _assert_blocks(c)
    # the B126(b) weighting front-loads decaying expected claims vs the
    # flat B126(a) line -- the revenue lines must actually differ
    assert not np.allclose(t.revenue, c.revenue)


# ---------------------------------------------------------------------------
# bundled sample (the OPEN-3 reason) and downstream arms
# ---------------------------------------------------------------------------

def test_bundled_paa_sample_settles():
    """The bundled PAA sample carries a settlement_tables sheet; settle
    must accept it (the OPEN-3 decision) so the cookbook runs the sample
    as shipped."""
    basis = fcf.samples.basis("paa")
    mp = fcf.samples.model_points("paa")
    full = fcf.paa.measure(mp, basis, full=True)
    em_open, em_close = 3, 6
    rows = np.arange(mp.n_mp)
    prior = full.cashflows.inforce[rows, em_open]
    close = full.cashflows.inforce[rows, em_close]
    from dataclasses import replace
    mp_c = replace(mp, elapsed_months=np.full(mp.n_mp, em_close),
                   count=close)
    state = InforceState(
        mp_id=mp.mp_id, elapsed_months=np.full(mp.n_mp, em_close),
        count=close, prior_csm=np.zeros(mp.n_mp), lock_in_rate=0.0,
        prior_count=prior)
    mv = settle(mp_c, state, basis, period_months=3)
    _assert_blocks(mv)
    assert np.all(mv.lic_closing > 0)          # the pattern leaves a tail
    # Both contracts are onerous AT INCEPTION, driven by the t=0 acquisition
    # expense. The paragraphs 57-58 re-test measures the REMAINING coverage: with
    # the acquisition outflow behind it, PA001's remaining book is already
    # profitable by month 3 (loss component zero at BOTH dates) while PA002
    # stays onerous -- the re-test discriminates where an inception carry
    # could not.
    assert mv.loss_component_opening[0] == 0.0
    assert mv.loss_component_closing[0] == 0.0
    assert mv.loss_component_closing[1] > 0.0


def test_reconcile_returns_a_footing_settlement_table():
    basis = _basis(mortality_q=0.002)
    mp, state = _book(**ONEROUS, prior_count=1.0, count=0.97)
    mv = settle(mp, state, basis, period_months=3)
    table = fcf.reconcile([mv])
    rec = table[0] if isinstance(table, list) else table
    np.testing.assert_allclose(rec.lrc_closing, float(mv.lrc_closing.sum()),
                               rtol=1e-10)
    # display convention: the draw-down rows are negative in the
    # reconciliation only (the movement keeps them positive)
    np.testing.assert_allclose(rec.revenue, -float(mv.revenue.sum()),
                               rtol=1e-10)
    np.testing.assert_allclose(rec.claims_paid, -float(mv.claims_paid.sum()),
                               rtol=1e-10)
    np.testing.assert_allclose(
        rec.loss_component_reversed,
        -float(mv.loss_component_reversed.sum()), rtol=1e-10)


def test_write_measurement_writes_the_movement_with_markers(tmp_path):
    import polars as pl
    mp, state = _book(**PROFITABLE)
    mv = settle(mp, state, _basis(), period_months=3)
    out = tmp_path / "paa_settle.parquet"
    fcf.write_measurement(mv, out)
    df = pl.read_parquet(out)
    for col in ("lrc_opening", "revenue", "lrc_closing", "lic_closing",
                "loss_component_closing", "measurement_basis"):
        assert col in df.columns
    assert df["measurement_basis"].to_list() == ["settlement"]
