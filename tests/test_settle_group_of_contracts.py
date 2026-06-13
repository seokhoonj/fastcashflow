"""portfolio.settle_group_of_contracts -- the per-GoC settlement (skeleton).

Authoritative skeleton (P-5c / settle-family pattern): written before the
implementation and activated unchanged once it lands. The anchor facts come
from dev/per-goc-settle-contract.md (the 2v2 panel synthesis in Sec. 10 is the
authoritative revision; O-7 = (beta) and OPEN-2 = (c) signed off 2026-06-12)
and the G2 gate hand-calcs in dev/scratch_goc_settle_handcalc.py (re-run here
through the public entry point; every pinned number below was captured from
that script).

What this entry computes (contract Sec. 2 -- the GoC-grain algebra):
on the per-MP pre-floor lines of ``gmm.settle``, the LINEAR lines are
group-summed (bel / ra blocks, finance_wedge, csm_opening / accretion, the
preserved unlocking line x, loss_component_opening, the coverage units), while
the NON-LINEAR step -- the paragraph-48/50(b) loss-component algebra and the
B119 release -- is applied ONCE at group grain:

    csm_after_g = algebra(sum csm_opening_i + sum accretion_i, sum x_i, sum lc_open_i)
    frac_g      = cu_provided_g / (cu_provided_g + cu_future_g)   (0 denom -> 1)
    csm_release_g = csm_after_g * frac_g

The per-MP ``csm_release`` / ``csm_closing`` / ``loss_component_*`` lines are
DROPPED (no double floor): they have already passed the per-MP floor, so the
group nets within the CSM/LC that the per-MP floor cannot see. The whole point
is the legitimate within-GoC mutualisation: per-MP floor sum != group floor.

Scope (contract Sec. 1, confirmed): v1 is GMM segments only. VFA is v1.1
(the observed account value is per-MP, so the k_obs re-summation needs its own
gate), PAA is rejected on purpose (no CSM / floor, so a per-GoC algebra is
meaningless -- ``paa.settle`` summed by the user's own groupby is enough), and
reinsurance is a stage-7 seam. A mixed book is rejected WHOLE, never partially.

coverage_units is REQUIRED with no default (O-7 confirmed: the B119 benefit-
amount axis is a place the standard leaves to entity judgement, so the library
does not silently pick one). profitability is an explicit grouping key -- never
derived from a carry (Sec. 24 lock at inception); it may ride on an InforceState
column. OPEN-2 = (c): lock_in_rate may be a per-MP array, required uniform
WITHIN each group (a GoC sits in one cohort, so one locked-in rate; B73 leaves
the cohort weighted-average to the user, the engine only checks uniformity).
"""
import numpy as np
import pytest
from dataclasses import replace

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from fastcashflow.basis import BasisRouter

# Contract-first: the per-GoC settle entry does not exist yet. Skip the whole
# module until it lands, so committing this spec keeps the suite green; the
# implementation then activates it unchanged.
import fastcashflow.portfolio as _pf
if not hasattr(_pf, "settle_group_of_contracts"):
    pytest.skip(
        "per-GoC settle (settle_group_of_contracts / GoCSettlement) is a "
        "contract skeleton, not yet implemented (redesign step 6)",
        allow_module_level=True)

from fastcashflow.portfolio import settle_group_of_contracts, GoCSettlement
from fastcashflow.numerics import _paragraph45_csm_algebra
from conftest import PATTERNS, make_death_basis


# ---------------------------------------------------------------------------
# Independent re-derivation of the group-grain algebra (contract Sec. 2).
# This is the oracle: the engine is used only to produce the per-MP pre-floor
# lines; the grouping arithmetic below is derived from paragraphs 48/50(b)/B119.
# ---------------------------------------------------------------------------
def algebra_4850b(accreted, x, lc_open):
    """Paragraphs 48 / 50(b), scalar, derived from the standard text: an
    unfavourable future-service change first exhausts the CSM, the excess is a
    loss; a favourable change first reverses the loss component, the excess
    re-establishes the CSM. Exactly one of recognised / reversed is positive."""
    assert not (accreted > 0 and lc_open > 0)
    balance = accreted - lc_open + x
    csm_after = max(0.0, balance)
    lc_close = max(0.0, -balance)
    lc_recognised = max(0.0, lc_close - lc_open)
    lc_reversed = max(0.0, lc_open - lc_close)
    return csm_after, lc_reversed, lc_recognised, lc_close


def goc_settle(mv, w):
    """The contract Sec. 2 formulas on the per-MP pre-floor lines of a
    GMMSettlementMovement: linear lines group-summed, the algebra and the
    B119 release once at group grain."""
    csm_open = float(mv.csm_opening.sum())
    accretion = float(mv.csm_accretion.sum())
    x = float(mv.csm_experience_unlocking.sum())
    lc_open = float(mv.loss_component_opening.sum())
    cu_p = float((w * mv.coverage_units_provided).sum())
    cu_f = float((w * mv.coverage_units_future).sum())
    csm_after, lc_rev, lc_rec, lc_close = algebra_4850b(
        csm_open + accretion, x, lc_open)
    denom = cu_p + cu_f
    frac = cu_p / denom if denom > 0 else 1.0
    release = csm_after * frac
    return {
        "csm_opening": csm_open, "csm_accretion": accretion,
        "csm_experience_unlocking": x, "loss_component_opening": lc_open,
        "csm_after": csm_after, "loss_component_reversed": lc_rev,
        "loss_component_recognised": lc_rec, "loss_component_closing": lc_close,
        "coverage_units_provided": cu_p, "coverage_units_future": cu_f,
        "frac": frac, "csm_release": release,
        "csm_closing": csm_after - release,
    }


# group-summed linear lines (compared straight to the per-MP sums)
_LINEAR = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "finance_wedge", "csm_opening", "csm_accretion",
    "csm_experience_unlocking", "loss_component_opening",
)
# lines that go through the group-grain algebra / B119 release
_NONLINEAR = (
    "csm_release", "csm_closing", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
)


def assert_group_matches_oracle(goc, mv, w, row=0):
    """Every GoCSettlement line of group ``row`` equals the oracle: linear
    lines are the per-MP sums (coverage-unit lines weighted by ``w``), the
    non-linear lines come from the group-grain algebra."""
    ref = goc_settle(mv, w)
    for name in _LINEAR:
        np.testing.assert_allclose(
            getattr(goc, name)[row], float(getattr(mv, name).sum()),
            rtol=1e-12, err_msg=name)
    for name in ("coverage_units_provided", "coverage_units_future"):
        np.testing.assert_allclose(
            getattr(goc, name)[row], float((w * getattr(mv, name)).sum()),
            rtol=1e-12, err_msg=name)
    for name in _NONLINEAR:
        np.testing.assert_allclose(
            getattr(goc, name)[row], ref[name], rtol=1e-12, atol=1e-12,
            err_msg=name)


# ---------------------------------------------------------------------------
# Fixtures -- mirror the G2 hand-calc book, plus the routing axes the per-GoC
# entry needs (channel for the router key, issue_date for the cohort).
# ---------------------------------------------------------------------------
BASIS = make_death_basis(mortality_q=0.002, lapse_q=0.005,
                         discount_annual=0.05, ra_confidence=0.75,
                         mortality_cv=0.10)
BENEFITS = [10_000.0, 100_000.0]
ROUTER = BasisRouter({("PROT_A", "GA"): BASIS})

# expected unit survival (same decrements for every row): s6=0.958786
_unit = ModelPoints(
    issue_age=np.full(1, 40, dtype=np.int64), premium=np.full(1, 250.0),
    term_months=np.full(1, 120, dtype=np.int64),
    premium_term_months=np.full(1, 120, dtype=np.int64),
    benefits={0: np.array([10_000.0])}, count=np.ones(1),
    product=np.full(1, "PROT_A"), calculation_methods=PATTERNS)
SURV = fcf.gmm.measure(_unit, BASIS, full=True).cashflows.inforce[0]


def book(*, benefits, em_close, counts, prior_counts, prior_csms, prior_lcs,
         premium=250.0, term=120, lock_in_rate=0.03):
    """One product, one cohort, two MPs with a 10x benefit gap (Sec. 9)."""
    n = len(benefits)
    ids = np.array([f"G{i}" for i in range(n)])
    rep = lambda v: np.asarray(v, dtype=np.float64)
    mp = ModelPoints(
        issue_age=np.full(n, 40, dtype=np.int64), premium=np.full(n, premium),
        term_months=np.full(n, term, dtype=np.int64),
        premium_term_months=np.full(n, term, dtype=np.int64),
        benefits={0: rep(benefits)}, count=rep(counts),
        elapsed_months=np.full(n, em_close, dtype=np.int64), mp_id=ids,
        product=np.full(n, "PROT_A"), channel=np.full(n, "GA"),
        issue_date=np.array(["2026-02-01"] * n, dtype="datetime64[D]"),
        calculation_methods=PATTERNS)
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(n, em_close, dtype=np.int64),
        count=rep(counts), prior_csm=rep(prior_csms),
        lock_in_rate=lock_in_rate, prior_count=rep(prior_counts),
        prior_loss_component=rep(prior_lcs))
    return mp, state


def settle_goc(mp, state, *, period, coverage_units="count", **kw):
    """Single-GoC call: one product, one cohort, one profitability class -> one
    group row whose lines must equal the oracle on the whole book."""
    n = mp.n_mp
    kw.setdefault("profitability", np.zeros(n, dtype=np.int64))
    return settle_group_of_contracts(
        mp, state, ROUTER, period_months=period,
        coverage_units=coverage_units, **kw)


# ===========================================================================
# Sec. 9 hand-calc (1): heterogeneous benefit -- per-MP floor sum != group floor
# ===========================================================================
def test_case1_per_mp_floor_sum_differs_from_group_floor():
    """One GoC, MP0 favourable off-track (+8% survivors), MP1 unfavourable
    (-20%) with a thin CSM. The per-MP algebra recognises a loss on MP1 while
    the group nets inside the CSM at group grain -- the within-GoC
    mutualisation the per-MP floor cannot see."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    mv = fcf.gmm.settle(mp, state, BASIS, period_months=6)
    goc = settle_goc(mp, state, period=6)

    assert goc.csm_closing.shape == (1,)                 # one group row
    assert_group_matches_oracle(goc, mv, w=np.ones(2))

    # the pinned discriminating numbers (dev/scratch_goc_settle_handcalc.py)
    per_mp_csm = float(mv.csm_closing.sum())
    per_mp_lc = float(mv.loss_component_closing.sum())
    np.testing.assert_allclose(per_mp_csm, 1202.737476, rtol=1e-6)
    np.testing.assert_allclose(per_mp_lc, 494.503166, rtol=1e-6)
    np.testing.assert_allclose(goc.csm_closing[0], 734.202745, rtol=1e-6)
    np.testing.assert_allclose(goc.csm_release[0], 60.987248, rtol=1e-6)
    np.testing.assert_allclose(goc.loss_component_closing[0], 0.0, atol=1e-9)
    # the per-MP floor sum and the group floor genuinely differ
    assert abs(per_mp_csm - goc.csm_closing[0]) > 1.0
    assert abs(per_mp_lc - goc.loss_component_closing[0]) > 1.0


# ===========================================================================
# Sec. 9 hand-calc (2): weighted coverage units change the release fraction
# ===========================================================================
def test_case2_weighted_units_change_the_release_fraction():
    """A CSM-positive book so the release line is alive. The benefit-weighted
    B119 fraction differs from the 'count' fraction -- the entity's choice of
    coverage_units actually moves the release."""
    counts = [SURV[6] * 1.02, SURV[6] * 0.98]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[600.0, 500.0],
                     prior_lcs=[0.0, 0.0])
    mv = fcf.gmm.settle(mp, state, BASIS, period_months=6)
    w_ben = np.asarray(BENEFITS)

    g_cnt = settle_goc(mp, state, period=6, coverage_units="count")
    g_ben = settle_goc(mp, state, period=6, coverage_units=w_ben)
    assert_group_matches_oracle(g_cnt, mv, w=np.ones(2))
    assert_group_matches_oracle(g_ben, mv, w=w_ben)

    np.testing.assert_allclose(goc_settle(mv, np.ones(2))["frac"],
                               0.0724267658, rtol=1e-8)
    np.testing.assert_allclose(goc_settle(mv, w_ben)["frac"],
                               0.0735430366, rtol=1e-8)
    np.testing.assert_allclose(g_cnt.csm_release[0], 99.435419, rtol=1e-6)
    np.testing.assert_allclose(g_ben.csm_release[0], 100.967959, rtol=1e-6)
    # weighting really moves the number
    assert abs(g_cnt.csm_release[0] - g_ben.csm_release[0]) > 1.0


def test_count_equals_unit_weight():
    """The 'count' string and an all-ones coverage_units array are the same
    degenerate weighting (contract Sec. 2)."""
    counts = [SURV[6] * 1.02, SURV[6] * 0.98]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[600.0, 500.0],
                     prior_lcs=[0.0, 0.0])
    g_str = settle_goc(mp, state, period=6, coverage_units="count")
    g_one = settle_goc(mp, state, period=6, coverage_units=np.ones(2))
    for name in _LINEAR + _NONLINEAR + ("coverage_units_provided",
                                        "coverage_units_future"):
        np.testing.assert_allclose(getattr(g_str, name), getattr(g_one, name),
                                   rtol=1e-12, err_msg=name)


# ===========================================================================
# Sec. 9 hand-calc (3): chain grain diverges; the GoC chain telescopes
# ===========================================================================
def _goc_closing_chain(grains, counts_path, prior_csms, prior_lcs):
    """The official C4 chain: settle -> group closing -> pro-rata allocation
    back to per-MP prior_* (closing-count share) -> next period. Uses the
    GoCSettlement closing_inputs() seed, which the contract Sec. 4 requires to
    return per-MP (ModelPoints, InforceState) by closing-count pro-rata."""
    em = 6
    csm0, lc0 = list(prior_csms), list(prior_lcs)
    mp, state = book(benefits=BENEFITS, em_close=em + grains[0],
                     prior_counts=[1.0, 1.0], counts=counts_path[0],
                     prior_csms=csm0, prior_lcs=lc0)
    goc = settle_goc(mp, state, period=grains[0])
    em += grains[0]
    for grain, counts in zip(grains[1:], counts_path[1:]):
        next_mp, next_state = goc.closing_inputs()
        arr = np.asarray(counts, dtype=np.float64)
        em += grain
        next_mp = replace(next_mp, elapsed_months=np.full(2, em), count=arr)
        next_state = replace(next_state,
                             elapsed_months=np.full(2, em), count=arr)
        goc = settle_goc(next_mp, next_state, period=grain)
    return goc


def test_case3_chain_grain_diverges():
    """Off-track, MP1 deteriorating: the GoC-grain chain (group closing ->
    pro-rata -> next period) and the per-MP chain summed at the end give
    DIFFERENT closings. Telescoping holds only within one grain."""
    counts_q = [[SURV[9] * 1.05, SURV[9] * 0.85],
                [SURV[12] * 1.06, SURV[12] * 0.75]]
    goc = _goc_closing_chain([3, 3], counts_q, [60.0, 12.0], [0.0, 0.0])
    np.testing.assert_allclose(goc.csm_closing[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(goc.loss_component_closing[0], 387.447062,
                               rtol=1e-6)
    # the per-MP chain summed at the end lands elsewhere (290.69 / 694.67) --
    # grains must not be mixed (pinned in the hand-calc script).
    assert abs(goc.loss_component_closing[0] - 694.665856) > 1.0


def test_case3_goc_grain_telescopes_on_track():
    """On-track counts (following unit survival): the GoC-grain chain 3m x 2
    equals the single 6m settle to rtol 1e-9 -- telescoping within one grain."""
    on_q = [[SURV[9] / SURV[6]] * 2, [SURV[12] / SURV[6]] * 2]
    on_h = [[SURV[12] / SURV[6]] * 2]
    q = _goc_closing_chain([3, 3], on_q, [600.0, 500.0], [0.0, 0.0])
    h = _goc_closing_chain([6], on_h, [600.0, 500.0], [0.0, 0.0])
    np.testing.assert_allclose(q.csm_closing[0], h.csm_closing[0], rtol=1e-9)
    np.testing.assert_allclose(q.csm_closing[0], 1032.8016234071, rtol=1e-9)
    np.testing.assert_allclose(q.loss_component_closing[0],
                               h.loss_component_closing[0], atol=1e-9)


# ===========================================================================
# Sec. 9 hand-calc (4): row-heterogeneous GoC -- a derecognized row accelerates
# the release through the B119 units channel, not an immediate full release
# ===========================================================================
def test_case4_derecognized_row_accelerates_via_b119_units():
    """Near break-even premium (95) so the full lapse of row A does not swamp
    the CSM through unlocking. Row A fully lapsed mid-period (count=0, em 9),
    row B exactly on-track (em 6). At per-MP grain A's csm_after releases in
    FULL this period (frac=1); at GROUP grain it stays in the pool and the
    survivor's release fraction is ACCELERATED (0.03719 -> 0.07172) over B's
    remaining life -- the adjudicated B119 group treatment (B1/P0-1)."""
    s_open, s_close = SURV[3], SURV[6]
    mp, state = book(benefits=[50_000.0, 50_000.0], em_close=6,
                     counts=[0.0, 1.0 * s_close / s_open],
                     prior_counts=[1.0, 1.0], prior_csms=[500.0, 80.0],
                     prior_lcs=[0.0, 0.0], premium=95.0)
    mp = replace(mp, elapsed_months=np.array([9, 6], dtype=np.int64))
    state = replace(state, elapsed_months=np.array([9, 6], dtype=np.int64))
    mv = fcf.gmm.settle(mp, state, BASIS, period_months=3)
    goc = settle_goc(mp, state, period=3)
    assert_group_matches_oracle(goc, mv, w=np.ones(2))

    np.testing.assert_allclose(goc.csm_closing[0], 1241.150935, rtol=1e-6)
    np.testing.assert_allclose(goc_settle(mv, np.ones(2))["frac"],
                               0.0717193853, rtol=1e-8)
    # row-A releases in full at per-MP grain (frac=1, the lapsed row)
    np.testing.assert_allclose(float(mv.csm_release[0]), 1256.449459,
                               rtol=1e-6)

    # the survivor-only group has the slower fraction; adding the lapsed row
    # ACCELERATES it (units channel), it does not release A immediately
    mpB, stB = book(benefits=[50_000.0], em_close=6,
                    counts=[1.0 * s_close / s_open], prior_counts=[1.0],
                    prior_csms=[80.0], prior_lcs=[0.0], premium=95.0)
    mvB = fcf.gmm.settle(mpB, stB, BASIS, period_months=3)
    np.testing.assert_allclose(goc_settle(mvB, np.ones(1))["frac"],
                               0.0371934379, rtol=1e-8)
    assert goc_settle(mv, np.ones(2))["frac"] > goc_settle(mvB, np.ones(1))["frac"]


def test_case4_all_dead_group_releases_in_full():
    """Every row derecognized mid-period (count=0): the future units are 0, so
    frac=1 and the whole group CSM is released -- closing 0."""
    s_open, s_close = SURV[3], SURV[6]
    mp, state = book(benefits=[50_000.0, 50_000.0], em_close=6,
                     counts=[0.0, 0.0], prior_counts=[1.0, 1.0],
                     prior_csms=[500.0, 80.0], prior_lcs=[0.0, 0.0],
                     premium=95.0)
    mp = replace(mp, elapsed_months=np.array([9, 6], dtype=np.int64))
    state = replace(state, elapsed_months=np.array([9, 6], dtype=np.int64))
    goc = settle_goc(mp, state, period=3)
    np.testing.assert_allclose(goc_settle(
        fcf.gmm.settle(mp, state, BASIS, period_months=3), np.ones(2))["frac"],
        1.0, atol=1e-12)
    np.testing.assert_allclose(goc.csm_closing[0], 0.0, atol=1e-9)


# ===========================================================================
# Sec. 2 structure: the algebra is the shared helper; the linear lines tie
# ===========================================================================
def test_group_algebra_is_the_shared_paragraph45_helper():
    """The group-grain CSM/LC step is the SAME numerics._paragraph45_csm_algebra
    used by the per-MP gmm.settle and vfa.settle -- not a re-implementation
    (A1 panel pin). Feeding the group-summed inputs through that helper
    reproduces the GoCSettlement non-linear lines."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    mv = fcf.gmm.settle(mp, state, BASIS, period_months=6)
    goc = settle_goc(mp, state, period=6)
    accreted = float((mv.csm_opening + mv.csm_accretion).sum())
    x = float(mv.csm_experience_unlocking.sum())
    lc_open = float(mv.loss_component_opening.sum())
    csm_after, lc_rev, lc_rec, lc_close = _paragraph45_csm_algebra(
        np.array(accreted), np.array(x), np.array(lc_open))
    # the helper's csm_after, before the B119 release, is csm_release + closing
    np.testing.assert_allclose(goc.csm_release[0] + goc.csm_closing[0],
                               float(csm_after), rtol=1e-12)
    np.testing.assert_allclose(goc.loss_component_closing[0], float(lc_close),
                               rtol=1e-12, atol=1e-12)


def test_finance_wedge_and_unlocking_tie_at_group_grain():
    """finance_wedge is a linear group sum (B97(a), outside the CSM block), and
    the three-term GMM cross-tie survives summation:
    sum(x) + sum(finance_wedge) == -(sum(bel_experience) + sum(ra_experience))."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    mv = fcf.gmm.settle(mp, state, BASIS, period_months=6)
    goc = settle_goc(mp, state, period=6)
    np.testing.assert_allclose(goc.finance_wedge[0],
                               float(mv.finance_wedge.sum()), rtol=1e-12)
    np.testing.assert_allclose(
        goc.csm_experience_unlocking[0] + goc.finance_wedge[0],
        -(goc.bel_experience[0] + goc.ra_experience[0]),
        rtol=1e-10, atol=1e-9)


# ===========================================================================
# Degeneracies: single-MP group == per-MP settle; chunk_size is a memory knob
# ===========================================================================
def test_single_mp_group_equals_per_mp_settle():
    """A group of one model point reproduces the per-MP gmm.settle closing
    lines exactly (no netting, frac as computed)."""
    mp, state = book(benefits=[50_000.0], em_close=6, prior_counts=[1.0],
                     counts=[SURV[6] * 0.98], prior_csms=[300.0],
                     prior_lcs=[0.0])
    mv = fcf.gmm.settle(mp, state, BASIS, period_months=6)
    goc = settle_goc(mp, state, period=6)
    np.testing.assert_allclose(goc.csm_closing[0], float(mv.csm_closing[0]),
                               rtol=1e-12)
    np.testing.assert_allclose(goc.csm_release[0], float(mv.csm_release[0]),
                               rtol=1e-12)
    np.testing.assert_allclose(goc.loss_component_closing[0],
                               float(mv.loss_component_closing[0]), atol=1e-12)


def test_chunk_size_is_a_numerical_noop():
    """chunk_size bounds memory, never the numbers: a group that spans chunks
    (chunk_size=1) equals the single-block result. The floor is applied ONCE on
    the fully-accumulated group, never per chunk."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    one = settle_goc(mp, state, period=6, chunk_size=1)
    big = settle_goc(mp, state, period=6, chunk_size=10_000)
    for name in _LINEAR + _NONLINEAR:
        np.testing.assert_allclose(getattr(one, name), getattr(big, name),
                                   rtol=1e-12, err_msg=name)


def test_rejects_non_positive_chunk_size():
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    with pytest.raises(ValueError, match="chunk_size"):
        settle_goc(mp, state, period=6, chunk_size=0)


# ===========================================================================
# Sec. 2: no double floor -- the per-MP floored lines are NOT summed
# ===========================================================================
def test_no_double_floor_uses_pre_floor_lines():
    """The group closing must come from accumulating the PRE-floor lines (x,
    csm_opening, accretion, lc_open) and flooring once -- not from summing the
    per-MP csm_closing / loss_component_closing (which already passed the
    per-MP floor). Case 1 is exactly where the two disagree."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    mv = fcf.gmm.settle(mp, state, BASIS, period_months=6)
    goc = settle_goc(mp, state, period=6)
    # the group closing is NOT the sum of per-MP floored closings
    assert not np.isclose(goc.csm_closing[0], float(mv.csm_closing.sum()))
    # it IS the floor of the accumulated pre-floor balance
    bal = float((mv.csm_opening + mv.csm_accretion
                 + mv.csm_experience_unlocking).sum())
    np.testing.assert_allclose(goc.csm_release[0] + goc.csm_closing[0],
                               max(0.0, bal), rtol=1e-12)


# ===========================================================================
# Sec. 3 / Sec. 10: keying and validation guards
# ===========================================================================
def test_coverage_units_is_required():
    """O-7 confirmed: coverage_units has NO default -- the B119 benefit-amount
    axis is entity judgement, the library will not silently pick one."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    with pytest.raises((TypeError, ValueError), match="coverage_units"):
        settle_group_of_contracts(mp, state, ROUTER, period_months=6,
                                  profitability=np.zeros(2, dtype=np.int64))


def test_profitability_must_be_explicit_not_derived():
    """profitability is an explicit grouping key -- never derived from the
    carry (Sec. 24 lock at inception). Omitting it is rejected, not silently
    classified by an inception-FCF sign."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    with pytest.raises((TypeError, ValueError), match="profitability"):
        settle_group_of_contracts(mp, state, ROUTER, period_months=6,
                                  coverage_units="count")


def test_rejects_missing_issue_date_for_default_cohort():
    """No silent single-cohort fallback (Sec. 22): the default cohort needs
    issue_date; issue_age / term are never a cohort substitute."""
    n = 2
    ids = np.array(["G0", "G1"])
    mp = ModelPoints(
        issue_age=np.full(n, 40, dtype=np.int64), premium=np.full(n, 250.0),
        term_months=np.full(n, 120, dtype=np.int64),
        premium_term_months=np.full(n, 120, dtype=np.int64),
        benefits={0: np.asarray(BENEFITS)}, count=np.array([0.5, 0.5]),
        elapsed_months=np.full(n, 6, dtype=np.int64), mp_id=ids,
        product=np.full(n, "PROT_A"), channel=np.full(n, "GA"),
        calculation_methods=PATTERNS)                       # no issue_date
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(n, 6, dtype=np.int64),
        count=np.array([0.5, 0.5]), prior_csm=np.array([60.0, 12.0]),
        lock_in_rate=0.03, prior_count=np.array([1.0, 1.0]),
        prior_loss_component=np.zeros(2))
    with pytest.raises(ValueError, match="issue_date|cohort"):
        settle_goc(mp, state, period=6)


def test_xor_rejects_csm_and_lc_summing_positive_in_one_group():
    """Within one group, a positive CSM sum AND a positive loss-component sum
    cannot coexist (the algebra precondition) -- reject, do not net silently."""
    counts = [SURV[6] * 1.0, SURV[6] * 1.0]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 0.0],
                     prior_lcs=[0.0, 50.0])                 # csm AND lc in one GoC
    with pytest.raises(ValueError, match="loss component|prior_csm|prior_loss"):
        settle_goc(mp, state, period=6)


def test_per_goc_lock_in_accepts_uniform_array():
    """OPEN-2 = (c): lock_in_rate may be a per-MP array. A GoC sits in one
    cohort, so the array must be uniform WITHIN each group; a uniform array
    equals the scalar result."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0], lock_in_rate=0.03)
    scalar = settle_goc(mp, state, period=6)
    _, state_arr = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                        counts=counts, prior_csms=[60.0, 12.0],
                        prior_lcs=[0.0, 0.0])
    state_arr = replace(state_arr, lock_in_rate=np.array([0.03, 0.03]))
    arr = settle_goc(mp, state_arr, period=6)
    np.testing.assert_allclose(arr.csm_closing, scalar.csm_closing, rtol=1e-12)


def test_rejects_non_uniform_lock_in_within_group():
    """A per-MP lock_in_rate that varies within one group is rejected with B73
    guidance: a cohort has one locked-in rate (the user supplies the cohort
    weighted average; the engine only checks uniformity)."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    state = replace(state, lock_in_rate=np.array([0.03, 0.04]))   # not uniform
    with pytest.raises(ValueError, match="lock|uniform|B73|cohort"):
        settle_goc(mp, state, period=6)


# ===========================================================================
# Sec. 1 / Sec. 10: the GoCSettlement container and its consumers
# ===========================================================================
def test_returns_marked_goc_settlement():
    """The result is a GoCSettlement: group_labels / group_sizes rows, every
    GMM settlement line mirrored as an (n_group,) scalar line, marked
    'settlement' and dated."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    goc = settle_goc(mp, state, period=6)
    assert isinstance(goc, GoCSettlement)
    assert goc.measurement_basis == "settlement"
    assert goc.period_months == 6
    assert len(goc.group_labels) == 1 and int(goc.group_sizes[0]) == 2
    for name in _LINEAR + _NONLINEAR:
        assert getattr(goc, name).shape == (1,)


def test_closing_inputs_seeds_per_mp_pro_rata():
    """Sec. 4 (B2/P0-1): closing_inputs() returns per-MP (ModelPoints,
    InforceState) by closing-count pro-rata of the group closing balances, so
    the official chain can roll forward. A custom allocation overrides it."""
    counts = [SURV[6] * 1.02, SURV[6] * 0.98]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[600.0, 500.0],
                     prior_lcs=[0.0, 0.0])
    goc = settle_goc(mp, state, period=6)
    next_mp, next_state = goc.closing_inputs()
    assert isinstance(next_mp, ModelPoints)
    assert isinstance(next_state, InforceState)
    # the per-MP seed sums back to the group closing (pro-rata is conservative)
    np.testing.assert_allclose(float(next_state.prior_csm.sum()),
                               goc.csm_closing[0], rtol=1e-12)
    # closing-count share is the default allocation
    share = np.asarray(counts) / np.sum(counts)
    np.testing.assert_allclose(next_state.prior_csm,
                               goc.csm_closing[0] * share, rtol=1e-12)
    # an explicit allocation overrides the default
    alloc = np.array([0.7, 0.3])
    _, override = goc.closing_inputs(allocation=alloc)
    np.testing.assert_allclose(override.prior_csm, goc.csm_closing[0] * alloc,
                               rtol=1e-12)


def test_reconcile_arm_negates_run_off_for_display():
    """Sec. 1 (B2/P0-2): fcf.reconcile(GoCSettlement) builds the group-grain
    Sec. 44 table, negating the run-off lines for display exactly as the per-MP
    reconcile does -- the GoCSettlement itself keeps them movement-positive."""
    counts = [SURV[6] * 1.02, SURV[6] * 0.98]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[600.0, 500.0],
                     prior_lcs=[0.0, 0.0])
    goc = settle_goc(mp, state, period=6)
    rec = fcf.reconcile(goc)
    assert goc.csm_release[0] > 0.0
    np.testing.assert_allclose(np.asarray(rec.csm_release).reshape(-1)[0],
                               -goc.csm_release[0], rtol=1e-12)


def test_write_measurement_arm_has_labels_and_marker(tmp_path):
    """Sec. 1 (B2/P0-2): write_measurement(GoCSettlement) writes the group
    labels and the settlement marker, so a disclosure reader can identify the
    grain."""
    import polars as pl
    counts = [SURV[6] * 1.02, SURV[6] * 0.98]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[600.0, 500.0],
                     prior_lcs=[0.0, 0.0])
    goc = settle_goc(mp, state, period=6)
    out = tmp_path / "goc.parquet"
    fcf.write_measurement(goc, out)
    df = pl.read_parquet(out)
    assert df.height == 1
    assert "measurement_basis" in df.columns
    assert df["measurement_basis"][0] == "settlement"


def test_profitability_carried_as_inforce_column():
    """Sec. 10 (B2/P0-4): InforceState carries an optional 'profitability'
    column (the inception-frozen class, Sec. 24), and the entry accepts the
    column name as the profitability key."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    state = replace(state,
                    profitability=np.array(["profitable", "profitable"]))
    by_col = settle_group_of_contracts(
        mp, state, ROUTER, period_months=6, coverage_units="count",
        profitability="profitability")
    by_arr = settle_goc(mp, state, period=6,
                        profitability=np.zeros(2, dtype=np.int64))
    assert by_col.csm_closing.shape == (1,)
    np.testing.assert_allclose(by_col.csm_closing, by_arr.csm_closing,
                               rtol=1e-12)


# ===========================================================================
# Sec. 1: model scope -- mixed book rejected whole; PAA / VFA out of v1
# ===========================================================================
def _two_product_router(model_of_p):
    return BasisRouter(
        {("PROT_A", "GA"): BASIS, ("OTHER", "GA"): BASIS},
        measurement_models={("OTHER", "GA"): model_of_p})


def _two_product_book(n_each=1):
    n = 2 * n_each
    ids = np.array([f"G{i}" for i in range(n)])
    prod = np.array(["PROT_A"] * n_each + ["OTHER"] * n_each)
    mp = ModelPoints(
        issue_age=np.full(n, 40, dtype=np.int64), premium=np.full(n, 250.0),
        term_months=np.full(n, 120, dtype=np.int64),
        premium_term_months=np.full(n, 120, dtype=np.int64),
        benefits={0: np.full(n, 50_000.0)}, count=np.full(n, 0.5),
        account_value=np.zeros(n),
        elapsed_months=np.full(n, 6, dtype=np.int64), mp_id=ids,
        product=prod, channel=np.full(n, "GA"),
        issue_date=np.array(["2026-02-01"] * n, dtype="datetime64[D]"),
        calculation_methods=PATTERNS)
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(n, 6, dtype=np.int64),
        count=np.full(n, 0.5), prior_csm=np.full(n, 100.0), lock_in_rate=0.03,
        prior_count=np.full(n, 1.0), prior_loss_component=np.zeros(n))
    return mp, state


def test_mixed_book_with_paa_is_rejected_whole():
    """Sec. 1 (B2/P0-3): if any used row routes to a non-GMM segment, the whole
    call is rejected (no partial processing) -- the message points to the PAA
    seam and per-model calls."""
    mp, state = _two_product_book()
    router = _two_product_router("PAA")
    with pytest.raises(ValueError, match="PAA|GMM|single model|paa.settle"):
        settle_group_of_contracts(
            mp, state, router, period_months=6, coverage_units="count",
            profitability=np.zeros(mp.n_mp, dtype=np.int64))


def test_pure_paa_book_is_rejected_with_guidance():
    """PAA is rejected on purpose: no CSM / floor, so a per-GoC algebra is
    meaningless -- the message tells the user to sum paa.settle by their own
    groupby instead."""
    n = 2
    ids = np.array(["G0", "G1"])
    router = BasisRouter({("PROT_A", "GA"): BASIS},
                         measurement_models={("PROT_A", "GA"): "PAA"})
    mp = ModelPoints(
        issue_age=np.full(n, 40, dtype=np.int64), premium=np.full(n, 250.0),
        term_months=np.full(n, 120, dtype=np.int64),
        premium_term_months=np.full(n, 120, dtype=np.int64),
        benefits={0: np.full(n, 50_000.0)}, count=np.full(n, 0.5),
        elapsed_months=np.full(n, 6, dtype=np.int64), mp_id=ids,
        product=np.full(n, "PROT_A"), channel=np.full(n, "GA"),
        issue_date=np.array(["2026-02-01"] * n, dtype="datetime64[D]"),
        calculation_methods=PATTERNS)
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(n, 6, dtype=np.int64),
        count=np.full(n, 0.5), prior_csm=np.zeros(n), lock_in_rate=0.03,
        prior_count=np.full(n, 1.0), prior_loss_component=np.zeros(n))
    with pytest.raises(ValueError, match="PAA|paa.settle|GMM"):
        settle_group_of_contracts(
            mp, state, router, period_months=6, coverage_units="count",
            profitability=np.zeros(n, dtype=np.int64))


def test_vfa_book_is_rejected_as_v1_1():
    """VFA is v1.1: the observed account value is per-MP, so the k_obs
    re-summation needs its own gate. The entry rejects a VFA book in v1."""
    mp, state = _two_product_book()
    router = _two_product_router("VFA")
    state = replace(state, account_value=np.zeros(mp.n_mp))
    with pytest.raises(ValueError, match="VFA|v1.1|GMM"):
        settle_group_of_contracts(
            mp, state, router, period_months=6, coverage_units="count",
            profitability=np.zeros(mp.n_mp, dtype=np.int64))


def test_requires_a_basis_router():
    """A single Basis cannot route a book by product -- a BasisRouter is
    required (use group_of_contracts on a single-model measurement otherwise)."""
    counts = [SURV[6] * 1.08, SURV[6] * 0.80]
    mp, state = book(benefits=BENEFITS, em_close=6, prior_counts=[1.0, 1.0],
                     counts=counts, prior_csms=[60.0, 12.0],
                     prior_lcs=[0.0, 0.0])
    with pytest.raises(TypeError, match="BasisRouter"):
        settle_group_of_contracts(mp, state, BASIS, period_months=6,
                                  coverage_units="count",
                                  profitability=np.zeros(2, dtype=np.int64))
