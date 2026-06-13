"""gmm.recognition_schedule -- the paragraph-109 maturity-band API (skeleton).

Authoritative skeleton (settle-family pattern): written before the
implementation and activated unchanged once it lands. Promotes the
workflow/settlement.md recipe (G4 gate, formula confirmed) into a first-class
function. The anchor facts:

* Paragraph 109 discloses, at the reporting date, WHEN the CSM remaining at
  period end is expected to be recognised in profit or loss, in maturity bands.
  The function allocates the closing CSM (the gmm.settle closing) to bands by
  each contract's forward coverage-unit fraction, so the bands SUM TO the
  closing CSM -- an allocation of the remaining balance, not the accreted
  release (the choice settled by tests/test_paragraph109_bands.py).
* The coverage-unit proxy is the in-force count, undiscounted -- the same proxy
  numerics._csm_kernel uses for the B119 amortisation, so the schedule matches
  the actual release pattern.
* Onerous contracts (csm_closing == 0) and any non-CSM rows contribute nothing.
* band_edges_months is configurable (default the Samsung 4-band axis
  12 / 36 / 60); the bands are [0, e0), [e0, e1), ..., [e_last, end).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from conftest import PATTERNS, make_death_basis

recognition_schedule = getattr(fcf.gmm, "recognition_schedule", None)
pytestmark = pytest.mark.skipif(
    recognition_schedule is None,
    reason="gmm.recognition_schedule not implemented yet (step 8 v1.1; "
           "skeleton activates unchanged once it lands)")


def _basis():
    return make_death_basis(mortality_q=0.0015, lapse_q=0.004,
                            discount_annual=0.06, ra_confidence=0.75,
                            mortality_cv=0.10)


def _book(*, n=3, elapsed=24, period=12, premium=600.0):
    """A profitable in-force GMM book (csm_closing > 0), prior_csm seeded from
    the inception CSM trajectory so the carry is alive."""
    basis = _basis()
    unit = ModelPoints.single(40, premium, 240, benefits={0: 100_000.0},
                              calculation_methods=PATTERNS)
    m = fcf.gmm.measure(unit, basis, full=True)
    surv = m.cashflows.inforce[0]
    csm_seed = float(m.csm_path[0, elapsed - period])
    ids = np.array([f"P{i}" for i in range(n)])
    scale = np.array([1000.0, 500.0, 2000.0])[:n]
    factor = np.array([0.8, 1.0, 1.3])[:n]
    count = scale * surv[elapsed] * factor
    rep = lambda v: np.full(n, v)
    mp = ModelPoints(
        issue_age=rep(40).astype(np.int64), premium=rep(premium),
        term_months=rep(240).astype(np.int64), benefits={0: rep(100_000.0)},
        count=count, elapsed_months=rep(elapsed).astype(np.int64), mp_id=ids,
        calculation_methods=PATTERNS)
    state = InforceState(
        mp_id=ids, elapsed_months=rep(elapsed).astype(np.int64), count=count,
        prior_csm=csm_seed * scale, lock_in_rate=basis.discount_annual,
        prior_count=scale * surv[elapsed - period])
    return mp, state, basis


def _oracle_bands(mp, state, basis, edges, period=12):
    """Independent re-derivation: per MP, allocate csm_closing by the forward
    coverage-unit fraction in each band; sum across MPs."""
    mv = fcf.gmm.settle(mp, state, basis, period_months=period)
    inforce = fcf.gmm.measure(mp, basis, full=True).cashflows.inforce
    em = np.asarray(mp.elapsed_months)
    boundary = np.asarray(mp.contract_boundary_months)
    bounds = (0,) + tuple(edges)
    n_bands = len(bounds)
    band = np.zeros(n_bands)
    for i in range(mp.n_mp):
        csm_i = float(mv.csm_closing[i])
        if csm_i <= 0.0:
            continue
        cu = inforce[i, em[i]:boundary[i]]
        total = cu.sum()
        for b in range(n_bands):
            lo = bounds[b]
            hi = bounds[b + 1] if b + 1 < n_bands else len(cu)
            band[b] += csm_i * cu[lo:hi].sum() / total
    return band


# ---------------------------------------------------------------------------
# the headline invariant: bands sum to the closing CSM
# ---------------------------------------------------------------------------
def test_bands_sum_to_closing_csm():
    mp, state, basis = _book()
    sched = recognition_schedule(mp, state, basis, period_months=12)
    closing = float(fcf.gmm.settle(mp, state, basis,
                                   period_months=12).csm_closing.sum())
    np.testing.assert_allclose(sched.csm.sum(), closing, rtol=1e-10)
    np.testing.assert_allclose(sched.closing_csm, closing, rtol=1e-10)


def test_matches_the_allocation_oracle():
    mp, state, basis = _book()
    edges = (12, 36, 60)
    sched = recognition_schedule(mp, state, basis, band_edges_months=edges,
                                 period_months=12)
    oracle = _oracle_bands(mp, state, basis, edges)
    np.testing.assert_allclose(sched.csm, oracle, rtol=1e-10)
    assert sched.csm.shape == (4,)               # 3 edges -> 4 bands


# ---------------------------------------------------------------------------
# onerous contracts carry no CSM, so they contribute nothing
# ---------------------------------------------------------------------------
def test_onerous_contracts_contribute_nothing():
    """A mixed book: a profitable contract plus an onerous one (zero premium,
    csm_closing == 0). The bands sum to the profitable contract's CSM alone."""
    mp, state, basis = _book(n=2)
    # make row 1 onerous: zero premium -> loss component, csm_closing == 0
    from dataclasses import replace
    prem = np.asarray(mp.premium, dtype=np.float64).copy()
    prem[1] = 0.0
    mp = replace(mp, premium=prem)
    state = InforceState(
        mp_id=state.mp_id, elapsed_months=state.elapsed_months,
        count=state.count, prior_csm=np.array([state.prior_csm[0], 0.0]),
        lock_in_rate=state.lock_in_rate, prior_count=state.prior_count,
        prior_loss_component=np.array([0.0, 500.0]))
    mv = fcf.gmm.settle(mp, state, basis, period_months=12)
    sched = recognition_schedule(mp, state, basis, period_months=12)
    # only profitable rows feed the schedule
    profitable = float(mv.csm_closing[mv.csm_closing > 0.0].sum())
    np.testing.assert_allclose(sched.csm.sum(), profitable, rtol=1e-10)


# ---------------------------------------------------------------------------
# band edges are configurable; the bands still reconcile to the closing CSM
# ---------------------------------------------------------------------------
def test_band_edges_are_configurable():
    mp, state, basis = _book()
    closing = float(fcf.gmm.settle(mp, state, basis,
                                   period_months=12).csm_closing.sum())
    fine = recognition_schedule(mp, state, basis,
                                band_edges_months=(12, 24, 36, 60, 120),
                                period_months=12)
    assert fine.csm.shape == (6,)                 # 5 edges -> 6 bands
    np.testing.assert_allclose(fine.csm.sum(), closing, rtol=1e-10)
    # a coarser split of the same book regroups the same total
    coarse = recognition_schedule(mp, state, basis, band_edges_months=(60,),
                                  period_months=12)
    assert coarse.csm.shape == (2,)               # 1 edge -> 2 bands
    np.testing.assert_allclose(coarse.csm.sum(), closing, rtol=1e-10)


def test_default_edges_are_the_four_band_axis():
    mp, state, basis = _book()
    sched = recognition_schedule(mp, state, basis, period_months=12)
    assert tuple(sched.band_edges_months) == (12, 36, 60)
    assert sched.csm.shape == (4,)


def test_rejects_bad_band_edges():
    mp, state, basis = _book()
    with pytest.raises(ValueError, match="band_edges|ascending|positive"):
        recognition_schedule(mp, state, basis, band_edges_months=(36, 12),
                             period_months=12)                 # not ascending
    with pytest.raises(ValueError, match="band_edges|ascending|positive"):
        recognition_schedule(mp, state, basis, band_edges_months=(0, 36),
                             period_months=12)                 # non-positive edge


# ---------------------------------------------------------------------------
# a single profitable contract -- the bands sum to its own closing CSM
# ---------------------------------------------------------------------------
def test_single_contract_sums_to_its_csm():
    mp, state, basis = _book(n=1)
    mv = fcf.gmm.settle(mp, state, basis, period_months=12)
    sched = recognition_schedule(mp, state, basis, period_months=12)
    np.testing.assert_allclose(sched.csm.sum(), float(mv.csm_closing[0]),
                               rtol=1e-10)
    assert np.all(sched.csm >= 0.0)
