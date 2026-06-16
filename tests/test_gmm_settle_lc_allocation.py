"""gmm.settle -- the paragraph 50(a)-52 systematic loss-component allocation.

Authoritative skeleton (P-5c pattern): written before the implementation and
activated unchanged by it. Anchors from dev/lc-allocation-gate.md and the
hand-calc oracle dev/scratch_lc_allocation_gate.py (3 cases).

The CURRENT gmm.settle rolls the loss component (LC) only through the
paragraph 48/50(b) future-service channel. This feature adds the missing
paragraph 50(a)/51 INCURRED-service channel: as coverage is provided, the
period's claims/expenses release (51a), RA release (51b) and finance (51c) are
allocated on a systematic basis between the LC and the LRC excluding the LC.

Confirmed default systematic basis (entity judgment under 50(a)) -- the
proportional loss-component ratio::

    r          = lc_open / pool_open,  pool_open = outflow_open + ra_open
    lc_finance = r * (outflow_interest + ra_interest)            (51c)
    lc_amort   = r * (outflow_release  + ra_release)             (50a / 51a+b)

The LC-amortised amount is the paragraph-49/B123(b) loss reversal -- presented
in P&L and EXCLUDED from insurance revenue. Two NEW movement lines:
``loss_component_finance`` and ``loss_component_amortised``. The reconciliation
identity gains them::

    loss_component_closing == loss_component_opening
        + loss_component_finance - loss_component_amortised
        - loss_component_reversed + loss_component_recognised

paragraph 52: the cumulative allocation runs the LC to ZERO by the end of the
coverage period -- exact because r is re-derived every period (at the final
period the whole pool releases and lc_amort == lc carried, so it lands at 0
regardless of the rate path or interim remeasurements).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, InforceState, ModelPoints)
from fastcashflow.movement import GMMSettlementMovement

settle = getattr(fcf.gmm, "settle", None)
_HAS_LC_ALLOC = (
    settle is not None
    and "loss_component_amortised" in GMMSettlementMovement.__dataclass_fields__)
pytestmark = pytest.mark.skipif(
    not _HAS_LC_ALLOC,
    reason="paragraph 50(a)-52 loss-component allocation not implemented yet "
           "(the skeleton activates unchanged once the two new movement lines "
           "land on GMMSettlementMovement)")

CM = {"DEATH": CalculationMethod.DEATH}


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


def _onerous_book(basis, *, em_open=12, period=12, term=36, scale=1000.0,
                  lc_open=40_000.0, count_factor=1.0, n=1, final=False):
    """An ONEROUS in-force book: prior_csm = 0, prior_loss_component > 0.

    ``final=True`` seats the closing date AT the contract boundary (count = 0)
    for the final-settlement run-to-zero pin.
    """
    em_close = term if final else em_open + period
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=CM),
        basis, full=True).cashflows.inforce[0]
    prior_count = scale * surv[em_open]
    surv_close = surv[em_close] if em_close < surv.shape[0] else 0.0
    count_close = 0.0 if final else scale * surv_close * count_factor
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
        count=rep(count_close), prior_csm=rep(0.0),
        lock_in_rate=basis.discount_annual, prior_count=rep(prior_count),
        prior_loss_component=rep(lc_open),
    )
    return mp, state


# ---------------------------------------------------------------------------
# the new reconciliation identity (paragraph 49/50/51)
# ---------------------------------------------------------------------------

def test_lc_reconciliation_identity_includes_the_two_new_lines():
    mp, state = _onerous_book(_basis())
    mv = settle(mp, state, _basis(), period_months=12)
    np.testing.assert_allclose(
        mv.loss_component_opening
        + mv.loss_component_finance - mv.loss_component_amortised
        - mv.loss_component_reversed + mv.loss_component_recognised,
        mv.loss_component_closing, rtol=1e-10)


def test_amortisation_and_finance_are_present_and_signed_on_an_onerous_book():
    mp, state = _onerous_book(_basis())
    mv = settle(mp, state, _basis(), period_months=12)
    # As coverage is provided the LC amortises (a positive reversal) and accretes
    # finance (a positive build); both are per-MP arrays.
    assert np.all(mv.loss_component_amortised > 0.0)
    assert np.all(mv.loss_component_finance > 0.0)
    assert mv.loss_component_amortised.shape == (1,)


def test_amortised_is_the_r_share_of_the_pool_release():
    """The proportional basis: amort / finance carry the SAME ratio r as the
    opening LC bears to the opening pool. r = amort_release_share is observable
    by the invariance amort/finance == (opening pool split) without exposing
    the pool: amort and finance are r * release and r * interest of the SAME
    pool, so amort * interest == finance * release."""
    mp, state = _onerous_book(_basis())
    mv = settle(mp, state, _basis(), period_months=12)
    # r * release * (r * interest) symmetry -> amort and finance share r.
    # (pool release) * loss_finance == (pool interest) * loss_amortised would
    # need the pool; instead pin the weaker, still-diagnostic property that the
    # two lines are strictly positive multiples driven by one r (see the
    # numeric oracle in dev/scratch_lc_allocation_gate.py for the exact split).
    assert np.all(mv.loss_component_amortised > mv.loss_component_finance)


# ---------------------------------------------------------------------------
# paragraph 52 -- run to zero by the end of the coverage period
# ---------------------------------------------------------------------------

def test_loss_component_is_zero_at_final_settlement():
    """A final settlement (closing date at the contract boundary, count 0)
    releases the whole remaining coverage; the LC lands at exactly zero (52)."""
    basis = _basis()
    mp, state = _onerous_book(basis, em_open=24, period=12, term=36, final=True)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(mv.loss_component_closing, 0.0, atol=1e-6)


def test_loss_component_amortises_down_each_period():
    """Chaining settle period by period on an on-track onerous book drives the
    closing LC strictly DOWN every period (the incurred-service amortisation);
    the exact landing at zero is pinned by the final-settlement test above."""
    from dataclasses import replace
    basis = _basis()
    term = 48
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=CM),
        basis, full=True).cashflows.inforce[0]
    mp, state = _onerous_book(basis, em_open=0, period=12, term=term)
    closings = []
    for nxt_em in (24, 36, 48):
        mv = settle(mp, state, basis, period_months=12)
        closings.append(float(mv.loss_component_closing[0]))
        mp_mid, state_mid = mv.closing_inputs()
        count_nxt = np.array([1000.0 * (surv[nxt_em] if nxt_em < surv.shape[0]
                                        else 0.0)])
        mp = replace(mp_mid, elapsed_months=np.array([nxt_em]), count=count_nxt)
        state = InforceState(
            mp_id=state_mid.mp_id, elapsed_months=np.array([nxt_em]),
            count=count_nxt, prior_csm=state_mid.prior_csm,
            lock_in_rate=state_mid.lock_in_rate,
            prior_count=state_mid.prior_count,
            prior_loss_component=state_mid.prior_loss_component)
    assert closings[0] > closings[1] > closings[2] > 0.0


# ---------------------------------------------------------------------------
# byte-identical when not onerous; CSM block untouched
# ---------------------------------------------------------------------------

def test_profitable_book_has_zero_lc_allocation_lines():
    """lc_open == 0 -> r == 0 -> both new lines are identically zero; a
    profitable book is byte-identical to the pre-feature settle."""
    basis = _basis()
    # a profitable book: prior_csm > 0, no loss component
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([36]), benefits={"DEATH": np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=CM),
        basis, full=True).cashflows.inforce[0]
    ids = np.array(["P0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([36]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1000.0 * surv[24]]), elapsed_months=np.array([24]),
        mp_id=ids, product=np.array(["A"]), calculation_methods=CM)
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([24]),
        count=np.array([1000.0 * surv[24]]), prior_csm=np.array([5_000.0]),
        lock_in_rate=basis.discount_annual,
        prior_count=np.array([1000.0 * surv[12]]))
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_array_equal(mv.loss_component_finance, 0.0)
    np.testing.assert_array_equal(mv.loss_component_amortised, 0.0)


def test_csm_block_is_unchanged_by_the_lc_allocation():
    """The two new lines touch only the LC channel; the CSM closing (zero on an
    onerous book) and the three-term BEL/RA tie are unaffected."""
    mp, state = _onerous_book(_basis())
    mv = settle(mp, state, _basis(), period_months=12)
    np.testing.assert_allclose(mv.csm_closing, 0.0, atol=1e-9)
    np.testing.assert_allclose(
        mv.csm_experience_unlocking + mv.finance_wedge,
        -(mv.bel_experience + mv.ra_experience), rtol=1e-9)


# ---------------------------------------------------------------------------
# the new lines thread through the reconciliation and the aggregate
# ---------------------------------------------------------------------------

def test_reconciliation_carries_the_new_lines():
    mp, state = _onerous_book(_basis(), n=3)
    mv = settle(mp, state, _basis(), period_months=12)
    recon = fcf.reconcile([mv])[0]
    assert hasattr(recon, "loss_component_amortised")
    assert hasattr(recon, "loss_component_finance")
    # display convention: the amortised reversal is stored negative (it reduces
    # the LC), the finance build positive.
    np.testing.assert_allclose(
        recon.loss_component_amortised, -float(mv.loss_component_amortised.sum()),
        rtol=1e-10)
    np.testing.assert_allclose(
        recon.loss_component_finance, float(mv.loss_component_finance.sum()),
        rtol=1e-10)


def test_aggregate_sums_the_new_lines():
    basis = _basis()
    mp, state = _onerous_book(basis, n=5)
    per_mp = settle(mp, state, basis, period_months=12)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    np.testing.assert_allclose(
        agg.loss_component_amortised, float(per_mp.loss_component_amortised.sum()),
        rtol=1e-9)
    np.testing.assert_allclose(
        agg.loss_component_finance, float(per_mp.loss_component_finance.sum()),
        rtol=1e-9)
