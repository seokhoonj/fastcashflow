"""gmm/vfa.settle_aggregate -- bounded-memory settlement totals (skeleton).

Authoritative skeleton (P-5c pattern): written before the implementation and
activated unchanged by it. The anchor facts, from dev/inforce-redesign-FINAL.md
(stage 3 and the master-invariant appendix):

* The aggregate is the per-MP settlement movement summed over the model-point
  axis, every line kept MOVEMENT-POSITIVE (release / loss-component-reversed
  stay positive run-offs; the display negation happens only in ``reconcile``).
  Oracle: fieldwise equality with the in-memory movement's sums, and
  ``reconcile(aggregate)`` equal to the movement's reconciliation table
  fieldwise.
* ``chunk_size`` is a memory knob, never a numbers knob: chunk_size=1 and one
  big chunk agree to machine precision.
* An aggregate cannot be chained -- ``closing_inputs()`` raises ValueError
  (the per-MP movement is the chaining citizen; FINAL Sec. 1.3).
* The scalar block carries period_months and the settlement marker
  (measurement_basis == 'settlement').
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, InforceState, ModelPoints)

pytestmark = pytest.mark.skipif(
    getattr(fcf.gmm, "settle_aggregate", None) is None
    or getattr(fcf.vfa, "settle_aggregate", None) is None,
    reason="settle_aggregate not implemented yet (redesign stage 3; skeleton "
           "activates unchanged once it lands)")

CM = {"DEATH": CalculationMethod.DEATH}


def _flat(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


# ---------------------------------------------------------------------------
# GMM fixtures -- a 3-row off-track book (heterogeneous balances and shocks)
# ---------------------------------------------------------------------------
def _gmm_basis(*, discount=0.03):
    return Basis(
        mortality_annual=_flat(0.012), lapse_annual=_flat(0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat(0.012)),),
    )


def _gmm_book(basis, *, em_open=12, period=12, term=36):
    """Three rows, deliberately heterogeneous: different sizes, prior CSMs
    and count shocks (favourable / on-track / unfavourable), so every
    movement line is exercised and a chunk split crosses unlike rows."""
    em_close = em_open + period
    unit = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1.0]), calculation_methods=CM,
    )
    surv = fcf.gmm.measure(unit, basis, full=True).cashflows.inforce[0]
    n = 3
    scale = np.array([1000.0, 500.0, 2000.0])
    count_factor = np.array([0.8, 1.0, 1.3])
    prior_count = scale * surv[em_open]
    count_close = scale * surv[em_close] * count_factor
    ids = np.array([f"P{i}" for i in range(n)])
    rep = lambda v: np.full(n, v)
    mp = ModelPoints(
        issue_age=rep(40).astype(np.int64), premium=rep(100.0),
        term_months=rep(term).astype(np.int64), benefits={"DEATH": rep(1e6)},
        count=count_close, elapsed_months=rep(em_close).astype(np.int64),
        mp_id=ids, product=np.full(n, "A"), calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=rep(em_close).astype(np.int64),
        count=count_close, prior_csm=np.array([5_000.0, 0.0, 20_000.0]),
        lock_in_rate=basis.discount_annual,
        prior_count=prior_count,
        prior_loss_component=np.array([0.0, 3_000.0, 0.0]),
    )
    return mp, state


_GMM_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_experience_unlocking",
    "finance_wedge", "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
    "coverage_units_provided", "coverage_units_future",
)


# ---------------------------------------------------------------------------
# VFA fixtures -- a 3-row off-track book (AV and count deviations)
# ---------------------------------------------------------------------------
def _vfa_basis(*, investment_return=0.05, fund_fee=0.015):
    death_fn = _flat(0.012)
    return Basis(
        mortality_annual=death_fn, lapse_annual=_flat(0.05),
        discount_annual=investment_return,   # VFA discounts at the return
        ra_confidence=0.75, mortality_cv=0.0, expense_cv=0.10,
        investment_return=investment_return, fund_fee=fund_fee,
        coverages=(CoverageRate("DEATH", death_fn),),
    )


def _vfa_book(basis, *, em_open=6, period=12, term=36):
    """Three rows with off-track observed account values AND counts."""
    em_close = em_open + period
    n = 3
    rep = lambda v: np.full(n, v)
    ids = np.array([f"P{i}" for i in range(n)])
    av0 = np.array([1e6, 5e5, 2e6])
    mp0 = ModelPoints(
        issue_age=np.array([40, 45, 50], dtype=np.int64), premium=rep(0.0),
        term_months=rep(term).astype(np.int64), benefits={"DEATH": rep(1e6)},
        count=rep(1.0), account_value=av0, mp_id=ids,
        calculation_methods=CM,
    )
    m0 = fcf.vfa.measure(mp0, basis)
    inforce = m0.cashflows.inforce
    rows = np.arange(n)
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0
    growth = (1.0 + r_m) * (1.0 - f_m)
    av_open = av0 * growth ** em_open
    av_close = av_open * growth ** period * np.array([1.10, 1.0, 0.92])
    count_open = inforce[rows, em_open]
    count_close = inforce[rows, em_close] * np.array([1.0, 0.85, 1.2])
    from dataclasses import replace
    mp = replace(mp0, elapsed_months=rep(em_close).astype(np.int64),
                 count=count_close)
    state = InforceState(
        mp_id=ids, elapsed_months=rep(em_close).astype(np.int64),
        count=count_close, prior_csm=m0.csm_path[rows, em_open],
        lock_in_rate=0.0, account_value=av_close,
        prior_count=count_open, prior_account_value=av_open,
    )
    return mp, state


_VFA_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_fv_share", "csm_future_service",
    "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
    "variable_fee_closing", "account_value_closing",
    "coverage_units_provided", "coverage_units_future",
)


# ---------------------------------------------------------------------------
# the oracle: aggregate == movement totals, movement-positive
# ---------------------------------------------------------------------------
def test_gmm_aggregate_equals_the_movement_totals():
    basis = _gmm_basis()
    mp, state = _gmm_book(basis)
    mv = fcf.gmm.settle(mp, state, basis, period_months=12)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    for name in _GMM_LINES:
        np.testing.assert_allclose(
            getattr(agg, name), float(getattr(mv, name).sum()),
            rtol=1e-12, err_msg=name)
    # movement-positive: the run-off lines keep their positive movement sign
    assert agg.csm_release > 0.0
    assert agg.bel_release > 0.0


def test_vfa_aggregate_equals_the_movement_totals():
    basis = _vfa_basis()
    mp, state = _vfa_book(basis)
    mv = fcf.vfa.settle(mp, state, basis, period_months=12)
    agg = fcf.vfa.settle_aggregate(mp, state, basis, period_months=12)
    for name in _VFA_LINES:
        np.testing.assert_allclose(
            getattr(agg, name), float(getattr(mv, name).sum()),
            rtol=1e-12, err_msg=name)
    assert agg.csm_release > 0.0


# ---------------------------------------------------------------------------
# chunk_size is a memory knob, never a numbers knob
# ---------------------------------------------------------------------------
def test_gmm_chunking_is_a_numerical_noop():
    basis = _gmm_basis()
    mp, state = _gmm_book(basis)
    one = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12,
                                   chunk_size=1)
    big = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    for name in _GMM_LINES:
        np.testing.assert_allclose(getattr(one, name), getattr(big, name),
                                   rtol=1e-12, err_msg=name)


def test_vfa_chunking_is_a_numerical_noop():
    basis = _vfa_basis()
    mp, state = _vfa_book(basis)
    one = fcf.vfa.settle_aggregate(mp, state, basis, period_months=12,
                                   chunk_size=1)
    big = fcf.vfa.settle_aggregate(mp, state, basis, period_months=12)
    for name in _VFA_LINES:
        np.testing.assert_allclose(getattr(one, name), getattr(big, name),
                                   rtol=1e-12, err_msg=name)


def test_chunk_size_must_be_positive():
    basis = _gmm_basis()
    mp, state = _gmm_book(basis)
    with pytest.raises(ValueError, match="chunk_size"):
        fcf.gmm.settle_aggregate(mp, state, basis, period_months=12,
                                 chunk_size=0)
    vbasis = _vfa_basis()
    vmp, vstate = _vfa_book(vbasis)
    with pytest.raises(ValueError, match="chunk_size"):
        fcf.vfa.settle_aggregate(vmp, vstate, vbasis, period_months=12,
                                 chunk_size=0)


# ---------------------------------------------------------------------------
# a state file in a different row order joins by mp_id before chunking
# ---------------------------------------------------------------------------
def test_gmm_shuffled_state_joins_by_mp_id_across_chunks():
    """A period-close state file arrives in its own order; the aggregate
    must align it to the model points ONCE before slicing chunks, or a
    chunk would pair one contract's rows with another's prior balances."""
    basis = _gmm_basis()
    mp, state = _gmm_book(basis)
    perm = np.array([2, 0, 1])
    shuffled = InforceState(
        mp_id=state.mp_id[perm], elapsed_months=state.elapsed_months[perm],
        count=state.count[perm], prior_csm=state.prior_csm[perm],
        lock_in_rate=state.lock_in_rate,
        prior_count=state.prior_count[perm],
        prior_loss_component=state.prior_loss_component[perm],
    )
    straight = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12,
                                        chunk_size=1)
    joined = fcf.gmm.settle_aggregate(mp, shuffled, basis, period_months=12,
                                      chunk_size=1)
    for name in _GMM_LINES:
        np.testing.assert_allclose(getattr(joined, name),
                                   getattr(straight, name),
                                   rtol=1e-12, err_msg=name)


# ---------------------------------------------------------------------------
# reconcile equivalence -- the aggregate's table IS the movement's table
# ---------------------------------------------------------------------------
def test_gmm_reconcile_arm_matches_the_movement_table():
    basis = _gmm_basis()
    mp, state = _gmm_book(basis)
    mv = fcf.gmm.settle(mp, state, basis, period_months=12)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    from_mv = fcf.reconcile([mv])[0]
    from_agg = fcf.reconcile(agg)
    assert type(from_agg) is type(from_mv)
    for name in ("period_months", "bel_opening", "bel_interest",
                 "bel_release", "bel_experience", "bel_closing",
                 "ra_opening", "ra_interest", "ra_release", "ra_experience",
                 "ra_closing", "csm_opening", "csm_accretion",
                 "csm_experience_unlocking", "finance_wedge",
                 "loss_component_reversed", "loss_component_recognised",
                 "csm_release", "csm_closing", "loss_component_opening",
                 "loss_component_closing"):
        np.testing.assert_allclose(getattr(from_agg, name),
                                   getattr(from_mv, name),
                                   rtol=1e-12, atol=1e-9, err_msg=name)
    # display convention happens in reconcile, not in the aggregate
    assert from_agg.csm_release < 0.0 < agg.csm_release


def test_vfa_reconcile_arm_matches_the_movement_table():
    basis = _vfa_basis()
    mp, state = _vfa_book(basis)
    mv = fcf.vfa.settle(mp, state, basis, period_months=12)
    agg = fcf.vfa.settle_aggregate(mp, state, basis, period_months=12)
    from_mv = fcf.reconcile([mv])[0]
    from_agg = fcf.reconcile(agg)
    assert type(from_agg) is type(from_mv)
    for name in ("period_months", "bel_opening", "bel_interest",
                 "bel_release", "bel_experience", "bel_closing",
                 "ra_opening", "ra_interest", "ra_release", "ra_experience",
                 "ra_closing", "csm_opening", "csm_accretion", "csm_fv_share",
                 "csm_future_service", "loss_component_reversed",
                 "loss_component_recognised", "csm_release", "csm_closing",
                 "loss_component_opening", "loss_component_closing"):
        np.testing.assert_allclose(getattr(from_agg, name),
                                   getattr(from_mv, name),
                                   rtol=1e-12, atol=1e-9, err_msg=name)
    assert from_agg.csm_release < 0.0 < agg.csm_release


# ---------------------------------------------------------------------------
# an aggregate is not a chaining citizen
# ---------------------------------------------------------------------------
def test_aggregates_cannot_be_chained():
    basis = _gmm_basis()
    mp, state = _gmm_book(basis)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    with pytest.raises(ValueError, match="per-MP|per model point"):
        agg.closing_inputs()
    vbasis = _vfa_basis()
    vmp, vstate = _vfa_book(vbasis)
    vagg = fcf.vfa.settle_aggregate(vmp, vstate, vbasis, period_months=12)
    with pytest.raises(ValueError, match="per-MP|per model point"):
        vagg.closing_inputs()


# ---------------------------------------------------------------------------
# the scalar block is marked and dated
# ---------------------------------------------------------------------------
def test_aggregates_carry_period_and_marker():
    basis = _gmm_basis()
    mp, state = _gmm_book(basis)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    assert agg.period_months == 12
    assert agg.measurement_basis == "settlement"
    assert isinstance(agg.bel_closing, float)
    vbasis = _vfa_basis()
    vmp, vstate = _vfa_book(vbasis)
    vagg = fcf.vfa.settle_aggregate(vmp, vstate, vbasis, period_months=12)
    assert vagg.period_months == 12
    assert vagg.measurement_basis == "settlement"
    assert isinstance(vagg.bel_closing, float)
