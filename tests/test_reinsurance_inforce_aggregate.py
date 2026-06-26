"""reinsurance.measure_inforce_aggregate -- the carry bridge (skeleton).

Authoritative skeleton (settle-family pattern): written before the
implementation and activated unchanged once it lands. The anchor facts come
from dev/inforce-redesign-FINAL.md (Sec. 1 signature, Sec. 3 step 7, the
bridge invariants and the A3 / O-3 dispositions):

* It is a BRIDGE, not a settle. The reinsurance leaf has no ``settle`` yet, so
  a carry-based scale variation fills the capability gap. headline-only,
  ``measurement_basis == 'settlement_carry'``; the docstring announces it is
  deprecated once ``reinsurance.settle`` lands (paragraph 66 unlocking and the
  loss-recovery component are absent in this bridge).
* The aggregate is the per-MP ``measure_inforce`` summed over the model-point
  axis: headline ``bel`` / ``ra`` / ``csm`` equal the per-MP sums to rtol
  1e-10. There is no loss component (paragraph 65).
* ``chunk_size`` is a memory knob, never a numbers knob: chunk_size=1 and one
  big chunk agree to machine precision. The period-close state is joined onto
  the model points ONCE before chunking (a shuffled state still aligns).
* zero-count rows are REJECTED -- the bridge is carry-only, so a derecognized
  row (paragraph 76 count=0) belongs to a future ``reinsurance.settle`` and the
  message says so. This diverges from ``gmm.settle``, which handles count=0 as
  normal derecognition.
* An aggregate cannot be chained -- ``closing_inputs()`` raises ValueError.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from conftest import PATTERNS, make_death_basis

measure_inforce_aggregate = getattr(
    fcf.reinsurance, "measure_inforce_aggregate", None)
pytestmark = [
    pytest.mark.skipif(
        measure_inforce_aggregate is None,
        reason="reinsurance.measure_inforce_aggregate not implemented yet "
               "(redesign step 7; skeleton activates unchanged once it lands)"),
    # The carry bridge is deprecated now that reinsurance.settle exists; this
    # file still verifies its behaviour, so silence its own deprecation notice.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

QS = fcf.reinsurance.QuotaShare


def _basis():
    return make_death_basis(mortality_q=0.002, lapse_q=0.005,
                            discount_annual=0.03, ra_confidence=0.75,
                            mortality_cv=0.10)


def _book(*, n=3, elapsed=36, period=12, cession=0.4):
    """A heterogeneous 3-row ceded book valued in force: different scales,
    counts and prior reinsurance CSMs, prior_csm seeded from the inception CSM
    trajectory so the carry is alive. A chunk split crosses unlike rows."""
    basis = _basis()
    treaty = QS(cession)
    unit = ModelPoints.single(40, 80_000.0, 240, benefits={"DEATH": 1e8},
                              calculation_methods=PATTERNS)
    m = fcf.reinsurance.measure(unit, basis, treaty=treaty)
    surv = m.cashflows.inforce[0]
    csm_seed = float(m.csm_path[0, elapsed - period])
    ids = np.array([f"R{i}" for i in range(n)])
    scale = np.array([1000.0, 500.0, 2000.0])[:n]
    factor = np.array([0.8, 1.0, 1.3])[:n]
    count = scale * surv[elapsed] * factor
    rep = lambda v: np.full(n, v)
    mp = ModelPoints(
        issue_age=rep(40).astype(np.int64), premium=rep(80_000.0),
        term_months=rep(240).astype(np.int64), benefits={"DEATH": rep(1e8)},
        count=count, elapsed_months=rep(elapsed).astype(np.int64), mp_id=ids,
        calculation_methods=PATTERNS)
    state = InforceState(
        mp_id=ids, elapsed_months=rep(elapsed).astype(np.int64), count=count,
        prior_csm=csm_seed * scale, lock_in_rate=basis.discount_annual)
    return mp, state, basis, treaty


# ---------------------------------------------------------------------------
# the oracle: aggregate == per-MP measure_inforce totals
# ---------------------------------------------------------------------------
def test_aggregate_equals_per_mp_inforce_sum():
    mp, state, basis, treaty = _book()
    agg = measure_inforce_aggregate(mp, state, basis, treaty=treaty,
                                    period_months=12)
    per = fcf.reinsurance.measure_inforce(mp, state, basis, treaty=treaty,
                                          period_months=12, full=False)
    for name in ("bel", "ra", "csm"):
        np.testing.assert_allclose(getattr(agg, name),
                                   float(getattr(per, name).sum()),
                                   rtol=1e-10, err_msg=name)
    # no loss component on a reinsurance asset (paragraph 65)
    assert not hasattr(agg, "loss_component")


# ---------------------------------------------------------------------------
# chunk_size is a memory knob, never a numbers knob
# ---------------------------------------------------------------------------
def test_chunking_is_a_numerical_noop():
    mp, state, basis, treaty = _book()
    one = measure_inforce_aggregate(mp, state, basis, treaty=treaty,
                                    period_months=12, chunk_size=1)
    big = measure_inforce_aggregate(mp, state, basis, treaty=treaty,
                                    period_months=12)
    for name in ("bel", "ra", "csm"):
        np.testing.assert_allclose(getattr(one, name), getattr(big, name),
                                   rtol=1e-12, err_msg=name)


def test_shuffled_state_joins_by_mp_id_once_before_chunking():
    """The period-close state arrives in its own order; the bridge aligns it to
    the model points ONCE before slicing chunks, or a chunk would pair one
    contract's rows with another's prior CSM."""
    mp, state, basis, treaty = _book()
    perm = np.array([2, 0, 1])
    shuffled = InforceState(
        mp_id=state.mp_id[perm], elapsed_months=state.elapsed_months[perm],
        count=state.count[perm], prior_csm=state.prior_csm[perm],
        lock_in_rate=state.lock_in_rate)
    straight = measure_inforce_aggregate(mp, state, basis, treaty=treaty,
                                         period_months=12, chunk_size=1)
    joined = measure_inforce_aggregate(mp, shuffled, basis, treaty=treaty,
                                       period_months=12, chunk_size=1)
    for name in ("bel", "ra", "csm"):
        np.testing.assert_allclose(getattr(joined, name),
                                   getattr(straight, name),
                                   rtol=1e-12, err_msg=name)


# ---------------------------------------------------------------------------
# zero-count rows are rejected -- the bridge is carry-only (paragraph 76 -> settle)
# ---------------------------------------------------------------------------
def test_rejects_zero_count_rows():
    mp, state, basis, treaty = _book()
    count = np.asarray(mp.count, dtype=np.float64).copy()
    count[1] = 0.0
    from dataclasses import replace
    mp0 = replace(mp, count=count)
    state0 = InforceState(
        mp_id=state.mp_id, elapsed_months=state.elapsed_months, count=count,
        prior_csm=state.prior_csm, lock_in_rate=state.lock_in_rate)
    with pytest.raises(ValueError, match="count|derecogni|settle"):
        measure_inforce_aggregate(mp0, state0, basis, treaty=treaty,
                                  period_months=12)


# ---------------------------------------------------------------------------
# an aggregate is not a chaining citizen
# ---------------------------------------------------------------------------
def test_cannot_be_chained():
    mp, state, basis, treaty = _book()
    agg = measure_inforce_aggregate(mp, state, basis, treaty=treaty,
                                    period_months=12)
    with pytest.raises(ValueError, match="per-MP|per model point|chain"):
        agg.closing_inputs()


# ---------------------------------------------------------------------------
# the scalar block is marked (carry) and dated
# ---------------------------------------------------------------------------
def test_carries_period_and_marker():
    mp, state, basis, treaty = _book()
    agg = measure_inforce_aggregate(mp, state, basis, treaty=treaty,
                                    period_months=12)
    assert agg.period_months == 12
    assert agg.measurement_basis == "settlement_carry"
    assert isinstance(agg.bel, float)


def test_rejects_non_positive_chunk_size():
    mp, state, basis, treaty = _book()
    with pytest.raises(ValueError, match="chunk_size"):
        measure_inforce_aggregate(mp, state, basis, treaty=treaty,
                                  period_months=12, chunk_size=0)
