"""``inforce_state.csv`` reader + ``apply_inforce_state`` helper.

Pins the period-close input layer: the per-MP closing state from the prior
reporting period flows in via :func:`read_inforce_state` and is folded into
a fresh :class:`ModelPoints` by :func:`apply_inforce_state`. End-to-end:
the resulting model points + ``prior_csm`` + ``lock_in_rate`` feed
:func:`_measure_inforce_full` to produce a settlement-mode measurement.
"""
import csv
from pathlib import Path

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow._measurement.gmm import _measure_inforce_full, _measure_inforce_fast


def _write_state(path: Path, rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mp_id", "elapsed_months", "count", "prior_csm",
                    "lock_in_rate"])
        for r in rows:
            w.writerow(r)


def test_read_inforce_state_roundtrips(tmp_path: Path):
    """The reader parses the schema and reproduces the per-column values."""
    p = tmp_path / "state.csv"
    _write_state(p, [
        ("A", 36, 0.9, 100_000.0, 0.03),
        ("B", 24, 0.95, 50_000.0, 0.03),
    ])
    state = fcf.read_inforce_state(p)
    assert list(state.mp_id) == ["A", "B"]
    assert np.array_equal(state.elapsed_months, np.array([36, 24]))
    assert np.allclose(state.count, [0.9, 0.95])
    assert np.allclose(state.prior_csm, [100_000.0, 50_000.0])
    assert state.lock_in_rate == 0.03


def test_read_inforce_state_nonuniform_lock_in_reads_per_row(tmp_path: Path):
    """A cohort-aware (non-uniform) lock_in_rate column reads into a per-MP array
    (gmm.settle then partitions by rate); a uniform column collapses to a scalar."""
    p = tmp_path / "state.csv"
    _write_state(p, [
        ("A", 36, 1.0, 0.0, 0.03),
        ("B", 24, 1.0, 0.0, 0.025),
    ])
    state = fcf.read_inforce_state(p)
    assert np.allclose(np.asarray(state.lock_in_rate, dtype=float), [0.03, 0.025])


def test_read_inforce_state_missing_column_errors(tmp_path: Path):
    """A schema mistake fails loudly rather than producing a silently
    incomplete state."""
    p = tmp_path / "state.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mp_id", "elapsed_months", "count", "prior_csm"])
        w.writerow(["A", 36, 1.0, 0.0])
    with pytest.raises(ValueError, match="missing required column 'lock_in_rate'"):
        fcf.read_inforce_state(p)


def test_apply_inforce_state_overrides_elapsed_and_count():
    """``apply_inforce_state`` substitutes ``elapsed_months`` and ``count``
    on a fresh ModelPoints from the policies file, joined on mp_id (the
    sample state and policies are in the same order, so it is a no-op here)."""
    mp = fcf.samples.model_points()
    state = fcf.samples.inforce_state()
    mp_settled = fcf.apply_inforce_state(mp, state)
    assert np.array_equal(mp_settled.elapsed_months, state.elapsed_months)
    assert np.allclose(mp_settled.count, state.count)
    # other fields preserved
    assert np.array_equal(mp_settled.issue_age, mp.issue_age)
    assert np.array_equal(mp_settled.term_months, mp.term_months)


def test_apply_inforce_state_length_mismatch_errors():
    """A wrong-length state (the two files were not aligned) errors out."""
    mp = fcf.samples.model_points()
    bad = fcf.InforceState(
        mp_id=np.array(["A", "B"]),
        elapsed_months=np.array([0, 0], dtype=np.int64),
        count=np.array([1.0, 1.0]),
        prior_csm=np.array([0.0, 0.0]),
        lock_in_rate=0.03,
    )
    with pytest.raises(ValueError, match="state has 2 rows"):
        fcf.apply_inforce_state(mp, bad)


def test_sample_inforce_end_to_end():
    """Sample policies + sample inforce state + _measure_inforce_full runs end
    to end; the settlement-mode CSM differs from the hypothetical one (or
    we have not actually exercised the carry-forward path)."""
    mp = fcf.samples.model_points()
    state = fcf.samples.inforce_state()
    mp_settled = fcf.apply_inforce_state(mp, state)

    basis = fcf.samples.basis()
    basis = basis.resolve(("TERM_LIFE_A", "FC"))

    mif_hyp = _measure_inforce_full(mp_settled, basis)
    mif_set = _measure_inforce_full(
        mp_settled, basis,
        prior_csm=state.prior_csm,
        lock_in_rate=state.lock_in_rate,
        period_months=12,
    )
    # The carried-forward CSM trajectory differs from the hypothetical one
    # (different starting point at t = elapsed - 12).
    assert not np.allclose(mif_hyp.csm, mif_set.csm)


def test_apply_inforce_state_joins_on_mp_id():
    """A state in a different mp_id order is reordered to match the model
    points -- the join is by mp_id, not by row position."""
    mp = fcf.samples.model_points()
    ids = mp.mp_id
    n = mp.n_mp
    rev = np.arange(n)[::-1]
    shuffled = fcf.InforceState(
        mp_id=ids[rev].copy(),
        elapsed_months=(np.arange(n) + 1)[rev].astype(np.int64),
        count=np.ones(n),
        prior_csm=np.zeros(n),
        lock_in_rate=0.03,
    )
    settled = fcf.apply_inforce_state(mp, shuffled)
    # reordered back to model-point order: each mp_id keeps its own elapsed
    assert np.array_equal(settled.elapsed_months, np.arange(n) + 1)


def test_apply_inforce_state_rejects_mismatched_mp_id():
    """A state covering different contracts (mp_id sets differ) is rejected
    rather than silently mis-assigned."""
    mp = fcf.samples.model_points()
    n = mp.n_mp
    wrong = fcf.InforceState(
        mp_id=np.array([f"X{i}" for i in range(n)]),
        elapsed_months=np.full(n, 12, dtype=np.int64),
        count=np.ones(n),
        prior_csm=np.zeros(n),
        lock_in_rate=0.03,
    )
    with pytest.raises(ValueError, match="mp_id sets"):
        fcf.apply_inforce_state(mp, wrong)


def test_measure_inforce_warns_surrender_is_sample_grade():
    """measure_inforce warns that the surrender value is sample-grade (no
    surrender table / no pre-valuation premiums) when the basis carries a
    surrender curve and there are in-force (elapsed > 0) contracts. The BEL / RA
    are re-based to the valuation date, so they no longer trigger a warning."""
    state = fcf.samples.inforce_state()
    mp = fcf.apply_inforce_state(fcf.samples.model_points(), state)
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "FC"))
    with pytest.warns(UserWarning, match="surrender"):
        fcf.gmm.measure_inforce(mp, state, basis, full=False)


def test_read_inforce_state_rejects_duplicate_mp_id(tmp_path: Path):
    """mp_id is the join key the state is matched on; a duplicate makes the
    join ambiguous, so the reader (via InforceState) rejects it."""
    p = tmp_path / "dup.csv"
    _write_state(p, [
        ("A", 36, 0.9, 100_000.0, 0.03),
        ("A", 24, 0.95, 50_000.0, 0.03),     # duplicate mp_id
    ])
    with pytest.raises(ValueError, match="mp_id must be unique"):
        fcf.read_inforce_state(p)


def test_inforce_state_rejects_duplicate_mp_id():
    """The InforceState dataclass guards its own identity key."""
    with pytest.raises(ValueError, match="mp_id must be unique"):
        fcf.InforceState(
            mp_id=np.array(["A", "A"]),
            elapsed_months=np.array([6, 6], dtype=np.int64),
            count=np.array([1.0, 1.0]), prior_csm=np.array([0.0, 0.0]),
            lock_in_rate=0.0)


def test_model_points_rejects_duplicate_mp_id():
    """ModelPoints.mp_id is the contract identity; duplicates are rejected so
    the in-force / grouping joins key on a unique id."""
    with pytest.raises(ValueError, match="mp_id must be unique"):
        fcf.ModelPoints(
            mp_id=np.array(["A", "A"]), issue_age=np.array([40, 50]),
            premium=np.array([0.0, 0.0]), term_months=np.array([12, 12]))


def test_mixed_type_mp_id_raises_clear_value_error():
    """A mixed-type mp_id (1 and "1") is str-keyed, so the uniqueness check sees
    a duplicate and raises a clear ValueError -- not a np.unique sort TypeError
    -- on both ModelPoints and InforceState."""
    with pytest.raises(ValueError, match="mp_id must be unique"):
        fcf.ModelPoints(mp_id=np.array([1, "1"], dtype=object),
                        issue_age=np.array([40.0, 50.0]),
                        premium=np.array([0.0, 0.0]),
                        term_months=np.array([12, 12]))
    with pytest.raises(ValueError, match="mp_id must be unique"):
        fcf.InforceState(mp_id=np.array([1, "1"], dtype=object),
                         elapsed_months=np.array([1, 2], dtype=np.int64),
                         count=np.array([1.0, 1.0]),
                         prior_csm=np.array([0.0, 0.0]), lock_in_rate=0.0)
