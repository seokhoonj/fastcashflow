"""``inforce_state.csv`` reader + ``apply_inforce_state`` helper.

Pins the period-close input layer: the per-MP closing state from the prior
reporting period flows in via :func:`read_inforce_state` and is folded into
a fresh :class:`ModelPoints` by :func:`apply_inforce_state`. End-to-end:
the resulting model points + ``prior_csm`` + ``lock_in_rate`` feed
:func:`measure_in_force` to produce a settlement-mode measurement.
"""
import csv
from pathlib import Path

import numpy as np
import pytest

import fastcashflow as fcf


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


def test_read_inforce_state_nonuniform_lock_in_errors(tmp_path: Path):
    """v1 takes a scalar locked-in rate; non-uniform rows error out so the
    detail is not silently dropped."""
    p = tmp_path / "state.csv"
    _write_state(p, [
        ("A", 36, 1.0, 0.0, 0.03),
        ("B", 24, 1.0, 0.0, 0.025),
    ])
    with pytest.raises(NotImplementedError, match="lock_in_rate must be uniform"):
        fcf.read_inforce_state(p)


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
    on a fresh ModelPoints from the policies file -- the join key is row
    position, the user having sorted both files by mp_id upstream."""
    mp = fcf.load_sample_model_points()
    state = fcf.load_sample_inforce_state()
    mp_settled = fcf.apply_inforce_state(mp, state)
    assert np.array_equal(mp_settled.elapsed_months, state.elapsed_months)
    assert np.allclose(mp_settled.count, state.count)
    # other fields preserved
    assert np.array_equal(mp_settled.issue_age, mp.issue_age)
    assert np.array_equal(mp_settled.term_months, mp.term_months)


def test_apply_inforce_state_length_mismatch_errors():
    """A wrong-length state (the two files were not aligned) errors out."""
    mp = fcf.load_sample_model_points()
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
    """Sample policies + sample inforce state + measure_in_force runs end
    to end; the settlement-mode CSM differs from the hypothetical one (or
    we have not actually exercised the carry-forward path)."""
    mp = fcf.load_sample_model_points()
    state = fcf.load_sample_inforce_state()
    mp_settled = fcf.apply_inforce_state(mp, state)

    basis = fcf.load_sample_assumptions()
    asmp = basis[("TERM_LIFE", "FC")]

    mif_hyp = fcf.measure_in_force(mp_settled, asmp)
    mif_set = fcf.measure_in_force(
        mp_settled, asmp,
        prior_csm=state.prior_csm,
        lock_in_rate=state.lock_in_rate,
        period_months=12,
    )
    # The carried-forward CSM trajectory differs from the hypothetical one
    # (different starting point at t = elapsed - 12).
    assert not np.allclose(mif_hyp.csm, mif_set.csm)
