"""Read economic scenarios from a file.

``read_scenarios(path)`` accepts a 2-D table (one row per scenario, one
column per projection month) in parquet / csv / xlsx / feather and
returns the numpy array shape :func:`measure_stochastic` and
:func:`measure_tvog` consume. A single-column file collapses to a 1-D
``(n_scenarios,)`` array of flat-rate scenarios.
"""
import fastcashflow as fcf
import numpy as np
import polars as pl
import pytest

from fastcashflow import read_scenarios, CoverageRate
from fastcashflow.basis import Basis
from fastcashflow.modelpoints import ModelPoints


def _flat_asmp() -> Basis:
    return Basis(
        mortality_annual=lambda s, ia, d: np.full(s.shape, 0.001),
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.02),
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", lambda s, ia, d: np.full(s.shape, 0.001)),),
    )


def test_read_scenarios_parquet_round_trip(tmp_path):
    """Write a wide parquet (n_scenarios x n_time) and read it back."""
    n_scenarios, n_time = 5, 12
    raw = np.linspace(0.01, 0.05, n_scenarios * n_time).reshape(n_scenarios, n_time)
    df = pl.DataFrame({f"m{i:03d}": raw[:, i] for i in range(n_time)})
    p = tmp_path / "scenarios.parquet"
    df.write_parquet(p)

    out = read_scenarios(p)
    assert out.shape == (n_scenarios, n_time)
    assert np.allclose(out, raw)


def test_read_scenarios_csv_round_trip(tmp_path):
    """CSV is supported too."""
    raw = np.array([[0.03, 0.04], [0.02, 0.05]])
    df = pl.DataFrame({"m0": raw[:, 0], "m1": raw[:, 1]})
    p = tmp_path / "scenarios.csv"
    df.write_csv(p)

    out = read_scenarios(p)
    assert out.shape == (2, 2)
    assert np.allclose(out, raw)


def test_read_scenarios_single_column_collapses_to_1d(tmp_path):
    """A one-column file means flat-rate scenarios -- returns a 1-D array."""
    flat = np.array([0.02, 0.03, 0.04, 0.05])
    df = pl.DataFrame({"rate": flat})
    p = tmp_path / "flat.parquet"
    df.write_parquet(p)

    out = read_scenarios(p)
    assert out.shape == (4,)
    assert np.allclose(out, flat)


def test_read_scenarios_feeds_measure_stochastic(tmp_path):
    """End-to-end: file -> read_scenarios -> measure_stochastic."""
    n_scenarios, n_time = 4, 24
    raw = np.array([[0.02] * n_time, [0.03] * n_time,
                    [0.04] * n_time, [0.05] * n_time])
    df = pl.DataFrame({f"m{i:03d}": raw[:, i] for i in range(n_time)})
    p = tmp_path / "scen.parquet"
    df.write_parquet(p)

    basis = _flat_asmp()
    mp = ModelPoints.single(issue_age=40, benefits={0: 1_000.0},
                            premium=10.0, term_months=24, count=1)

    scenarios = read_scenarios(p)
    result = fcf.gmm.stochastic(mp, basis, scenarios)
    assert result.bel.shape == (n_scenarios,)
    # The four scenarios must give four distinct BEL values (the discount
    # curve actually drives the kernel, not silently ignored).
    assert len(np.unique(np.round(result.bel, 6))) == n_scenarios
