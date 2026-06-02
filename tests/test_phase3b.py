"""Phase 3b validation -- polars file I/O for model points and results."""
import fastcashflow as fcf
import numpy as np
import polars as pl
import pytest

from fastcashflow import ExpenseItem, ModelPoints, read_model_points, sample_data_dir, write_measurement
from fastcashflow.gmm import measure
from conftest import (PATTERNS, annual_from_monthly as _annual,
                      make_death_assumptions, mp_to_wide, mp_to_long)


def _portfolio(n: int = 400) -> ModelPoints:
    rng = np.random.default_rng(3)
    return ModelPoints(
        issue_age=rng.integers(25, 60, n),
        benefits={0: rng.integers(10, 100, n) * 1_000_000},
        level_premium=rng.integers(3, 15, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        calculation_methods=PATTERNS,
    )


def _death_benefits(mps: ModelPoints) -> np.ndarray:
    """Reconstruct the per-policy death benefit from the CSR -- the
    post-(B) replacement for the previous ``ModelPoints.death_benefit``
    field. Sums every coverage_amount whose coverage_index is 0 (the
    DEATH coverage in these tests)."""
    mp_of_cov = np.repeat(np.arange(mps.n_mp), np.diff(mps.coverage_offset))
    mask = mps.coverage_index == 0
    return np.bincount(mp_of_cov[mask], weights=mps.coverage_amount[mask],
                       minlength=mps.n_mp)


def _assumptions():
    return make_death_assumptions(
        mortality_q       = 0.001,
        lapse_q           = 0.01,
        discount_annual   = 0.03,
        expense_inflation = 0.02,
        expense_items     = (
            ExpenseItem("acquisition",  "alpha_fixed",    200_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  48_000.0),
        ),
        ra_confidence     = 0.85,
        mortality_cv      = 0.10,
    )


def _frame(mps: ModelPoints) -> pl.DataFrame:
    # Wide form -- one row per policy, one column per coverage. The DEATH
    # coverage in these tests sits at coverage_code "DEATH" (the first and
    # only registered coverage in ``_assumptions()``); its wide column is
    # ``DEATH_benefit`` per the reader convention.
    return pl.DataFrame({
        "issue_age": mps.issue_age,
        "DEATH_benefit": _death_benefits(mps),
        "level_premium": mps.level_premium,
        "term_months": mps.term_months,
    })


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_model_points_round_trip(tmp_path, suffix):
    """read_model_points reconstructs a ModelPoints written to disk."""
    mps = _portfolio()
    path = tmp_path / f"mps{suffix}"
    df = _frame(mps)
    df.write_parquet(path) if suffix == ".parquet" else df.write_csv(path)

    loaded = read_model_points(path, calculation_methods=PATTERNS)
    assert loaded.n_mp == mps.n_mp
    assert np.allclose(loaded.issue_age, mps.issue_age)
    assert np.allclose(_death_benefits(loaded), _death_benefits(mps))
    assert np.allclose(loaded.level_premium, mps.level_premium)
    assert np.allclose(loaded.term_months, mps.term_months)


def test_read_model_points_reads_count(tmp_path):
    """read_model_points reads an optional count column, else defaults to one."""
    mps = _portfolio(50)
    counts = np.arange(1, mps.n_mp + 1, dtype=float)
    _frame(mps).with_columns(pl.Series("count", counts)).write_parquet(
        tmp_path / "with_count.parquet"
    )
    assert np.allclose(
        read_model_points(tmp_path / "with_count.parquet", calculation_methods=PATTERNS).count, counts
    )

    _frame(mps).write_parquet(tmp_path / "no_count.parquet")
    assert np.allclose(
        read_model_points(tmp_path / "no_count.parquet", calculation_methods=PATTERNS).count,
        np.ones(mps.n_mp),
    )


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_write_measurement_round_trip(tmp_path, suffix):
    """write_measurement persists the four valuation columns with an id column."""
    mps = _portfolio()
    val = measure(mps, _assumptions(), full=False)
    ids = np.arange(mps.n_mp)

    path = tmp_path / f"out{suffix}"
    write_measurement(val, path, ids=ids)
    df = pl.read_parquet(path) if suffix == ".parquet" else pl.read_csv(path)

    assert df.columns == ["id", "bel", "ra", "csm", "loss_component"]
    assert np.allclose(df["id"].to_numpy(), ids)
    assert np.allclose(df["bel"].to_numpy(), val.bel)
    assert np.allclose(df["ra"].to_numpy(), val.ra)
    assert np.allclose(df["csm"].to_numpy(), val.csm)
    assert np.allclose(df["loss_component"].to_numpy(), val.loss_component)


def test_write_measurement_without_ids(tmp_path):
    """ids are optional -- omitting them writes just the four result columns."""
    val = measure(_portfolio(50), _assumptions(), full=False)
    path = tmp_path / "out.parquet"
    write_measurement(val, path)
    assert pl.read_parquet(path).columns == ["bel", "ra", "csm", "loss_component"]


def test_write_measurement_dispatches_per_model(tmp_path):
    """write_measurement is singledispatch: PAA writes lrc, VFA adds
    variable_fee / time_value, and an unsupported type errors loudly."""
    import fastcashflow as fcf
    import numpy as np
    mp = fcf.samples.model_points()
    b = fcf.samples.basis()[("HEALTH_A", "FC")]
    idx = np.where((np.asarray(mp.product_code) == "HEALTH_A") &
                   (np.asarray(mp.channel_code) == "FC"))[0]
    sub = mp.subset(idx)

    fcf.write_measurement(fcf.paa.measure(sub, b), tmp_path / "paa.parquet")
    assert pl.read_parquet(tmp_path / "paa.parquet").columns == [
        "lrc", "loss_component"]

    vmp = fcf.samples.model_points(kind="vfa")
    vb = fcf.samples.basis(kind="vfa")
    fcf.write_measurement(fcf.vfa.measure(vmp, vb), tmp_path / "vfa.parquet")
    assert pl.read_parquet(tmp_path / "vfa.parquet").columns == [
        "bel", "ra", "csm", "variable_fee", "time_value", "loss_component"]

    with pytest.raises(TypeError, match="does not handle"):
        fcf.write_measurement("not a measurement", tmp_path / "x.csv")


def test_read_ignores_extra_columns_and_flags_missing(tmp_path):
    """Extra columns are ignored; a missing required column is an error."""
    mps = _portfolio(50)
    full = _frame(mps).with_columns(
        pl.Series("mp_id", np.arange(mps.n_mp))
    )
    full.write_parquet(tmp_path / "full.parquet")
    assert read_model_points(
        tmp_path / "full.parquet"
    ).n_mp == mps.n_mp

    pl.DataFrame({"issue_age": mps.issue_age}).write_parquet(tmp_path / "partial.parquet")
    with pytest.raises(ValueError, match="missing required column"):
        read_model_points(tmp_path / "partial.parquet")


def test_file_workflow_matches_in_memory(tmp_path):
    """A file round-trip produces the same valuation as the in-memory path."""
    mps = _portfolio()
    basis = _assumptions()
    _frame(mps).write_parquet(tmp_path / "mps.parquet")

    from_file = measure(read_model_points(tmp_path / "mps.parquet", calculation_methods=PATTERNS), basis, full=False)
    in_memory = measure(mps, basis, full=False)

    assert np.allclose(from_file.bel, in_memory.bel)
    assert np.allclose(from_file.ra, in_memory.ra)
    assert np.allclose(from_file.csm, in_memory.csm)
    assert np.allclose(from_file.loss_component, in_memory.loss_component)


def test_measure_stream_streaming_matches_in_memory(tmp_path):
    """Chunked file-to-file valuation equals the in-memory valuation exactly."""
    rng = np.random.default_rng(5)
    n = 1000
    mps = ModelPoints(
        issue_age=rng.integers(25, 60, n),
        benefits={0: rng.integers(10, 100, n) * 1_000_000},
        level_premium=rng.integers(3, 15, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        calculation_methods=PATTERNS,
    )
    basis = _assumptions()

    in_path = tmp_path / "mps.parquet"
    _frame(mps).with_columns(pl.Series("id", np.arange(n))).write_parquet(in_path)

    out_dir = tmp_path / "results"
    processed = fcf.gmm.measure_stream(in_path, out_dir, basis, chunk_size=300, id_column="id",
                           calculation_methods=PATTERNS)
    assert processed == n
    assert len(sorted(out_dir.glob("part-*.parquet"))) == 4  # 300+300+300+100

    results = pl.read_parquet(str(out_dir / "part-*.parquet")).sort("id")
    in_memory = measure(mps, basis, full=False)
    assert np.array_equal(results["id"].to_numpy(), np.arange(n))
    assert np.array_equal(results["bel"].to_numpy(), in_memory.bel)
    assert np.array_equal(results["ra"].to_numpy(), in_memory.ra)
    assert np.array_equal(results["csm"].to_numpy(), in_memory.csm)
    assert np.array_equal(results["loss_component"].to_numpy(), in_memory.loss_component)


def test_measure_stream_rejects_existing_output(tmp_path):
    """A directory that already holds part files is rejected."""
    mps = _portfolio(100)
    in_path = tmp_path / "mps.parquet"
    _frame(mps).write_parquet(in_path)
    out_dir = tmp_path / "results"

    fcf.gmm.measure_stream(in_path, out_dir, _assumptions())
    with pytest.raises(ValueError, match="already contains part"):
        fcf.gmm.measure_stream(in_path, out_dir, _assumptions())


def test_load_sample_data_runs():
    """The bundled sample data loads and values without error."""
    mps = fcf.samples.model_points()
    basis = next(iter(fcf.samples.basis().values()))
    assert mps.n_mp > 0
    val = measure(mps, basis, full=False)
    assert val.bel.shape == (mps.n_mp,)
    assert val.csm.shape == (mps.n_mp,)


def test_sample_data_dir_exposes_bundled_files():
    """sample_data_dir() points at the directory holding the bundled inputs."""
    d = sample_data_dir()
    assert d.is_dir()
    names = {p.name for p in d.iterdir()}
    assert {"sample_basis.xlsx",
            "sample_policies.csv",
            "sample_coverages.csv"}.issubset(names)


def test_describe_basis_renders_both_shapes(capsys):
    """describe_basis prints a tree for an Basis and for a dict."""
    from fastcashflow import describe_basis
    basis = fcf.samples.basis()
    asmp = next(iter(basis.values()))

    describe_basis(asmp)
    out_one = capsys.readouterr().out
    assert out_one.startswith("Basis")
    assert "상태 전이율" in out_one
    assert "state_model" in out_one
    assert "coverages" in out_one

    describe_basis(basis)
    out_dict = capsys.readouterr().out
    assert "(7 segments)" in out_dict
    # every segment unfolded -- both ('TERM_LIFE_A', 'GA') and ('TERM_LIFE_A', 'FC') appear
    assert "('TERM_LIFE_A', 'GA')" in out_dict
    assert "('TERM_LIFE_A', 'FC')" in out_dict


def test_long_form_round_trips(tmp_path):
    """A long-form policies + coverages pair written out and re-read through
    read_model_points reproduces the valuation."""

    basis = next(iter(fcf.samples.basis().values()))
    patterns = fcf.samples.calculation_methods()
    mps = fcf.samples.model_points()
    policies, coverages = mp_to_long(mps, basis)
    policies.write_csv(tmp_path / "pol.csv")
    coverages.write_csv(tmp_path / "cov.csv")
    back = read_model_points(tmp_path / "pol.csv",
                             coverages=tmp_path / "cov.csv",
                             calculation_methods=patterns)
    a, b = measure(mps, basis, full=False), measure(back, basis, full=False)
    assert np.allclose(a.bel, b.bel)
    assert np.allclose(a.csm, b.csm)


def test_wide_form_round_trips(tmp_path):
    """A wide one-row-per-policy frame written out and re-read through
    read_model_points reproduces the valuation."""

    basis = next(iter(fcf.samples.basis().values()))
    patterns = fcf.samples.calculation_methods()
    mps = fcf.samples.model_points()
    mp_to_wide(mps, basis).write_csv(tmp_path / "wide.csv")
    back = read_model_points(tmp_path / "wide.csv",
                             calculation_methods=patterns)
    a, b = measure(mps, basis, full=False), measure(back, basis, full=False)
    assert np.allclose(a.bel, b.bel)
    assert np.allclose(a.csm, b.csm)


def test_measure_stream_streams_long_form(tmp_path):
    """gmm.measure_stream streams a long-form policies + coverages pair in chunks."""
    
    basis = next(iter(fcf.samples.basis().values()))
    patterns = fcf.samples.calculation_methods()
    mps = fcf.samples.model_points()
    policies, coverages = mp_to_long(mps, basis)
    policies.write_parquet(tmp_path / "pol.parquet")
    coverages.write_parquet(tmp_path / "cov.parquet")

    out_dir = tmp_path / "results"
    processed = fcf.gmm.measure_stream(
        tmp_path / "pol.parquet", out_dir, basis,
        coverages=tmp_path / "cov.parquet",
        calculation_methods=patterns, chunk_size=3,
    )
    assert processed == mps.n_mp

    results = pl.read_parquet(str(out_dir / "part-*.parquet")).sort("id")
    in_memory = measure(mps, basis, full=False)
    assert np.allclose(results["bel"].to_numpy(), in_memory.bel)
    assert np.allclose(results["csm"].to_numpy(), in_memory.csm)
