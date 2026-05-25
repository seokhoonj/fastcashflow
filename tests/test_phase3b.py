"""Phase 3b validation -- polars file I/O for model points and results."""
import numpy as np
import polars as pl
import pytest

from fastcashflow import (
    Assumptions,
    ModelPoints,
    load_sample_assumptions,
    load_sample_model_points,
    read_model_points,
    sample_data_dir,
    value,
    value_file,
    write_valuation,
)


def _annual(m):
    """Convert a monthly rate to its annual equivalent (engine converts back)."""
    return 1.0 - (1.0 - m) ** 12


def _portfolio(n: int = 400) -> ModelPoints:
    rng = np.random.default_rng(3)
    return ModelPoints(
        issue_age=rng.integers(25, 60, n),
        death_benefit=rng.integers(10, 100, n) * 1_000_000,
        level_premium=rng.integers(3, 15, n) * 10_000,
        term_months=rng.integers(60, 180, n),
    )


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.001)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.01)),
        discount_annual=0.03,
        alpha_flat=200_000.0,
        gamma_flat=48_000.0,
        expense_inflation=0.02,
        ra_confidence=0.85,
        mortality_cv=0.10,
    )


def _frame(mps: ModelPoints) -> pl.DataFrame:
    return pl.DataFrame({
        "issue_age": mps.issue_age,
        "death_benefit": mps.death_benefit,
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

    loaded = read_model_points(path)
    assert loaded.n_mp == mps.n_mp
    assert np.allclose(loaded.issue_age, mps.issue_age)
    assert np.allclose(loaded.death_benefit, mps.death_benefit)
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
        read_model_points(tmp_path / "with_count.parquet").count, counts
    )

    _frame(mps).write_parquet(tmp_path / "no_count.parquet")
    assert np.allclose(
        read_model_points(tmp_path / "no_count.parquet").count,
        np.ones(mps.n_mp),
    )


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_write_valuation_round_trip(tmp_path, suffix):
    """write_valuation persists the four valuation columns with an id column."""
    mps = _portfolio()
    val = value(mps, _assumptions())
    ids = np.arange(mps.n_mp)

    path = tmp_path / f"out{suffix}"
    write_valuation(val, path, ids=ids)
    df = pl.read_parquet(path) if suffix == ".parquet" else pl.read_csv(path)

    assert df.columns == ["id", "bel", "ra", "csm", "loss_component"]
    assert np.allclose(df["id"].to_numpy(), ids)
    assert np.allclose(df["bel"].to_numpy(), val.bel)
    assert np.allclose(df["ra"].to_numpy(), val.ra)
    assert np.allclose(df["csm"].to_numpy(), val.csm)
    assert np.allclose(df["loss_component"].to_numpy(), val.loss_component)


def test_write_valuation_without_ids(tmp_path):
    """ids are optional -- omitting them writes just the four result columns."""
    val = value(_portfolio(50), _assumptions())
    path = tmp_path / "out.parquet"
    write_valuation(val, path)
    assert pl.read_parquet(path).columns == ["bel", "ra", "csm", "loss_component"]


def test_read_ignores_extra_columns_and_flags_missing(tmp_path):
    """Extra columns are ignored; a missing required column is an error."""
    mps = _portfolio(50)
    full = _frame(mps).with_columns(
        pl.Series("mp_id", np.arange(mps.n_mp))
    )
    full.write_parquet(tmp_path / "full.parquet")
    assert read_model_points(tmp_path / "full.parquet").n_mp == mps.n_mp

    pl.DataFrame({"issue_age": mps.issue_age}).write_parquet(tmp_path / "partial.parquet")
    with pytest.raises(ValueError, match="missing required column"):
        read_model_points(tmp_path / "partial.parquet")


def test_file_workflow_matches_in_memory(tmp_path):
    """A file round-trip produces the same valuation as the in-memory path."""
    mps = _portfolio()
    asmp = _assumptions()
    _frame(mps).write_parquet(tmp_path / "mps.parquet")

    from_file = value(read_model_points(tmp_path / "mps.parquet"), asmp)
    in_memory = value(mps, asmp)

    assert np.allclose(from_file.bel, in_memory.bel)
    assert np.allclose(from_file.ra, in_memory.ra)
    assert np.allclose(from_file.csm, in_memory.csm)
    assert np.allclose(from_file.loss_component, in_memory.loss_component)


def test_value_file_streaming_matches_in_memory(tmp_path):
    """Chunked file-to-file valuation equals the in-memory valuation exactly."""
    rng = np.random.default_rng(5)
    n = 1000
    mps = ModelPoints(
        issue_age=rng.integers(25, 60, n),
        death_benefit=rng.integers(10, 100, n) * 1_000_000,
        level_premium=rng.integers(3, 15, n) * 10_000,
        term_months=rng.integers(60, 180, n),
    )
    asmp = _assumptions()

    in_path = tmp_path / "mps.parquet"
    _frame(mps).with_columns(pl.Series("id", np.arange(n))).write_parquet(in_path)

    out_dir = tmp_path / "results"
    processed = value_file(in_path, out_dir, asmp, chunk_size=300, id_column="id")
    assert processed == n
    assert len(sorted(out_dir.glob("part-*.parquet"))) == 4  # 300+300+300+100

    results = pl.read_parquet(str(out_dir / "part-*.parquet")).sort("id")
    in_memory = value(mps, asmp)
    assert np.array_equal(results["id"].to_numpy(), np.arange(n))
    assert np.array_equal(results["bel"].to_numpy(), in_memory.bel)
    assert np.array_equal(results["ra"].to_numpy(), in_memory.ra)
    assert np.array_equal(results["csm"].to_numpy(), in_memory.csm)
    assert np.array_equal(results["loss_component"].to_numpy(), in_memory.loss_component)


def test_value_file_rejects_existing_output(tmp_path):
    """A directory that already holds part files is rejected."""
    mps = _portfolio(100)
    in_path = tmp_path / "mps.parquet"
    _frame(mps).write_parquet(in_path)
    out_dir = tmp_path / "results"

    value_file(in_path, out_dir, _assumptions())
    with pytest.raises(ValueError, match="already contains part"):
        value_file(in_path, out_dir, _assumptions())


def test_load_sample_data_runs():
    """The bundled sample data loads and values without error."""
    mps = load_sample_model_points()
    asmp = next(iter(load_sample_assumptions().values()))
    assert mps.n_mp > 0
    val = value(mps, asmp)
    assert val.bel.shape == (mps.n_mp,)
    assert val.csm.shape == (mps.n_mp,)


def test_sample_data_dir_exposes_bundled_files():
    """sample_data_dir() points at the directory holding the bundled inputs."""
    d = sample_data_dir()
    assert d.is_dir()
    names = {p.name for p in d.iterdir()}
    assert {"sample_assumptions.xlsx",
            "sample_policies.csv",
            "sample_coverages.csv"}.issubset(names)


def test_describe_assumptions_renders_both_shapes(capsys):
    """describe_assumptions prints a tree for an Assumptions and for a dict."""
    from fastcashflow import describe_assumptions
    basis = load_sample_assumptions()
    asmp = next(iter(basis.values()))

    describe_assumptions(asmp)
    out_one = capsys.readouterr().out
    assert out_one.startswith("Assumptions")
    assert "상태 전이율" in out_one
    assert "state_model" in out_one
    assert "coverages" in out_one

    describe_assumptions(basis)
    out_dict = capsys.readouterr().out
    assert "(7 segments)" in out_dict
    # every segment unfolded -- both ('TERM_LIFE', 'GA') and ('TERM_LIFE', 'FC') appear
    assert "('TERM_LIFE', 'GA')" in out_dict
    assert "('TERM_LIFE', 'FC')" in out_dict


def test_to_long_round_trips(tmp_path):
    """ModelPoints.to_long written out and re-read reproduces the valuation."""
    asmp = next(iter(load_sample_assumptions().values()))
    mps = load_sample_model_points()
    policies, coverages = mps.to_long(asmp)
    policies.write_csv(tmp_path / "pol.csv")
    coverages.write_csv(tmp_path / "cov.csv")
    back = read_model_points(tmp_path / "pol.csv", asmp,
                             coverages=tmp_path / "cov.csv")
    a, b = value(mps, asmp), value(back, asmp)
    assert np.allclose(a.bel, b.bel)
    assert np.allclose(a.csm, b.csm)


def test_to_wide_round_trips(tmp_path):
    """ModelPoints.to_wide written out and re-read reproduces the valuation."""
    asmp = next(iter(load_sample_assumptions().values()))
    mps = load_sample_model_points()
    mps.to_wide(asmp).write_csv(tmp_path / "wide.csv")
    back = read_model_points(tmp_path / "wide.csv", asmp)
    a, b = value(mps, asmp), value(back, asmp)
    assert np.allclose(a.bel, b.bel)
    assert np.allclose(a.csm, b.csm)


def test_value_file_streams_long_form(tmp_path):
    """value_file streams a long-form policies + coverages pair in chunks."""
    asmp = next(iter(load_sample_assumptions().values()))
    mps = load_sample_model_points()
    policies, coverages = mps.to_long(asmp)
    policies.write_parquet(tmp_path / "pol.parquet")
    coverages.write_parquet(tmp_path / "cov.parquet")

    out_dir = tmp_path / "results"
    processed = value_file(
        tmp_path / "pol.parquet", out_dir, asmp,
        coverages=tmp_path / "cov.parquet", chunk_size=3,
    )
    assert processed == mps.n_mp

    results = pl.read_parquet(str(out_dir / "part-*.parquet")).sort("id")
    in_memory = value(mps, asmp)
    assert np.allclose(results["bel"].to_numpy(), in_memory.bel)
    assert np.allclose(results["csm"].to_numpy(), in_memory.csm)
