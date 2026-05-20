"""Phase 3b validation -- polars file I/O for model points and results."""
import numpy as np
import polars as pl
import pytest

from fastcashflow import (
    Assumptions,
    ModelPointSet,
    read_model_points,
    value,
    write_valuation,
)


def _portfolio(n: int = 400) -> ModelPointSet:
    rng = np.random.default_rng(3)
    return ModelPointSet(
        issue_age=rng.integers(25, 60, n),
        sum_assured=rng.integers(10, 100, n) * 1_000_000,
        monthly_premium=rng.integers(3, 15, n) * 10_000,
        term_months=rng.integers(60, 180, n),
    )


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.001),
        lapse_monthly=lambda duration: np.full(duration.shape, 0.01),
        discount_annual=0.03,
        expense_acquisition=200_000.0,
        expense_maintenance_annual=48_000.0,
        expense_inflation=0.02,
        ra_confidence=0.85,
        claims_cv=0.10,
    )


def _frame(mps: ModelPointSet) -> pl.DataFrame:
    return pl.DataFrame({
        "issue_age": mps.issue_age,
        "sum_assured": mps.sum_assured,
        "monthly_premium": mps.monthly_premium,
        "term_months": mps.term_months,
    })


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_model_points_round_trip(tmp_path, suffix):
    """read_model_points reconstructs a ModelPointSet written to disk."""
    mps = _portfolio()
    path = tmp_path / f"mps{suffix}"
    df = _frame(mps)
    df.write_parquet(path) if suffix == ".parquet" else df.write_csv(path)

    loaded = read_model_points(path)
    assert loaded.n_mp == mps.n_mp
    assert np.allclose(loaded.issue_age, mps.issue_age)
    assert np.allclose(loaded.sum_assured, mps.sum_assured)
    assert np.allclose(loaded.monthly_premium, mps.monthly_premium)
    assert np.allclose(loaded.term_months, mps.term_months)


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
        pl.Series("policy_id", np.arange(mps.n_mp))
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
