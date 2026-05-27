"""save_sample_* helpers -- drop the packaged sample files on disk so a
reader's tutorial code can take a real path through read_*. The four
helpers cover the four file types the cookbook / tutorials show
(assumptions workbook, policies, coverages, benefit_patterns).
"""
from pathlib import Path

import fastcashflow as fcf


def test_save_sample_assumptions_round_trips_via_read_assumptions(tmp_path):
    """The dropped workbook reads back through read_assumptions to the
    same dict of Assumptions the in-memory loader produces."""
    path = fcf.save_sample_assumptions(tmp_path / "assumptions.xlsx")
    assert path.exists()
    assert path.suffix == ".xlsx"

    basis_from_file = fcf.read_assumptions(path)
    basis_in_memory = fcf.load_sample_assumptions()
    assert sorted(basis_from_file) == sorted(basis_in_memory)


def test_save_sample_full_round_trip(tmp_path):
    """The four save_* helpers, together with the three read_* arguments,
    reproduce the same ModelPoints as load_sample_model_points."""
    fcf.save_sample_assumptions(tmp_path / "assumptions.xlsx")
    fcf.save_sample_policies(tmp_path / "policies.csv")
    fcf.save_sample_coverages(tmp_path / "coverages.csv")
    fcf.save_sample_benefit_patterns(tmp_path / "benefit_patterns.csv")

    basis = fcf.read_assumptions(tmp_path / "assumptions.xlsx")
    asmp = next(iter(basis.values()))
    mp_file = fcf.read_model_points(
        tmp_path / "policies.csv", asmp,
        coverages=tmp_path / "coverages.csv",
        benefit_patterns=tmp_path / "benefit_patterns.csv",
    )
    mp_mem = fcf.load_sample_model_points()
    assert mp_file.n_mp == mp_mem.n_mp
    assert list(mp_file.product_code) == list(mp_mem.product_code)


def test_save_sample_accepts_directory(tmp_path):
    """Passing a directory writes the file inside with its packaged name."""
    target = fcf.save_sample_assumptions(tmp_path)
    assert target == tmp_path / "sample_assumptions.xlsx"
    assert target.exists()


def test_save_sample_policies_returns_destination_path(tmp_path):
    """The return value points at the file on disk -- the caller can chain
    it straight into read_model_points without restating the path."""
    path = fcf.save_sample_policies(tmp_path / "mp.csv")
    assert isinstance(path, Path)
    assert path == tmp_path / "mp.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("mp_id")


def test_save_sample_converts_to_parquet(tmp_path):
    """A non-csv extension routes through polars and converts the format,
    preserving row count and column order."""
    csv_path = fcf.save_sample_policies(tmp_path / "policies.csv")
    parq_path = fcf.save_sample_policies(tmp_path / "policies.parquet")

    import polars as pl
    csv_df = pl.read_csv(csv_path)
    parq_df = pl.read_parquet(parq_path)
    assert csv_df.shape == parq_df.shape
    assert csv_df.columns == parq_df.columns


def test_save_sample_converts_to_feather(tmp_path):
    """.feather / .arrow extension also routes through polars."""
    path = fcf.save_sample_coverages(tmp_path / "coverages.feather")
    assert path.exists()
    import polars as pl
    df = pl.read_ipc(path)
    assert "mp_id" in df.columns
    assert "coverage_code" in df.columns


def test_save_sample_converts_to_xlsx_single_sheet(tmp_path):
    """The three single-table sample files can land as .xlsx and round-trip
    through read_model_points just like their .csv source."""
    fcf.save_sample_assumptions(tmp_path / "assumptions.xlsx")
    fcf.save_sample_policies(tmp_path / "policies.xlsx")
    fcf.save_sample_coverages(tmp_path / "coverages.xlsx")
    fcf.save_sample_benefit_patterns(tmp_path / "benefit_patterns.xlsx")

    basis = fcf.read_assumptions(tmp_path / "assumptions.xlsx")
    mp = fcf.read_model_points(
        tmp_path / "policies.xlsx",
        basis[("TERM_LIFE_A", "GA")],
        coverages=tmp_path / "coverages.xlsx",
        benefit_patterns=tmp_path / "benefit_patterns.xlsx",
    )
    assert mp.n_mp == fcf.load_sample_model_points().n_mp


def test_save_sample_rejects_unsupported_extension(tmp_path):
    """A path the writer cannot route (no recognised extension) errors
    clearly instead of writing a silently empty file."""
    import pytest
    with pytest.raises(ValueError, match="unsupported file type"):
        fcf.save_sample_benefit_patterns(tmp_path / "bp.json")


def test_save_sample_assumptions_rejects_non_xlsx(tmp_path):
    """The assumptions workbook is multi-sheet -- single-table formats
    cannot represent it. A non-.xlsx path errors clearly."""
    import pytest
    with pytest.raises(ValueError, match="expected an .xlsx path"):
        fcf.save_sample_assumptions(tmp_path / "assumptions.csv")
