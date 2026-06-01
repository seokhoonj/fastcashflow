"""save_sample_* helpers -- drop the packaged sample files on disk so a
reader's tutorial code can take a real path through read_*. The four
helpers cover the four file types the cookbook / tutorials show
(basis workbook, policies, coverages, calculation_methods).
"""
from pathlib import Path

import fastcashflow as fcf


def test_save_sample_basis_round_trips_via_read_basis(tmp_path):
    """The dropped workbook reads back through read_basis to the
    same dict of Basis the in-memory loader produces."""
    path = fcf.save_sample_basis(tmp_path / "basis.xlsx")
    assert path.exists()
    assert path.suffix == ".xlsx"

    basis_from_file = fcf.read_basis(path)
    basis_in_memory = fcf.samples.basis()
    assert sorted(basis_from_file) == sorted(basis_in_memory)


def test_save_sample_full_round_trip(tmp_path):
    """The four save_* helpers, together with the three read_* arguments,
    reproduce the same ModelPoints as load_sample_model_points."""
    fcf.save_sample_basis(tmp_path / "basis.xlsx")
    fcf.save_sample_policies(tmp_path / "policies.csv")
    fcf.save_sample_coverages(tmp_path / "coverages.csv")
    fcf.save_sample_calculation_methods(tmp_path / "calculation_methods.csv")

    basis = fcf.read_basis(tmp_path / "basis.xlsx")
    asmp = next(iter(basis.values()))
    mp_file = fcf.read_model_points(
        tmp_path / "policies.csv",
        coverages=tmp_path / "coverages.csv",
        calculation_methods=tmp_path / "calculation_methods.csv",
    )
    mp_mem = fcf.samples.model_points()
    assert mp_file.n_mp == mp_mem.n_mp
    assert list(mp_file.product_code) == list(mp_mem.product_code)


def test_save_sample_accepts_directory(tmp_path):
    """Passing a directory writes the file inside with its packaged name."""
    target = fcf.save_sample_basis(tmp_path)
    assert target == tmp_path / "sample_basis.xlsx"
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


def test_read_inforce_policies_matches_two_file_workflow(tmp_path):
    """The single-file inforce reader yields the same ModelPoints (after
    state fold) and the same InforceState as the two-step workflow that
    reads policies + inforce_state separately. So the same downstream
    valuation."""
    import numpy as np

    fcf.save_sample_basis(tmp_path / "basis.xlsx")
    fcf.save_sample_policies(tmp_path / "policies.csv")
    fcf.save_sample_coverages(tmp_path / "coverages.csv")
    fcf.save_sample_calculation_methods(tmp_path / "calculation_methods.csv")
    fcf.save_sample_inforce_state(tmp_path / "inforce_state.csv")
    fcf.save_sample_inforce_policies(tmp_path / "inforce_policies.csv")

    basis = fcf.read_basis(tmp_path / "basis.xlsx")
    asmp = basis[("TERM_LIFE_A", "GA")]

    # Two-file workflow
    mp_a = fcf.read_model_points(
        tmp_path / "policies.csv",
        coverages=tmp_path / "coverages.csv",
        calculation_methods=tmp_path / "calculation_methods.csv",
    )
    state_a = fcf.read_inforce_state(tmp_path / "inforce_state.csv")
    mp_a = fcf.apply_inforce_state(mp_a, state_a)

    # One-file workflow
    mp_b, state_b = fcf.read_inforce_policies(
        tmp_path / "inforce_policies.csv",
        coverages=tmp_path / "coverages.csv",
        calculation_methods=tmp_path / "calculation_methods.csv",
    )

    assert mp_a.n_mp == mp_b.n_mp
    assert np.array_equal(mp_a.elapsed_months, mp_b.elapsed_months)
    assert np.allclose(mp_a.count, mp_b.count)
    assert np.allclose(state_a.prior_csm, state_b.prior_csm)
    assert state_a.lock_in_rate == state_b.lock_in_rate

    val_a = fcf.value_in_force(mp_a, asmp, period_months=3,
                               prior_csm=state_a.prior_csm,
                               lock_in_rate=state_a.lock_in_rate)
    val_b = fcf.value_in_force(mp_b, asmp, period_months=3,
                               prior_csm=state_b.prior_csm,
                               lock_in_rate=state_b.lock_in_rate)
    assert np.allclose(val_a.bel, val_b.bel)
    assert np.allclose(val_a.ra, val_b.ra)
    assert np.allclose(val_a.csm, val_b.csm)


def test_read_inforce_policies_rejects_missing_state_columns(tmp_path):
    """Missing one of the state columns surfaces as a clear ValueError,
    not as a silent fall-back to new-business defaults."""
    import pytest
    import polars as pl

    fcf.save_sample_policies(tmp_path / "broken.csv")  # spec only, no state
    fcf.save_sample_basis(tmp_path / "basis.xlsx")
    fcf.save_sample_coverages(tmp_path / "coverages.csv")
    fcf.save_sample_calculation_methods(tmp_path / "calculation_methods.csv")
    asmp = fcf.read_basis(tmp_path / "basis.xlsx")[("TERM_LIFE_A", "GA")]

    with pytest.raises(ValueError, match="missing required column"):
        fcf.read_inforce_policies(
            tmp_path / "broken.csv",
            coverages=tmp_path / "coverages.csv",
            calculation_methods=tmp_path / "calculation_methods.csv",
        )


def test_save_sample_inforce_policies_round_trip(tmp_path):
    """The combined file written by save_sample_inforce_policies reads
    back through read_inforce_policies to a usable ModelPoints / state
    pair."""
    path = fcf.save_sample_inforce_policies(tmp_path / "inforce.csv")
    assert path.exists()

    fcf.save_sample_basis(tmp_path / "basis.xlsx")
    fcf.save_sample_coverages(tmp_path / "coverages.csv")
    fcf.save_sample_calculation_methods(tmp_path / "calculation_methods.csv")
    asmp = fcf.read_basis(tmp_path / "basis.xlsx")[("TERM_LIFE_A", "GA")]
    mp, state = fcf.read_inforce_policies(
        path,
        coverages=tmp_path / "coverages.csv",
        calculation_methods=tmp_path / "calculation_methods.csv",
    )
    assert mp.n_mp > 0
    assert state.lock_in_rate == 0.03


def test_save_sample_inforce_state_round_trips(tmp_path):
    """The dropped in-force state file reads back through
    read_inforce_state to the same state load_sample_inforce_state
    produces in memory."""
    import numpy as np
    path = fcf.save_sample_inforce_state(tmp_path / "inforce_state.csv")
    assert path.exists()

    state_file = fcf.read_inforce_state(path)
    state_mem = fcf.samples.inforce_state()
    assert np.array_equal(state_file.mp_id, state_mem.mp_id)
    assert np.allclose(state_file.count, state_mem.count)
    assert np.allclose(state_file.elapsed_months, state_mem.elapsed_months)


def test_save_sample_converts_to_xlsx_single_sheet(tmp_path):
    """The three single-table sample files can land as .xlsx and round-trip
    through read_model_points just like their .csv source."""
    fcf.save_sample_basis(tmp_path / "basis.xlsx")
    fcf.save_sample_policies(tmp_path / "policies.xlsx")
    fcf.save_sample_coverages(tmp_path / "coverages.xlsx")
    fcf.save_sample_calculation_methods(tmp_path / "calculation_methods.xlsx")

    basis = fcf.read_basis(tmp_path / "basis.xlsx")
    mp = fcf.read_model_points(
        tmp_path / "policies.xlsx",
        coverages=tmp_path / "coverages.xlsx",
        calculation_methods=tmp_path / "calculation_methods.xlsx",
    )
    assert mp.n_mp == fcf.samples.model_points().n_mp


def test_save_sample_rejects_unsupported_extension(tmp_path):
    """A path the writer cannot route (no recognised extension) errors
    clearly instead of writing a silently empty file."""
    import pytest
    with pytest.raises(ValueError, match="unsupported file type"):
        fcf.save_sample_calculation_methods(tmp_path / "bp.json")


def test_save_sample_basis_rejects_non_xlsx(tmp_path):
    """The basis workbook is multi-sheet -- single-table formats
    cannot represent it. A non-.xlsx path errors clearly."""
    import pytest
    with pytest.raises(ValueError, match="expected an .xlsx path"):
        fcf.save_sample_basis(tmp_path / "basis.csv")
