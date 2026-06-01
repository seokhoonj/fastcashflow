"""Model-point I/O formats -- read_model_points reads xlsx and feather,
and the long-form reader picks up the optional benefit-rule columns.

xlsx: a single-sheet wide workbook and a two-sheet (policies + coverages)
long-form workbook. feather: the Arrow IPC format. All round-trip to the
same valuation as the bundled in-memory sample.
"""
import numpy as np
import openpyxl
import polars as pl

from fastcashflow import (
    load_sample_basis,
    load_sample_model_points,
    read_model_points,
    measure,
    write_measurement,
)


def _write_sheets(path, sheets):
    """Write ``(name, polars-frame)`` pairs to an .xlsx file."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, frame in sheets:
        ws = wb.create_sheet(name)
        ws.append(list(frame.columns))
        for row in frame.iter_rows():
            ws.append(list(row))
    wb.save(path)


def test_read_wide_xlsx(tmp_path):
    """A wide .xlsx reads to the same valuation as the in-memory book."""
    from fastcashflow import load_sample_calculation_methods
    asmp = next(iter(load_sample_basis().values()))
    patterns = load_sample_calculation_methods()
    mps = load_sample_model_points()
    path = tmp_path / "wide.xlsx"
    _write_sheets(path, [("model_points", mps.to_wide(asmp))])

    back = read_model_points(path, calculation_methods=patterns)
    assert back.n_mp == mps.n_mp
    assert np.allclose(measure(back, asmp, full=False).bel, measure(mps, asmp, full=False).bel)


def test_read_long_xlsx(tmp_path):
    """A long-form .xlsx -- policies and coverages sheets in one workbook."""
    from fastcashflow import load_sample_calculation_methods
    asmp = next(iter(load_sample_basis().values()))
    patterns = load_sample_calculation_methods()
    mps = load_sample_model_points()
    policies, coverages = mps.to_long(asmp)
    path = tmp_path / "long.xlsx"
    _write_sheets(path, [("policies", policies), ("coverages", coverages)])

    back = read_model_points(path, calculation_methods=patterns)
    assert back.n_mp == mps.n_mp
    assert np.allclose(measure(back, asmp, full=False).bel, measure(mps, asmp, full=False).bel)


def test_read_feather(tmp_path):
    """A .feather (Arrow IPC) model-point file round-trips."""
    from fastcashflow import load_sample_calculation_methods
    asmp = next(iter(load_sample_basis().values()))
    patterns = load_sample_calculation_methods()
    mps = load_sample_model_points()
    path = tmp_path / "wide.feather"
    mps.to_wide(asmp).write_ipc(path)

    back = read_model_points(path, calculation_methods=patterns)
    assert back.n_mp == mps.n_mp
    assert np.allclose(measure(back, asmp, full=False).bel, measure(mps, asmp, full=False).bel)


def test_write_valuation_feather(tmp_path):
    """write_measurement writes a .feather result file."""
    asmp = next(iter(load_sample_basis().values()))
    mps = load_sample_model_points()
    path = tmp_path / "results.feather"
    write_measurement(measure(mps, asmp, full=False), path)
    assert path.exists()


def test_long_form_reads_benefit_rules(tmp_path):
    """The long-form coverages frame reads the waiting / reduction columns."""
    from fastcashflow import load_sample_calculation_methods
    asmp = next(iter(load_sample_basis().values()))
    patterns = load_sample_calculation_methods()
    mps = load_sample_model_points()
    policies, coverages = mps.to_long(asmp)
    coverages = coverages.with_columns(
        pl.lit(6).alias("waiting"),
        pl.lit(24).alias("reduction_end"),
        pl.lit(0.5).alias("reduction_factor"),
    )
    pol_path = tmp_path / "policies.csv"
    cov_path = tmp_path / "coverages.csv"
    policies.write_csv(pol_path)
    coverages.write_csv(cov_path)

    back = read_model_points(pol_path, coverages=cov_path,
                             calculation_methods=patterns)
    assert np.all(back.coverage_waiting == 6)
    assert np.all(back.coverage_reduction_end == 24)
    assert np.allclose(back.coverage_reduction_factor, 0.5)


# ---------------------------------------------------------------------------
# elapsed_months -- inforce_state is the source of truth, the policies frame
# silently ignores the column. The warning surfaces a common misuse.
# ---------------------------------------------------------------------------

def test_wide_policies_elapsed_months_emits_warning(tmp_path):
    """A wide policies frame with an ``elapsed_months`` column triggers
    a UserWarning -- the reader silently drops it and inforce_state is
    the source of truth."""
    import warnings
    asmp = next(iter(load_sample_basis().values()))
    mps = load_sample_model_points()
    wide = mps.to_wide(asmp).with_columns(pl.lit(12).alias("elapsed_months"))
    path = tmp_path / "wide_with_em.xlsx"
    _write_sheets(path, [("model_points", wide)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = read_model_points(path)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("elapsed_months" in m for m in msgs), msgs
    # And the column was indeed ignored -- the ModelPoints default of zero
    # stands (every contract is treated as just-issued, per show_trace's
    # source-of-truth note).
    assert np.all(back.elapsed_months == 0)


def test_long_policies_elapsed_months_emits_warning(tmp_path):
    """Same guard fires on the long-form (policies + coverages) path."""
    import warnings
    from fastcashflow import load_sample_calculation_methods
    asmp = next(iter(load_sample_basis().values()))
    patterns = load_sample_calculation_methods()
    mps = load_sample_model_points()
    policies, coverages = mps.to_long(asmp)
    policies = policies.with_columns(pl.lit(12).alias("elapsed_months"))
    pol_path = tmp_path / "policies.csv"
    cov_path = tmp_path / "coverages.csv"
    policies.write_csv(pol_path)
    coverages.write_csv(cov_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = read_model_points(pol_path, coverages=cov_path,
                                 calculation_methods=patterns)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("elapsed_months" in m for m in msgs), msgs
    assert np.all(back.elapsed_months == 0)
