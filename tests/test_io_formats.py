"""Model-point I/O formats -- read_model_points reads xlsx and feather.

xlsx: a single-sheet wide workbook and a two-sheet (policies + coverages)
long-form workbook. feather: the Arrow IPC format. All round-trip to the
same valuation as the bundled in-memory sample.
"""
import numpy as np
import openpyxl

from fastcashflow import (
    load_sample_assumptions,
    load_sample_model_points,
    read_model_points,
    value,
    write_valuation,
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
    asmp = load_sample_assumptions()
    mps = load_sample_model_points()
    path = tmp_path / "wide.xlsx"
    _write_sheets(path, [("model_points", mps.to_wide(asmp))])

    back = read_model_points(path, asmp)
    assert back.n_mp == mps.n_mp
    assert np.allclose(value(back, asmp).bel, value(mps, asmp).bel)


def test_read_long_xlsx(tmp_path):
    """A long-form .xlsx -- policies and coverages sheets in one workbook."""
    asmp = load_sample_assumptions()
    mps = load_sample_model_points()
    policies, coverages = mps.to_long(asmp)
    path = tmp_path / "long.xlsx"
    _write_sheets(path, [("policies", policies), ("coverages", coverages)])

    back = read_model_points(path, asmp)
    assert back.n_mp == mps.n_mp
    assert np.allclose(value(back, asmp).bel, value(mps, asmp).bel)


def test_read_feather(tmp_path):
    """A .feather (Arrow IPC) model-point file round-trips."""
    asmp = load_sample_assumptions()
    mps = load_sample_model_points()
    path = tmp_path / "wide.feather"
    mps.to_wide(asmp).write_ipc(path)

    back = read_model_points(path, asmp)
    assert back.n_mp == mps.n_mp
    assert np.allclose(value(back, asmp).bel, value(mps, asmp).bel)


def test_write_valuation_feather(tmp_path):
    """write_valuation writes a .feather result file."""
    asmp = load_sample_assumptions()
    mps = load_sample_model_points()
    path = tmp_path / "results.feather"
    write_valuation(value(mps, asmp), path)
    assert path.exists()
