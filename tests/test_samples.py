"""fcf.samples.* -- the single packaged-sample surface.

Replaces the old per-file ``save_sample_*`` helpers: :func:`samples.export`
writes a template's starter files (in a chosen format) to a directory, and the
loaders return assembled objects. A round-trip through ``read_*`` must
reproduce the bundled in-memory sample.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.gmm import measure


def test_templates_lists_available():
    assert fcf.samples.templates() == ["gmm", "vfa"]


def test_export_gmm_round_trips(tmp_path):
    """export writes the gmm set; reading it back reproduces the bundled
    sample's measurement."""
    fcf.samples.export(tmp_path, template="gmm")
    for name in ("basis.xlsx", "policies.csv", "coverages.csv",
                 "calculation_methods.csv", "inforce_state.csv",
                 "inforce_policies.csv"):
        assert (tmp_path / name).exists(), name
    mp = fcf.read_model_points(
        tmp_path / "policies.csv",
        coverages=tmp_path / "coverages.csv",
        calculation_methods=tmp_path / "calculation_methods.csv")
    basis = fcf.read_basis(tmp_path / "basis.xlsx")
    a = measure(mp, basis, full=False)
    b = measure(fcf.samples.model_points(), fcf.samples.basis(), full=False)
    assert np.allclose(a.bel, b.bel) and np.allclose(a.csm, b.csm)


def test_export_combined_inforce_round_trips(tmp_path):
    """The combined inforce_policies file reads back via read_inforce_policies
    with the period-close state folded in."""
    fcf.samples.export(tmp_path, template="gmm")
    mp, state = fcf.read_inforce_policies(
        tmp_path / "inforce_policies.csv",
        coverages=tmp_path / "coverages.csv",
        calculation_methods=tmp_path / "calculation_methods.csv")
    assert mp.n_mp == state.elapsed_months.shape[0]
    assert np.all(np.asarray(mp.elapsed_months) > 0)  # state folded in


@pytest.mark.parametrize("fmt,ext", [("csv", ".csv"), ("parquet", ".parquet"),
                                     ("feather", ".feather"), ("xlsx", ".xlsx")])
def test_export_format_picks_data_extension(tmp_path, fmt, ext):
    """format= sets the data-file extension; the basis stays .xlsx; reads back."""
    fcf.samples.export(tmp_path, template="gmm", format=fmt)
    assert (tmp_path / "basis.xlsx").exists()
    assert (tmp_path / f"policies{ext}").exists()
    mp = fcf.read_model_points(
        tmp_path / f"policies{ext}",
        coverages=tmp_path / f"coverages{ext}",
        calculation_methods=tmp_path / f"calculation_methods{ext}")
    assert mp.n_mp == 11


def test_export_vfa(tmp_path):
    fcf.samples.export(tmp_path, template="vfa")
    assert (tmp_path / "basis.xlsx").exists()
    assert (tmp_path / "policies.csv").exists()


def test_export_returns_directory(tmp_path):
    out = fcf.samples.export(tmp_path / "fresh", template="gmm")
    assert out == tmp_path / "fresh" and out.is_dir()


def test_export_rejects_unknown_template_and_format(tmp_path):
    with pytest.raises(ValueError, match="template must be one of"):
        fcf.samples.export(tmp_path, template="paa")
    with pytest.raises(ValueError, match="format must be one of"):
        fcf.samples.export(tmp_path, format="json")
    with pytest.raises(ValueError, match="template must be one of"):
        fcf.samples.basis(template="paa")
