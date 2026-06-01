"""Bundled synthetic sample data -- ``fcf.samples.*``.

Two uses, deliberately distinct:

* **load** -- get an assembled object to play with or feed straight to a
  measurement: :func:`basis`, :func:`model_points`, :func:`calculation_methods`,
  :func:`inforce_state`. ``kind="vfa"`` returns the variable (account-value)
  sample instead of the default GMM sample.
* **export** -- write a starter set of input template files to a directory
  (edit them, then read back with ``fcf.read_model_points`` / ``fcf.read_basis``).

The data is synthetic (calibrated demo figures), never sourced from real
portfolios.
"""
import shutil
from importlib import resources
from pathlib import Path

from fastcashflow import io as _io


def basis(kind: str = "gmm"):
    """Bundled sample basis. ``kind="gmm"`` (default) returns the per-segment
    ``{(product_code, channel_code): Basis}`` dict; ``kind="vfa"`` returns the single
    variable-contract :class:`~fastcashflow.Basis`."""
    if kind == "vfa":
        return _io.load_sample_vfa_basis()
    if kind == "gmm":
        return _io.load_sample_basis()
    raise ValueError(f"kind must be 'gmm' or 'vfa', got {kind!r}")


def model_points(kind: str = "gmm"):
    """Bundled sample model points (``kind="gmm"`` default, ``"vfa"`` for the
    variable account-value contracts)."""
    if kind == "vfa":
        return _io.load_sample_vfa_model_points()
    if kind == "gmm":
        return _io.load_sample_model_points()
    raise ValueError(f"kind must be 'gmm' or 'vfa', got {kind!r}")


def calculation_methods():
    """Bundled sample coverage-code -> calculation-method taxonomy."""
    return _io.load_sample_calculation_methods()


def inforce_state():
    """Bundled sample in-force state (elapsed_months / count / prior_csm / ...)."""
    return _io.load_sample_inforce_state()


def export(dest_dir, kind: str = "gmm") -> Path:
    """Write a starter set of input template files to ``dest_dir``.

    ``kind="gmm"`` writes ``basis.xlsx``, ``policies.csv``, ``coverages.csv``
    and ``calculation_methods.csv`` (plus ``inforce_state.csv``); edit them and
    read back with :func:`~fastcashflow.read_model_points` /
    :func:`~fastcashflow.read_basis`.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if kind == "gmm":
        _io.save_sample_basis(dest / "basis.xlsx")
        _io.save_sample_policies(dest / "policies.csv")
        _io.save_sample_coverages(dest / "coverages.csv")
        _io.save_sample_calculation_methods(dest / "calculation_methods.csv")
        _io.save_sample_inforce_state(dest / "inforce_state.csv")
        return dest
    if kind == "vfa":
        base = resources.files("fastcashflow") / "sample_data"
        for packaged, out in (("sample_vfa_basis.xlsx", "basis.xlsx"),
                              ("sample_vfa_policies.csv", "policies.csv")):
            with resources.as_file(base / packaged) as src:
                shutil.copyfile(src, dest / out)
        return dest
    raise ValueError(f"kind must be 'gmm' or 'vfa', got {kind!r}")


__all__ = ["basis", "model_points", "calculation_methods", "inforce_state", "export"]
