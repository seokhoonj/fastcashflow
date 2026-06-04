"""Bundled synthetic sample data -- ``fcf.samples.*``.

The single surface for the packaged demo data. Two uses, deliberately distinct:

* **load** -- get an assembled object to play with or feed straight to a
  measurement: :func:`basis`, :func:`model_points`, :func:`calculation_methods`,
  :func:`inforce_state`.
* **export** -- write a starter set of input *template files* to a directory
  (edit them, then read back with ``fcf.read_model_points`` / ``fcf.read_basis``).
  :func:`templates` lists the available template names.

``template="gmm"`` (default) is the protection portfolio; ``template="vfa"``
is the variable (account-value) contract set. The data is synthetic
(calibrated demo figures), never sourced from real portfolios.
"""
from pathlib import Path

from fastcashflow import io as _io

#: Available sample templates -- see :func:`templates`.
_TEMPLATES = ("gmm", "vfa")

#: ``format=`` choices for :func:`export` -> the data-file extension. The
#: basis is always a multi-sheet ``.xlsx`` workbook regardless of this.
_FORMATS = {"csv": ".csv", "parquet": ".parquet",
            "feather": ".feather", "xlsx": ".xlsx"}


def templates() -> list[str]:
    """The available :func:`export` / load template names (``["gmm", "vfa"]``)."""
    return list(_TEMPLATES)


def basis(template: str = "gmm"):
    """Bundled sample basis. ``template="gmm"`` (default) returns the per-segment
    ``{(product_code, channel_code): Basis}`` dict; ``template="vfa"`` returns the
    single variable-contract :class:`~fastcashflow.Basis`."""
    if template == "vfa":
        return _io.load_sample_vfa_basis()
    if template == "gmm":
        return _io.load_sample_basis()
    raise ValueError(f"template must be one of {_TEMPLATES}, got {template!r}")


def model_points(template: str = "gmm"):
    """Bundled sample model points (``template="gmm"`` default, ``"vfa"`` for the
    variable account-value contracts)."""
    if template == "vfa":
        return _io.load_sample_vfa_model_points()
    if template == "gmm":
        return _io.load_sample_model_points()
    raise ValueError(f"template must be one of {_TEMPLATES}, got {template!r}")


def calculation_methods():
    """Bundled sample coverage-code -> calculation-method taxonomy."""
    return _io.load_sample_calculation_methods()


def inforce_state():
    """Bundled sample in-force state (elapsed_months / count / prior_csm / ...)."""
    return _io.load_sample_inforce_state()


def _export_tree(dest: Path, files: list[str]) -> str:
    """An ASCII tree of the files :func:`export` wrote, expanding the
    ``basis.xlsx`` workbook into its sheets -- a one-glance map of what landed
    in the directory and which assumption sheets the basis carries."""
    import openpyxl
    lines = [f"{dest}/"]
    for i, name in enumerate(files):
        last_file = i == len(files) - 1
        lines.append(f"{'└── ' if last_file else '├── '}{name}")
        if name.endswith(".xlsx"):
            wb = openpyxl.load_workbook(dest / name, read_only=True)
            sheets = wb.sheetnames
            wb.close()
            pad = "    " if last_file else "│   "
            for j, sheet in enumerate(sheets):
                last_sheet = j == len(sheets) - 1
                lines.append(f"{pad}{'└── ' if last_sheet else '├── '}{sheet}")
    return "\n".join(lines)


def export(output_dir, template: str = "gmm", format: str = "csv",
           *, quiet: bool = False) -> Path:
    """Write a starter set of input template files to ``output_dir``.

    ``template="gmm"`` writes ``basis.xlsx`` plus ``policies`` / ``coverages``
    / ``calculation_methods`` / ``inforce_state`` and the combined
    ``inforce_policies`` (the period-close one-file form); ``template="vfa"``
    writes the variable-contract ``basis.xlsx`` and ``policies``. Edit them and
    read back with :func:`~fastcashflow.read_model_points` /
    :func:`~fastcashflow.read_basis`.

    ``format`` picks the data-file extension -- ``"csv"`` (default),
    ``"parquet"``, ``"feather"`` or ``"xlsx"``. The basis is always a
    multi-sheet ``.xlsx`` workbook (it cannot be a flat table), so ``format``
    applies only to the policies / coverages / state files. Use ``"parquet"``
    for a portfolio large enough to stream with
    :func:`~fastcashflow.gmm.measure_stream`.

    Prints a tree of the files written -- expanding ``basis.xlsx`` into its
    sheets -- so it is clear what landed where. Pass ``quiet=True`` to suppress
    (e.g. in scripts). Returns the destination directory.
    """
    if template not in _TEMPLATES:
        raise ValueError(f"template must be one of {_TEMPLATES}, got {template!r}")
    if format not in _FORMATS:
        raise ValueError(
            f"format must be one of {tuple(_FORMATS)}, got {format!r}")
    ext = _FORMATS[format]
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if template == "gmm":
        _io._save_sample_basis(dest / "basis.xlsx")
        _io._save_sample_policies(dest / f"policies{ext}")
        _io._save_sample_coverages(dest / f"coverages{ext}")
        _io._save_sample_calculation_methods(dest / f"calculation_methods{ext}")
        _io._save_sample_inforce_state(dest / f"inforce_state{ext}")
        _io._save_sample_inforce_policies(dest / f"inforce_policies{ext}")
        files = ["basis.xlsx", f"policies{ext}", f"coverages{ext}",
                 f"calculation_methods{ext}", f"inforce_state{ext}",
                 f"inforce_policies{ext}"]
    else:  # vfa
        _io._drop_sample_table("sample_vfa_basis.xlsx", dest / "basis.xlsx")
        _io._drop_sample_table("sample_vfa_policies.csv", dest / f"policies{ext}")
        files = ["basis.xlsx", f"policies{ext}"]
    if not quiet:
        print(f"fastcashflow sample export -- template={template!r}, "
              f"{len(files)} files")
        print(_export_tree(dest, files))
    return dest


__all__ = ["templates", "basis", "model_points", "calculation_methods",
           "inforce_state", "export"]
