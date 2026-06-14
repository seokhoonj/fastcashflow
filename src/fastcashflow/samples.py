"""Bundled synthetic sample data -- ``fcf.samples.*``.

The single surface for the packaged demo data. Two uses, deliberately distinct:

* **load** -- get an assembled object to play with or feed straight to a
  measurement: :func:`basis`, :func:`model_points`, :func:`calculation_methods`,
  :func:`inforce_state`, :func:`return_scenarios` (toy fund returns for the VFA
  time-value-of-guarantees example) and :func:`rate_scenarios` (toy discount
  rates for the stochastic GMM valuation).
* **export** -- write a starter set of input *template files* to a directory
  (edit them, then read back with ``fcf.read_model_points`` / ``fcf.read_basis``).
  :func:`templates` lists the available template names.

``template="gmm"`` (default) is the protection portfolio; ``template="vfa"``
is the variable (account-value) contract set. The data is synthetic
(calibrated demo figures), never sourced from real portfolios.
"""
from pathlib import Path

import numpy as np

from fastcashflow import io as _io

#: Available sample templates -- see :func:`templates`.
_TEMPLATES = ("gmm", "vfa", "paa")

#: Fixed seed for :func:`scenarios` -- a reproducible toy path set, not a
#: calibration parameter.
_SCENARIO_SEED = 20260605

#: ``format=`` choices for :func:`export` -> the data-file extension. The
#: basis is always a multi-sheet ``.xlsx`` workbook regardless of this.
_FORMATS = {"csv": ".csv", "parquet": ".parquet",
            "feather": ".feather", "xlsx": ".xlsx"}


def templates() -> list[str]:
    """The available :func:`export` / load template names
    (``["gmm", "vfa", "paa"]``)."""
    return list(_TEMPLATES)


def basis(template: str = "gmm"):
    """Bundled sample basis. ``template="gmm"`` (default) returns the per-segment
    :class:`~fastcashflow.BasisRouter` (a ``(product, channel)`` -> ``Basis``
    mapping); ``template="vfa"`` returns the single variable-contract
    :class:`~fastcashflow.Basis`."""
    if template == "vfa":
        return _io.load_sample_vfa_basis()
    if template == "paa":
        return _io.load_sample_paa_basis()
    if template == "gmm":
        return _io.load_sample_basis()
    raise ValueError(f"template must be one of {_TEMPLATES}, got {template!r}")


def model_points(template: str = "gmm"):
    """Bundled sample model points (``template="gmm"`` default, ``"vfa"`` for the
    variable account-value contracts)."""
    if template == "vfa":
        return _io.load_sample_vfa_model_points()
    if template == "paa":
        return _io.load_sample_paa_model_points()
    if template == "gmm":
        return _io.load_sample_model_points()
    raise ValueError(f"template must be one of {_TEMPLATES}, got {template!r}")


def calculation_methods():
    """Bundled sample coverage-code -> calculation-method taxonomy."""
    return _io.load_sample_calculation_methods()


def treaty(cession: float = 0.30):
    """Bundled sample reinsurance treaty -- a quota share ceding ``cession`` of
    the direct book (default 30%).

    A treaty is a parameter object, not a data file, so this is the one
    reinsurance-specific sample object: the underlying ceded contracts are the
    same :func:`model_points` / :func:`basis` portfolio. Pass it to
    :func:`~fastcashflow.reinsurance.measure` or
    :func:`~fastcashflow.reinsurance.settle` over a segment of the sample book
    (reinsurance is measured on a single :class:`~fastcashflow.Basis`)."""
    from fastcashflow.reinsurance import QuotaShare
    return QuotaShare(cession=cession)


def inforce_state():
    """Bundled sample in-force state (elapsed_months / count / prior_csm / ...)."""
    return _io.load_sample_inforce_state()


def return_scenarios(template: str = "vfa", n_scenarios: int = 1000):
    """Toy *fund-return* scenarios, shape ``(n_scenarios, n_time)``, for the
    variable (VFA) time-value-of-guarantees example -- the ``return_scenarios``
    input to :func:`~fastcashflow.vfa.measure`.

    Generated in memory (no bundled file): deterministic, modest-volatility
    monthly fund returns so the guarantee shows a believable time value (~3% of
    account value on the sample). This is NOT a calibrated economic scenario
    generator -- the engine *consumes* scenarios, it does not certify
    valuation-grade ones. For a real valuation supply your own set via
    :func:`~fastcashflow.read_scenarios`.

    Each cell is a one-month *fund return* (not an interest-rate path -- that is
    the separate ``scenarios`` input to :func:`~fastcashflow.gmm.stochastic`);
    ``n_time`` matches the bundled VFA sample's term and the fixed seed keeps the
    output stable.
    """
    if template != "vfa":
        raise ValueError(
            "return_scenarios are a variable-contract (VFA) input; template "
            f"must be 'vfa', got {template!r}"
        )
    mp = model_points("vfa")
    n_time = int(np.asarray(mp.term_months).max())
    rng = np.random.default_rng(_SCENARIO_SEED)
    central = (1.0 + 0.06) ** (1.0 / 12.0) - 1.0   # ~6% annual, monthly return
    vol = 0.005                                    # modest monthly sd -- a toy
    return central + vol * rng.standard_normal((n_scenarios, n_time))


def rate_scenarios(n_scenarios: int = 1000):
    """Toy *discount-rate* scenarios, shape ``(n_scenarios,)``, for the
    stochastic GMM valuation -- the ``scenarios`` input to
    :func:`~fastcashflow.gmm.stochastic`. The interest-rate counterpart to
    :func:`return_scenarios` (which is fund returns).

    Generated in memory: one flat annual discount rate per scenario, modest
    dispersion around ~3%, deterministic (fixed seed). This is NOT a calibrated
    economic scenario generator -- for a real valuation supply your own rate set
    (Hull-White / Vasicek / regulator-prescribed) via
    :func:`~fastcashflow.read_scenarios`. Flat (1-D) rates so the toy is
    portfolio-agnostic; a real run can pass a 2-D ``(n_scenarios, n_time)`` curve
    set instead.
    """
    rng = np.random.default_rng(_SCENARIO_SEED + 1)   # a stream distinct from returns
    rates = 0.03 + 0.01 * rng.standard_normal(n_scenarios)
    return np.maximum(rates, 1e-4)                     # keep the discount rate positive


def _export_tree(dest: Path, files: list[str]) -> str:
    """An ASCII tree of the files :func:`export` wrote, expanding the
    ``basis.xlsx`` workbook into its sheets -- a one-glance map of what landed
    in the directory and which assumption sheets the basis carries."""
    import openpyxl
    lines = [f"{dest}/"]
    for i, name in enumerate(files):
        last_file = i == len(files) - 1
        lines.append(f"{'`-- ' if last_file else '+-- '}{name}")
        if name.endswith(".xlsx"):
            wb = openpyxl.load_workbook(dest / name, read_only=True)
            sheets = wb.sheetnames
            wb.close()
            pad = "    " if last_file else "|   "
            for j, sheet in enumerate(sheets):
                last_sheet = j == len(sheets) - 1
                lines.append(f"{pad}{'`-- ' if last_sheet else '+-- '}{sheet}")
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
    elif template == "vfa":
        _io._drop_sample_table("sample_vfa_basis.xlsx", dest / "basis.xlsx")
        _io._drop_sample_table("sample_vfa_policies.csv", dest / f"policies{ext}")
        files = ["basis.xlsx", f"policies{ext}"]
    else:  # paa
        _io._drop_sample_table("sample_paa_basis.xlsx", dest / "basis.xlsx")
        _io._drop_sample_table("sample_paa_policies.csv", dest / f"policies{ext}")
        _io._drop_sample_table("sample_paa_coverages.csv", dest / f"coverages{ext}")
        files = ["basis.xlsx", f"policies{ext}", f"coverages{ext}"]
    if not quiet:
        print(f"fastcashflow sample export -- template={template!r}, "
              f"{len(files)} files")
        print(_export_tree(dest, files))
    return dest


__all__ = ["templates", "basis", "model_points", "calculation_methods",
           "treaty", "inforce_state", "return_scenarios", "rate_scenarios",
           "export"]
