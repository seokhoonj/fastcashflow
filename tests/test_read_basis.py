"""Basis workbook reader.

A single ``basis.xlsx`` carries the segment mapping plus the named rate
tables. The ``segments`` sheet has a ``defaults`` row whose values blank
cells inherit, and one row per (product, channel) segment; the reader returns
one ``Basis`` per segment. See docs/basis-format.md.
"""
import fastcashflow as fcf
import numpy as np

from fastcashflow import ModelPoints
from fastcashflow.gmm import measure


def test_segments_resolve():
    """The sample workbook resolves to several (product, channel) segments."""
    basis = fcf.samples.basis()
    # The sample carries three products on FC/GA (HEALTH also adds TM).
    assert set(basis) >= {
        ("TERM_LIFE_A", "FC"), ("TERM_LIFE_A", "GA"),
        ("HEALTH_A", "FC"), ("HEALTH_A", "GA"), ("HEALTH_A", "TM"),
        ("WHOLE_LIFE_A", "FC"), ("WHOLE_LIFE_A", "GA"),
    }


def test_defaults_inherited():
    """Blank cells in a segment row inherit from the ``defaults`` row."""
    basis = fcf.samples.basis()
    ga, fc = basis[("TERM_LIFE_A", "GA")], basis[("TERM_LIFE_A", "FC")]
    # ra_confidence / mortality_cv / morbidity_cv live only on the defaults row
    assert ga.ra_confidence == 0.75 and fc.ra_confidence == 0.75
    assert ga.mortality_cv == 0.10 and fc.mortality_cv == 0.10
    assert ga.morbidity_cv == 0.12 and fc.morbidity_cv == 0.12
    # Shared economic curve -- inherited identically.
    assert ga.discount_annual == 0.03 and fc.discount_annual == 0.03
    # Shared maintenance row in both segments' expense ledgers --
    # 60_000 per-policy; the 2% inflation is the global economic
    # assumption on the Basis object, not on the row itself.
    for basis in (ga, fc):
        maint = [r for r in basis.expense_items
                 if r.basis == "gamma_fixed"]
        assert len(maint) == 1
        assert maint[0].value == 60_000.0
        assert basis.expense_inflation == 0.02


def test_channel_segmented_lapse():
    """GA and FC reference different lapse tables -- the per-segment table
    reference. GA persistency is worse than FC."""
    basis = fcf.samples.basis()
    dur = np.arange(6)
    zero = np.zeros_like(dur)
    ga_lapse = basis[("TERM_LIFE_A", "GA")].lapse_annual(zero, zero, dur, zero, zero)
    fc_lapse = basis[("TERM_LIFE_A", "FC")].lapse_annual(zero, zero, dur, zero, zero)
    assert np.all(ga_lapse > fc_lapse)


def test_per_segment_acquisition_amount():
    """Acquisition cost differs per segment row (GA vs FC commission) --
    each segment points to its own expense_table_id in the
    ``expense_tables`` sheet."""
    basis = fcf.samples.basis()
    for (key, expected_acq) in (
        (("TERM_LIFE_A", "GA"), 150_000.0),
        (("TERM_LIFE_A", "FC"),  80_000.0),
        (("HEALTH_A",    "FC"), 100_000.0),
        (("HEALTH_A",    "GA"), 180_000.0),
        (("HEALTH_A",    "TM"),  40_000.0),
        (("WHOLE_LIFE_A","FC"), 200_000.0),
        (("WHOLE_LIFE_A","GA"), 350_000.0),
    ):
        rows = basis[key].expense_items
        acq = [r for r in rows if r.basis == "alpha_fixed"]
        assert len(acq) == 1, f"{key}: expected one per_policy_init row"
        assert acq[0].value == expected_acq, (key, expected_acq, acq[0].value)


def test_every_segment_has_expense_items():
    """The sample workbook attaches an ``expense_table`` to every segment;
    the loader populates ``Basis.expense_items`` on each."""
    basis = fcf.samples.basis()
    for basis in basis.values():
        assert basis.expense_items                       # populated


def test_coverages_resolved():
    """Rate-driven coverages resolve from ``incidence_rate_tables`` (or
    ``mortality_tables`` for the general death coverage); the pattern
    taxonomy now lives in ``calculation_methods.csv`` (read by
    :func:`load_sample_calculation_methods`), no longer on the
    :class:`Basis`. The order matches the workbook's ``coverages``
    sheet rows; the engine treats every entry as an ordinary rate-driven
    coverage (no slot reserved)."""
    from fastcashflow import CalculationMethod

    basis = fcf.samples.basis()
    basis = basis[("TERM_LIFE_A", "GA")]
    assert [r.code for r in basis.coverages] == [
        "DEATH", "INPATIENT", "CANCER", "ADB", "DISEASE_DEATH",
    ]
    assert fcf.samples.calculation_methods() == {
        "DEATH":         CalculationMethod.DEATH,
        "INPATIENT":     CalculationMethod.MORBIDITY,
        "CANCER":        CalculationMethod.DIAGNOSIS,
        "ADB":           CalculationMethod.DEATH,
        "DISEASE_DEATH": CalculationMethod.DEATH,
        "ANNUITY":       CalculationMethod.ANNUITY,
        "MATURITY":      CalculationMethod.MATURITY,
    }


def test_resolved_basis_values():
    """A resolved ``Basis`` runs through ``value`` and ``measure``; the
    GA and FC segments give different BEL because lapse differs (channel
    segmentation actually bites the valuation)."""
    
    basis = fcf.samples.basis()
    mp = ModelPoints.single(issue_age=40, benefits={0: 100_000_000.0},
                            level_premium=50_000.0, term_months=120,
                            calculation_methods=fcf.samples.calculation_methods())
    # Use a copy of the basis without surrender for the measure() / measure()
    # equivalence assertion -- the measure() fast path doesn't yet include
    # surrender cash flows (see surrender-value-gap memory); only measure()
    # does. With surrender disabled the two paths agree to machine precision.
    import dataclasses
    asmp_ga_no_surr = dataclasses.replace(
        basis[("TERM_LIFE_A", "GA")], surrender_value_curve=None)
    asmp_fc_no_surr = dataclasses.replace(
        basis[("TERM_LIFE_A", "FC")], surrender_value_curve=None)
    ga = measure(mp, asmp_ga_no_surr, full=False).bel[0]
    fc = measure(mp, asmp_fc_no_surr, full=False).bel[0]
    assert np.isfinite(ga) and np.isfinite(fc)
    assert not np.isclose(ga, fc)
    # fused and detailed paths agree (when surrender is disabled).
    assert np.isclose(measure(mp, asmp_ga_no_surr).bel_path[0, 0], ga)


# ---------------------------------------------------------------------------
# state_model column + STATE_MODELS registry (U+W)
# ---------------------------------------------------------------------------


def test_state_model_column_resolves_to_registry_entry():
    """The sample workbook's ``defaults`` row carries
    ``state_model = WAIVER`` -- both segments inherit and resolve to
    ``STATE_MODELS['WAIVER']``.
    """
    from fastcashflow import STATE_MODELS
    basis = fcf.samples.basis()
    for basis in basis.values():
        assert basis.state_model is STATE_MODELS["WAIVER"]


def test_state_model_column_blank_keeps_none():
    """A segment row with a blank ``state_model`` cell falls back to the
    defaults row; an empty defaults cell leaves ``Basis.state_model``
    as ``None`` (matching the engine's pre-registry behaviour).
    """
    import openpyxl, tempfile, shutil
    import importlib.resources as resources
    from fastcashflow import read_basis
    # Copy the sample workbook and clear the defaults row's state_model.
    sample = resources.files("fastcashflow").joinpath(
        "sample_data/sample_basis.xlsx")
    with tempfile.TemporaryDirectory() as d:
        dst = f"{d}/blank_state.xlsx"
        shutil.copy(sample, dst)
        wb = openpyxl.load_workbook(dst)
        ws = wb["segments"]
        col = None
        for c in range(1, ws.max_column + 1):
            if ws.cell(row=1, column=c).value == "state_model":
                col = c; break
        for r in range(2, ws.max_row + 1):
            ws.cell(row=r, column=col).value = None
        wb.save(dst)
        basis = read_basis(dst)
        for basis in basis.values():
            assert basis.state_model is None


def test_state_model_unknown_key_raises():
    """An unrecognised state_model key is rejected at read time, with a
    hint listing the registry contents."""
    import openpyxl, tempfile, shutil, pytest
    import importlib.resources as resources
    from fastcashflow import read_basis
    sample = resources.files("fastcashflow").joinpath(
        "sample_data/sample_basis.xlsx")
    with tempfile.TemporaryDirectory() as d:
        dst = f"{d}/bad_state.xlsx"
        shutil.copy(sample, dst)
        wb = openpyxl.load_workbook(dst)
        ws = wb["segments"]
        for c in range(1, ws.max_column + 1):
            if ws.cell(row=1, column=c).value == "state_model":
                ws.cell(row=2, column=c).value = "NOT_A_MODEL"
                break
        wb.save(dst)
        with pytest.raises(ValueError, match="NOT_A_MODEL"):
            read_basis(dst)


# ---------------------------------------------------------------------------
# schema_version
# ---------------------------------------------------------------------------

def test_meta_sheet_carries_schema_version():
    """The sample workbook ships with a ``_meta`` sheet declaring
    ``schema_version = v1`` -- absence triggers no error, presence
    matching a supported version is silently accepted."""
    import openpyxl, importlib.resources as resources
    sample = resources.files("fastcashflow").joinpath(
        "sample_data/sample_basis.xlsx")
    wb = openpyxl.load_workbook(sample, read_only=True)
    assert "_meta" in wb.sheetnames
    rows = list(wb["_meta"].iter_rows(values_only=True))
    rows = [(str(r[0]).strip(), r[1]) for r in rows[1:] if r and r[0]]
    assert dict(rows)["schema_version"] == "v1"


def test_legacy_workbook_without_meta_sheet_is_v1():
    """A workbook predating the _meta sheet still reads -- legacy = v1."""
    import openpyxl, tempfile, shutil, importlib.resources as resources
    from fastcashflow import read_basis
    sample = resources.files("fastcashflow").joinpath(
        "sample_data/sample_basis.xlsx")
    with tempfile.TemporaryDirectory() as d:
        dst = f"{d}/legacy.xlsx"
        shutil.copy(sample, dst)
        wb = openpyxl.load_workbook(dst)
        del wb["_meta"]
        wb.save(dst)
        basis = read_basis(dst)
        assert basis                                  # reads fine


def test_unsupported_schema_version_raises():
    """A version the build does not recognise is rejected loudly rather
    than silently mis-interpreted."""
    import openpyxl, tempfile, shutil, pytest
    import importlib.resources as resources
    from fastcashflow import read_basis
    sample = resources.files("fastcashflow").joinpath(
        "sample_data/sample_basis.xlsx")
    with tempfile.TemporaryDirectory() as d:
        dst = f"{d}/futuristic.xlsx"
        shutil.copy(sample, dst)
        wb = openpyxl.load_workbook(dst)
        wb["_meta"].cell(row=2, column=2).value = "v99"
        wb.save(dst)
        with pytest.raises(ValueError, match="schema_version"):
            read_basis(dst)
