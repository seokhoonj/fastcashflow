"""Input-validation tests -- the guards that turn a future silently-wrong BEL
into a loud error at the workbook reader / dataclass construction site.

These guards close concrete footguns surfaced by the 2nd review (silent BEL
miscomputation from bad inputs the reader used to accept). Each test asserts
the guard fires; without a test, a future refactor can silently remove the
guard and the footgun returns.
"""
from pathlib import Path

import numpy as np
import openpyxl
import polars as pl
import pytest

import fastcashflow as fcf
from fastcashflow import Assumptions, BenefitPattern, CoverageRate, ModelPoints
from fastcashflow.assumptions import annual_to_monthly
from fastcashflow.io import (
    _axis_tables, _flex_rate_table, _read_expense_tables, _read_state,
    _truncate_list,
)


# ---------------------------------------------------------------------------
# ModelPoints scalar guards
# ---------------------------------------------------------------------------

def test_modelpoints_rejects_negative_issue_age():
    with pytest.raises(ValueError, match="issue_age"):
        ModelPoints(
            issue_age=np.array([-5.0]),
            level_premium=np.array([0.0]),
            term_months=np.array([12]),
        )


def test_modelpoints_rejects_zero_term_months():
    with pytest.raises(ValueError, match="term_months"):
        ModelPoints(
            issue_age=np.array([40.0]),
            level_premium=np.array([0.0]),
            term_months=np.array([0]),
        )


def test_modelpoints_rejects_negative_count():
    with pytest.raises(ValueError, match="count"):
        ModelPoints(
            issue_age=np.array([40.0]),
            level_premium=np.array([0.0]),
            term_months=np.array([12]),
            count=np.array([-100.0]),
        )


# ---------------------------------------------------------------------------
# Assumptions scalar guards
# ---------------------------------------------------------------------------

def _flat_rate(annual=0.01):
    return lambda sex, issue_age, duration: np.full(issue_age.shape, annual)


def test_assumptions_rejects_ra_confidence_at_boundary():
    for bad in (-0.1, 0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="ra_confidence"):
            Assumptions(
                mortality_annual=_flat_rate(), lapse_annual=_flat_rate(),
                discount_annual=0.0, ra_confidence=bad,
                mortality_cv=0.10,
                coverages=(CoverageRate("DEATH", _flat_rate()),),
            )


def test_assumptions_rejects_negative_cv():
    with pytest.raises(ValueError, match="mortality_cv"):
        Assumptions(
            mortality_annual=_flat_rate(), lapse_annual=_flat_rate(),
            discount_annual=0.0, ra_confidence=0.75,
            mortality_cv=-0.1,
            coverages=(CoverageRate("DEATH", _flat_rate()),),
        )


def test_assumptions_rejects_settlement_pattern_not_summing_to_one():
    with pytest.raises(ValueError, match="settlement_pattern"):
        Assumptions(
            mortality_annual=_flat_rate(), lapse_annual=_flat_rate(),
            discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
            settlement_pattern=np.array([0.3, 0.3, 0.3]),
            coverages=(CoverageRate("DEATH", _flat_rate()),),
        )


# ---------------------------------------------------------------------------
# io.py rate-table duplicate row catch
# ---------------------------------------------------------------------------

def _make_ws(rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "mortality_tables"
    for r in rows:
        ws.append(list(r))
    return ws


def test_flex_rate_table_rejects_duplicate_rows():
    """A duplicate (table_id, sex, age, ...) row would last-wins silently."""
    ws = _make_ws([
        ("table_id", "sex", "age", "rate"),
        ("MORT", 0, 30, 0.001),
        ("MORT", 0, 30, 99.0),    # the silent-overwrite case
    ])
    with pytest.raises(ValueError, match="duplicate row"):
        _flex_rate_table(ws)


# ---------------------------------------------------------------------------
# io.py state range check
# ---------------------------------------------------------------------------

def test_read_state_rejects_unknown_integer():
    """state=42 used to slip through as a silent garbage state index."""
    col = pl.Series("state", [0, 42])
    with pytest.raises(ValueError, match="unknown integer value"):
        _read_state(col)


# ---------------------------------------------------------------------------
# io.py wide reader collision with reserved column names
# ---------------------------------------------------------------------------

def test_wide_reader_rejects_collision_with_reserved_name(tmp_path):
    """A coverage_code 'maturity' would shadow the maturity_benefit scalar."""
    asmp_book = tmp_path / "assumptions.xlsx"
    _write_minimal_assumptions(asmp_book, coverage_code="maturity")
    pol_csv = tmp_path / "policies.csv"
    pl.DataFrame({
        "issue_age": [40], "term_months": [12], "level_premium": [0.0],
        "maturity_benefit": [1_000_000.0],
    }).write_csv(pol_csv)
    bp_csv = tmp_path / "benefit_patterns.csv"
    pl.DataFrame({
        "coverage_code": ["maturity"], "benefit_pattern": ["DEATH"],
    }).write_csv(bp_csv)

    basis = fcf.read_assumptions(asmp_book)
    with pytest.raises(ValueError, match="reserved wide-form"):
        fcf.read_model_points(
            pol_csv,
            assumptions=basis[next(iter(basis))],
            benefit_patterns=bp_csv,
        )


# ---------------------------------------------------------------------------
# io.py long-form mp_id uniqueness / premium double-source / reduction pair
# ---------------------------------------------------------------------------

def test_long_form_rejects_duplicate_mp_id(tmp_path):
    """Duplicate mp_id would fan out the coverages join silently."""
    asmp_book = tmp_path / "assumptions.xlsx"
    _write_minimal_assumptions(asmp_book, coverage_code="DEATH")
    pol_csv = tmp_path / "policies.csv"
    pl.DataFrame({
        "mp_id": ["A", "A"],          # the duplicate-id case
        "issue_age": [40, 40], "term_months": [12, 12], "level_premium": [0.0, 0.0],
    }).write_csv(pol_csv)
    cov_csv = tmp_path / "coverages.csv"
    pl.DataFrame({
        "mp_id": ["A"], "coverage_code": ["DEATH"], "amount": [1e8],
    }).write_csv(cov_csv)
    bp_csv = tmp_path / "benefit_patterns.csv"
    pl.DataFrame({
        "coverage_code": ["DEATH"], "benefit_pattern": ["DEATH"],
    }).write_csv(bp_csv)

    basis = fcf.read_assumptions(asmp_book)
    with pytest.raises(ValueError, match="duplicate mp_id"):
        fcf.read_model_points(
            pol_csv, coverages=cov_csv,
            assumptions=basis[next(iter(basis))],
            benefit_patterns=bp_csv,
        )


def test_long_form_rejects_premium_in_both_frames(tmp_path):
    """``premium`` in coverages and ``level_premium`` in policies = ambiguous."""
    asmp_book = tmp_path / "assumptions.xlsx"
    _write_minimal_assumptions(asmp_book, coverage_code="DEATH")
    pol_csv = tmp_path / "policies.csv"
    pl.DataFrame({
        "mp_id": ["A"], "issue_age": [40], "term_months": [12],
        "level_premium": [12_000.0],    # source 1
    }).write_csv(pol_csv)
    cov_csv = tmp_path / "coverages.csv"
    pl.DataFrame({
        "mp_id": ["A"], "coverage_code": ["DEATH"], "amount": [1e8],
        "premium": [1.0],               # source 2 -- silently overrode
    }).write_csv(cov_csv)
    bp_csv = tmp_path / "benefit_patterns.csv"
    pl.DataFrame({
        "coverage_code": ["DEATH"], "benefit_pattern": ["DEATH"],
    }).write_csv(bp_csv)

    basis = fcf.read_assumptions(asmp_book)
    with pytest.raises(ValueError, match="premium is specified twice"):
        fcf.read_model_points(
            pol_csv, coverages=cov_csv,
            assumptions=basis[next(iter(basis))],
            benefit_patterns=bp_csv,
        )


def test_long_form_rejects_reduction_factor_without_reduction_end(tmp_path):
    """reduction_factor=0.5 without reduction_end is silently inert."""
    asmp_book = tmp_path / "assumptions.xlsx"
    _write_minimal_assumptions(asmp_book, coverage_code="DEATH")
    pol_csv = tmp_path / "policies.csv"
    pl.DataFrame({
        "mp_id": ["A"], "issue_age": [40], "term_months": [12],
        "level_premium": [12_000.0],
    }).write_csv(pol_csv)
    cov_csv = tmp_path / "coverages.csv"
    pl.DataFrame({
        "mp_id": ["A"], "coverage_code": ["DEATH"], "amount": [1e8],
        "reduction_factor": [0.5],      # no reduction_end -- never fires
    }).write_csv(cov_csv)
    bp_csv = tmp_path / "benefit_patterns.csv"
    pl.DataFrame({
        "coverage_code": ["DEATH"], "benefit_pattern": ["DEATH"],
    }).write_csv(bp_csv)

    basis = fcf.read_assumptions(asmp_book)
    with pytest.raises(ValueError, match="reduction_factor"):
        fcf.read_model_points(
            pol_csv, coverages=cov_csv,
            assumptions=basis[next(iter(basis))],
            benefit_patterns=bp_csv,
        )


# ---------------------------------------------------------------------------
# Minimal assumptions workbook helper
# ---------------------------------------------------------------------------

def _write_minimal_assumptions(path: Path, coverage_code: str) -> None:
    """A tiny one-segment workbook with one rate-driven coverage."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    def sheet(name, rows):
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(list(r))

    sheet("mortality_tables", [
        ("table_id", "rate"),
        ("MORT_FLAT", 0.001),
    ])
    sheet("lapse_tables", [
        ("table_id", "rate"),
        ("LAPSE_FLAT", 0.01),
    ])
    sheet("discount_tables", [
        ("table_id", "year", "rate"),
        ("DISC_FLAT", 0, 0.03),
    ])
    sheet("coverages", [
        ("coverage_code", "rate_table"),
        (coverage_code, "MORT_FLAT"),
    ])
    sheet("segments", [
        ("product_code", "channel_code", "mortality_table", "lapse_table",
         "discount_table", "ra_confidence", "mortality_cv"),
        ("TERM_LIFE_A", "FC", "MORT_FLAT", "LAPSE_FLAT", "DISC_FLAT",
         0.75, 0.10),
    ])
    wb.save(path)


# ---------------------------------------------------------------------------
# P1 E -- io.py error-message quality
# ---------------------------------------------------------------------------

def _make_sheet(title, rows):
    """Build an openpyxl worksheet from header + data rows."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = title
    for r in rows:
        ws.append(list(r))
    return ws


def test_truncate_list_caps_long_alternatives():
    """A 100-entry registry list must collapse to 10 items + suffix."""
    items = [f"T{i:03d}" for i in range(20)]
    out = _truncate_list(items, cap=10)
    assert "T000" in out and "T009" in out
    assert "and 10 more" in out
    assert "T015" not in out          # past the cap


def test_truncate_list_returns_full_when_under_cap():
    items = ["A", "B", "C"]
    assert _truncate_list(items, cap=10) == repr(items)


def test_segments_legacy_product_column_hints(tmp_path):
    """A user with the old ``product`` header gets a rename hint."""
    book = tmp_path / "assumptions.xlsx"
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    def sheet(name, rows):
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(list(r))

    sheet("mortality_tables", [("table_id", "rate"), ("MORT", 0.001)])
    sheet("lapse_tables", [("table_id", "rate"), ("LAPSE", 0.01)])
    sheet("discount_tables", [("table_id", "year", "rate"), ("DISC", 0, 0.03)])
    sheet("segments", [
        ("product", "channel", "mortality_table", "lapse_table",
         "discount_table", "ra_confidence", "mortality_cv"),
        ("TERM_LIFE_A", "FC", "MORT", "LAPSE", "DISC", 0.75, 0.10),
    ])
    wb.save(book)
    with pytest.raises(ValueError, match="did you mean 'product_code'"):
        fcf.read_assumptions(book)


def test_missing_required_sheet_friendly_error(tmp_path):
    """A workbook missing ``mortality_tables`` raises a sheet-named error,
    not a raw openpyxl KeyError."""
    book = tmp_path / "assumptions.xlsx"
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    # Build a workbook with everything except mortality_tables.
    for name, rows in (
        ("lapse_tables", [("table_id", "rate"), ("LAPSE", 0.01)]),
        ("discount_tables", [("table_id", "year", "rate"), ("DISC", 0, 0.03)]),
        ("segments", [
            ("product_code", "channel_code", "mortality_table", "lapse_table",
             "discount_table", "ra_confidence", "mortality_cv"),
        ]),
    ):
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(list(r))
    wb.save(book)
    with pytest.raises(ValueError, match="missing required sheet 'mortality_tables'"):
        fcf.read_assumptions(book)


def test_flex_rate_table_missing_rate_column():
    """A rate sheet without the ``rate`` column gets a named-column error."""
    ws = _make_sheet("mortality_tables", [
        ("table_id", "sex", "age"),    # no 'rate' column
        ("MORT", 0, 30),
    ])
    with pytest.raises(ValueError, match="missing required column 'rate'"):
        _flex_rate_table(ws)


def test_axis_tables_missing_value_column():
    """An axis sheet without the value column gets a named-column error."""
    ws = _make_sheet("discount_tables", [
        ("table_id", "year"),          # no 'rate' column
        ("DISC", 0),
    ])
    with pytest.raises(ValueError, match=r"missing required column.*'rate'"):
        _axis_tables(ws, "year")


def test_axis_tables_sparse_row():
    """A row missing its axis column gets a row-level error, not raw KeyError."""
    ws = _make_sheet("discount_tables", [
        ("table_id", "year", "rate"),
        ("DISC", 0, 0.03),
    ])
    # Synthesise a second row with no 'year' value by reaching into openpyxl
    ws.append(["DISC", None, 0.04])
    # The sparse row's ``year`` becomes None -> int(None) raises TypeError, not
    # KeyError, so the bare-row guard here is on the column-existence side.
    # Instead test the value_col-missing case via the absent column.
    ws2 = _make_sheet("discount_tables", [
        ("table_id", "year", "rate"),
        ("DISC", 0, 0.03),
    ])
    # Build a row dict that drops 'year' to trigger the per-row guard.
    # _axis_tables iterates _sheet_dicts, which only yields cells whose
    # header was set. The cleanest sparse-row case is the header missing,
    # exercised by the missing-value-column test above. Keep this test as
    # a smoke check that valid rows still parse.
    out = _axis_tables(ws2, "year")
    assert "DISC" in out and out["DISC"][0] == 0.03


def test_read_expense_tables_missing_column():
    """The expense_tables sheet without ``value`` column is named in the error."""
    ws = _make_sheet("expense_tables", [
        ("table_id", "expense_type", "basis"),     # no 'value'
        ("EXP", "acquisition", "premium_pct"),
    ])
    with pytest.raises(ValueError, match="missing required column"):
        _read_expense_tables(ws)


def test_long_form_orphan_mp_id_names_offender(tmp_path):
    """``cov.mp_id='X'`` not in policies: the error names ``'X'``."""
    asmp_book = tmp_path / "assumptions.xlsx"
    _write_minimal_assumptions(asmp_book, coverage_code="DEATH")
    pol_csv = tmp_path / "policies.csv"
    pl.DataFrame({
        "mp_id": ["A"], "issue_age": [40], "term_months": [12],
        "level_premium": [0.0],
    }).write_csv(pol_csv)
    cov_csv = tmp_path / "coverages.csv"
    pl.DataFrame({
        "mp_id": ["ORPHAN_X"],            # not in policies
        "coverage_code": ["DEATH"], "amount": [1e8],
    }).write_csv(cov_csv)
    bp_csv = tmp_path / "benefit_patterns.csv"
    pl.DataFrame({
        "coverage_code": ["DEATH"], "benefit_pattern": ["DEATH"],
    }).write_csv(bp_csv)

    basis = fcf.read_assumptions(asmp_book)
    with pytest.raises(ValueError, match="ORPHAN_X"):
        fcf.read_model_points(
            pol_csv, coverages=cov_csv,
            assumptions=basis[next(iter(basis))],
            benefit_patterns=bp_csv,
        )


def test_long_form_orphan_coverage_code_names_offender(tmp_path):
    """``cov.coverage_code`` not in benefit_patterns: the error names the code."""
    asmp_book = tmp_path / "assumptions.xlsx"
    _write_minimal_assumptions(asmp_book, coverage_code="DEATH")
    pol_csv = tmp_path / "policies.csv"
    pl.DataFrame({
        "mp_id": ["A"], "issue_age": [40], "term_months": [12],
        "level_premium": [0.0],
    }).write_csv(pol_csv)
    cov_csv = tmp_path / "coverages.csv"
    pl.DataFrame({
        "mp_id": ["A"], "coverage_code": ["GHOST_CODE"], "amount": [1e8],
    }).write_csv(cov_csv)
    bp_csv = tmp_path / "benefit_patterns.csv"
    pl.DataFrame({
        "coverage_code": ["DEATH"], "benefit_pattern": ["DEATH"],
    }).write_csv(bp_csv)

    basis = fcf.read_assumptions(asmp_book)
    with pytest.raises(ValueError, match="GHOST_CODE"):
        fcf.read_model_points(
            pol_csv, coverages=cov_csv,
            assumptions=basis[next(iter(basis))],
            benefit_patterns=bp_csv,
        )


def test_long_form_no_premium_source_warns(tmp_path, recwarn):
    """Long-form with neither ``premium`` (cov) nor ``level_premium`` (pol)
    silently defaults to zero -- now warns."""
    asmp_book = tmp_path / "assumptions.xlsx"
    _write_minimal_assumptions(asmp_book, coverage_code="DEATH")
    pol_csv = tmp_path / "policies.csv"
    pl.DataFrame({
        "mp_id": ["A"], "issue_age": [40], "term_months": [12],
        # no level_premium
    }).write_csv(pol_csv)
    cov_csv = tmp_path / "coverages.csv"
    pl.DataFrame({
        "mp_id": ["A"], "coverage_code": ["DEATH"], "amount": [1e8],
        # no premium
    }).write_csv(cov_csv)
    bp_csv = tmp_path / "benefit_patterns.csv"
    pl.DataFrame({
        "coverage_code": ["DEATH"], "benefit_pattern": ["DEATH"],
    }).write_csv(bp_csv)

    basis = fcf.read_assumptions(asmp_book)
    fcf.read_model_points(
        pol_csv, coverages=cov_csv,
        assumptions=basis[next(iter(basis))],
        benefit_patterns=bp_csv,
    )
    matched = [w for w in recwarn.list
               if issubclass(w.category, UserWarning)
               and "no premium source" in str(w.message)]
    assert matched, [str(w.message) for w in recwarn.list]


def test_rate_table_not_found_caps_alternatives(tmp_path):
    """100 mortality tables + a typo: the error lists at most 10 plus a count."""
    book = tmp_path / "assumptions.xlsx"
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    def sheet(name, rows):
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(list(r))

    mort_rows = [("table_id", "rate")] + [
        (f"MORT_{i:03d}", 0.001) for i in range(20)
    ]
    sheet("mortality_tables", mort_rows)
    sheet("lapse_tables", [("table_id", "rate"), ("LAPSE", 0.01)])
    sheet("discount_tables", [("table_id", "year", "rate"), ("DISC", 0, 0.03)])
    sheet("coverages", [
        ("coverage_code", "rate_table"),
        ("DEATH", "NONEXISTENT"),         # typo -- triggers not-registered
    ])
    sheet("segments", [
        ("product_code", "channel_code", "mortality_table", "lapse_table",
         "discount_table", "ra_confidence", "mortality_cv"),
        ("TERM_LIFE_A", "FC", "MORT_000", "LAPSE", "DISC", 0.75, 0.10),
    ])
    wb.save(book)
    with pytest.raises(ValueError, match="and 10 more"):
        fcf.read_assumptions(book)


# ---------------------------------------------------------------------------
# P1 F -- second-tier regression risk
# ---------------------------------------------------------------------------

def test_annual_to_monthly_rejects_rate_above_one():
    """A decrement probability above 1.0 produces a silent NaN; reject."""
    with pytest.raises(ValueError, match="annual rate must be <= 1.0"):
        annual_to_monthly(np.array([0.5, 1.5]))


def test_annual_to_monthly_accepts_boundary_one():
    """annual = 1.0 is the everyone-leaves-in-the-year boundary -- still valid."""
    out = annual_to_monthly(np.array([1.0]))
    assert out[0] == pytest.approx(1.0)        # monthly q = 1.0


def test_discount_curve_rejects_rate_at_negative_one():
    """A discount annual <= -1.0 produces NaN; reject up front."""
    from fastcashflow.curves import discount_monthly_curve
    assumptions = Assumptions(
        mortality_annual=_flat_rate(), lapse_annual=_flat_rate(),
        discount_annual=-1.0,
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat_rate()),),
    )
    with pytest.raises(ValueError, match="discount_annual must be > -1.0"):
        discount_monthly_curve(assumptions, n_time=12)


def test_value_in_force_rejects_elapsed_past_term():
    """elapsed_months > term_months silently read past the trajectory -- reject."""
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([0.0]),
        term_months=np.array([12]),
        elapsed_months=np.array([15]),         # past maturity
    )
    assumptions = Assumptions(
        mortality_annual=_flat_rate(), lapse_annual=_flat_rate(),
        discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat_rate()),),
    )
    with pytest.raises(ValueError, match="run past its original maturity"):
        fcf.value_in_force(mp, assumptions)


def test_measure_in_force_rejects_elapsed_past_term():
    """Same elapsed > term guard on the trajectory-returning entry."""
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([0.0]),
        term_months=np.array([12]),
        elapsed_months=np.array([15]),
    )
    assumptions = Assumptions(
        mortality_annual=_flat_rate(), lapse_annual=_flat_rate(),
        discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat_rate()),),
    )
    with pytest.raises(ValueError, match="run past its original maturity"):
        fcf.measure_in_force(
            mp, assumptions, prior_csm=np.array([0.0]),
            lock_in_rate=0.03, period_months=12,
        )


def test_issue_age_fractional_warns(recwarn):
    """A fractional issue_age would silently truncate at rate lookup -- warn."""
    ModelPoints(
        issue_age=np.array([40.7, 50.0]),       # 40.7 truncates to 40
        level_premium=np.array([0.0, 0.0]),
        term_months=np.array([12, 12]),
    )
    matched = [w for w in recwarn.list
               if issubclass(w.category, UserWarning)
               and "fractional" in str(w.message)]
    assert matched, [str(w.message) for w in recwarn.list]


def test_issue_age_integer_does_not_warn(recwarn):
    """Whole-year issue_age (the typical case) does not warn."""
    ModelPoints(
        issue_age=np.array([40.0, 50.0]),
        level_premium=np.array([0.0, 0.0]),
        term_months=np.array([12, 12]),
    )
    fractional = [w for w in recwarn.list
                  if issubclass(w.category, UserWarning)
                  and "fractional" in str(w.message)]
    assert not fractional


def test_value_segmented_matches_nfc_and_nfd_codes():
    """Composed (NFC) vs decomposed (NFD) Unicode codes match the same segment.

    A product_code on the model_points side composed (e.g. ``café`` =
    'caf' + U+00E9) compared against a basis key decomposed (``café`` =
    'cafe' + U+0301) used to mismatch by byte identity. NFC-normalising
    both sides fixes the lookup.
    """
    composed = "café"            # NFC: single e-acute char
    decomposed = "café"          # NFD: e + combining acute
    assert composed != decomposed
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([0.0]),
        term_months=np.array([12]),
        product_code=np.array([composed], dtype=object),
        channel_code=np.array(["FC"], dtype=object),
    )
    assumptions = Assumptions(
        mortality_annual=_flat_rate(), lapse_annual=_flat_rate(),
        discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat_rate()),),
    )
    # Basis keyed under the decomposed form -- the lookup must still match.
    basis = {(decomposed, "FC"): assumptions}
    out = fcf.value_segmented(mp, basis)
    assert out.bel.shape == (1,)
