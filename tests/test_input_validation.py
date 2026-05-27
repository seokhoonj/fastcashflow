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
from fastcashflow.io import _flex_rate_table, _read_state


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
