"""Optional A/E factor layer.

A workbook may carry an ``ae_factors`` sheet with rows
``(product, channel, coverage_code, factor)`` plus optional axis columns
``{sex, age, issue_age, duration}``. The reader looks up the (product,
channel, coverage_code) factor and wraps the base rate so the engine
multiplies by the factor at lookup time. Main mortality uses
``coverage_code = 'DEATH_MAIN'`` to match the ``riders`` sheet convention.
Missing sheet or missing key = no adjustment (factor 1.0).
"""
from pathlib import Path

import numpy as np
import openpyxl

from fastcashflow import read_assumptions


def _build(path: Path, *, ae_rows=None):
    """A small assumptions workbook with one segment + an optional ae_factors sheet."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    seg = wb.create_sheet("segments")
    seg.append(["product", "channel", "mortality_table", "lapse_table",
                "discount_table", "inflation_table",
                "alpha_flat", "ra_confidence", "mortality_cv",
                "morbidity_cv"])
    seg.append(["TERM_A", "GA", "MORT", "LAPSE", "DISC", "INFL",
                100_000, 0.75, 0.10, 0.10])

    rd = wb.create_sheet("coverages")
    rd.append(["coverage_code", "coverage_name", "benefit_type", "rate_table"])
    rd.append(["DEATH_MAIN", "main death", "DEATH_MAIN", None])
    rd.append(["INPATIENT", "hospitalization", "MORBIDITY", "HOSP"])

    mt = wb.create_sheet("mortality_tables")
    mt.append(["table_id", "sex", "age", "rate"])
    for sex in (0, 1):
        for age in range(20, 81):
            mt.append(["MORT", sex, age, 0.001])

    rr = wb.create_sheet("incidence_rate_tables")
    rr.append(["table_id", "sex", "age", "rate"])
    for sex in (0, 1):
        for age in range(20, 81):
            rr.append(["HOSP", sex, age, 0.02])

    lp = wb.create_sheet("lapse_tables")
    lp.append(["table_id", "duration", "rate"])
    lp.append(["LAPSE", 0, 0.05])

    dt = wb.create_sheet("discount_tables")
    dt.append(["table_id", "year", "rate"])
    dt.append(["DISC", 0, 0.03])

    inf = wb.create_sheet("inflation_tables")
    inf.append(["table_id", "year", "rate"])
    inf.append(["INFL", 0, 0.02])

    if ae_rows is not None:
        ae = wb.create_sheet("ae_factors")
        ae.append(ae_rows[0])
        for r in ae_rows[1:]:
            ae.append(r)

    wb.save(path)


def _segment(path):
    return read_assumptions(path)[("TERM_A", "GA")]


def test_no_ae_factor_sheet(tmp_path):
    """Workbook with no ae_factors sheet leaves rates unchanged."""
    p = tmp_path / "a.xlsx"
    _build(p)
    asmp = _segment(p)
    s = np.array([0]); a = np.array([30]); d = np.array([0])
    assert asmp.mortality_annual(s, a, d, np.zeros_like(d), np.zeros_like(d))[0] == 0.001
    assert asmp.riders[0].rate(s, a, d, np.zeros_like(d), np.zeros_like(d))[0] == 0.02


def test_scalar_ae_factor_per_rider(tmp_path):
    """A single (product, channel, coverage_code, factor) row scales the rider rate."""
    p = tmp_path / "a.xlsx"
    _build(p, ae_rows=[
        ["product", "channel", "coverage_code", "factor"],
        ["TERM_A", "GA", "INPATIENT", 1.5],          # 손해율 150%
    ])
    asmp = _segment(p)
    s = np.array([0]); a = np.array([30]); d = np.array([0])
    # base 0.02 * factor 1.5 = 0.03
    assert np.isclose(asmp.riders[0].rate(s, a, d, np.zeros_like(d), np.zeros_like(d))[0], 0.03)
    # mortality unchanged (no row for dth_main)
    assert asmp.mortality_annual(s, a, d, np.zeros_like(d), np.zeros_like(d))[0] == 0.001


def test_ae_factor_on_main_mortality(tmp_path):
    """coverage_code 'DEATH_MAIN' wires to the main mortality table."""
    p = tmp_path / "a.xlsx"
    _build(p, ae_rows=[
        ["product", "channel", "coverage_code", "factor"],
        ["TERM_A", "GA", "DEATH_MAIN", 0.80],     # CI 80% — pricing 위험률에 마진 있음
    ])
    asmp = _segment(p)
    s = np.array([0]); a = np.array([30]); d = np.array([0])
    assert np.isclose(asmp.mortality_annual(s, a, d, np.zeros_like(d), np.zeros_like(d))[0], 0.001 * 0.80)


def test_ae_factor_varies_by_age(tmp_path):
    """A factor table with an age column captures age-dependent A/E (e.g., 20s anti-selection).

    The dense-grid rule applies to factor tables like to rate tables -- list
    every age, even when bands repeat. Three bands here: 25-29 = 3.0
    (heavy anti-selection), 30-49 = 1.5, 50-60 = 1.0 (ultimate).
    """
    rows = [["product", "channel", "coverage_code", "age", "factor"]]
    for age in range(25, 30):
        rows.append(["TERM_A", "GA", "INPATIENT", age, 3.0])
    for age in range(30, 50):
        rows.append(["TERM_A", "GA", "INPATIENT", age, 1.5])
    for age in range(50, 61):
        rows.append(["TERM_A", "GA", "INPATIENT", age, 1.0])
    p = tmp_path / "a.xlsx"
    _build(p, ae_rows=rows)
    asmp = _segment(p)
    s = np.array([0, 0, 0]); a = np.array([25, 40, 60]); d = np.array([0, 0, 0])
    out = asmp.riders[0].rate(s, a, d, np.zeros_like(d), np.zeros_like(d))
    assert np.allclose(out, [0.02 * 3.0, 0.02 * 1.5, 0.02 * 1.0])


def test_ae_factor_only_applies_to_matching_segment(tmp_path):
    """A factor for (TERM_A, FC, hosp) does NOT apply to (TERM_A, GA, hosp)."""
    p = tmp_path / "a.xlsx"
    _build(p, ae_rows=[
        ["product", "channel", "coverage_code", "factor"],
        ["TERM_A", "FC", "INPATIENT", 1.5],         # different channel
    ])
    asmp = _segment(p)
    s = np.array([0]); a = np.array([30]); d = np.array([0])
    # GA segment: no matching A/E -> base rate unchanged
    assert asmp.riders[0].rate(s, a, d, np.zeros_like(d), np.zeros_like(d))[0] == 0.02


def test_ae_factor_composes_with_age_shift(tmp_path):
    """age_shift and A/E factor compose -- the shift moves the lookup, the factor scales it."""
    p = tmp_path / "a.xlsx"
    # Mortality rate is flat 0.001 (so age shift doesn't change the lookup value),
    # but the factor scales it.
    wb = openpyxl.load_workbook(p) if p.exists() else None
    # Easier to re-run _build but add age_shift column too. Inline:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    seg = wb.create_sheet("segments")
    seg.append(["product", "channel", "mortality_table", "lapse_table",
                "discount_table", "inflation_table",
                "alpha_flat", "ra_confidence", "mortality_cv",
                "morbidity_cv", "mortality_age_shift"])
    seg.append(["TERM_A", "GA", "MORT", "LAPSE", "DISC", "INFL",
                100_000, 0.75, 0.10, 0.10, 5])
    rd = wb.create_sheet("coverages")
    rd.append(["coverage_code", "coverage_name", "benefit_type", "rate_table"])
    rd.append(["DEATH_MAIN", "main death", "DEATH_MAIN", None])
    mt = wb.create_sheet("mortality_tables")
    mt.append(["table_id", "sex", "age", "rate"])
    for sex in (0, 1):
        for age in range(20, 81):
            mt.append(["MORT", sex, age, 0.001 * (age - 20 + 1)])   # 0.001, 0.002, ...
    for sn, header, row in [
        ("lapse_tables", ["table_id", "duration", "rate"], ["LAPSE", 0, 0.05]),
        ("discount_tables", ["table_id", "year", "rate"], ["DISC", 0, 0.03]),
        ("inflation_tables", ["table_id", "year", "rate"], ["INFL", 0, 0.02]),
    ]:
        s_ws = wb.create_sheet(sn)
        s_ws.append(header)
        s_ws.append(row)
    ae = wb.create_sheet("ae_factors")
    ae.append(["product", "channel", "coverage_code", "factor"])
    ae.append(["TERM_A", "GA", "DEATH_MAIN", 0.5])
    wb.save(p)

    asmp = _segment(p)
    s = np.array([0]); a = np.array([30]); d = np.array([0])
    # apparent age 30 + 5 = 35 -> base rate 0.001 * 16 = 0.016; A/E 0.5 -> 0.008
    assert np.isclose(asmp.mortality_annual(s, a, d, np.zeros_like(d), np.zeros_like(d))[0], 0.001 * 16 * 0.5)
