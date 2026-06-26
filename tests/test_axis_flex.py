"""Schema-detecting rate-table reader -- axis-flex variants.

The workbook rate tables (`mortality_tables`, `incidence_rate_tables`,
`waiver_tables`, `lapse_tables`) accept any subset of
``{sex, age, issue_age, duration, issue_class}`` as columns. The reader
detects which axes are present, builds a numpy grid, and wraps it in the
standard ``(sex, issue_age, duration, issue_class) -> rate`` callable.
Axes the table does not carry broadcast; lookups past the table's range
clip to the edge; ``age`` (attained) and ``issue_age`` / ``duration``
(select schema) are mutually exclusive.
"""
import numpy as np
import openpyxl
import pytest

from fastcashflow import Basis, CoverageRate
from fastcashflow.io import _flex_rate_table
from conftest import PATTERNS

def _sheet(rows):
    """Build a worksheet from a list of (header, row, row, ...) tuples."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "test"
    for r in rows:
        ws.append(list(r))
    return ws


def _call(callable_, sex, issue_age, duration, issue_class=None, elapsed=None):
    s = np.atleast_1d(np.asarray(sex, dtype=np.int64))
    a = np.atleast_1d(np.asarray(issue_age, dtype=np.int64))
    d = np.atleast_1d(np.asarray(duration, dtype=np.int64))
    c = (np.zeros_like(s) if issue_class is None
         else np.atleast_1d(np.asarray(issue_class, dtype=np.int64)))
    e = (np.zeros_like(s) if elapsed is None
         else np.atleast_1d(np.asarray(elapsed, dtype=np.int64)))
    return callable_(s, a, d, c, e)


def test_scalar_schema():
    """Just (table_id, rate) -- one row per table, broadcast over everything."""
    ws = _sheet([
        ("table_id", "rate"),
        ("FLAT", 0.005),
    ])
    out = _flex_rate_table(ws)
    fn = out["FLAT"]
    assert _call(fn, [0, 1, 0], [30, 40, 50], [0, 5, 10]).tolist() == [0.005, 0.005, 0.005]


def test_age_only_schema():
    """(table_id, age, rate) -- attained age; sex/duration broadcast."""
    ws = _sheet([
        ("table_id", "age", "rate"),
        ("AGE", 30, 0.001),
        ("AGE", 31, 0.002),
        ("AGE", 32, 0.003),
    ])
    out = _flex_rate_table(ws)
    fn = out["AGE"]
    # attained = issue_age + duration
    assert _call(fn, [0], [30], [0])[0] == 0.001       # attained 30
    assert _call(fn, [0], [30], [1])[0] == 0.002       # attained 31
    assert _call(fn, [1], [31], [1])[0] == 0.003       # attained 32; sex broadcast


def test_sex_age_schema():
    """(table_id, sex, age, rate) -- the historical default."""
    ws = _sheet([
        ("table_id", "sex", "age", "rate"),
        ("MORT", 0, 30, 0.001),
        ("MORT", 0, 31, 0.002),
        ("MORT", 1, 30, 0.0008),
        ("MORT", 1, 31, 0.0016),
    ])
    fn = _flex_rate_table(ws)["MORT"]
    assert _call(fn, [0, 1], [30, 30], [0, 0]).tolist() == [0.001, 0.0008]
    assert _call(fn, [0, 1], [30, 30], [1, 1]).tolist() == [0.002, 0.0016]


def test_duration_only_schema_for_lapse():
    """(table_id, duration, rate) -- lapse-style, sex/age broadcast."""
    ws = _sheet([
        ("table_id", "duration", "rate"),
        ("LAPSE", 0, 0.20),
        ("LAPSE", 1, 0.15),
        ("LAPSE", 2, 0.10),
    ])
    fn = _flex_rate_table(ws)["LAPSE"]
    # sex / issue_age vary, lapse depends only on duration
    assert _call(fn, [0, 1, 0], [25, 40, 55], [0, 1, 2]).tolist() == [0.20, 0.15, 0.10]


def test_select_and_ultimate_schema():
    """(table_id, sex, issue_age, duration, rate) -- full select grid."""
    ws = _sheet([
        ("table_id", "sex", "issue_age", "duration", "rate"),
        ("SEL", 0, 30, 0, 0.0003),
        ("SEL", 0, 30, 1, 0.0004),
        ("SEL", 0, 30, 2, 0.0005),
        ("SEL", 0, 31, 0, 0.00035),
        ("SEL", 0, 31, 1, 0.00045),
        ("SEL", 0, 31, 2, 0.00055),
        ("SEL", 1, 30, 0, 0.00025),
        ("SEL", 1, 30, 1, 0.00035),
        ("SEL", 1, 30, 2, 0.00045),
        ("SEL", 1, 31, 0, 0.00030),
        ("SEL", 1, 31, 1, 0.00040),
        ("SEL", 1, 31, 2, 0.00050),
    ])
    fn = _flex_rate_table(ws)["SEL"]
    # Same issue_age (30), duration grows -> select effect wears off
    assert _call(fn, [0, 0, 0], [30, 30, 30], [0, 1, 2]).tolist() == [0.0003, 0.0004, 0.0005]
    # Same duration (0), different issue_age
    assert _call(fn, [0, 0], [30, 31], [0, 0]).tolist() == [0.0003, 0.00035]
    # Different sex
    assert _call(fn, [0, 1], [30, 30], [0, 0]).tolist() == [0.0003, 0.00025]


def test_age_with_select_is_rejected():
    """Mixing 'age' (attained) with 'issue_age'/'duration' (select) raises."""
    ws = _sheet([
        ("table_id", "age", "issue_age", "duration", "rate"),
        ("BAD", 30, 30, 0, 0.001),
    ])
    with pytest.raises(ValueError, match="mixes 'age'"):
        _flex_rate_table(ws)


def test_missing_grid_cell_is_rejected():
    """A non-dense grid (cartesian product has holes) raises."""
    ws = _sheet([
        ("table_id", "sex", "age", "rate"),
        ("HOLE", 0, 30, 0.001),
        ("HOLE", 0, 32, 0.003),                # age 31 missing for sex 0
        ("HOLE", 1, 30, 0.0008),
        ("HOLE", 1, 31, 0.0009),
        ("HOLE", 1, 32, 0.0010),
    ])
    with pytest.raises(ValueError, match="not dense"):
        _flex_rate_table(ws)


def test_clip_past_table_range():
    """Lookups past the table's range clip to the edge."""
    ws = _sheet([
        ("table_id", "age", "rate"),
        ("CLAMP", 30, 0.001),
        ("CLAMP", 31, 0.002),
        ("CLAMP", 32, 0.003),
    ])
    fn = _flex_rate_table(ws)["CLAMP"]
    # attained = 100 -> clipped to age 32 -> rate 0.003
    assert _call(fn, [0], [80], [20])[0] == 0.003
    # attained = 10 -> clipped to age 30 -> rate 0.001
    assert _call(fn, [0], [5], [5])[0] == 0.001


def test_multiple_tables_in_one_sheet():
    """One sheet, two table_ids -- each becomes its own callable."""
    ws = _sheet([
        ("table_id", "sex", "age", "rate"),
        ("A", 0, 30, 0.001),
        ("A", 1, 30, 0.0008),
        ("B", 0, 30, 0.005),
        ("B", 1, 30, 0.004),
    ])
    out = _flex_rate_table(ws)
    assert set(out) == {"A", "B"}
    assert _call(out["A"], [0, 1], [30, 30], [0, 0]).tolist() == [0.001, 0.0008]
    assert _call(out["B"], [0, 1], [30, 30], [0, 0]).tolist() == [0.005, 0.004]


def test_broadcasts_to_engine_grid_shape():
    """The callable matches numpy meshgrid shapes the engine passes in."""
    ws = _sheet([
        ("table_id", "sex", "age", "rate"),
        ("MORT", 0, 30, 0.001),
        ("MORT", 0, 31, 0.002),
        ("MORT", 1, 30, 0.0008),
        ("MORT", 1, 31, 0.0016),
    ])
    fn = _flex_rate_table(ws)["MORT"]
    # Engine call style: meshgrid of sex / age / duration
    sex_g, age_g, dur_g = np.meshgrid(
        np.array([0, 1]), np.array([30]), np.array([0, 1]), indexing="ij",
    )
    out = fn(sex_g, age_g, dur_g, np.zeros_like(dur_g), np.zeros_like(dur_g))
    assert out.shape == sex_g.shape
    # sex=0, age=30, dur=0 -> attained 30 -> 0.001 ; dur=1 -> attained 31 -> 0.002
    assert out[0, 0, 0] == 0.001
    assert out[0, 0, 1] == 0.002
    assert out[1, 0, 0] == 0.0008
    assert out[1, 0, 1] == 0.0016


def test_issue_class_axis_is_recognised():
    """A table with an ``issue_class`` column varies by that axis; lookups
    pass the per-policy issue_class through as the fourth callable arg."""
    ws = _sheet([
        ("table_id", "sex", "issue_class", "rate"),
        ("CLS", 0, 0, 0.0010),
        ("CLS", 0, 1, 0.0020),
        ("CLS", 0, 2, 0.0040),
    ])
    fn = _flex_rate_table(ws)["CLS"]
    # Same (sex, age, dur) but different issue_class -> different rate.
    s = np.array([0, 0, 0]); a = np.array([40, 40, 40]); d = np.array([0, 0, 0])
    c = np.array([0, 1, 2]); e = np.zeros_like(s)
    out = fn(s, a, d, c, e)
    assert np.allclose(out, [0.0010, 0.0020, 0.0040])
    # An issue_class beyond the table's range clips to the edge.
    assert fn(np.array([0]), np.array([40]), np.array([0]),
               np.array([99]), np.array([0]))[0] == 0.0040


def test_legacy_three_arg_user_lambda_is_adapted():
    """User-supplied 3-arg rate lambdas are auto-wrapped to the 5-arg shape
    the engine now passes; the issue_class and elapsed arguments are
    discarded by the wrapper."""
    basis = Basis(
        # Pre-Phase-1A user lambda style -- 3 positional args.
        mortality_annual=lambda sex, age, dur: np.full(dur.shape, 0.001),
        lapse_annual=lambda sex, age, dur: np.full(dur.shape, 0.01),
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", lambda sex, age, dur: np.full(dur.shape, 0.001)),),
    )
    # The adapter accepts the new 5-arg engine call without error.
    s = np.array([0]); a = np.array([40]); d = np.array([0])
    c = np.array([0]); e = np.array([0])
    assert basis.mortality_annual(s, a, d, c, e)[0] == 0.001
    assert basis.lapse_annual(s, a, d, c, e)[0] == 0.01


def test_legacy_four_arg_duration_lambda_is_adapted():
    """A pre-unification 4-arg DurationRateFn lambda (sex, age, dur, cohort)
    is auto-wrapped: the wrapper maps the original 4th argument to the
    new 5th ``elapsed`` slot, so the engine's 5-arg call routes the
    cohort dimension through correctly."""
    basis = Basis(
        mortality_annual=lambda s, a, d: np.full(d.shape, 0.001),
        lapse_annual=lambda s, a, d: np.full(d.shape, 0.01),
        # 4-arg legacy DurationRateFn user lambda -- rate doubles per
        # cohort step (an exclusion-window style probe).
        ci_reincidence_annual=lambda s, a, d, cohort: np.where(
            cohort == 0, 0.10, 0.20,
        ),
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", lambda s, a, d: np.full(d.shape, 0.001)),),
    )
    s = np.array([0, 0]); a = np.array([40, 40]); d = np.array([0, 0])
    c = np.array([0, 0])
    # cohort_index lands in the new 5th positional slot (``elapsed``).
    elapsed_cohort_zero = np.array([0, 0])
    elapsed_cohort_one = np.array([0, 1])
    assert basis.ci_reincidence_annual(s, a, d, c, elapsed_cohort_zero)[0] == 0.10
    out = basis.ci_reincidence_annual(s, a, d, c, elapsed_cohort_one)
    assert out[0] == 0.10 and out[1] == 0.20


def test_elapsed_axis_is_recognised():
    """A table with an ``elapsed`` column varies by that sojourn axis;
    lookups pass the per-call elapsed value through as the fifth
    callable argument."""
    ws = _sheet([
        ("table_id", "sex", "elapsed", "rate"),
        ("CAN_RE", 0, 0, 0.00),       # exclusion window -- no recurrence at t=0
        ("CAN_RE", 0, 1, 0.05),
        ("CAN_RE", 0, 2, 0.04),
        ("CAN_RE", 0, 3, 0.03),
    ])
    fn = _flex_rate_table(ws)["CAN_RE"]
    s = np.array([0, 0, 0, 0]); a = np.array([50, 50, 50, 50])
    d = np.array([0, 0, 0, 0]); c = np.zeros_like(s)
    e = np.array([0, 1, 2, 3])
    out = fn(s, a, d, c, e)
    assert np.allclose(out, [0.00, 0.05, 0.04, 0.03])
    # An elapsed beyond the table's range clips to the edge.
    assert fn(np.array([0]), np.array([50]), np.array([0]),
               np.array([0]), np.array([99]))[0] == 0.03
