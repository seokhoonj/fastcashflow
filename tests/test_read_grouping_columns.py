"""read_model_points reads issue_date / issue_class as fields and any other
policies column as a grouping attribute -- so the file -> measure -> group path
works end to end.
"""
import numpy as np
import polars as pl

from fastcashflow import read_model_points, group, group_of_contracts
from fastcashflow.gmm import measure
from conftest import PATTERNS, make_death_assumptions


def _basis():
    return make_death_assumptions(mortality_q=0.002, lapse_q=0.01,
                                  discount_annual=0.03, ra_confidence=0.75,
                                  mortality_cv=0.10)


def _read(tmp_path, n=2):
    policies = pl.DataFrame({
        "mp_id":         np.arange(n),
        "issue_age":     np.full(n, 40),
        "term_months":   np.full(n, 120),
        "premium": np.full(n, 200_000.0),
        "product":  ["TL"] * n,
        "channel":  ["GA"] * n,
        "issue_class":   np.zeros(n, dtype=int),
        "issue_date":    ["2025-06-01", "2026-02-01"][:n],
        "risk_class":    ["A", "B"][:n],          # extra column -> attribute
    })
    coverages = pl.DataFrame({
        "mp_id":         np.arange(n),
        "coverage": ["DEATH"] * n,
        "amount":        np.full(n, 1e8),
    })
    pp, cp = tmp_path / "pol.csv", tmp_path / "cov.csv"
    policies.write_csv(pp)
    coverages.write_csv(cp)
    return read_model_points(pp, coverages=cp, calculation_methods=PATTERNS)


def test_reader_reads_issue_date_and_attributes(tmp_path):
    mp = _read(tmp_path)
    assert mp.issue_date is not None
    assert mp.axis("issue_year").tolist() == [2025, 2026]      # derived from issue_date
    assert mp.attributes is not None
    assert mp.attributes["risk_class"].tolist() == ["A", "B"]
    # recognised fields do NOT leak into attributes
    for reserved in ("issue_class", "issue_date", "product", "mp_id"):
        assert reserved not in mp.attributes


def test_reader_issue_date_drives_group_of_contracts_cohort(tmp_path):
    mp = _read(tmp_path)
    m = measure(mp, _basis())
    g = group_of_contracts(m)            # cohort from issue_year -> 2025 vs 2026
    assert g.bel.shape[0] == 2           # two annual cohorts -> two groups


def test_reader_attribute_is_a_group_axis(tmp_path):
    mp = _read(tmp_path)
    m = measure(mp, _basis())
    assert group(m, by=["risk_class"]).bel.shape[0] == 2   # A vs B from the file


def test_reader_does_not_leak_internal_columns(tmp_path):
    """The internal `_mp` row index must not surface as a grouping attribute."""
    mp = _read(tmp_path)
    assert mp.attributes is not None
    assert not any(str(k).startswith("_") for k in mp.attributes)   # no `_mp`
