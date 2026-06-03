"""IFRS 17 level of aggregation -- general group(by=...) + group_of_contracts.

``group`` aggregates a measurement to any axis (resolved from the model points
``measure`` stamps on the result, or passed as an array); ``group_of_contracts``
is the IFRS 17 preset (portfolio x annual cohort x profitability). The
architecture point is ``CSM(sum FCF)``, not ``sum CSM(CF)`` -- the floor nets
within a group, not across.
"""
import numpy as np
import pytest

from fastcashflow import ExpenseItem, ModelPoints, group, group_of_contracts
from fastcashflow.gmm import measure
from conftest import PATTERNS, make_death_assumptions


def _assumptions():
    return make_death_assumptions(
        mortality_q       = 0.002,
        lapse_q           = 0.01,
        discount_annual   = 0.03,
        expense_inflation = 0.02,
        expense_items     = (
            ExpenseItem("acquisition", "alpha_fixed", 100_000.0),
            ExpenseItem("maintenance", "gamma_fixed",  60_000.0),
        ),
        ra_confidence     = 0.75,
        mortality_cv      = 0.10,
    )


def _two_contracts(**extra) -> ModelPoints:
    """Two term-life model points -- the first profitable, the second onerous."""
    return ModelPoints(
        issue_age=np.array([40, 40]),
        benefits={0: np.array([1e8, 1e8])},
        level_premium=np.array([300_000.0, 60_000.0]),
        term_months=np.array([120, 120]),
        calculation_methods=PATTERNS,
        **extra,
    )


# -- group: the general aggregator ------------------------------------------

def test_group_by_precomputed_array():
    """A raw label array still works -- group(m, ids)."""
    m = measure(_two_contracts(), _assumptions())
    assert group(m, np.array([0, 0])).bel.shape[0] == 1
    assert group(m, np.array([0, 1])).bel.shape[0] == 2


def test_group_by_single_axis_name():
    """A bare string is a single axis name."""
    mp = _two_contracts(product_code=np.array(["TL", "TL"]),
                        channel_code=np.array(["TM", "GA"]))
    m = measure(mp, _assumptions())
    assert group(m, "product_code").bel.shape[0] == 1   # one product
    assert group(m, "channel").bel.shape[0] == 2        # two channels


def test_group_by_axis_names_from_stamped_model_points():
    """group(m, by=[names]) resolves from the model points measure() stamped."""
    mp = _two_contracts(product_code=np.array(["TL", "TL"]),
                        channel_code=np.array(["TM", "GA"]))
    m = measure(mp, _assumptions())                 # stamps mp on m
    assert group(m, by=["product_code"]).bel.shape[0] == 1   # one product
    assert group(m, by=["product_code", "channel"]).bel.shape[0] == 2


def test_group_by_mixed_name_and_array():
    """A list may mix axis names and precomputed (n_mp,) label arrays."""
    mp = _two_contracts(product_code=np.array(["TL", "TL"]))
    m = measure(mp, _assumptions())
    onerous = np.where(m.loss_component > 0.0, "onerous", "remaining")
    g = group(m, by=["product_code", onerous])      # same product, split by onerous
    assert g.bel.shape[0] == 2


def test_group_by_arbitrary_attribute_axis():
    """Any attributes column is a valid axis -- not just the IFRS ones."""
    mp = _two_contracts(product_code=np.array(["TL", "TL"]),
                        attributes={"risk_class": np.array(["A", "B"])})
    m = measure(mp, _assumptions())
    g = group(m, by=["product_code", "risk_class"])
    assert g.bel.shape[0] == 2     # same product, different risk_class -> 2 groups


def test_group_by_names_needs_model_points():
    m = measure(_two_contracts(), _assumptions())
    object.__setattr__(m, "model_points", None)      # a stamp-less result
    with pytest.raises(ValueError, match="needs the model points"):
        group(m, by=["product_code"])


def test_group_unknown_axis_rejected():
    mp = _two_contracts(product_code=np.array(["TL", "TL"]))
    m = measure(mp, _assumptions())
    with pytest.raises(KeyError, match="unknown grouping axis"):
        group(m, by=["nonexistent"])


# -- group_of_contracts: the IFRS 17 preset ---------------------------------

def test_group_of_contracts_defaults_from_stamped_model_points():
    """m only: cohort from issue_year, portfolio from product_code, prof derived."""
    mp = _two_contracts(
        product_code=np.array(["TL", "TL"]),
        issue_date=np.array(["2026-03-01", "2026-07-01"], dtype="datetime64[D]"),
    )
    m = measure(mp, _assumptions())
    g = group_of_contracts(m)                   # mp stamped -> no second arg
    assert g.bel.shape[0] == 2                  # same 2026 cohort, onerous vs remaining
    assert g.loss_component.sum() > 0.0         # the onerous one stands alone


def test_group_of_contracts_single_cohort_without_issue_date():
    """No issue_date: the default cohort falls back to a single annual cohort."""
    mp = _two_contracts(product_code=np.array(["TL", "TL"]))   # no issue_date
    m = measure(mp, _assumptions())
    g = group_of_contracts(m)                   # one product, one cohort, prof derived
    assert g.bel.shape[0] == 2                  # split only by onerous / remaining


def test_group_of_contracts_portfolio_override():
    """Pass another column to group on a different portfolio definition."""
    mp = _two_contracts(
        product_code=np.array(["TL", "TL"]),
        attributes={"portfolio_id": np.array(["P1", "P2"])},
    )
    m = measure(mp, _assumptions())
    # same product / cohort, but two portfolio_id values -> two groups
    assert group_of_contracts(m, portfolio="portfolio_id").bel.shape[0] == 2
    assert group_of_contracts(m).bel.shape[0] == 2          # product_code: still 2 (onerous split)


def test_group_of_contracts_splits_by_issue_year():
    mp = _two_contracts(
        product_code=np.array(["TL", "TL"]),
        issue_date=np.array(["2025-12-01", "2026-01-01"], dtype="datetime64[D]"),
    )
    m = measure(mp, _assumptions())
    # 2025 vs 2026 cohort (paragraph 22) -- two groups regardless of profitability
    assert group_of_contracts(m).bel.shape[0] == 2


def test_group_of_contracts_finer_cohort_from_data_column():
    """A finer cohort (issue_quarter) carried in the data groups by that column."""
    mp = _two_contracts(
        product_code=np.array(["TL", "TL"]),
        attributes={"issue_quarter": np.array(["2026Q1", "2026Q2"])},
    )
    m = measure(mp, _assumptions())
    assert group_of_contracts(m, cohort="issue_quarter").bel.shape[0] == 2


def test_group_of_contracts_profitability_override_pools():
    """A uniform profitability override pools the onerous loss against the gain."""
    mp = _two_contracts(product_code=np.array(["TL", "TL"]))
    m = measure(mp, _assumptions())
    uniform = np.array(["remaining", "remaining"])
    together = group_of_contracts(m, profitability=uniform)   # forced one group
    apart = group_of_contracts(m)                             # derived -> 2 groups
    assert together.bel.shape[0] == 1
    assert apart.bel.shape[0] == 2
    assert together.loss_component.sum() < apart.loss_component.sum()


def test_group_of_contracts_profitability_from_column():
    """A string profitability names a stored (locked) classification column."""
    mp = _two_contracts(
        product_code=np.array(["TL", "TL"]),
        attributes={"locked_prof": np.array(["remaining", "remaining"])},
    )
    m = measure(mp, _assumptions())
    assert group_of_contracts(m, profitability="locked_prof").bel.shape[0] == 1


def test_group_of_contracts_needs_model_points():
    m = measure(_two_contracts(), _assumptions())
    object.__setattr__(m, "model_points", None)
    with pytest.raises(ValueError, match="needs the model points"):
        group_of_contracts(m)


def test_group_of_contracts_explicit_unknown_axis_raises():
    """An explicit (non-default) axis name that is missing is a typo, not a fallback."""
    m = measure(_two_contracts(product_code=np.array(["TL", "TL"])), _assumptions())
    with pytest.raises(KeyError, match="unknown grouping axis"):
        group_of_contracts(m, portfolio="typo_axis")


def test_group_of_contracts_unsupported_type_raises():
    """The base dispatch refuses a type with no registration (e.g. reinsurance)."""
    class _Other:
        pass
    with pytest.raises(TypeError, match="not implemented"):
        group_of_contracts(_Other())


# -- review fixes (5-way code review) ---------------------------------------

def test_group_rejects_pipe_in_axis_value():
    """A '|' in an axis value is rejected -- it would collide distinct groups."""
    mp = _two_contracts(product_code=np.array(["TL", "TL"]),
                        attributes={"x": np.array(["a|b", "a"])})
    m = measure(mp, _assumptions())
    with pytest.raises(ValueError, match="character"):
        group(m, by=["product_code", "x"])


def test_axis_resolves_engine_native_field_issue_class():
    """issue_class (위험등급) is a grouping axis even though it is a reserved field."""
    m = measure(_two_contracts(issue_class=np.array([0, 1])), _assumptions())
    assert group(m, by=["issue_class"]).bel.shape[0] == 2      # two classes -> two groups
