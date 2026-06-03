"""IFRS 17 level of aggregation -- general group(by=...) + group_into_gic preset.

``group`` aggregates a measurement to any axis (resolved from the model points
``measure`` stamps on the result, or passed explicitly); ``group_into_gic`` is
the IFRS 17 preset. The architecture point is ``CSM(sum FCF)``, not
``sum CSM(CF)`` -- the floor nets within a group, not across.
"""
import numpy as np
import pytest

from fastcashflow import (ExpenseItem, ModelPoints, assign_gic, group,
                          group_into_gic)
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


# -- assign_gic: the pure composite-key builder -----------------------------

def test_assign_gic_composes_three_axes():
    gic = assign_gic(
        portfolio     = np.array(["healthA", "healthA", "healthA"]),
        cohort        = np.array([2026, 2026, 2025]),
        profitability = np.array(["remaining", "onerous", "remaining"]),
    )
    assert gic[0] == "healthA|2026|remaining"
    assert gic[0] != gic[1]      # profitability differs (paragraph 16)
    assert gic[0] != gic[2]      # cohort differs (paragraph 22)


def test_assign_gic_is_label_agnostic_two_or_three_way():
    pf, co = np.array(["A", "A"]), np.array([2026, 2026])
    two = assign_gic(pf, co, np.array(["onerous", "remaining"]))
    three = assign_gic(pf, co, np.array(["onerous", "no_significant_possibility"]))
    assert two[0] != two[1] and three[0] != three[1]   # no engine vocabulary


def test_assign_gic_length_mismatch_rejected():
    with pytest.raises(ValueError, match="same length"):
        assign_gic(portfolio=np.array(["A", "B"]), cohort=np.array([2026]),
                   profitability=np.array(["x", "y"]))


# -- group: the general aggregator ------------------------------------------

def test_group_by_precomputed_array():
    """A raw label array still works -- group(m, ids)."""
    m = measure(_two_contracts(), _assumptions())
    assert group(m, np.array([0, 0])).bel.shape[0] == 1
    assert group(m, np.array([0, 1])).bel.shape[0] == 2


def test_group_by_axis_names_from_stamped_model_points():
    """group(m, by=[names]) resolves from the model points measure() stamped."""
    mp = _two_contracts(product_code=np.array(["TL", "TL"]),
                        channel_code=np.array(["TM", "GA"]))
    m = measure(mp, _assumptions())                 # stamps mp on m
    assert group(m, by=["product_code"]).bel.shape[0] == 1   # one product
    assert group(m, by=["channel"]).bel.shape[0] == 2        # two channels


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


# -- group_into_gic: the IFRS 17 preset -------------------------------------

def test_group_into_gic_defaults_from_stamped_model_points():
    """m only: cohort from issue_date, portfolio from product_code, prof derived."""
    mp = _two_contracts(
        product_code=np.array(["TL", "TL"]),
        issue_date=np.array(["2026-03-01", "2026-07-01"], dtype="datetime64[D]"),
    )
    m = measure(mp, _assumptions())
    g = group_into_gic(m)                       # mp stamped -> no second arg
    assert g.bel.shape[0] == 2                  # same 2026 cohort, onerous vs remaining
    assert g.loss_component.sum() > 0.0         # the onerous one stands alone


def test_group_into_gic_portfolio_id_overrides_product():
    mp = _two_contracts(
        product_code=np.array(["TL", "TL"]),
        attributes={"portfolio_id": np.array(["P1", "P2"]),
                    "profitability_group": np.array(["remaining", "remaining"])},
    )
    m = measure(mp, _assumptions())
    # different portfolio_id -> two GICs even with the same product / cohort / prof
    assert group_into_gic(m).bel.shape[0] == 2


def test_group_into_gic_splits_by_issue_year():
    mp = _two_contracts(
        product_code=np.array(["TL", "TL"]),
        issue_date=np.array(["2025-12-01", "2026-01-01"], dtype="datetime64[D]"),
        attributes={"profitability_group": np.array(["remaining", "remaining"])},
    )
    m = measure(mp, _assumptions())
    assert group_into_gic(m).bel.shape[0] == 2   # 2025 vs 2026 cohort (paragraph 22)


def test_group_into_gic_pools_when_one_gic():
    """CSM(sum CF): one GIC nets the onerous loss against the profitable one."""
    pooled = measure(_two_contracts(
        product_code=np.array(["TL", "TL"]),
        attributes={"profitability_group": np.array(["remaining", "remaining"])},
    ), _assumptions())
    split = measure(_two_contracts(           # no profitability_group -> derive
        product_code=np.array(["TL", "TL"]),
    ), _assumptions())
    together = group_into_gic(pooled)         # forced one GIC
    apart = group_into_gic(split)             # derived onerous / remaining -> 2 GICs
    assert together.bel.shape[0] == 1
    assert apart.bel.shape[0] == 2
    assert together.loss_component.sum() < apart.loss_component.sum()


def test_group_into_gic_needs_model_points():
    m = measure(_two_contracts(), _assumptions())
    object.__setattr__(m, "model_points", None)
    with pytest.raises(ValueError, match="needs the model points"):
        group_into_gic(m)


# -- review fixes (5-way code review) ---------------------------------------

def test_group_rejects_pipe_in_axis_value():
    """A '|' in an axis value is rejected -- it would collide distinct groups."""
    mp = _two_contracts(product_code=np.array(["TL", "TL"]),
                        attributes={"x": np.array(["a|b", "a"])})
    m = measure(mp, _assumptions())
    with pytest.raises(ValueError, match="character"):
        group(m, by=["x"])


def test_assign_gic_rejects_pipe_in_axis_value():
    with pytest.raises(ValueError, match="character"):
        assign_gic(np.array(["p|q"]), np.array([2026]), np.array(["onerous"]))


def test_axis_resolves_engine_native_field_issue_class():
    """issue_class (위험등급) is a grouping axis even though it is a reserved field."""
    m = measure(_two_contracts(issue_class=np.array([0, 1])), _assumptions())
    assert group(m, by=["issue_class"]).bel.shape[0] == 2      # two classes -> two groups


def test_group_into_gic_explicit_unknown_axis_raises():
    """An explicit (non-default) axis name that is missing is a typo, not a fallback."""
    m = measure(_two_contracts(product_code=np.array(["TL", "TL"])), _assumptions())
    with pytest.raises(KeyError, match="unknown grouping axis"):
        group_into_gic(m, portfolio="typo_axis")
