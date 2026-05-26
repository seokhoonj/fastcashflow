"""Aggregation validation -- grouping a measurement into IFRS 17 groups.

The CSM floor applies to the group: contracts within a group are netted
before ``max(0, ...)``, contracts in different groups are not.
"""
import numpy as np
import pytest

from fastcashflow import Assumptions, ExpenseRow, ModelPoints, group, measure, roll_forward


def _annual(m):
    """Convert a monthly rate to the equivalent annual rate the engine expects."""
    return 1.0 - (1.0 - m) ** 12


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.002)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.01)),
        discount_annual=0.03,
        expense_inflation=0.02,
        expense_rows=(
            ExpenseRow("acquisition",  "per_policy_init",    100_000.0),
            ExpenseRow("maintenance",  "per_policy_monthly",  60_000.0),
        ),
        ra_confidence=0.75,
        mortality_cv=0.10,
    )


def _portfolio(n: int = 60) -> ModelPoints:
    rng = np.random.default_rng(7)
    return ModelPoints(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(20, 90, n) * 1_000_000,
        level_premium=rng.integers(8, 20, n) * 10_000,
        term_months=np.full(n, 120),
    )


def _two_contracts() -> ModelPoints:
    """Two term-life model points -- the first profitable, the second onerous."""
    return ModelPoints(
        issue_age=np.array([40, 40]),
        death_benefit=np.array([1e8, 1e8]),
        level_premium=np.array([300_000.0, 60_000.0]),
        term_months=np.array([120, 120]),
    )


def test_group_count_and_sums():
    """group() returns one row per group; BEL is summed within each."""
    m = measure(_portfolio(), _assumptions())
    g = group(m, np.arange(len(m.bel)) % 3)
    assert g.bel.shape[0] == 3
    assert np.isclose(g.bel[:, 0].sum(), m.bel[:, 0].sum())   # BEL is additive


def test_group_nets_within_a_group_not_across():
    """A profitable contract absorbs an onerous one only inside the same group."""
    m = measure(_two_contracts(), _assumptions())
    together = group(m, np.array([0, 0]))          # one group
    apart = group(m, np.array([0, 1]))             # two groups
    assert apart.loss_component.sum() > 0.0        # the onerous one stands alone
    assert together.loss_component[0] < apart.loss_component.sum()


def test_group_csm_reconciles():
    """The grouped CSM trajectory reconciles."""
    m = measure(_portfolio(), _assumptions())
    g = group(m, np.arange(len(m.bel)) % 4)
    step = g.csm[:, :-1] + g.csm_accretion - g.csm_release
    assert np.allclose(step, g.csm[:, 1:])


def test_group_composes_with_roll_forward():
    """A grouped measurement flows into the period-close roll-forward."""
    m = measure(_portfolio(), _assumptions())
    g = group(m, np.arange(len(m.bel)) % 5)
    periods = roll_forward(g, 12)
    assert periods[0].csm_opening.shape == (5,)


def test_group_rejects_wrong_length():
    """group_ids must have one entry per model point."""
    m = measure(_portfolio(), _assumptions())
    with pytest.raises(ValueError, match="one entry per model point"):
        group(m, np.array([0, 1, 2]))
