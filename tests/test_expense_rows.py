"""ExpenseRow + derive_expense_components -- the row-form authoring shape
for the expense ledger and its projection onto the five kernel-side
primitives the compiled time loop consumes.

The dataclass and helper land first (this commit); the kernel,
io.py and sample workbook migrate to consume the row form in
follow-up commits.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import EXPENSE_BASES, ExpenseRow, derive_expense_components


def test_empty_rows_emit_zero_primitives():
    """Empty rows: every primitive is zero across the projection horizon."""
    a_pct, a_flat, b_pct, gamma, claim_pct = derive_expense_components((), 24)
    assert a_pct == 0.0 and a_flat == 0.0 and b_pct == 0.0
    assert gamma.shape == (24,) and claim_pct.shape == (24,)
    assert np.all(gamma == 0.0) and np.all(claim_pct == 0.0)


def test_per_policy_init_lands_in_alpha_flat():
    """A ``per_policy_init`` row contributes only to alpha_flat."""
    rows = (ExpenseRow("acquisition", "per_policy_init", 50_000.0),)
    a_pct, a_flat, b_pct, gamma, claim_pct = derive_expense_components(rows, 12)
    assert a_flat == 50_000.0
    assert a_pct == 0.0 and b_pct == 0.0
    assert np.all(gamma == 0.0) and np.all(claim_pct == 0.0)


def test_premium_pct_init_lands_in_alpha_pct():
    """A ``premium_pct_init`` row contributes only to alpha_pct."""
    rows = (ExpenseRow("acquisition", "premium_pct_init", 1.20),)
    a_pct, a_flat, b_pct, gamma, claim_pct = derive_expense_components(rows, 12)
    assert a_pct == 1.20
    assert a_flat == 0.0 and b_pct == 0.0


def test_premium_pct_lands_in_beta_pct():
    """A ``premium_pct`` row contributes only to beta_pct."""
    rows = (ExpenseRow("collection", "premium_pct", 0.01),)
    a_pct, a_flat, b_pct, gamma, claim_pct = derive_expense_components(rows, 12)
    assert b_pct == 0.01


def test_per_policy_monthly_grows_with_inflation():
    """A maintenance row's monthly amount is ``value/12 * (1+infl)^(t/12)``.

    At ``t=0`` the factor is 1; at ``t=12`` it equals ``1+inflation``.
    """
    rows = (
        ExpenseRow("maintenance", "per_policy_monthly", 36_000.0,
                   inflation_rate=0.03),
    )
    _, _, _, gamma, _ = derive_expense_components(rows, 24)
    assert gamma[0] == pytest.approx(36_000.0 / 12.0)
    assert gamma[12] == pytest.approx((36_000.0 / 12.0) * 1.03)
    assert gamma[24 - 1] == pytest.approx(
        (36_000.0 / 12.0) * (1.03) ** (23 / 12.0)
    )


def test_claim_pct_grows_with_inflation():
    """A ``claim_pct`` row's monthly fraction grows like ``(1+infl)^(t/12)``."""
    rows = (ExpenseRow("claim_handling", "claim_pct", 0.02,
                       inflation_rate=0.02),)
    _, _, _, _, claim_pct = derive_expense_components(rows, 24)
    assert claim_pct[0] == pytest.approx(0.02)
    assert claim_pct[12] == pytest.approx(0.02 * 1.02)


def test_multiple_rows_sum_into_each_primitive():
    """When several rows share a basis, their values add up."""
    rows = (
        ExpenseRow("acquisition", "per_policy_init", 50_000.0),
        ExpenseRow("acquisition_extra", "per_policy_init", 10_000.0),
        ExpenseRow("maintenance", "per_policy_monthly", 36_000.0,
                   inflation_rate=0.03),
        ExpenseRow("overhead", "per_policy_monthly", 12_000.0),
    )
    _, a_flat, _, gamma, _ = derive_expense_components(rows, 12)
    assert a_flat == 60_000.0
    # Two maintenance rows sum, the second has zero inflation.
    assert gamma[0] == pytest.approx((36_000.0 + 12_000.0) / 12.0)


def test_unknown_basis_raises():
    """An unrecognised basis is flagged loudly with the supported list."""
    rows = (ExpenseRow("???", "yearly_payroll", 1000.0),)
    with pytest.raises(ValueError, match="unknown expense basis"):
        derive_expense_components(rows, 12)


def test_expense_bases_constant_is_complete():
    """``EXPENSE_BASES`` matches the dispatch table the helper implements."""
    for basis in EXPENSE_BASES:
        rows = (ExpenseRow("x", basis, 1.0),)
        derive_expense_components(rows, 12)             # no error
    assert len(EXPENSE_BASES) == 5


def test_helper_exported_at_package_level():
    """The new authoring surface is reachable as ``fcf.*``."""
    assert hasattr(fcf, "ExpenseRow")
    assert hasattr(fcf, "EXPENSE_BASES")
    assert hasattr(fcf, "derive_expense_components")
