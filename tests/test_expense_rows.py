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


# ---------------------------------------------------------------------------
# Wiring -- expense_rows reaches the kernels and produces the same BEL / CSM
# as the equivalent legacy alpha / beta / gamma scalars.
# ---------------------------------------------------------------------------

def _term_life_mp():
    """A single-policy fixture exercising the value() and measure() paths."""
    return fcf.ModelPoints.single(
        issue_age=40,
        death_benefit=100_000_000.0,
        level_premium=50_000.0,
        term_months=120,
    )


def _basis_legacy():
    """Legacy alpha / beta / gamma scalars (the pre-row form)."""
    import numpy as np

    def mort(s, ia, d, ic, em):
        return np.full(d.shape, 0.0008)

    def lapse(s, ia, d, ic, em):
        return np.full(d.shape, 0.05)

    return fcf.Assumptions(
        mortality_annual=mort, lapse_annual=lapse,
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.05,
        alpha_flat=120_000.0,
        alpha_pct=0.20,
        beta_pct=0.02,
        gamma_flat=36_000.0,
        expense_inflation=0.03,
    )


def _basis_rows():
    """The same expense ledger expressed as ExpenseRow tuples."""
    import dataclasses
    legacy = _basis_legacy()
    rows = (
        ExpenseRow("acquisition", "per_policy_init", 120_000.0),
        ExpenseRow("acquisition", "premium_pct_init", 0.20),
        ExpenseRow("collection", "premium_pct", 0.02),
        ExpenseRow("maintenance", "per_policy_monthly", 36_000.0,
                   inflation_rate=0.03),
    )
    # Row-form takes precedence; clear the legacy scalars so the dispatch
    # is unambiguous (Assumptions accepts both during the migration).
    return dataclasses.replace(
        legacy,
        alpha_flat=0.0, alpha_pct=0.0, beta_pct=0.0,
        gamma_flat=0.0, expense_inflation=0.0,
        expense_rows=rows,
    )


def test_rows_match_legacy_value_path():
    """value() with the row-form basis matches the legacy scalar basis
    when the rows describe the same expense ledger."""
    import numpy as np
    mp = _term_life_mp()
    v_legacy = fcf.value(mp, _basis_legacy())
    v_rows = fcf.value(mp, _basis_rows())
    assert np.isclose(v_legacy.bel[0], v_rows.bel[0])
    assert np.isclose(v_legacy.ra[0], v_rows.ra[0])
    assert np.isclose(v_legacy.csm[0], v_rows.csm[0])
    assert np.isclose(v_legacy.loss_component[0], v_rows.loss_component[0])


def test_rows_match_legacy_measure_path():
    """measure() likewise agrees -- the detailed projection sees the same
    expense_cf trajectory from both routes."""
    import numpy as np
    mp = _term_life_mp()
    m_legacy = fcf.measure(mp, _basis_legacy())
    m_rows = fcf.measure(mp, _basis_rows())
    assert np.allclose(m_legacy.cashflows.expense_cf,
                       m_rows.cashflows.expense_cf)
    assert np.isclose(m_legacy.bel[0, 0], m_rows.bel[0, 0])


def test_claim_pct_row_raises_expense_above_legacy():
    """A claim_pct row adds a new line the legacy scalars cannot express
    (지급비), so the same baseline plus a claim_pct row produces a strictly
    higher expense_cf and a different BEL."""
    import dataclasses
    import numpy as np
    mp = _term_life_mp()
    base_rows = _basis_rows()
    with_claim = dataclasses.replace(
        base_rows,
        expense_rows=base_rows.expense_rows + (
            ExpenseRow("claim_handling", "claim_pct", 0.02),
        ),
    )
    m_base = fcf.measure(mp, base_rows)
    m_claim = fcf.measure(mp, with_claim)
    assert np.all(m_claim.cashflows.expense_cf >=
                  m_base.cashflows.expense_cf - 1e-9)
    # Strictly higher in months where the policy has any claim flow.
    has_claim = (m_base.cashflows.claim_cf[0]
                 + m_base.cashflows.morbidity_cf[0]) > 0.0
    assert np.any(m_claim.cashflows.expense_cf[0, has_claim]
                  > m_base.cashflows.expense_cf[0, has_claim])


def test_empty_expense_rows_uses_legacy_path():
    """Default ``expense_rows=()`` is the migration-friendly state: the
    legacy alpha / beta / gamma scalars drive the kernel exactly as
    before."""
    import dataclasses
    import numpy as np
    mp = _term_life_mp()
    legacy = _basis_legacy()
    legacy_with_empty_rows = dataclasses.replace(legacy, expense_rows=())
    v_legacy = fcf.value(mp, legacy)
    v_empty = fcf.value(mp, legacy_with_empty_rows)
    assert np.isclose(v_legacy.bel[0], v_empty.bel[0])
