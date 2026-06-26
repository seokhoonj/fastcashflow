"""ExpenseItem + derive_expense_components -- the item-form authoring shape
for the expense ledger and its projection onto the six kernel-side
primitives the compiled time loop consumes.

Basis names follow the Korean actuarial alpha / beta / gamma convention
plus a dedicated LAE (Loss Adjustment Expense) slot and a
surrender-value-linked maintenance slot:

    alpha_pro_rata / alpha_fixed -- acquisition (init, t=0)
    beta_pro_rata                -- maintenance pro-rated on premium
    gamma_fixed                  -- maintenance per-policy fixed
    lae_pro_rata                 -- LAE on claim-type outflow
    surrender_value_pro_rata     -- maintenance on the in-force surrender value
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import EXPENSE_BASES, ExpenseItem, derive_expense_components, CoverageRate
from conftest import PATTERNS

def test_empty_rows_emit_zero_primitives():
    """Empty rows: every primitive is zero across the projection horizon."""
    (alpha_pro_rata, alpha_fixed, beta_pro_rata,
     gamma_fixed, lae_pro_rata, _surr) = derive_expense_components((), 24)
    assert alpha_pro_rata == 0.0 and alpha_fixed == 0.0 and beta_pro_rata == 0.0
    assert gamma_fixed.shape == (24,) and lae_pro_rata.shape == (24,)
    assert np.all(gamma_fixed == 0.0) and np.all(lae_pro_rata == 0.0)


def test_alpha_fixed_row_lands_in_alpha_fixed_primitive():
    """An ``alpha_fixed`` row contributes only to the alpha_fixed primitive."""
    rows = (ExpenseItem("acquisition", "alpha_fixed", 50_000.0),)
    (alpha_pro_rata, alpha_fixed, beta_pro_rata,
     gamma_fixed, lae_pro_rata, _surr) = derive_expense_components(rows, 12)
    assert alpha_fixed == 50_000.0
    assert alpha_pro_rata == 0.0 and beta_pro_rata == 0.0
    assert np.all(gamma_fixed == 0.0) and np.all(lae_pro_rata == 0.0)


def test_alpha_pro_rata_row_lands_in_alpha_pro_rata_primitive():
    """An ``alpha_pro_rata`` row contributes only to alpha_pro_rata."""
    rows = (ExpenseItem("acquisition", "alpha_pro_rata", 1.20),)
    (alpha_pro_rata, alpha_fixed, beta_pro_rata,
     gamma_fixed, lae_pro_rata, _surr) = derive_expense_components(rows, 12)
    assert alpha_pro_rata == 1.20
    assert alpha_fixed == 0.0 and beta_pro_rata == 0.0


def test_beta_pro_rata_row_lands_in_beta_pro_rata_primitive():
    """A ``beta_pro_rata`` row contributes only to beta_pro_rata."""
    rows = (ExpenseItem("collection", "beta_pro_rata", 0.01),)
    (alpha_pro_rata, alpha_fixed, beta_pro_rata,
     gamma_fixed, lae_pro_rata, _surr) = derive_expense_components(rows, 12)
    assert beta_pro_rata == 0.01


def test_gamma_fixed_grows_with_inflation():
    """A ``gamma_fixed`` row's monthly amount is ``value/12 * inflation_index[t]``.

    Inflation is the macro-economic assumption on ``Basis``, not a
    row attribute; the helper takes the curve as a parameter. At ``t=0``
    the multiplier is 1; at ``t=12`` it equals ``1 + inflation``.
    """
    rows = (
        ExpenseItem("maintenance", "gamma_fixed", 36_000.0),
    )
    n_time = 24
    infl = (1.03) ** (np.arange(n_time) / 12.0)
    _, _, _, gamma_fixed, _, _ = derive_expense_components(rows, n_time, infl)
    assert gamma_fixed[0] == pytest.approx(36_000.0 / 12.0)
    assert gamma_fixed[12] == pytest.approx((36_000.0 / 12.0) * 1.03)
    assert gamma_fixed[n_time - 1] == pytest.approx(
        (36_000.0 / 12.0) * (1.03) ** ((n_time - 1) / 12.0)
    )


def test_lae_pro_rata_grows_with_inflation():
    """A ``lae_pro_rata`` row's monthly fraction grows with the inflation curve."""
    rows = (ExpenseItem("LAE", "lae_pro_rata", 0.02),)
    n_time = 24
    infl = (1.02) ** (np.arange(n_time) / 12.0)
    _, _, _, _, lae_pro_rata, _ = derive_expense_components(rows, n_time, infl)
    assert lae_pro_rata[0] == pytest.approx(0.02)
    assert lae_pro_rata[12] == pytest.approx(0.02 * 1.02)


def test_multiple_rows_sum_into_each_primitive():
    """When several rows share a basis, their values add up."""
    rows = (
        ExpenseItem("acquisition", "alpha_fixed", 50_000.0),
        ExpenseItem("acquisition_extra", "alpha_fixed", 10_000.0),
        ExpenseItem("maintenance", "gamma_fixed", 36_000.0),
        ExpenseItem("overhead", "gamma_fixed", 12_000.0),
    )
    _, alpha_fixed, _, gamma_fixed, _, _ = derive_expense_components(rows, 12)
    assert alpha_fixed == 60_000.0
    # Two maintenance rows sum, the second has zero inflation.
    assert gamma_fixed[0] == pytest.approx((36_000.0 + 12_000.0) / 12.0)


def test_unknown_basis_raises():
    """An unrecognised basis is flagged loudly at construction, with the
    supported list -- not deferred to derive_expense_components at measure."""
    with pytest.raises(ValueError, match="unknown expense basis"):
        ExpenseItem("???", "yearly_payroll", 1000.0)


def test_expense_bases_constant_is_complete():
    """``EXPENSE_BASES`` matches the dispatch table the helper implements."""
    for basis in EXPENSE_BASES:
        rows = (ExpenseItem("x", basis, 1.0),)
        derive_expense_components(rows, 12)             # no error
    assert len(EXPENSE_BASES) == 6


def test_helper_exported_at_package_level():
    """The new authoring surface is reachable as ``fcf.*``."""
    assert hasattr(fcf, "ExpenseItem")
    assert hasattr(fcf, "EXPENSE_BASES")
    assert hasattr(fcf, "derive_expense_components")


# ---------------------------------------------------------------------------
# Wiring -- expense_items reaches the kernels and the empty-tuple state is
# a clean no-expense basis (zero expense_cf, value/measure agree).
# ---------------------------------------------------------------------------

def _term_life_mp():
    """A single-policy fixture exercising the measure() and measure() paths."""
    return fcf.ModelPoints.single(
        issue_age=40,
        benefits={"DEATH": 100_000_000.0},
        premium=50_000.0,
        term_months=120,
        calculation_methods=PATTERNS,
    )


def _basis_rows():
    """A populated expense ledger -- 4 rows covering every pre-claim basis."""
    import numpy as np

    def mort(s, ia, d, ic, em):
        return np.full(d.shape, 0.0008)

    def lapse(s, ia, d, ic, em):
        return np.full(d.shape, 0.05)

    return fcf.Basis(
        mortality_annual=mort, lapse_annual=lapse,
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.05,
        expense_inflation=0.03,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    120_000.0),
            ExpenseItem("acquisition",  "alpha_pro_rata",        0.20),
            ExpenseItem("collection",   "beta_pro_rata",         0.02),
            ExpenseItem("maintenance",  "gamma_fixed",       36_000.0),
        ),
        coverages=(CoverageRate("DEATH", mort),),
    )


def test_lae_pro_rata_row_lifts_expense():
    """Adding an LAE row raises the expense cash flow in every month with
    any claim activity -- the new line the engine could not express
    before the item form."""
    import dataclasses
    import numpy as np
    mp = _term_life_mp()
    base_rows = _basis_rows()
    with_lae = dataclasses.replace(
        base_rows,
        expense_items=base_rows.expense_items + (
            ExpenseItem("LAE", "lae_pro_rata", 0.02),
        ),
    )
    m_base = fcf.gmm.measure(mp, base_rows)
    m_lae = fcf.gmm.measure(mp, with_lae)
    assert np.all(m_lae.cashflows.expense_cf >=
                  m_base.cashflows.expense_cf - 1e-9)
    # Strictly higher in months where the policy has any claim flow.
    has_claim = (m_base.cashflows.mortality_cf[0]
                 + m_base.cashflows.morbidity_cf[0]) > 0.0
    assert np.any(m_lae.cashflows.expense_cf[0, has_claim]
                  > m_base.cashflows.expense_cf[0, has_claim])


def test_empty_expense_items_is_zero_expense_basis():
    """Default ``expense_items=()`` produces a zero expense cash flow and
    a strictly lower (more profitable) BEL than the populated basis."""
    import dataclasses
    import numpy as np
    mp = _term_life_mp()
    populated = _basis_rows()
    empty = dataclasses.replace(populated, expense_items=())
    m_empty = fcf.gmm.measure(mp, empty)
    v_empty = fcf.gmm.measure(mp, empty, full=False)
    assert np.all(m_empty.cashflows.expense_cf == 0.0)
    assert np.isclose(m_empty.bel_path[0, 0], v_empty.bel[0])
    # populated basis has expense outflows, so it must have a higher BEL
    populated_bel = fcf.gmm.measure(mp, populated, full=False).bel[0]
    assert populated_bel > v_empty.bel[0]


# ---------------------------------------------------------------------------
# surrender_value_pro_rata -- maintenance charged on the in-force surrender
# value (the Korean "% of surrender-reserve" reserve-linked maintenance). Unlike the
# other bases its kernel input is a single scalar rate: the base (the in-force
# surrender value) is built post-projection, where the in-force path is known.
# ---------------------------------------------------------------------------

def test_surrender_value_pro_rata_lands_in_sixth_primitive():
    """A ``surrender_value_pro_rata`` row contributes only to the 6th primitive
    -- a scalar annual rate with NO inflation applied (it rides the surrender
    curve's own growth)."""
    rows = (ExpenseItem("maintenance", "surrender_value_pro_rata", 0.004),)
    infl = (1.05) ** (np.arange(12) / 12.0)        # would inflate an inflating basis
    a_pr, a_fx, b_pr, gamma, lae, surr = derive_expense_components(rows, 12, infl)
    assert surr == 0.004                            # scalar, not inflated
    assert a_pr == 0.0 and a_fx == 0.0 and b_pr == 0.0
    assert np.all(gamma == 0.0) and np.all(lae == 0.0)


def _surrender_basis(extra_items=(), value=2_000_000.0, n_time=120):
    """Term-life basis with a flat amount-per-policy surrender curve."""
    def mort(s, ia, d, ic, em):
        return np.full(d.shape, 0.01)

    def lapse(s, ia, d, ic, em):
        return np.full(d.shape, 0.06)

    basis = fcf.Basis(
        mortality_annual=mort, lapse_annual=lapse,
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.05,
        surrender_value_curve=np.full(n_time, value),
        surrender_value_basis="amount_per_policy",
        expense_items=extra_items,
        coverages=(CoverageRate("DEATH", mort),),
    )
    return basis, value


def test_surrender_value_pro_rata_hand_calc():
    """The surrender-linked expense each month equals ``rate/12 x value x
    inforce[t]`` -- charged on the begin-of-month in-force surrender value, not
    the lapsing exits. Hand-checked against the in-force path (read off the
    baseline run; the expense does not change the in-force)."""
    mp = _term_life_mp()
    rate = 0.004
    base, value = _surrender_basis()
    with_surr, _ = _surrender_basis(
        (ExpenseItem("maintenance", "surrender_value_pro_rata", rate),))
    m0 = fcf.gmm.measure(mp, base)
    m1 = fcf.gmm.measure(mp, with_surr)
    inforce = m0.cashflows.inforce              # identical in both runs
    delta = m1.cashflows.expense_cf - m0.cashflows.expense_cf
    expected = (rate / 12.0) * value * inforce
    assert np.allclose(delta, expected, rtol=0, atol=1e-6)
    # flat curve, no inflation -- the per-unit-of-in-force charge is the same
    # at t=0 and t=12.
    assert delta[0, 0] == pytest.approx((rate / 12.0) * value * inforce[0, 0])
    # a positive expense leg raises the BEL.
    assert m1.bel_path[0, 0] > m0.bel_path[0, 0]


def test_surrender_value_pro_rata_routes_fast_to_full():
    """The fused fast path does not carry the in-force surrender value, so a book
    with this item routes to the full path (``requires_full``); the fast call
    returns the full-path BEL."""
    mp = _term_life_mp()
    with_surr, _ = _surrender_basis(
        (ExpenseItem("maintenance", "surrender_value_pro_rata", 0.004),))
    m_full = fcf.gmm.measure(mp, with_surr, full=True)
    m_fast = fcf.gmm.measure(mp, with_surr, full=False)
    assert np.isclose(m_full.bel_path[0, 0], m_fast.bel[0])


def test_surrender_value_pro_rata_requires_curve():
    """A ``surrender_value_pro_rata`` item with no ``surrender_value_curve``
    errors at measure time -- the expense base would otherwise be silently
    zero."""
    def mort(s, ia, d, ic, em):
        return np.full(d.shape, 0.01)

    def lapse(s, ia, d, ic, em):
        return np.full(d.shape, 0.06)

    basis = fcf.Basis(
        mortality_annual=mort, lapse_annual=lapse, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.05,
        expense_items=(
            ExpenseItem("maintenance", "surrender_value_pro_rata", 0.004),),
        coverages=(CoverageRate("DEATH", mort),),
    )
    with pytest.raises(ValueError, match="requires Basis.surrender_value_curve"):
        fcf.gmm.measure(_term_life_mp(), basis)


def test_surrender_value_pro_rata_rejected_on_account_book():
    """The item is undefined on an account-backed (UL / VFA) book -- the account
    fund_fee already charges the account value, so a second charge here would
    double-count. Rejected at measure time."""
    import dataclasses
    basis = fcf.samples.basis(template="ul")
    mp = fcf.samples.model_points(template="ul")
    n_time = basis.contract_boundary if hasattr(basis, "contract_boundary") else 600
    account_basis = dataclasses.replace(
        basis,
        surrender_value_curve=np.full(n_time, 1_000_000.0),
        surrender_value_basis="amount_per_policy",
        expense_items=basis.expense_items + (
            ExpenseItem("maintenance", "surrender_value_pro_rata", 0.004),),
    )
    with pytest.raises(ValueError, match="account-backed"):
        fcf.gmm.measure(mp, account_basis)
