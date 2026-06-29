"""ExpenseItem + derive_expense_components -- the item-form authoring shape
for the expense ledger and its projection onto the seven kernel-side
primitives the compiled time loop consumes.

An expense is two axes: ``category`` (WHAT for) x ``base`` (proportional to
WHAT). The first three categories map to the Korean actuarial alpha / beta /
gamma convention -- acquisition = alpha, maintenance = beta, collection =
gamma; ``lae`` is the Loss Adjustment Expense. The internal primitive is named
``<category>_<base>``, matching the external pair:

    (acquisition, per_policy) / (acquisition, premium)  -- at issue (t=0)
    (maintenance, premium)                              -- % premium
    (maintenance, per_policy)                           -- per-policy fixed
    (maintenance, surrender_value)                      -- % in-force CSV
    (maintenance, face)                                 -- % sum assured
    (collection,  premium)                              -- % premium (collection)
    (lae,         claim)                                -- on claim outflow
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import EXPENSE_BASES, ExpenseItem, derive_expense_components, CoverageRate
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis

def test_empty_rows_emit_zero_primitives():
    """Empty rows: every primitive is zero across the projection horizon."""
    (acquisition_premium, acquisition_per_policy, maintenance_premium,
     maintenance_per_policy, lae, _surr, _face) = derive_expense_components((), 24)
    assert acquisition_premium == 0.0 and acquisition_per_policy == 0.0 and maintenance_premium == 0.0
    assert maintenance_per_policy.shape == (24,) and lae.shape == (24,)
    assert np.all(maintenance_per_policy == 0.0) and np.all(lae == 0.0)


def test_acquisition_per_policy_row_lands_in_its_primitive():
    """An ``acquisition_per_policy`` row contributes only to the acquisition_per_policy primitive."""
    rows = (ExpenseItem("acquisition", "per_policy", 50_000.0),)
    (acquisition_premium, acquisition_per_policy, maintenance_premium,
     maintenance_per_policy, lae, _surr, _face) = derive_expense_components(rows, 12)
    assert acquisition_per_policy == 50_000.0
    assert acquisition_premium == 0.0 and maintenance_premium == 0.0
    assert np.all(maintenance_per_policy == 0.0) and np.all(lae == 0.0)


def test_acquisition_premium_row_lands_in_its_primitive():
    """An ``acquisition_premium`` row contributes only to acquisition_premium."""
    rows = (ExpenseItem("acquisition", "premium", 1.20),)
    (acquisition_premium, acquisition_per_policy, maintenance_premium,
     maintenance_per_policy, lae, _surr, _face) = derive_expense_components(rows, 12)
    assert acquisition_premium == 1.20
    assert acquisition_per_policy == 0.0 and maintenance_premium == 0.0


def test_maintenance_premium_row_lands_in_its_primitive():
    """A ``maintenance_premium`` row contributes only to maintenance_premium."""
    rows = (ExpenseItem("maintenance", "premium", 0.01),)
    (acquisition_premium, acquisition_per_policy, maintenance_premium,
     maintenance_per_policy, lae, _surr, _face) = derive_expense_components(rows, 12)
    assert maintenance_premium == 0.01


def test_maintenance_per_policy_grows_with_inflation():
    """A ``maintenance_per_policy`` row's monthly amount is ``value/12 * inflation_index[t]``.

    Inflation is the macro-economic assumption on ``Basis``, not a
    row attribute; the helper takes the curve as a parameter. At ``t=0``
    the multiplier is 1; at ``t=12`` it equals ``1 + inflation``.
    """
    rows = (
        ExpenseItem("maintenance", "per_policy", 36_000.0),
    )
    n_time = 24
    infl = (1.03) ** (np.arange(n_time) / 12.0)
    _, _, _, maintenance_per_policy, _, _, _ = derive_expense_components(rows, n_time, infl)
    assert maintenance_per_policy[0] == pytest.approx(36_000.0 / 12.0)
    assert maintenance_per_policy[12] == pytest.approx((36_000.0 / 12.0) * 1.03)
    assert maintenance_per_policy[n_time - 1] == pytest.approx(
        (36_000.0 / 12.0) * (1.03) ** ((n_time - 1) / 12.0)
    )


def test_lae_grows_with_inflation():
    """A ``lae`` row's monthly fraction grows with the inflation curve."""
    rows = (ExpenseItem("lae", "claim", 0.02),)
    n_time = 24
    infl = (1.02) ** (np.arange(n_time) / 12.0)
    _, _, _, _, lae, _, _ = derive_expense_components(rows, n_time, infl)
    assert lae[0] == pytest.approx(0.02)
    assert lae[12] == pytest.approx(0.02 * 1.02)


def test_multiple_rows_sum_into_each_primitive():
    """When several rows share a basis, their values add up."""
    rows = (
        ExpenseItem("acquisition", "per_policy", 50_000.0),
        ExpenseItem("acquisition", "per_policy", 10_000.0),
        ExpenseItem("maintenance", "per_policy", 36_000.0),
        ExpenseItem("maintenance", "per_policy", 12_000.0),
    )
    _, acquisition_per_policy, _, maintenance_per_policy, _, _, _ = derive_expense_components(rows, 12)
    assert acquisition_per_policy == 60_000.0
    # Two maintenance rows sum, the second has zero inflation.
    assert maintenance_per_policy[0] == pytest.approx((36_000.0 + 12_000.0) / 12.0)


def test_unknown_pair_raises():
    """An unrecognised (category, base) pair is flagged loudly at construction,
    with the supported list -- not deferred to derive at measure."""
    with pytest.raises(ValueError, match="unknown expense .category, base. pair"):
        ExpenseItem("overhead", "yearly_payroll", 1000.0)
    # a valid category with an invalid base for it also raises (lae has no
    # premium base).
    with pytest.raises(ValueError, match="unknown expense .category, base. pair"):
        ExpenseItem("lae", "premium", 0.02)


def test_every_valid_pair_dispatches():
    """Every supported (category, base) pair projects without error."""
    from fastcashflow.basis import _EXPENSE_DISPATCH
    for (category, base) in _EXPENSE_DISPATCH:
        rows = (ExpenseItem(category, base, 1.0),)
        derive_expense_components(rows, 12)             # no error
    assert len(EXPENSE_BASES) == 5              # per_policy / premium / surrender_value / face / claim


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
            ExpenseItem("acquisition", "per_policy",    120_000.0),
            ExpenseItem("acquisition", "premium",        0.20),
            ExpenseItem("maintenance", "premium",         0.02),
            ExpenseItem("maintenance", "per_policy",       36_000.0),
        ),
        coverages=(CoverageRate("DEATH", mort),),
    )


def test_lae_row_lifts_expense():
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
            ExpenseItem("lae", "claim", 0.02),
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
# maintenance_surrender_value -- maintenance charged on the in-force surrender
# value (the Korean "% of surrender-reserve" reserve-linked maintenance). Unlike the
# other bases its kernel input is a single scalar rate: the base (the in-force
# surrender value) is built post-projection, where the in-force path is known.
# ---------------------------------------------------------------------------

def test_maintenance_surrender_value_lands_in_sixth_primitive():
    """A ``maintenance_surrender_value`` row contributes only to the 6th primitive
    -- a scalar annual rate with NO inflation applied (it rides the surrender
    curve's own growth)."""
    rows = (ExpenseItem("maintenance", "surrender_value", 0.004),)
    infl = (1.05) ** (np.arange(12) / 12.0)        # would inflate an inflating basis
    a_pr, a_fx, b_pr, gamma, lae, surr, _face = derive_expense_components(rows, 12, infl)
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


def test_maintenance_surrender_value_hand_calc():
    """The surrender-linked expense each month equals ``rate/12 x value x
    inforce[t]`` -- charged on the begin-of-month in-force surrender value, not
    the lapsing exits. Hand-checked against the in-force path (read off the
    baseline run; the expense does not change the in-force)."""
    mp = _term_life_mp()
    rate = 0.004
    base, value = _surrender_basis()
    with_surr, _ = _surrender_basis(
        (ExpenseItem("maintenance", "surrender_value", rate),))
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


def test_maintenance_surrender_value_routes_fast_to_full():
    """The fused fast path does not carry the in-force surrender value, so a book
    with this item routes to the full path (``requires_full``); the fast call
    returns the full-path BEL."""
    mp = _term_life_mp()
    with_surr, _ = _surrender_basis(
        (ExpenseItem("maintenance", "surrender_value", 0.004),))
    m_full = fcf.gmm.measure(mp, with_surr, full=True)
    m_fast = fcf.gmm.measure(mp, with_surr, full=False)
    assert np.isclose(m_full.bel_path[0, 0], m_fast.bel[0])


def test_maintenance_surrender_value_requires_curve():
    """A ``maintenance_surrender_value`` item with no ``surrender_value_curve``
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
            ExpenseItem("maintenance", "surrender_value", 0.004),),
        coverages=(CoverageRate("DEATH", mort),),
    )
    with pytest.raises(ValueError, match="requires Basis.surrender_value_curve"):
        fcf.gmm.measure(_term_life_mp(), basis)


def test_maintenance_surrender_value_rejected_on_account_book():
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
            ExpenseItem("maintenance", "surrender_value", 0.004),),
    )
    with pytest.raises(ValueError, match="account-backed"):
        fcf.gmm.measure(mp, account_basis)


# ---------------------------------------------------------------------------
# (maintenance, face) -- maintenance charged on the policy's sum assured (the
# main coverage's amount, flagged by ModelPoints.coverage_is_main).
# ---------------------------------------------------------------------------

def test_face_lands_in_seventh_primitive():
    """A (maintenance, face) row contributes only to the 7th primitive (a scalar
    annual rate); no inflation is applied inside derive (it is applied
    post-projection on the level sum assured)."""
    rows = (ExpenseItem("maintenance", "face", 0.0002),)
    out = derive_expense_components(rows, 12)
    assert len(out) == 7
    assert out[6] == 0.0002                          # face primitive
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0


def _face_mp(is_main, face=100_000_000.0):
    """Single DEATH-coverage policy; ``is_main`` flags it as the main contract."""
    import dataclasses
    mp = fcf.ModelPoints.single(
        issue_age=40, benefits={"DEATH": face}, premium=50_000.0,
        term_months=120, calculation_methods={"DEATH": "DEATH"})
    flag = (np.ones_like(mp.coverage_is_main) if is_main
            else np.zeros_like(mp.coverage_is_main))
    return dataclasses.replace(mp, coverage_is_main=flag)


def _face_basis(rate):
    import dataclasses
    def mort(s, ia, d, ic, em): return np.full(d.shape, 0.005)
    def lapse(s, ia, d, ic, em): return np.full(d.shape, 0.03)
    base = fcf.Basis(mortality_annual=mort, lapse_annual=lapse, discount_annual=0.03,
                     ra_confidence=0.75, mortality_cv=0.05,
                     coverages=(CoverageRate("DEATH", mort),))
    with_face = dataclasses.replace(
        base, expense_items=(ExpenseItem("maintenance", "face", rate),))
    return base, with_face


def test_face_hand_calc():
    """The face expense each month equals ``rate/12 x sum_assured x inforce[t]``
    (inflation 1.0 at t=0). Hand-checked against the in-force path."""
    rate, face = 0.0002, 100_000_000.0
    mp = _face_mp(is_main=True, face=face)
    base, with_face = _face_basis(rate)
    m0 = fcf.gmm.measure(mp, base)
    m1 = fcf.gmm.measure(mp, with_face)
    delta = m1.cashflows.expense_cf[0] - m0.cashflows.expense_cf[0]
    inforce = m0.cashflows.inforce[0]
    assert delta[0] == pytest.approx(rate / 12.0 * face * inforce[0])
    assert m1.bel[0] > m0.bel[0]


def test_face_requires_a_main_coverage():
    """A (maintenance, face) item with no coverage flagged main errors -- the
    sum-assured base would otherwise be undefined."""
    _base, with_face = _face_basis(0.0002)
    mp_no_main = _face_mp(is_main=False)
    with pytest.raises(ValueError, match="coverage_is_main"):
        fcf.gmm.measure(mp_no_main, with_face)


def test_face_routes_fast_to_full():
    """The fused fast path does not carry the sum assured, so a face book routes
    to the full path; the fast call returns the full-path BEL."""
    mp = _face_mp(is_main=True)
    _base, with_face = _face_basis(0.0002)
    m_full = fcf.gmm.measure(mp, with_face, full=True)
    m_fast = fcf.gmm.measure(mp, with_face, full=False)
    assert np.isclose(m_full.bel_path[0, 0], m_fast.bel[0])


def _death_basis(**overrides):
    """1%/month mortality, 2%/month lapse, zero discount, RA off."""
    kw = dict(
        mortality_q     = 0.01,
        lapse_q         = 0.02,
        discount_annual = 0.0,
        ra_confidence   = 0.75,
        mortality_cv    = 0.0,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


def test_acquisition_and_maintenance_expense_hand_calc():
    """Acquisition (t=0) and maintenance expense, hand-checked through measure()."""
    res = fcf.gmm.measure(
        fcf.ModelPoints.single(
            issue_age=40, benefits={"DEATH": 1_000_000.0},
            premium=12_000.0, term_months=2,
            calculation_methods=PATTERNS,
        ),
        _death_basis(
            expense_items=(
                ExpenseItem("acquisition", "per_policy",    500.0),
                ExpenseItem("maintenance", "per_policy", 120.0),  # 10 per month
            ),
        ),
    )
    inforce = [1.0, 0.99 * 0.98]
    # expense_cf[0] = acquisition + maintenance = 1*500 + 1*(120/12) = 510
    # expense_cf[1] = maintenance only         = 0.9702*(120/12)     = 9.702
    assert np.isclose(res.cashflows.expense_cf[0, 0], 510.0)
    assert np.isclose(res.cashflows.expense_cf[0, 1], 9.702)

    # BEL = PV(claims) + PV(expenses) - PV(premiums)
    pv_claims = 19702.0
    pv_expenses = 510.0 + 9.702
    pv_premiums = 12_000.0 + inforce[1] * 12_000.0
    assert np.isclose(res.bel_path[0, 0], pv_claims + pv_expenses - pv_premiums)


def test_maintenance_expense_grows_with_inflation():
    """Maintenance expense grows with inflation; acquisition does not recur."""
    res = fcf.gmm.measure(
        fcf.ModelPoints.single(
            issue_age=40, benefits={"DEATH": 1_000_000.0},
            premium=12_000.0, term_months=13,
            calculation_methods=PATTERNS,
        ),
        _death_basis(
            mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.0)),
            lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.0)),
            expense_inflation=0.06,
            expense_items=(
                ExpenseItem("maintenance", "per_policy", 120.0),  # 10 per month
            ),
        ),
    )
    # no mortality/lapse -> in force stays 1.0
    # maintenance[t] = 1.0 * 10 * (1.06)^(t/12)
    assert np.isclose(res.cashflows.expense_cf[0, 0], 10.0)
    assert np.isclose(res.cashflows.expense_cf[0, 12], 10.0 * 1.06)
