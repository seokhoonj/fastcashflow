"""vfa.settle -- the IFRS 17 paragraph-45 settlement movement.

The anchor facts, from the design contract:

* With on-track experience the single period-end release telescopes to the
  monthly carry of ``vfa.measure_inforce`` exactly.
* The future-service change is x = -(bel_experience + ra_experience) -- the
  engine's own observed-vs-expected FCF difference -- and the 45(b)/45(c)
  lines are a disclosure split of that exact total: a pure account-value move
  lands entirely in 45(b); a binding crediting floor or an in-the-money
  guarantee lands in 45(c).
* The loss-component algebra satisfies a conservation identity in every
  sign case, and every block reconciles by construction.
"""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CoverageRate, ExpenseItem, InforceState, ModelPoints)
from fastcashflow._vfa import _paragraph45_csm_algebra, _vfa_project


def _basis(*, investment_return=0.05, fund_fee=0.015, expense=1_000.0,
           settlement_pattern=None):
    death_fn = lambda s, ia, d: np.full(s.shape, 0.012)
    items = ((ExpenseItem("maintenance", "gamma_fixed", expense),)
             if expense else ())
    return Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=investment_return,   # VFA discounts at the return
        ra_confidence=0.75,
        mortality_cv=0.0,
        expense_cv=0.10,
        investment_return=investment_return,
        fund_fee=fund_fee,
        expense_items=items,
        settlement_pattern=settlement_pattern,
        coverages=(CoverageRate("DEATH", death_fn),),
    )


def _growth(basis, mp):
    """The engine's monthly account-growth factor, per model point."""
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + basis.fund_fee) ** (1.0 / 12.0) - 1.0
    g = np.asarray(mp.minimum_crediting_rate, dtype=np.float64)
    g_m = np.where(g >= 0.0, (1.0 + np.maximum(g, 0.0)) ** (1.0 / 12.0) - 1.0,
                   r_m)
    credit = np.where(g >= 0.0, np.maximum(r_m, g_m), r_m)
    return (1.0 + credit) * (1.0 - f_m)


def _book(basis, mp0, *, em_open, period, av_close=None, count_close=None,
          prior_csm=None, lc_open=None):
    """Seat an in-force book at em_open + period with on-track defaults.

    Returns ``(model_points, state)`` ready for ``vfa.settle``: the observed
    closing account value / count default to the expected (on-track) path,
    and ``prior_csm`` defaults to the inception measurement's CSM at
    ``em_open``.
    """
    em_close = em_open + period
    m0 = fcf.vfa.measure(mp0, basis)
    inforce = m0.cashflows.inforce
    n_mp = mp0.n_mp
    rows = np.arange(n_mp)
    growth = _growth(basis, mp0)
    av_open = np.asarray(mp0.account_value, dtype=np.float64) * growth ** em_open
    if av_close is None:
        av_close = av_open * growth ** period
    av_close = np.atleast_1d(np.asarray(av_close, dtype=np.float64))
    inforce_pad = np.concatenate([inforce, np.zeros((n_mp, 1))], axis=1)
    count_open = inforce[rows, em_open]
    if count_close is None:
        boundary = np.asarray(mp0.contract_boundary_months)
        count_close = inforce_pad[rows, np.minimum(em_close, boundary)]
    if prior_csm is None:
        prior_csm = m0.csm_path[rows, em_open]
    mp = replace(mp0, elapsed_months=np.full(n_mp, em_close, dtype=np.int64),
                 count=np.asarray(count_close, dtype=np.float64))
    state = InforceState(
        mp_id=np.array([f"P{i}" for i in range(n_mp)]),
        elapsed_months=np.full(n_mp, em_close, dtype=np.int64),
        count=np.asarray(count_close, dtype=np.float64),
        prior_csm=np.asarray(prior_csm, dtype=np.float64),
        lock_in_rate=0.0,
        account_value=np.asarray(av_close, dtype=np.float64),
        prior_count=np.asarray(count_open, dtype=np.float64),
        prior_account_value=av_open,
        prior_loss_component=lc_open,
    )
    return mp, state


def _assert_blocks_foot(mv):
    """Every reconciliation identity of the movement, by construction."""
    np.testing.assert_allclose(
        mv.bel_closing,
        mv.bel_opening + mv.bel_interest - mv.bel_release + mv.bel_experience,
        rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(
        mv.ra_closing,
        mv.ra_opening + mv.ra_interest - mv.ra_release + mv.ra_experience,
        rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(
        mv.csm_closing,
        mv.csm_opening + mv.csm_accretion + mv.csm_fv_share
        + mv.csm_future_service - mv.loss_component_reversed
        + mv.loss_component_recognised - mv.csm_release,
        rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(
        mv.loss_component_closing,
        mv.loss_component_opening - mv.loss_component_reversed
        + mv.loss_component_recognised,
        rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(
        mv.csm_fv_share + mv.csm_future_service,
        -(mv.bel_experience + mv.ra_experience),
        rtol=1e-10, atol=1e-9)


# ---------------------------------------------------------------------------
# anchors: carry equivalence, pure-AV split, floor wedge, ITM direction
# ---------------------------------------------------------------------------
def test_on_track_settle_equals_the_monthly_carry():
    """x ~ 0 with on-track experience, and the single period-end release
    telescopes to vfa.measure_inforce's monthly _csm_kernel carry exactly."""
    basis = _basis()
    mp, state = _book(basis, ModelPoints.single(40, 0.0, 24, account_value=1e6),
                      em_open=6, period=3)
    mv = fcf.vfa.settle(mp, state, basis, period_months=3)
    money = float(state.prior_account_value[0])
    assert abs(mv.bel_experience[0]) < 1e-9 * money
    assert abs(mv.ra_experience[0]) < 1e-9 * money
    ref = fcf.vfa.measure_inforce(mp, state, basis, period_months=3)
    np.testing.assert_allclose(mv.csm_closing, ref.csm, rtol=1e-10)
    _assert_blocks_foot(mv)


def test_pure_av_move_lands_entirely_in_fv_share():
    """No guarantee, no floor, on-track count, observed AV +10%: the
    paragraph-45(c) line is exactly zero and 45(b) carries all of x."""
    basis = _basis()
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6)
    m0 = fcf.vfa.measure(mp0, basis)
    growth = _growth(basis, mp0)
    av_exp = 1e6 * growth ** 9
    mp, state = _book(basis, mp0, em_open=6, period=3, av_close=av_exp * 1.10)
    mv = fcf.vfa.settle(mp, state, basis, period_months=3)
    x = -(mv.bel_experience + mv.ra_experience)
    assert x[0] > 0.0                                     # fund up -> fee up
    np.testing.assert_allclose(mv.csm_fv_share, x, rtol=1e-10)
    np.testing.assert_allclose(mv.csm_future_service, 0.0,
                               atol=1e-7 * abs(x[0]))
    _assert_blocks_foot(mv)


def test_crediting_floor_cost_lands_in_future_service():
    """A binding minimum crediting rate (floor 5% > return 3%) leaves a
    first-order cost inside the BEL that is in neither the guarantee-excess
    nor the fee PV -- the x primitive keeps the identities exact where the
    old feePV - (dG+dE+dRA) formula would not, and the 45(c) line is
    materially non-zero."""
    basis = _basis(investment_return=0.03)
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6,
                             minimum_crediting_rate=0.05)
    growth = _growth(basis, mp0)
    av_exp = 1e6 * growth[0] ** 9
    mp, state = _book(basis, mp0, em_open=6, period=3, av_close=av_exp * 1.10)
    mv = fcf.vfa.settle(mp, state, basis, period_months=3)
    x = -(mv.bel_experience + mv.ra_experience)
    np.testing.assert_allclose(mv.csm_fv_share + mv.csm_future_service, x,
                               rtol=1e-10)
    assert abs(mv.csm_future_service[0]) > 1e-6 * abs(x[0])
    _assert_blocks_foot(mv)


def test_itm_guarantee_pushes_the_csm_down():
    """GMDB well above the fund, observed AV 20% below expected: the
    future-service change is unfavourable (guarantee cost up) and the fee
    line falls too."""
    basis = _basis()
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6,
                             minimum_death_benefit=2e6)
    growth = _growth(basis, mp0)
    av_exp = 1e6 * growth[0] ** 9
    mp, state = _book(basis, mp0, em_open=6, period=3, av_close=av_exp * 0.80,
                      prior_csm=np.array([50_000.0]))
    mv = fcf.vfa.settle(mp, state, basis, period_months=3)
    x = -(mv.bel_experience + mv.ra_experience)
    assert x[0] < 0.0
    assert mv.csm_fv_share[0] < 0.0           # fee PV falls with the fund
    assert mv.csm_future_service[0] < 0.0     # the guarantee bites
    _assert_blocks_foot(mv)


# ---------------------------------------------------------------------------
# loss-component algebra (scalar, exact)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "accreted, x, lc_open, csm_after, lc_reversed, lc_recognised, lc_closing",
    [
        (100.0,   50.0,  80.0, 100.0, 50.0,  0.0,  30.0),
        (100.0,  120.0,  80.0, 140.0, 80.0,  0.0,   0.0),
        (100.0,  -60.0,   0.0,  40.0,  0.0,  0.0,   0.0),
        (100.0, -150.0,   0.0,   0.0,  0.0, 50.0,  50.0),
        (0.0,     70.0, 100.0,   0.0, 70.0,  0.0,  30.0),
        (0.0,    -70.0, 100.0,   0.0,  0.0, 70.0, 170.0),
    ],
)
def test_lc_algebra_sign_grid(accreted, x, lc_open, csm_after, lc_reversed,
                              lc_recognised, lc_closing):
    """The paragraph-48/50(b) step, every sign case, exact scalar values --
    plus the conservation identity (csm_after - A) - (dLC) == x."""
    after, reversed_, recognised, closing = _paragraph45_csm_algebra(
        np.array([accreted]), np.array([x]), np.array([lc_open]))
    assert after[0] == csm_after
    assert reversed_[0] == lc_reversed
    assert recognised[0] == lc_recognised
    assert closing[0] == lc_closing
    assert (after[0] - accreted) - (closing[0] - lc_open) == x


def test_rejects_a_state_carrying_both_csm_and_loss_component():
    basis = _basis()
    mp, state = _book(basis, ModelPoints.single(40, 0.0, 24, account_value=1e6),
                      em_open=6, period=3,
                      prior_csm=np.array([10.0]), lc_open=np.array([5.0]))
    with pytest.raises(ValueError, match=r"both positive at row\(s\) \[0\]"):
        fcf.vfa.settle(mp, state, basis, period_months=3)


# ---------------------------------------------------------------------------
# opening figures and the expected projection's conventions
# ---------------------------------------------------------------------------
def test_opening_lines_match_measure_inforce_at_the_opening_date():
    """The expected leg sliced at em_open IS the measure_inforce valuation
    of the opening date -- zero drift between the two entry points."""
    basis = _basis()
    em_open, period = 6, 3
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6)
    mp, state = _book(basis, mp0, em_open=em_open, period=period)
    mv = fcf.vfa.settle(mp, state, basis, period_months=period)

    mp_open = replace(
        mp0, elapsed_months=np.array([em_open], dtype=np.int64),
        count=np.asarray(state.prior_count, dtype=np.float64))
    state_open = InforceState(
        mp_id=np.array(["P0"]),
        elapsed_months=np.array([em_open], dtype=np.int64),
        count=np.asarray(state.prior_count),
        prior_csm=np.asarray(state.prior_csm),
        lock_in_rate=0.0,
        account_value=np.asarray(state.prior_account_value))
    ref = fcf.vfa.measure_inforce(mp_open, state_open, basis,
                                  period_months=em_open)
    np.testing.assert_allclose(mv.bel_opening, ref.bel, rtol=1e-12)
    np.testing.assert_allclose(mv.ra_opening, ref.ra, rtol=1e-12)


def test_zero_fee_zero_expense_book_has_zero_bel_lines():
    """With no fee and no expenses every exit repays the account value and
    the discount equals the growth, so the BEL is identically zero -- every
    BEL line of the movement vanishes while the CSM still rolls."""
    basis = _basis(fund_fee=0.0, expense=0.0)
    mp, state = _book(basis, ModelPoints.single(40, 0.0, 24, account_value=1e6),
                      em_open=6, period=3, prior_csm=np.array([1_000.0]))
    mv = fcf.vfa.settle(mp, state, basis, period_months=3)
    for line in (mv.bel_opening, mv.bel_interest, mv.bel_release,
                 mv.bel_experience, mv.bel_closing):
        np.testing.assert_allclose(line, 0.0, atol=1e-6)
    assert mv.csm_accretion[0] > 0.0
    assert mv.csm_release[0] > 0.0
    _assert_blocks_foot(mv)


def test_interest_line_is_the_rate_times_the_expected_trajectory():
    """bel_interest is computed directly -- r_m times the expected
    trajectory's BEL over the period months (re-based to the opening count),
    the roll-forward convention -- pinned against an independent
    recomputation from _vfa_project."""
    basis = _basis()
    em_open, period = 6, 3
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6)
    mp, state = _book(basis, mp0, em_open=em_open, period=period)
    mv = fcf.vfa.settle(mp, state, basis, period_months=period)

    p = _vfa_project(mp, basis,
                     elapsed_months=np.array([em_open], dtype=np.int64),
                     account_value=np.asarray(state.prior_account_value))
    r_m = p.r_m
    k_exp = float(state.prior_count[0]) / p.inforce[0, em_open]
    expected = r_m * k_exp * p.bel[0, em_open:em_open + period].sum()
    np.testing.assert_allclose(mv.bel_interest[0], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# boundary / maturity / dead-cohort edges
# ---------------------------------------------------------------------------
def test_final_settlement_releases_the_whole_csm():
    """A contract whose boundary falls inside the period closes at zero:
    no remaining units, the post-adjustment CSM releases in full."""
    basis = _basis()
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6)
    mp, state = _book(basis, mp0, em_open=22, period=2,
                      av_close=np.array([0.0]), count_close=np.array([0.0]),
                      prior_csm=np.array([5_000.0]))
    mv = fcf.vfa.settle(mp, state, basis, period_months=2)
    assert mv.bel_closing[0] == 0.0
    assert mv.ra_closing[0] == 0.0
    csm_after = (mv.csm_opening + mv.csm_accretion + mv.csm_fv_share
                 + mv.csm_future_service - mv.loss_component_reversed
                 + mv.loss_component_recognised)
    np.testing.assert_allclose(mv.csm_release, csm_after, rtol=1e-10)
    np.testing.assert_allclose(mv.csm_closing, 0.0, atol=1e-9)
    _assert_blocks_foot(mv)


def test_final_settlement_requires_a_zero_closing_snapshot():
    basis = _basis()
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6)
    mp, state = _book(basis, mp0, em_open=22, period=2,
                      av_close=np.array([0.0]),
                      count_close=np.array([0.5]))   # matured but count > 0
    with pytest.raises(ValueError, match="final settlement"):
        fcf.vfa.settle(mp, state, basis, period_months=2)


def test_mixed_book_straddling_one_boundary():
    """A 24m and a 120m contract settled in one call, the period crossing
    the short contract's boundary: per-MP clamps hold, the short row fully
    releases, the long row releases its B119 fraction, the table foots."""
    basis = _basis()
    mp0 = ModelPoints(
        issue_age=np.array([40, 40]), premium=np.zeros(2),
        term_months=np.array([24, 120]),
        benefits={0: np.zeros(2)},
        account_value=np.array([1e6, 1e6]))
    mp, state = _book(basis, mp0, em_open=22, period=2)
    # the short row matured inside the period: zero its closing snapshot
    count = state.count.copy(); count[0] = 0.0
    av = state.account_value.copy(); av[0] = 0.0
    state = replace(state, count=count, account_value=av)
    mp = replace(mp, count=count)
    mv = fcf.vfa.settle(mp, state, basis, period_months=2)
    np.testing.assert_allclose(mv.csm_closing[0], 0.0, atol=1e-9)
    assert mv.csm_closing[1] > 0.0
    assert mv.csm_release[1] < (mv.csm_opening[1] + mv.csm_accretion[1])
    _assert_blocks_foot(mv)
    table = fcf.reconcile([mv])[0]
    assert np.isclose(
        table.csm_closing,
        table.csm_opening + table.csm_accretion + table.csm_fv_share
        + table.csm_future_service + table.loss_component_reversed
        + table.loss_component_recognised + table.csm_release)


def test_dead_cohort_mid_life_fully_derecognises():
    """count = 0 on a live column (mass lapse): the observed factor is zero,
    no future units remain, the whole post-adjustment CSM releases."""
    basis = _basis()
    mp0 = ModelPoints.single(40, 0.0, 60, account_value=1e6)
    mp, state = _book(basis, mp0, em_open=6, period=3,
                      av_close=np.array([0.0]), count_close=np.array([0.0]))
    mv = fcf.vfa.settle(mp, state, basis, period_months=3)
    assert mv.bel_closing[0] == 0.0
    np.testing.assert_allclose(
        mv.csm_release, mv.csm_opening + mv.csm_accretion + mv.csm_fv_share
        + mv.csm_future_service - mv.loss_component_reversed
        + mv.loss_component_recognised, rtol=1e-10)
    np.testing.assert_allclose(mv.csm_closing, 0.0, atol=1e-9)
    _assert_blocks_foot(mv)


def test_onerous_book_loss_component_is_static_between_remeasurements():
    """The documented v1 cut: with x ~ 0 an onerous book's loss component
    carries over unchanged and no profit emerges."""
    basis = _basis()
    mp, state = _book(basis, ModelPoints.single(40, 0.0, 24, account_value=1e6),
                      em_open=6, period=3,
                      prior_csm=np.array([0.0]), lc_open=np.array([1_000.0]))
    mv = fcf.vfa.settle(mp, state, basis, period_months=3)
    assert mv.csm_release[0] == 0.0
    assert mv.csm_closing[0] == 0.0
    np.testing.assert_allclose(mv.loss_component_closing,
                               mv.loss_component_opening, atol=1e-6)
    _assert_blocks_foot(mv)


def test_heterogeneous_elapsed_months_in_one_call():
    """Two cohorts with different valuation elapsed months settled in one
    call: every gather / clamp is per-row, and each row equals its own
    single-row settle exactly."""
    basis = _basis()
    period = 3
    em_close = np.array([9, 17], dtype=np.int64)
    em_open = em_close - period
    mp0 = ModelPoints(
        issue_age=np.array([40, 45]), premium=np.zeros(2),
        term_months=np.array([24, 36]),
        benefits={0: np.zeros(2)},
        account_value=np.array([1e6, 2e6]))
    m0 = fcf.vfa.measure(mp0, basis)
    inforce = m0.cashflows.inforce
    rows = np.arange(2)
    growth = _growth(basis, mp0)
    av_open = np.asarray(mp0.account_value) * growth ** em_open
    av_close = av_open * growth ** period * np.array([1.08, 0.95])  # off-track
    count_open = inforce[rows, em_open]
    count_close = inforce[rows, em_close]
    prior_csm = m0.csm_path[rows, em_open]

    mp = replace(mp0, elapsed_months=em_close, count=count_close)
    state = InforceState(
        mp_id=np.array(["P0", "P1"]), elapsed_months=em_close,
        count=count_close, prior_csm=prior_csm, lock_in_rate=0.0,
        account_value=av_close, prior_count=count_open,
        prior_account_value=av_open)
    mv = fcf.vfa.settle(mp, state, basis, period_months=period)
    _assert_blocks_foot(mv)

    # each row reproduces its own single-row settle bit-for-bit
    for i in (0, 1):
        mp_i = replace(mp0.subset([i]),
                       elapsed_months=em_close[[i]], count=count_close[[i]])
        state_i = state.subset([i])
        mv_i = fcf.vfa.settle(mp_i, state_i, basis, period_months=period)
        for field in ("bel_opening", "bel_interest", "bel_release",
                      "bel_experience", "bel_closing", "csm_fv_share",
                      "csm_future_service", "csm_release", "csm_closing"):
            np.testing.assert_allclose(getattr(mv, field)[i],
                                       getattr(mv_i, field)[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_guards():
    basis = _basis()
    mp0 = ModelPoints.single(40, 0.0, 24, account_value=1e6)
    mp, state = _book(basis, mp0, em_open=6, period=3)

    with pytest.raises(NotImplementedError, match="settlement_pattern"):
        fcf.vfa.settle(mp, state,
                       _basis(settlement_pattern=np.array([0.6, 0.4])),
                       period_months=3)
    with pytest.raises(ValueError, match="prior_account_value"):
        fcf.vfa.settle(mp, replace(state, prior_account_value=None), basis,
                       period_months=3)
    with pytest.raises(ValueError, match="prior_count"):
        fcf.vfa.settle(mp, replace(state, prior_count=None), basis,
                       period_months=3)
    with pytest.raises(ValueError, match="account_value"):
        fcf.vfa.settle(mp, replace(state, account_value=None), basis,
                       period_months=3)
    with pytest.raises(ValueError, match="period_months must be >= 1"):
        fcf.vfa.settle(mp, state, basis, period_months=0)
    with pytest.raises(ValueError, match="precedes inception"):
        fcf.vfa.settle(mp, state, basis, period_months=10)  # em_open < 0

    # opening date at/past the contract boundary -> nothing to settle.
    # Hand-built (the _book helper cannot index an opening past the horizon).
    mp_old = replace(mp0, elapsed_months=np.array([26], dtype=np.int64),
                     count=np.array([0.0]))
    state_old = InforceState(
        mp_id=np.array(["P0"]), elapsed_months=np.array([26], dtype=np.int64),
        count=np.array([0.0]), prior_csm=np.array([100.0]),
        lock_in_rate=0.0, account_value=np.array([0.0]),
        prior_count=np.array([0.5]), prior_account_value=np.array([1e6]))
    with pytest.raises(ValueError, match="nothing to settle"):
        fcf.vfa.settle(mp_old, state_old, basis, period_months=2)


# ---------------------------------------------------------------------------
# integration: reconcile, closing_measurement, csm_basis guards, chaining
# ---------------------------------------------------------------------------
def test_reconcile_returns_a_footing_settlement_table():
    basis = _basis()
    mp, state = _book(basis, ModelPoints.single(40, 0.0, 24, account_value=1e6),
                      em_open=6, period=3)
    mv = fcf.vfa.settle(mp, state, basis, period_months=3)
    table = fcf.reconcile([mv])[0]
    assert isinstance(table, fcf.VFASettlementReconciliation)
    assert table.period_months == 3
    # every printed block foots: opening + rows == closing (signed rows)
    assert np.isclose(table.bel_closing, table.bel_opening + table.bel_interest
                      + table.bel_release + table.bel_experience)
    assert np.isclose(table.ra_closing, table.ra_opening + table.ra_interest
                      + table.ra_release + table.ra_experience)
    assert np.isclose(
        table.csm_closing,
        table.csm_opening + table.csm_accretion + table.csm_fv_share
        + table.csm_future_service + table.loss_component_reversed
        + table.loss_component_recognised + table.csm_release)
    assert np.isclose(
        table.loss_component_closing,
        table.loss_component_opening + table.loss_component_reversed
        + table.loss_component_recognised)
    assert "VFA settlement reconciliation" in str(table)


def test_closing_measurement_is_settlement_grade():
    """The paragraph-45 tag passes the carry-only guard: write_measurement
    serialises it (report / group / roll_forward still need trajectories,
    which a headline-only result does not carry); a carry-only measurement
    stays rejected everywhere (regression)."""
    basis = _basis()
    mp, state = _book(basis, ModelPoints.single(40, 0.0, 24, account_value=1e6),
                      em_open=6, period=3)
    mv = fcf.vfa.settle(mp, state, basis, period_months=3)
    closing = mv.closing_measurement()
    assert closing.csm_basis == "paragraph_45_settlement"
    np.testing.assert_allclose(closing.csm, mv.csm_closing)

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "close.parquet")
        fcf.write_measurement(closing, path)      # no carry-only rejection
        assert os.path.exists(path)
    with pytest.raises(ValueError, match="full=True"):
        fcf.roll_forward(closing, period_months=12)   # headline-only

    carry = fcf.vfa.measure_inforce(mp, state, basis, period_months=3)
    with pytest.raises(ValueError, match="carry-only"):
        fcf.write_measurement(carry, "unused.parquet")


def test_two_six_month_settles_chain_to_one_twelve_month():
    """Chaining: settle 6 months, seed the next state from the closing
    figures, settle 6 more -- with on-track experience the closing CSM
    equals one 12-month settle (the telescoping identity again)."""
    basis = _basis()
    mp0 = ModelPoints.single(40, 0.0, 36, account_value=1e6)
    m0 = fcf.vfa.measure(mp0, basis)
    growth = _growth(basis, mp0)
    inforce = m0.cashflows.inforce[0]

    # one 12-month settle: open at 6, close at 18
    mp12, state12 = _book(basis, mp0, em_open=6, period=12)
    mv12 = fcf.vfa.settle(mp12, state12, basis, period_months=12)

    # two 6-month settles
    mp_a, state_a = _book(basis, mp0, em_open=6, period=6)
    mv_a = fcf.vfa.settle(mp_a, state_a, basis, period_months=6)
    em_mid, em_end = 12, 18
    av_end = 1e6 * growth[0] ** em_end
    mp_b = replace(mp0, elapsed_months=np.array([em_end], dtype=np.int64),
                   count=np.array([inforce[em_end]]))
    state_b = InforceState(
        mp_id=np.array(["P0"]),
        elapsed_months=np.array([em_end], dtype=np.int64),
        count=np.array([inforce[em_end]]),
        prior_csm=mv_a.csm_closing,
        lock_in_rate=0.0,
        account_value=np.array([av_end]),
        prior_count=state_a.count,
        prior_account_value=state_a.account_value,
        prior_loss_component=mv_a.loss_component_closing,
    )
    mv_b = fcf.vfa.settle(mp_b, state_b, basis, period_months=6)
    np.testing.assert_allclose(mv_b.csm_closing, mv12.csm_closing, rtol=1e-10)
    _assert_blocks_foot(mv_a)
    _assert_blocks_foot(mv_b)
    _assert_blocks_foot(mv12)


# ---------------------------------------------------------------------------
# InforceState extension and the readers
# ---------------------------------------------------------------------------
def test_inforce_state_optional_fields_roundtrip(tmp_path):
    import polars as pl

    pl.DataFrame({
        "mp_id": ["A", "B"],
        "elapsed_months": [12, 12],
        "count": [0.9, 0.8],
        "prior_csm": [10.0, 20.0],
        "lock_in_rate": [0.03, 0.03],
        "account_value": [1e6, 2e6],
        "prior_count": [0.95, 0.85],
        "prior_account_value": [9e5, 1.8e6],
        "prior_loss_component": [0.0, 5.0],
    }).write_csv(tmp_path / "state.csv")
    state = fcf.read_inforce_state(tmp_path / "state.csv")
    np.testing.assert_allclose(state.prior_count, [0.95, 0.85])
    np.testing.assert_allclose(state.prior_account_value, [9e5, 1.8e6])
    np.testing.assert_allclose(state.prior_loss_component, [0.0, 5.0])
    sub = state.subset([1])
    np.testing.assert_allclose(sub.prior_account_value, [1.8e6])

    # absent optional columns stay None (the GMM/PAA state shape)
    pl.DataFrame({
        "mp_id": ["A"], "elapsed_months": [12], "count": [0.9],
        "prior_csm": [10.0], "lock_in_rate": [0.03],
    }).write_csv(tmp_path / "plain.csv")
    plain = fcf.read_inforce_state(tmp_path / "plain.csv")
    assert plain.prior_count is None
    assert plain.prior_account_value is None
    assert plain.prior_loss_component is None


def test_inforce_state_validates_optional_fields():
    with pytest.raises(ValueError, match="prior_count must be >= 0"):
        InforceState(mp_id=np.array(["A"]), elapsed_months=[1], count=[1.0],
                     prior_csm=[0.0], lock_in_rate=0.0,
                     prior_count=np.array([-1.0]))
    with pytest.raises(ValueError, match="prior_account_value has length"):
        InforceState(mp_id=np.array(["A"]), elapsed_months=[1], count=[1.0],
                     prior_csm=[0.0], lock_in_rate=0.0,
                     prior_account_value=np.array([1.0, 2.0]))
