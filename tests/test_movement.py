"""Period-close roll-forward validation -- the expected-basis analysis of change.

``roll_forward`` slices a GMM measurement into reporting-period movements.
Each period must reconcile (opening + movements = closing) and consecutive
periods must chain.
"""
import fastcashflow as fcf
from dataclasses import replace

import numpy as np
import pytest

from fastcashflow import CoverageRate, ExpenseItem, ModelPoints, reconcile, roll_forward
from fastcashflow.gmm import measure
from fastcashflow.basis import BasisRouter
from conftest import annual_from_monthly as _annual, make_death_basis, PATTERNS


def _basis():
    return make_death_basis(
        mortality_q       = 0.001,
        lapse_q           = 0.01,
        discount_annual   = 0.03,
        expense_inflation = 0.02,
        expense_items     = (
            ExpenseItem("acquisition",  "alpha_fixed",    200_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  60_000.0),
        ),
        ra_confidence     = 0.75,
        mortality_cv      = 0.10,
    )


def _portfolio(n: int = 50) -> ModelPoints:
    rng = np.random.default_rng(4)
    return ModelPoints(
        issue_age=rng.integers(30, 55, n),
        benefits={"DEATH": rng.integers(20, 90, n) * 1_000_000},
        premium=rng.integers(8, 20, n) * 10_000,
        term_months=np.full(n, 120),
        calculation_methods=PATTERNS,
    )


def test_roll_forward_period_count():
    """A 120-month horizon in yearly periods gives ten movements."""
    periods = roll_forward(measure(_portfolio(), _basis()), period_months=12)
    assert len(periods) == 10


def test_roll_forward_csm_reconciles():
    """Each period: opening + accretion - release = closing CSM."""
    for p in roll_forward(measure(_portfolio(), _basis()), 12):
        assert np.allclose(
            p.csm_opening + p.csm_accretion - p.csm_release, p.csm_closing
        )


def test_roll_forward_bel_and_ra_reconcile():
    """Each period: opening + interest - release = closing, for BEL and RA."""
    for p in roll_forward(measure(_portfolio(), _basis()), 12):
        assert np.allclose(
            p.bel_opening + p.bel_interest - p.bel_release, p.bel_closing
        )
        assert np.allclose(
            p.ra_opening + p.ra_interest - p.ra_release, p.ra_closing
        )


def test_roll_forward_periods_chain():
    """Each period's closing balances are the next period's opening balances."""
    periods = roll_forward(measure(_portfolio(), _basis()), 12)
    for prev, nxt in zip(periods, periods[1:]):
        assert prev.month_end == nxt.month_start
        assert np.allclose(prev.csm_closing, nxt.csm_opening)
        assert np.allclose(prev.bel_closing, nxt.bel_opening)
        assert np.allclose(prev.ra_closing, nxt.ra_opening)


def test_roll_forward_opening_is_inception():
    """The first period opens at the inception measurement."""
    m = measure(_portfolio(), _basis())
    first = roll_forward(m, 12)[0]
    assert first.month_start == 0
    assert np.allclose(first.csm_opening, m.csm_path[:, 0])
    assert np.allclose(first.bel_opening, m.bel_path[:, 0])


def test_roll_forward_runs_off_to_zero():
    """The final period closes the contract -- CSM and BEL run off to zero."""
    last = roll_forward(measure(_portfolio(), _basis()), 12)[-1]
    assert np.allclose(last.csm_closing, 0.0, atol=1.0)
    assert np.allclose(last.bel_closing, 0.0, atol=1.0)


def test_roll_forward_uneven_last_period():
    """A horizon not divisible by the period gives a short final period."""
    periods = roll_forward(measure(_portfolio(), _basis()), period_months=7)
    assert periods[-1].month_end == 120
    assert sum(p.month_end - p.month_start for p in periods) == 120


def test_roll_forward_rejects_bad_period():
    """A non-positive period length is an error."""
    with pytest.raises(ValueError, match="period_months"):
        roll_forward(measure(_portfolio(), _basis()), 0)


def _actuals(m, factors):
    """Actual in-force at successive boundaries -- expected scaled by factors."""
    return np.array([m.cashflows.inforce[:, (j + 1) * 12] * f
                     for j, f in enumerate(factors)])


def test_roll_forward_multi_experience_reconciles():
    """Experience at every period boundary -- each period reconciles exactly."""
    m = measure(_portfolio(), _basis())
    actuals = _actuals(m, [0.97, 0.93, 0.88])
    for p in roll_forward(m, 12, actual_inforce=actuals):
        assert np.allclose(
            p.bel_opening + p.bel_assumption_change + p.bel_experience
            + p.bel_interest - p.bel_release, p.bel_closing)
        assert np.allclose(
            p.csm_opening + p.csm_assumption_change + p.csm_experience
            + p.csm_accretion - p.csm_release, p.csm_closing)


def test_roll_forward_multi_experience_chains_and_isolates():
    """Periods chain, and an experience line sits in each boundary period."""
    m = measure(_portfolio(), _basis())
    # distinct ratios per boundary, so each period has its own experience
    periods = roll_forward(m, 12, actual_inforce=_actuals(m, [0.95, 0.90, 0.85]))
    for prev, nxt in zip(periods, periods[1:]):
        assert np.allclose(prev.csm_closing, nxt.csm_opening)
        assert np.allclose(prev.bel_closing, nxt.bel_opening)
    for p in periods:
        if p.month_start in (12, 24, 36):
            assert not np.allclose(p.bel_experience, 0.0)
        else:
            assert np.allclose(p.bel_experience, 0.0)


def test_roll_forward_multi_single_row_matches_single_experience():
    """A one-row 2-D actual_inforce equals the single-experience roll."""
    m = measure(_portfolio(), _basis())
    actual_12 = m.cashflows.inforce[:, 12] * 0.9
    multi = roll_forward(m, 12, actual_inforce=actual_12.reshape(1, -1))
    single = roll_forward(m, 12, actual_inforce=actual_12, experience_at=12)
    for p_multi, p_single in zip(multi, single):
        assert np.allclose(p_multi.bel_closing, p_single.bel_closing)
        assert np.allclose(p_multi.csm_closing, p_single.csm_closing)
        assert np.allclose(p_multi.csm_experience, p_single.csm_experience)


def test_roll_forward_multi_rejects_revision_or_experience_at():
    """A 2-D actual_inforce does not combine with revised or experience_at."""
    m = measure(_portfolio(), _basis())
    actuals = _actuals(m, [0.95, 0.95])
    with pytest.raises(ValueError, match="do not apply"):
        roll_forward(m, 12, actual_inforce=actuals, revised=m, revised_at=24)


def test_roll_forward_multi_rejects_too_many_rows():
    """Experience rows past the projection horizon are an error."""
    m = measure(_portfolio(), _basis())          # 120-month horizon
    actuals = np.ones((11, m.bel.shape[0]))            # boundary 132 > 120
    with pytest.raises(ValueError, match="horizon"):
        roll_forward(m, 12, actual_inforce=actuals)


def _revised(mps: ModelPoints):
    """A measurement of the same book under markedly higher mortality."""
    worse_mort = lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.003))
    worse = replace(
        _basis(),
        mortality_annual=worse_mort,
        coverages=(CoverageRate("DEATH", worse_mort),),
    )
    return measure(mps, worse)


def test_roll_forward_assumption_change_reconciles():
    """With a revision, every period still reconciles exactly."""
    mps = _portfolio()
    periods = roll_forward(measure(mps, _basis()), 12,
                           revised=_revised(mps), revised_at=24)
    for p in periods:
        assert np.allclose(
            p.bel_opening + p.bel_assumption_change + p.bel_interest
            - p.bel_release, p.bel_closing)
        assert np.allclose(
            p.csm_opening + p.csm_assumption_change + p.csm_accretion
            - p.csm_release, p.csm_closing)


def test_roll_forward_assumption_change_only_in_revision_period():
    """The assumption-change line is non-zero only in the revision period."""
    mps = _portfolio()
    periods = roll_forward(measure(mps, _basis()), 12,
                           revised=_revised(mps), revised_at=24)
    for p in periods:
        if p.month_start == 24:
            assert not np.allclose(p.csm_assumption_change, 0.0)
        else:
            assert np.allclose(p.csm_assumption_change, 0.0)
            assert np.allclose(p.bel_assumption_change, 0.0)


def test_roll_forward_worse_assumptions_reduce_csm():
    """A revision that raises the liability adjusts the CSM downwards."""
    mps = _portfolio()
    periods = roll_forward(measure(mps, _basis()), 12,
                           revised=_revised(mps), revised_at=24)
    rev = next(p for p in periods if p.month_start == 24)
    assert np.all(rev.bel_assumption_change > 0.0)        # higher claims
    assert np.all(rev.csm_assumption_change <= 0.0)       # CSM absorbs it


def test_roll_forward_pre_revision_periods_unaffected():
    """Periods before the revision match the no-revision roll, and chain."""
    mps = _portfolio()
    m = measure(mps, _basis())
    plain = roll_forward(m, 12)
    revised = roll_forward(m, 12, revised=_revised(mps), revised_at=24)
    for plain_p, rev_p in zip(plain[:2], revised[:2]):
        assert np.allclose(plain_p.csm_closing, rev_p.csm_closing)
        assert np.allclose(plain_p.bel_closing, rev_p.bel_closing)
    for prev, nxt in zip(revised, revised[1:]):
        assert np.allclose(prev.csm_closing, nxt.csm_opening)
        assert np.allclose(prev.bel_closing, nxt.bel_opening)


def test_roll_forward_rejects_lonely_revised():
    """revised and revised_at must be passed together."""
    m = measure(_portfolio(), _basis())
    with pytest.raises(ValueError, match="together"):
        roll_forward(m, 12, revised=m)


def test_roll_forward_rejects_off_boundary_revision():
    """The change month must be a multiple of the period length."""
    mps = _portfolio()
    m = measure(mps, _basis())
    with pytest.raises(ValueError, match="change month"):
        roll_forward(m, 12, revised=_revised(mps), revised_at=20)


def test_roll_forward_experience_scales_the_fcf():
    """In-force experience scales the closing FCF by the actual/expected ratio."""
    m = measure(_portfolio(), _basis())
    k = 24
    actual = 0.5 * m.cashflows.inforce[:, k]          # half the book remains
    periods = roll_forward(m, 12, actual_inforce=actual, experience_at=k)
    exp = next(p for p in periods if p.month_start == k)
    assert np.allclose(exp.bel_experience, m.bel_path[:, k] * -0.5)
    assert np.allclose(exp.ra_experience, m.ra_path[:, k] * -0.5)


def test_roll_forward_experience_reconciles():
    """With an experience adjustment, every period still reconciles exactly."""
    m = measure(_portfolio(), _basis())
    actual = 0.8 * m.cashflows.inforce[:, 24]
    for p in roll_forward(m, 12, actual_inforce=actual, experience_at=24):
        assert np.allclose(
            p.bel_opening + p.bel_assumption_change + p.bel_experience
            + p.bel_interest - p.bel_release, p.bel_closing)
        assert np.allclose(
            p.csm_opening + p.csm_assumption_change + p.csm_experience
            + p.csm_accretion - p.csm_release, p.csm_closing)


def test_roll_forward_experience_isolated_to_its_period():
    """The experience line is non-zero only in the experience period."""
    m = measure(_portfolio(), _basis())
    actual = 0.8 * m.cashflows.inforce[:, 24]
    for p in roll_forward(m, 12, actual_inforce=actual, experience_at=24):
        if p.month_start == 24:
            assert not np.allclose(p.bel_experience, 0.0)
        else:
            assert np.allclose(p.bel_experience, 0.0)
            assert np.allclose(p.csm_experience, 0.0)


def test_roll_forward_experience_pre_periods_unaffected():
    """Periods before the experience match the no-experience roll, and chain."""
    m = measure(_portfolio(), _basis())
    actual = 0.8 * m.cashflows.inforce[:, 24]
    plain = roll_forward(m, 12)
    exp = roll_forward(m, 12, actual_inforce=actual, experience_at=24)
    for plain_p, exp_p in zip(plain[:2], exp[:2]):
        assert np.allclose(plain_p.csm_closing, exp_p.csm_closing)
        assert np.allclose(plain_p.bel_closing, exp_p.bel_closing)
    for prev, nxt in zip(exp, exp[1:]):
        assert np.allclose(prev.csm_closing, nxt.csm_opening)
        assert np.allclose(prev.bel_closing, nxt.bel_opening)


def test_roll_forward_rejects_experience_and_revision_together():
    """v1 recognises a revision or experience, not both in one call."""
    mps = _portfolio()
    m = measure(mps, _basis())
    actual = 0.8 * m.cashflows.inforce[:, 24]
    with pytest.raises(ValueError, match="not both"):
        roll_forward(m, 12, revised=_revised(mps), revised_at=24,
                     actual_inforce=actual, experience_at=24)


def test_roll_forward_rejects_lonely_actual_inforce():
    """actual_inforce and experience_at must be passed together."""
    m = measure(_portfolio(), _basis())
    with pytest.raises(ValueError, match="actual_inforce"):
        roll_forward(m, 12, actual_inforce=m.cashflows.inforce[:, 24])


def test_reconcile_period_count_and_aggregation():
    """reconcile gives one portfolio-total table per movement."""
    m = measure(_portfolio(), _basis())
    movements = roll_forward(m, 12)
    recons = reconcile(movements)
    assert len(recons) == len(movements)
    assert np.isclose(recons[0].csm_opening, movements[0].csm_opening.sum())
    assert np.isclose(recons[0].bel_closing, movements[0].bel_closing.sum())


def test_reconcile_reconciles():
    """opening + future service + finance + release == closing, per column."""
    m = measure(_portfolio(), _basis())
    for r in reconcile(roll_forward(m, 12)):
        assert np.isclose(
            r.bel_opening + r.bel_future_service + r.bel_finance
            + r.bel_release, r.bel_closing)
        assert np.isclose(
            r.ra_opening + r.ra_future_service + r.ra_finance
            + r.ra_release, r.ra_closing)
        assert np.isclose(
            r.csm_opening + r.csm_future_service + r.csm_finance
            + r.csm_release, r.csm_closing)


def test_reconcile_carries_the_assumption_change():
    """The future-service row carries an assumption revision, and reconciles."""
    mps = _portfolio()
    m = measure(mps, _basis())
    recons = reconcile(roll_forward(m, 12, revised=_revised(mps), revised_at=24))
    rev = next(r for r in recons if r.month_start == 24)
    assert not np.isclose(rev.csm_future_service, 0.0)
    assert np.isclose(
        rev.csm_opening + rev.csm_future_service + rev.csm_finance
        + rev.csm_release, rev.csm_closing)


def test_reconcile_renders_a_table():
    """str(Reconciliation) is a readable three-column table."""
    m = measure(_portfolio(), _basis())
    text = str(reconcile(roll_forward(m, 12))[0])
    assert "Opening" in text and "Closing" in text and "CSM" in text


def test_roll_forward_paa_reconciles_the_lrc():
    """The PAA movement reconciles: opening + premiums - revenue = closing."""
    m = fcf.paa.measure(_portfolio(), _basis())
    for p in roll_forward(m, 12):
        assert np.allclose(p.lrc_opening + p.premiums - p.revenue, p.lrc_closing)


def test_roll_forward_paa_chains_from_zero():
    """The PAA LRC opens at zero and each period chains to the next."""
    periods = roll_forward(fcf.paa.measure(_portfolio(), _basis()), 12)
    assert np.allclose(periods[0].lrc_opening, 0.0)
    for prev, nxt in zip(periods, periods[1:]):
        assert prev.month_end == nxt.month_start
        assert np.allclose(prev.lrc_closing, nxt.lrc_opening)


def test_roll_forward_paa_rejects_gmm_options():
    """The revision and experience options do not apply to a PAA measurement."""
    paa = fcf.paa.measure(_portfolio(), _basis())
    gmm = measure(_portfolio(), _basis())
    with pytest.raises(ValueError, match="GMM"):
        roll_forward(paa, 12, revised=gmm, revised_at=24)


def test_roll_forward_paa_reconciles_all_three_components():
    """LRC, loss component and LIC each reconcile, with a settlement pattern."""
    basis = replace(_basis(), settlement_pattern=np.array([0.5, 0.3, 0.2]))
    for p in roll_forward(fcf.paa.measure(_portfolio(), basis), 12):
        assert np.allclose(p.lrc_opening + p.premiums - p.revenue, p.lrc_closing)
        assert np.allclose(p.loss_component_opening - p.loss_component_release, p.loss_component_closing)
        assert np.allclose(
            p.lic_opening + p.claims_incurred - p.claims_paid, p.lic_closing)


def test_paa_no_phantom_claims_paid_past_segment_horizon():
    """A stitched short PAA segment's parked LIC residual is not booked as paid.

    Two PAA segments (term 3 vs 8) share a [0.2]*5 settlement pattern that runs
    past the short term. The stitch carries the short row's beyond-horizon
    residual flat to the global terminal (month 8), so the roll-forward sees no
    settlement past month 3 -- claims_paid stays 0 there. The bug dropped the
    residual to zero at the segment horizon and booked a phantom claims_paid in
    the period that spanned the drop. The raw-movement invariant
    ``lic_opening + claims_incurred - claims_paid == lic_closing`` holds for
    every period either way.
    """
    def _sb():
        return make_death_basis(mortality_q=0.01, lapse_q=0.0,
                                discount_annual=0.0, settlement_pattern=np.full(5, 0.2))

    router = BasisRouter(
        {("P", "GA"): _sb(), ("Q", "GA"): _sb()},
        measurement_models={("P", "GA"): "PAA", ("Q", "GA"): "PAA"})
    mp = ModelPoints(
        issue_age=np.array([40, 40]), premium=np.array([0.0, 0.0]),
        term_months=np.array([3, 8]), benefits={"DEATH": np.array([1e6, 1e6])},
        calculation_methods=PATTERNS,
        product=np.array(["P", "Q"]), channel=np.array(["GA", "GA"]))
    pm = fcf.portfolio.measure(mp, router)
    periods = roll_forward(pm.paa.measurement, period_months=2)

    short = 0
    for p in periods:
        assert np.allclose(
            p.lic_opening + p.claims_incurred - p.claims_paid, p.lic_closing)
        if p.month_start >= 4:                       # fully past the short term (3)
            assert p.claims_paid[short] == pytest.approx(0.0)
            assert p.claims_incurred[short] == pytest.approx(0.0)
    assert periods[-1].lic_closing[short] > 0.0      # residual parked, not paid


def test_paa_lic_builds_with_a_settlement_pattern():
    """A settlement pattern makes the LIC non-zero; immediate settlement zeroes it."""
    lagged = fcf.paa.measure(
        _portfolio(),
        replace(_basis(), settlement_pattern=np.array([0.5, 0.3, 0.2])),
    )
    immediate = fcf.paa.measure(_portfolio(), _basis())
    assert np.any(lagged.lic > 0.0)
    assert np.allclose(immediate.lic, 0.0)


def test_gmm_lic_builds_with_a_settlement_pattern():
    """A settlement pattern gives the GMM measurement a non-zero LIC."""
    pattern = np.array([0.5, 0.3, 0.2])
    lagged = measure(_portfolio(), replace(_basis(), settlement_pattern=pattern))
    immediate = measure(_portfolio(), _basis())
    assert np.any(lagged.lic > 0.0)
    assert np.allclose(immediate.lic, 0.0)


def test_vfa_lic_builds_with_a_settlement_pattern():
    """A settlement pattern gives the VFA measurement a non-zero LIC."""
    pattern = np.array([0.6, 0.4])
    lagged = fcf.vfa.measure(_vfa_contract(),
                         replace(_vfa_assumptions(), settlement_pattern=pattern))
    immediate = fcf.vfa.measure(_vfa_contract(), _vfa_assumptions())
    assert np.any(lagged.lic > 0.0)
    assert np.allclose(immediate.lic, 0.0)


def test_settlement_lag_lowers_the_bel():
    """A settlement lag discounts claims to their payment dates -- a lower BEL."""
    immediate = measure(_portfolio(), _basis())
    lagged = measure(_portfolio(), replace(
        _basis(), settlement_pattern=np.array([0.2, 0.3, 0.5])))
    assert np.all(lagged.bel_path[:, 0] <= immediate.bel_path[:, 0])
    assert lagged.bel_path[:, 0].sum() < immediate.bel_path[:, 0].sum()


def test_settlement_lag_value_matches_measure():
    """measure() and measure() agree once the settlement lag is reflected."""
    basis = replace(_basis(), settlement_pattern=np.array([0.4, 0.6]))
    mps = _portfolio()
    v = measure(mps, basis, full=False)
    m = measure(mps, basis)
    assert np.allclose(v.bel, m.bel_path[:, 0])
    assert np.allclose(v.ra, m.ra_path[:, 0])
    assert np.allclose(v.csm, m.csm_path[:, 0])


def test_reconcile_paa():
    """The PAA reconciliation aggregates the three components and renders."""
    basis = replace(_basis(), settlement_pattern=np.array([0.6, 0.4]))
    recons = reconcile(roll_forward(fcf.paa.measure(_portfolio(), basis), 12))
    assert len(recons) == 10
    for r in recons:
        assert np.isclose(r.lrc_opening + r.premiums + r.revenue, r.lrc_closing)
        assert np.isclose(
            r.lic_opening + r.claims_incurred + r.claims_paid, r.lic_closing)
    text = str(recons[0])
    assert "LRC" in text and "incurred claims" in text


def _vfa_assumptions():
    return make_death_basis(
        mortality_q       = 0.002,
        lapse_q           = 0.004,
        discount_annual   = 0.03,
        ra_confidence     = 0.75,
        mortality_cv      = 0.10,
        investment_return = 0.06,
        fund_fee          = 0.015,
    )


def _vfa_contract() -> ModelPoints:
    return ModelPoints.single(40, 0.0, 120, account_value=1e8,
                              calculation_methods=PATTERNS)


def test_roll_forward_vfa_reconciles():
    """The VFA movement reconciles BEL, RA and CSM, opening to closing."""
    m = fcf.vfa.measure(_vfa_contract(), _vfa_assumptions())
    for p in roll_forward(m, 12):
        assert np.allclose(
            p.bel_opening + p.bel_interest - p.bel_release, p.bel_closing)
        assert np.allclose(
            p.ra_opening + p.ra_interest - p.ra_release, p.ra_closing)
        assert np.allclose(
            p.csm_opening + p.csm_accretion - p.csm_release, p.csm_closing)


def test_roll_forward_vfa_chains_and_runs_off():
    """The VFA balances build at inception and run off to zero, periods chaining."""
    periods = roll_forward(fcf.vfa.measure(_vfa_contract(), _vfa_assumptions()), 12)
    assert periods[0].csm_opening[0] > 0.0
    assert np.allclose(periods[-1].csm_closing, 0.0, atol=1.0)
    assert np.allclose(periods[-1].bel_closing, 0.0, atol=1.0)
    for prev, nxt in zip(periods, periods[1:]):
        assert prev.month_end == nxt.month_start
        assert np.allclose(prev.csm_closing, nxt.csm_opening)
        assert np.allclose(prev.bel_closing, nxt.bel_opening)


def test_roll_forward_vfa_rejects_gmm_options():
    """The revision and experience options do not apply to a VFA measurement."""
    m = fcf.vfa.measure(_vfa_contract(), _vfa_assumptions())
    with pytest.raises(ValueError, match="GMM"):
        roll_forward(m, 12, revised=measure(_portfolio(), _basis()),
                     revised_at=24)


def test_reconcile_vfa():
    """The VFA reconciliation aggregates, reconciles and renders."""
    recons = reconcile(roll_forward(fcf.vfa.measure(_vfa_contract(),
                                                _vfa_assumptions()), 12))
    assert len(recons) == 10
    for r in recons:
        assert np.isclose(
            r.bel_opening + r.bel_finance + r.bel_release, r.bel_closing)
        assert np.isclose(
            r.csm_opening + r.csm_finance + r.csm_release, r.csm_closing)
    assert "BEL" in str(recons[0]) and "CSM" in str(recons[0])


def test_roll_forward_experience_chain_on_segmented_measurement():
    """Regression (P0): the experience-chain interest accrual indexed
    monthly_rate[a:b] / monthly_rate[a] -- correct for a single-basis 1-D rate,
    but for a SEGMENTED measurement the rate is per-MP (n_mp, n_time) and that
    sliced the model-point axis, crashing / mis-computing. With the time-axis
    ellipsis it reconciles. Drives the chain path (2-D actual_inforce) on a
    dict-basis (segmented) measurement."""
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis()
    m = measure(mp, basis)                                 # segmented -> 2-D discount_bom
    assert m.discount_bom.ndim == 2
    actuals = np.stack([m.cashflows.inforce[:, 12] * 0.97,
                        m.cashflows.inforce[:, 24] * 0.93,
                        m.cashflows.inforce[:, 36] * 0.88])
    periods = roll_forward(m, 12, actual_inforce=actuals)
    assert len(periods) > 3
    for p in periods:
        assert np.allclose(
            p.bel_opening + p.bel_assumption_change + p.bel_experience
            + p.bel_interest - p.bel_release, p.bel_closing)
        assert np.allclose(
            p.csm_opening + p.csm_assumption_change + p.csm_experience
            + p.csm_accretion - p.csm_release, p.csm_closing)
