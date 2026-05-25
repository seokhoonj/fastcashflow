"""Period-close roll-forward validation -- the expected-basis analysis of change.

``roll_forward`` slices a GMM measurement into reporting-period movements.
Each period must reconcile (opening + movements = closing) and consecutive
periods must chain.
"""
from dataclasses import replace

import numpy as np
import pytest

from fastcashflow import (
    Assumptions,
    ModelPoints,
    measure,
    measure_paa,
    measure_vfa,
    reconcile,
    roll_forward,
    value,
)


def _annual(m: float) -> float:
    """Convert a monthly rate to its annual equivalent so the engine converts back."""
    return 1.0 - (1.0 - m) ** 12


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.001)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.01)),
        discount_annual=0.03,
        alpha_flat=200_000.0,
        gamma_flat=60_000.0,
        expense_inflation=0.02,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )


def _portfolio(n: int = 50) -> ModelPoints:
    rng = np.random.default_rng(4)
    return ModelPoints(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(20, 90, n) * 1_000_000,
        level_premium=rng.integers(8, 20, n) * 10_000,
        term_months=np.full(n, 120),
    )


def test_roll_forward_period_count():
    """A 120-month horizon in yearly periods gives ten movements."""
    periods = roll_forward(measure(_portfolio(), _assumptions()), period_months=12)
    assert len(periods) == 10


def test_roll_forward_csm_reconciles():
    """Each period: opening + accretion - release = closing CSM."""
    for p in roll_forward(measure(_portfolio(), _assumptions()), 12):
        assert np.allclose(
            p.csm_opening + p.csm_accretion - p.csm_release, p.csm_closing
        )


def test_roll_forward_bel_and_ra_reconcile():
    """Each period: opening + interest - release = closing, for BEL and RA."""
    for p in roll_forward(measure(_portfolio(), _assumptions()), 12):
        assert np.allclose(
            p.bel_opening + p.bel_interest - p.bel_release, p.bel_closing
        )
        assert np.allclose(
            p.ra_opening + p.ra_interest - p.ra_release, p.ra_closing
        )


def test_roll_forward_periods_chain():
    """Each period's closing balances are the next period's opening balances."""
    periods = roll_forward(measure(_portfolio(), _assumptions()), 12)
    for prev, nxt in zip(periods, periods[1:]):
        assert prev.month_end == nxt.month_start
        assert np.allclose(prev.csm_closing, nxt.csm_opening)
        assert np.allclose(prev.bel_closing, nxt.bel_opening)
        assert np.allclose(prev.ra_closing, nxt.ra_opening)


def test_roll_forward_opening_is_inception():
    """The first period opens at the inception measurement."""
    m = measure(_portfolio(), _assumptions())
    first = roll_forward(m, 12)[0]
    assert first.month_start == 0
    assert np.allclose(first.csm_opening, m.csm[:, 0])
    assert np.allclose(first.bel_opening, m.bel[:, 0])


def test_roll_forward_runs_off_to_zero():
    """The final period closes the contract -- CSM and BEL run off to zero."""
    last = roll_forward(measure(_portfolio(), _assumptions()), 12)[-1]
    assert np.allclose(last.csm_closing, 0.0, atol=1.0)
    assert np.allclose(last.bel_closing, 0.0, atol=1.0)


def test_roll_forward_uneven_last_period():
    """A horizon not divisible by the period gives a short final period."""
    periods = roll_forward(measure(_portfolio(), _assumptions()), period_months=7)
    assert periods[-1].month_end == 120
    assert sum(p.month_end - p.month_start for p in periods) == 120


def test_roll_forward_rejects_bad_period():
    """A non-positive period length is an error."""
    with pytest.raises(ValueError, match="period_months"):
        roll_forward(measure(_portfolio(), _assumptions()), 0)


def _actuals(m, factors):
    """Actual in-force at successive boundaries -- expected scaled by factors."""
    return np.array([m.cashflows.inforce[:, (j + 1) * 12] * f
                     for j, f in enumerate(factors)])


def test_roll_forward_multi_experience_reconciles():
    """Experience at every period boundary -- each period reconciles exactly."""
    m = measure(_portfolio(), _assumptions())
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
    m = measure(_portfolio(), _assumptions())
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
    m = measure(_portfolio(), _assumptions())
    actual_12 = m.cashflows.inforce[:, 12] * 0.9
    multi = roll_forward(m, 12, actual_inforce=actual_12.reshape(1, -1))
    single = roll_forward(m, 12, actual_inforce=actual_12, experience_at=12)
    for p_multi, p_single in zip(multi, single):
        assert np.allclose(p_multi.bel_closing, p_single.bel_closing)
        assert np.allclose(p_multi.csm_closing, p_single.csm_closing)
        assert np.allclose(p_multi.csm_experience, p_single.csm_experience)


def test_roll_forward_multi_rejects_revision_or_experience_at():
    """A 2-D actual_inforce does not combine with revised or experience_at."""
    m = measure(_portfolio(), _assumptions())
    actuals = _actuals(m, [0.95, 0.95])
    with pytest.raises(ValueError, match="do not apply"):
        roll_forward(m, 12, actual_inforce=actuals, revised=m, revised_at=24)


def test_roll_forward_multi_rejects_too_many_rows():
    """Experience rows past the projection horizon are an error."""
    m = measure(_portfolio(), _assumptions())          # 120-month horizon
    actuals = np.ones((11, m.bel.shape[0]))            # boundary 132 > 120
    with pytest.raises(ValueError, match="horizon"):
        roll_forward(m, 12, actual_inforce=actuals)


def _revised(mps: ModelPoints):
    """A measurement of the same book under markedly higher mortality."""
    worse = replace(
        _assumptions(),
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.003)),
    )
    return measure(mps, worse)


def test_roll_forward_assumption_change_reconciles():
    """With a revision, every period still reconciles exactly."""
    mps = _portfolio()
    periods = roll_forward(measure(mps, _assumptions()), 12,
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
    periods = roll_forward(measure(mps, _assumptions()), 12,
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
    periods = roll_forward(measure(mps, _assumptions()), 12,
                           revised=_revised(mps), revised_at=24)
    rev = next(p for p in periods if p.month_start == 24)
    assert np.all(rev.bel_assumption_change > 0.0)        # higher claims
    assert np.all(rev.csm_assumption_change <= 0.0)       # CSM absorbs it


def test_roll_forward_pre_revision_periods_unaffected():
    """Periods before the revision match the no-revision roll, and chain."""
    mps = _portfolio()
    m = measure(mps, _assumptions())
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
    m = measure(_portfolio(), _assumptions())
    with pytest.raises(ValueError, match="together"):
        roll_forward(m, 12, revised=m)


def test_roll_forward_rejects_off_boundary_revision():
    """The change month must be a multiple of the period length."""
    mps = _portfolio()
    m = measure(mps, _assumptions())
    with pytest.raises(ValueError, match="change month"):
        roll_forward(m, 12, revised=_revised(mps), revised_at=20)


def test_roll_forward_experience_scales_the_fcf():
    """In-force experience scales the closing FCF by the actual/expected ratio."""
    m = measure(_portfolio(), _assumptions())
    k = 24
    actual = 0.5 * m.cashflows.inforce[:, k]          # half the book remains
    periods = roll_forward(m, 12, actual_inforce=actual, experience_at=k)
    exp = next(p for p in periods if p.month_start == k)
    assert np.allclose(exp.bel_experience, m.bel[:, k] * -0.5)
    assert np.allclose(exp.ra_experience, m.ra[:, k] * -0.5)


def test_roll_forward_experience_reconciles():
    """With an experience adjustment, every period still reconciles exactly."""
    m = measure(_portfolio(), _assumptions())
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
    m = measure(_portfolio(), _assumptions())
    actual = 0.8 * m.cashflows.inforce[:, 24]
    for p in roll_forward(m, 12, actual_inforce=actual, experience_at=24):
        if p.month_start == 24:
            assert not np.allclose(p.bel_experience, 0.0)
        else:
            assert np.allclose(p.bel_experience, 0.0)
            assert np.allclose(p.csm_experience, 0.0)


def test_roll_forward_experience_pre_periods_unaffected():
    """Periods before the experience match the no-experience roll, and chain."""
    m = measure(_portfolio(), _assumptions())
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
    m = measure(mps, _assumptions())
    actual = 0.8 * m.cashflows.inforce[:, 24]
    with pytest.raises(ValueError, match="not both"):
        roll_forward(m, 12, revised=_revised(mps), revised_at=24,
                     actual_inforce=actual, experience_at=24)


def test_roll_forward_rejects_lonely_actual_inforce():
    """actual_inforce and experience_at must be passed together."""
    m = measure(_portfolio(), _assumptions())
    with pytest.raises(ValueError, match="actual_inforce"):
        roll_forward(m, 12, actual_inforce=m.cashflows.inforce[:, 24])


def test_reconcile_period_count_and_aggregation():
    """reconcile gives one portfolio-total table per movement."""
    m = measure(_portfolio(), _assumptions())
    movements = roll_forward(m, 12)
    recons = reconcile(movements)
    assert len(recons) == len(movements)
    assert np.isclose(recons[0].csm_opening, movements[0].csm_opening.sum())
    assert np.isclose(recons[0].bel_closing, movements[0].bel_closing.sum())


def test_reconcile_reconciles():
    """opening + future service + finance + release == closing, per column."""
    m = measure(_portfolio(), _assumptions())
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
    m = measure(mps, _assumptions())
    recons = reconcile(roll_forward(m, 12, revised=_revised(mps), revised_at=24))
    rev = next(r for r in recons if r.month_start == 24)
    assert not np.isclose(rev.csm_future_service, 0.0)
    assert np.isclose(
        rev.csm_opening + rev.csm_future_service + rev.csm_finance
        + rev.csm_release, rev.csm_closing)


def test_reconcile_renders_a_table():
    """str(Reconciliation) is a readable three-column table."""
    m = measure(_portfolio(), _assumptions())
    text = str(reconcile(roll_forward(m, 12))[0])
    assert "Opening" in text and "Closing" in text and "CSM" in text


def test_roll_forward_paa_reconciles_the_lrc():
    """The PAA movement reconciles: opening + premiums - revenue = closing."""
    m = measure_paa(_portfolio(), _assumptions())
    for p in roll_forward(m, 12):
        assert np.allclose(p.lrc_opening + p.premiums - p.revenue, p.lrc_closing)


def test_roll_forward_paa_chains_from_zero():
    """The PAA LRC opens at zero and each period chains to the next."""
    periods = roll_forward(measure_paa(_portfolio(), _assumptions()), 12)
    assert np.allclose(periods[0].lrc_opening, 0.0)
    for prev, nxt in zip(periods, periods[1:]):
        assert prev.month_end == nxt.month_start
        assert np.allclose(prev.lrc_closing, nxt.lrc_opening)


def test_roll_forward_paa_rejects_gmm_options():
    """The revision and experience options do not apply to a PAA measurement."""
    paa = measure_paa(_portfolio(), _assumptions())
    gmm = measure(_portfolio(), _assumptions())
    with pytest.raises(ValueError, match="GMM"):
        roll_forward(paa, 12, revised=gmm, revised_at=24)


def test_roll_forward_paa_reconciles_all_three_components():
    """LRC, loss component and LIC each reconcile, with a settlement pattern."""
    asmp = replace(_assumptions(), settlement_pattern=np.array([0.5, 0.3, 0.2]))
    for p in roll_forward(measure_paa(_portfolio(), asmp), 12):
        assert np.allclose(p.lrc_opening + p.premiums - p.revenue, p.lrc_closing)
        assert np.allclose(p.lc_opening - p.lc_release, p.lc_closing)
        assert np.allclose(
            p.lic_opening + p.claims_incurred - p.claims_paid, p.lic_closing)


def test_paa_lic_builds_with_a_settlement_pattern():
    """A settlement pattern makes the LIC non-zero; immediate settlement zeroes it."""
    lagged = measure_paa(
        _portfolio(),
        replace(_assumptions(), settlement_pattern=np.array([0.5, 0.3, 0.2])),
    )
    immediate = measure_paa(_portfolio(), _assumptions())
    assert np.any(lagged.lic > 0.0)
    assert np.allclose(immediate.lic, 0.0)


def test_gmm_lic_builds_with_a_settlement_pattern():
    """A settlement pattern gives the GMM measurement a non-zero LIC."""
    pattern = np.array([0.5, 0.3, 0.2])
    lagged = measure(_portfolio(), replace(_assumptions(), settlement_pattern=pattern))
    immediate = measure(_portfolio(), _assumptions())
    assert np.any(lagged.lic > 0.0)
    assert np.allclose(immediate.lic, 0.0)


def test_vfa_lic_builds_with_a_settlement_pattern():
    """A settlement pattern gives the VFA measurement a non-zero LIC."""
    pattern = np.array([0.6, 0.4])
    lagged = measure_vfa(_vfa_contract(),
                         replace(_vfa_assumptions(), settlement_pattern=pattern))
    immediate = measure_vfa(_vfa_contract(), _vfa_assumptions())
    assert np.any(lagged.lic > 0.0)
    assert np.allclose(immediate.lic, 0.0)


def test_settlement_lag_lowers_the_bel():
    """A settlement lag discounts claims to their payment dates -- a lower BEL."""
    immediate = measure(_portfolio(), _assumptions())
    lagged = measure(_portfolio(), replace(
        _assumptions(), settlement_pattern=np.array([0.2, 0.3, 0.5])))
    assert np.all(lagged.bel[:, 0] <= immediate.bel[:, 0])
    assert lagged.bel[:, 0].sum() < immediate.bel[:, 0].sum()


def test_settlement_lag_value_matches_measure():
    """value() and measure() agree once the settlement lag is reflected."""
    asmp = replace(_assumptions(), settlement_pattern=np.array([0.4, 0.6]))
    mps = _portfolio()
    v = value(mps, asmp)
    m = measure(mps, asmp)
    assert np.allclose(v.bel, m.bel[:, 0])
    assert np.allclose(v.ra, m.ra[:, 0])
    assert np.allclose(v.csm, m.csm[:, 0])


def test_reconcile_paa():
    """The PAA reconciliation aggregates the three components and renders."""
    asmp = replace(_assumptions(), settlement_pattern=np.array([0.6, 0.4]))
    recons = reconcile(roll_forward(measure_paa(_portfolio(), asmp), 12))
    assert len(recons) == 10
    for r in recons:
        assert np.isclose(r.lrc_opening + r.premiums + r.revenue, r.lrc_closing)
        assert np.isclose(
            r.lic_opening + r.claims_incurred + r.claims_paid, r.lic_closing)
    text = str(recons[0])
    assert "LRC" in text and "incurred claims" in text


def _vfa_assumptions() -> Assumptions:
    return Assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.002)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.004)),
        discount_annual=0.03,
        alpha_flat=0.0,
        gamma_flat=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
        investment_return=0.06,
        fund_fee=0.015,
    )


def _vfa_contract() -> ModelPoints:
    return ModelPoints.single(40, 0.0, 0.0, 120, account_value=1e8)


def test_roll_forward_vfa_reconciles():
    """The VFA movement reconciles BEL, RA and CSM, opening to closing."""
    m = measure_vfa(_vfa_contract(), _vfa_assumptions())
    for p in roll_forward(m, 12):
        assert np.allclose(
            p.bel_opening + p.bel_interest - p.bel_release, p.bel_closing)
        assert np.allclose(
            p.ra_opening + p.ra_interest - p.ra_release, p.ra_closing)
        assert np.allclose(
            p.csm_opening + p.csm_accretion - p.csm_release, p.csm_closing)


def test_roll_forward_vfa_chains_and_runs_off():
    """The VFA balances build at inception and run off to zero, periods chaining."""
    periods = roll_forward(measure_vfa(_vfa_contract(), _vfa_assumptions()), 12)
    assert periods[0].csm_opening[0] > 0.0
    assert np.allclose(periods[-1].csm_closing, 0.0, atol=1.0)
    assert np.allclose(periods[-1].bel_closing, 0.0, atol=1.0)
    for prev, nxt in zip(periods, periods[1:]):
        assert prev.month_end == nxt.month_start
        assert np.allclose(prev.csm_closing, nxt.csm_opening)
        assert np.allclose(prev.bel_closing, nxt.bel_opening)


def test_roll_forward_vfa_rejects_gmm_options():
    """The revision and experience options do not apply to a VFA measurement."""
    m = measure_vfa(_vfa_contract(), _vfa_assumptions())
    with pytest.raises(ValueError, match="GMM"):
        roll_forward(m, 12, revised=measure(_portfolio(), _assumptions()),
                     revised_at=24)


def test_reconcile_vfa():
    """The VFA reconciliation aggregates, reconciles and renders."""
    recons = reconcile(roll_forward(measure_vfa(_vfa_contract(),
                                                _vfa_assumptions()), 12))
    assert len(recons) == 10
    for r in recons:
        assert np.isclose(
            r.bel_opening + r.bel_finance + r.bel_release, r.bel_closing)
        assert np.isclose(
            r.csm_opening + r.csm_finance + r.csm_release, r.csm_closing)
    assert "BEL" in str(recons[0]) and "CSM" in str(recons[0])
