"""Period-close roll-forward validation -- the expected-basis analysis of change.

``roll_forward`` slices a GMM measurement into reporting-period movements.
Each period must reconcile (opening + movements = closing) and consecutive
periods must chain.
"""
from dataclasses import replace

import numpy as np
import pytest

from fastcashflow import Assumptions, ModelPointSet, measure, roll_forward


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.001),
        lapse_monthly=lambda duration: np.full(duration.shape, 0.01),
        discount_annual=0.03,
        expense_acquisition=200_000.0,
        expense_maintenance_annual=60_000.0,
        expense_inflation=0.02,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )


def _portfolio(n: int = 50) -> ModelPointSet:
    rng = np.random.default_rng(4)
    return ModelPointSet(
        issue_age=rng.integers(30, 55, n),
        death_benefit=rng.integers(20, 90, n) * 1_000_000,
        monthly_premium=rng.integers(8, 20, n) * 10_000,
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


def _revised(mps: ModelPointSet):
    """A measurement of the same book under markedly higher mortality."""
    worse = replace(
        _assumptions(),
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.003),
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
