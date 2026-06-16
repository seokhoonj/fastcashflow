"""Phase-1 refactor: correctness-seam guards.

Three inputs the dataclass permits but the settle paths handled badly -- a
per-MP lock_in_rate that crashed opaquely on one path while another guarded it
cleanly, the settlement within-period experience arrays that were sliced and
consumed but never validated, and experience inputs silently dropped by
reinsurance.settle. Each is number-safe (it only rejects inputs that were
previously mis-handled), so no existing measurement moves.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from conftest import PATTERNS, make_death_basis


def _gmm_book(**state_over):
    basis = make_death_basis(mortality_q=0.01, lapse_q=0.02, discount_annual=0.03,
                             ra_confidence=0.75, mortality_cv=0.10)
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([36]), benefits={"DEATH": np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=PATTERNS),
        basis, full=True).cashflows.inforce[0]
    eo, ec = 12, 24
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([36]), benefits={"DEATH": np.array([1e6])},
        count=np.array([surv[ec]]), elapsed_months=np.array([ec]),
        mp_id=np.array(["P0"]), product=np.array(["A"]),
        calculation_methods=PATTERNS)
    kw = dict(mp_id=np.array(["P0"]), elapsed_months=np.array([ec]),
              count=np.array([surv[ec]]), prior_csm=np.array([0.0]),
              lock_in_rate=0.03, prior_count=np.array([surv[eo]]))
    kw.update(state_over)
    return mp, InforceState(**kw), basis


# ---------------------------------------------------------------------------
# guard 1: a per-MP lock_in_rate is rejected with a clear v1-scope message on
# every settle entry (it used to crash opaquely in float(state.lock_in_rate))
# ---------------------------------------------------------------------------

def test_gmm_settle_rejects_per_mp_lock_in_rate():
    mp, state, basis = _gmm_book(lock_in_rate=np.array([0.03]))
    with pytest.raises(NotImplementedError, match="lock_in_rate must be uniform"):
        fcf.gmm.settle(mp, state, basis, period_months=12)


def test_gmm_settle_aggregate_rejects_per_mp_lock_in_rate():
    mp, state, basis = _gmm_book(lock_in_rate=np.array([0.03]))
    with pytest.raises(NotImplementedError, match="lock_in_rate must be uniform"):
        fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)


def test_scalar_lock_in_rate_still_settles():
    mp, state, basis = _gmm_book()             # scalar lock_in_rate
    mv = fcf.gmm.settle(mp, state, basis, period_months=12)
    assert mv.bel_closing.shape == (1,)


# ---------------------------------------------------------------------------
# guard 2: the within-period experience arrays are validated (finite + length),
# but may be negative (favourable experience / a refund)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["actual_claims", "actual_expenses",
                                   "actual_investment_component", "actual_premium"])
def test_inforce_state_actual_fields_must_be_finite(field):
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        InforceState(mp_id=np.array(["P0"]), elapsed_months=np.array([24]),
                     count=np.array([1.0]), prior_csm=np.array([0.0]),
                     lock_in_rate=0.03, prior_count=np.array([1.0]),
                     **{field: np.array([np.nan])})


@pytest.mark.parametrize("field", ["actual_claims", "actual_expenses",
                                   "actual_investment_component", "actual_premium"])
def test_inforce_state_actual_fields_length_checked(field):
    with pytest.raises(ValueError, match=f"{field} has length"):
        InforceState(mp_id=np.array(["P0"]), elapsed_months=np.array([24]),
                     count=np.array([1.0]), prior_csm=np.array([0.0]),
                     lock_in_rate=0.03, prior_count=np.array([1.0]),
                     **{field: np.array([1.0, 2.0])})


def test_inforce_state_actual_fields_may_be_negative():
    # favourable experience / a net refund is legitimate -- no >= 0 rejection
    st = InforceState(mp_id=np.array(["P0"]), elapsed_months=np.array([24]),
                      count=np.array([1.0]), prior_csm=np.array([0.0]),
                      lock_in_rate=0.03, prior_count=np.array([1.0]),
                      actual_claims=np.array([-500.0]),
                      actual_premium=np.array([-50.0]))
    np.testing.assert_array_equal(st.actual_claims, [-500.0])


# ---------------------------------------------------------------------------
# guard 3: reinsurance.settle rejects within-period experience inputs it does
# not model (rather than silently dropping a reused gmm.settle state)
# ---------------------------------------------------------------------------

def _reins_book(**state_over):
    basis = make_death_basis(mortality_q=0.002, lapse_q=0.005, discount_annual=0.03,
                             ra_confidence=0.75, mortality_cv=0.10)
    treaty = fcf.reinsurance.QuotaShare(0.4)
    unit = ModelPoints.single(40, 400_000.0, 240, benefits={"DEATH": 1e8},
                              calculation_methods=PATTERNS)
    m = fcf.reinsurance.measure(unit, basis, treaty=treaty)
    surv = m.cashflows.inforce[0]
    eo, ec, scale = 24, 36, 1000.0
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([400_000.0]),
        term_months=np.array([240]), benefits={"DEATH": np.array([1e8])},
        count=np.array([scale * surv[ec]]), elapsed_months=np.array([ec]),
        mp_id=np.array(["R0"]), calculation_methods=PATTERNS)
    kw = dict(mp_id=np.array(["R0"]), elapsed_months=np.array([ec]),
              count=np.array([scale * surv[ec]]),
              prior_csm=np.array([float(m.csm_path[0, eo]) * scale]),
              lock_in_rate=0.03, prior_count=np.array([scale * surv[eo]]))
    kw.update(state_over)
    return mp, InforceState(**kw), basis, treaty


@pytest.mark.parametrize("field", ["actual_premium", "actual_claims",
                                   "actual_expenses", "actual_investment_component"])
def test_reinsurance_settle_rejects_experience_inputs(field):
    mp, state, basis, treaty = _reins_book(**{field: np.array([100.0])})
    with pytest.raises(NotImplementedError, match="within-period experience"):
        fcf.reinsurance.settle(mp, state, basis, treaty=treaty, period_months=12)


def test_reinsurance_settle_clean_state_still_settles():
    mp, state, basis, treaty = _reins_book()
    mv = fcf.reinsurance.settle(mp, state, basis, treaty=treaty, period_months=12)
    assert mv.bel_closing.shape == (1,)
