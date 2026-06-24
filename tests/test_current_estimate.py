"""Current-estimate trajectory accessor -- ``Measurement.estimate_at(t)``.

``estimate_at(t)`` returns the cohort BEL / RA / CSM / LIC at a future month t
(IFRS 17 Sec. 40, the deterministic nested-projection view). It is column t of
the measurement's trajectories, so it must agree with two independent code
paths:

* a fresh in-force measurement at that elapsed month carrying the deterministic
  survivor count (``gmm.measure_inforce(elapsed=t)``), and
* the period waterfall (``roll_forward``), whose opening / closing balances are
  the same trajectory columns.

The hand-calc anchor pins the arithmetic on a tiny case derived by hand.
"""
import numpy as np
import pytest

from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow.gmm import measure
# _measure_inforce_fast is the engine-internal workhorse behind the public
# gmm.measure_inforce; called directly here (as in tests/test_inforce.py) so the
# re-anchor sweep does not need to build an InforceState.
from fastcashflow.gmm._engine import _measure_inforce_fast
from fastcashflow.movement import roll_forward


def _flat_rate(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis():
    return Basis(
        mortality_annual=_flat_rate(0.005),
        lapse_annual=_flat_rate(0.05),
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat_rate(0.005)),),
    )


def _single(term_months=120):
    return ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=50_000.0, term_months=term_months,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )


# ---------------------------------------------------------------------------
# Inception identity + hand-calc anchor
# ---------------------------------------------------------------------------

def test_estimate_at_zero_is_inception_headline():
    """``estimate_at(0)`` is the inception column, so it equals the headline."""
    m = measure(_single(), _basis())
    e0 = m.estimate_at(0)
    assert np.allclose(e0.bel, m.bel)
    assert np.allclose(e0.ra, m.ra)
    assert np.allclose(e0.csm, m.csm)


def test_estimate_at_bel_handcalc():
    """A 3-month, 1-policy contract with zero discount, 1% mortality, no lapse
    and no expense, so every BEL_t is derived by hand: the cohort BEL at month t
    is the (undiscounted) sum of future death claims minus future premiums over
    the in-force projected from inception."""
    q, db, prem, term = 0.01, 1_000_000.0, 12_000.0, 3
    basis = Basis(
        mortality_annual=_flat_rate(q), lapse_annual=_flat_rate(0.0),
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", _flat_rate(q)),),
    )
    mp = ModelPoints.single(
        issue_age=40, benefits={"DEATH": db}, premium=prem, term_months=term,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    m = measure(mp, basis)

    mq = 1.0 - (1.0 - q) ** (1.0 / 12.0)        # engine converts annual -> monthly
    inforce = (1.0 - mq) ** np.arange(term)     # [1, 1-mq, (1-mq)^2]
    deaths = inforce * mq
    for t in range(term):
        bel_t = float(np.sum(deaths[t:] * db) - np.sum(inforce[t:] * prem))
        assert np.isclose(m.estimate_at(t).bel[0], bel_t, rtol=1e-9), f"t={t}"


# ---------------------------------------------------------------------------
# Cross-validation: trajectory == in-force re-anchor at every t
# ---------------------------------------------------------------------------

def test_estimate_at_matches_inforce_reanchor():
    """For every t, ``estimate_at(t)`` equals a fresh in-force measurement at
    elapsed t carrying the deterministic survivor count -- the trajectory and
    the independent re-anchor code path agree. This generalises
    ``test_inforce_fast_matches_trajectory_slice`` from one elapsed point to a
    sweep across the horizon."""
    basis = _basis()
    m = measure(_single(term_months=120), basis)
    n_time = m.bel_path.shape[1] - 1

    ts = np.array([0, 1, 12, 36, 60, 119])      # spread incl. both edges, < n_time
    assert ts.max() < n_time
    survivors = m.cashflows.inforce[0, ts]      # deterministic in-force at each t
    k = len(ts)
    mp_inforce = ModelPoints(
        issue_age=np.full(k, 40),
        premium=np.full(k, 50_000.0),
        term_months=np.full(k, 120),
        benefits={"DEATH": np.full(k, 100_000_000.0)},
        count=survivors,                        # as-of count = deterministic survivors
        elapsed_months=ts,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    v = _measure_inforce_fast(mp_inforce, basis)
    for i, t in enumerate(ts):
        e = m.estimate_at(int(t))
        assert np.isclose(v.bel[i], e.bel[0], rtol=1e-9), f"bel t={t}"
        assert np.isclose(v.ra[i], e.ra[0], rtol=1e-9), f"ra t={t}"
    # CSM is deliberately NOT asserted against this re-anchor: the hypothetical
    # in-force CSM (prior_csm=None) is re-floored from the in-force contract's own
    # inception and is path-dependent, so it legitimately differs from the
    # from-inception trajectory slice (the measure_inforce docstring warns of
    # exactly this). The CSM trajectory column is cross-validated instead against
    # the roll_forward waterfall below, and -- transitively -- against the
    # settlement-carry path in tests/test_inforce.py.


# ---------------------------------------------------------------------------
# Cross-validation: trajectory == roll_forward waterfall
# ---------------------------------------------------------------------------

def test_estimate_at_reconciles_with_roll_forward():
    """The roll-forward period openings / closings are the same trajectory
    columns, so each period's opening equals ``estimate_at(month_start)`` and its
    closing equals ``estimate_at(month_end)``."""
    m = measure(_single(term_months=120), _basis())
    for p in roll_forward(m, period_months=12):
        eo = m.estimate_at(p.month_start)
        ec = m.estimate_at(p.month_end)
        assert np.allclose(eo.bel, p.bel_opening)
        assert np.allclose(eo.ra, p.ra_opening)
        assert np.allclose(eo.csm, p.csm_opening)
        assert np.allclose(ec.bel, p.bel_closing)
        assert np.allclose(ec.ra, p.ra_closing)
        assert np.allclose(ec.csm, p.csm_closing)


# ---------------------------------------------------------------------------
# Views and guards
# ---------------------------------------------------------------------------

def test_estimate_at_derived_views():
    """fcf = BEL + RA, lrc = FCF + CSM, per_survivor = money / inforce."""
    m = measure(_single(), _basis())
    e = m.estimate_at(60)
    assert np.allclose(e.fcf, e.bel + e.ra)
    assert np.allclose(e.lrc, e.bel + e.ra + e.csm)
    ps = e.per_survivor
    assert np.allclose(ps.bel, e.bel / e.inforce)
    assert np.allclose(ps.inforce, 1.0)


def test_estimate_at_terminal_column_runs_off():
    """The terminal column (t == n_time) uses the maturity exit count for
    in-force and carries the run-off balance (BEL near zero for a protection
    book that has fully matured)."""
    m = measure(_single(term_months=120), _basis())
    n_time = m.bel_path.shape[1] - 1
    et = m.estimate_at(n_time)
    assert np.allclose(et.inforce, m.cashflows.maturity_survivors)
    assert np.all(np.isfinite(et.per_survivor.bel))


def test_estimate_at_requires_full():
    """The fast path (full=False) carries no trajectory, so estimate_at raises."""
    m = measure(_single(), _basis(), full=False)
    with pytest.raises(ValueError, match="full=True"):
        m.estimate_at(0)


def test_estimate_at_rejects_out_of_range():
    m = measure(_single(term_months=120), _basis())
    n_time = m.bel_path.shape[1] - 1
    with pytest.raises(ValueError, match="horizon"):
        m.estimate_at(n_time + 1)
    with pytest.raises(ValueError, match="horizon"):
        m.estimate_at(-1)
