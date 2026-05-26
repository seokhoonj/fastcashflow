"""Fast path surrender_cf -- value() must match measure() BEL.

``measure()`` has carried surrender value (해약환급금) since the surrender
mechanism landed; ``value()`` (the fused fast path) ignored it. This file
pins the invariant that the two paths return the same BEL when a
``surrender_value_curve`` is set, across all CPU kernels:

* scalar fast path (no StateModel, no waiver),
* Markov codegen (WAIVER_MODEL),
* semi-Markov codegen (sojourn-aware cohort tracking).
"""
import numpy as np

from fastcashflow import (
    Assumptions, ExpenseItem, ModelPoints, STATE_MODELS, measure, value,
    CoverageRate,
)


def _flat_rate(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis(**overrides):
    base = dict(
        mortality_annual=_flat_rate(0.005),
        lapse_annual=_flat_rate(0.05),
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.10,
        morbidity_cv=0.10,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    100_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  30_000.0),
        ),
        coverages=(CoverageRate("DEATH", _flat_rate(0.005)),),
    )
    base.update(overrides)
    return Assumptions(**base)


def _mp():
    return ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        level_premium=50_000.0, term_months=240,
    )


def test_value_scalar_matches_measure_with_surrender():
    """Scalar fast path -- no StateModel, no waiver. value() must agree
    with measure() to floating-point tolerance once surrender is on."""
    n_time = 240
    # A non-trivial monotone surrender curve: 0 in years 1-2 (typical
    # surrender penalty), ramping up to 1.0 by the end of the term.
    curve = np.clip((np.arange(n_time) - 24) / (n_time - 24.0), 0.0, 1.0)
    asmp = _basis(surrender_value_curve=curve)
    mp = _mp()
    v = value(mp, asmp)
    m = measure(mp, asmp)
    assert np.isclose(v.bel[0], m.bel[0, 0])


def test_value_scalar_zero_curve_matches_no_surrender():
    """``surrender_value_curve = 0`` everywhere collapses to the legacy
    no-surrender BEL."""
    n_time = 240
    asmp_off = _basis()
    asmp_on = _basis(surrender_value_curve=np.zeros(n_time))
    mp = _mp()
    assert np.isclose(value(mp, asmp_off).bel[0], value(mp, asmp_on).bel[0])


def test_value_surrender_increases_bel():
    """Adding a positive surrender curve adds an insurer outflow on lapse;
    BEL (claims - premiums + surrender) goes up."""
    n_time = 240
    asmp_off = _basis()
    asmp_on = _basis(surrender_value_curve=np.full(n_time, 0.5))
    mp = _mp()
    assert value(mp, asmp_on).bel[0] > value(mp, asmp_off).bel[0]


def test_surrender_scales_linearly_in_count():
    """A 10-policy grouped MP must produce 10x the surrender flow of an
    otherwise-identical 1-policy MP -- not 100x. Earlier the projection
    multiplied lapse_flow (already inforce-weighted) by cum_premium (also
    inforce-weighted), giving a cnt^2 scaling."""
    n_time = 120
    asmp = _basis(surrender_value_curve=np.full(n_time, 0.5))
    mp_single = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        level_premium=50_000.0, term_months=n_time, count=1.0,
    )
    mp_grouped = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        level_premium=50_000.0, term_months=n_time, count=10.0,
    )
    m_single = measure(mp_single, asmp)
    m_grouped = measure(mp_grouped, asmp)
    ratio = m_grouped.cashflows.surrender_cf.sum() / m_single.cashflows.surrender_cf.sum()
    # Linear in count -- ratio must be ~10, not ~100.
    assert np.isclose(ratio, 10.0)


def test_value_state_model_matches_measure_with_surrender():
    """Markov codegen path (WAIVER_MODEL) -- value() must agree with
    measure() once surrender is on."""
    n_time = 240
    curve = np.clip((np.arange(n_time) - 24) / (n_time - 24.0), 0.0, 1.0)
    asmp = _basis(
        surrender_value_curve=curve,
        state_model=STATE_MODELS["WAIVER"],
        waiver_incidence_annual=_flat_rate(0.001),
    )
    mp = _mp()
    v = value(mp, asmp)
    m = measure(mp, asmp)
    assert np.isclose(v.bel[0], m.bel[0, 0])
