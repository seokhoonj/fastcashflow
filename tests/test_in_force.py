"""In-force valuation (IFRS 17 subsequent measurement) -- MVP tests.

``value_in_force(mp, basis)`` returns the BEL / RA / CSM at each
contract's valuation date, ``elapsed_months[mp]`` months after that
contract's inception. The projection runs from inception and the
trajectory is sliced at ``t = elapsed_months[mp]``.

The headline equivalence test: an in-force MP with ``elapsed_months = E``
and term ``T`` must produce the same in-force BEL as the matching
new-business MP measured ``E`` months into its life. This is the BEL
trajectory-slice property, and it pins the in-force semantics.
"""
import numpy as np

from fastcashflow import (
    Assumptions, ModelPoints, measure, value, value_in_force,
)


def _flat_rate(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis():
    return Assumptions(
        mortality_annual=_flat_rate(0.005),
        lapse_annual=_flat_rate(0.05),
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.10,
        morbidity_cv=0.10,
        alpha_flat=100_000.0,
        gamma_flat=30_000.0,
        expense_inflation=0.02,
    )


def test_value_in_force_zero_elapsed_matches_value():
    """When every ``elapsed_months`` is 0 the in-force valuation collapses
    to the new-business :func:`value` (= ``Measurement.bel[:, 0]``)."""
    mp = ModelPoints.single(
        issue_age=40, death_benefit=100_000_000.0,
        level_premium=50_000.0, term_months=120,
    )
    asmp = _basis()
    v_new = value(mp, asmp)
    v_inf = value_in_force(mp, asmp)
    assert np.isclose(v_inf.bel[0], v_new.bel[0])
    assert np.isclose(v_inf.ra[0], v_new.ra[0])
    assert np.isclose(v_inf.csm[0], v_new.csm[0])


def test_value_in_force_matches_trajectory_slice():
    """An in-force MP with ``elapsed_months = E`` returns the trajectory
    slice ``Measurement.bel[mp, E]`` -- the PV of future cash flows from
    the valuation date forward."""
    elapsed = 36
    mp_new = ModelPoints.single(
        issue_age=40, death_benefit=100_000_000.0,
        level_premium=50_000.0, term_months=120,
    )
    asmp = _basis()
    m = measure(mp_new, asmp)

    mp_inforce = ModelPoints(
        issue_age=np.array([40]),
        level_premium=np.array([50_000.0]),
        term_months=np.array([120]),
        death_benefit=np.array([100_000_000.0]),
        elapsed_months=np.array([elapsed]),
    )
    v_inf = value_in_force(mp_inforce, asmp)
    # The in-force BEL is the trajectory slice at t = elapsed.
    assert np.isclose(v_inf.bel[0], m.bel[0, elapsed])
    assert np.isclose(v_inf.ra[0], m.ra[0, elapsed])


def test_in_force_bel_smaller_term_left():
    """As ``elapsed_months`` grows (less of the term left), the absolute
    value of the in-force BEL shrinks -- there are fewer future cash flows
    to discount."""
    asmp = _basis()
    def in_force_bel(e):
        mp = ModelPoints(
            issue_age=np.array([40]),
            level_premium=np.array([50_000.0]),
            term_months=np.array([240]),
            death_benefit=np.array([100_000_000.0]),
            elapsed_months=np.array([e]),
        )
        return abs(value_in_force(mp, asmp).bel[0])
    # Strictly decreasing in elapsed -- the future shortens.
    bels = [in_force_bel(e) for e in (0, 60, 120, 180)]
    assert bels[0] > bels[1] > bels[2] > bels[3]
