"""In-force valuation (IFRS 17 subsequent measurement) -- MVP tests.

``_measure_inforce_fast(mp, basis)`` returns the BEL / RA / CSM at each
contract's valuation date, ``elapsed_months[mp]`` months after that
contract's inception. The projection runs from inception and the
trajectory is sliced at ``t = elapsed_months[mp]``.

The headline equivalence test: an in-force MP with ``elapsed_months = E``
and term ``T`` must produce the same in-force BEL as the matching
new-business MP measured ``E`` months into its life. This is the BEL
trajectory-slice property, and it pins the in-force semantics.
"""
import numpy as np
import pytest

from fastcashflow import Basis, ExpenseItem, ModelPoints, CoverageRate
# _measure_inforce_fast / _measure_inforce_full are the engine-internal workhorses behind
# the public fcf.gmm.measure_inforce; tested directly here.
from fastcashflow.engine import _measure_inforce_full, _measure_inforce_fast
from fastcashflow.gmm import measure


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
        morbidity_cv=0.10,
        expense_inflation=0.02,
        expense_items=(
            ExpenseItem("acquisition",  "alpha_fixed",    100_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  30_000.0),
        ),
        coverages=(CoverageRate("DEATH", _flat_rate(0.005)),),
    )


def test_inforce_fast_zero_elapsed_matches_value():
    """When every ``elapsed_months`` is 0 the in-force valuation collapses
    to the new-business :func:`measure` (= ``GMMMeasurement.bel_path[:, 0]``)."""
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        premium=50_000.0, term_months=120,
    )
    basis = _basis()
    v_new = measure(mp, basis, full=False)
    v_inf = _measure_inforce_fast(mp, basis)
    assert np.isclose(v_inf.bel[0], v_new.bel[0])
    assert np.isclose(v_inf.ra[0], v_new.ra[0])
    assert np.isclose(v_inf.csm[0], v_new.csm[0])


def test_inforce_fast_matches_trajectory_slice():
    """An in-force MP with ``elapsed_months = E`` returns the trajectory
    slice ``GMMMeasurement.bel_path[mp, E]`` -- the PV of future cash flows from
    the valuation date forward."""
    elapsed = 36
    mp_new = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        premium=50_000.0, term_months=120,
    )
    basis = _basis()
    m = measure(mp_new, basis)

    mp_inforce = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([120]),
        benefits={0: np.array([100_000_000.0])},
        elapsed_months=np.array([elapsed]),
    )
    v_inf = _measure_inforce_fast(mp_inforce, basis)
    # The in-force BEL is the trajectory slice at t = elapsed.
    assert np.isclose(v_inf.bel[0], m.bel_path[0, elapsed])
    assert np.isclose(v_inf.ra[0], m.ra_path[0, elapsed])


def test_inforce_fast_settlement_matches_trajectory():
    """Settlement-mode carry-forward: with ``prior_csm`` taken from the
    measure() CSM trajectory at ``E - period_months`` and a ``lock_in_rate``
    equal to the current discount, rolling one period forward must
    reproduce the same trajectory's CSM at ``E``. This pins the §44
    accretion + coverage-unit release path."""
    basis = _basis()
    mp_new = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        premium=50_000.0, term_months=240,
    )
    m = measure(mp_new, basis)
    elapsed, period = 36, 12
    prior_t = elapsed - period
    prior_csm = m.csm_path[:, prior_t]

    mp_inforce = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([240]),
        benefits={0: np.array([100_000_000.0])},
        elapsed_months=np.array([elapsed]),
    )
    v = _measure_inforce_fast(
        mp_inforce, basis,
        prior_csm=prior_csm,
        lock_in_rate=basis.discount_annual,
        period_months=period,
    )
    assert np.isclose(v.bel[0], m.bel_path[0, elapsed])
    assert np.isclose(v.ra[0], m.ra_path[0, elapsed])
    assert np.isclose(v.csm[0], m.csm_path[0, elapsed])


def test_inforce_fast_period_months_rejected_in_hypothetical_mode():
    """period_months only applies in settlement mode; passing it without
    prior_csm / lock_in_rate is a no-op trap and now raises."""
    basis = _basis()
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        premium=50_000.0, term_months=120,
    )
    with pytest.raises(ValueError, match="period_months applies only in"):
        _measure_inforce_fast(mp, basis, period_months=12)


def test_inforce_fast_settlement_paired_args():
    """``prior_csm`` and ``lock_in_rate`` must be supplied together; one
    without the other is a silent-wrong-result trap and raises."""
    basis = _basis()
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        premium=50_000.0, term_months=240,
    )
    with pytest.raises(ValueError, match="both be given.*both omitted"):
        _measure_inforce_fast(mp, basis, prior_csm=np.array([0.0]))
    with pytest.raises(ValueError, match="both be given.*both omitted"):
        _measure_inforce_fast(mp, basis, lock_in_rate=0.03)


def test_inforce_fast_settlement_elapsed_too_small():
    """``elapsed_months < period_months`` means the prior closing date
    precedes inception -- no CSM to carry forward, so the call errors out
    rather than silently using a zero or out-of-range slice."""
    basis = _basis()
    mp = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([240]),
        benefits={0: np.array([100_000_000.0])},
        elapsed_months=np.array([6]),
    )
    with pytest.raises(ValueError, match="precedes inception"):
        _measure_inforce_fast(
            mp, basis,
            prior_csm=np.array([1.0]),
            lock_in_rate=0.03,
            period_months=12,
        )


def test_inforce_full_hypothetical_is_measure():
    """``_measure_inforce_full`` without prior_csm returns the measure() result
    unchanged -- it is the trajectory-variant of the hypothetical mode."""
    basis = _basis()
    mp = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([240]),
        benefits={0: np.array([100_000_000.0])},
        elapsed_months=np.array([36]),
    )
    m = measure(mp, basis)
    mif = _measure_inforce_full(mp, basis)
    assert np.allclose(m.csm, mif.csm)
    assert np.allclose(m.csm_accretion, mif.csm_accretion)
    assert np.allclose(m.csm_release, mif.csm_release)


def test_inforce_full_settlement_matches__measure_inforce_fast():
    """Settlement-mode ``_measure_inforce_full`` at the valuation date equals
    the _measure_inforce_fast settlement-mode CSM scalar."""
    basis = _basis()
    mp = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([240]),
        benefits={0: np.array([100_000_000.0])},
        elapsed_months=np.array([36]),
    )
    m_baseline = measure(mp, basis)
    period = 12
    prior_csm = m_baseline.csm_path[:, 36 - period]
    lock_in = basis.discount_annual
    v = _measure_inforce_fast(mp, basis, prior_csm=prior_csm,
                       lock_in_rate=lock_in, period_months=period)
    mif = _measure_inforce_full(mp, basis, prior_csm=prior_csm,
                            lock_in_rate=lock_in, period_months=period)
    rows = np.arange(1)
    assert np.isclose(mif.csm_path[rows, 36][0], v.csm[0])
    assert np.isclose(mif.loss_component[0], v.loss_component[0])


def test_inforce_full_settlement_roundtrip_to_measure():
    """When prior_csm and lock_in_rate are seeded from the engine's own
    trajectory and discount, the carried-forward CSM trajectory from
    t=prior_t onwards matches the measure() trajectory bit for bit."""
    basis = _basis()
    mp = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([240]),
        benefits={0: np.array([100_000_000.0])},
        elapsed_months=np.array([36]),
    )
    m = measure(mp, basis)
    period = 12
    prior_t = 36 - period
    mif = _measure_inforce_full(
        mp, basis,
        prior_csm=m.csm_path[:, prior_t],
        lock_in_rate=basis.discount_annual,
        period_months=period,
    )
    assert np.allclose(m.csm_path[:, prior_t:], mif.csm_path[:, prior_t:])
    assert np.allclose(m.csm_accretion[:, prior_t:], mif.csm_accretion[:, prior_t:])
    assert np.allclose(m.csm_release[:, prior_t:], mif.csm_release[:, prior_t:])


def test_inforce_bel_smaller_term_left():
    """As ``elapsed_months`` grows (less of the term left), the absolute
    value of the in-force BEL shrinks -- there are fewer future cash flows
    to discount."""
    basis = _basis()
    def inforce_bel(e):
        mp = ModelPoints(
            issue_age=np.array([40]),
            premium=np.array([50_000.0]),
            term_months=np.array([240]),
            benefits={0: np.array([100_000_000.0])},
            elapsed_months=np.array([e]),
        )
        return abs(_measure_inforce_fast(mp, basis).bel[0])
    # Strictly decreasing in elapsed -- the future shortens.
    bels = [inforce_bel(e) for e in (0, 60, 120, 180)]
    assert bels[0] > bels[1] > bels[2] > bels[3]


def test_gmm_measure_inforce_headline_csm_is_as_of_and_tracks_prior():
    """Public fcf.gmm.measure_inforce headline .csm must be the as-of
    valuation-date CSM -- sensitive to state.prior_csm, equal to
    csm_path[:, elapsed_months], and identical for full=False / full=True.

    Regression for the bug where the full=True headline echoed the inception
    column (csm_new[:, 0]), ignoring prior_csm and contradicting full=False.
    """
    import fastcashflow as fcf
    from dataclasses import replace

    mp = fcf.apply_inforce_state(
        fcf.samples.model_points(), fcf.samples.inforce_state()
    )
    basis = fcf.samples.basis()[("TERM_LIFE_A", "GA")]
    rows = np.arange(mp.n_mp)
    em = np.asarray(mp.elapsed_months, dtype=np.int64)

    def run(prior, full):
        state = replace(
            fcf.samples.inforce_state(),
            prior_csm=np.full(mp.n_mp, float(prior)),
            lock_in_rate=0.03,
        )
        return fcf.gmm.measure_inforce(mp, basis, state, period_months=12, full=full)

    head0 = run(0.0, full=False)
    head5k = run(5_000.0, full=False)
    # Headline CSM responds to prior_csm (the bug echoed inception, ignoring it).
    assert not np.allclose(head0.csm, head5k.csm)

    full5k = run(5_000.0, full=True)
    # full=False headline == full=True headline == csm_path at the as-of month.
    assert np.allclose(head5k.csm, full5k.csm)
    assert np.allclose(full5k.csm, full5k.csm_path[rows, em])


def test_inforce_state_subset_is_consistent_and_drives_segment_measure():
    """InforceState.subset slices every per-MP field together (not a ragged
    prior_csm-only replace), so per-segment measure_inforce -- the demo /
    cookbook close pattern -- sees a self-consistent state."""
    import fastcashflow as fcf
    portfolio = fcf.samples.model_points()
    state = fcf.samples.inforce_state()
    basis = fcf.samples.basis()
    mp = fcf.apply_inforce_state(portfolio, state)

    key = ("HEALTH_A", "FC")
    idx = np.where((np.asarray(mp.product_code) == key[0]) &
                   (np.asarray(mp.channel_code) == key[1]))[0]
    sub_state = state.subset(idx)
    assert sub_state.prior_csm.shape[0] == len(idx)
    assert sub_state.elapsed_months.shape[0] == len(idx)
    assert sub_state.count.shape[0] == len(idx)
    assert np.asarray(sub_state.mp_id).shape[0] == len(idx)
    assert sub_state.lock_in_rate == state.lock_in_rate
    assert np.allclose(sub_state.prior_csm, np.asarray(state.prior_csm)[idx])

    val = fcf.gmm.measure_inforce(mp.subset(idx), basis[key], sub_state,
                                  period_months=3)
    assert np.all(np.isfinite(val.bel)) and np.all(np.isfinite(val.csm))
