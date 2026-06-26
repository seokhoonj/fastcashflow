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

from fastcashflow import Basis, CalculationMethod, ExpenseItem, ModelPoints, CoverageRate
# _measure_inforce_fast / _measure_inforce_full are the engine-internal workhorses behind
# the public fcf.gmm.measure_inforce; tested directly here.
from fastcashflow._measurement.gmm import _measure_inforce_full, _measure_inforce_fast
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
            ExpenseItem("acquisition", "per_policy",    100_000.0),
            ExpenseItem("maintenance", "per_policy",  30_000.0),
        ),
        coverages=(CoverageRate("DEATH", _flat_rate(0.005)),),
    )


def test_inforce_rescale_matches_fresh_valuation_date_start():
    """Re-basing is exact: under flat rates with no inception-only flows, an
    in-force contract valued at elapsed E equals a fresh contract started at
    the valuation date (attained age, remaining term). The count / inforce[E]
    rescale removes the inception double-decrement exactly (the original
    motivation -- without it the in-force BEL understates by survival(0->E))."""
    basis = Basis(
        mortality_annual=_flat_rate(0.012), lapse_annual=_flat_rate(0.05),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", _flat_rate(0.012)),),
    )
    CM = {"DEATH": CalculationMethod.DEATH}
    inforce = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([24]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1000.0]), elapsed_months=np.array([12]),
        calculation_methods=CM,
    )
    bel_inforce = _measure_inforce_fast(inforce, basis).bel[0]
    # fresh contract from the valuation date: attained age 41, remaining 12m
    fresh = ModelPoints(
        issue_age=np.array([41]), premium=np.array([100.0]),
        term_months=np.array([12]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1000.0]), calculation_methods=CM,
    )
    bel_fresh = measure(fresh, basis).bel_path[0, 0]
    assert np.isclose(bel_inforce, bel_fresh, rtol=1e-9)


def test_inforce_fast_zero_elapsed_matches_value():
    """When every ``elapsed_months`` is 0 the in-force valuation collapses
    to the new-business :func:`measure` (= ``Measurement.bel_path[:, 0]``)."""
    mp = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=50_000.0, term_months=120,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    basis = _basis()
    v_new = measure(mp, basis, full=False)
    v_inf = _measure_inforce_fast(mp, basis)
    assert np.isclose(v_inf.bel[0], v_new.bel[0])
    assert np.isclose(v_inf.ra[0], v_new.ra[0])
    assert np.isclose(v_inf.csm[0], v_new.csm[0])


def test_inforce_fast_matches_trajectory_slice():
    """An in-force MP with ``elapsed_months = E`` returns the trajectory slice
    ``bel_path[mp, E]`` re-based to the valuation date: the projection runs from
    inception, so the slice is scaled by ``count / inforce[E]`` (here count = 1,
    so ``1 / survival(0->E)``) to set the as-of in-force to the input count."""
    elapsed = 36
    mp_new = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=50_000.0, term_months=120,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    basis = _basis()
    m = measure(mp_new, basis)

    mp_inforce = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([120]),
        benefits={"DEATH": np.array([100_000_000.0])},
        elapsed_months=np.array([elapsed]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    v_inf = _measure_inforce_fast(mp_inforce, basis)
    # The in-force BEL is the trajectory slice at t = elapsed, re-based so the
    # as-of in-force is the input count (here 1): x count / inforce[elapsed].
    rescale = 1.0 / m.cashflows.inforce[0, elapsed]
    assert np.isclose(v_inf.bel[0], m.bel_path[0, elapsed] * rescale)
    assert np.isclose(v_inf.ra[0], m.ra_path[0, elapsed] * rescale)


def test_inforce_fast_settlement_matches_trajectory():
    """Settlement-mode carry-forward: with ``prior_csm`` taken from the
    measure() CSM trajectory at ``E - period_months`` and a ``lock_in_rate``
    equal to the current discount, rolling one period forward must
    reproduce the same trajectory's CSM at ``E``. This pins the paragraph 44
    accretion + coverage-unit release path."""
    basis = _basis()
    mp_new = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=50_000.0, term_months=240,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    m = measure(mp_new, basis)
    elapsed, period = 36, 12
    prior_t = elapsed - period
    prior_csm = m.csm_path[:, prior_t]

    mp_inforce = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([240]),
        benefits={"DEATH": np.array([100_000_000.0])},
        elapsed_months=np.array([elapsed]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    v = _measure_inforce_fast(
        mp_inforce, basis,
        prior_csm=prior_csm,
        lock_in_rate=basis.discount_annual,
        period_months=period,
    )
    rescale = 1.0 / m.cashflows.inforce[0, elapsed]   # count = 1
    assert np.isclose(v.bel[0], m.bel_path[0, elapsed] * rescale)
    assert np.isclose(v.ra[0], m.ra_path[0, elapsed] * rescale)
    assert np.isclose(v.csm[0], m.csm_path[0, elapsed])   # CSM is scale-invariant


def test_inforce_fast_period_months_rejected_in_hypothetical_mode():
    """period_months only applies in settlement mode; passing it without
    prior_csm / lock_in_rate is a no-op trap and now raises."""
    basis = _basis()
    mp = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=50_000.0, term_months=120,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    with pytest.raises(ValueError, match="period_months applies only in"):
        _measure_inforce_fast(mp, basis, period_months=12)


def test_inforce_fast_settlement_paired_args():
    """``prior_csm`` and ``lock_in_rate`` must be supplied together; one
    without the other is a silent-wrong-result trap and raises."""
    basis = _basis()
    mp = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=50_000.0, term_months=240,
        calculation_methods={"DEATH": CalculationMethod.DEATH},
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
        benefits={"DEATH": np.array([100_000_000.0])},
        elapsed_months=np.array([6]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    with pytest.raises(ValueError, match="precedes inception"):
        _measure_inforce_fast(
            mp, basis,
            prior_csm=np.array([1.0]),
            lock_in_rate=0.03,
            period_months=12,
        )


def test_inforce_fast_at_the_contract_boundary_is_rejected():
    """Valuing exactly at the contract boundary (== term with no cut) leaves no
    remaining coverage; the call raises a clear ValueError rather than indexing
    the in-force out of bounds (em == boundary == n_time)."""
    basis = _basis()
    mp = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([240]),
        benefits={"DEATH": np.array([100_000_000.0])},
        elapsed_months=np.array([240]),        # == boundary -> no remaining coverage
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    with pytest.raises(ValueError, match="no remaining coverage"):
        _measure_inforce_fast(mp, basis)


def test_inforce_full_hypothetical_is_measure():
    """``_measure_inforce_full`` without prior_csm returns the measure() result
    unchanged -- it is the trajectory-variant of the hypothetical mode."""
    basis = _basis()
    mp = ModelPoints(
        issue_age=np.array([40]),
        premium=np.array([50_000.0]),
        term_months=np.array([240]),
        benefits={"DEATH": np.array([100_000_000.0])},
        elapsed_months=np.array([36]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
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
        benefits={"DEATH": np.array([100_000_000.0])},
        elapsed_months=np.array([36]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
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
        benefits={"DEATH": np.array([100_000_000.0])},
        elapsed_months=np.array([36]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
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


def _two_mp_book():
    """A two-contract book seated 36 months in, identical save the row index."""
    return ModelPoints(
        issue_age=np.array([40, 40]),
        premium=np.array([50_000.0, 50_000.0]),
        term_months=np.array([240, 240]),
        benefits={"DEATH": np.array([100_000_000.0, 100_000_000.0])},
        elapsed_months=np.array([36, 36]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )


@pytest.mark.parametrize("full", [False, True])
def test_inforce_cohort_aware_lock_in_matches_per_mp(full):
    """A mixed locked-in-rate in-force book (Sec. B72(b)) carries each model
    point's CSM forward at its OWN rate -- the per-MP kernel path. Oracle: every
    row equals that row measured alone under its scalar rate, and the two
    cohorts' as-of CSM genuinely differ because the locked rate differs."""
    basis = _basis()
    mp = _two_mp_book()
    period = 12
    prior_csm = measure(mp, basis).csm_path[:, 36 - period]
    fn = _measure_inforce_full if full else _measure_inforce_fast
    mixed = fn(mp, basis, prior_csm=prior_csm,
               lock_in_rate=np.array([0.03, 0.05]), period_months=period)
    a = fn(mp.subset([0]), basis, prior_csm=prior_csm[[0]],
           lock_in_rate=0.03, period_months=period)
    b = fn(mp.subset([1]), basis, prior_csm=prior_csm[[1]],
           lock_in_rate=0.05, period_months=period)
    assert np.isclose(mixed.csm[0], a.csm[0])
    assert np.isclose(mixed.csm[1], b.csm[0])
    # the locked-in rate genuinely separates the two cohorts' as-of CSM
    assert not np.isclose(mixed.csm[0], mixed.csm[1])


@pytest.mark.parametrize("full", [False, True])
def test_inforce_uniform_lock_in_array_equals_scalar(full):
    """A uniform lock_in_rate carried as a per-row array gives the scalar result
    (collapsed to the shared-rate kernel, not the per-MP one)."""
    basis = _basis()
    mp = _two_mp_book()
    period = 12
    prior_csm = measure(mp, basis).csm_path[:, 36 - period]
    fn = _measure_inforce_full if full else _measure_inforce_fast
    arr = fn(mp, basis, prior_csm=prior_csm,
             lock_in_rate=np.array([0.03, 0.03]), period_months=period)
    sca = fn(mp, basis, prior_csm=prior_csm, lock_in_rate=0.03,
             period_months=period)
    assert np.allclose(arr.csm, sca.csm)


def test_inforce_bel_smaller_term_left():
    """As ``elapsed_months`` grows (less of the term left), the as-of in-force
    BEL of a claims-only contract shrinks -- fewer future claims to discount.

    (A claims-only contract is used on purpose: the *net* BEL with premiums
    need not be monotonic in elapsed -- that monotonicity in earlier versions
    was an artifact of the inception-decremented projection, now corrected so
    the as-of in-force equals the input count.)"""
    basis = _basis()
    def inforce_bel(e):
        mp = ModelPoints(
            issue_age=np.array([40]),
            premium=np.array([0.0]),               # claims-only
            term_months=np.array([240]),
            benefits={"DEATH": np.array([100_000_000.0])},
            elapsed_months=np.array([e]),
            calculation_methods={"DEATH": CalculationMethod.DEATH},
        )
        return abs(_measure_inforce_fast(mp, basis).bel[0])
    # Strictly decreasing in elapsed -- the remaining claims shorten.
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
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "GA"))
    rows = np.arange(mp.n_mp)
    em = np.asarray(mp.elapsed_months, dtype=np.int64)

    def run(prior, full):
        state = replace(
            fcf.samples.inforce_state(),
            prior_csm=np.full(mp.n_mp, float(prior)),
            lock_in_rate=0.03,
        )
        return fcf.gmm.measure_inforce(mp, state, basis, period_months=12, full=full)

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
    idx = np.where((np.asarray(mp.product) == key[0]) &
                   (np.asarray(mp.channel) == key[1]))[0]
    sub_state = state.subset(idx)
    assert sub_state.prior_csm.shape[0] == len(idx)
    assert sub_state.elapsed_months.shape[0] == len(idx)
    assert sub_state.count.shape[0] == len(idx)
    assert np.asarray(sub_state.mp_id).shape[0] == len(idx)
    assert sub_state.lock_in_rate == state.lock_in_rate
    assert np.allclose(sub_state.prior_csm, np.asarray(state.prior_csm)[idx])

    val = fcf.gmm.measure_inforce(mp.subset(idx), sub_state, basis.resolve(key),
                                  period_months=3)
    assert np.all(np.isfinite(val.bel)) and np.all(np.isfinite(val.csm))


def test_measure_inforce_requires_reconciled_state():
    """measure_inforce reads each contract's as-of duration / size off
    model_points; a model_points not reconciled with the state (via
    apply_inforce_state) is rejected, so a stale snapshot cannot borrow the
    fresh state's prior_csm. A reconciled pair passes unchanged."""
    import fastcashflow as fcf
    from dataclasses import replace
    portfolio = fcf.samples.model_points()
    state = fcf.samples.inforce_state()
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "GA"))

    # reconciled pair (what read_inforce_policies returns) -> passes
    mp = fcf.apply_inforce_state(portfolio, state)
    assert np.all(np.isfinite(
        fcf.gmm.measure_inforce(mp, state, basis, full=False).bel))

    # a state whose elapsed disagrees with the reconciled model_points -> reject
    stale = replace(state,
                    elapsed_months=np.asarray(state.elapsed_months) + 1)
    with pytest.raises(ValueError, match="do not match"):
        fcf.gmm.measure_inforce(mp, stale, basis, full=False)

    # an un-reconciled new-business model_points (elapsed backfilled to 0)
    # paired with a real period-close state is likewise rejected
    with pytest.raises(ValueError, match="do not match"):
        fcf.gmm.measure_inforce(portfolio, state, basis, full=False)


def test_measure_inforce_prior_csm_is_order_independent():
    """A state whose mp_id order differs from the policies must give the same
    per-contract CSM as the matched-order state -- prior_csm is re-aligned by
    mp_id, not read positionally (else one contract carries another's CSM)."""
    import fastcashflow as fcf
    mort = lambda s, a, d: np.full(np.shape(a), 0.01)
    zero = lambda s, a, d: np.zeros(np.shape(d))
    basis = Basis(mortality_annual=mort, lapse_annual=zero, discount_annual=0.0,
                  ra_confidence=0.75, mortality_cv=0.10,
                  coverages=(CoverageRate("DEATH", mort),))
    mp0 = ModelPoints(
        mp_id=np.array(["A", "B"]),
        issue_age=np.array([40, 40], dtype=np.int64),
        benefits={"DEATH": np.array([100_000.0, 100_000.0])},
        premium=np.array([100.0, 100.0]),
        term_months=np.array([24, 24], dtype=np.int64),
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})
    # A: elapsed 12 / prior_csm 0 ; B: elapsed 6 / prior_csm 5000
    shuffled = fcf.InforceState(                # rows in [B, A] order
        mp_id=np.array(["B", "A"]),
        elapsed_months=np.array([6, 12], dtype=np.int64),
        count=np.array([1.0, 1.0]), prior_csm=np.array([5_000.0, 0.0]),
        lock_in_rate=0.03)
    matched = fcf.InforceState(                 # the same data in [A, B] order
        mp_id=np.array(["A", "B"]),
        elapsed_months=np.array([12, 6], dtype=np.int64),
        count=np.array([1.0, 1.0]), prior_csm=np.array([0.0, 5_000.0]),
        lock_in_rate=0.03)
    v_shuf = fcf.gmm.measure_inforce(
        fcf.apply_inforce_state(mp0, shuffled), shuffled, basis,
        period_months=3, full=False)
    v_match = fcf.gmm.measure_inforce(
        fcf.apply_inforce_state(mp0, matched), matched, basis,
        period_months=3, full=False)
    assert np.allclose(v_shuf.csm, v_match.csm)
    # and the CSM sits on B (the contract that carried prior_csm), not A
    assert v_match.csm[0] == pytest.approx(0.0)
    assert v_match.csm[1] > 0.0


def test_paa_measure_inforce_lrc_is_rebased_slice():
    """PAA in-force: the LRC is the inception LRC trajectory sliced at each
    contract's elapsed_months and re-based by count / inforce[elapsed]. There is
    no CSM, so loss_component is zero and fcf is None (subsequent onerous re-test
    deferred); full=False gives the same headline as full=True."""
    import fastcashflow as fcf

    mp = fcf.apply_inforce_state(
        fcf.samples.model_points(), fcf.samples.inforce_state())
    state = fcf.samples.inforce_state()
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "GA"))

    m = fcf.paa.measure(mp, basis, full=True)
    rows = np.arange(mp.n_mp)
    em = np.asarray(mp.elapsed_months, dtype=np.int64)
    inforce_em = m.cashflows.inforce[rows, em]
    count = np.asarray(mp.count, dtype=np.float64)
    rescale = np.where(inforce_em > 0.0,
                       count / np.where(inforce_em > 0.0, inforce_em, 1.0), 1.0)
    expected = m.lrc_path[rows, em] * rescale

    inf = fcf.paa.measure_inforce(mp, state, basis)
    assert np.allclose(inf.lrc, expected)
    assert np.allclose(inf.loss_component, 0.0)
    assert inf.fcf is None
    # full=False headline matches full=True
    assert np.allclose(
        fcf.paa.measure_inforce(mp, state, basis, full=False).lrc, inf.lrc)


def test_paa_measure_inforce_at_the_contract_boundary_is_rejected():
    """PAA in-force at the contract boundary has no remaining coverage -- a clear
    ValueError, not an out-of-bounds index (the same guard as the GMM path)."""
    import fastcashflow as fcf
    from dataclasses import replace

    base_mp = fcf.samples.model_points()
    boundary = np.asarray(base_mp.contract_boundary_months, dtype=np.int64)
    state = replace(fcf.samples.inforce_state(), elapsed_months=boundary.copy())
    mp = fcf.apply_inforce_state(base_mp, state)
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "GA"))
    with pytest.raises(ValueError, match="no remaining coverage"):
        fcf.paa.measure_inforce(mp, state, basis)


def test_paa_measure_inforce_ignores_prior_csm_and_rejects_stale_state():
    """PAA has no CSM, so prior_csm / lock_in_rate on the state are ignored; and
    a model_points not reconciled with the state (apply_inforce_state) is
    rejected, the same guard as the GMM path."""
    import fastcashflow as fcf
    from dataclasses import replace

    mp = fcf.apply_inforce_state(
        fcf.samples.model_points(), fcf.samples.inforce_state())
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "GA"))
    base = fcf.paa.measure_inforce(mp, fcf.samples.inforce_state(), basis)
    # a wildly different prior_csm changes nothing (PAA has no CSM)
    weird = replace(fcf.samples.inforce_state(),
                    prior_csm=np.full(mp.n_mp, 1e9), lock_in_rate=0.99)
    assert np.allclose(
        fcf.paa.measure_inforce(mp, weird, basis).lrc, base.lrc)
    # an unreconciled (raw) model_points is rejected
    with pytest.raises(ValueError, match="elapsed_months / count"):
        fcf.paa.measure_inforce(
            fcf.samples.model_points(), fcf.samples.inforce_state(), basis)


def test_inforce_state_account_value_optional_carried_and_validated():
    """InforceState.account_value is optional (None for GMM/PAA, backward
    compatible); when given it is carried by subset, reordered by
    align_inforce_state (by mp_id), and validated (length / finite / >= 0)."""
    import fastcashflow as fcf

    # default None -- existing states keep working
    s0 = fcf.InforceState(
        mp_id=np.array(["A", "B"]), elapsed_months=np.array([12, 12]),
        count=np.array([1.0, 1.0]), prior_csm=np.zeros(2), lock_in_rate=0.03)
    assert s0.account_value is None

    s = fcf.InforceState(
        mp_id=np.array(["A", "B"]), elapsed_months=np.array([12, 24]),
        count=np.array([1.0, 2.0]), prior_csm=np.zeros(2), lock_in_rate=0.03,
        account_value=np.array([1000.0, 2000.0]))
    # subset carries account_value
    assert np.allclose(s.subset([1]).account_value, [2000.0])
    # align reorders account_value to model-points (mp_id) order [B, A]
    mp = fcf.ModelPoints(
        issue_age=np.array([40, 40]), premium=np.zeros(2),
        term_months=np.full(2, 60), benefits={"DEATH": np.full(2, 1e4)},
        mp_id=np.array(["B", "A"]))
    aligned = fcf.align_inforce_state(mp, s)
    assert np.allclose(aligned.account_value, [2000.0, 1000.0])

    with pytest.raises(ValueError, match="account_value has length|per-MP arrays"):
        fcf.InforceState(
            mp_id=np.array(["A", "B"]), elapsed_months=np.array([12, 12]),
            count=np.array([1.0, 1.0]), prior_csm=np.zeros(2), lock_in_rate=0.03,
            account_value=np.array([1000.0]))
    with pytest.raises(ValueError, match="account_value must be >= 0"):
        fcf.InforceState(
            mp_id=np.array(["A", "B"]), elapsed_months=np.array([12, 12]),
            count=np.array([1.0, 1.0]), prior_csm=np.zeros(2), lock_in_rate=0.03,
            account_value=np.array([1000.0, -5.0]))
