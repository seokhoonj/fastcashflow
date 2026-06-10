"""Full-trajectory measurement of a multi-segment (per-basis-dict) portfolio.

``measure(model_points, basis_dict, full=True)`` runs each (product,
channel) segment under its own basis and stitches the per-segment
trajectories into one ``(n_mp, n_time+1)`` result. The correctness anchor is
equivalence: a model point's stitched trajectory must equal the trajectory it
gets when its segment is measured alone, zero-padded to the portfolio horizon.
discount_bom / discount_mid become per-MP (2-D) because segments discount on
different curves; the operations that do not yet handle that shape must refuse
loudly rather than compute a wrong number.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.basis import BasisRouter
from conftest import PATTERNS, make_death_basis


def _segments(mp, basis):
    pc = np.array(mp.product)
    ch = np.array(mp.channel)
    for key in basis.segments:
        idx = np.nonzero((pc == key[0]) & (ch == key[1]))[0]
        if idx.size:
            yield key, idx


def test_segmented_full_equals_per_segment():
    """Each MP's stitched trajectory equals its segment measured alone, padded."""
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis()
    m = fcf.gmm.measure(mp, basis)                      # full=True + dict

    n_mp = mp.n_mp
    assert m.bel_path.shape[0] == n_mp
    assert m.discount_bom.ndim == 2                     # per-MP curves
    assert m.discount_bom.shape == m.bel_path.shape
    assert m.discount_mid.shape == (n_mp, m.bel_path.shape[1] - 1)
    for arr in (m.bel_path, m.ra_path, m.csm_path, m.discount_bom):
        assert not np.isnan(arr).any() and not np.isinf(arr).any()

    seen = 0
    for key, idx in _segments(mp, basis):
        ref = fcf.gmm.measure(mp.subset(idx), basis.resolve(key))   # single basis -> full
        t = ref.bel_path.shape[1] - 1
        for field in ("bel_path", "ra_path", "csm_path"):
            block = getattr(m, field)[idx, :t + 1]
            np.testing.assert_allclose(block, getattr(ref, field), rtol=0, atol=1e-9)
            tail = getattr(m, field)[idx, t + 1:]
            assert np.allclose(tail, 0.0)               # matured -> zero
        for field in ("bel", "ra", "csm", "loss_component"):
            np.testing.assert_allclose(getattr(m, field)[idx], getattr(ref, field),
                                       rtol=0, atol=1e-9)
        seen += idx.size
    assert seen == n_mp                                 # every row covered


def test_segmented_full_rollforward_matches_aggregated_segments():
    """A combined roll-forward reconciles to the sum of per-segment ones."""
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis()

    combined = fcf.reconcile(fcf.roll_forward(fcf.gmm.measure(mp, basis),
                                              period_months=12))

    agg = {k: 0.0 for k in ("csm_opening", "csm_finance", "csm_release",
                            "csm_closing", "bel_opening", "ra_opening")}
    for key, idx in _segments(mp, basis):
        r0 = fcf.reconcile(fcf.roll_forward(
            fcf.gmm.measure(mp.subset(idx), basis.resolve(key)), period_months=12))[0]
        for k in agg:
            agg[k] += float(np.asarray(getattr(r0, k)).sum())

    c0 = combined[0]
    for k in agg:
        assert np.isclose(float(np.asarray(getattr(c0, k)).sum()), agg[k], atol=1e-6)
    # the CSM block reconciles: opening + finance + release == closing
    assert np.isclose(agg["csm_opening"] + agg["csm_finance"] + agg["csm_release"],
                      agg["csm_closing"], atol=1e-6)


def test_single_segment_basis_stays_1d():
    """A single Basis (not a dict) keeps the 1-D discount curve."""
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "FC"))
    m = fcf.gmm.measure(mp.subset([0]), basis)
    assert m.discount_bom.ndim == 1
    assert m.discount_mid.ndim == 1


def test_segmented_full_ops_match_per_segment():
    """transition / report / group / revised roll-forward on a segmented (2-D
    discount) result match doing each on the single-basis segment alone."""
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis()
    m = fcf.gmm.measure(mp, basis)
    fv = (m.bel_path[:, 0] + m.ra_path[:, 0]) * 0.5 + 100_000.0

    ct = fcf.transition(m, fv)
    cr = fcf.report(m)
    seg_id = np.empty(mp.n_mp, dtype=int)
    segs = list(_segments(mp, basis))
    for g, (key, idx) in enumerate(segs):
        seg_id[idx] = g
    gm = fcf.group(m, seg_id)

    for g, (key, idx) in enumerate(segs):
        sm = fcf.gmm.measure(mp.subset(idx), basis.resolve(key))
        t = sm.bel_path.shape[1] - 1
        # transition: per-MP CSM matches the segment measured alone
        st = fcf.transition(sm, fv[idx])
        np.testing.assert_allclose(ct.csm[idx], st.csm, rtol=0, atol=1e-6)
        np.testing.assert_allclose(ct.csm_path[idx, :t + 1], st.csm_path, rtol=0, atol=1e-6)
        # report: per-MP rows match
        sr = fcf.report(sm)
        np.testing.assert_allclose(cr.csm_accretion[idx, :t], sr.csm_accretion[:, :t],
                                   rtol=0, atol=1e-6)
        # group: group g (= this segment) matches the segment aggregated alone
        sg = fcf.group(sm, np.zeros(len(idx), dtype=int))
        assert np.isclose(gm.csm[g], sg.csm[0]) and np.isclose(gm.bel[g], sg.bel[0])

    # a revised (assumption-change) roll-forward now runs on the segmented result
    rf = fcf.reconcile(fcf.roll_forward(m, period_months=12, revised=m, revised_at=12))
    assert len(rf) > 0


def _two_seg_mp(terms):
    """Two model points in two channels of one product, given terms."""
    return fcf.ModelPoints(
        issue_age=np.array([40, 40]), benefits={0: np.array([1e8, 1e8])},
        premium=np.array([200_000.0, 200_000.0]),
        term_months=np.array(terms), calculation_methods=PATTERNS,
        product=np.array(["A", "A"]), channel=np.array(["X", "Y"]),
    )


def _disc(rate):
    return make_death_basis(mortality_q=0.002, lapse_q=0.01,
                                  discount_annual=rate, ra_confidence=0.75,
                                  mortality_cv=0.10)


def test_segmented_full_group_rejects_mixed_curves():
    """group() refuses a group spanning genuinely different discount curves --
    a group must sit in one portfolio (basis)."""
    m = fcf.gmm.measure(_two_seg_mp([120, 120]),
                        BasisRouter({("A", "X"): _disc(0.03), ("A", "Y"): _disc(0.05)}), full=True)
    with pytest.raises(ValueError, match="different discount curves"):
        fcf.group(m, np.zeros(2, dtype=int))            # 3% and 5% in one group


def test_segmented_full_group_allows_same_curve_different_terms():
    """Same curve, different terms is fine: the padded tail past each contract's
    maturity discounts zero in-force and never reaches the CSM, so it must not
    be mistaken for a different discount curve."""
    m = fcf.gmm.measure(_two_seg_mp([120, 240]),
                        BasisRouter({("A", "X"): _disc(0.03), ("A", "Y"): _disc(0.03)}), full=True)
    g = fcf.group(m, np.zeros(2, dtype=int))            # same 3%, terms 120 & 240
    assert g.bel.shape[0] == 1
    assert np.isclose(g.bel.sum(), m.bel.sum())         # BEL additive, totals match


def test_measure_aggregate_matches_full_summed_and_is_chunk_independent():
    """measure_aggregate returns the per-model-point full=True trajectories
    summed over the model-point axis, in bounded memory; the result must not
    depend on chunk_size and must reproduce the full measure's headline totals."""
    import fastcashflow as fcf
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis()
    full = fcf.gmm.measure(mp, basis, full=True)

    n_time = int(np.asarray(mp.contract_boundary_months).max()) + 1
    ref = np.zeros(n_time)
    ref[:full.bel_path.shape[1]] = full.bel_path.sum(axis=0)

    multi = fcf.gmm.measure_aggregate(mp, basis, chunk_size=3)   # several chunks
    one = fcf.gmm.measure_aggregate(mp, basis, chunk_size=10_000)  # single chunk
    assert np.allclose(multi.bel_path, ref)
    assert np.allclose(multi.bel_path, one.bel_path)
    assert np.allclose(multi.csm_path, one.csm_path)
    # headline totals reproduce the full measure's per-MP sums
    assert np.isclose(multi.bel, float(full.bel.sum()))
    assert np.isclose(multi.csm, float(full.csm.sum()))
    assert np.isclose(multi.loss_component, float(full.loss_component.sum()))
    # column 0 of each path is the inception total
    assert np.isclose(multi.bel, multi.bel_path[0])


def test_stitched_lic_residual_persists_past_segment_horizon():
    """A short segment's beyond-horizon LIC residual is carried to the global terminal.

    A claim settling past a segment's own term stays outstanding -- the stitch
    must hold its parked residual flat, not zero-pad it. Two segments (term 3
    vs 8) share a [0.2]*5 settlement pattern that runs past the short term;
    the global horizon is 8. The short row's stitched LIC must equal the short
    policy measured ALONE (terminal residual), carried flat to month 8 -- not
    dropped to zero after column 3 (the bug this guards).
    """
    pat = np.full(5, 0.2)

    def _sb():
        return make_death_basis(mortality_q=0.01, lapse_q=0.0,
                                discount_annual=0.0, settlement_pattern=pat)

    mp = fcf.ModelPoints(
        issue_age=np.array([40, 40]), benefits={0: np.array([1_000_000.0, 1_000_000.0])},
        premium=np.array([0.0, 0.0]), term_months=np.array([3, 8]),
        calculation_methods=PATTERNS,
        product=np.array(["A", "A"]), channel=np.array(["X", "Y"]),
    )
    m = fcf.gmm.measure(mp, BasisRouter({("A", "X"): _sb(), ("A", "Y"): _sb()}),
                        full=True)

    mp_alone = mp.subset([0])
    m_alone = fcf.gmm.measure(mp_alone, _sb(), full=True)
    residual = m_alone.lic[0, -1]

    assert residual > 0.0                                   # the tail is genuinely parked
    # cols 0..3 match the alone measure, cols 3..8 hold the residual flat
    np.testing.assert_allclose(m.lic[0, :4], m_alone.lic[0])
    np.testing.assert_allclose(m.lic[0, 3:], residual)
    # the horizon-defining segment (term 8 == global horizon) is laid in
    # directly with no tail to fill -- it must equal its alone measurement.
    m_long = fcf.gmm.measure(mp.subset([1]), _sb(), full=True)
    np.testing.assert_allclose(m.lic[1], m_long.lic[0])
