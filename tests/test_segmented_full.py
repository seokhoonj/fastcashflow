"""Full-trajectory measurement of a multi-segment (per-basis-dict) portfolio.

``measure(model_points, basis_dict, full=True)`` runs each (product_code,
channel_code) segment under its own basis and stitches the per-segment
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


def _segments(mp, basis):
    pc = np.array(mp.product_code)
    ch = np.array(mp.channel_code)
    for key in basis:
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
        ref = fcf.gmm.measure(mp.subset(idx), basis[key])   # single basis -> full
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
            fcf.gmm.measure(mp.subset(idx), basis[key]), period_months=12))[0]
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
    basis = fcf.samples.basis()[("TERM_LIFE_A", "FC")]
    m = fcf.gmm.measure(mp.subset([0]), basis)
    assert m.discount_bom.ndim == 1
    assert m.discount_mid.ndim == 1


def test_segmented_full_unsupported_ops_refuse():
    """Operations that don't yet handle the per-MP (2-D) result raise clearly."""
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis()
    m = fcf.gmm.measure(mp, basis)
    n_mp = mp.n_mp

    with pytest.raises(NotImplementedError):
        fcf.transition(m, np.zeros(n_mp))
    with pytest.raises(NotImplementedError):
        fcf.group(m, np.zeros(n_mp))
    with pytest.raises(NotImplementedError):
        fcf.report(m)
    # an assumption-revision roll-forward is not supported on a segmented result
    with pytest.raises(NotImplementedError):
        fcf.roll_forward(m, period_months=12, revised=m, revised_at=12)
    # a clean roll-forward, by contrast, works
    rf = fcf.roll_forward(m, period_months=12)
    assert len(rf) > 0
