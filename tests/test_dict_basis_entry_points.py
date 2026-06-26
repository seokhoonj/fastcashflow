"""Non-GMM measure entry points accept the dict basis read_basis returns.

Regression for P1-2: ``read_basis`` ALWAYS returns a ``BasisRouter`` (a
dict subclass), so even a single-segment workbook arrives as a one-entry
dict. ``gmm.measure`` and ``gmm.measure_inforce`` ROUTE a dict per segment;
the other entry points (``vfa.measure`` / ``paa.measure`` /
``reinsurance.measure``) used to crash on it with a deep ``AttributeError``.
They now unwrap a single-segment dict and reject a genuinely multi-segment
dict with an actionable ``ValueError`` -- so the documented file -> measure
workflow works on the shipped sample.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import CalculationMethod, ModelPoints
from fastcashflow.io import BasisRouter
from conftest import make_death_basis


def _death_portfolio():
    mp = ModelPoints(
        issue_age           = np.array([40, 45], dtype=np.int64),
        benefits            = {"DEATH": np.array([1e8, 8e7])},
        premium             = np.array([50_000.0, 40_000.0]),
        term_months         = np.array([120, 120], dtype=np.int64),
        calculation_methods = {"DEATH": CalculationMethod.DEATH},
    )
    return mp, make_death_basis(mortality_q=0.005, lapse_q=0.01)


def _wrap(basis):
    return BasisRouter({("PROD_A", "FC"): basis})


# ---------------------------------------------------------------------------
# single-segment dict unwraps and equals the plain Basis
# ---------------------------------------------------------------------------
def test_paa_unwraps_single_segment_dict():
    mp, basis = _death_portfolio()
    plain = fcf.paa.measure(mp, basis)
    viadict = fcf.paa.measure(mp, _wrap(basis))
    assert np.allclose(viadict.lrc, plain.lrc)


def test_reinsurance_unwraps_single_segment_dict():
    mp, basis = _death_portfolio()
    treaty = fcf.reinsurance.QuotaShare(cession=0.5)
    plain = fcf.reinsurance.measure(mp, basis, treaty=treaty)
    viadict = fcf.reinsurance.measure(mp, _wrap(basis), treaty=treaty)
    assert np.allclose(viadict.bel, plain.bel)


def test_vfa_unwraps_single_segment_dict():
    mpv = fcf.samples.model_points(template="vfa")
    bv = fcf.samples.basis(template="vfa")
    plain = fcf.vfa.measure(mpv, bv)
    viadict = fcf.vfa.measure(mpv, _wrap(bv))
    assert np.allclose(viadict.bel, plain.bel)


# ---------------------------------------------------------------------------
# a genuinely multi-segment dict is rejected with an actionable message
# (the unwrap path is shared via basis._single_basis, exercised above)
# ---------------------------------------------------------------------------
def test_non_gmm_entry_points_reject_multi_segment_dict():
    """PAA / reinsurance / VFA take a single Basis only -- a multi-segment dict
    is rejected. (measure and measure_inforce DO route a dict per segment.)"""
    mp, basis = _death_portfolio()
    multi = BasisRouter({("A", "FC"): basis, ("B", "FC"): basis})
    treaty = fcf.reinsurance.QuotaShare(cession=0.5)
    mpv = fcf.samples.model_points(template="vfa")
    with pytest.raises(ValueError, match="single Basis"):
        fcf.paa.measure(mp, multi)
    with pytest.raises(ValueError, match="single Basis"):
        fcf.reinsurance.measure(mp, multi, treaty=treaty)
    with pytest.raises(ValueError, match="single Basis"):
        fcf.vfa.measure(mpv, multi)


def test_measure_inforce_routes_a_multi_segment_dict(tmp_path):
    """measure_inforce settles a multi-segment portfolio in one call: each
    (product, channel) routes to its own Basis, and the routed result equals
    the per-segment single-basis settlement -- no manual subsetting needed."""
    from fastcashflow._measurement.inforce import _reconcile_state
    fcf.samples.export(str(tmp_path), template="gmm", quiet=True)
    basis = fcf.read_basis(str(tmp_path / "basis.xlsx"))     # dict, 7 segments
    mp, state = fcf.read_inforce_policies(
        str(tmp_path / "inforce_policies.csv"),
        coverages=str(tmp_path / "coverages.csv"),
        calculation_methods=str(tmp_path / "calculation_methods.csv"),
    )
    routed = fcf.gmm.measure_inforce(mp, state, basis, period_months=3)
    assert routed.bel.shape[0] == mp.n_mp
    st = _reconcile_state(mp, state)
    prod, chan = np.asarray(mp.product), np.asarray(mp.channel)
    seen = 0
    for key in basis.segments:
        idx = np.nonzero((prod == key[0]) & (chan == key[1]))[0]
        if idx.size == 0:
            continue
        ref = fcf.gmm.measure_inforce(mp.subset(idx), st.subset(idx),
                                      basis.resolve(key), period_months=3)
        assert np.allclose(routed.bel[idx], ref.bel)
        assert np.allclose(routed.csm[idx], ref.csm)
        seen += idx.size
    assert seen == mp.n_mp                              # every contract routed
