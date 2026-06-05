"""Non-GMM measure entry points accept the dict basis read_basis returns.

Regression for P1-2: ``read_basis`` ALWAYS returns a ``SegmentedBasis`` (a
dict subclass), so even a single-segment workbook arrives as a one-entry
dict. ``gmm.measure`` routes a dict; the other entry points
(``measure_vfa`` / ``measure_paa`` / ``measure_reinsurance`` /
``measure_inforce``) used to crash on it with a deep ``AttributeError``. They
now unwrap a single-segment dict and reject a genuinely multi-segment dict
with an actionable ``ValueError`` -- so the documented file -> measure
workflow works on the shipped single-segment sample.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import CalculationMethod, ModelPoints
from fastcashflow.io import SegmentedBasis
from conftest import make_death_basis


def _death_portfolio():
    mp = ModelPoints(
        issue_age           = np.array([40, 45], dtype=np.int64),
        benefits            = {0: np.array([1e8, 8e7])},
        premium             = np.array([50_000.0, 40_000.0]),
        term_months         = np.array([120, 120], dtype=np.int64),
        calculation_methods = {"DEATH": CalculationMethod.DEATH},
    )
    return mp, make_death_basis(mortality_q=0.005, lapse_q=0.01)


def _wrap(basis):
    return SegmentedBasis({("PROD_A", "FC"): basis})


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
    plain = fcf.reinsurance.measure(mp, basis, treaty)
    viadict = fcf.reinsurance.measure(mp, _wrap(basis), treaty)
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
    mp, basis = _death_portfolio()
    multi = SegmentedBasis({("A", "FC"): basis, ("B", "FC"): basis})
    treaty = fcf.reinsurance.QuotaShare(cession=0.5)
    mpv = fcf.samples.model_points(template="vfa")
    state = fcf.InforceState(
        mp_id=np.array([0, 1]), elapsed_months=np.array([12, 24], dtype=np.int64),
        count=np.array([1.0, 1.0]), prior_csm=np.array([0.0, 0.0]), lock_in_rate=0.03,
    )
    with pytest.raises(ValueError, match="single Basis"):
        fcf.paa.measure(mp, multi)
    with pytest.raises(ValueError, match="single Basis"):
        fcf.reinsurance.measure(mp, multi, treaty)
    with pytest.raises(ValueError, match="single Basis"):
        fcf.vfa.measure(mpv, multi)
    with pytest.raises(ValueError, match="single Basis"):
        fcf.gmm.measure_inforce(mp, multi, state)
