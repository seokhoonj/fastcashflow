"""Reinsurance-held validation -- a quota-share treaty over a direct portfolio.

The cedant cedes a fraction of its claims (recovered) and the same fraction
of its premiums (paid to the reinsurer). The CSM carries the net cost or
gain of the cover -- it may be negative, and there is no loss component.
"""
import fastcashflow as fcf
import numpy as np
import pytest

from fastcashflow import ModelPoints
from fastcashflow.numerics import _norm_ppf
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


Q = 0.002          # flat monthly mortality
LAPSE = 0.005      # flat monthly lapse
MORTALITY_CV = 0.10


def _assumptions():
    return make_death_basis(
        mortality_q     = Q,
        lapse_q         = LAPSE,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = MORTALITY_CV,
    )


def test_reinsurance_hand_calc():
    """Single quota-share treaty -- hand-checked BEL, RA and CSM."""
    basis = _assumptions()
    death_benefit, premium, term, cession = 1e8, 80_000.0, 60, 0.4
    res = fcf.reinsurance.measure(
        ModelPoints.single(40, premium, term, benefits={0: death_benefit}, calculation_methods=PATTERNS),
        basis, fcf.reinsurance.QuotaShare(cession=cession)
    )

    i = basis.discount_monthly
    surv = (1.0 - Q) * (1.0 - LAPSE)
    half = (1.0 + i) ** (-0.5)
    full = 1.0 / (1.0 + i)
    geom = float(np.sum((surv * full) ** np.arange(term)))

    pv_recovery = cession * Q * death_benefit * half * geom
    pv_reinsurance_premium = cession * premium * geom
    bel = pv_reinsurance_premium - pv_recovery
    ra = _norm_ppf(basis.ra_confidence) * MORTALITY_CV * pv_recovery

    assert np.isclose(res.bel[0], bel)
    assert np.isclose(res.ra[0], ra)
    assert np.isclose(res.csm_path[0, 0], -(bel - ra))


def test_reinsurance_csm_can_be_negative():
    """Ceding a profitable book has a net cost -- a negative CSM, no loss component."""
    res = fcf.reinsurance.measure(
        ModelPoints.single(40, 300_000.0, 60, benefits={0: 1e8}, calculation_methods=PATTERNS),
        _assumptions(), fcf.reinsurance.QuotaShare(cession=0.5)
    )
    assert res.bel[0] > 0.0           # reinsurance premiums ceded exceed recoveries
    assert res.csm_path[0, 0] < 0.0        # the net cost is carried as a negative CSM


def test_reinsurance_csm_analysis_of_change_reconciles():
    """The reinsurance CSM waterfall reconciles opening to closing."""
    res = fcf.reinsurance.measure(
        ModelPoints.single(40, 80_000.0, 120, benefits={0: 1e8}, calculation_methods=PATTERNS),
        _assumptions(), fcf.reinsurance.QuotaShare(cession=0.3)
    )
    assert np.allclose(
        res.csm_path[:, :-1] + res.csm_accretion - res.csm_release, res.csm_path[:, 1:]
    )


def test_reinsurance_zero_cession_is_nothing():
    """A zero cession rate cedes nothing -- every figure is zero."""
    res = fcf.reinsurance.measure(
        ModelPoints.single(40, 80_000.0, 60, benefits={0: 1e8}, calculation_methods=PATTERNS),
        _assumptions(), fcf.reinsurance.QuotaShare(cession=0.0)
    )
    assert np.allclose(res.bel, 0.0)
    assert np.allclose(res.ra, 0.0)
    assert np.allclose(res.csm, 0.0)
    assert np.allclose(res.recovery, 0.0)


def test_reinsurance_rejects_bad_cession_rate():
    """A cession rate outside [0, 1] is an error."""
    with pytest.raises(ValueError, match="cession"):
        fcf.reinsurance.measure(
            ModelPoints.single(40, 80_000.0, 60, benefits={0: 1e8}, calculation_methods=PATTERNS),
            _assumptions(), fcf.reinsurance.QuotaShare(cession=1.5)
        )
