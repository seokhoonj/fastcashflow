"""Reinsurance-held validation -- a quota-share treaty over a direct portfolio.

The cedant cedes a fraction of its claims (recovered) and the same fraction
of its premiums (paid to the reinsurer). The CSM carries the net cost or
gain of the cover -- it may be negative, and there is no loss component.
"""
import numpy as np
import pytest

from fastcashflow import Assumptions, ModelPoints, measure_reinsurance
from fastcashflow.gmm import _norm_ppf

Q = 0.002          # flat monthly mortality
LAPSE = 0.005      # flat monthly lapse
MORTALITY_CV = 0.10


def _annual(m: float) -> float:
    """Convert a monthly rate to its annual equivalent so the engine converts back."""
    return 1.0 - (1.0 - m) ** 12


def _assumptions() -> Assumptions:
    return Assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(Q)),
        lapse_annual=lambda duration: np.full(duration.shape, _annual(LAPSE)),
        discount_annual=0.03,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=MORTALITY_CV,
    )


def test_reinsurance_hand_calc():
    """Single quota-share treaty -- hand-checked BEL, RA and CSM."""
    asmp = _assumptions()
    death_benefit, premium, term, cession = 1e8, 80_000.0, 60, 0.4
    res = measure_reinsurance(
        ModelPoints.single(40, death_benefit, premium, term), asmp, cession
    )

    i = asmp.discount_monthly
    surv = (1.0 - Q) * (1.0 - LAPSE)
    half = (1.0 + i) ** (-0.5)
    full = 1.0 / (1.0 + i)
    geom = float(np.sum((surv * full) ** np.arange(term)))

    pv_recovery = cession * Q * death_benefit * half * geom
    pv_reins_premium = cession * premium * geom
    bel = pv_reins_premium - pv_recovery
    ra = _norm_ppf(asmp.ra_confidence) * MORTALITY_CV * pv_recovery

    assert np.isclose(res.bel[0], bel)
    assert np.isclose(res.ra[0], ra)
    assert np.isclose(res.csm[0, 0], -(bel - ra))


def test_reinsurance_csm_can_be_negative():
    """Ceding a profitable book has a net cost -- a negative CSM, no loss component."""
    res = measure_reinsurance(
        ModelPoints.single(40, 1e8, 300_000.0, 60), _assumptions(), 0.5
    )
    assert res.bel[0] > 0.0           # reinsurance premiums ceded exceed recoveries
    assert res.csm[0, 0] < 0.0        # the net cost is carried as a negative CSM


def test_reinsurance_csm_analysis_of_change_reconciles():
    """The reinsurance CSM waterfall reconciles opening to closing."""
    res = measure_reinsurance(
        ModelPoints.single(40, 1e8, 80_000.0, 120), _assumptions(), 0.3
    )
    assert np.allclose(
        res.csm[:, :-1] + res.csm_accretion - res.csm_release, res.csm[:, 1:]
    )


def test_reinsurance_zero_cession_is_nothing():
    """A zero cession rate cedes nothing -- every figure is zero."""
    res = measure_reinsurance(
        ModelPoints.single(40, 1e8, 80_000.0, 60), _assumptions(), 0.0
    )
    assert np.allclose(res.bel, 0.0)
    assert np.allclose(res.ra, 0.0)
    assert np.allclose(res.csm, 0.0)
    assert np.allclose(res.recovery, 0.0)


def test_reinsurance_rejects_bad_cession_rate():
    """A cession rate outside [0, 1] is an error."""
    with pytest.raises(ValueError, match="cession_rate"):
        measure_reinsurance(
            ModelPoints.single(40, 1e8, 80_000.0, 60), _assumptions(), 1.5
        )
