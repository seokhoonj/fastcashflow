"""Risk Adjustment validation -- the confidence-level and cost-of-capital methods.

The confidence-level RA is a percentile margin on the benefit present values.
The cost-of-capital RA holds that margin as non-financial-risk capital and
charges the cost-of-capital rate on it over the contract's run-off.
"""
import numpy as np
import pytest

from fastcashflow import ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


Q = 0.002          # flat monthly mortality
LAPSE = 0.005      # flat monthly lapse


def _basis(**overrides):
    kw = dict(
        mortality_q     = Q,
        lapse_q         = LAPSE,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = 0.10,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


def test_cost_of_capital_ra_hand_calc():
    """The CoC RA at inception is the cost-of-capital rate times the present
    value of the confidence-level margin held as capital."""
    mp = ModelPoints.single(40, 60_000.0, 60, benefits={0: 1e8}, calculation_methods=PATTERNS)
    coc_rate = 0.06
    cl = measure(mp, _basis())
    coc = measure(mp, _basis(ra_method="cost_of_capital",
                                   cost_of_capital_rate=coc_rate))

    # the confidence-level run's RA trajectory is the capital held under CoC
    full = cl.discount_bom[1]                       # (1 + i)^-1
    capital = cl.ra_path[0]
    cap_pv0 = float(np.sum(capital * full ** np.arange(capital.shape[0])))
    assert np.isclose(coc.ra_path[0, 0], coc_rate * cap_pv0)


def test_coc_ra_scales_with_the_rate():
    """The cost-of-capital RA is linear in the cost-of-capital rate."""
    mp = ModelPoints.single(40, 60_000.0, 60, benefits={0: 1e8}, calculation_methods=PATTERNS)
    coc1 = measure(mp, _basis(ra_method="cost_of_capital",
                                    cost_of_capital_rate=0.04))
    coc2 = measure(mp, _basis(ra_method="cost_of_capital",
                                    cost_of_capital_rate=0.08))
    assert np.isclose(coc2.ra_path[0, 0], 2.0 * coc1.ra_path[0, 0])


def test_coc_ra_differs_from_confidence_level():
    """The two methods give genuinely different RAs of the same order."""
    mp = ModelPoints.single(40, 60_000.0, 60, benefits={0: 1e8}, calculation_methods=PATTERNS)
    cl = measure(mp, _basis())
    coc = measure(mp, _basis(ra_method="cost_of_capital"))
    assert not np.isclose(cl.ra_path[0, 0], coc.ra_path[0, 0])
    assert 0.1 < coc.ra_path[0, 0] / cl.ra_path[0, 0] < 5.0


def test_value_rejects_cost_of_capital():
    """measure() computes the confidence-level RA only."""
    mp = ModelPoints.single(40, 60_000.0, 60, benefits={0: 1e8}, calculation_methods=PATTERNS)
    with pytest.raises(ValueError, match="confidence-level"):
        measure(mp, _basis(ra_method="cost_of_capital"), full=False)


def test_invalid_ra_method_is_rejected():
    """An unrecognised ra_method is an error."""
    mp = ModelPoints.single(40, 60_000.0, 60, benefits={0: 1e8}, calculation_methods=PATTERNS)
    with pytest.raises(ValueError, match="ra_method"):
        measure(mp, _basis(ra_method="margins"))
