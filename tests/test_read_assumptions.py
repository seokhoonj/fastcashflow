"""read_assumptions validation -- loading an actuarial basis from a workbook."""
from pathlib import Path

import numpy as np

from fastcashflow import Assumptions, ModelPointSet, measure, read_assumptions

_SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_basis.xlsx"


def test_read_assumptions_loads_the_sample_basis():
    """The sample workbook loads into an Assumptions with the right scalars."""
    asmp = read_assumptions(_SAMPLE)
    assert isinstance(asmp, Assumptions)
    assert asmp.discount_annual == 0.03
    assert asmp.ra_confidence == 0.75
    assert asmp.mortality_cv == 0.10


def test_read_assumptions_builds_working_rate_callables():
    """The mortality and lapse callables return sensible monthly rates."""
    asmp = read_assumptions(_SAMPLE)
    young = asmp.mortality_monthly(np.array([30]), np.array([0]))
    old = asmp.mortality_monthly(np.array([60]), np.array([0]))
    assert 0.0 < young[0] < old[0] < 1.0          # mortality rises with age
    lapse = asmp.lapse_monthly(np.array([0, 5]))
    assert np.all((lapse > 0.0) & (lapse < 1.0))


def test_read_assumptions_basis_measures_a_portfolio():
    """A basis read from the workbook drives a measurement end to end."""
    asmp = read_assumptions(_SAMPLE)
    mps = ModelPointSet(
        issue_age=np.array([40, 45]),
        death_benefit=np.array([1e8, 8e7]),
        monthly_premium=np.array([25_000.0, 22_000.0]),
        term_months=np.array([120, 120]),
    )
    m = measure(mps, asmp)
    assert m.bel.shape == (2, 121)
    assert np.all(np.isfinite(m.csm[:, 0]))
