"""read_assumptions validation -- loading an actuarial basis from a workbook."""
import numpy as np

from fastcashflow import (
    Assumptions,
    RiderRate,
    load_sample_assumptions,
    load_sample_model_points,
    measure,
)


def test_read_assumptions_loads_the_sample_basis():
    """The bundled workbook loads into an Assumptions with the right scalars."""
    asmp = load_sample_assumptions()
    assert isinstance(asmp, Assumptions)
    assert asmp.discount_annual == 0.03
    assert asmp.ra_confidence == 0.75
    assert asmp.mortality_cv == 0.10


def test_read_assumptions_builds_working_rate_callables():
    """Mortality and lapse callables return sensible monthly rates."""
    asmp = load_sample_assumptions()
    young = asmp.mortality_annual(np.array([0]), np.array([30]), np.array([0]))
    old = asmp.mortality_annual(np.array([0]), np.array([60]), np.array([0]))
    assert 0.0 < young[0] < old[0] < 1.0          # mortality rises with age
    lapse = asmp.lapse_annual(np.array([0, 5]))
    assert np.all((lapse > 0.0) & (lapse < 1.0))


def test_read_assumptions_registers_riders():
    """The riders master and rates sheets become rate-driven RiderRates."""
    asmp = load_sample_assumptions()
    assert len(asmp.riders) > 0
    assert all(isinstance(r, RiderRate) for r in asmp.riders)
    assert asmp.coverage_types is not None
    for r in asmp.riders:                         # each rate is a monthly rate
        rate = r.rate(np.array([0]), np.array([45]), np.array([0]))
        assert 0.0 <= rate[0] < 1.0


def test_read_assumptions_basis_measures_a_portfolio():
    """A basis read from the workbook drives a measurement end to end."""
    asmp = load_sample_assumptions()
    mps = load_sample_model_points()
    m = measure(mps, asmp)
    assert m.bel.shape[0] == mps.n_mp
    assert np.all(np.isfinite(m.csm[:, 0]))
