"""Mid-month discounting precision validation.

Claims arise during the month (discounted mid-month) and premiums at the
start of the month. The present-value timing is hand-checked against an
independent recomputation.
"""
import numpy as np

from fastcashflow import ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, make_death_basis


def _flat_assumptions(**overrides):
    kw = dict(
        mortality_q     = 0.01,
        lapse_q         = 0.0,
        discount_annual = 0.06,
        ra_confidence   = 0.75,
        mortality_cv    = 0.0,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


def test_mid_month_discounting():
    """Claims discounted mid-month, premiums start-of-month -- hand-checked."""
    death_benefit = 1_000_000.0
    premium = 12_000.0
    term = 2
    q = 0.01

    res = measure(
        ModelPoints.single(
            issue_age=40, benefits={"DEATH": death_benefit},
            premium=premium, term_months=term,
            calculation_methods=PATTERNS,
        ),
        _flat_assumptions(),
    )

    i = (1.0 + 0.06) ** (1.0 / 12.0) - 1.0
    inforce = np.array([1.0, 1.0 - q])           # zero lapse
    deaths = inforce * q
    t = np.arange(term)
    d_start = (1.0 + i) ** (-t)
    d_mid = (1.0 + i) ** (-(t + 0.5))

    pv_claims = float(np.sum(deaths * death_benefit * d_mid))
    pv_premiums = float(np.sum(inforce * premium * d_start))
    assert np.isclose(res.bel_path[0, 0], pv_claims - pv_premiums)

    # the timing genuinely differs from the old all-start-of-month basis
    bel_all_start = float(np.sum(deaths * death_benefit * d_start)
                          - np.sum(inforce * premium * d_start))
    assert not np.isclose(res.bel_path[0, 0], bel_all_start)
