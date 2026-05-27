"""Tests for the load-bearing footguns the 2nd review surfaced.

Three positional / silent-overwrite traps that previously slipped through
without an error -- each now blocked at engine entry.
"""
import numpy as np
import pytest

from fastcashflow import (
    Assumptions, BenefitPattern, CoverageRate, ModelPoints,
    measure, value, value_segmented,
)
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_assumptions


def _flat(annual_q):
    return lambda sex, issue_age, duration: np.full(issue_age.shape, annual_q)


# ---------------------------------------------------------------------------
# value_segmented: '|' in product_code / channel_code is the key separator
# ---------------------------------------------------------------------------

def test_value_segmented_rejects_pipe_in_product_code():
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([12_000.0]),
        term_months=np.array([60]),
        product_code=np.array(["TERM|2020"]),         # the trap
        channel_code=np.array(["FC"]),
        benefits={0: np.array([1e8])},
        benefit_patterns=PATTERNS,
    )
    basis = {("TERM|2020", "FC"): make_death_assumptions(
        mortality_q=0.005, lapse_q=0.01)}
    with pytest.raises(ValueError, match="product_code.*'\\|'"):
        value_segmented(mp, basis)


def test_value_segmented_rejects_pipe_in_channel_code():
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([12_000.0]),
        term_months=np.array([60]),
        product_code=np.array(["TERM_LIFE_A"]),
        channel_code=np.array(["FC|GA"]),            # the trap
        benefits={0: np.array([1e8])},
        benefit_patterns=PATTERNS,
    )
    basis = {("TERM_LIFE_A", "FC|GA"): make_death_assumptions(
        mortality_q=0.005, lapse_q=0.01)}
    with pytest.raises(ValueError, match="channel_code.*'\\|'"):
        value_segmented(mp, basis)


# ---------------------------------------------------------------------------
# validate_csr_codes catalogue-consistency check
# ---------------------------------------------------------------------------

def test_engine_rejects_catalogue_mismatch():
    """An Assumptions.coverages code that's absent from the model points'
    benefit_patterns catalogue lands without a routing pattern and the
    engine falls back silently. Catch it loudly."""
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 1e8},
        level_premium=12_000.0, term_months=60,
        benefit_patterns={"DEATH": BenefitPattern.DEATH},  # catalogue: DEATH
    )
    asmp = Assumptions(
        mortality_annual=_flat(_annual(0.005)),
        lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("CANCER", _flat(_annual(0.005))),),  # mismatch
    )
    with pytest.raises(ValueError, match="catalogue"):
        measure(mp, asmp)
    with pytest.raises(ValueError, match="catalogue"):
        value(mp, asmp)
