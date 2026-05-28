"""Tests for the load-bearing footguns the 2nd review surfaced.

Three positional / silent-overwrite traps that previously slipped through
without an error -- each now blocked at engine entry.
"""
import numpy as np
import pytest

from fastcashflow import (
    Assumptions, CalculationMethod, CoverageRate, ModelPoints,
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
        calculation_methods=PATTERNS,
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
        calculation_methods=PATTERNS,
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
    calculation_methods catalogue lands without a routing pattern and the
    engine falls back silently. Catch it loudly."""
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 1e8},
        level_premium=12_000.0, term_months=60,
        calculation_methods={"DEATH": CalculationMethod.DEATH},  # catalogue: DEATH
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


# ---------------------------------------------------------------------------
# validate_csr_codes: ordered coverage_codes guard
# ---------------------------------------------------------------------------

def test_engine_reorders_coverages_by_code():
    """ModelPoints carry ``coverage_codes=(DEATH, CANCER)``; the engine
    aligns ``Assumptions.coverages`` to that order by *code* at entry. So an
    Assumptions registered in a different order (``(CANCER, DEATH)``) yields
    the **same** result -- DEATH amounts always meet DEATH rates regardless
    of registration order. This is the decouple: reading the portfolio never
    has to know the assumptions' internal coverage order."""
    rate_death = _flat(_annual(0.005))
    rate_cancer = _flat(_annual(0.003))
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([12_000.0]),
        term_months=np.array([60]),
        benefits={0: np.array([1e8]), 1: np.array([1e7])},
        calculation_methods={"DEATH": CalculationMethod.DEATH,
                          "CANCER": CalculationMethod.DIAGNOSIS},
        coverage_codes=("DEATH", "CANCER"),
    )
    asmp_ordered = Assumptions(
        mortality_annual=rate_death, lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", rate_death),
                   CoverageRate("CANCER", rate_cancer)),
    )
    asmp_swapped = Assumptions(
        mortality_annual=rate_death, lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("CANCER", rate_cancer),   # swapped order
                   CoverageRate("DEATH", rate_death)),
    )
    # Reorder by code makes the two equivalent -- same BEL whichever order
    # the assumptions register the coverages in.
    assert np.allclose(np.asarray(measure(mp, asmp_ordered).bel),
                       np.asarray(measure(mp, asmp_swapped).bel))
    assert np.isclose(float(np.asarray(value(mp, asmp_ordered).bel).ravel()[0]),
                      float(np.asarray(value(mp, asmp_swapped).bel).ravel()[0]))


def test_engine_rejects_unregistered_coverage():
    """V4: a code the model points reference but the assumptions do not
    register has no rate_table. The engine raises at entry naming the
    missing code, rather than silently scoring it zero."""
    rate = _flat(_annual(0.005))
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([12_000.0]),
        term_months=np.array([60]),
        benefits={0: np.array([1e8]), 1: np.array([1e7])},
        calculation_methods={"DEATH": CalculationMethod.DEATH,
                          "CANCER": CalculationMethod.DIAGNOSIS},
        coverage_codes=("DEATH", "CANCER"),
    )
    asmp = Assumptions(
        mortality_annual=rate, lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", rate),),   # CANCER not registered
    )
    with pytest.raises(ValueError, match="no registered coverage"):
        measure(mp, asmp)
    with pytest.raises(ValueError, match="no registered coverage"):
        value(mp, asmp)


def test_engine_accepts_matching_coverage_codes():
    """Sanity check the order guard: when the assumptions ordering matches
    the pinned tuple, both engine entry points run as before."""
    rate_death = _flat(_annual(0.005))
    rate_cancer = _flat(_annual(0.003))
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([12_000.0]),
        term_months=np.array([60]),
        benefits={0: np.array([1e8]), 1: np.array([1e7])},
        calculation_methods={"DEATH": CalculationMethod.DEATH,
                          "CANCER": CalculationMethod.DIAGNOSIS},
        coverage_codes=("DEATH", "CANCER"),
    )
    asmp = Assumptions(
        mortality_annual=rate_death,
        lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", rate_death),
                   CoverageRate("CANCER", rate_cancer)),
    )
    # No exception; both paths produce finite results.
    r = measure(mp, asmp)
    assert np.all(np.isfinite(np.asarray(r.bel)))
    v = value(mp, asmp)
    assert np.all(np.isfinite(np.asarray(v.bel)))


def test_wide_reader_populates_coverage_codes(tmp_path):
    """The wide-form reader pins ``coverage_codes`` to the assumptions
    ordering so a later reordered Assumptions is refused by the engine
    without any extra wiring on the user's side."""
    import polars as pl
    from fastcashflow import read_model_points
    asmp = Assumptions(
        mortality_annual=_flat(_annual(0.005)),
        lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat(_annual(0.005))),
                   CoverageRate("CANCER", _flat(_annual(0.003)))),
    )
    path = tmp_path / "mp.csv"
    pl.DataFrame({
        "issue_age": [40.0], "term_months": [60], "level_premium": [12_000.0],
        "DEATH_benefit": [1e8], "CANCER_benefit": [1e7],
    }).write_csv(path)
    mp = read_model_points(path, calculation_methods={
        "DEATH": CalculationMethod.DEATH, "CANCER": CalculationMethod.DIAGNOSIS,
    })
    assert mp.coverage_codes == ("DEATH", "CANCER")


def test_engine_ignores_unreferenced_assumptions_coverage():
    """An Assumptions that registers more coverages than the portfolio uses
    is fine -- the engine builds rates only for the codes the model points
    reference (via coverage_codes), ignoring the extras. Code-based
    alignment means a registered-but-unused coverage cannot cause the
    position-drift the old length guard worried about. The result matches a
    slim Assumptions carrying only the referenced coverage."""
    rate = _flat(_annual(0.005))
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        level_premium=np.array([12_000.0]),
        term_months=np.array([60]),
        benefits={0: np.array([1e8])},
        calculation_methods={"DEATH": CalculationMethod.DEATH,
                          "CANCER": CalculationMethod.DIAGNOSIS},
        coverage_codes=("DEATH",),
    )
    asmp_extra = Assumptions(
        mortality_annual=rate, lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", rate),
                   CoverageRate("CANCER", rate)),   # extra, unused
    )
    asmp_slim = Assumptions(
        mortality_annual=rate, lapse_annual=_flat(_annual(0.01)),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", rate),),
    )
    assert np.allclose(np.asarray(measure(mp, asmp_extra).bel),
                       np.asarray(measure(mp, asmp_slim).bel))
