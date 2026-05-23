"""Assumptions workbook reader.

A single ``assumptions.xlsx`` carries the segment mapping plus the named rate
tables. The ``segments`` sheet has a ``defaults`` row whose values blank
cells inherit, and one row per (product, channel) segment; the reader returns
one ``Assumptions`` per segment. See docs/assumptions-format.md.
"""
import numpy as np

from fastcashflow import (
    ModelPoints, load_sample_assumptions, measure, value,
)


def test_segments_resolve():
    """The sample workbook resolves to two segments -- term_a on GA and FC."""
    basis = load_sample_assumptions()
    assert set(basis) == {("term_a", "GA"), ("term_a", "FC")}


def test_defaults_inherited():
    """Blank cells in a segment row inherit from the ``defaults`` row."""
    basis = load_sample_assumptions()
    ga, fc = basis[("term_a", "GA")], basis[("term_a", "FC")]
    # ra_confidence / mortality_cv / morbidity_cv live only on the defaults row
    assert ga.ra_confidence == 0.75 and fc.ra_confidence == 0.75
    assert ga.mortality_cv == 0.10 and fc.mortality_cv == 0.10
    assert ga.morbidity_cv == 0.12 and fc.morbidity_cv == 0.12
    # shared economic / maintenance tables -- inherited identically
    assert ga.discount_annual == 0.03 and fc.discount_annual == 0.03
    assert ga.expense_inflation == 0.02 and fc.expense_inflation == 0.02
    assert ga.expense_maintenance_annual == 60_000.0
    assert fc.expense_maintenance_annual == 60_000.0


def test_channel_segmented_lapse():
    """GA and FC reference different lapse tables -- the per-segment table
    reference. GA persistency is worse than FC."""
    basis = load_sample_assumptions()
    dur = np.arange(6)
    zero = np.zeros_like(dur)
    ga_lapse = basis[("term_a", "GA")].lapse_annual(zero, zero, dur)
    fc_lapse = basis[("term_a", "FC")].lapse_annual(zero, zero, dur)
    assert np.all(ga_lapse > fc_lapse)


def test_per_segment_scalar():
    """``expense_acquisition`` is filled per segment row (GA vs FC commission)."""
    basis = load_sample_assumptions()
    assert basis[("term_a", "GA")].expense_acquisition == 150_000.0
    assert basis[("term_a", "FC")].expense_acquisition == 80_000.0


def test_riders_resolved():
    """Rate-driven riders resolve from ``rider_rate_tables``; non-rate-driven
    types stay in ``coverage_types`` only."""
    basis = load_sample_assumptions()
    asmp = basis[("term_a", "GA")]
    # adb is rate-driven (death-type), so it joins the riders tuple too.
    assert [r.code for r in asmp.riders] == ["hosp", "cancer", "adb"]
    assert asmp.coverage_types == {
        "dth_main": "death_main",
        "hosp": "morbidity",
        "cancer": "diagnosis",
        "adb": "death",
        "ann": "annuity",
        "mat": "maturity",
    }


def test_resolved_basis_values():
    """A resolved ``Assumptions`` runs through ``value`` and ``measure``; the
    GA and FC segments give different BEL because lapse differs (channel
    segmentation actually bites the valuation)."""
    basis = load_sample_assumptions()
    mp = ModelPoints.single(issue_age=40, death_benefit=100_000_000.0,
                            level_premium=50_000.0, term_months=120)
    ga = value(mp, basis[("term_a", "GA")]).bel[0]
    fc = value(mp, basis[("term_a", "FC")]).bel[0]
    assert np.isfinite(ga) and np.isfinite(fc)
    assert not np.isclose(ga, fc)
    # fused and detailed paths agree
    assert np.isclose(measure(mp, basis[("term_a", "GA")]).bel[0, 0], ga)
