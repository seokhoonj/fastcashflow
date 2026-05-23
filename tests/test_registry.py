"""Registry-format actuarial basis reader (v1).

The registry format splits the basis into two workbooks: a table-registry
(named rate tables) plus a basis workbook with a ``defaults`` row and one
row per (product, channel) segment. Blank cells inherit the defaults; the
reader returns one ``Assumptions`` per segment. See docs/assumptions-format.md.
"""
import numpy as np

from fastcashflow import (
    ModelPoints, load_sample_registry, measure, value,
)


def test_registry_resolves_segments():
    """The sample workbook resolves to two segments -- termA on GA and FC."""
    reg = load_sample_registry()
    assert set(reg) == {("termA", "GA"), ("termA", "FC")}


def test_defaults_inherited():
    """Blank cells in a segment row inherit from the ``defaults`` row."""
    reg = load_sample_registry()
    ga, fc = reg[("termA", "GA")], reg[("termA", "FC")]
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
    reg = load_sample_registry()
    dur = np.arange(6)
    ga_lapse = reg[("termA", "GA")].lapse_annual(dur)
    fc_lapse = reg[("termA", "FC")].lapse_annual(dur)
    assert np.all(ga_lapse > fc_lapse)


def test_per_segment_scalar():
    """``expense_acquisition`` is filled per segment row (GA vs FC commission)."""
    reg = load_sample_registry()
    assert reg[("termA", "GA")].expense_acquisition == 150_000.0
    assert reg[("termA", "FC")].expense_acquisition == 80_000.0


def test_riders_resolved():
    """Rate-driven riders resolve from ``rider_rate_tables``; non-rate-driven
    types stay in ``coverage_types`` only."""
    reg = load_sample_registry()
    asmp = reg[("termA", "GA")]
    assert [r.code for r in asmp.riders] == ["hosp", "cancer"]
    assert asmp.coverage_types == {
        "dth_main": "death_main",
        "hosp": "morbidity",
        "cancer": "diagnosis",
    }


def test_resolved_basis_values():
    """A resolved ``Assumptions`` runs through ``value`` and ``measure``; the
    GA and FC segments give different BEL because lapse differs (channel
    segmentation actually bites the valuation)."""
    reg = load_sample_registry()
    mp = ModelPoints.single(issue_age=40, death_benefit=100_000_000.0,
                            level_premium=50_000.0, term_months=120)
    ga = value(mp, reg[("termA", "GA")]).bel[0]
    fc = value(mp, reg[("termA", "FC")]).bel[0]
    assert np.isfinite(ga) and np.isfinite(fc)
    assert not np.isclose(ga, fc)
    # fused and detailed paths agree
    assert np.isclose(measure(mp, reg[("termA", "GA")]).bel[0, 0], ga)
