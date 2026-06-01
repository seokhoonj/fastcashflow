"""ModelPoints.subset + value_segmented -- per-segment portfolio valuation.

`ModelPoints` may carry per-row `product` / `channel` strings naming each
contract's segment. `measure(mp, basis, full=False)` splits the portfolio by
those keys, looks each segment's `Basis` up in the
`{(product, channel): Basis}` dict, calls :func:`value` per segment,
and writes the per-mp results back to a single ``(n_mp,)`` `GMMMeasurement`.
"""
import numpy as np
import pytest

from fastcashflow import (
    Basis, ModelPoints, load_sample_basis, measure, measure,
    CoverageRate,
)


def _flat_asmp(*, discount=0.05) -> Basis:
    return Basis(
        mortality_annual=lambda s, ia, d: np.full(s.shape, 0.001),
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=discount,
        ra_confidence=0.75,
        mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", lambda s, ia, d: np.full(s.shape, 0.001)),),
    )


# ---------------------------------------------------------------------------
# ModelPoints.subset
# ---------------------------------------------------------------------------

def test_subset_keeps_selected_rows():
    """Subsetting by indices preserves per-row scalar fields."""
    mp = ModelPoints(
        issue_age=np.array([30, 40, 50, 60]),
        level_premium=np.array([100.0, 200.0, 300.0, 400.0]),
        term_months=np.array([120, 120, 120, 120]),
        benefits={0: np.array([1_000.0, 2_000.0, 3_000.0, 4_000.0])},
    )
    sub = mp.subset([0, 2])
    assert sub.n_mp == 2
    assert sub.issue_age.tolist() == [30, 50]
    assert sub.level_premium.tolist() == [100.0, 300.0]
    # Per-coverage amounts survive the subset (the CSR is rebuilt for the
    # selected rows). The death coverage's per-mp amount is at coverage_index=0.
    assert sub.coverage_amount.tolist() == [1_000.0, 3_000.0]


def test_subset_rebuilds_csr_coverages():
    """Sub-MP's CSR coverage arrays are densified for the selected rows only."""
    # mp 0 -> 1 coverage; mp 1 -> 2 coverages; mp 2 -> 1 coverage
    mp = ModelPoints(
        issue_age=np.array([30, 40, 50]),
        level_premium=np.zeros(3),
        term_months=np.array([120, 120, 120]),
        benefits={0: np.array([1_000.0, 2_000.0, 3_000.0]), 2: np.array([0.0, 500.0, 0.0])},      # second coverage on mp 1
    )
    assert mp.coverage_offset.tolist() == [0, 1, 3, 4]       # 1 + 2 + 1

    sub = mp.subset([0, 2])                              # skip mp 1 (2 coverages)
    assert sub.coverage_offset.tolist() == [0, 1, 2]          # 1 + 1
    assert sub.coverage_index.tolist() == [0, 0]               # both DEATH
    assert sub.coverage_amount.tolist() == [1_000.0, 3_000.0]


def test_subset_slices_product_and_channel_when_set():
    """Segment metadata is sliced alongside per-row fields."""
    mp = ModelPoints(
        issue_age=np.array([30, 40, 50]),
        level_premium=np.zeros(3),
        term_months=np.array([120, 120, 120]),
        benefits={0: np.array([1_000.0, 2_000.0, 3_000.0])},
        product_code=np.array(["TERM_A", "TERM_A", "term_b"]),
        channel_code=np.array(["GA", "FC", "GA"]),
    )
    sub = mp.subset([1, 2])
    assert sub.product_code.tolist() == ["TERM_A", "term_b"]
    assert sub.channel_code.tolist() == ["FC", "GA"]


def test_subset_preserves_issue_class_and_elapsed_months():
    """The newer per-row fields (issue_class for the UW class axis, and
    elapsed_months for the in-force valuation date) must round-trip
    through subset(); otherwise value_segmented silently resets them to
    zero on the segmented portfolio."""
    mp = ModelPoints(
        issue_age=np.array([30, 40, 50]),
        level_premium=np.zeros(3),
        term_months=np.array([120, 120, 120]),
        benefits={0: np.array([1_000.0, 2_000.0, 3_000.0])},
        issue_class=np.array([0, 1, 2], dtype=np.int64),
        elapsed_months=np.array([0, 24, 60], dtype=np.int64),
    )
    sub = mp.subset([1, 2])
    assert sub.issue_class.tolist() == [1, 2]
    assert sub.elapsed_months.tolist() == [24, 60]


def test_subset_leaves_product_none_when_unset():
    mp = ModelPoints(
        issue_age=np.array([30, 40]),
        level_premium=np.zeros(2),
        term_months=np.array([120, 120]),
        benefits={0: np.array([1_000.0, 2_000.0])},
    )
    assert mp.subset([0]).product_code is None
    assert mp.subset([0]).channel_code is None


# ---------------------------------------------------------------------------
# value_segmented
# ---------------------------------------------------------------------------

def test_value_segmented_routes_each_mp_to_its_segment():
    """Each mp's BEL should equal the measure() result on its own segment."""
    asmp_high = _flat_asmp(discount=0.03)               # lower discount -> larger BEL
    asmp_low = _flat_asmp(discount=0.10)                # higher discount -> smaller BEL
    basis = {("TERM_A", "GA"): asmp_high, ("TERM_A", "FC"): asmp_low}

    mp = ModelPoints(
        issue_age=np.array([40, 40, 40]),
        level_premium=np.zeros(3),
        term_months=np.array([60, 60, 60]),
        benefits={0: np.array([10_000.0, 10_000.0, 10_000.0])},
        product_code=np.array(["TERM_A", "TERM_A", "TERM_A"]),
        channel_code=np.array(["GA", "FC", "GA"]),
    )
    val = measure(mp, basis, full=False)

    # The two GA mps should match measure() on a single-GA portfolio.
    ga_only = mp.subset([0, 2])
    expected_ga = measure(ga_only, asmp_high, full=False)
    assert np.allclose(val.bel[[0, 2]], expected_ga.bel)
    # The FC mp matches the FC valuation.
    fc_only = mp.subset([1])
    expected_fc = measure(fc_only, asmp_low, full=False)
    assert np.allclose(val.bel[1], expected_fc.bel[0])
    # GA and FC give different per-mp BEL (different discount).
    assert not np.isclose(val.bel[0], val.bel[1])


def test_value_segmented_falls_back_to_single_segment_when_no_product():
    """A single-segment basis works even when product/channel aren't set."""
    asmp = _flat_asmp()
    basis = {("TERM_A", ""): asmp}
    mp = ModelPoints(
        issue_age=np.array([40, 40]),
        level_premium=np.zeros(2),
        term_months=np.array([60, 60]),
        benefits={0: np.array([10_000.0, 20_000.0])},
    )
    val = measure(mp, basis, full=False)
    expected = measure(mp, asmp, full=False)
    assert np.allclose(val.bel, expected.bel)


def test_value_segmented_rejects_multi_segment_basis_without_keys():
    """Multi-segment basis + no product/channel on MPs -> raise."""
    basis = {("TERM_A", "GA"): _flat_asmp(), ("TERM_A", "FC"): _flat_asmp(discount=0.10)}
    mp = ModelPoints(
        issue_age=np.array([40]),
        level_premium=np.zeros(1),
        term_months=np.array([60]),
        benefits={0: np.array([10_000.0])},
    )
    with pytest.raises(ValueError, match="product_code"):
        measure(mp, basis, full=False)


def test_value_segmented_rejects_unknown_segment():
    """A model point pointing at a segment not in basis -> raise."""
    basis = {("TERM_A", "GA"): _flat_asmp()}
    mp = ModelPoints(
        issue_age=np.array([40, 40]),
        level_premium=np.zeros(2),
        term_months=np.array([60, 60]),
        benefits={0: np.array([10_000.0, 10_000.0])},
        product_code=np.array(["TERM_A", "term_b"]),
        channel_code=np.array(["GA", "GA"]),
    )
    with pytest.raises(ValueError, match="not in the basis"):
        measure(mp, basis, full=False)


def test_value_segmented_with_sample_basis():
    """End-to-end smoke -- the bundled sample basis has two segments and
    ``value_segmented`` routes per-mp valuations through it."""
    from fastcashflow import load_sample_calculation_methods
    basis = load_sample_basis()                    # multi-segment sample
    mp = ModelPoints(
        issue_age=np.array([40, 50, 45]),
        level_premium=np.array([50_000.0, 60_000.0, 55_000.0]),
        term_months=np.array([120, 120, 120]),
        benefits={0: np.array([100_000_000.0, 80_000_000.0, 90_000_000.0])},
        product_code=np.array(["TERM_LIFE_A", "TERM_LIFE_A", "TERM_LIFE_A"]),
        channel_code=np.array(["GA", "FC", "GA"]),
        calculation_methods=load_sample_calculation_methods(),
    )
    val = measure(mp, basis, full=False)
    assert val.bel.shape == (3,)
    # GA segment has worse persistency than FC (different LAPSE table) ->
    # the two GA mps should not match the FC mp's pattern.
    expected_ga = measure(mp.subset([0, 2]), basis[("TERM_LIFE_A", "GA")], full=False)
    expected_fc = measure(mp.subset([1]), basis[("TERM_LIFE_A", "FC")], full=False)
    assert np.allclose(val.bel[[0, 2]], expected_ga.bel)
    assert np.allclose(val.bel[1], expected_fc.bel[0])
