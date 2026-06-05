"""Generalised segment routing -- measure(segment_by=[...]) for N-tuple bases.

The assumption-routing key defaults to (product, channel) but can be
any axes (resolved via ModelPoints.axis), so a basis can vary by, e.g.,
(product, channel, risk_class). Cost scales with the number of distinct
segments, not the number of axes.
"""
import numpy as np
import pytest

from fastcashflow import ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, make_death_basis


def _b(lapse_q):
    return make_death_basis(mortality_q=0.002, lapse_q=lapse_q,
                                  discount_annual=0.03, ra_confidence=0.75,
                                  mortality_cv=0.10)


def _mp():
    return ModelPoints(
        issue_age=np.full(4, 40), benefits={0: np.full(4, 1e8)},
        premium=np.full(4, 200_000.0), term_months=np.full(4, 120),
        calculation_methods=PATTERNS,
        product=np.array(["TL", "TL", "TL", "TL"]),
        channel=np.array(["GA", "GA", "TM", "TM"]),
        attributes={"risk_class": np.array(["A", "B", "A", "B"])},
    )


def _basis_3():
    return {
        ("TL", "GA", "A"): _b(0.05), ("TL", "GA", "B"): _b(0.10),
        ("TL", "TM", "A"): _b(0.08), ("TL", "TM", "B"): _b(0.15),
    }


def test_segment_by_three_axes_routes_per_combo():
    """(product, channel, risk_class) routes each combo to its own basis."""
    m = measure(_mp(), _basis_3(),
                segment_by=["product", "channel", "risk_class"])
    assert len({round(float(x), 3) for x in m.bel}) == 4   # all 4 combos differ
    assert not np.isclose(m.bel[0], m.bel[1])              # risk_class A vs B


def test_segment_by_default_is_product_channel():
    """Omitting segment_by keeps the (product, channel) default."""
    basis2 = {("TL", "GA"): _b(0.05), ("TL", "TM"): _b(0.08)}
    m = measure(_mp(), basis2)              # risk_class ignored
    assert np.isclose(m.bel[0], m.bel[1])  # both GA -> one basis
    assert np.isclose(m.bel[2], m.bel[3])  # both TM -> one basis


def test_segment_by_full_matches_fast():
    """The full=True segmented path routes the same axes as the fast path."""
    by = ["product", "channel", "risk_class"]
    fast = measure(_mp(), _basis_3(), full=False, segment_by=by)
    full = measure(_mp(), _basis_3(), full=True, segment_by=by)
    assert np.allclose(fast.bel, full.bel)
    assert np.allclose(fast.csm, full.csm)


def test_segment_by_single_basis_fallback_when_axis_unset():
    """A one-entry basis applies to everything when the axes are not set."""
    mp = ModelPoints(issue_age=np.full(2, 40), benefits={0: np.full(2, 1e8)},
                     premium=np.full(2, 200_000.0), term_months=np.full(2, 120),
                     calculation_methods=PATTERNS)        # no product / channel
    assert measure(mp, {("only",): _b(0.05)}).bel.shape[0] == 2


def test_segment_by_unknown_segment_rejected():
    incomplete = {("TL", "GA", "A"): _b(0.05)}            # missing 3 combos
    with pytest.raises(ValueError, match="not in the"):
        measure(_mp(), incomplete,
                segment_by=["product", "channel", "risk_class"])
