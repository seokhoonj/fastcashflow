"""Sanity tests for ``show_trace`` -- the per-mp calculation walk."""
import io

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.assumptions import Assumptions
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.trace import show_trace


def _basis():
    return fcf.load_sample_assumptions()


def _portfolio():
    return fcf.load_sample_model_points()


def test_show_trace_renders_all_sections():
    """The eight headline tree sections all appear in the output."""
    buf = io.StringIO()
    show_trace(0, _portfolio(), _basis(), file=buf)
    text = buf.getvalue()
    for section in (
        "Assumptions (segment-level)",
        "Coverages",
        "Rates (annual",
        "Cash flows",
        "Discount factors",
        "BEL roll-forward",
        "CSM roll-forward",
        "Final",
    ):
        assert section in text, f"missing section: {section}"


def test_show_trace_routes_dict_basis_by_segment():
    """Passing the read_assumptions dict picks the right segment from
    the model point's (product, channel)."""
    mp = _portfolio()
    basis = _basis()
    buf = io.StringIO()
    show_trace(0, mp, basis, file=buf)
    text = buf.getvalue()
    seg = f"({mp.product[0]}/{mp.channel[0]}"
    assert seg in text


def test_show_trace_accepts_single_assumptions():
    """A plain :class:`Assumptions` (not a dict) bypasses the segment
    lookup and is used directly."""
    mp = _portfolio()
    asmp = _basis()[(str(mp.product[0]), str(mp.channel[0]))]
    buf = io.StringIO()
    show_trace(0, mp, asmp, file=buf)
    assert "Assumptions (segment-level)" in buf.getvalue()


def test_show_trace_bel_and_ra_agree_with_measure():
    """The headline numbers printed in the tree match :func:`measure`
    on the same portfolio for the same row -- the trace is just a view,
    not a recalculation."""
    mp = _portfolio()
    asmp = _basis()[(str(mp.product[0]), str(mp.channel[0]))]
    m = fcf.measure(mp.subset([0]), asmp)
    buf = io.StringIO()
    show_trace(0, mp, asmp, file=buf)
    text = buf.getvalue()
    assert f"{m.bel[0, 0]:,.2f}" in text
    assert f"{m.ra[0, 0]:,.2f}" in text


def test_show_trace_rejects_out_of_range_index():
    mp = _portfolio()
    with pytest.raises(IndexError, match="mp_index"):
        show_trace(mp.n_mp, mp, _basis(), file=io.StringIO())


def test_show_trace_dict_basis_requires_segment_columns():
    """A dict basis cannot be routed when model_points has no product /
    channel columns."""
    bare = ModelPoints(
        issue_age=np.array([35.0]),
        level_premium=np.array([50_000.0]),
        term_months=np.array([120]),
        death_benefit=np.array([100_000_000.0]),
    )
    with pytest.raises(ValueError, match="product / channel"):
        show_trace(0, bare, _basis(), file=io.StringIO())


def test_show_trace_dict_basis_unknown_segment_raises():
    """An unmapped (product, channel) is flagged with available keys."""
    mp = _portfolio()
    partial = {k: v for k, v in _basis().items() if k[0] != mp.product[0]}
    if partial:                           # only meaningful when dict is shrinkable
        with pytest.raises(KeyError, match="no assumptions for segment"):
            show_trace(0, mp, partial, file=io.StringIO())
