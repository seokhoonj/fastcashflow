"""Plain-annuity payout forms -- deferred start / term-certain / guaranteed period.

These three IntArray ModelPoints fields shape the ``annuity_payment`` income:

* ``annuity_start_months``     -- deferred payout start (0 = from inception)
* ``annuity_term_months``      -- term-certain payout count (0 = unlimited/life)
* ``annuity_guarantee_months`` -- certain-and-life guarantee window (0 = pure life)

This module holds the input-layer validation (the behaviour hand-calcs land with
each form's kernel stage). All three default to 0, so an existing book is a no-op.
"""
import numpy as np
import pytest

from fastcashflow import ModelPoints


def _annuity_mp(**overrides):
    kw = dict(issue_age=40, premium=0.0, term_months=120, annuity_payment=10.0)
    kw.update(overrides)
    return ModelPoints.single(**kw)


def test_annuity_form_fields_default_zero():
    """An ordinary book leaves all three forms at 0 (the historical behaviour)."""
    mp = _annuity_mp()
    assert mp.annuity_start_months[0] == 0
    assert mp.annuity_term_months[0] == 0
    assert mp.annuity_guarantee_months[0] == 0


def test_annuity_form_fields_set_and_subset():
    """The fields round-trip through ``.single`` and survive ``subset``."""
    mp = _annuity_mp(annuity_start_months=12, annuity_term_months=60,
                     annuity_guarantee_months=24)
    assert (mp.annuity_start_months[0], mp.annuity_term_months[0],
            mp.annuity_guarantee_months[0]) == (12, 60, 24)
    sub = mp.subset([0])
    assert sub.annuity_start_months[0] == 12
    assert sub.annuity_term_months[0] == 60
    assert sub.annuity_guarantee_months[0] == 24


def test_negative_form_field_raises():
    for field in ("annuity_start_months", "annuity_term_months",
                  "annuity_guarantee_months"):
        with pytest.raises(ValueError, match=">= 0"):
            _annuity_mp(**{field: -1})


def test_guarantee_exceeds_term_raises():
    """The guarantee window cannot outlast the payout term."""
    with pytest.raises(ValueError, match="guarantee_months must be <="):
        _annuity_mp(annuity_term_months=12, annuity_guarantee_months=24)


def test_start_at_or_past_boundary_raises():
    """A deferred start must leave at least one payout month in the horizon."""
    with pytest.raises(ValueError, match="start_months must be <"):
        _annuity_mp(term_months=120, annuity_start_months=120)


def test_new_form_with_annuitization_raises():
    """The plain-annuity forms are not yet supported with UL annuitization."""
    with pytest.raises(NotImplementedError, match="annuitization"):
        ModelPoints.single(
            issue_age=40, premium=0.0, term_months=120, account_value=1000.0,
            premium_term_months=0, annuitization_months=60,
            annuitization_rate=0.004, annuity_guarantee_months=24)
