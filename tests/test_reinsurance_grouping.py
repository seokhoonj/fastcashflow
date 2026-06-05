"""Grouping a reinsurance-held measurement to the IFRS 17 unit of account.

Reinsurance held has no loss component and no floor (paragraph 65): the CSM is
the net cost or gain, ``csm0 = -(BEL - RA)``, which is linear -- so the grouped
CSM is the sum of the contract CSMs, and any grouping preserves the total.
``group_of_contracts`` splits profitability by the net gain at initial
recognition (paragraph 61, ``csm > 0``) instead of the onerous test. The
underlying reinsurance measure is hand-validated in test_reinsurance.py.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import ModelPoints, ReinsuranceMeasurement, group, group_of_contracts
from conftest import PATTERNS, make_death_assumptions


def _basis():
    return make_death_assumptions(
        mortality_q     = 0.002,
        lapse_q         = 0.005,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = 0.10,
    )


def _two_reins(**extra) -> ModelPoints:
    """Two cedant policies with very different premiums -- one net cost, one net gain."""
    return ModelPoints(
        issue_age=np.array([40, 40]),
        premium=np.array([500_000.0, 30_000.0]),
        term_months=np.array([60, 60]),
        benefits={0: np.array([1e8, 1e8])},
        calculation_methods=PATTERNS,
        **extra,
    )


def _measure(**extra):
    return fcf.reinsurance.measure(
        _two_reins(**extra), _basis(), fcf.reinsurance.QuotaShare(cession=0.4)
    )


def test_reinsurance_group_per_mp_reproduces_original():
    """Each MP in its own group reproduces the per-MP measurement (rate + additivity)."""
    m = _measure()
    g = group(m, np.arange(2))                      # one group per model point
    assert isinstance(g, ReinsuranceMeasurement)
    np.testing.assert_allclose(g.bel, m.bel, rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.ra, m.ra, rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.csm, m.csm, rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.csm_path, m.csm_path, rtol=0, atol=1e-6)


def test_reinsurance_group_csm_is_additive_no_floor():
    """No floor (paragraph 65): the grouped CSM is the sum of the contract CSMs."""
    m = _measure()
    pooled = group(m, np.zeros(2, dtype=int))       # one group
    split = group(m, np.array([0, 1]))              # two groups

    assert pooled.bel.shape[0] == 1 and split.bel.shape[0] == 2
    # additive parts
    assert np.isclose(pooled.bel[0], m.bel.sum())
    assert np.isclose(pooled.ra[0], m.ra.sum())
    # CSM is linear -- pooling does not net (no floor), totals match the per-MP sum
    assert np.isclose(pooled.csm[0], m.csm.sum())
    assert np.isclose(pooled.csm.sum(), split.csm.sum())
    assert np.isclose(split.csm.sum(), m.csm.sum())


def test_reinsurance_group_csm_path_reconciles():
    m = _measure()
    g = group(m, np.zeros(2, dtype=int))
    step = g.csm_path[:, :-1] + g.csm_accretion - g.csm_release
    np.testing.assert_allclose(g.csm_path[:, 1:], step, rtol=0, atol=1e-6)


def test_reinsurance_group_of_contracts_net_gain_split():
    """group_of_contracts splits by the net gain at initial recognition (paragraph 61)."""
    mp = _two_reins(product=np.array(["RE", "RE"]))
    m = fcf.reinsurance.measure(mp, _basis(), fcf.reinsurance.QuotaShare(cession=0.4))
    g = group_of_contracts(m)
    assert isinstance(g, ReinsuranceMeasurement)
    # one product, one cohort -> groups split only by the net-gain classification
    n_classes = np.unique(m.csm > 0.0).size
    assert g.bel.shape[0] == n_classes
    assert n_classes == 2                               # the fixture has one gain, one cost
    assert np.isclose(g.csm.sum(), m.csm.sum())         # total CSM preserved (linear)
    assert g.group_labels is not None and g.group_labels.shape[0] == g.bel.shape[0]


def test_reinsurance_group_by_axis_name():
    mp = _two_reins(product=np.array(["RA", "RB"]))
    m = fcf.reinsurance.measure(mp, _basis(), fcf.reinsurance.QuotaShare(cession=0.4))
    assert group(m, "product").bel.shape[0] == 2   # two products
