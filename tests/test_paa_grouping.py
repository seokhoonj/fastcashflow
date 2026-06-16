"""Grouping a PAA (premium-allocation) measurement to the IFRS 17 unit of account.

The PAA has no CSM (paragraphs 53-59): the LRC, insurance revenue, service
expense and LIC are undiscounted and additive, so they sum within a group. The
only non-linear part is the onerous loss (paragraph 57), re-derived on the
group's aggregate fulfilment cash flows. The underlying PAA measure is
hand-validated in test_paa.py.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import ModelPoints, PAAMeasurement, group, group_of_contracts
from fastcashflow.paa import measure as measure_paa
from conftest import PATTERNS, make_death_basis


def _basis():
    return make_death_basis(
        mortality_q     = 0.002,
        lapse_q         = 0.005,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = 0.10,
    )


def _two_contracts(**extra) -> ModelPoints:
    """Two short-coverage policies: a high-premium (profitable) and a low (onerous)."""
    return ModelPoints(
        issue_age=np.array([40, 40]),
        benefits={"DEATH": np.array([1e8, 1e8])},
        premium=np.array([400_000.0, 50_000.0]),
        term_months=np.array([12, 12]),
        calculation_methods=PATTERNS,
        **extra,
    )


def test_paa_group_per_mp_reproduces_original():
    """Each MP in its own group reproduces the per-MP measurement."""
    m = measure_paa(_two_contracts(), _basis())
    g = group(m, np.arange(2))
    assert isinstance(g, PAAMeasurement)
    np.testing.assert_allclose(g.lrc, m.lrc, rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.fcf, m.fcf, rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.loss_component, m.loss_component, rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.revenue, m.revenue, rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.service_expense, m.service_expense, rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.lic_path, m.lic_path, rtol=0, atol=1e-6)


def test_paa_group_is_additive_and_refloors_loss():
    """LRC / revenue / service expense sum; the onerous loss re-floors on the aggregate."""
    m = measure_paa(_two_contracts(), _basis())
    g = group(m, np.zeros(2, dtype=int))                 # one group
    assert g.lrc.shape[0] == 1
    assert np.isclose(g.lrc[0], m.lrc.sum())
    assert np.isclose(g.revenue.sum(), m.revenue.sum())
    assert np.isclose(g.service_expense.sum(), m.service_expense.sum())
    assert np.isclose(g.lic_path[:, 0].sum(), m.lic_path[:, 0].sum())
    # the onerous loss is re-derived on the aggregate fulfilment cash flows
    assert np.isclose(g.loss_component[0], max(0.0, m.fcf.sum()))


def test_paa_group_pooling_nets_the_loss():
    m = measure_paa(_two_contracts(), _basis())
    split = group(m, np.array([0, 1]))
    pooled = group(m, np.zeros(2, dtype=int))
    assert split.loss_component.sum() > 0.0              # one contract onerous standalone
    assert pooled.loss_component.sum() < split.loss_component.sum()


def test_paa_group_service_result_holds():
    """The service_result property works on the grouped measurement."""
    m = measure_paa(_two_contracts(), _basis())
    g = group(m, np.zeros(2, dtype=int))
    np.testing.assert_allclose(g.service_result, g.revenue - g.service_expense)


def test_paa_group_of_contracts_onerous_split():
    mp = _two_contracts(product=np.array(["MED", "MED"]))
    m = measure_paa(mp, _basis())
    g = group_of_contracts(m)                            # product x cohort x onerous
    assert isinstance(g, PAAMeasurement)
    assert g.lrc.shape[0] == 2                           # one onerous, one remaining
    assert g.loss_component.sum() > 0.0
    assert g.group_labels is not None and g.group_labels.shape[0] == 2


def test_paa_group_by_axis_name():
    mp = _two_contracts(product=np.array(["MA", "MB"]))
    m = measure_paa(mp, _basis())
    assert group(m, "product").lrc.shape[0] == 2
