"""Grouping a VFA (account-value) measurement to the IFRS 17 unit of account.

``group`` and ``group_of_contracts`` dispatch on the measurement type. For VFA
the additive parts (BEL, RA, variable fee, time value) sum within a group; the
CSM and loss component are re-derived on the aggregate -- the floor on
``Sigma BEL + Sigma RA + Sigma time_value`` -- with the CSM accreted at the
underlying-items return (the single VFA curve). The underlying VFA measure is
hand-validated in test_vfa.py; here the grouping invariants are checked.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import ExpenseItem, ModelPoints, VFAMeasurement, group, group_of_contracts
from conftest import make_death_assumptions


def _vfa_basis(**overrides):
    kw = dict(
        mortality_q       = 0.002,
        lapse_q           = 0.004,
        discount_annual   = 0.03,
        ra_confidence     = 0.75,
        mortality_cv      = 0.10,
        investment_return = 0.06,
        fund_fee          = 0.015,
        # a fixed acquisition cost makes the tiny-account-value policy onerous
        # (expense-dominated) and the large one profitable (fee-dominated)
        expense_items     = (ExpenseItem("acquisition", "alpha_fixed", 5_000_000.0),),
    )
    kw.update(overrides)
    return make_death_assumptions(**kw)


def _two_vfa(**extra) -> ModelPoints:
    """Two account-value model points: a large (profitable) and a tiny (onerous)."""
    n = 2
    return ModelPoints(
        issue_age=np.array([40, 40]),
        premium=np.array([0.0, 0.0]),
        term_months=np.array([60, 60]),
        account_value=np.array([1e8, 1e6]),
        minimum_crediting_rate=np.zeros(n),
        minimum_death_benefit=np.zeros(n),
        minimum_accumulation_benefit=np.zeros(n),
        **extra,
    )


def test_vfa_group_is_additive_and_refloors():
    """One group: additive parts sum; CSM/loss re-floor on the aggregate."""
    m = fcf.vfa.measure(_two_vfa(), _vfa_basis())
    g = group(m, np.zeros(2, dtype=int))                 # both into one group

    assert isinstance(g, VFAMeasurement)
    assert g.bel.shape[0] == 1
    # additive parts
    assert np.isclose(g.bel[0], m.bel.sum())
    assert np.isclose(g.ra[0], m.ra.sum())
    assert np.isclose(g.variable_fee[0], m.variable_fee.sum())
    assert np.isclose(g.time_value[0], m.time_value.sum())
    # CSM and loss re-derived on the aggregate fulfilment cash flows
    fcf0 = m.bel.sum() + m.ra.sum() + m.time_value.sum()
    assert np.isclose(g.csm[0], max(0.0, -fcf0))
    assert np.isclose(g.loss_component[0], max(0.0, fcf0))


def test_vfa_group_csm_path_reconciles():
    """The grouped CSM trajectory closes: csm[t+1] = csm[t] + accretion - release."""
    m = fcf.vfa.measure(_two_vfa(), _vfa_basis())
    g = group(m, np.zeros(2, dtype=int))
    step = g.csm_path[:, :-1] + g.csm_accretion - g.csm_release
    np.testing.assert_allclose(g.csm_path[:, 1:], step, rtol=0, atol=1e-6)


def test_vfa_group_pooling_nets_the_loss():
    """Pooling an onerous and a profitable contract nets the loss (floor)."""
    m = fcf.vfa.measure(_two_vfa(), _vfa_basis())
    split = group(m, np.array([0, 1]))                   # two groups
    pooled = group(m, np.zeros(2, dtype=int))            # one group

    assert split.bel.shape[0] == 2
    # the split per-group additive parts match the per-MP figures
    np.testing.assert_allclose(np.sort(split.bel), np.sort(m.bel), rtol=0, atol=1e-6)
    # one contract is onerous standalone, so the split carries a real loss...
    assert split.loss_component.sum() > 0.0
    # ...which pooling nets against the profitable contract's surplus
    assert pooled.loss_component.sum() <= split.loss_component.sum() + 1e-6
    assert pooled.loss_component.sum() < split.loss_component.sum()


def test_vfa_group_of_contracts_preset():
    """group_of_contracts on a VFA measurement: product x cohort x onerous."""
    mp = _two_vfa(product_code=np.array(["VA", "VA"]))
    m = fcf.vfa.measure(mp, _vfa_basis())
    g = group_of_contracts(m)                            # one product, one cohort, prof derived
    assert isinstance(g, VFAMeasurement)
    assert g.bel.shape[0] == 2                           # split only by onerous / remaining
    assert g.loss_component.sum() > 0.0
    assert g.group_labels is not None and g.group_labels.shape[0] == 2


def test_vfa_group_by_axis_name():
    """A VFA measurement carries its model points, so axis names resolve."""
    mp = _two_vfa(product_code=np.array(["VA", "VB"]))
    m = fcf.vfa.measure(mp, _vfa_basis())
    assert group(m, "product_code").bel.shape[0] == 2    # two products
