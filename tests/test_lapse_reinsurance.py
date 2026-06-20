"""Mass-lapse reinsurance (lapse-XL) -- hand-calc anchors.

Phase A: the loss density (the mass-lapse own-funds strain per unit of excess
lapse) and the excess-of-loss layer mechanics (attachment / detachment /
capacity / recovery).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import lapse_reinsurance as lre
from fastcashflow import ModelPoints
from fastcashflow.engine import inforce_surrender_value, measure

from conftest import make_death_basis, PATTERNS


def _basis(**over):
    kw = dict(mortality_q=0.002, lapse_q=0.004, discount_annual=0.03, mortality_cv=0.10)
    kw.update(over)
    return make_death_basis(**kw)


def _two_contracts(**extra) -> ModelPoints:
    """A profitable (high-premium) and an onerous (low-premium) policy."""
    return ModelPoints(
        issue_age=np.array([40, 40]),
        benefits={"DEATH": np.array([1e8, 1e8])},
        premium=np.array([400_000.0, 50_000.0]),
        term_months=np.array([120, 120]),
        calculation_methods=PATTERNS,
        **extra,
    )


# ---------------------------------------------------------------------------
# Loss density S = sum_MP max(0, surrender_value - BEL)
# ---------------------------------------------------------------------------

def test_loss_density_is_per_policy_max_not_aggregate():
    """S sums max(0, isv - bel) per model point (the Art. 142(6) per-policy
    worst-discontinuance selection), so an onerous MP contributes 0 rather than
    netting against the profitable MP. S therefore exceeds the aggregate form
    (which solvency.mass_lapse uses)."""
    curve = np.full(121, 5_000.0)
    basis = _basis(surrender_value_curve=curve, surrender_value_basis="amount_per_policy")
    mp = _two_contracts()
    bel = measure(mp, basis, full=False).bel
    isv = inforce_surrender_value(mp, basis)
    per_mp_loss = np.maximum(0.0, isv - bel)

    S = lre.loss_density(mp, basis)
    assert np.isclose(S, per_mp_loss.sum())                       # per-policy max
    # one MP is profitable (loss > 0), the other onerous (clamped to 0)
    assert (per_mp_loss > 0).sum() == 1 and (per_mp_loss == 0).sum() == 1
    aggregate = max(0.0, float((isv - bel).sum()))                # the netting form
    assert S > aggregate                                          # per-policy >= aggregate


def test_loss_density_without_surrender_is_lost_business_value():
    """No surrender curve: S is still the lost embedded value of profitable
    business, sum max(0, -BEL) -- a mass lapse loses the future profit even
    when nothing is paid out."""
    basis = _basis()                                              # no surrender_value_curve
    mp = _two_contracts()
    bel = measure(mp, basis, full=False).bel
    assert np.isclose(inforce_surrender_value(mp, basis).sum(), 0.0)
    S = lre.loss_density(mp, basis)
    assert np.isclose(S, np.maximum(0.0, -bel).sum())
    assert S > 0.0                                                # the profitable MP


def test_loss_density_zero_when_all_onerous():
    """A book where surrender is a gain on every MP (onerous, no surrender
    value) has S = 0 -- a mass lapse releases liability, no own-funds loss."""
    basis = _basis()
    mp = ModelPoints(
        issue_age=np.array([40]), benefits={"DEATH": np.array([1e8])},
        premium=np.array([50_000.0]), term_months=np.array([120]),
        calculation_methods=PATTERNS,
    )
    assert measure(mp, basis, full=False).bel[0] > 0.0           # onerous
    assert np.isclose(lre.loss_density(mp, basis), 0.0)


# ---------------------------------------------------------------------------
# LapseXL layer mechanics
# ---------------------------------------------------------------------------

def test_lapsexl_capacity_and_covered_fraction():
    """capacity = detachment - attachment; covered fraction clips the excess
    lapse into the layer."""
    t = lre.LapseXL(0.15, 0.40)
    assert np.isclose(t.capacity, 0.25)
    assert np.isclose(t.covered_fraction(0.10), 0.0)             # below attachment
    assert np.isclose(t.covered_fraction(0.25), 0.10)            # inside: 0.25 - 0.15
    assert np.isclose(t.covered_fraction(0.40), 0.25)            # at detachment: full layer
    assert np.isclose(t.covered_fraction(0.55), 0.25)            # above: capped


def test_lapsexl_recovery_is_loss_in_the_layer():
    """recovery = loss_density x clip(excess - attachment, 0, capacity) -- linear
    in the loss (no cliff)."""
    t = lre.LapseXL(0.15, 0.40)
    S = 8_000_000.0
    assert np.isclose(t.recovery(0.10, S), 0.0)
    assert np.isclose(t.recovery(0.25, S), S * 0.10)
    assert np.isclose(t.recovery(0.40, S), S * 0.25)            # full layer = S x capacity
    assert np.isclose(t.recovery(0.55, S), S * 0.25)            # capped at detachment


def test_lapsexl_rejects_bad_points():
    with pytest.raises(ValueError, match="attachment < detachment"):
        lre.LapseXL(0.40, 0.15)
    with pytest.raises(ValueError, match="attachment < detachment"):
        lre.LapseXL(0.20, 0.20)
    with pytest.raises(ValueError, match="attachment < detachment"):
        lre.LapseXL(-0.1, 0.40)


# ---------------------------------------------------------------------------
# Cedant capital relief (the headline sales number)
# ---------------------------------------------------------------------------

def test_capital_relief_attach_15_detach_40():
    """gross = 0.40 S; recovery = 0.25 S (the 15%-40% layer at the 40% shock);
    net = 0.15 S = attachment layer; relief = recovery."""
    curve = np.full(121, 5_000.0)
    basis = _basis(surrender_value_curve=curve, surrender_value_basis="amount_per_policy")
    mp = _two_contracts()
    treaty = lre.LapseXL(0.15, 0.40)
    r = lre.capital_relief(mp, basis, treaty)

    S = lre.loss_density(mp, basis)
    assert np.isclose(r.loss_density, S)
    assert np.isclose(r.gross_scr, 0.40 * S)
    assert np.isclose(r.recovery, 0.25 * S)
    assert np.isclose(r.net_scr, 0.15 * S)          # retained attachment layer
    assert np.isclose(r.net_scr, treaty.attachment * S)
    assert np.isclose(r.relief, r.recovery)
    assert np.isclose(r.relief, 0.25 * S)


def test_capital_relief_group_pension_70_shock():
    """A 70% group-pension shock with detachment at 40%: the treaty caps at its
    capacity (0.25 S), so net = (0.70 - 0.25) S -- the cedant retains the
    attachment layer plus everything above detachment up to the 70% shock."""
    curve = np.full(121, 5_000.0)
    basis = _basis(surrender_value_curve=curve, surrender_value_basis="amount_per_policy")
    mp = _two_contracts()
    treaty = lre.LapseXL(0.15, 0.40)
    r = lre.capital_relief(mp, basis, treaty, shock=lre.SF_MASS_LAPSE_SHOCK_GROUP_PENSION)

    S = lre.loss_density(mp, basis)
    assert np.isclose(r.gross_scr, 0.70 * S)
    assert np.isclose(r.recovery, 0.25 * S)         # full capacity (0.70 - 0.15 > 0.25)
    assert np.isclose(r.net_scr, (0.70 - 0.25) * S)


# ---------------------------------------------------------------------------
# Counterparty default risk on the reinsurer (DR Art 192/199/200/201)
# ---------------------------------------------------------------------------

def test_credit_quality_step_pd_table():
    """DR Art 199 probability-of-default table, steps 0..6."""
    assert lre.CREDIT_QUALITY_STEP_PD == (
        0.00002, 0.0001, 0.0005, 0.0024, 0.012, 0.042, 0.042)


def test_counterparty_default_lgd_formula():
    """LGD = 0.50 x (recoverables + 0.50 x RM_re) - collateral_factor x collateral
    (DR Art 192(2)). At PD in the first Art-200 case, SCR = 3 x LGD x sqrt(PD(1-PD))."""
    import math
    recoverables, rm_re = 1_000_000.0, 8_000_000.0
    pd = 0.0005                                       # CQS 2, first case
    lgd = 0.50 * (recoverables + 0.50 * rm_re)
    expected = 3.0 * lgd * math.sqrt(pd * (1.0 - pd))
    got = lre.counterparty_default_scr(recoverables, rm_re, pd)
    assert np.isclose(got, expected)


def test_counterparty_default_three_art200_cases():
    """The three Art 200 thresholds on sqrt(PD(1-PD)): CQS3 -> 3 sigma,
    CQS4 -> 5 sigma, CQS5 -> sum LGD."""
    import math
    rm_re = 8_000_000.0
    lgd = 0.50 * 0.50 * rm_re                          # recoverables 0, no collateral

    pd1 = 0.0024                                       # sqrt(pd(1-pd)) ~ 0.0489 <= 0.07
    assert math.sqrt(pd1 * (1 - pd1)) <= 0.07
    assert np.isclose(lre.counterparty_default_scr(0.0, rm_re, pd1),
                      3.0 * lgd * math.sqrt(pd1 * (1 - pd1)))

    pd2 = 0.012                                        # ~0.1089, in (0.07, 0.20]
    assert 0.07 < math.sqrt(pd2 * (1 - pd2)) <= 0.20
    assert np.isclose(lre.counterparty_default_scr(0.0, rm_re, pd2),
                      5.0 * lgd * math.sqrt(pd2 * (1 - pd2)))

    pd3 = 0.042                                        # ~0.2006 > 0.20 -> sum LGD
    assert math.sqrt(pd3 * (1 - pd3)) > 0.20
    assert np.isclose(lre.counterparty_default_scr(0.0, rm_re, pd3), lgd)


def test_counterparty_default_collateral_floors_at_zero():
    """Collateral above the recoverable+mitigation drives LGD (and SCR) to zero."""
    scr = lre.counterparty_default_scr(
        1_000_000.0, 0.0, 0.0005, collateral=10_000_000.0, collateral_factor=1.0)
    assert scr == 0.0
