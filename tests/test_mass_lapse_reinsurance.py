"""Mass-lapse reinsurance (lapse-XL) -- hand-calc anchors.

Phase A: the loss density (the mass-lapse own-funds strain per unit of excess
lapse) and the excess-of-loss layer mechanics (attachment / detachment /
capacity / recovery).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import mass_lapse_reinsurance as lre
from fastcashflow import solvency as sv
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


# ---------------------------------------------------------------------------
# Cedant solvency relief (Phase B2 -- full diversified picture)
# ---------------------------------------------------------------------------

def _mass_biting_book():
    """A profitable book with a high surrender value and a low base lapse, so the
    mass-lapse stress is the biting lapse leg (the +/-50% gradual stresses are
    small)."""
    mp = ModelPoints(
        issue_age=np.array([45, 50]), benefits={"DEATH": np.array([1e6, 1e6])},
        premium=np.array([300_000.0, 250_000.0]), term_months=np.array([120, 120]),
        count=np.array([2_000.0, 1_500.0]), calculation_methods=PATTERNS)
    basis = make_death_basis(
        mortality_q=0.001, lapse_q=0.008, discount_annual=0.03, mortality_cv=0.10,
        surrender_value_curve=np.full(121, 120_000.0),
        surrender_value_basis="amount_per_policy")
    return mp, basis


def test_cedant_relief_lapse_net_floored_by_next_leg():
    """The treaty cuts only the mass leg; the net lapse capital cannot fall below
    the next-biting lapse leg (lapse up / down). Here mass bites gross, and after
    the treaty the up/down floor bites instead -- exactly the standard-formula
    'lapse up/down may bite instead' effect."""
    mp, basis = _mass_biting_book()
    r = lre.cedant_solvency_relief(mp, basis, lre.LapseXL(0.15, 0.40),
                                   regime=sv.SOLVENCY2,
                                   reinsurer_pd=lre.CREDIT_QUALITY_STEP_PD[2])
    # mass is the gross biting leg; net mass is below the up/down floor
    assert np.isclose(r.lapse_gross_scr, r.mass_gross_scr)
    assert r.mass_net_scr < r.lapse_net_scr            # floored by up/down
    assert r.lapse_net_scr > 0.0
    assert r.lapse_relief > 0.0


def test_cedant_relief_reaggregates_life_module():
    """insurance_gross / net are the life module re-aggregated with the gross /
    net lapse capital (the other sub-risks from one required_capital run)."""
    mp, basis = _mass_biting_book()
    treaty = lre.LapseXL(0.15, 0.40)
    r = lre.cedant_solvency_relief(mp, basis, treaty, regime=sv.SOLVENCY2,
                                   reinsurer_pd=lre.CREDIT_QUALITY_STEP_PD[2])
    caps = dict(sv.required_capital(mp, basis, regime=sv.SOLVENCY2).sub_risk_capital)
    exp_gross = sv.aggregate({**caps, "lapse": r.lapse_gross_scr}, sv.SOLVENCY2)
    exp_net = sv.aggregate({**caps, "lapse": r.lapse_net_scr}, sv.SOLVENCY2)
    assert np.isclose(r.insurance_gross_scr, exp_gross)
    assert np.isclose(r.insurance_net_scr, exp_net)
    # diversification: the module relief does not exceed the standalone lapse relief
    assert r.insurance_relief <= r.lapse_relief + 1.0


def test_cedant_relief_counterparty_default_on_module_relief():
    """The counterparty-default add-back uses the diversified insurance relief as
    the risk-mitigating effect (Art 192 RM_re)."""
    mp, basis = _mass_biting_book()
    pd = lre.CREDIT_QUALITY_STEP_PD[2]
    r = lre.cedant_solvency_relief(mp, basis, lre.LapseXL(0.15, 0.40),
                                   regime=sv.SOLVENCY2, reinsurer_pd=pd)
    assert np.isclose(
        r.counterparty_default,
        lre.counterparty_default_scr(0.0, r.insurance_relief, pd))
    assert r.counterparty_default > 0.0


def test_cedant_relief_total_composition():
    """net SCR benefit = insurance relief - counterparty default; total benefit
    adds the risk-margin relief."""
    mp, basis = _mass_biting_book()
    r = lre.cedant_solvency_relief(mp, basis, lre.LapseXL(0.15, 0.40),
                                   regime=sv.SOLVENCY2,
                                   reinsurer_pd=lre.CREDIT_QUALITY_STEP_PD[2])
    assert np.isclose(r.net_scr_benefit, r.insurance_relief - r.counterparty_default)
    assert np.isclose(r.risk_margin_relief, r.risk_margin_gross - r.risk_margin_net)
    assert np.isclose(r.total_benefit, r.net_scr_benefit + r.risk_margin_relief)
    assert r.total_benefit > 0.0


# ---------------------------------------------------------------------------
# Lapse tail distribution F(L) -- the reinsurer baseline (Phase D)
# ---------------------------------------------------------------------------

def test_lapse_distribution_reproduces_anchors():
    """from_anchors calibrates the lognormal so the two public tail anchors are
    reproduced: P(L > 15%) = 1/30, P(L > 40%) = 1/200."""
    f = lre.LapseTailDistribution.from_anchors()
    assert np.isclose(f.survival(0.15), 1.0 / 30.0, atol=1e-9)
    assert np.isclose(f.survival(0.40), 1.0 / 200.0, atol=1e-9)
    # survival is monotone decreasing, and 0 lapse is certain to be exceeded
    assert f.survival(0.0) == 1.0
    assert f.survival(0.05) > f.survival(0.15) > f.survival(0.40) > f.survival(0.80)


def test_expected_layer_equals_survival_integral():
    """The pluggable-interface identity: E[clip(L-a,0,b-a)] = integral_a^b
    survival(x) dx. Validates the closed form against survival-only integration
    (any drop-in F(L) can price via survival alone)."""
    f = lre.LapseTailDistribution.from_anchors()
    a, b = 0.15, 0.40
    grid = np.linspace(a, b, 200_001)
    surv = np.array([f.survival(x) for x in grid])
    integral = np.trapezoid(surv, grid)
    assert np.isclose(f.expected_layer(a, b), integral, rtol=1e-4)


def test_lapse_distribution_rejects_bad_anchors():
    with pytest.raises(ValueError, match="a higher lapse is rarer"):
        lre.LapseTailDistribution.from_anchors((0.40, 1/200), (0.15, 1/30))  # swapped


def test_value_at_risk_returns_sf_anchor():
    """VaR_99.5(L) reproduces the 40% standard-formula stress (the calibration
    upper anchor)."""
    f = lre.LapseTailDistribution.from_anchors()
    assert np.isclose(f.value_at_risk(0.995), 0.40, atol=1e-9)
    assert np.isclose(f.value_at_risk(1.0 - 1/30), 0.15, atol=1e-9)


# ---------------------------------------------------------------------------
# Reinsurer pricing (Phase D)
# ---------------------------------------------------------------------------

def test_price_treaty_components():
    """expected_recovery = S x E[layer]; capital = (S x covered(VaR) - E[rec]) x
    div; premium = E[rec] + coc x capital."""
    S = 10_000_000_000.0
    treaty = lre.LapseXL(0.15, 0.40)
    f = lre.LapseTailDistribution.from_anchors()
    coc, div = 0.06, 1.0
    p = lre.price_treaty(S, treaty, f, cost_of_capital=coc, diversification_factor=div)

    exp_rec = S * f.expected_layer(0.15, 0.40)
    var_lapse = f.value_at_risk(0.995)                       # = 0.40
    capital = (S * treaty.covered_fraction(var_lapse) - exp_rec) * div
    assert np.isclose(p.expected_recovery, exp_rec)
    assert np.isclose(p.capacity_at_risk, S * treaty.capacity)
    assert np.isclose(p.capital, capital)
    assert np.isclose(p.premium, exp_rec + coc * capital)
    assert np.isclose(p.expected_profit, coc * capital)
    # at the 40% VaR the layer is fully covered -> capacity saturates the VaR loss
    assert np.isclose(S * treaty.covered_fraction(var_lapse), p.capacity_at_risk)


def test_price_treaty_diversification_scales_the_load():
    """The diversification factor scales the assumed capital (and so the cost-of-
    capital load) linearly; the expected recovery is unchanged."""
    S = 10_000_000_000.0
    treaty = lre.LapseXL(0.15, 0.40)
    f = lre.LapseTailDistribution.from_anchors()
    full = lre.price_treaty(S, treaty, f, diversification_factor=1.0)
    quarter = lre.price_treaty(S, treaty, f, diversification_factor=0.25)
    assert np.isclose(quarter.capital, 0.25 * full.capital)
    assert np.isclose(quarter.expected_recovery, full.expected_recovery)
    assert np.isclose(quarter.expected_profit, 0.25 * full.expected_profit)
    assert quarter.premium < full.premium


def test_price_treaty_is_distribution_pluggable():
    """price_treaty depends on the distribution only through expected_layer and
    value_at_risk -- any object providing them prices the treaty (the drop-in
    F(L) contract)."""
    class FlatF:                                              # a trivial stand-in F(L)
        def expected_layer(self, a, b):
            return 0.02
        def value_at_risk(self, q):
            return 0.40
    S = 1_000_000_000.0
    treaty = lre.LapseXL(0.15, 0.40)
    p = lre.price_treaty(S, treaty, FlatF(), cost_of_capital=0.06)
    assert np.isclose(p.expected_recovery, S * 0.02)
    assert np.isclose(p.capital, S * treaty.capacity - S * 0.02)
    assert np.isclose(p.premium, S * 0.02 + 0.06 * (S * treaty.capacity - S * 0.02))


# ---------------------------------------------------------------------------
# Measurement period -- the time axis (EIOPA Annex 3.8 / 3.9 / footnote 10)
# ---------------------------------------------------------------------------

S_TEST = 10_000_000.0
TREATY = lre.LapseXL(0.20, 0.40)            # attach 20%, detach 40% over BE, capacity 0.20


def test_reset_window_claims_per_period():
    """reset windows accumulate within each period and sum the claims. A single
    24-month event with 30% in year 1 alone triggers the 20% attachment."""
    incr = np.zeros(36)
    incr[0] = 0.30                            # 30% excess lapse in month 0
    m = lre.MeasurementPeriod(months=12, mode="reset")
    claim = lre.windowed_claim(incr, TREATY, S_TEST, m, duration_months=36)
    # only window 0 sees the 0.30 -> clip(0.30 - 0.20, 0, 0.20) = 0.10
    assert np.isclose(claim, S_TEST * 0.10)


def test_reset_12mo_misses_multiyear_event():
    """EIOPA Annex 3.9: a 2-year event of 20% per year never reaches a 20%
    attachment in any single 12-month window, so a 12-month reset treaty pays
    nothing."""
    incr = np.zeros(24)
    incr[0] = 0.20                            # year 1: 20%
    incr[12] = 0.20                           # year 2: 20%
    m12 = lre.MeasurementPeriod(months=12, mode="reset")
    assert np.isclose(lre.windowed_claim(incr, TREATY, S_TEST, m12, duration_months=24), 0.0)


def test_longer_window_catches_multiyear_event():
    """The same 20% + 20% event, measured over a 24-month window, accumulates to
    40% and pays the full capacity (40% - 20% = 20%, capped at the 0.20 layer)."""
    incr = np.zeros(24)
    incr[0] = 0.20
    incr[12] = 0.20
    m24 = lre.MeasurementPeriod(months=24, mode="reset")
    claim = lre.windowed_claim(incr, TREATY, S_TEST, m24, duration_months=24)
    assert np.isclose(claim, S_TEST * 0.20)   # clip(0.40 - 0.20, 0, 0.20) = 0.20


def test_rolling_high_water_mark_pays_once():
    """EIOPA footnote 10: a single event falling in several overlapping windows
    is paid once (the high-water mark), not summed across the windows."""
    incr = np.zeros(36)
    incr[12] = 0.30                           # one event at month 12
    roll = lre.MeasurementPeriod(months=12, mode="rolling", step_months=3)
    claim = lre.windowed_claim(incr, TREATY, S_TEST, roll, duration_months=36)
    # windows starting at 3,6,9,12 all contain month 12 -> each claims S*0.10;
    # high-water mark pays the max once, not 4x
    assert np.isclose(claim, S_TEST * 0.10)
    # a naive reset over the same path would also see it in one window here
    reset = lre.MeasurementPeriod(months=12, mode="reset")
    assert np.isclose(lre.windowed_claim(incr, TREATY, S_TEST, reset, duration_months=36),
                      S_TEST * 0.10)


def test_measurement_period_validation():
    with pytest.raises(ValueError, match="mode must be"):
        lre.MeasurementPeriod(months=12, mode="sliding")
    with pytest.raises(ValueError, match="months must be positive"):
        lre.MeasurementPeriod(months=0)


@pytest.mark.parametrize("regime, regime_shock", [
    (sv.SOLVENCY2, 0.40),                         # DR Art 142(6)(b)
    (sv.KICS, 0.30),                              # K-ICS handbook
])
def test_cedant_relief_uses_regime_mass_lapse_shock(regime, regime_shock):
    """The mass-lapse shock defaults to the regime's own fraction -- 40% under
    Solvency II, 30% under K-ICS -- so the same call works for both regimes.
    The solvency module carries both calibrations."""
    mp, basis = _mass_biting_book()
    treaty = lre.LapseXL(0.10, regime_shock)      # detach at the regime shock
    r = lre.cedant_solvency_relief(mp, basis, treaty, regime=regime,
                                   reinsurer_pd=lre.CREDIT_QUALITY_STEP_PD[2])
    assert np.isclose(r.mass_gross_scr, regime_shock * r.loss_density)
    assert r.lapse_gross_scr >= r.mass_gross_scr  # mass is a (the) biting leg here
    assert r.total_benefit > 0.0


def test_cedant_relief_shock_override():
    """An explicit shock overrides the regime default."""
    mp, basis = _mass_biting_book()
    r = lre.cedant_solvency_relief(mp, basis, lre.LapseXL(0.10, 0.35),
                                   regime=sv.KICS,
                                   reinsurer_pd=lre.CREDIT_QUALITY_STEP_PD[2],
                                   shock=0.35)
    assert np.isclose(r.mass_gross_scr, 0.35 * r.loss_density)


def test_kics_tail_distribution_anchor():
    """A K-ICS tail distribution calibrates to the 30% / 1-in-200 anchor (the
    K-ICS mass-lapse stress), so VaR_99.5 returns 30%, not the Solvency II 40%."""
    fk = lre.LapseTailDistribution.from_anchors((0.10, 1 / 30), (0.30, 1 / 200))
    assert np.isclose(fk.value_at_risk(0.995), 0.30, atol=1e-9)
    assert np.isclose(fk.survival(0.30), 1 / 200, atol=1e-9)


def test_cedant_relief_zero_when_updown_dominates():
    """When lapse up/down already bites harder than mass, cutting the mass leg
    gives no lapse relief -- the treaty does not help."""
    mp = ModelPoints(
        issue_age=np.array([45]), benefits={"DEATH": np.array([1e6])},
        premium=np.array([300_000.0]), term_months=np.array([120]),
        count=np.array([2_000.0]), calculation_methods=PATTERNS)
    basis = make_death_basis(                          # high base lapse -> up/down dominates
        mortality_q=0.001, lapse_q=0.05, discount_annual=0.03, mortality_cv=0.10,
        surrender_value_curve=np.full(121, 80_000.0),
        surrender_value_basis="amount_per_policy")
    r = lre.cedant_solvency_relief(mp, basis, lre.LapseXL(0.15, 0.40),
                                   regime=sv.SOLVENCY2,
                                   reinsurer_pd=lre.CREDIT_QUALITY_STEP_PD[2])
    assert r.lapse_gross_scr > r.mass_gross_scr        # up/down is the biting leg
    assert np.isclose(r.lapse_relief, 0.0)
    assert np.isclose(r.total_benefit, 0.0)
