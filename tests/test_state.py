"""Contract state -- waiver of premium and paid-up.

A model point in the WAIVER or PAIDUP state keeps its coverage in force but
collects no premium (IFRS 17 Sec. 33-34 -- the fulfilment cash flows reflect
the contract's actual terms at the measurement date). The hand case is the
2-month flat-rate contract of test_phase0, valued once active and once with
the premium stopped: the two differ only by the dropped premium stream.
"""
import numpy as np

from fastcashflow import (
    STATE_ACTIVE,
    STATE_PAIDUP,
    STATE_WAIVER,
    Assumptions,
    ModelPoints,
    measure,
    read_model_points,
    value,
)

# Standard-normal 75th percentile -- used so the RA check does not depend on
# the engine's own quantile code.
Z_75 = 0.6744897501960817


def _assumptions(**overrides) -> Assumptions:
    """Flat-rate, zero-discount, zero-expense basis -- every figure by hand."""
    base = dict(
        mortality_monthly=lambda sex, issue_age, duration: np.full(issue_age.shape, 0.01),
        lapse_monthly=lambda duration: np.full(duration.shape, 0.02),
        discount_annual=0.0,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
    )
    base.update(overrides)
    return Assumptions(**base)


def test_waiver_hand_calculation():
    """Waiver zeroes the premium inflow; claims, RA and decrements stand."""
    kw = dict(issue_age=40, death_benefit=1_000_000.0,
              monthly_premium=12_000.0, term_months=2)
    asmp = _assumptions()

    active = value(ModelPoints.single(**kw, state=STATE_ACTIVE), asmp)
    waiver = value(ModelPoints.single(**kw, state=STATE_WAIVER), asmp)

    # in force [1.0, 0.99 * 0.98]; claims at 1e6, premiums at 12e3, no discount.
    inforce = [1.0, 0.99 * 0.98]
    pv_claims = sum(i * 0.01 * 1_000_000.0 for i in inforce)   # 19702.0
    pv_premiums = sum(i * 12_000.0 for i in inforce)           # 23642.4

    # active: BEL = PV(claims) - PV(premiums); waiver: BEL = PV(claims).
    assert np.isclose(active.bel[0], pv_claims - pv_premiums)
    assert np.isclose(waiver.bel[0], pv_claims)
    # the only difference between the two is the dropped premium stream.
    assert np.isclose(waiver.bel[0] - active.bel[0], pv_premiums)

    # RA is a load on the claims -- the premium does not enter it.
    ra = Z_75 * 0.10 * pv_claims
    assert np.isclose(active.ra[0], ra)
    assert np.isclose(waiver.ra[0], ra)

    # dropping the premium turns FCF positive -> the waiver contract is
    # onerous: no CSM, a loss component equal to the FCF.
    assert np.isclose(waiver.csm[0], 0.0)
    assert np.isclose(waiver.loss_component[0], pv_claims + ra)
    # the active contract is profitable -- it carries a CSM, no loss.
    assert active.csm[0] > 0.0
    assert np.isclose(active.loss_component[0], 0.0)


def test_waiver_drops_exactly_the_premium_pv():
    """On a full-term contract the BEL rises by exactly the projected
    premium PV -- zero discount makes that the nominal premium sum."""
    kw = dict(issue_age=45, death_benefit=50_000_000.0,
              monthly_premium=30_000.0, term_months=120)
    asmp = _assumptions()

    active = value(ModelPoints.single(**kw, state=STATE_ACTIVE), asmp)
    waiver = value(ModelPoints.single(**kw, state=STATE_WAIVER), asmp)

    proj = measure(ModelPoints.single(**kw, state=STATE_ACTIVE), asmp)
    pv_premiums = proj.cashflows.premium_cf[0].sum()   # 0% discount: PV = sum

    assert pv_premiums > 0.0
    assert np.isclose(waiver.bel[0] - active.bel[0], pv_premiums)


def test_waiver_default_is_active():
    """A model point with no `state` is an ordinary active contract."""
    kw = dict(issue_age=40, death_benefit=1_000_000.0,
              monthly_premium=12_000.0, term_months=12)
    asmp = _assumptions()

    default = ModelPoints.single(**kw)
    assert np.all(default.state == STATE_ACTIVE)
    assert np.isclose(
        value(default, asmp).bel[0],
        value(ModelPoints.single(**kw, state=STATE_ACTIVE), asmp).bel[0],
    )


def test_state_column_round_trips(tmp_path):
    """A wide file's `state` column reads back to the waiver state, and the
    waiver row is valued with no premium."""
    asmp = _assumptions()
    mp = ModelPoints(
        issue_age=np.array([40, 40]),
        monthly_premium=np.array([12_000.0, 12_000.0]),
        term_months=np.array([24, 24]),
        death_benefit=np.array([1_000_000.0, 1_000_000.0]),
        state=np.array([STATE_ACTIVE, STATE_WAIVER]),
    )
    path = tmp_path / "model_points.csv"
    mp.to_wide(asmp).write_csv(path)

    back = read_model_points(path, asmp)
    assert list(back.state) == [STATE_ACTIVE, STATE_WAIVER]

    val = value(back, asmp)
    # same contract, but the waiver row collects no premium -> larger liability.
    assert val.bel[1] > val.bel[0]


def test_paidup_drops_premium():
    """A paid-up contract collects no premium -- the BEL rises by the
    projected premium PV, exactly as a waiver does."""
    kw = dict(issue_age=40, death_benefit=1_000_000.0,
              monthly_premium=12_000.0, term_months=2)
    asmp = _assumptions()

    active = value(ModelPoints.single(**kw, state=STATE_ACTIVE), asmp)
    paidup = value(ModelPoints.single(**kw, state=STATE_PAIDUP), asmp)

    inforce = [1.0, 0.99 * 0.98]
    pv_premiums = sum(i * 12_000.0 for i in inforce)           # 23642.4
    assert np.isclose(paidup.bel[0] - active.bel[0], pv_premiums)


def test_paidup_matches_waiver():
    """Paid-up and waiver differ in cause, not in cash flows -- a contract
    valued in either state gives identical BEL, RA, CSM and loss component."""
    kw = dict(issue_age=42, death_benefit=80_000_000.0,
              monthly_premium=40_000.0, term_months=180)
    asmp = _assumptions()

    waiver = value(ModelPoints.single(**kw, state=STATE_WAIVER), asmp)
    paidup = value(ModelPoints.single(**kw, state=STATE_PAIDUP), asmp)

    for field in ("bel", "ra", "csm", "loss_component"):
        assert np.isclose(getattr(paidup, field)[0], getattr(waiver, field)[0])


def test_paidup_state_spelling_is_normalised(tmp_path):
    """The `state` column accepts paid-up spellings -- case, spaces, hyphens
    and underscores are ignored."""
    path = tmp_path / "model_points.csv"
    path.write_text(
        "issue_age,term_months,monthly_premium,death_benefit,state\n"
        "40,24,12000,1000000,Paid-up\n"
        "40,24,12000,1000000,paid_up\n"
        "40,24,12000,1000000,paid up\n"
        "40,24,12000,1000000,PAIDUP\n"
    )
    back = read_model_points(path, _assumptions())
    assert list(back.state) == [STATE_PAIDUP] * 4
