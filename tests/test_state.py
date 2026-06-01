"""Contract state -- input states and the dynamic waiver transition.

In-force is carried on two tracks: an active track (paying premium) and a
waiver track (premium waived, coverage continuing, no lapse). The input
``state`` seats a model point's count on one track; during the projection
the waiver-inception rate moves a fraction of the active track to the
waiver track each month. With no waiver-inception assumption the waiver
track stays empty and an active contract reduces to the ordinary single-
track projection.

The reference here is a plain-Python re-run of the two-track recursion --
an independent check on the compiled kernels.
"""
import numpy as np

from fastcashflow import STATE_ACTIVE, STATE_MODELS, STATE_PAIDUP, STATE_WAIVER, Basis, CalculationMethod, ModelPoints, CoverageRate, measure, read_model_points, measure

from conftest import annual_from_monthly as _annual

# Standard-normal 75th percentile -- used so the RA check does not depend on
# the engine's own quantile code.
Z_75 = 0.6744897501960817

PATTERNS = {
    "DEATH": CalculationMethod.DEATH,
    "dx":    CalculationMethod.DIAGNOSIS,
    "hosp":  CalculationMethod.MORBIDITY,
}


def _assumptions(waiver_rate: float = 0.0, **overrides) -> Basis:
    """Flat-rate, zero-discount, zero-expense basis -- every figure by hand.

    ``waiver_rate`` is a flat monthly waiver-inception rate; 0 leaves the
    assumption unset (no transitions).
    """
    waiver = None
    if waiver_rate != 0.0:
        def waiver(sex, issue_age, duration):
            return np.full(issue_age.shape, _annual(waiver_rate))
    base = dict(
        mortality_annual=lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.01)),
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, _annual(0.02)),
        waiver_incidence_annual=waiver,
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.01))),),
    )
    if waiver is not None:
        # Set state_model explicitly to silence the implicit-fallback warning.
        base["state_model"] = STATE_MODELS["WAIVER"]
    base.update(overrides)
    return Basis(**base)


def _two_track_bel(death_benefit, premium, term, state, *,
                   q=0.01, lapse=0.02, w=0.0, premium_term=None):
    """Plain-Python two-track recursion -- the independent BEL reference.

    Zero discount, so a present value is the plain sum. A death-benefit-only
    contract: BEL = PV(claims, on both tracks) - PV(premiums, active track).
    """
    premium_term = term if premium_term is None else premium_term
    act = 1.0 if state == STATE_ACTIVE else 0.0
    wav = 0.0 if state == STATE_ACTIVE else 1.0
    pv_claims = 0.0
    pv_premiums = 0.0
    inforce = []
    for t in range(term):
        total = act + wav
        inforce.append(total)
        pv_claims += total * q * death_benefit
        if t < premium_term:
            pv_premiums += act * premium
        act, wav = (act * (1.0 - q) * (1.0 - w) * (1.0 - lapse),
                    wav * (1.0 - q) + act * (1.0 - q) * w)
    return pv_claims - pv_premiums, inforce


def test_state_default_is_active():
    """A model point with no `state` is an ordinary active contract."""
    kw = dict(issue_age=40, benefits={0: 1_000_000.0},
              level_premium=12_000.0, term_months=12)
    asmp = _assumptions()
    default = ModelPoints.single(**kw, calculation_methods=PATTERNS)
    assert np.all(default.state == STATE_ACTIVE)
    assert np.isclose(
        measure(default, asmp, full=False).bel[0],
        measure(ModelPoints.single(**kw, state=STATE_ACTIVE, calculation_methods=PATTERNS), asmp, full=False).bel[0],
    )


def test_waiver_track_does_not_lapse():
    """A waiver contract's in-force decays by mortality alone -- no lapse."""
    mp = ModelPoints.single(issue_age=40, benefits={0: 1_000_000.0},
                            level_premium=12_000.0, term_months=3,
                            state=STATE_WAIVER,
                            calculation_methods=PATTERNS,
                            )
    res = measure(mp, _assumptions())
    # mortality only: 1 -> 0.99 -> 0.99**2.
    assert np.allclose(res.cashflows.inforce[0], [1.0, 0.99, 0.99 ** 2])


def test_waiver_hand_calculation():
    """Input-waiver, 2-month term: coverage continues, no premium, no lapse."""
    death_benefit = 1_000_000.0
    mp = ModelPoints.single(issue_age=40, benefits={0: death_benefit},
                            level_premium=12_000.0, term_months=2,
                            state=STATE_WAIVER,
                            calculation_methods=PATTERNS,
                            )
    asmp = _assumptions()
    val = measure(mp, asmp, full=False)

    # waiver in force [1.0, 0.99]; claims at 1e6, no premium, zero discount.
    inforce = [1.0, 0.99]
    pv_claims = sum(i * 0.01 * death_benefit for i in inforce)   # 19900.0
    assert np.isclose(val.bel[0], pv_claims)

    ra = Z_75 * 0.10 * pv_claims
    assert np.isclose(val.ra[0], ra)
    # no premium -> FCF positive -> onerous: no CSM, a loss component.
    assert np.isclose(val.csm[0], 0.0)
    assert np.isclose(val.loss_component[0], pv_claims + ra)


def test_waiver_collects_no_premium():
    """The waiver track pays no premium -- every premium cash flow is zero."""
    mp = ModelPoints.single(issue_age=40, benefits={0: 1_000_000.0},
                            level_premium=12_000.0, term_months=24,
                            state=STATE_WAIVER,
                            calculation_methods=PATTERNS,
                            )
    res = measure(mp, _assumptions())
    assert np.all(res.cashflows.premium_cf[0] == 0.0)


def test_paidup_matches_waiver():
    """Paid-up and waiver differ in cause, not cash flows -- identical
    BEL, RA, CSM and loss component."""
    kw = dict(issue_age=42, benefits={0: 80_000_000.0},
              level_premium=40_000.0, term_months=180)
    asmp = _assumptions()
    waiver = measure(ModelPoints.single(**kw, state=STATE_WAIVER, calculation_methods=PATTERNS), asmp, full=False)
    paidup = measure(ModelPoints.single(**kw, state=STATE_PAIDUP, calculation_methods=PATTERNS), asmp, full=False)
    for field in ("bel", "ra", "csm", "loss_component"):
        assert np.isclose(getattr(paidup, field)[0], getattr(waiver, field)[0])


def test_zero_waiver_rate_is_no_transition():
    """With no waiver-inception assumption the active track never leaks --
    the result is the ordinary single-track projection."""
    kw = dict(issue_age=45, benefits={0: 50_000_000.0},
              level_premium=30_000.0, term_months=120)
    plain = measure(ModelPoints.single(**kw, calculation_methods=PATTERNS), _assumptions(), full=False)
    with_zero = measure(ModelPoints.single(**kw, calculation_methods=PATTERNS), _assumptions(waiver_rate=0.0), full=False)
    assert np.isclose(plain.bel[0], with_zero.bel[0])


def test_dynamic_transition_hand_calculation():
    """Active contract, flat waiver-inception rate, 2-month term -- every
    figure derived by hand from the two-track recursion."""
    death_benefit = 1_000_000.0
    premium = 12_000.0
    asmp = _assumptions(waiver_rate=0.05)
    mp = ModelPoints.single(issue_age=40, benefits={0: death_benefit},
                            level_premium=premium, term_months=2,
                            calculation_methods=PATTERNS,
                            )

    # t=0: act=1, wav=0, total=1.
    #   act[1] = 1 * 0.99 * 0.95 * 0.98 = 0.92169
    #   wav[1] = 0 + 1 * 0.99 * 0.05    = 0.0495
    act1 = 0.99 * 0.95 * 0.98
    wav1 = 0.99 * 0.05
    inforce = [1.0, act1 + wav1]
    pv_claims = sum(i * 0.01 * death_benefit for i in inforce)
    pv_premiums = 1.0 * premium + act1 * premium    # premium on the active track
    bel = pv_claims - pv_premiums

    res = measure(mp, asmp)
    assert np.allclose(res.cashflows.inforce[0], inforce)
    assert np.isclose(res.bel_path[0, 0], bel)
    assert np.isclose(measure(mp, asmp, full=False).bel[0], bel)


def test_dynamic_transition_matches_reference():
    """measure() reproduces the plain-Python two-track recursion across a
    range of waiver-inception rates and starting states."""
    death_benefit = 1_000_000.0
    premium = 20_000.0
    term = 60
    for w in (0.0, 0.01, 0.05, 0.2):
        for state in (STATE_ACTIVE, STATE_WAIVER):
            asmp = _assumptions(waiver_rate=w)
            mp = ModelPoints.single(
                issue_age=40, benefits={0: death_benefit},
                level_premium=premium, term_months=term, state=state,
                calculation_methods=PATTERNS,
            )
            ref_bel, ref_inforce = _two_track_bel(
                death_benefit, premium, term, state, w=w)
            assert np.isclose(measure(mp, asmp, full=False).bel[0], ref_bel)
            assert np.allclose(measure(mp, asmp).cashflows.inforce[0],
                               ref_inforce)


def test_measure_and_value_agree_under_transition():
    """The detailed and the fused path give the same BEL with a transition."""
    mp = ModelPoints.single(issue_age=50, benefits={0: 30_000_000.0},
                            level_premium=25_000.0, term_months=240,
                            calculation_methods=PATTERNS,
                            )
    asmp = _assumptions(waiver_rate=0.03)
    assert np.isclose(measure(mp, asmp).bel_path[0, 0], measure(mp, asmp, full=False).bel[0])


def test_state_column_round_trips(tmp_path):
    """A wide file's `state` column reads back, and the waiver row -- no
    premium and no lapse -- carries the larger liability."""
    asmp = _assumptions()
    mp = ModelPoints(
        issue_age=np.array([40, 40]),
        level_premium=np.array([12_000.0, 12_000.0]),
        term_months=np.array([24, 24]),
        benefits={0: np.array([1_000_000.0, 1_000_000.0])},
        state=np.array([STATE_ACTIVE, STATE_WAIVER]),
        calculation_methods=PATTERNS,
    )
    path = tmp_path / "model_points.csv"
    mp.to_wide(asmp).write_csv(path)

    back = read_model_points(path)
    assert list(back.state) == [STATE_ACTIVE, STATE_WAIVER]
    val = measure(back, asmp, full=False)
    assert val.bel[1] > val.bel[0]


def test_paidup_state_spelling_is_normalised(tmp_path):
    """The `state` column accepts paid-up spellings -- case, spaces, hyphens
    and underscores are ignored."""
    path = tmp_path / "model_points.csv"
    path.write_text(
        "issue_age,term_months,level_premium,DEATH_benefit,state\n"
        "40,24,12000,1000000,Paid-up\n"
        "40,24,12000,1000000,paid_up\n"
        "40,24,12000,1000000,paid up\n"
        "40,24,12000,1000000,PAIDUP\n"
    )
    back = read_model_points(path, calculation_methods=PATTERNS)
    assert list(back.state) == [STATE_PAIDUP] * 4


def _flat(rate):
    """A flat ``(sex, issue_age, duration)`` rate callable -- returns annual equivalent."""
    return lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(rate))


def test_diagnosis_transition_measure_value_agree():
    """A diagnosis coverage attached to a death contract under a waiver
    transition -- the fused measure() and the detailed measure() agree,
    cross-checking the two-track diagnosis pool against the projection
    kernel over mixed input states."""
    mort_fn = lambda sex, issue_age, duration: np.full(issue_age.shape, _annual(0.01))
    asmp = _assumptions(
        waiver_rate=0.03,
        coverages=(
            CoverageRate("DEATH", mort_fn),
            CoverageRate("dx", _flat(0.004)),
        ),
    )
    rng = np.random.default_rng(11)
    n = 60
    mps = ModelPoints(
        issue_age=rng.integers(30, 55, n).astype(float),
        benefits={
            0: rng.integers(10, 80, n) * 1_000_000.0,
            1: rng.integers(5, 30, n) * 1_000_000.0,
        },
        level_premium=rng.integers(2, 10, n) * 10_000.0,
        term_months=np.full(n, 120),
        state=rng.integers(0, 3, n),
        calculation_methods={"DEATH": CalculationMethod.DEATH, "dx": CalculationMethod.DIAGNOSIS},
    )
    assert np.allclose(measure(mps, asmp).bel_path[:, 0], measure(mps, asmp, full=False).bel)


def test_waiting_rule_transition_measure_value_agree():
    """A coverage with a waiting period under a waiver transition -- measure()
    and measure() agree, cross-checking the two-track rule pass."""
    asmp = _assumptions(
        waiver_rate=0.04,
        coverages=(CoverageRate("hosp", _flat(0.02)),),
    )
    mps = ModelPoints(
        issue_age=np.array([40.0, 45.0]),
        level_premium=np.array([30_000.0, 30_000.0]),
        term_months=np.array([120, 120]),
        coverage_index=np.array([0, 0]),
        coverage_amount=np.array([2_000_000.0, 2_000_000.0]),
        coverage_offset=np.array([0, 1, 2]),
        coverage_waiting=np.array([12, 12]),
        state=np.array([STATE_ACTIVE, STATE_WAIVER]),
        calculation_methods={"hosp": CalculationMethod.MORBIDITY},
    )
    assert np.allclose(measure(mps, asmp).bel_path[:, 0], measure(mps, asmp, full=False).bel)
