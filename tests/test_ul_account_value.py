"""Universal-life account-value roll-forward -- hand-calc anchor.

The recursive account value cannot be derived in closed form (the COI depends on
the net amount at risk, which depends on the account value the COI reduces), so
this pins a tiny case derived entirely by hand, plus an independent plain-Python
recursion as a cross-check of the vectorised/parallel kernel.
"""
import numpy as np

from fastcashflow._ul import _ul_av_kernel, _ul_benefits, _ul_project
from fastcashflow.basis import Basis, annual_to_monthly
from fastcashflow.model_points import ModelPoints


def test_ul_av_kernel_hand_calc():
    # 1 policy, sum assured 1e8, no initial AV, 2 months. 500k premium credited
    # each month, COI 0.0001/month, no maintenance fee, 0.3%/month crediting.
    av0 = np.array([0.0])
    prem = np.array([[500_000.0, 500_000.0]])
    sa = np.array([100_000_000.0])
    coi_r = np.full((1, 2), 0.0001)
    admin = np.zeros((1, 2))
    credit = np.full((1, 2), 0.003)

    av, coi, av_mid, nar = _ul_av_kernel(av0, prem, sa, coi_r, admin, credit)

    # Month 0: BEF_FEE = 0 + 500_000; NAR = 1e8 - 500_000; COI = 1e-4 * NAR;
    # BEF_INV = 500_000 - COI; AV[1] = BEF_INV * 1.003.
    assert np.isclose(nar[0, 0], 99_500_000.0)
    assert np.isclose(coi[0, 0], 9_950.0)
    bef_inv0 = 500_000.0 - 9_950.0
    assert np.isclose(av[0, 1], bef_inv0 * 1.003)
    assert np.isclose(av_mid[0, 0], bef_inv0 * 1.003 ** 0.5)

    # Month 1: carries the month-0 closing AV.
    bef_fee1 = av[0, 1] + 500_000.0
    nar1 = 100_000_000.0 - bef_fee1
    coi1 = 1e-4 * nar1
    bef_inv1 = bef_fee1 - coi1
    assert np.isclose(nar[0, 1], nar1)
    assert np.isclose(coi[0, 1], coi1)
    assert np.isclose(av[0, 2], bef_inv1 * 1.003)


def _reference(av0, prem, sa, coi_r, admin, credit):
    n_mp, n_time = prem.shape
    av = np.zeros((n_mp, n_time + 1))
    coi = np.zeros((n_mp, n_time))
    av_mid = np.zeros((n_mp, n_time))
    nar = np.zeros((n_mp, n_time))
    for i in range(n_mp):
        a = av0[i]
        av[i, 0] = a
        for t in range(n_time):
            a += prem[i, t]
            risk = max(0.0, sa[i] - a)
            c = coi_r[i, t] * risk
            nar[i, t] = risk
            coi[i, t] = c
            a = max(0.0, a - admin[i, t] - c)
            av_mid[i, t] = a * (1.0 + credit[i, t]) ** 0.5
            a = a * (1.0 + credit[i, t])
            av[i, t + 1] = a
    return av, coi, av_mid, nar


def test_ul_av_kernel_matches_reference_recursion():
    # A deterministic multi-policy, multi-month case (different SA / premium /
    # COI / credit per policy) vs an independent plain-Python recursion.
    av0 = np.array([0.0, 1_000_000.0, 50_000.0])
    n_time = 24
    prem = np.array([
        np.full(n_time, 300_000.0),
        np.concatenate([np.full(12, 200_000.0), np.zeros(12)]),  # paid-up after a year
        np.full(n_time, 80_000.0),
    ])
    sa = np.array([50_000_000.0, 100_000_000.0, 30_000_000.0])
    coi_r = np.stack([np.full(n_time, 0.00008 + 0.000002 * t) for t in range(3)])
    admin = np.full((3, n_time), 1_000.0)
    credit = np.array([
        np.full(n_time, 0.0025),
        np.full(n_time, 0.0030),
        np.full(n_time, 0.0020),
    ])
    got = _ul_av_kernel(av0, prem, sa, coi_r, admin, credit)
    exp = _reference(av0, prem, sa, coi_r, admin, credit)
    for g, e in zip(got, exp):
        assert np.allclose(g, e)


def test_ul_benefits_hand_calc():
    # 1 policy, 3 months, matures at month 3 (term_idx = 2). SA 1e8.
    av_mid = np.array([[400_000.0, 900_000.0, 1_400_000.0]])
    av_end = np.array([[0.0, 466_000.0, 982_000.0, 1_500_000.0]])  # n_time+1
    deaths = np.array([[0.001, 0.001, 0.001]])
    lapses = np.array([[0.05, 0.05, 0.0]])
    maturity_survivors = np.array([0.8])
    term_idx = np.array([2])
    sa = np.array([100_000_000.0])
    surr_charge = np.array([[50_000.0, 30_000.0, 0.0]])
    gmab = np.array([0.0])

    benefit_cf, death_cf, surr_cf, mat_cf = _ul_benefits(
        av_end, av_mid, deaths, lapses, maturity_survivors, term_idx,
        sa, surr_charge, gmab)

    # death: max(av_mid, SA) = SA here (face >> account)
    assert np.allclose(death_cf[0], [100_000.0, 100_000.0, 100_000.0])
    # surrender: lapse * max(0, av_mid - charge)
    assert np.allclose(surr_cf[0], [0.05 * 350_000.0, 0.05 * 870_000.0, 0.0])
    # maturity: survivors * max(av_at_maturity=av_end[3]=1.5e6, gmab=0)
    assert np.isclose(mat_cf[0], 0.8 * 1_500_000.0)
    # combined, with maturity entered at term_idx=2
    assert np.allclose(benefit_cf[0], [117_500.0, 143_500.0, 100_000.0 + 1_200_000.0])


def test_ul_benefits_gmab_floor_and_account_exceeds_face():
    # av_mid exceeds the face -> death pays the account, not the face.
    # av_at_maturity below the GMAB -> maturity pays the GMAB.
    av_mid = np.array([[120_000_000.0]])
    av_end = np.array([[100_000_000.0, 110_000_000.0]])
    deaths = np.array([[0.01]])
    lapses = np.array([[0.0]])
    maturity_survivors = np.array([0.5])
    term_idx = np.array([0])
    sa = np.array([100_000_000.0])
    surr_charge = np.array([[0.0]])
    gmab = np.array([130_000_000.0])

    benefit_cf, death_cf, surr_cf, mat_cf = _ul_benefits(
        av_end, av_mid, deaths, lapses, maturity_survivors, term_idx,
        sa, surr_charge, gmab)
    assert np.isclose(death_cf[0, 0], 0.01 * 120_000_000.0)        # account > face
    assert np.isclose(mat_cf[0], 0.5 * 130_000_000.0)              # GMAB > matured av


def test_ul_av_nar_floored_when_account_exceeds_sum_assured():
    # A large account relative to the face -> NAR and COI go to zero.
    av0 = np.array([200_000_000.0])
    prem = np.zeros((1, 3))
    sa = np.array([100_000_000.0])
    coi_r = np.full((1, 3), 0.001)
    admin = np.zeros((1, 3))
    credit = np.zeros((1, 3))
    av, coi, av_mid, nar = _ul_av_kernel(av0, prem, sa, coi_r, admin, credit)
    assert np.all(nar[0] == 0.0)
    assert np.all(coi[0] == 0.0)
    assert np.allclose(av[0], 200_000_000.0)  # no premium, no charge, no credit


def _ul_basis(coi_annual_value, **overrides):
    """A minimal UL basis -- flat COI, no expenses; rates overridable per test."""
    kw = dict(
        mortality_annual=0.0,
        lapse_annual=0.0,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.1,
        investment_return=0.0,            # r_m = 0 -> credit = guarantee floor
        coi_annual=coi_annual_value,
    )
    kw.update(overrides)
    return Basis(**kw)


def test_ul_project_maturity_only_hand_calc():
    # No decrements (mortality = lapse = 0): the single policy survives to its
    # 2-month maturity, so the only benefit is maturity = the matured account
    # value. With zero return and a 0% guarantee floor, crediting is nil, so the
    # account simply runs down by the COI each month -- the whole AV path, the
    # fund and the maturity benefit are hand-derivable.
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        premium=np.array([0.0]),                       # no premium into the account
        term_months=np.array([2]),
        account_value=np.array([1_000_000.0]),         # single-premium account at issue
        minimum_death_benefit=np.array([10_000_000.0]),  # the UL face
        minimum_crediting_rate=np.array([0.0]),        # 0% floor; r_m = 0 -> credit 0
        sex=np.array([0]),
    )
    coi_a = 0.0012
    basis = _ul_basis(coi_a)
    p = _ul_project(mp, basis)

    q = annual_to_monthly(np.array(coi_a)).item()      # monthly COI rate
    face = 10_000_000.0
    # Account roll by hand (no premium, no credit, no admin fee):
    av0 = 1_000_000.0
    coi0 = q * (face - av0)
    av1 = av0 - coi0
    coi1 = q * (face - av1)
    av2 = av1 - coi1
    assert np.allclose(p.av[0], [av0, av1, av2])
    assert np.allclose(p.coi[0], [coi0, coi1])
    assert np.allclose(p.nar[0], [face - av0, face - av1])

    # Survivor reaches maturity at term_idx = 1 with the matured value av2.
    assert p.term_idx[0] == 1
    assert np.isclose(p.maturity_survivors[0], 1.0)
    assert np.isclose(p.maturity_cf[0], av2)           # GMAB = 0 -> matured av
    # No deaths / surrenders; maturity enters benefit_cf at term_idx.
    assert np.allclose(p.death_cf[0], [0.0, 0.0])
    assert np.allclose(p.surrender_cf[0], [0.0, 0.0])
    assert np.allclose(p.benefit_cf[0], [0.0, av2])
    # Fund = in-force-weighted account value; in force = 1 through maturity, the
    # padded month-end column 0.
    assert np.allclose(p.fund[0], [av0, av1, 0.0])


def test_ul_project_weaves_decrements_and_load():
    # With non-zero mortality / lapse, the orchestration must weave the right
    # arrays: death on the occupancy deaths at max(av_mid, face), surrender on
    # the non-maturity exits at av_mid, fund in-force weighted -- and the premium
    # load must reach the account (prem_to_av = premium * (1 - load)).
    mp = ModelPoints(
        issue_age=np.array([45.0]),
        premium=np.array([300_000.0]),
        term_months=np.array([12]),
        account_value=np.array([0.0]),
        minimum_death_benefit=np.array([50_000_000.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([1]),
    )
    basis = _ul_basis(
        0.001, mortality_annual=0.005, lapse_annual=0.03,
        investment_return=0.024, premium_load=0.1)
    p = _ul_project(mp, basis)
    proj = p.cashflows

    # The premium load reaches the account: month-0 NAR = face - prem_to_av,
    # prem_to_av = 300_000 * (1 - 0.1) = 270_000 (av0 = 0).
    assert np.isclose(p.nar[0, 0], 50_000_000.0 - 270_000.0)

    # Benefit weaving, re-derived from the projection decrements and AV path.
    inforce_pad = np.concatenate([p.inforce, np.zeros((1, 1))], axis=1)
    exits = inforce_pad[:, :-1] - inforce_pad[:, 1:]
    deaths = proj.deaths
    non_maturity_exits = exits - deaths
    term_idx = p.term_idx[0]
    non_maturity_exits[0, term_idx] -= p.maturity_survivors[0]

    exp_death = deaths * np.maximum(p.av_mid, mp.minimum_death_benefit[:, None])
    exp_surr = non_maturity_exits * p.av_mid          # surr_charge = 0
    assert np.allclose(p.death_cf, exp_death)
    assert np.allclose(p.surrender_cf, exp_surr)
    assert np.allclose(p.fund, inforce_pad * p.av)
    # benefit_cf = death + surrender, with maturity added at term_idx.
    exp_benefit = exp_death + exp_surr
    exp_benefit[0, term_idx] += p.maturity_cf[0]
    assert np.allclose(p.benefit_cf, exp_benefit)
