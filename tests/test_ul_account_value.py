"""Universal-life account-value roll-forward -- hand-calc anchor.

The recursive account value cannot be derived in closed form (the COI depends on
the net amount at risk, which depends on the account value the COI reduces), so
this pins a tiny case derived entirely by hand, plus an independent plain-Python
recursion as a cross-check of the vectorised/parallel kernel.

A universal-life contract is expressed as an account-backed DEATH coverage --
``CoverageRate("DEATH", coi_annual, funds_from_account=True,
pays_account_balance=True)`` with the face carried as ``minimum_death_benefit``.
The shared projection (``gmm.measure`` / ``vfa.measure``) then rolls the account
forward and exposes the trajectory on the ``cashflows.account`` sidecar
(:class:`fastcashflow.projection.AccountTrajectory`): ``av`` (n_mp, n_time+1),
``av_mid`` (n_mp, n_time), ``coi`` (n_mp, n_time), ``fund`` (n_mp, n_time+1).
The net amount at risk is NOT stored -- it is recomputed as
``max(0, face - av_mid)`` where needed. The account-driven benefits stay in-band
on :class:`fastcashflow.projection.Cashflows`: the death leg in ``mortality_cf``
(deaths * max(av_mid, face)), the surrender in ``surrender_cf``, and the
maturity in ``maturity_cf`` (a ``(n_mp,)`` payment entered at the term column).
"""
import numpy as np

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow.basis import annual_to_monthly


# ---------------------------------------------------------------------------
# Constructors -- a universal-life contract as an account-backed DEATH coverage.
# ---------------------------------------------------------------------------

def _ul_basis(coi_annual_value, **overrides):
    """A minimal UL basis -- flat COI, no expenses; rates overridable per test.

    The DEATH coverage carries the account-chassis flags so the shared
    projection routes its death leg through the account roll.
    """
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
    kw["coverages"] = (
        CoverageRate("DEATH", coi_annual_value, funds_from_account=True,
                     pays_account_balance=True),
    )
    return Basis(**kw)


def _ul_mp(face, **fields):
    """A UL ModelPoints -- the face on ``minimum_death_benefit`` and the
    account-backed DEATH coverage registered (its amount is unused -- the
    account death reads the account balance, not the coverage amount)."""
    face_arr = np.atleast_1d(np.asarray(face, dtype=float))
    fields["minimum_death_benefit"] = face_arr
    fields.setdefault("benefits", {"DEATH": face_arr})
    fields.setdefault("calculation_methods", {"DEATH": CalculationMethod.DEATH})
    return ModelPoints(**fields)


def _account(measurement):
    return measurement.cashflows.account


def _nar(measurement, face):
    """The net amount at risk is not stored -- recompute max(0, face - av_mid)."""
    acct = measurement.cashflows.account
    return np.maximum(0.0, np.atleast_1d(np.asarray(face, dtype=float))[:, None]
                      - acct.av_mid)


# ---------------------------------------------------------------------------
# Account-roll hand calc -- single policy, premium credited, COI on the NAR.
# ---------------------------------------------------------------------------

def test_ul_av_kernel_hand_calc():
    # 1 policy, sum assured 1e8, no initial AV, 2 months. 500k premium credited
    # each month, COI ~0.0001/month (annual COI back-solved so the monthly charge
    # is 1e-4), no maintenance fee, ~0.3%/month crediting (annual investment
    # return back-solved so the monthly credit is 0.003).
    coi_m = 0.0001
    credit_m = 0.003
    coi_annual = 1.0 - (1.0 - coi_m) ** 12        # annual_to_monthly inverse
    inv_return = (1.0 + credit_m) ** 12 - 1.0
    face = 100_000_000.0
    mp = _ul_mp(
        face,
        issue_age=np.array([40.0]),
        premium=np.array([500_000.0]),
        term_months=np.array([2]),
        account_value=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _ul_basis(coi_annual, investment_return=inv_return)
    m = fcf.gmm.measure(mp, basis, full=True)
    acct = _account(m)
    nar = _nar(m, face)

    # Confirm the back-solved annual rates reproduce the intended monthly charges.
    assert np.isclose(annual_to_monthly(np.array(coi_annual)).item(), coi_m)
    assert np.isclose((1.0 + inv_return) ** (1.0 / 12.0) - 1.0, credit_m)

    # Month 0: BEF_FEE = 0 + 500_000; NAR = 1e8 - 500_000; COI = 1e-4 * NAR;
    # BEF_INV = 500_000 - COI; AV[1] = BEF_INV * 1.003.
    assert np.isclose(acct.coi[0, 0], 9_950.0)
    bef_inv0 = 500_000.0 - 9_950.0
    assert np.isclose(acct.av[0, 1], bef_inv0 * 1.003)
    assert np.isclose(acct.av_mid[0, 0], bef_inv0 * 1.003 ** 0.5)
    # NAR is recomputed from av_mid (max(0, face - av_mid)).
    assert np.isclose(nar[0, 0], face - acct.av_mid[0, 0])

    # Month 1: carries the month-0 closing AV.
    bef_fee1 = acct.av[0, 1] + 500_000.0
    nar1 = face - bef_fee1
    coi1 = coi_m * nar1
    bef_inv1 = bef_fee1 - coi1
    assert np.isclose(acct.coi[0, 1], coi1)
    assert np.isclose(acct.av[0, 2], bef_inv1 * 1.003)
    assert np.isclose(nar[0, 1], face - acct.av_mid[0, 1])


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
    # A deterministic multi-policy, multi-month case vs an independent plain-Python
    # recursion. Per-policy COI differs via a callable rate keyed on sex; per-policy
    # crediting differs via the per-policy minimum_crediting_rate guarantee floor
    # (the basis investment_return is 0, so credit = floor). No maintenance fee.
    n_time = 24
    coi_annual_p = np.array([0.001, 0.0015])      # annual COI, per policy (by sex)
    credit_m_p = np.array([0.0025, 0.0030])       # target monthly credit, per policy
    min_annual = (1.0 + credit_m_p) ** 12 - 1.0   # floor that yields that monthly credit
    sa = np.array([50_000_000.0, 100_000_000.0])
    av0 = np.array([0.0, 1_000_000.0])
    prem_p = np.array([300_000.0, 200_000.0])

    coi_fn = lambda s, a, d: np.where(np.asarray(s) == 0, 0.001, 0.0015)
    mp = _ul_mp(
        sa,
        issue_age=np.array([40.0, 50.0]),
        premium=prem_p,
        term_months=np.array([n_time, n_time]),
        account_value=av0,
        minimum_crediting_rate=min_annual,
        sex=np.array([0, 1]),
    )
    basis = Basis(
        mortality_annual=0.0, lapse_annual=0.0, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1, investment_return=0.0,
        coi_annual=coi_fn,
        coverages=(CoverageRate("DEATH", coi_fn, funds_from_account=True,
                                pays_account_balance=True),))
    m = fcf.gmm.measure(mp, basis, full=True)
    acct = _account(m)

    # Reconstruct the effective monthly arrays the basis implies and run the
    # independent recursion -- the folded account roll must reproduce it.
    coi_r = annual_to_monthly(coi_annual_p)[:, None] * np.ones((2, n_time))
    credit = credit_m_p[:, None] * np.ones((2, n_time))
    prem = prem_p[:, None] * np.ones((2, n_time))
    admin = np.zeros((2, n_time))
    av, coi, av_mid, nar = _reference(av0, prem, sa, coi_r, admin, credit)

    assert np.allclose(acct.av, av)
    assert np.allclose(acct.coi, coi)
    assert np.allclose(acct.av_mid, av_mid)
    # NAR is recomputed from av_mid; cross-check it equals the same recompute on
    # the reference av_mid.
    assert np.allclose(_nar(m, sa), np.maximum(0.0, sa[:, None] - av_mid))


def test_ul_benefits_hand_calc():
    # Death / surrender / maturity benefits from a real account roll. 1 policy,
    # 3 months, matures at month 3, SA 1e8 (face >> account, so death pays the
    # face). With mortality / lapse on, the benefit legs are re-derived from the
    # projection decrements and the produced AV path.
    face = 100_000_000.0
    mp = _ul_mp(
        face,
        issue_age=np.array([45.0]),
        premium=np.array([400_000.0]),
        term_months=np.array([3]),
        account_value=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _ul_basis(
        0.0012, mortality_annual=0.012, lapse_annual=0.4,
        investment_return=0.03)
    m = fcf.gmm.measure(mp, basis, full=True)
    proj = m.cashflows
    acct = _account(m)
    term_idx = int(mp.term_months[0]) - 1

    # The face dominates the small account -> death pays max(av_mid, face) = face.
    assert np.all(acct.av_mid[0] < face)
    exp_death = proj.deaths * np.maximum(acct.av_mid, face)
    assert np.allclose(proj.mortality_cf, exp_death)

    # Surrender = (non-maturity exits) * av_mid (no surrender charge in v1).
    inforce_pad = np.concatenate([proj.inforce, np.zeros((1, 1))], axis=1)
    exits = inforce_pad[:, :-1] - inforce_pad[:, 1:]
    non_maturity_exits = exits - proj.deaths
    non_maturity_exits[0, term_idx] -= proj.maturity_survivors[0]
    exp_surr = non_maturity_exits * acct.av_mid
    assert np.allclose(proj.surrender_cf, exp_surr)

    # Maturity = survivors * max(matured av, GMAB=0) = survivors * matured av.
    av_at_maturity = acct.av[0, term_idx + 1]
    assert np.isclose(proj.maturity_cf[0], proj.maturity_survivors[0] * av_at_maturity)

    # Combined benefit, with maturity entered at the term column.
    exp_benefit = exp_death + exp_surr
    exp_benefit[0, term_idx] += proj.maturity_cf[0]
    got_benefit = proj.mortality_cf + proj.surrender_cf
    got_benefit[0, term_idx] += proj.maturity_cf[0]
    assert np.allclose(got_benefit, exp_benefit)


def test_ul_benefits_gmab_floor_and_account_exceeds_face():
    # av_mid exceeds the face -> death pays the account, not the face.
    # av_at_maturity below the GMAB -> maturity pays the GMAB. The GMAB floor in
    # the folded path is carried as maturity_benefit (account_face has no GMAB).
    face = 100_000_000.0
    gmab = 130_000_000.0
    mp = _ul_mp(
        face,
        issue_age=np.array([45.0]),
        premium=np.array([0.0]),
        term_months=np.array([1]),
        account_value=np.array([120_000_000.0]),
        maturity_benefit=np.array([gmab]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _ul_basis(0.0, mortality_annual=0.01, investment_return=0.0)
    m = fcf.gmm.measure(mp, basis, full=True)
    proj = m.cashflows
    acct = _account(m)

    # No premium / COI / credit -> the account stays at 120e6 (av_mid = 120e6);
    # 1 policy, monthly mortality q, no lapse -> deaths = q, survivors = 1 - q.
    q = annual_to_monthly(np.array(0.01)).item()
    assert np.isclose(acct.av_mid[0, 0], 120_000_000.0)
    assert np.isclose(proj.deaths[0, 0], q)
    assert np.isclose(proj.maturity_survivors[0], 1.0 - q)
    # account (120e6) > face (100e6): death pays the account, not the face.
    assert acct.av_mid[0, 0] > face
    assert np.isclose(proj.mortality_cf[0, 0], q * 120_000_000.0)
    # matured av (120e6) < GMAB (130e6): maturity pays the GMAB floor (130e6).
    assert acct.av[0, 1] < gmab
    assert np.isclose(proj.maturity_cf[0], (1.0 - q) * gmab)


def test_ul_av_nar_floored_when_account_exceeds_sum_assured():
    # A large account relative to the face -> NAR and COI go to zero.
    face = 100_000_000.0
    mp = _ul_mp(
        face,
        issue_age=np.array([45.0]),
        premium=np.array([0.0]),
        term_months=np.array([3]),
        account_value=np.array([200_000_000.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _ul_basis(0.001, investment_return=0.0)
    m = fcf.gmm.measure(mp, basis, full=True)
    acct = _account(m)
    nar = _nar(m, face)
    assert np.all(nar[0] == 0.0)            # account >> face -> NAR floored to 0
    assert np.all(acct.coi[0] == 0.0)       # no NAR -> no COI
    assert np.allclose(acct.av[0], 200_000_000.0)  # no premium, no charge, no credit


# ---------------------------------------------------------------------------
# Orchestration -- decrements woven into the account-driven benefits.
# ---------------------------------------------------------------------------

def test_ul_project_maturity_only_hand_calc():
    # No decrements (mortality = lapse = 0): the single policy survives to its
    # 2-month maturity, so the only benefit is maturity = the matured account
    # value. With zero return and a 0% guarantee floor, crediting is nil, so the
    # account simply runs down by the COI each month -- the whole AV path, the
    # fund and the maturity benefit are hand-derivable.
    face = 10_000_000.0
    coi_a = 0.0012
    mp = _ul_mp(
        face,
        issue_age=np.array([40.0]),
        premium=np.array([0.0]),                       # no premium into the account
        term_months=np.array([2]),
        account_value=np.array([1_000_000.0]),         # single-premium account at issue
        minimum_crediting_rate=np.array([0.0]),        # 0% floor; r_m = 0 -> credit 0
        sex=np.array([0]),
    )
    basis = _ul_basis(coi_a)
    m = fcf.gmm.measure(mp, basis, full=True)
    proj = m.cashflows
    acct = _account(m)
    nar = _nar(m, face)

    q = annual_to_monthly(np.array(coi_a)).item()      # monthly COI rate
    # Account roll by hand (no premium, no credit, no admin fee):
    av0 = 1_000_000.0
    coi0 = q * (face - av0)
    av1 = av0 - coi0
    coi1 = q * (face - av1)
    av2 = av1 - coi1
    assert np.allclose(acct.av[0], [av0, av1, av2])
    assert np.allclose(acct.coi[0], [coi0, coi1])
    # NAR is recomputed from av_mid; with zero crediting av_mid == av_end, so
    # av_mid = [av1, av2] and NAR = [face - av1, face - av2].
    assert np.allclose(acct.av_mid[0], [av1, av2])
    assert np.allclose(nar[0], [face - av1, face - av2])

    # Survivor reaches maturity at term_idx = 1 with the matured value av2.
    term_idx = int(mp.term_months[0]) - 1
    assert term_idx == 1
    assert np.isclose(proj.maturity_survivors[0], 1.0)
    assert np.isclose(proj.maturity_cf[0], av2)        # GMAB = 0 -> matured av
    # No deaths / surrenders; maturity enters benefit_cf at term_idx.
    assert np.allclose(proj.mortality_cf[0], [0.0, 0.0])
    assert np.allclose(proj.surrender_cf[0], [0.0, 0.0])
    benefit_cf = proj.mortality_cf + proj.surrender_cf
    benefit_cf[0, term_idx] += proj.maturity_cf[0]
    assert np.allclose(benefit_cf[0], [0.0, av2])
    # Fund = in-force-weighted account value; in force = 1 through maturity, the
    # padded month-end column 0.
    inforce_pad = np.concatenate([proj.inforce, np.zeros((1, 1))], axis=1)
    assert np.allclose(acct.fund[0], inforce_pad[0] * acct.av[0])
    assert np.allclose(acct.fund[0], [av0, av1, 0.0])


def test_ul_project_weaves_decrements_and_load():
    # With non-zero mortality / lapse, the orchestration must weave the right
    # arrays: death on the occupancy deaths at max(av_mid, face), surrender on
    # the non-maturity exits at av_mid, fund in-force weighted -- and the premium
    # load must reach the account (prem_to_av = premium * (1 - load)).
    face = 50_000_000.0
    mp = _ul_mp(
        face,
        issue_age=np.array([45.0]),
        premium=np.array([300_000.0]),
        term_months=np.array([12]),
        account_value=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([1]),
    )
    basis = _ul_basis(
        0.001, mortality_annual=0.005, lapse_annual=0.03,
        investment_return=0.024, premium_load=0.1)
    m = fcf.gmm.measure(mp, basis, full=True)
    proj = m.cashflows
    acct = _account(m)

    # The premium load reaches the account: prem_to_av = 300_000 * (1 - 0.1) =
    # 270_000 (av0 = 0), the month-0 post-premium balance. The COI is charged on
    # that month's NAR = face - 270_000 -- pin the hand COI value (this proves the
    # load reached the account AND the COI is on the right net amount at risk).
    q_coi = annual_to_monthly(np.array(0.001)).item()
    assert np.isclose(acct.coi[0, 0], q_coi * (face - 270_000.0))

    # Benefit weaving, re-derived from the projection decrements and AV path.
    inforce_pad = np.concatenate([proj.inforce, np.zeros((1, 1))], axis=1)
    exits = inforce_pad[:, :-1] - inforce_pad[:, 1:]
    deaths = proj.deaths
    non_maturity_exits = exits - deaths
    term_idx = int(mp.term_months[0]) - 1
    non_maturity_exits[0, term_idx] -= proj.maturity_survivors[0]

    exp_death = deaths * np.maximum(acct.av_mid, mp.minimum_death_benefit[:, None])
    exp_surr = non_maturity_exits * acct.av_mid       # surr_charge = 0
    assert np.allclose(proj.mortality_cf, exp_death)
    assert np.allclose(proj.surrender_cf, exp_surr)
    assert np.allclose(acct.fund, inforce_pad * acct.av)
    # benefit_cf = death + surrender, with maturity added at term_idx.
    exp_benefit = exp_death + exp_surr
    exp_benefit[0, term_idx] += proj.maturity_cf[0]
    got_benefit = proj.mortality_cf + proj.surrender_cf
    got_benefit[0, term_idx] += proj.maturity_cf[0]
    assert np.allclose(got_benefit, exp_benefit)


# ---------------------------------------------------------------------------
# Measurement -- BEL / RA / CSM the account-driven cash flows discount into.
# ---------------------------------------------------------------------------

def test_measure_ul_margin_hand_calc():
    # 1 policy, 1-month term, premium 1e6 (load 10%), face 1e7, no decrements
    # other than maturity, zero return / 0% floor / no admin, zero discount. The
    # account takes in the net-of-load premium, the COI is charged on the NAR
    # (but no death occurs), and the survivor takes the matured account. The BEL
    # is then exactly -(load amount + COI) -- the insurer keeps the load and the
    # COI as margin -- and the CSM is its negative.
    premium = 1_000_000.0
    load = 0.1
    face = 10_000_000.0
    coi_a = 0.012
    mp = _ul_mp(
        face,
        issue_age=np.array([40.0]),
        premium=np.array([premium]),
        term_months=np.array([1]),
        account_value=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _ul_basis(
        coi_a, mortality_annual=0.0, lapse_annual=0.0, discount_annual=0.0,
        premium_load=load)
    m = fcf.gmm.measure(mp, basis, full=True)

    q = annual_to_monthly(np.array(coi_a)).item()
    prem_to_av = premium * (1.0 - load)
    coi0 = q * (face - prem_to_av)               # NAR = face - prem_to_av
    av1 = prem_to_av - coi0                       # matured account (no credit / admin)
    # BEL = PV(maturity) - PV(premium) - fund0; fund0 = 0, no discount.
    assert np.isclose(m.bel[0], av1 - premium)   # = -(load amount + COI)
    assert np.isclose(m.bel[0], -(load * premium + coi0))
    assert np.isclose(m.ra[0], 0.0)              # no death -> no NAR claim; expense_cv 0
    assert np.isclose(m.csm[0], load * premium + coi0)
    assert np.isclose(m.loss_component[0], 0.0)
    assert isinstance(m, fcf.gmm.GMMMeasurement)     # routed through the GMM path


def test_measure_ul_single_premium_pass_through_is_zero():
    # A pure pass-through: single-premium account, no load, no COI, no admin, no
    # decrements but maturity, zero return and zero discount. The account is
    # returned in full at maturity with no margin, so BEL / RA / CSM are all
    # zero -- and the fund netting (BEL = PV(maturity) - fund0, premium_cf = 0)
    # reduces to the VFA single-premium form.
    mp = _ul_mp(
        10_000_000.0,
        issue_age=np.array([50.0]),
        premium=np.array([0.0]),                 # single premium sits in the account
        term_months=np.array([1]),
        account_value=np.array([1_000_000.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([1]),
    )
    basis = _ul_basis(
        0.0, mortality_annual=0.0, lapse_annual=0.0, discount_annual=0.0)
    m = fcf.gmm.measure(mp, basis, full=True)
    assert np.isclose(m.bel[0], 0.0)
    assert np.isclose(m.ra[0], 0.0)
    assert np.isclose(m.csm[0], 0.0)
    assert np.isclose(m.loss_component[0], 0.0)


def test_measure_ul_routing_changes_only_the_discount():
    # GMM discounts at the locked-in rate, VFA at the underlying-items return.
    # With a profitable spread (COI > mortality, plus a load) both give a
    # positive CSM, and the two differ purely through the discount basis.
    mp = _ul_mp(
        50_000_000.0,
        issue_age=np.array([45.0]),
        premium=np.array([400_000.0]),
        term_months=np.array([60]),
        account_value=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _ul_basis(
        0.003, mortality_annual=0.002, lapse_annual=0.03,
        discount_annual=0.03, investment_return=0.05, premium_load=0.08)
    gmm = fcf.gmm.measure(mp, basis, full=True)
    vfa = fcf.vfa.measure(mp, basis)
    assert gmm.csm[0] > 0 and vfa.csm[0] > 0
    assert isinstance(gmm, fcf.gmm.GMMMeasurement)   # GMM routing
    assert isinstance(vfa, fcf.vfa.VFAMeasurement)   # VFA routing
    # Different discount basis -> different BEL / CSM.
    assert not np.isclose(gmm.bel[0], vfa.bel[0])


def test_measure_ul_headline_matches_full_path_inception():
    mp = _ul_mp(
        80_000_000.0,
        issue_age=np.array([40.0]),
        premium=np.array([500_000.0]),
        term_months=np.array([36]),
        account_value=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _ul_basis(
        0.0025, mortality_annual=0.0018, lapse_annual=0.04,
        discount_annual=0.025, investment_return=0.03, premium_load=0.06)
    full = fcf.gmm.measure(mp, basis, full=True)
    head = fcf.gmm.measure(mp, basis, full=False)
    assert np.isclose(full.bel[0], head.bel[0])
    assert np.isclose(full.ra[0], head.ra[0])
    assert np.isclose(full.csm[0], head.csm[0])
    assert head.csm_path is None and full.csm_path is not None
    # The full path's column 0 is the headline.
    assert np.isclose(full.csm_path[0, 0], full.csm[0])
    assert np.isclose(full.bel_path[0, 0], full.bel[0])
