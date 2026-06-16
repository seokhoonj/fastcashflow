"""IFRS 17 paragraph-indexed validation.

A navigable index from IFRS 17 paragraph -> engine mechanic. Each test
docstring leads with the paragraph it validates, with a tiny hand-validated
input so the figure can be derived on paper.

The standard text is authoritative; a literal IFRS 17 Illustrative Examples
reproduction is intentionally not attempted here because IFRS 17 IE assumes
year-end annual cash flows while this engine projects monthly with mid-month
discounting -- the two discretisations disagree at the percent level on the
same gross inputs, and the engine's discretisation is the mechanic under test.

Paragraph -> test name:

* Sec.32        -- test_sec32_bel_is_pv_of_future_cashflows
* Sec.34 / B65  -- test_sec34_b65_contract_boundary
* Sec.37        -- test_sec37_ra_addition_to_bel
* Sec.38(b)     -- test_sec38_initial_csm_profitable
* Sec.38(c)     -- test_sec38_loss_component_onerous
* Sec.44(b)     -- test_sec44_csm_accretion_at_locked_in_rate
* Sec.44(e)+B119 -- test_sec44_b119_csm_release_proportional_to_coverage_units
* B96           -- test_b96_higher_discount_reduces_pv_of_claims
"""
import numpy as np

from fastcashflow import ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


def _flat_assumptions(**overrides):
    """Defaults: 1%/year mortality, 0 lapse, 0 discount, RA off, no expenses."""
    kw = dict(
        mortality_q     = 0.01,
        lapse_q         = 0.0,
        discount_annual = 0.0,
        ra_confidence   = 0.75,
        mortality_cv    = 0.0,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


# ---------------------------------------------------------------------------
# Sec.32 -- BEL is the present value of expected future cash flows
# ---------------------------------------------------------------------------

def test_sec32_bel_is_pv_of_future_cashflows():
    """Sec.32: BEL = PV(future cash outflows) - PV(future cash inflows).

    A 2-month, 1-policy contract with zero discount, 1% mortality, no lapse
    and no expenses, so every figure is derived by hand.
    """
    death_benefit = 1_000_000.0
    premium = 12_000.0
    term = 2
    q = 0.01

    res = measure(
        ModelPoints.single(
            issue_age=40, benefits={"DEATH": death_benefit},
            premium=premium, term_months=term,
            calculation_methods=PATTERNS,
        ),
        _flat_assumptions(),
    )
    inforce = np.array([1.0, 1.0 - q])
    deaths = inforce * q
    pv_claims = float(np.sum(deaths * death_benefit))
    pv_premiums = float(np.sum(inforce * premium))
    assert np.isclose(res.bel_path[0, 0], pv_claims - pv_premiums)


# ---------------------------------------------------------------------------
# Sec.34 + B65 -- contract boundary
# ---------------------------------------------------------------------------

def test_sec34_b65_contract_boundary():
    """Sec.34 / B65: cash flows outside the contract boundary are excluded.

    The projection horizon ends at ``term_months``; no premium, claim or
    expense cash flow is projected beyond it.
    """
    term = 12
    res = measure(
        ModelPoints.single(
            issue_age=40, benefits={"DEATH": 1_000_000.0},
            premium=12_000.0, term_months=term,
            calculation_methods=PATTERNS,
        ),
        _flat_assumptions(),
    )
    assert res.cashflows.premium_cf.shape == (1, term)
    assert res.cashflows.claim_cf.shape    == (1, term)
    assert res.cashflows.expense_cf.shape  == (1, term)


# ---------------------------------------------------------------------------
# Sec.37 -- RA addition to BEL
# ---------------------------------------------------------------------------

def test_sec37_ra_addition_to_bel():
    """Sec.37: FCF = BEL + RA; CSM and loss component derive from this sum.

    Identity at initial recognition: ``csm - loss_component = -(bel + ra)``.
    For a profitable group csm = -FCF and loss = 0; for an onerous group
    csm = 0 and loss = FCF. Either way the identity holds.
    """
    res = measure(
        ModelPoints.single(
            issue_age=40, benefits={"DEATH": 1_000_000.0},
            premium=12_000.0, term_months=24,
            calculation_methods=PATTERNS,
        ),
        _flat_assumptions(mortality_cv=0.10),
    )
    fcf = res.bel_path[0, 0] + res.ra_path[0, 0]
    assert np.isclose(res.csm_path[0, 0] - res.loss_component[0], -fcf)


# ---------------------------------------------------------------------------
# Sec.38 -- initial CSM and loss component
# ---------------------------------------------------------------------------

def test_sec38_initial_csm_profitable():
    """Sec.38(b): for a profitable group CSM_0 = max(0, -FCF), loss = 0."""
    res = measure(
        ModelPoints.single(
            issue_age=35, benefits={"DEATH": 1_000_000.0},
            premium=15_000.0, term_months=36,
            calculation_methods=PATTERNS,
        ),
        _flat_assumptions(),
    )
    fcf = res.bel_path[0, 0] + res.ra_path[0, 0]
    assert fcf < 0.0                              # profitable -> negative FCF
    assert np.isclose(res.csm_path[0, 0], -fcf)
    assert res.loss_component[0] == 0.0


def test_sec38_loss_component_onerous():
    """Sec.38(c): for an onerous group CSM_0 = 0, loss component = max(0, FCF)."""
    res = measure(
        ModelPoints.single(
            issue_age=40, benefits={"DEATH": 1_000_000.0},
            premium=100.0,                  # premium far too low
            term_months=12,
            calculation_methods=PATTERNS,
        ),
        _flat_assumptions(
            mortality_annual=lambda sex, issue_age, duration:
                np.full(issue_age.shape, _annual(0.05)),
        ),
    )
    fcf = res.bel_path[0, 0] + res.ra_path[0, 0]
    assert fcf > 0.0                              # onerous -> positive FCF
    assert res.csm_path[0, 0] == 0.0
    assert np.isclose(res.loss_component[0], fcf)


# ---------------------------------------------------------------------------
# Sec.44 -- subsequent measurement of CSM
# ---------------------------------------------------------------------------

def test_sec44_csm_accretion_at_locked_in_rate():
    """Sec.44(b): CSM accretes interest at the discount rate locked in at inception.

    The roll-forward decomposes as ``csm[t+1] = csm[t] + accretion[t] -
    release[t]``; accretion is opening CSM times the locked-in monthly rate.
    """
    basis = _flat_assumptions(discount_annual=0.06)
    res = measure(
        ModelPoints.single(
            issue_age=35, benefits={"DEATH": 1_000_000.0},
            premium=15_000.0, term_months=36,
            calculation_methods=PATTERNS,
        ),
        basis,
    )
    opening = res.csm_path[:, :-1]
    closing = res.csm_path[:, 1:]
    assert np.array_equal(closing, opening + res.csm_accretion - res.csm_release)
    assert np.array_equal(res.csm_accretion, opening * basis.discount_monthly)


def test_sec44_b119_csm_release_proportional_to_coverage_units():
    """Sec.44(e) + B119: CSM released in proportion to coverage units provided.

    The engine uses in-force as the coverage-unit series. With zero mortality
    and zero lapse the in-force stays constant, so each month's release is
    the same fraction of the opening CSM and the CSM runs off linearly to
    zero. With zero discount no interest accretes, making the release per
    month exactly ``csm_0 / term``.
    """
    term = 3
    basis = _flat_assumptions(
        mortality_annual=lambda sex, issue_age, duration: np.zeros(issue_age.shape),
        lapse_annual=lambda sex, issue_age, duration: np.zeros(duration.shape),
    )
    res = measure(
        ModelPoints.single(
            issue_age=40, benefits={"DEATH": 1_000_000.0},
            premium=12_000.0, term_months=term,
            calculation_methods=PATTERNS,
        ),
        basis,
    )
    csm0 = res.csm_path[0, 0]
    assert csm0 > 0.0                             # profitable -- premium, no claims
    expected_release = csm0 / term
    assert np.allclose(res.csm_release[0], expected_release)
    assert np.isclose(res.csm_path[0, -1], 0.0)


# ---------------------------------------------------------------------------
# B96 -- discount rate reflects characteristics of the cash flows
# ---------------------------------------------------------------------------

def test_b96_higher_discount_reduces_pv_of_claims():
    """B96: a higher discount rate reduces the PV of future cash flows.

    Computed directly on a single cash-flow component so the mechanic is
    isolated from any premium / claim balance effect: the PV of the
    projected death-claim cash flow at 10% annual discount is strictly
    less than at 0%. The same contract under both bases produces an
    identical ``claim_cf`` series (in-force, mortality and benefit are
    unchanged), so only the discount factors differ.
    """
    kwargs = dict(
        issue_age=40, benefits={"DEATH": 1_000_000.0},
        premium=12_000.0, term_months=60,
    )
    res_lo = measure(ModelPoints.single(**kwargs, calculation_methods=PATTERNS), _flat_assumptions(discount_annual=0.0))
    res_hi = measure(ModelPoints.single(**kwargs, calculation_methods=PATTERNS), _flat_assumptions(discount_annual=0.10))
    pv_claims_lo = float(np.sum(res_lo.cashflows.claim_cf[0] * res_lo.discount_factor_mid))
    pv_claims_hi = float(np.sum(res_hi.cashflows.claim_cf[0] * res_hi.discount_factor_mid))
    assert pv_claims_hi < pv_claims_lo
