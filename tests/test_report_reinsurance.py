"""Reinsurance-held report validation -- hand-calc + reconcile tie-out.

``report`` on a reinsurance-held measurement returns a ReinsuranceReport (IFRS
17 paragraphs 82 + 86): premiums paid and amounts recovered disaggregated, the
service result built from the risk-transferred and CSM releases, and the
finance unwind split by source. The report's monthly figures must sum over a
reporting period to exactly the ReinsuranceReconciliation -- the report and the
reconciliation read off the same measurement.
"""
import fastcashflow as fcf
import numpy as np

from fastcashflow import ModelPoints
from fastcashflow.curves import forward_rates
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


Q = 0.002          # flat monthly mortality
LAPSE = 0.005      # flat monthly lapse
MORTALITY_CV = 0.10


def _basis():
    return make_death_basis(
        mortality_q     = Q,
        lapse_q         = LAPSE,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = MORTALITY_CV,
    )


def _measurement(term=60, premium=80_000.0, death_benefit=1e8, cession=0.4):
    return fcf.reinsurance.measure(
        ModelPoints.single(40, premium, term, benefits={"DEATH": death_benefit},
                           calculation_methods=PATTERNS),
        _basis(), treaty=fcf.reinsurance.QuotaShare(cession=cession),
    )


def test_reinsurance_report_field_formulas():
    """Hand-calc the report fields straight off the measurement arrays."""
    m = _measurement()
    rep = fcf.report(m)
    assert isinstance(rep, fcf.ReinsuranceReport)

    bel, ra, csm = m.bel_path, m.ra_path, m.csm_path
    discount_monthly = forward_rates(m.discount_bom)

    # Disaggregated cash flows are passed straight through (positive both).
    assert np.allclose(rep.reinsurance_premium_allocated, m.reinsurance_premium)
    assert np.allclose(rep.amounts_recovered, m.recovery)
    assert np.all(rep.reinsurance_premium_allocated >= 0.0)
    assert np.all(rep.amounts_recovered >= 0.0)

    # Net presentation property = recoveries - premiums (paragraph 86).
    assert np.allclose(
        rep.net_reinsurance_result, m.recovery - m.reinsurance_premium)

    # RA release the issuer revenue form: opening - closing discounted (the RA
    # interest is in the finance line, not the service result).
    monthly_discount = 1.0 / (1.0 + discount_monthly)
    ra_release = ra[:, :-1] - ra[:, 1:] * monthly_discount
    assert np.allclose(rep.ra_release, ra_release)

    # Service result = ra_release + csm_release (mirrors _report_gmm).
    assert np.allclose(
        rep.reinsurance_service_result, ra_release + m.csm_release)

    # Finance: interest on BEL + RA + CSM accretion, disaggregated by source.
    assert np.allclose(rep.bel_finance_expense, discount_monthly * bel[:, :-1])
    assert np.allclose(rep.ra_finance_expense, discount_monthly * ra[:, :-1])
    assert np.allclose(rep.csm_finance_expense, m.csm_accretion)
    assert np.allclose(
        rep.reinsurance_finance_expense,
        discount_monthly * (bel[:, :-1] + ra[:, :-1]) + m.csm_accretion)
    # The three parts sum to the aggregate (B130-B136).
    assert np.allclose(
        rep.bel_finance_expense + rep.ra_finance_expense
        + rep.csm_finance_expense,
        rep.reinsurance_finance_expense)

    # CSM analysis of change reconciles.
    assert np.allclose(rep.csm_opening, csm[:, :-1])
    assert np.allclose(rep.csm_closing, csm[:, 1:])
    assert np.allclose(
        rep.csm_opening + rep.csm_accretion - rep.csm_release, rep.csm_closing)


def test_reinsurance_report_csm_can_be_negative():
    """Ceding a profitable book has a net cost: a negative CSM, no loss component.

    Sec. 65 -- the reinsurance CSM is the net cost / gain of the cover and may
    be negative; the report carries the negative trajectory through with no
    loss-component floor (the ReinsuranceReport has no loss_component field).
    """
    m = _measurement(premium=300_000.0, cession=0.5)
    rep = fcf.report(m)
    assert m.bel[0] > 0.0                  # premiums ceded exceed recoveries
    assert rep.csm_opening[0, 0] < 0.0     # the net cost is a negative CSM
    assert rep.csm_closing[0, -1] <= 0.0   # stays a net cost through run-off
    assert not hasattr(rep, "loss_component")


def test_reinsurance_report_ties_out_to_reconciliation():
    """The presentation-independent report lines sum per period to the reconciliation.

    The report (a P&L view) and the reconciliation (a liability roll-forward)
    decompose the same opening->closing transition differently, so they share
    the lines that do not depend on that split: the finance lines (interest by
    source) and the CSM release. The RA run-off line legitimately differs -- the
    report's ``ra_release`` excludes interest (the issuer revenue form, with the
    RA interest in finance) whereas the reconciliation's is the movement residual
    (opening + interest - closing) -- so it is NOT asserted equal here, mirroring
    how the issuer GMM report does not tie its ``ra_release`` to its reconcile.
    """
    m = _measurement(term=60, cession=0.4)
    rep = fcf.report(m)

    period = 12
    movements = fcf.roll_forward(m, period)
    recs = fcf.reconcile(movements)

    for rec in recs:
        a, b = rec.month_start, rec.month_end

        # Finance lines (signed positive, an expense) tie out by source -- the
        # interest decomposition is the same in both views.
        assert np.isclose(
            rep.bel_finance_expense[:, a:b].sum(), rec.bel_finance)
        assert np.isclose(
            rep.ra_finance_expense[:, a:b].sum(), rec.ra_finance)
        assert np.isclose(
            rep.csm_finance_expense[:, a:b].sum(), rec.csm_finance)

        # The CSM release is shared (both read m.csm_release); the reconciliation
        # shows it negative (opening plus every row equals closing), hence the flip.
        assert np.isclose(rep.csm_release[:, a:b].sum(), -rec.csm_release)


def test_reinsurance_report_str_renders():
    """The annual table renders without error and lists the disaggregated rows."""
    rep = fcf.report(_measurement())
    text = str(rep)
    assert "reinsurance-held report" in text
    assert "Reinsurance premium" in text
    assert "Amounts recovered" in text
    assert "Service result" in text
