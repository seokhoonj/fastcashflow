"""Universal-life trace + the bundled ``"ul"`` sample template.

``gmm.trace`` is GMM's calculation trace; for an account-backed universal-life
contract (``cashflows.account is not None``) it grows an extra "Universal-life
account" section -- the account value carried forward, the COI charged, the net
amount at risk it prices, the in-force-weighted fund and the death = max(av_mid,
face) top-up. These tests pin that the section renders for a UL contract, that a
non-account GMM contract's trace is byte-identical (the section is gated off),
and that the bundled ``samples.model_points("ul")`` / ``basis("ul")`` measure
cleanly through both ``gmm.measure`` and ``vfa.measure``.
"""
import io

import numpy as np

import fastcashflow as fcf


def _trace_text(mp, basis, index=0):
    buf = io.StringIO()
    fcf.gmm.trace(index, mp, basis, file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# The "ul" sample template -- load + measure end to end.
# ---------------------------------------------------------------------------

def test_ul_template_listed():
    assert "ul" in fcf.samples.templates()


def test_ul_sample_measures_through_gmm_and_vfa():
    mp = fcf.samples.model_points("ul")
    basis = fcf.samples.basis("ul")
    g = fcf.gmm.measure(mp, basis)
    v = fcf.vfa.measure(mp, basis)
    # Both paths produce finite headline numbers for every contract.
    for m in (g, v):
        assert np.all(np.isfinite(m.bel))
        assert np.all(np.isfinite(m.ra))
        assert np.all(np.isfinite(m.csm))
        assert np.all(m.csm >= 0.0)
    # The sample is an account book -- the GMM measurement carries the sidecar.
    assert g.cashflows.account is not None
    assert isinstance(g, fcf.gmm.Measurement)
    assert isinstance(v, fcf.vfa.Measurement)
    # GMM (locked-in discount) and VFA (underlying return) differ on BEL.
    assert not np.allclose(g.bel, v.bel)


def test_ul_sample_export_is_load_only():
    import pytest
    with pytest.raises(NotImplementedError):
        fcf.samples.export("/tmp/should_not_be_written", template="ul")


# ---------------------------------------------------------------------------
# gmm.trace -- the universal-life account section.
# ---------------------------------------------------------------------------

def test_gmm_trace_ul_shows_account_section():
    mp = fcf.samples.model_points("ul")
    basis = fcf.samples.basis("ul")
    text = _trace_text(mp, basis, 0)
    # The new account section header and its rows.
    assert "Universal-life account" in text
    assert "account_value0" in text
    assert "av=" in text
    assert "av_mid=" in text
    assert "coi=" in text
    assert "nar=" in text
    assert "fund=" in text
    # The death top-up rule is spelled out.
    assert "death = max(av_mid, face)" in text


def test_gmm_trace_ul_runs_for_every_row():
    mp = fcf.samples.model_points("ul")
    basis = fcf.samples.basis("ul")
    for i in range(mp.n_mp):
        text = _trace_text(mp, basis, i)
        assert "Universal-life account" in text


def test_gmm_trace_ul_output_is_ascii():
    mp = fcf.samples.model_points("ul")
    basis = fcf.samples.basis("ul")
    text = _trace_text(mp, basis, 0)
    text.encode("ascii")  # raises if any non-ASCII slipped into the output


def test_gmm_trace_non_account_has_no_account_section():
    # A plain protection contract is not an account book -- the account section
    # must be entirely absent (the gate keeps the existing output unchanged).
    mp = fcf.samples.model_points("gmm")
    basis = fcf.samples.basis("gmm")
    text = _trace_text(mp, basis, 0)
    assert "Universal-life account" not in text
    assert "account_value0" not in text


# ---------------------------------------------------------------------------
# The "ul-annuity" sample template + the annuitization trace section.
# ---------------------------------------------------------------------------

def test_ul_annuity_template_listed():
    assert "ul-annuity" in fcf.samples.templates()


def test_ul_annuity_sample_measures_through_gmm():
    mp = fcf.samples.model_points("ul-annuity")
    basis = fcf.samples.basis("ul-annuity")
    m = fcf.gmm.measure(mp, basis)
    assert np.all(np.isfinite(m.bel))
    assert np.all(np.isfinite(m.ra))
    assert np.all(np.isfinite(m.csm))
    assert np.all(m.csm >= 0.0)
    # It is an account book -- the GMM measurement carries the account sidecar.
    assert m.cashflows.account is not None
    # Annuitizing contracts pay a survival income (phase 2) and no maturity lump.
    assert np.all(m.cashflows.annuity_cf.sum(axis=1) > 0.0)
    assert np.allclose(m.cashflows.maturity_cf, 0.0)
    assert isinstance(m, fcf.gmm.Measurement)


def test_ul_annuity_sample_export_is_load_only():
    import pytest
    with pytest.raises(NotImplementedError):
        fcf.samples.export("/tmp/should_not_be_written", template="ul-annuity")


def test_gmm_trace_ul_annuity_shows_annuitization_section():
    mp = fcf.samples.model_points("ul-annuity")
    basis = fcf.samples.basis("ul-annuity")
    text = _trace_text(mp, basis, 0)
    # The conversion section header and its load-bearing rows.
    assert "Universal-life annuitization (conversion + payout)" in text
    assert "annuitization_months" in text
    assert "balance at conversion" in text
    assert "GMAB floor" in text
    assert "converted_balance" in text
    assert "locked_annuity_payment" in text
    # The conversion arithmetic is spelled out.
    assert "(= max(balance, GMAB))" in text
    assert "(= converted_balance x rate)" in text
    # The account section flags that the roll stops at conversion.
    assert "annuitizes at t=180m" in text


def test_gmm_trace_ul_annuity_output_is_ascii():
    mp = fcf.samples.model_points("ul-annuity")
    basis = fcf.samples.basis("ul-annuity")
    for i in range(mp.n_mp):
        _trace_text(mp, basis, i).encode("ascii")  # raises on non-ASCII


def test_gmm_trace_non_annuitizing_ul_has_no_annuitization_section():
    # A plain (non-annuitizing) UL account book grows the account section but
    # NOT the annuitization section -- the latter is gated on a conversion month.
    mp = fcf.samples.model_points("ul")
    basis = fcf.samples.basis("ul")
    text = _trace_text(mp, basis, 0)
    assert "Universal-life account" in text
    assert "Universal-life annuitization" not in text
    assert "annuitizes at t=" not in text
