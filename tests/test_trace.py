"""Sanity tests for ``show_trace`` / ``show_trace_diff`` -- the per-mp
calculation walk and the two-basis comparison."""
import io
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.basis import Basis, BasisRouter
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.trace import (
    _resolve_basis,
    show_trace_bel_step, show_trace_csm_step, show_trace, show_trace_diff, show_trace_paa,
    show_trace_vfa,
)


def _shock_mortality(rate_fn, factor: float):
    """Wrap a rate callable to multiply its return by ``factor``.

    Preserves ``_fcf_table_id`` and appends a modifier tag so the diff
    can attribute the change to mortality.
    """
    def wrapped(sex, issue_age, duration, issue_class, elapsed):
        return rate_fn(sex, issue_age, duration, issue_class, elapsed) * factor
    wrapped._fcf_table_id = getattr(rate_fn, "_fcf_table_id", None)
    wrapped._fcf_modifiers = (
        getattr(rate_fn, "_fcf_modifiers", ()) + (f"x{factor:g}",)
    )
    return wrapped


def _basis():
    return fcf.samples.basis()


def _portfolio():
    return fcf.samples.model_points()


def test_show_trace_renders_all_sections():
    """The eight headline tree sections all appear in the output."""
    buf = io.StringIO()
    fcf.gmm.trace(0, _portfolio(), _basis(), file=buf)
    text = buf.getvalue()
    for section in (
        "Basis (segment-level)",
        "Coverages",
        "Rates (annual",
        "Cash flows",
        "Discount factors",
        "BEL roll-forward",
        "CSM roll-forward",
        "Final",
    ):
        assert section in text, f"missing section: {section}"


def test_show_trace_emits_diagnosis_pool_only_when_present():
    """The "Undiagnosed share" node appears for portfolios with a
    DIAGNOSIS-pattern coverage and is omitted otherwise."""
    # DIAGNOSIS coverage: sample has CANCER on every MP.
    buf = io.StringIO()
    fcf.gmm.trace(0, _portfolio(), _basis(), file=buf)
    text_with = buf.getvalue()
    assert "Undiagnosed share" in text_with
    assert "'CANCER':" in text_with

    # DEATH-only: no DIAGNOSIS, so the node is suppressed.
    death_fn = lambda s, a, d: np.full(a.shape, 0.001)
    mp_death = fcf.ModelPoints.single(
        issue_age=40, benefits={0: 1_000_000},
        premium=100, term_months=12,
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH},
    )
    basis_death = Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda s, a, d: np.full(d.shape, 0.0),
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.0,
        coverages=(fcf.CoverageRate("DEATH", death_fn),),
    )
    buf = io.StringIO()
    fcf.gmm.trace(0, mp_death, basis_death, file=buf)
    assert "Undiagnosed share" not in buf.getvalue()


def test_show_trace_undiagnosed_matches_hand_calc():
    """The undiagnosed scalar depletes by (1 - monthly_q) each month --
    a single coverage with a flat annual rate must reproduce the
    closed-form (1 - monthly_q)**t at every key month the tree prints."""
    annual_q = 1 - (1 - 0.01) ** 12        # monthly q = 0.01
    cancer_fn = lambda s, a, d: np.full(a.shape, annual_q)
    no_decr = lambda s, a, d: np.full(a.shape, 0.0)
    basis = Basis(
        mortality_annual=no_decr, lapse_annual=no_decr,
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.0,
        coverages=(fcf.CoverageRate("CANCER", cancer_fn),),
    )
    mp = fcf.ModelPoints.single(
        issue_age=40, benefits={0: 1_000_000},
        premium=0, term_months=60,
        calculation_methods={"CANCER": fcf.CalculationMethod.DIAGNOSIS},
    )
    buf = io.StringIO()
    fcf.gmm.trace(0, mp, basis, file=buf)
    text = buf.getvalue()

    # Expected: undiagnosed(t) = (1 - 0.01)**t -- closed-form
    for t in (0, 12, 60):
        expected = (1.0 - 0.01) ** t
        assert f"t={t:>4d}m: undiagnosed={expected:.6f}" in text, (
            f"missing or wrong undiagnosed at t={t}m"
        )


def test_show_trace_routes_dict_basis_by_segment():
    """Passing the read_basis dict picks the right segment from
    the model point's (product, channel)."""
    mp = _portfolio()
    basis = _basis()
    buf = io.StringIO()
    fcf.gmm.trace(0, mp, basis, file=buf)
    text = buf.getvalue()
    seg = f"({mp.product[0]}/{mp.channel[0]}"
    assert seg in text


def test_show_trace_accepts_single_assumptions():
    """A plain :class:`Basis` (not a dict) bypasses the segment
    lookup and is used directly."""
    mp = _portfolio()
    basis = _basis().resolve((str(mp.product[0]), str(mp.channel[0])))
    buf = io.StringIO()
    fcf.gmm.trace(0, mp, basis, file=buf)
    assert "Basis (segment-level)" in buf.getvalue()


def test_show_trace_bel_and_ra_agree_with_measure():
    """The headline numbers printed in the tree match :func:`measure`
    on the same portfolio for the same row -- the trace is just a view,
    not a recalculation."""
    mp = _portfolio()
    basis = _basis().resolve((str(mp.product[0]), str(mp.channel[0])))
    m = fcf.gmm.measure(mp.subset([0]), basis)
    buf = io.StringIO()
    fcf.gmm.trace(0, mp, basis, file=buf)
    text = buf.getvalue()
    assert f"{m.bel_path[0, 0]:,.2f}" in text
    assert f"{m.ra_path[0, 0]:,.2f}" in text


def test_show_trace_rejects_out_of_range_index():
    mp = _portfolio()
    with pytest.raises(IndexError, match="mp_index"):
        fcf.gmm.trace(mp.n_mp, mp, _basis(), file=io.StringIO())


def test_show_trace_dict_basis_requires_segment_columns():
    """A dict basis cannot be routed when model_points has no product /
    channel columns."""
    bare = ModelPoints(
        issue_age=np.array([35.0]),
        premium=np.array([50_000.0]),
        term_months=np.array([120]),
        benefits={0: np.array([100_000_000.0])},
    )
    with pytest.raises(ValueError, match="product / channel"):
        fcf.gmm.trace(0, bare, _basis(), file=io.StringIO())


def test_show_trace_dict_basis_unknown_segment_raises():
    """An unmapped (product, channel) is flagged with available keys."""
    mp = _portfolio()
    partial = BasisRouter(
        {k: v for k, v in _basis().segments.items() if k[0] != mp.product[0]},
        segment_axes=_basis().axes)
    if partial:                           # only meaningful when dict is shrinkable
        with pytest.raises(KeyError, match="no basis for segment"):
            fcf.gmm.trace(0, mp, partial, file=io.StringIO())


# ---------------------------------------------------------------------------
# show_trace_diff
# ---------------------------------------------------------------------------

def test_show_trace_diff_renders_all_sections():
    """The diff prints the seven headline sections plus the labels line."""
    mp = _portfolio()
    basis = _basis().resolve((str(mp.product[0]), str(mp.channel[0])))
    shocked = replace(basis, mortality_annual=_shock_mortality(
        basis.mortality_annual, 1.10,
    ))
    buf = io.StringIO()
    fcf.gmm.trace_diff(0, mp, basis, shocked,
                    label_a="baseline", label_b="mort+10%", file=buf)
    text = buf.getvalue()
    for section in (
        "labels:",
        "Assumption changes",
        "Rate deltas",
        "Cash flow deltas",
        "Discount factor deltas",
        "BEL deltas",
        "CSM deltas",
        "Final",
    ):
        assert section in text, f"missing section: {section}"
    assert "'baseline'" in text and "'mort+10%'" in text


def test_show_trace_diff_identical_basis_reports_no_changes():
    """Diffing a basis against itself surfaces no rate / cash-flow
    changes -- only the all-zero anchor-month and Final lines remain,
    and the change-only sections explicitly say so."""
    mp = _portfolio()
    basis = _basis().resolve((str(mp.product[0]), str(mp.channel[0])))
    buf = io.StringIO()
    fcf.gmm.trace_diff(0, mp, basis, basis, file=buf)
    text = buf.getvalue()
    assert "(no changes in tracked fields)" in text
    assert "(no rate changes at sampled years)" in text
    assert "(no cash flow changes)" in text


def test_show_trace_diff_mortality_shock_raises_claim_and_bel():
    """A +10% mortality shock increases claim cash flows and the BEL
    monotonically -- the propagation is visible in the printed diff. The
    shock applies to BOTH the in-force decrement and the death coverage's
    payment rate (the convention for the sample basis, where they share
    a mortality table)."""
    mp = _portfolio()
    basis = _basis().resolve((str(mp.product[0]), str(mp.channel[0])))
    shocked_mort = _shock_mortality(basis.mortality_annual, 1.10)
    # Shock the DEATH coverage too so the payment rate moves with the
    # decrement -- otherwise the higher decrement just lowers in-force
    # without raising claims.
    new_coverages = tuple(
        fcf.CoverageRate(r.code, _shock_mortality(r.rate, 1.10))
        if r.code in ("DEATH_GENERAL", "DEATH")
        else r
        for r in basis.coverages
    )
    shocked = replace(basis, mortality_annual=shocked_mort, coverages=new_coverages)
    ma = fcf.gmm.measure(mp.subset([0]), basis)
    mb = fcf.gmm.measure(mp.subset([0]), shocked)
    assert mb.cashflows.claim_cf.sum() > ma.cashflows.claim_cf.sum()
    assert mb.bel_path[0, 0] > ma.bel_path[0, 0]
    # And the diff renders without raising.
    buf = io.StringIO()
    fcf.gmm.trace_diff(0, mp, basis, shocked, file=buf)
    assert "mortality(annual)" in buf.getvalue()
    assert "+10.00%" in buf.getvalue()


def test_show_trace_diff_routes_dict_bases_independently():
    """Each basis (dict) is routed by the model point's segment, so
    comparing two segment-keyed dicts that map the row to the same
    Basis yields a no-change diff."""
    mp = _portfolio()
    basis = _basis()
    buf = io.StringIO()
    fcf.gmm.trace_diff(0, mp, basis, basis, file=buf)
    assert "(no changes in tracked fields)" in buf.getvalue()


def test_show_trace_diff_rejects_out_of_range_index():
    mp = _portfolio()
    with pytest.raises(IndexError, match="mp_index"):
        fcf.gmm.trace_diff(mp.n_mp, mp, _basis(), _basis(),
                        file=io.StringIO())


# ---------------------------------------------------------------------------
# show_trace_bel_step
# ---------------------------------------------------------------------------

def test_show_trace_bel_step_renders_recursion_and_steps():
    """The step view prints the recursion equation, the seed and at
    least the inception step."""
    buf = io.StringIO()
    fcf.gmm.trace_bel_step(0, _portfolio(), _basis(), file=buf)
    text = buf.getvalue()
    assert "BEL[t] = annuity[t] - premium[t]" in text
    assert "seed:" in text
    assert "t=   0" in text
    assert "Inception BEL" in text
    assert "residual" in text


def test_show_trace_bel_step_residuals_are_machine_zero():
    """The recomputed BEL[t] in every printed step must agree with the
    engine's BEL[t] to within float64 noise -- that is the contract the
    step view is supposed to surface."""
    buf = io.StringIO()
    fcf.gmm.trace_bel_step(0, _portfolio(), _basis(), file=buf)
    for line in buf.getvalue().splitlines():
        if "residual" not in line:
            continue
        # e.g. "... (residual +0.0000e+00)"
        token = line.rsplit("residual ", 1)[1].rstrip(")").strip()
        assert abs(float(token)) < 1e-6, line


def test_show_trace_bel_step_accepts_custom_months():
    """Passing ``months=`` overrides the default anchor set."""
    buf = io.StringIO()
    fcf.gmm.trace_bel_step(0, _portfolio(), _basis(),
                  months=[0, 24, 36], file=buf)
    text = buf.getvalue()
    assert "t=   0" in text
    assert "t=  24" in text
    assert "t=  36" in text
    assert "t=  12" not in text                 # not requested


def test_show_trace_bel_step_rejects_out_of_range_index():
    mp = _portfolio()
    with pytest.raises(IndexError, match="mp_index"):
        fcf.gmm.trace_bel_step(mp.n_mp, mp, _basis(), file=io.StringIO())


def test_show_trace_bel_step_seed_month_prints_only_the_seed():
    """At ``t = term`` the recursion has no below, so the step row
    states only the seed value, not a full equation expansion."""
    mp = _portfolio()
    term = int(mp.term_months[0])
    buf = io.StringIO()
    fcf.gmm.trace_bel_step(0, mp, _basis(), months=[term], file=buf)
    text = buf.getvalue()
    assert "seed -- no recursion below" in text
    # The component lines that only show up in a recursion expansion
    # ("tail piece", "mid-month piece", "recomputed BEL[t]") are absent
    # at the seed row.
    assert "tail piece" not in text
    assert "recomputed BEL[t]" not in text


# ---------------------------------------------------------------------------
# show_trace_csm_step
# ---------------------------------------------------------------------------

def _profitable_basis_and_mp():
    """A tiny single-cell profitable contract -- CSM > 0 at inception."""
    def mort(s, ia, d, ic, em):
        return np.full(d.shape, 0.0005)
    def lapse(s, ia, d, ic, em):
        return np.full(d.shape, 0.02)
    basis = Basis(
        mortality_annual=mort, lapse_annual=lapse,
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.05,
        coverages=(fcf.CoverageRate("DEATH", mort),),
    )
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        premium=np.array([200_000.0]),
        term_months=np.array([60]),
        benefits={0: np.array([100_000_000.0])},
    )
    return basis, mp


def test_show_trace_csm_step_renders_seed_and_steps():
    """Recursion equations, seed values and at least one step row print."""
    buf = io.StringIO()
    fcf.gmm.trace_csm_step(0, _portfolio(), _basis(), file=buf)
    text = buf.getvalue()
    assert "csm[t]   = csm[t-1] + accretion[t-1] - release[t-1]" in text
    assert "Seed (t = 0)" in text
    assert "t=   1" in text
    assert "End CSM" in text


def _onerous_basis_and_mp():
    """A tiny single-cell onerous contract -- FCF > 0, CSM = 0 throughout.

    Heavy mortality on a large death benefit with a token premium drives
    PV(claims) well above PV(premiums), so the contract is onerous at
    inception. Built explicitly so the test does not depend on any sample
    contract's profitability.
    """
    def mort(s, ia, d, ic, em):
        return np.full(d.shape, 0.01)
    def lapse(s, ia, d, ic, em):
        return np.full(d.shape, 0.02)
    basis = Basis(
        mortality_annual=mort, lapse_annual=lapse,
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.05,
        coverages=(fcf.CoverageRate("DEATH", mort),),
    )
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        premium=np.array([1_000.0]),          # far too low for the cover
        term_months=np.array([60]),
        benefits={0: np.array([100_000_000.0])},
    )
    return basis, mp


def test_show_trace_csm_step_onerous_notes_zero_throughout():
    """An onerous contract surfaces the explicit \"csm = 0 throughout\"
    note in the seed block."""
    basis, mp = _onerous_basis_and_mp()
    buf = io.StringIO()
    fcf.gmm.trace_csm_step(0, mp, basis, file=buf)
    assert "onerous contract -- csm = 0 throughout" in buf.getvalue()


def test_show_trace_csm_step_profitable_residuals_are_zero():
    """On a profitable contract every printed recursion step holds the
    ``csm[t-1] + acc - rel == csm[t]`` identity to float64 noise."""
    basis, mp = _profitable_basis_and_mp()
    buf = io.StringIO()
    fcf.gmm.trace_csm_step(0, mp, basis, months=[1, 12, 30, 60], file=buf)
    for line in buf.getvalue().splitlines():
        if "residual" not in line:
            continue
        token = line.rsplit("residual ", 1)[1].rstrip(")").strip()
        assert abs(float(token)) < 1e-6, line


def test_show_trace_csm_step_terminal_release_drains_the_csm():
    """At ``t = term`` the release fraction equals 1 and the CSM drops
    to (essentially) zero -- the boundary condition the kernel enforces."""
    basis, mp = _profitable_basis_and_mp()
    buf = io.StringIO()
    fcf.gmm.trace_csm_step(0, mp, basis, months=[60], file=buf)
    text = buf.getvalue()
    # The terminal release fraction prints exactly as "= 1.000000".
    assert "= 1.000000" in text
    # End CSM is essentially zero (printed with two decimals).
    assert "csm[60] =            0.00" in text


def test_show_trace_csm_step_rejects_out_of_range_index():
    mp = _portfolio()
    with pytest.raises(IndexError, match="mp_index"):
        fcf.gmm.trace_csm_step(mp.n_mp, mp, _basis(), file=io.StringIO())


# ---------------------------------------------------------------------------
# show_trace_vfa -- the VFA (account-value) tracer
# ---------------------------------------------------------------------------

def _vfa_setup():
    death_fn = lambda s, a, d: np.full(np.shape(d), 0.005)
    lapse_fn = lambda s, a, d: np.full(np.shape(d), 0.04)
    basis = Basis(
        mortality_annual=death_fn, lapse_annual=lapse_fn,
        discount_annual=0.03, ra_confidence=0.95, mortality_cv=0.10,
        expense_cv=0.10, investment_return=0.06, fund_fee=0.025,
    )
    mp = ModelPoints.single(
        40, 0.0, 120, account_value=1.0e8,
        minimum_death_benefit=1.02e8,
        minimum_accumulation_benefit=1.05e8,
    )
    return mp, basis


def test_show_trace_vfa_renders_and_matches_measure_vfa():
    """The VFA tracer renders its sections and shows the engine's CSM."""
    mp, basis = _vfa_setup()
    buf = io.StringIO()
    fcf.vfa.trace(0, mp, basis, file=buf)
    text = buf.getvalue()
    for section in ("VFA inputs", "Account value & in-force",
                    "Guarantee floors", "BEL / CSM trajectory",
                    "CSM roll-forward", "Final"):
        assert section in text, f"missing section: {section}"
    m = fcf.vfa.measure(mp, basis)
    assert f"{m.csm_path[0, 0]:,.2f}" in text          # trace shows the engine CSM
    assert f"{m.variable_fee[0]:,.2f}" in text     # and the variable fee


def test_show_trace_vfa_scenarios_show_tvog():
    """With return_scenarios the trace surfaces the (non-zero) guarantee TVOG."""
    mp, basis = _vfa_setup()
    rng = np.random.default_rng(7)
    scen = (1.06 ** (1 / 12) - 1) + 0.005 * rng.standard_normal((500, 120))
    buf = io.StringIO()
    fcf.vfa.trace(0, mp, basis, return_scenarios=scen, file=buf)
    text = buf.getvalue()
    m = fcf.vfa.measure(mp, basis, return_scenarios=scen)
    assert m.time_value[0] != 0.0
    assert f"{m.time_value[0]:,.2f}" in text       # TVOG shown matches the engine


def test_show_trace_vfa_rejects_out_of_range_index():
    mp, basis = _vfa_setup()
    with pytest.raises(IndexError, match="mp_index"):
        fcf.vfa.trace(mp.n_mp, mp, basis, file=io.StringIO())


# ---------------------------------------------------------------------------
# show_trace_paa -- the PAA (LRC / revenue / LIC) tracer
# ---------------------------------------------------------------------------

def test_show_trace_paa_renders_and_matches_measure_paa():
    """The PAA tracer renders its sections and shows the engine's numbers."""
    mp, basis = _portfolio(), _basis()
    buf = io.StringIO()
    fcf.paa.trace(0, mp, basis, file=buf)
    text = buf.getvalue()
    for section in ("PAA inputs", "LRC roll-forward",
                    "Insurance service result", "LIC", "Final"):
        assert section in text, f"missing section: {section}"
    sub = mp.subset([0])
    b = _resolve_basis(basis, mp, 0)
    m = fcf.paa.measure(sub, b)
    assert f"{m.loss_component[0]:,.2f}" in text         # onerous loss shown
    assert f"{float(m.revenue[0].sum()):,.2f}" in text    # total revenue


def test_show_trace_paa_rejects_out_of_range_index():
    mp, basis = _portfolio(), _basis()
    with pytest.raises(IndexError, match="mp_index"):
        fcf.paa.trace(mp.n_mp, mp, basis, file=io.StringIO())
