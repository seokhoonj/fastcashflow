"""IFRS 17 paragraph-109 maturity-band disclosure -- the G4 gate, pinned.

Paragraph 109 discloses, at the reporting date, when the CSM remaining at
period end is expected to be recognised in profit or loss, in maturity bands
(Samsung 2026Q1 XBRL: <=1y / 1-3y / 3-5y / 5y+). The settled question, from
dev/inforce-redesign-FINAL.md (gate G4) and the Samsung evidence:

  (1) allocation     band = csm_closing * (cu_in_band / cu_total)
                     -- the no-accretion coverage-unit split; bands SUM TO the
                        closing CSM. THIS is the disclosure form.
  (2) chain-runoff   project the closing CSM forward, accreting at the
                     locked-in rate and releasing by the B119 fraction; the
                     NOMINAL releases per band sum to MORE than the closing CSM.

Samsung's disclosed bands sum to the closing CSM total (0.90 + 1.46 + 1.18 +
10.35 = 13.89 = closing CSM), so the standard/practice form is (1). This test
pins that the two formulas genuinely diverge, that only (1) reconciles to the
closing CSM, and that the divergence IS the accretion (they coincide at a zero
lock-in rate). The cookbook recipe (workflow/settlement.md, doc-exec checked)
shows formula (1) on the bundled book.
"""
import numpy as np

import fastcashflow as fcf
from fastcashflow import ModelPoints
from conftest import PATTERNS, make_death_basis

BANDS = [(0, 12), (12, 36), (36, 60), (60, None)]


def _closing_and_forward_units(lock_in=0.06):
    """A profitable in-force contract: its closing CSM (the on-track carry, =
    csm_path at the valuation date) and the forward coverage-unit profile."""
    basis = make_death_basis(mortality_q=0.0015, lapse_q=0.004,
                             discount_annual=lock_in, ra_confidence=0.75,
                             mortality_cv=0.10)
    unit = ModelPoints.single(40, 600.0, 240, benefits={0: 100_000.0},
                              calculation_methods=PATTERNS)
    m = fcf.gmm.measure(unit, basis, full=True)
    elapsed = 24
    csm_closing = float(m.csm_path[0, elapsed])
    cu = m.cashflows.inforce[0, elapsed:240].astype(np.float64)
    assert csm_closing > 0.0, "fixture must be profitable (CSM > 0)"
    return csm_closing, cu, lock_in


def _allocation_bands(csm_closing, cu):
    """Formula (1): csm_closing * coverage-unit fraction per band."""
    total = cu.sum()
    return np.array([csm_closing * cu[lo:(len(cu) if hi is None else hi)].sum()
                     / total for lo, hi in BANDS])


def _chain_runoff_bands(csm_closing, cu, lock_in):
    """Formula (2): each month accrete at the locked-in rate FIRST, then release
    the post-accretion CSM by the B119 fraction -- the engine kernel order
    (numerics._csm_kernel: "the end-of-period, i.e. post-accretion, CSM is
    spread over the coverage units"). Accumulate the nominal release per band."""
    r_m = (1.0 + lock_in) ** (1.0 / 12.0) - 1.0
    cu_remaining = cu[::-1].cumsum()[::-1]
    release = np.zeros(cu.shape[0])
    csm = csm_closing
    for t in range(cu.shape[0]):
        accreted = csm * (1.0 + r_m)                 # accrete first
        rem = cu_remaining[t]
        rel = accreted * (cu[t] / rem) if rem > 0 else accreted
        release[t] = rel
        csm = accreted - rel                         # release the remainder
    return np.array([release[lo:(len(release) if hi is None else hi)].sum()
                     for lo, hi in BANDS])


def test_allocation_reconciles_to_the_closing_csm():
    """Formula (1) bands sum to the closing CSM -- it is an allocation of the
    remaining balance (matches Samsung's bands summing to the closing CSM)."""
    csm_closing, cu, _ = _closing_and_forward_units()
    band1 = _allocation_bands(csm_closing, cu)
    np.testing.assert_allclose(band1.sum(), csm_closing, rtol=1e-12)
    assert np.all(band1 > 0.0)


def test_chain_runoff_overstates_by_the_accretion():
    """Formula (2) sums to MORE than the closing CSM -- the excess is the future
    interest accretion, which the paragraph-109 allocation (1) excludes. So (2)
    also exceeds (1), whose bands sum exactly to the closing CSM."""
    csm_closing, cu, lock_in = _closing_and_forward_units()
    band1 = _allocation_bands(csm_closing, cu)
    band2 = _chain_runoff_bands(csm_closing, cu, lock_in)
    assert band2.sum() > csm_closing
    assert band2.sum() > band1.sum()                 # (2) > (1) directly
    np.testing.assert_allclose(band1.sum(), csm_closing, rtol=1e-12)
    # the overstatement is material, not a rounding wobble
    assert band2.sum() - csm_closing > 0.1 * csm_closing


def test_the_two_formulas_genuinely_diverge_per_band():
    """The choice is not academic: the bands differ materially, most in the long
    tail where accretion has compounded the longest."""
    csm_closing, cu, lock_in = _closing_and_forward_units()
    band1 = _allocation_bands(csm_closing, cu)
    band2 = _chain_runoff_bands(csm_closing, cu, lock_in)
    max_diff = np.max(np.abs(band2 - band1))
    assert max_diff > 0.01 * csm_closing
    # the long-tail band (5y+) is where they diverge most
    assert np.argmax(np.abs(band2 - band1)) == len(BANDS) - 1


def test_divergence_is_exactly_the_accretion():
    """At a zero locked-in rate (no accretion) the two formulas COINCIDE -- so
    the divergence is the accretion and nothing else, confirming that formula
    (1) is the no-accretion B119 runoff allocation."""
    csm_closing, cu, _ = _closing_and_forward_units(lock_in=0.0)
    band1 = _allocation_bands(csm_closing, cu)
    band2 = _chain_runoff_bands(csm_closing, cu, lock_in=0.0)
    np.testing.assert_allclose(band1, band2, rtol=1e-12)
