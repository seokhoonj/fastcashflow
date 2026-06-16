"""The bundled PAA sample -- a short-tail group-accident product whose claims
settle over four months.

The shipped sample is the regression anchor for the settlement-pattern paths on
real data: that PAA discounts incurred claims to their settlement dates exactly
as GMM does (commit 5be855f), and that the incurred-claims liability (LIC) holds
its settlement-tail residual past the coverage term (commit 245a7c0). Before
this sample no shipped basis set ``settlement_pattern`` at all, so both paths
were byte-identical no-ops on the bundled data.
"""
import os
import tempfile

import numpy as np

import fastcashflow as fcf


def test_paa_sample_carries_a_scalar_discount_and_settlement_pattern():
    """The basis is built for settlement: a 4-month run-off and a scalar
    discount (a per-year curve would be rejected with a settlement pattern)."""
    basis = fcf.samples.basis("paa")
    assert np.allclose(basis.settlement_pattern, [0.4, 0.3, 0.2, 0.1])
    assert np.asarray(basis.discount_annual).size == 1   # scalar, not a curve


def test_paa_sample_is_onerous_and_matches_gmm_on_settled_claims():
    """The sample is priced below break-even, so the PAA onerous test reports a
    loss; and that loss equals the GMM loss on the identical settled claims --
    both discount each claim to its settlement date (5be855f)."""
    basis = fcf.samples.basis("paa")
    mp = fcf.samples.model_points("paa")
    paa = fcf.paa.measure(mp, basis)
    gmm = fcf.gmm.measure(mp, basis)
    assert np.all(paa.loss_component > 0.0)               # onerous block
    assert np.allclose(paa.loss_component, gmm.loss_component)


def test_paa_sample_lic_holds_its_settlement_tail_past_term():
    """The incurred-claims liability keeps a positive residual past the 12-month
    coverage term -- claims incurred late still settle over the run-off, and the
    residual is held, not zero-padded (245a7c0)."""
    basis = fcf.samples.basis("paa")
    mp = fcf.samples.model_points("paa")
    lic_path = np.asarray(fcf.paa.measure(mp, basis, full=True).lic_path)
    assert lic_path.ndim == 2
    assert np.all(lic_path[:, -1] > 0.0)                       # tail residual held, not 0


def test_paa_sample_round_trips_through_export():
    """Exporting the template and reading it back reproduces the in-memory
    measure -- the settlement_pattern survives the .xlsx serialization."""
    basis = fcf.samples.basis("paa")
    mp = fcf.samples.model_points("paa")
    ref = fcf.paa.measure(mp, basis)
    with tempfile.TemporaryDirectory() as d:
        fcf.samples.export(d, template="paa", quiet=True)
        b2 = fcf.read_basis(os.path.join(d, "basis.xlsx")).resolve(("ACCIDENT_A", "GA"))
        mp2 = fcf.read_model_points(
            os.path.join(d, "policies.csv"),
            coverages=os.path.join(d, "coverages.csv"),
            calculation_methods=fcf.samples.calculation_methods(),
        )
        assert np.allclose(b2.settlement_pattern, basis.settlement_pattern)
        out = fcf.paa.measure(mp2, b2)
        assert np.allclose(out.loss_component, ref.loss_component)
