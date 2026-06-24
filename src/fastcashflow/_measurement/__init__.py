"""Shared measurement layer -- the cross-model primitives the GMM / VFA / PAA /
reinsurance measurements all build on.

Private package (no public ``fcf.measurement`` namespace; the public API is
model-first: ``fcf.gmm`` / ``fcf.vfa`` / ``fcf.paa`` / ``fcf.reinsurance``).
Import the submodules directly:

* ``projection`` -- model-neutral cash-flow -> BEL / RA valuation core.
* ``inforce``    -- in-force rescale, state reconcile, surrender value.
* ``account``    -- universal-life account-chassis detection and roll inputs.
* ``recognition``-- IFRS 17 para-109 CSM recognition (GMM / VFA share it).

These were extracted from the GMM engine so the non-GMM models stop borrowing
from a GMM-named module; nothing here imports back from ``engine`` / ``_gmm``,
so the package sits at the base of the measurement import graph.
"""
