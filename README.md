# fastcashflow

A fast IFRS 17 GMM (General Measurement Model) cash flow projection engine.

Takes model points and actuarial assumptions, projects monthly cash flows, and
measures the insurance contract liability -- BEL, RA and CSM.

## Design

- **Speed first.** Model points are the vectorised axis; the time axis is a
  sequential loop (the in-force recursion is genuinely sequential in time).
  This is the opposite of per-model-point iteration and is what makes the
  engine fast at portfolio scale.
- **Fixed projection structure.** The GMM recursion is built into the engine;
  new products are added as code modules, not user-written formulas. The
  trade-off -- less flexibility -- buys raw speed.
- **From scratch.** All code is original. The methodology references the
  IFRS 17 standard (paragraphs) directly; no third-party code is copied.

## Status

- **Phase 0** -- single fixed-benefit protection product, deterministic
  projection, BEL / RA / CSM, validated against hand calculation.
- **Phase 1** -- confidence-level RA, acquisition + maintenance expenses.
- **Phase 3a** -- numba parallel (`@njit` + `prange`) kernels.
- **Phase 3 (fusion)** -- `value()`: a single fused kernel that
  materialises no per-month arrays -- the memory-minimal fast path.

Later phases: duration-based lapse and select-ultimate mortality, polars
I/O at scale, monthly roll-forward / movement analysis.

## Quick start

```python
import numpy as np
from fastcashflow import Assumptions, ModelPointSet, run

asmp = Assumptions(
    mortality_monthly=lambda ages: np.full(ages.shape, 1.0 - (1.0 - 0.001) ** (1.0 / 12.0)),
    lapse_monthly=0.01,
    discount_annual=0.03,
    expense_acquisition=300_000.0,
    expense_maintenance_annual=60_000.0,
    expense_inflation=0.02,
    ra_confidence=0.75,
    claims_cv=0.10,
)
mps = ModelPointSet.single(
    issue_age=40, sum_assured=100_000_000,
    monthly_premium=70_000, term_months=120,
)
res = run(mps, asmp)
print(res.bel[0], res.ra[0], res.csm0[0])
```

## License

MPL-2.0 (Mozilla Public License 2.0).
