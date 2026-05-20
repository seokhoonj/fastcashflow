# fastcashflow

A fast IFRS 17 GMM (General Measurement Model) cash flow projection engine.

Takes model points and actuarial assumptions, projects monthly cash flows, and
measures the insurance contract liability — BEL, RA and CSM.

## Design

- **Speed first.** Model points are the vectorised axis; the time axis is a
  sequential loop (the in-force recursion is genuinely sequential in time).
  This is the opposite of per-model-point iteration and is what makes the
  engine fast at portfolio scale.
- **Fixed projection structure.** The GMM recursion is built into the engine;
  new products are added as code modules, not user-written formulas. The
  trade-off — less flexibility — buys raw speed.
- **From scratch.** All code is original. The methodology references the
  IFRS 17 standard (paragraphs) directly; no third-party code is copied.

## Status — Phase 0

Phase 0 is the correctness foundation:

- single fixed-benefit protection product (level premium)
- deterministic projection, no assumption changes
- BEL, a placeholder RA, and CSM (initial recognition + roll-forward)
- validated against an independent hand calculation (`tests/test_phase0.py`)

Later phases: proper RA methodology, multi-product, 1e8-row scale
(numba / polars), monthly roll-forward / movement analysis.

## Quick start

```python
import numpy as np
from fastcashflow import Assumptions, ModelPointSet, run

asmp = Assumptions(
    mortality_monthly=lambda ages: 1 - (1 - 0.001) ** (1 / 12),
    lapse_monthly=0.01,
    discount_annual=0.03,
    ra_rate=0.05,
)
mps = ModelPointSet.single(
    issue_age=40, sum_assured=100_000_000,
    monthly_premium=70_000, term_months=120,
)
res = run(mps, asmp)
print(res.bel[0], res.ra[0], res.csm0[0])
```

## License

TBD.
