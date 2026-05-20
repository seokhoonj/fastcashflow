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
- **Phase 1b** -- select-and-ultimate mortality, duration-based lapse.
- **Phase 3a** -- numba parallel (`@njit` + `prange`) kernels.
- **Phase 3 (fusion)** -- `value()`: a single fused kernel that
  materialises no per-month arrays and derives BEL / RA / CSM in the
  same pass -- the memory-minimal fast path.
- **GPU backend** -- `value(..., backend="gpu")` runs the same kernel
  on a CUDA device (optional; requires a CUDA GPU).

Later phases: polars I/O at scale, monthly roll-forward / movement
analysis.

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
res = run(mps, asmp)   # mps: model points, asmp: assumptions
print(res.bel[0], res.ra[0], res.csm0[0])
```

`run()` returns the full detail -- cash flow and CSM trajectories. For
portfolio-scale valuation use `value()`: it returns only the headline
numbers (BEL, RA, CSM, loss component) per model point and is much
faster.

```python
from fastcashflow import value

val = value(mps, asmp)                      # parallel CPU kernel
val_gpu = value(mps, asmp, backend="gpu")   # CUDA device, if available
print(val.bel, val.ra, val.csm, val.loss_component)
```

## Performance

`value()` carries the in-force amount as a scalar through the time loop
and derives BEL / RA / CSM in one pass, so no per-month or intermediate
arrays are materialised. Measured on an 8-core / 16-thread desktop CPU
(Ryzen 7 3700X), 120-month projection:

| Model points | `value()` |
|---|---|
| 1,000,000 | 0.045 s |
| 5,000,000 | 0.218 s |

That is roughly 2.7 billion cell-updates per second (one cell = one
model point x one month). Run `examples/benchmark.py` to reproduce.

## GPU backend

`value(..., backend="gpu")` runs the same kernel on a CUDA device.
Before reaching for it:

- **It needs a CUDA setup** -- a CUDA-capable GPU and driver, and a numba
  build with CUDA support. Without one, `backend="gpu"` raises; the
  default `backend="cpu"` always works.
- **The fixed cost is only amortised at scale.** Each call pays a kernel
  launch and a host-to-device transfer, so the GPU is slower than the CPU
  for small portfolios and roughly breaks even near a million model
  points.
- **The first call is slow** -- a one-time CUDA JIT compile and context
  initialisation of a few hundred milliseconds.
- **GPU memory bounds the portfolio** -- device arrays take ~64 bytes per
  model point, so an 8 GB card holds on the order of 100M.
- **Consumer cards give no speedup.** float64 throughput is deliberately
  capped on consumer GeForce hardware, so there the GPU only matches the
  CPU; the advantage shows on data-centre cards with full-rate float64.

## License

MPL-2.0 (Mozilla Public License 2.0).
