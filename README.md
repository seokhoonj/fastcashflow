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
- **Phase 2** -- mid-month discounting of claims and expenses, CSM
  movement detail (per-month interest accretion).
- **Phase 3a** -- numba parallel (`@njit` + `prange`) kernels.
- **Phase 3 (fusion)** -- `value()`: a single fused kernel that
  materialises no per-month arrays and derives BEL / RA / CSM in the
  same pass -- the memory-minimal fast path.
- **Phase 3b** -- polars file I/O (parquet / CSV); a chunked streaming
  path (`value_file`) values portfolios past what memory holds, to ~1e9
  model points and beyond.
- **Phase 4** -- BEL / RA / CSM roll-forward: month-by-month liability
  trajectories with the CSM movement decomposition.
- **GPU backend** -- `value(..., backend="gpu")` runs the same kernel
  on a CUDA device (optional; requires a CUDA GPU).

Beyond the phase plan:

- **Measurement** -- all three IFRS 17 models: the GMM (BEL / RA / CSM with
  roll-forward), the PAA (the simplified model for short-coverage contracts)
  and the VFA (the variable-fee model for direct-participation /
  account-value contracts).
- **Disclosure** -- the IFRS 17 insurance service result (insurance revenue,
  service expense, finance expense), the loss component and the CSM analysis
  of change, assembled from a GMM, PAA or VFA measurement.
- **Analysis of change** -- `roll_forward` slices a GMM or PAA measurement
  into reporting-period movements (the BEL, RA and CSM, or the liability
  for remaining coverage), recognising assumption revisions and in-force
  experience as CSM adjustments; `reconcile` aggregates them into IFRS 17
  reconciliation tables -- the period-close reporting cycle.
- **Reinsurance** -- reinsurance contracts held, measured as a quota-share
  treaty over a direct portfolio; the CSM carries the net cost or gain of
  the cover.
- **Products** -- term and whole life, endowment, pure endowment, immediate
  annuity, and health (inpatient, surgery, outpatient, diagnosis), built as
  a variable-length coverage list per policy.
- **Risk Adjustment** -- the confidence-level and cost-of-capital methods,
  with separate mortality, morbidity and longevity components, and an
  expense-risk component for account-value contracts.
- **Pricing** -- `solve_premium` solves the level premium for a break-even,
  margin or target-CSM objective.
- **Stochastic** -- `value_stochastic` values a portfolio under many
  economic scenarios -- a flat rate or a full discount-rate curve each --
  and reports the liability distribution, for the percentile-based risk and
  capital measures a single run cannot give.
- **Guarantees** -- `measure_tvog` values a VFA minimum-rate guarantee over
  return scenarios, splitting its cost into intrinsic value and time value
  (TVOG); `measure_vfa` folds that time value into the BEL, where the CSM
  absorbs it.

Further out: a non-financial-risk adjustment for guarantees.

## Quick start

```python
import numpy as np
from fastcashflow import Assumptions, ModelPointSet, measure

asmp = Assumptions(
    mortality_monthly=lambda issue_age, duration: np.full(
        issue_age.shape, 1.0 - (1.0 - 0.001) ** (1.0 / 12.0)
    ),
    lapse_monthly=lambda duration: np.full(duration.shape, 0.01),
    discount_annual=0.03,
    expense_acquisition=300_000.0,
    expense_maintenance_annual=60_000.0,
    expense_inflation=0.02,
    ra_confidence=0.75,
    mortality_cv=0.10,
)
mps = ModelPointSet.single(
    issue_age=40, death_benefit=100_000_000,
    monthly_premium=70_000, term_months=120,
)
res = measure(mps, asmp)   # mps: model points, asmp: assumptions
print(res.bel[0, 0], res.ra[0, 0], res.csm[0, 0])   # [model point, month]
```

`measure()` returns the full detail -- cash flows and the BEL / RA / CSM
roll-forward. For portfolio-scale valuation use `value()`: it returns only
the headline numbers (BEL, RA, CSM, loss component) per model point.

```python
from fastcashflow import value

val = value(mps, asmp)                      # parallel CPU kernel
val_gpu = value(mps, asmp, backend="gpu")   # CUDA device, if available
print(val.bel, val.ra, val.csm, val.loss_component)
```

The product is a combination of benefits -- a positive `maturity_benefit`
makes the contract an endowment, and `solve_premium` prices it:

```python
from fastcashflow import solve_premium

endowment = ModelPointSet.single(
    issue_age=40, death_benefit=100_000_000,
    monthly_premium=0, term_months=120, maturity_benefit=50_000_000,
)
premium = solve_premium(endowment, asmp, margin=0.10)   # 10% profit margin
```

At portfolio scale, read model points from a parquet or CSV file and
write the results back:

```python
from fastcashflow import read_model_points, write_valuation

mps = read_model_points("policies.parquet")
val = value(mps, asmp)
write_valuation(val, "results.parquet")     # pass ids=... to keep a join key
```

Past what fits in memory, stream a parquet file chunk by chunk straight
to a result dataset:

```python
from fastcashflow import value_file

value_file("policies.parquet", "results/", asmp, id_column="id")
# -> results/part-00000.parquet, part-00001.parquet, ...
```

## Performance

`value()` carries the in-force amount as a scalar through the time loop
and derives BEL / RA / CSM in one pass, so no per-month or intermediate
arrays are materialised. Measured on an 8-core / 16-thread desktop CPU
(Ryzen 7 3700X), 120-month projection:

| Model points | `value()` |
|---|---|
| 1,000,000 | 0.05 s |
| 5,000,000 | 0.30 s |

That is roughly 2 billion cell-updates per second (one cell = one model
point x one month). Run `examples/benchmark.py` to reproduce.

File I/O scales on the same budget: a 10M-model-point parquet round-trip
-- read, value, write -- takes about one second. Past what memory holds,
`value_file` streams a parquet file chunk by chunk -- 50M model points in
under five seconds, peak memory one chunk -- so portfolio size is bounded
by disk and time, not RAM.

Stochastic valuation runs the same kernel once per scenario: 500 scenarios
over 1,000,000 model points complete in about 35 seconds -- seriatim
stochastic at a scale a slow engine cannot reach.

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
