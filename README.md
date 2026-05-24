# fastcashflow

A fast IFRS 17 measurement engine in Python -- open source.

Takes model points and actuarial assumptions, projects monthly cash flows, and
measures the insurance contract liability (BEL, RA and CSM) under the
General Measurement Model. PAA and VFA are supported too.

**Goal**: an open-source measurement engine that matches enterprise platforms
on speed and correctness, so any actuary can open it, read the source, and
run a real valuation -- no licence wall, no closed binaries. The IFRS 17
standard is the only reference; everything is written from scratch with
hand-validated tests.

## Installation

Not yet on PyPI -- install directly from GitHub:

```bash
pip install git+https://github.com/seokhoonj/fastcashflow.git
```

The plotting helpers (`plot_liability`, `plot_csm_runoff`, ...) additionally
need matplotlib:

```bash
pip install "git+https://github.com/seokhoonj/fastcashflow.git#egg=fastcashflow[viz]"
```

fastcashflow requires Python 3.10 or newer. Its core dependencies -- numpy,
numba and polars -- install automatically.

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

## Features

fastcashflow measures the IFRS 17 insurance contract liability and the
reporting that surrounds it.

- **Measurement** -- all three IFRS 17 models: the GMM (BEL / RA / CSM),
  the PAA (the simplified model for short-coverage contracts) and the VFA
  (the variable-fee model for direct-participation / account-value
  contracts).
- **Projection** -- deterministic monthly cash flows; select-and-ultimate
  mortality, duration-based lapse, mid-month discounting of claims and
  expenses, and acquisition and maintenance expenses.
- **Assumption input layers** -- one workbook (`assumptions.xlsx`) with
  schema-detecting axis-flex base rate tables (`sex` / `age` /
  `issue_age` / `duration` columns, any subset), plus optional
  `ae_factors` (A/E multipliers, vendor-style runtime calibration),
  optional integer `age_shift` columns on segments, optional
  `improvement_tables` (mortality improvement scales), and curve-shaped
  discount / inflation / maintenance. Each layer is no-op when omitted,
  so a simple workbook stays simple.
- **Per-segment portfolios** -- `(product, channel)` model-point columns
  let `value_segmented(mp, basis)` route each row to its segment's
  `Assumptions` and stitch the results back to a single per-row
  `Valuation`.
- **Contract states** -- active, waiver of premium and paid-up. In-force is
  carried on an active track and a premium-waived track; a waiver-inception
  rate moves in-force from one to the other during the projection.
- **Risk Adjustment** -- the confidence-level and cost-of-capital methods,
  with separate mortality, morbidity and longevity components, and an
  expense-risk component for account-value contracts.
- **Roll-forward** -- month-by-month BEL / RA / CSM trajectories with the
  CSM movement decomposition; `roll_forward` and `reconcile` assemble the
  reporting-period analysis of change into IFRS 17 reconciliation tables.
- **Reporting** -- the IFRS 17 insurance service result (insurance
  revenue, service expense, finance expense), the loss component and the
  CSM analysis of change.
- **Aggregation** -- `group` re-expresses a measurement at the IFRS 17
  unit of account (portfolio x annual cohort x profitability bucket).
- **Transition** -- `transition` re-sets the CSM on the fair value
  approach for in-force contracts at first adoption.
- **Reinsurance** -- reinsurance contracts held, measured as a quota-share
  treaty over a direct portfolio.
- **Pricing** -- `solve_premium` solves the level premium for a
  break-even, margin or target-CSM objective.
- **Stochastic** -- `value_stochastic` values a portfolio under many
  economic scenarios and reports the liability distribution.
- **Guarantees** -- `measure_tvog` values a VFA minimum-rate guarantee,
  splitting its cost into intrinsic value and time value (TVOG).
- **Products** -- term and whole life, endowment, pure endowment,
  immediate annuity, and health (inpatient, surgery, outpatient,
  diagnosis), built as a variable-length coverage list per policy --
  each coverage able to carry a waiting or reduced-benefit period.
- **Speed and scale** -- numba parallel (`@njit` + `prange`) kernels and a
  fused `value()` path that materialises no per-month arrays; polars
  parquet / CSV I/O; a chunked `value_file` stream for portfolios past
  what memory holds; an optional CUDA GPU backend.

**Maturity.** The deterministic GMM core -- projection, BEL / RA / CSM,
`measure()` and `value()` -- is the most exercised path and is validated
against hand calculations. The wider surface (PAA, VFA, reinsurance,
reporting, roll-forward, stochastic) is implemented and tested, but the
package is pre-1.0 and its API may still change.

*Planned:* re-diagnosis benefits and a non-financial-risk adjustment on
the guarantee time value.

## Quick start

After installing, this runs with no files to prepare -- it measures
fastcashflow's bundled sample portfolio:

```python
import fastcashflow as fcf

model_points = fcf.load_sample_model_points()              # bundled sample portfolio
basis        = fcf.load_sample_assumptions()               # {(product, channel): Assumptions}
assumptions  = basis[("TERM_A", "GA")]                     # pick one segment
m            = fcf.measure(model_points, assumptions)
print(m.bel[:, 0], m.ra[:, 0], m.csm[:, 0])   # BEL / RA / CSM at issue
```

Outside the samples you build the two inputs yourself -- a set of model
points and the actuarial assumptions:

```python
import numpy as np
import fastcashflow as fcf

assumptions = fcf.Assumptions(
    mortality_annual=lambda sex, issue_age, duration: np.full(
        issue_age.shape, 0.001,
    ),
    lapse_annual=lambda sex, issue_age, duration: np.full(
        duration.shape, 0.01,
    ),
    discount_annual=0.03,
    expense_acquisition=300_000.0,
    expense_maintenance_annual=60_000.0,
    expense_inflation=0.02,
    ra_confidence=0.75,
    mortality_cv=0.10,
)
model_points = fcf.ModelPoints.single(
    issue_age=40, death_benefit=100_000_000,
    level_premium=70_000, term_months=120,
)
res = fcf.measure(model_points, assumptions)
print(res.bel[0, 0], res.ra[0, 0], res.csm[0, 0])   # [model point, month]
```

`measure()` returns the full detail -- cash flows and the BEL / RA / CSM
roll-forward. For portfolio-scale valuation use `value()`: it returns only
the headline numbers (BEL, RA, CSM, loss component) per model point.

```python
val     = fcf.value(model_points, assumptions)                  # parallel CPU kernel
val_gpu = fcf.value(model_points, assumptions, backend="gpu")   # CUDA device, if available
print(val.bel, val.ra, val.csm, val.loss_component)
```

The product is a combination of benefits -- a positive `maturity_benefit`
makes the contract an endowment, and `solve_premium` prices it:

```python
endowment = fcf.ModelPoints.single(
    issue_age=40, death_benefit=100_000_000,
    level_premium=0, term_months=120, maturity_benefit=50_000_000,
)
premium = fcf.solve_premium(endowment, assumptions, margin=0.10)   # 10% profit margin
```

At portfolio scale, read model points from a parquet or CSV file and
write the results back:

```python
model_points = fcf.read_model_points("policies.parquet", assumptions)
val = fcf.value(model_points, assumptions)
fcf.write_valuation(val, "results.parquet")   # pass ids=... to keep a join key
```

Past what fits in memory, stream a parquet file chunk by chunk straight
to a result dataset:

```python
fcf.value_file("policies.parquet", "results/", assumptions, id_column="id")
# -> results/part-00000.parquet, part-00001.parquet, ...
```

The [`examples/` directory](https://github.com/seokhoonj/fastcashflow/tree/main/examples)
has runnable scripts -- `quickstart.py`, `reporting.py`, `pricing.py` and
more. Each reads its inputs from the Excel files in `examples/data/`, so a
practitioner can value their own book by editing those files alone.

## Documentation

The full documentation -- a guided tutorial that builds up the IFRS 17
measurement step by step, and the API reference -- is at
<https://fastcashflow.readthedocs.io>.

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

Licensed under the Mozilla Public License 2.0 (MPL-2.0). See
[LICENSE](https://github.com/seokhoonj/fastcashflow/blob/main/LICENSE).
