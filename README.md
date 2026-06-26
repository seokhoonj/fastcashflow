# fastcashflow

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://github.com/seokhoonj/fastcashflow/blob/main/LICENSE)

An open-source IFRS 17 measurement engine in Python. Takes model points and a
valuation basis, projects monthly cash flows, and measures the insurance
contract liability — BEL, RA and CSM — under the GMM, PAA and VFA models.

The goal: an engine that matches enterprise platforms on speed and correctness,
so any actuary can open it, read the source, and run a real valuation with no
licence wall and no closed binaries.

## Installation

> [!WARNING]
> **Work in progress** — fastcashflow is in active development. The public API,
> namespace layout and numerical results may still change between commits, and it
> is not yet on PyPI. Install from GitHub for the latest, and pin a commit if you
> need a stable surface.

```bash
pip install git+https://github.com/seokhoonj/fastcashflow.git
```

Requires Python 3.10 or newer. numpy, numba, polars and matplotlib install
automatically. (Not yet on PyPI.)

## Quick start

No files to prepare — measure the whole bundled sample portfolio. The sample
mixes segments (term life, whole life, health; several channels), so a dict
basis lets `measure` route each policy to its own segment's assumptions:

```python
import fastcashflow as fcf

# load the bundled sample inputs
basis = fcf.samples.basis()   # basis = {(product, channel): Basis}
mp    = fcf.samples.model_points()

# measure the whole portfolio -- each policy uses its segment's assumptions
# (full=False is the fast headline-only path; full=True also works on a dict
#  basis and returns the full trajectories)
val = fcf.gmm.measure(mp, basis, full=False)
print(f"model points : {val.bel.shape[0]:>15,}")
print(f"BEL          : {val.bel.sum():>15,.0f}")
print(f"RA           : {val.ra.sum():>15,.0f}")
print(f"CSM          : {val.csm.sum():>15,.0f}")
```

```text
model points :              11
BEL          :     -10,182,300
RA           :       1,309,817
CSM          :      10,280,704
```

Or build a single contract by hand and measure it in full detail:

```python
import numpy as np
import fastcashflow as fcf

# mortality -- flat 0.1% annual rate (same for every sex/age/duration)
mortality_fn = lambda sex, issue_age, duration: np.full(issue_age.shape, 0.001)

# lapse -- flat 1% annual rate
lapse_fn = lambda sex, issue_age, duration: np.full(duration.shape, 0.01)

# the valuation basis (mortality / lapse / discount / risk adjustment)
basis = fcf.Basis(
    mortality_annual = mortality_fn,   # in-force decrement (mortality_fn above)
    lapse_annual     = lapse_fn,       # lapse rate (lapse_fn above)
    discount_annual  = 0.03,           # annual discount rate
    ra_confidence    = 0.75,           # risk-adjustment confidence level (75th pct)
    mortality_cv     = 0.10,           # mortality coefficient of variation
    coverages        = (
        fcf.CoverageRate("DEATH", mortality_fn),  # one death coverage (claim rate = mortality_fn)
    ),
)

# one policy -- age 40, 100M death benefit, 70k monthly premium, 10-year term
mp = fcf.ModelPoints.single(
    issue_age           = 40,                                      # age at inception
    benefits            = {"DEATH": 100_000_000},                  # 100M DEATH benefit
    premium             = 70_000,                                  # monthly premium
    term_months         = 120,                                     # 10-year term (in months)
    calculation_methods = {"DEATH": fcf.CalculationMethod.DEATH},  # coverage code -> method
)

m = fcf.gmm.measure(mp, basis)
print(m)
```

```text
<gmm.Measurement -- 1 model point>
                   BEL            RA           CSM          loss
    mp 0    -6,092,691        55,484     6,037,206             0
   Total    -6,092,691        55,484     6,037,206             0
```

`measure(mp, basis)` returns the full per-month detail; `measure(mp, basis,
full=False)` returns only the headline BEL / RA / CSM per policy, on a numba
parallel kernel that is far faster at portfolio scale.

## Features

- **IFRS 17 models** — GMM (BEL / RA / CSM), PAA and VFA (variable-fee /
  account-value contracts with GMDB / GMAB guarantees).
- **Projection** — deterministic monthly cash flows; select-and-ultimate
  mortality, duration-based lapse, mid-month discounting, α / β / γ expenses,
  surrender value, contract states (active / waiver / paid-up).
- **Reporting** — roll-forward, reconciliation tables, insurance service result,
  loss component, aggregation to IFRS 17 unit of account.
- **I/O** — Excel workbook basis, polars parquet / CSV model points,
  `gmm.measure_stream` for portfolios larger than RAM.
- **More** — reinsurance, stochastic valuation, premium pricing, TVOG, first-adoption
  transition, GPU backend (`backend="gpu"`).

## Performance

Measured on an 8-core desktop (Ryzen 7 3700X), 120-month projection, a
single-coverage term-life portfolio (`examples/benchmark.py`); multi-coverage
portfolios scale roughly linearly in the coverage count:

| Model points | `measure(full=False)` |
|---|---|
| 1,000,000 | 0.07 s |
| 5,000,000 | 0.41 s |

`measure(full=False)` carries in-force as a scalar and materialises no
intermediate arrays. A 10M-row parquet round-trip — read, measure, write —
takes about 2.5 seconds, of which the measurement itself is under one second.
Run `examples/benchmark.py` to reproduce on your machine.

## Documentation

Full tutorial and API reference: <https://docs.fastcashflow.org>

Live demo: <https://demo.fastcashflow.org>

## License

Mozilla Public License 2.0 — see
[LICENSE](https://github.com/seokhoonj/fastcashflow/blob/main/LICENSE).
