# fastcashflow

[![PyPI](https://img.shields.io/pypi/v/fastcashflow)](https://pypi.org/project/fastcashflow/)
[![Python](https://img.shields.io/pypi/pyversions/fastcashflow)](https://pypi.org/project/fastcashflow/)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://github.com/seokhoonj/fastcashflow/blob/main/LICENSE)

An open-source IFRS 17 measurement engine in Python. Takes model points and
actuarial assumptions, projects monthly cash flows, and measures the insurance
contract liability — BEL, RA and CSM — under the GMM, PAA and VFA models.

The goal: an engine that matches enterprise platforms on speed and correctness,
so any actuary can open it, read the source, and run a real valuation with no
licence wall and no closed binaries.

## Installation

```bash
pip install fastcashflow
```

Requires Python 3.10 or newer. numpy, numba, polars and matplotlib install
automatically.

## Quick start

No files to prepare — measure the bundled sample portfolio:

```python
import fastcashflow as fcf

# load the bundled sample inputs
basis       = fcf.load_sample_assumptions()   # {(product, channel): Assumptions}
assumptions = basis[("TERM_LIFE_A", "FC")]    # pick one segment
mp          = fcf.load_sample_model_points()

# measure the contract liability -- portfolio totals at issue
m = fcf.measure(mp, assumptions)
print(f"model points : {m.bel.shape[0]:>15,}")
print(f"BEL          : {m.bel[:, 0].sum():>15,.0f}")
print(f"RA           : {m.ra[:, 0].sum():>15,.0f}")
print(f"CSM          : {m.csm[:, 0].sum():>15,.0f}")
```

```text
model points :              11
BEL          :      20,955,426
RA           :       1,854,622
CSM          :       1,488,802
```

Or build a single contract by hand:

```python
import numpy as np
import fastcashflow as fcf

# mortality -- flat 0.1% annual rate (same for every sex/age/duration)
death_fn = lambda sex, issue_age, duration: np.full(issue_age.shape, 0.001)

# lapse -- flat 1% annual rate
lapse_fn = lambda sex, issue_age, duration: np.full(duration.shape, 0.01)

# actuarial assumptions
asmp = fcf.Assumptions(
    mortality_annual = death_fn,   # in-force decrement (death_fn above)
    lapse_annual     = lapse_fn,   # lapse rate (lapse_fn above)
    discount_annual  = 0.03,       # annual discount rate
    ra_confidence    = 0.75,       # risk-adjustment confidence level (75th pct)
    mortality_cv     = 0.10,       # mortality coefficient of variation
    coverages        = (
        fcf.CoverageRate("DEATH", death_fn),  # one death coverage (claim rate = death_fn)
    ),
)

# one policy -- age 40, 100M death benefit, 70k monthly premium, 10-year term
mp = fcf.ModelPoints.single(
    issue_age           = 40,
    benefits            = {0: 100_000_000},
    level_premium       = 70_000,
    term_months         = 120,
    calculation_methods = {"DEATH": fcf.CalculationMethod.DEATH},  # coverage code -> method
)

r = fcf.measure(mp, asmp)
print(f"BEL : {r.bel[0, 0]:>12,.0f}")
print(f"RA  : {r.ra[0, 0]:>12,.0f}")
print(f"CSM : {r.csm[0, 0]:>12,.0f}")
```

```text
BEL :   -6,092,691
RA  :       55,484
CSM :    6,037,206
```

For portfolio-scale valuation, `value()` runs a numba parallel kernel and
returns the same headline numbers, far faster:

```python
# same sample portfolio, fast scalar kernel -- headline numbers only
val = fcf.value(fcf.load_sample_model_points(), assumptions)
print(f"BEL : {val.bel.sum():>15,.0f}")
print(f"RA  : {val.ra.sum():>15,.0f}")
print(f"CSM : {val.csm.sum():>15,.0f}")
```

```text
BEL :      20,955,426
RA  :       1,854,622
CSM :       1,488,802
```

## Features

- **IFRS 17 models** — GMM (BEL / RA / CSM), PAA and VFA (variable-fee /
  account-value contracts with GMDB / GMAB guarantees).
- **Projection** — deterministic monthly cash flows; select-and-ultimate
  mortality, duration-based lapse, mid-month discounting, α / β / γ expenses,
  surrender value, contract states (active / waiver / paid-up).
- **Reporting** — roll-forward, reconciliation tables, insurance service result,
  loss component, aggregation to IFRS 17 unit of account.
- **I/O** — Excel workbook assumptions, polars parquet / CSV model points,
  `value_file` for portfolios larger than RAM.
- **More** — reinsurance, stochastic valuation, premium pricing, TVOG, first-adoption
  transition, GPU backend (`backend="gpu"`).

## Performance

Measured on an 8-core desktop (Ryzen 7 3700X), 120-month projection:

| Model points | `value()` |
|---|---|
| 1,000,000 | 0.07 s |
| 5,000,000 | 0.41 s |

`value()` carries in-force as a scalar and materialises no intermediate arrays.
A 10M-row parquet round-trip — read, value, write — takes about 2.5 seconds,
of which `value()` itself is under one second.
Run `examples/benchmark.py` to reproduce on your machine.

## Documentation

Full tutorial and API reference: <https://docs.fastcashflow.org>

Live demo: <https://demo.fastcashflow.org>

## License

Mozilla Public License 2.0 — see
[LICENSE](https://github.com/seokhoonj/fastcashflow/blob/main/LICENSE).
