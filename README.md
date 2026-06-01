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

basis      = fcf.load_sample_assumptions()       # {(product, channel): Assumptions}
assumptions = basis[("TERM_LIFE_A", "FC")]
m          = fcf.measure(fcf.load_sample_model_points(), assumptions)
print(m.bel[:, 0], m.ra[:, 0], m.csm[:, 0])    # BEL / RA / CSM at issue
```

For a single hand-built contract:

```python
import numpy as np
import fastcashflow as fcf

mort = lambda sex, age, dur: np.full(age.shape, 0.001)
asmp = fcf.Assumptions(
    mortality_annual = mort,
    lapse_annual     = lambda sex, age, dur: np.full(dur.shape, 0.01),
    discount_annual  = 0.03,
    ra_confidence    = 0.75,
    mortality_cv     = 0.10,
    coverages        = (fcf.CoverageRate("DEATH", mort),),
)
mp = fcf.ModelPoints.single(
    issue_age=40, benefits={0: 100_000_000},
    level_premium=70_000, term_months=120,
    calculation_methods={"DEATH": fcf.CalculationMethod.DEATH},
)
r = fcf.measure(mp, asmp)
print(r.bel[0, 0], r.ra[0, 0], r.csm[0, 0])
```

For portfolio-scale valuation, `value()` runs a numba parallel kernel and
returns only the headline numbers per model point:

```python
val = fcf.value(model_points, assumptions)
print(val.bel, val.ra, val.csm)
```

## Features

- **Three IFRS 17 models** — GMM (BEL / RA / CSM), PAA and VFA (variable-fee /
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

Measured on an 8-core desktop (Ryzen 7 3700X), 120-month term projection:

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
