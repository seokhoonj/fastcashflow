# Getting started

## Installation

Until the package is published to PyPI, install it from GitHub:

```bash
pip install "fastcashflow[viz] @ git+https://github.com/seokhoonj/fastcashflow.git"
```

The `viz` extra adds the charting helpers used below; drop it for the core
engine alone.

## A first valuation

A valuation needs two inputs -- a set of model points (the policies) and an
actuarial basis (the assumptions). The simplest way to supply them is from
files: an Excel workbook for the basis, a CSV for the portfolio.

```python
import fastcashflow as fcf

asmp = fcf.read_assumptions("sample_basis.xlsx")
mps = fcf.read_model_points("sample_policies.csv")

m = fcf.measure(mps, asmp)
print(m.bel[:, 0])      # best estimate liability at inception
print(m.ra[:, 0])       # risk adjustment
print(m.csm[:, 0])      # contractual service margin
```

`measure` projects every policy month by month and rolls the IFRS 17
liability forward. One more line charts the result:

```python
fcf.plot_liability(m)
```

```{image} images/first-valuation.png
:alt: BEL, RA and CSM trajectories over the contract's life
:class: hero
```

For the memory-minimal fast path that returns only the headline numbers,
use `value` in place of `measure`.

## Next steps

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} The worked example
:link: https://github.com/seokhoonj/fastcashflow/blob/main/examples/worked_example.py
:link-type: url

A full valuation end to end -- pricing, measurement, the disclosure and the
period-close analysis of change.
:::

:::{grid-item-card} Concepts
:link: concepts
:link-type: doc

The IFRS 17 ideas behind the engine -- the measurement models and the
building blocks.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc

Every function and result type, with full signatures.
:::

::::
