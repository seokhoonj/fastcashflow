# Getting started

## Installation

```bash
pip install fastcashflow[viz]
```

The `viz` extra adds the charting helpers used below; drop it for the core
engine alone.

## A first valuation

A valuation needs two inputs -- a set of model points (the policies) and an
actuarial basis (the assumptions). The quickest start is fastcashflow's
bundled sample, which loads with no files to prepare.

```python
import fastcashflow as fcf

assumptions = fcf.load_sample_assumptions()
model_points = fcf.load_sample_model_points()

m = fcf.measure(model_points, assumptions)
print(m.bel[:, 0])      # best estimate liability at inception
print(m.ra[:, 0])       # risk adjustment
print(m.csm[:, 0])      # contractual service margin
```

`measure` projects every policy month by month and rolls the IFRS 17
liability forward. One more line charts the result:

```python
fcf.plot_liability(m)
```

```{image} ../images/first-valuation.png
:alt: BEL, RA and CSM trajectories over the contract's life
:class: hero
```

For the memory-minimal fast path that returns only the headline numbers,
use `value` in place of `measure`.

## Next steps

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Examples
:link: https://github.com/seokhoonj/fastcashflow/tree/main/examples
:link-type: url

Runnable scripts -- quickstart, reporting, pricing and more. Each reads
its inputs from the Excel files in examples/data/.
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
