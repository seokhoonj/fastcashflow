# Getting started

## Installation

Until the package is published to PyPI, install it from GitHub:

```bash
pip install git+https://github.com/seokhoonj/fastcashflow.git
```

## A first valuation

A valuation needs two inputs: a set of model points (the policies) and an
actuarial basis (the assumptions). The simplest way to supply them is from
files -- an Excel workbook for the basis, a CSV for the portfolio:

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
liability forward. For the memory-minimal fast path that returns only the
headline numbers, use `value` instead.

## The worked example

The repository ships a complete worked example at
[`examples/worked_example.py`](https://github.com/seokhoonj/fastcashflow/blob/main/examples/worked_example.py),
with the sample `sample_basis.xlsx` and `sample_policies.csv` beside it. It
runs a full IFRS 17 valuation end to end: pricing, measurement, the
disclosure, the period-close analysis of change, aggregation into groups, and
a tour of the other measurement models. Copy the sample files, edit them with
your own numbers, and point the two `read_*` calls at them.
