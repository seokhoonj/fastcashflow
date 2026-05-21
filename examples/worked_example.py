"""A worked IFRS 17 valuation with fastcashflow, end to end.

Run it::

    python examples/worked_example.py

It loads an actuarial basis and a portfolio from the sample files here,
prices a contract, measures the IFRS 17 liability, assembles the
disclosure, rolls a reporting period forward into the analysis of change,
aggregates to IFRS 17 groups, and closes with a tour of the other
measurement models.

To value your own book: copy ``sample_basis.xlsx`` and
``sample_policies.csv``, edit them with your own numbers, and point the two
``read_*`` calls below at your files.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

import fastcashflow as fcf

HERE = Path(__file__).parent


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# --- the inputs ------------------------------------------------------------
# Two inputs, both from files. The actuarial basis is an Excel workbook --
# the form a practitioner keeps assumptions in; read_assumptions turns its
# mortality and lapse tables into the monthly-rate functions the engine
# uses. The portfolio is a CSV of model points.
section("1. The inputs")
asmp = fcf.read_assumptions(HERE / "sample_basis.xlsx")
mps = fcf.read_model_points(HERE / "sample_policies.csv")
print(f"Loaded {mps.n_mp} model points and the actuarial basis.")


# --- pricing ---------------------------------------------------------------
# solve_premium finds the level monthly premium meeting a profitability
# target -- here a 10% margin. (The sample policies were priced this way.)
section("2. Pricing a contract")
new_contract = fcf.ModelPointSet.single(45, 50_000_000, 0, 120)
premium = fcf.solve_premium(new_contract, asmp, margin=0.10)
print("A 45-year-old, 50m cover, 10-year term, priced for a 10% margin:")
print(f"  level premium  {premium[0]:,.0f} per month")


# --- measuring the liability ----------------------------------------------
# measure() projects every contract month by month and rolls the IFRS 17
# liability forward: the best-estimate liability, the risk adjustment, the
# contractual service margin (unearned profit), and any loss component.
section("3. Measuring the liability (GMM)")
m = fcf.measure(mps, asmp)
print("Inception measurement (portfolio totals):")
print(f"  BEL             {m.bel[:, 0].sum():>16,.0f}")
print(f"  RA              {m.ra[:, 0].sum():>16,.0f}")
print(f"  CSM             {m.csm[:, 0].sum():>16,.0f}")
print(f"  loss component  {m.loss_component.sum():>16,.0f}")
# value() is the fast path -- the same headline numbers, no trajectories.
v = fcf.value(mps, asmp)
print(f"value() agrees -- CSM {v.csm.sum():,.0f}")


# --- the disclosure --------------------------------------------------------
# report() turns the measurement into the IFRS 17 insurance service result.
section("4. The disclosure")
rep = fcf.report(m)
print(f"Insurance revenue over the contract  {rep.insurance_revenue.sum():>16,.0f}")
print(f"Insurance service result             "
      f"{rep.insurance_service_result.sum():>16,.0f}")


# --- the period-close roll-forward ----------------------------------------
# roll_forward slices the measurement into reporting periods; reconcile
# aggregates each into the IFRS 17 analysis of change -- opening balance,
# interest, release, closing balance, every column reconciling exactly.
section("5. The period-close analysis of change")
reconciliations = fcf.reconcile(fcf.roll_forward(m, period_months=12))
print(f"The contract runs off over {len(reconciliations)} reporting periods.\n")
print(reconciliations[0])


# --- aggregation -----------------------------------------------------------
# group() re-expresses the measurement at the IFRS 17 unit of account. Here
# the book is split into two groups by issue age; the CSM is re-derived at
# the group level, so the floor nets contracts within a group.
section("6. Aggregation into IFRS 17 groups")
group_ids = (mps.issue_age >= 45).astype(int)        # 0 = under 45, 1 = 45+
grouped = fcf.group(m, group_ids)
for g, label in enumerate(("under 45", "45 and over")):
    print(f"  group '{label}':  CSM {grouped.csm[g, 0]:>14,.0f}")


# --- the other measurement models -----------------------------------------
# The same engine measures the other two IFRS 17 models.
section("7. The other measurement models")

# PAA -- the simplified model for short-coverage business. A one-year
# contract carries a far smaller acquisition cost than a ten-year sale, so
# the basis is adjusted accordingly.
short_book = fcf.ModelPointSet(
    issue_age=np.array([40, 45]),
    death_benefit=np.array([3e7, 3e7]),
    monthly_premium=np.array([18_000.0, 20_000.0]),
    term_months=np.array([12, 12]),
)
paa = fcf.measure_paa(short_book, replace(asmp, expense_acquisition=20_000.0))
print(f"  PAA  -- insurance service result  {paa.service_result.sum():>14,.0f}")

# VFA -- account-value (unit-linked / with-profits) contracts.
account = fcf.ModelPointSet.single(40, 0.0, 0.0, 60, account_value=1e8)
vfa = fcf.measure_vfa(account, replace(asmp, investment_return=0.06, fund_fee=0.015))
print(f"  VFA  -- CSM (the entity's variable fee)  {vfa.csm[:, 0].sum():>9,.0f}")

# Stochastic -- the liability distribution over economic scenarios.
dist = fcf.value_stochastic(mps, asmp, np.array([0.02, 0.03, 0.04, 0.05]))
print(f"  Stochastic -- BEL from {dist.bel.min():,.0f} to {dist.bel.max():,.0f}"
      f" across the discount-rate scenarios")
