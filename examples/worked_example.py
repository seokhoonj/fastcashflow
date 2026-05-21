"""A worked IFRS 17 valuation with fastcashflow, end to end.

Run it::

    python examples/worked_example.py

It loads an actuarial basis and a portfolio from the sample files in this
directory, measures the IFRS 17 liability, assembles the disclosure, and
rolls one reporting period forward into the analysis of change.

To value your own book: copy ``sample_basis.xlsx`` and
``sample_policies.csv``, edit them with your own numbers, and point the two
``read_*`` calls below at your files.
"""
from pathlib import Path

import fastcashflow as fcf

HERE = Path(__file__).parent


# --- 1. The inputs ---------------------------------------------------------
# Two inputs, both from files. The actuarial basis is an Excel workbook --
# the form a practitioner keeps assumptions in; read_assumptions turns its
# mortality and lapse tables into the monthly-rate functions the engine
# uses. The portfolio is a CSV of model points.
asmp = fcf.read_assumptions(HERE / "sample_basis.xlsx")
mps = fcf.read_model_points(HERE / "sample_policies.csv")
print(f"Loaded {mps.n_mp} model points and the actuarial basis.\n")


# --- 2. Measure the liability ---------------------------------------------
# measure() projects every contract month by month and rolls the IFRS 17
# liability forward. The headline at inception: the best-estimate liability,
# the risk adjustment, the contractual service margin (the unearned profit),
# and any loss component on onerous contracts.
m = fcf.measure(mps, asmp)
print("Inception measurement (portfolio totals):")
print(f"  BEL             {m.bel[:, 0].sum():>16,.0f}")
print(f"  RA              {m.ra[:, 0].sum():>16,.0f}")
print(f"  CSM             {m.csm[:, 0].sum():>16,.0f}")
print(f"  loss component  {m.loss_component.sum():>16,.0f}\n")

# value() is the fast path -- the same headline numbers, no trajectories,
# the route for portfolio-scale runs.
v = fcf.value(mps, asmp)
print(f"value() agrees with measure() -- CSM {v.csm.sum():,.0f}\n")


# --- 3. The disclosure -----------------------------------------------------
# report() turns the measurement into the IFRS 17 insurance service result.
rep = fcf.report(m)
print(f"Insurance revenue over the contract  {rep.insurance_revenue.sum():>16,.0f}")
print(f"Insurance service result             "
      f"{rep.insurance_service_result.sum():>16,.0f}\n")


# --- 4. The period-close roll-forward -------------------------------------
# roll_forward slices the measurement into reporting periods; reconcile
# aggregates each into the IFRS 17 analysis of change. Here, twelve-month
# periods on the expected basis -- opening balance, interest, release,
# closing balance, each column reconciling exactly.
reconciliations = fcf.reconcile(fcf.roll_forward(m, period_months=12))
print(f"The contract runs off over {len(reconciliations)} reporting periods.\n")
print(reconciliations[0])
print()
print(reconciliations[-1])


# --- where to go next ------------------------------------------------------
# The same measure -> report -> roll_forward -> reconcile pattern carries
# across the engine: measure_paa for short-coverage business, measure_vfa
# for account-value contracts; group() aggregates to IFRS 17 groups;
# solve_premium prices a contract; value_stochastic runs economic scenarios.
