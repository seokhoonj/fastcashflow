"""Reporting -- the IFRS 17 report, the analysis of change and aggregation.

Inputs are in examples/data/ (Excel files).

    python examples/reporting.py
"""
from pathlib import Path

import fastcashflow as fcf

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    basis = fcf.read_assumptions(DATA / "assumptions.xlsx")
    assumptions = basis[("TERM_LIFE_A", "FC")]
    book = fcf.read_model_points(DATA / "model_points_wide.xlsx", calculation_methods=DATA / "calculation_methods.csv")
    m = fcf.measure(book, assumptions)

    # The IFRS 17 report -- insurance revenue, service expense, service result.
    print(fcf.report(m))
    print()

    # The period-close analysis of change, first reporting year.
    recon = fcf.reconcile(fcf.roll_forward(m, period_months=12))
    print(recon[0])
    print()

    # Aggregation to the IFRS 17 unit of account -- here, two age groups.
    group_ids = (book.issue_age >= 45).astype(int)
    grouped = fcf.group(m, group_ids)
    for g, label in enumerate(("under 45", "45 and over")):
        print(f"group '{label}'  CSM {grouped.csm[g, 0]:>14,.0f}")


if __name__ == "__main__":
    main()
