"""Reporting -- the IFRS 17 report, the analysis of change and aggregation.

The inputs are the bundled sample portfolio (``fcf.samples``).

    python examples/reporting.py
"""
import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis()
    book = fcf.samples.model_points()
    m = fcf.gmm.measure(book, basis)

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
        print(f"group '{label}'  CSM {grouped.csm_path[g, 0]:>14,.0f}")


if __name__ == "__main__":
    main()
