API reference
=============

Inputs
------

.. autoclass:: fastcashflow.ModelPoints
   :members:

.. autoclass:: fastcashflow.Basis
   :members:

.. autoclass:: fastcashflow.BasisRouter
   :members:

.. autoclass:: fastcashflow.CoverageRate
   :members:

.. autoclass:: fastcashflow.CalculationMethod

.. autoclass:: fastcashflow.ExpenseItem

.. autofunction:: fastcashflow.derive_expense_components

.. autodata:: fastcashflow.EXPENSE_BASES

.. autodata:: fastcashflow.RA_METHODS

.. autodata:: fastcashflow.SURRENDER_VALUE_BASES

.. autodata:: fastcashflow.RISK_MORTALITY

.. autodata:: fastcashflow.RISK_MORBIDITY

Measurement (GMM)
-----------------

.. autofunction:: fastcashflow.gmm.measure

.. autofunction:: fastcashflow.gmm.measure_aggregate

.. autofunction:: fastcashflow.gmm.measure_inforce

.. autofunction:: fastcashflow.gmm.settle

.. autofunction:: fastcashflow.gmm.settle_aggregate

.. autofunction:: fastcashflow.gmm.settle_stream

.. autofunction:: fastcashflow.gmm.recognition_schedule

.. autoclass:: fastcashflow.GMMSettlementMovement
   :members:

.. autoclass:: fastcashflow.GMMSettlementReconciliation
   :members:

.. autoclass:: fastcashflow.GMMSettlementAggregate
   :members:

.. autoclass:: fastcashflow.CSMRecognitionSchedule
   :members:

.. autoclass:: fastcashflow.GMMMeasurement
   :members:

.. autoclass:: fastcashflow.GMMAggregate
   :members:

Premium allocation approach
---------------------------

.. autofunction:: fastcashflow.paa.measure

.. autofunction:: fastcashflow.paa.measure_aggregate

.. autofunction:: fastcashflow.paa.measure_inforce

.. autofunction:: fastcashflow.paa.settle

.. autoclass:: fastcashflow.PAASettlementMovement
   :members:

.. autoclass:: fastcashflow.PAASettlementReconciliation
   :members:

.. autoclass:: fastcashflow.PAAMeasurement
   :members:

.. autoclass:: fastcashflow.PAAAggregate
   :members:

Variable fee approach
---------------------

.. autofunction:: fastcashflow.vfa.measure

.. autofunction:: fastcashflow.vfa.measure_aggregate

.. autofunction:: fastcashflow.vfa.measure_inforce

.. autofunction:: fastcashflow.vfa.settle

.. autofunction:: fastcashflow.vfa.settle_aggregate

.. autofunction:: fastcashflow.vfa.settle_stream

.. autoclass:: fastcashflow.VFASettlementMovement
   :members:

.. autoclass:: fastcashflow.VFASettlementReconciliation
   :members:

.. autoclass:: fastcashflow.VFASettlementAggregate
   :members:

.. autoclass:: fastcashflow.VFAMeasurement
   :members:

.. autodata:: fastcashflow.vfa.CSM_BASES

.. autoclass:: fastcashflow.VFAAggregate
   :members:

.. autofunction:: fastcashflow.vfa.tvog

.. autoclass:: fastcashflow.TVOGResult
   :members:

Portfolio (mixed-model orchestration)
-------------------------------------

One heterogeneous portfolio -- GMM, PAA and VFA contracts in a single routed
file -- measured in one call. Each contract is routed to its segment's
measurement model and each model's native result is kept separate (a BEL and an
LRC are never summed into one array). ``measure`` returns the per-model-point
:class:`~fastcashflow.portfolio.PortfolioMeasurement`; ``measure_aggregate``
returns the chunked, bounded-memory
:class:`~fastcashflow.portfolio.PortfolioAggregate` (a scalable sum of the
measured model-point results -- not an IFRS group remeasurement and not a group of contracts
re-floor). ``loss_component`` is the lone quantity summed across models.

.. autofunction:: fastcashflow.portfolio.measure

.. autofunction:: fastcashflow.portfolio.measure_aggregate

.. autoclass:: fastcashflow.portfolio.PortfolioMeasurement
   :members:

.. autoclass:: fastcashflow.portfolio.ModelMeasurement
   :members:

.. autoclass:: fastcashflow.portfolio.PortfolioAggregate
   :members:

Per-group aggregate (scalable group of contracts)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``measure_group_of_contracts`` is the chunked, bounded-memory form of
:func:`fastcashflow.group_of_contracts`: the IFRS 17 unit of account
(portfolio x annual cohort x profitability) computed where holding the
per-model-point ``measure(full=True)`` would not fit in memory.
``measure_group`` is the same machinery on any axis (the scalable
:func:`fastcashflow.group`). Both return a
:class:`~fastcashflow.portfolio.PortfolioGroups` holding each model's native
grouped measurement -- its rows the groups -- so the group rows flow on into
:func:`fastcashflow.roll_forward`, :func:`fastcashflow.reconcile` and
:func:`fastcashflow.report`.

The floor unit is what distinguishes this from ``measure_aggregate``:

* ``measure_aggregate`` floors **per model point**, then sums --
  ``sum max(0, -FCF_i)``. It is a scalable sum of the already-floored
  per-contract results, never re-grouping them.
* ``measure_group_of_contracts`` / ``measure_group`` re-floor **per group**, on the summed
  fulfilment cash flows -- ``max(0, -sum FCF_in_group)`` per group, applied once
  on the fully-accumulated group (a group spans chunks, so it is never floored
  per chunk).

At initial recognition these agree: under any paragraph-16-compliant grouping a
group never mixes inception-FCF signs, so ``CSM(sum FCF) == sum CSM(FCF)`` and
``measure_group_of_contracts`` and ``measure_aggregate`` report the same totals. The re-floor
changes the number only for a deliberately coarser, sign-mixing grouping (e.g.
``measure_group(by="product")`` with no profitability axis -- within-group
mutualisation) or in subsequent measurement (out of scope here). So at inception
``measure_group_of_contracts``'s value over ``measure_aggregate`` is the **per-group rows**
(disclosure, roll-forward, the paragraph-44 foundation), not a different number.

.. autofunction:: fastcashflow.portfolio.measure_group_of_contracts

.. autofunction:: fastcashflow.portfolio.settle_group_of_contracts

.. autofunction:: fastcashflow.portfolio.measure_group

.. autoclass:: fastcashflow.portfolio.PortfolioGroups
   :members:

.. autoclass:: fastcashflow.portfolio.GoCSettlement
   :members:

.. autoclass:: fastcashflow.portfolio.VFAGoCSettlement
   :members:

Tracing and validation
----------------------

Per-contract tracers that unfold a single model point's measurement as an
ASCII tree -- which segment, table and rate feed each step, the year-by-year
rates and cash flows, and the anchor-month discount / BEL / CSM. Used for
hand-calculation validation, learning and debugging. Each measurement
approach has its own tracer.

.. autofunction:: fastcashflow.gmm.trace

.. autofunction:: fastcashflow.gmm.trace_diff

.. autofunction:: fastcashflow.gmm.trace_bel_step

.. autofunction:: fastcashflow.gmm.trace_csm_step

.. autofunction:: fastcashflow.vfa.trace

.. autofunction:: fastcashflow.paa.trace

Reinsurance
-----------

.. autofunction:: fastcashflow.reinsurance.measure

.. autoclass:: fastcashflow.reinsurance.QuotaShare
   :members:

.. autoclass:: fastcashflow.reinsurance.ReinsuranceMeasurement
   :members:

Pricing
-------

.. autofunction:: fastcashflow.solve_premium

Reporting
---------

.. autofunction:: fastcashflow.report

.. autoclass:: fastcashflow.Report
   :members:

``report`` also accepts a mixed-portfolio container
(:class:`~fastcashflow.portfolio.PortfolioMeasurement` or
:class:`~fastcashflow.portfolio.PortfolioGroups`) and returns a
:class:`~fastcashflow.portfolio.PortfolioReport` -- one :class:`Report` per model,
never merged.

.. autoclass:: fastcashflow.portfolio.PortfolioReport
   :members:

Period-close analysis of change
-------------------------------

.. autofunction:: fastcashflow.roll_forward

.. autofunction:: fastcashflow.reconcile

.. autoclass:: fastcashflow.PeriodMovement
   :members:

.. autoclass:: fastcashflow.Reconciliation
   :members:

.. autoclass:: fastcashflow.PAAPeriodMovement
   :members:

.. autoclass:: fastcashflow.PAAReconciliation
   :members:

.. autoclass:: fastcashflow.VFAPeriodMovement
   :members:

.. autoclass:: fastcashflow.VFAReconciliation
   :members:

``roll_forward`` and ``reconcile`` also accept the mixed-portfolio containers:
``roll_forward`` on a :class:`~fastcashflow.portfolio.PortfolioMeasurement` /
:class:`~fastcashflow.portfolio.PortfolioGroups` returns a
:class:`~fastcashflow.portfolio.PortfolioMovements` (one movement list per model),
which :func:`reconcile` turns into a
:class:`~fastcashflow.portfolio.PortfolioReconciliation` -- a GMM CSM movement and
a PAA LRC movement are never merged.

.. autoclass:: fastcashflow.portfolio.PortfolioMovements
   :members:

.. autoclass:: fastcashflow.portfolio.PortfolioReconciliation
   :members:

Aggregation and transition
--------------------------

.. autofunction:: fastcashflow.group

.. autofunction:: fastcashflow.group_of_contracts

.. autofunction:: fastcashflow.transition

State models
------------

The in-force state machine driving multi-state products (waiver, paid-up,
disability income, reincidence, long-term care). A :class:`StateModel` is a
tuple of :class:`State` objects, each carrying its :class:`Transition` edges;
``STATE_MODELS`` holds the bundled models.

.. autoclass:: fastcashflow.StateModel

.. autoclass:: fastcashflow.State

.. autoclass:: fastcashflow.Transition

.. autodata:: fastcashflow.STATE_MODELS

.. autodata:: fastcashflow.STATE_ACTIVE

.. autodata:: fastcashflow.STATE_WAIVER

.. autodata:: fastcashflow.STATE_PAIDUP

Stochastic valuation
--------------------

.. autofunction:: fastcashflow.gmm.stochastic

.. autoclass:: fastcashflow.StochasticResult
   :members:

Input and output
-----------------

.. autofunction:: fastcashflow.read_model_points

.. autofunction:: fastcashflow.read_vfa_model_points

.. autofunction:: fastcashflow.read_basis

.. autofunction:: fastcashflow.read_inforce_policies

.. autofunction:: fastcashflow.read_inforce_state

.. autofunction:: fastcashflow.apply_inforce_state

.. autofunction:: fastcashflow.align_inforce_state

.. autoclass:: fastcashflow.InforceState

.. autofunction:: fastcashflow.read_scenarios

.. autofunction:: fastcashflow.describe_basis

.. autofunction:: fastcashflow.samples.model_points

.. autofunction:: fastcashflow.samples.basis

.. autofunction:: fastcashflow.samples.inforce_state

.. autofunction:: fastcashflow.samples.calculation_methods

.. autofunction:: fastcashflow.samples.export

.. autofunction:: fastcashflow.samples.templates

.. autofunction:: fastcashflow.write_measurement

.. autofunction:: fastcashflow.gmm.measure_stream

.. autofunction:: fastcashflow.sample_data_dir

.. autofunction:: fastcashflow.clear_codegen_cache

Visualisation
-------------

The plotting helpers use matplotlib, which is included in the standard install.
The measurement and reconciliation charts dispatch on the result type -- GMM,
PAA, VFA and reinsurance held each draw their own model's quantities (a PAA
result draws its LRC and LIC; a reinsurance-held result draws the ceded
streams and its possibly-negative CSM). A portfolio container is refused --
plot one model slot's native result instead.

.. autofunction:: fastcashflow.plot_liability

.. autofunction:: fastcashflow.plot_cashflows

.. autofunction:: fastcashflow.plot_csm_runoff

.. autofunction:: fastcashflow.plot_risk_adjustment

.. autofunction:: fastcashflow.plot_analysis_of_change

.. autofunction:: fastcashflow.plot_stochastic
