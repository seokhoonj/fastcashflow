API reference
=============

Inputs
------

.. autoclass:: fastcashflow.ModelPoints
   :members:

.. autoclass:: fastcashflow.Basis
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

.. autofunction:: fastcashflow.gmm.measure_inforce

.. autoclass:: fastcashflow.GMMMeasurement
   :members:

Premium allocation approach
---------------------------

.. autofunction:: fastcashflow.paa.measure

.. autoclass:: fastcashflow.PAAMeasurement
   :members:

Variable fee approach
---------------------

.. autofunction:: fastcashflow.vfa.measure

.. autoclass:: fastcashflow.VFAMeasurement
   :members:

.. autofunction:: fastcashflow.vfa.tvog

.. autoclass:: fastcashflow.TVOGResult
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

.. autofunction:: fastcashflow.plot_liability

.. autofunction:: fastcashflow.plot_cashflows

.. autofunction:: fastcashflow.plot_csm_runoff

.. autofunction:: fastcashflow.plot_risk_adjustment

.. autofunction:: fastcashflow.plot_analysis_of_change

.. autofunction:: fastcashflow.plot_stochastic
