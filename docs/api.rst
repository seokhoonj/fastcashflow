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

Measurement (GMM)
-----------------

.. autofunction:: fastcashflow.gmm.measure

.. autofunction:: fastcashflow.measure_in_force

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

.. autofunction:: fastcashflow.transition

Stochastic valuation
--------------------

.. autofunction:: fastcashflow.gmm.stochastic

.. autoclass:: fastcashflow.StochasticResult
   :members:

Input and output
-----------------

.. autofunction:: fastcashflow.read_model_points

.. autofunction:: fastcashflow.read_basis

.. autofunction:: fastcashflow.samples.model_points

.. autofunction:: fastcashflow.samples.basis

.. autofunction:: fastcashflow.write_measurement

.. autofunction:: fastcashflow.gmm.measure_stream

Visualisation
-------------

The plotting helpers use matplotlib, which is included in the standard install.

.. autofunction:: fastcashflow.plot_liability

.. autofunction:: fastcashflow.plot_cashflows

.. autofunction:: fastcashflow.plot_csm_runoff

.. autofunction:: fastcashflow.plot_risk_adjustment

.. autofunction:: fastcashflow.plot_analysis_of_change

.. autofunction:: fastcashflow.plot_stochastic
