API reference
=============

Inputs
------

.. autoclass:: fastcashflow.ModelPoints
   :members:

.. autoclass:: fastcashflow.Assumptions
   :members:

.. autoclass:: fastcashflow.CoverageRate
   :members:

Measurement (GMM)
-----------------

.. autofunction:: fastcashflow.measure

.. autofunction:: fastcashflow.value

.. autoclass:: fastcashflow.Measurement
   :members:

.. autoclass:: fastcashflow.Valuation
   :members:

Premium allocation approach
---------------------------

.. autofunction:: fastcashflow.measure_paa

.. autoclass:: fastcashflow.PAAMeasurement
   :members:

Variable fee approach
---------------------

.. autofunction:: fastcashflow.measure_vfa

.. autoclass:: fastcashflow.VFAMeasurement
   :members:

.. autofunction:: fastcashflow.measure_tvog

.. autoclass:: fastcashflow.TVOGResult
   :members:

Reinsurance
-----------

.. autofunction:: fastcashflow.measure_reinsurance

.. autoclass:: fastcashflow.ReinsuranceMeasurement
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

.. autofunction:: fastcashflow.value_stochastic

.. autoclass:: fastcashflow.StochasticResult
   :members:

Input and output
-----------------

.. autofunction:: fastcashflow.read_model_points

.. autofunction:: fastcashflow.read_assumptions

.. autofunction:: fastcashflow.load_sample_model_points

.. autofunction:: fastcashflow.load_sample_assumptions

.. autofunction:: fastcashflow.write_valuation

.. autofunction:: fastcashflow.value_file

Visualisation
-------------

The plotting helpers require the ``viz`` extra (``pip install
fastcashflow[viz]``).

.. autofunction:: fastcashflow.plot_liability

.. autofunction:: fastcashflow.plot_cashflows

.. autofunction:: fastcashflow.plot_csm_runoff

.. autofunction:: fastcashflow.plot_risk_adjustment

.. autofunction:: fastcashflow.plot_analysis_of_change

.. autofunction:: fastcashflow.plot_stochastic
