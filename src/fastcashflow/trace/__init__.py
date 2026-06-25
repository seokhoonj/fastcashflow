"""Step-by-step calculation traces -- the ``fcf.<model>.trace`` tools.

Each ``show_trace_*`` walks one model point's measurement and prints it as an
ASCII tree for hand-checking against an external pricing system. The public
call names are the namespaced ``.trace`` re-exports on the model facades
(``fcf.gmm.trace``, ``fcf.vfa.trace``, ``fcf.paa.trace``,
``fcf.reinsurance.trace``, ``fcf.gmm.trace_diff``); the ``show_trace_*``
functions here are the implementations those facades alias.

The per-model trace logic lives in the sibling modules (:mod:`.gmm`,
:mod:`.vfa`, :mod:`.paa`, :mod:`.reinsurance`); the shared rendering and
diff-formatting primitives live in :mod:`._common`.
"""
from fastcashflow.trace._common import _resolve_basis
from fastcashflow.trace.gmm import (
    show_trace, show_trace_diff, show_trace_bel_step, show_trace_csm_step)
from fastcashflow.trace.vfa import show_trace_vfa, show_trace_diff_vfa
from fastcashflow.trace.paa import show_trace_paa, show_trace_diff_paa
from fastcashflow.trace.reinsurance import (
    show_trace_reinsurance, show_trace_diff_reinsurance)

__all__ = [
    "show_trace", "show_trace_diff", "show_trace_bel_step",
    "show_trace_csm_step",
    "show_trace_vfa", "show_trace_diff_vfa",
    "show_trace_paa", "show_trace_diff_paa",
    "show_trace_reinsurance", "show_trace_diff_reinsurance",
    "_resolve_basis",
]
