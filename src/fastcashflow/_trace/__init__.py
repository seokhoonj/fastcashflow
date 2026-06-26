"""Step-by-step calculation traces -- the private impl behind ``fcf.<model>.trace``.

Not a public namespace. The public call names are the ``.trace`` / ``.trace_diff``
re-exports on the model facades (``fcf.gmm.trace``, ``fcf.vfa.trace``,
``fcf.paa.trace``, ``fcf.reinsurance.trace``, ``fcf.gmm.trace_bel_step`` /
``trace_csm_step``). Each per-model module (:mod:`.gmm`, :mod:`.vfa`, :mod:`.paa`,
:mod:`.reinsurance`) exposes a bare ``trace`` / ``trace_diff``; shared
rendering / diff helpers live in :mod:`.common`. Facades and ``portfolio`` import
the per-model modules directly.
"""

from fastcashflow._trace import gmm, vfa, paa, reinsurance  # noqa: F401 (submodule access)
