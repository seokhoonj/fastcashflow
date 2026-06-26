"""Measurement identity -- which engine measurement produced a result.

IFRS 17 has THREE measurement models: the General Measurement Model (GMM, the
default building blocks, paragraphs 32-52), the Premium Allocation Approach (PAA, the
short-duration simplification, paragraphs 53-59) and the Variable Fee Approach (VFA,
the variation for direct participating contracts). Reinsurance contracts held
are NOT a fourth model -- they are measured under GMM with the modifications of
paragraphs 60-70 (a net-cost-or-gain CSM that may be negative, no loss component, a
loss-recovery component). But in this engine reinsurance has its own ``measure``
entry point and its own result type, so it carries a fourth family tag
alongside the three models.

This module is the single source of that vocabulary: the four canonical
lowercase tokens, the structural tag every per-model result carries
(:class:`ModelTagged`), and the two accessors that read it (:func:`model_tag`,
:func:`supported_model_tags`). Centralising it here lets the result types, the
portfolio partition keys and the disclosure frames name the measurements from
one place, instead of repeating string literals (with the identity riding
implicitly on each result class's name). The sibling
:mod:`fastcashflow._measurement.basis` holds the orthogonal axis -- the time
basis (inception / settlement-carry / hypothetical) of a result. This module
has no internal dependencies, so any module may import it without a cycle.
"""
from __future__ import annotations

from typing import ClassVar, Protocol

# Canonical family tokens -- the single source of the measurement vocabulary.
# GMM / PAA / VFA are the three IFRS 17 measurement models (in the standard's
# order: general model, then the PAA simplification, then the VFA variation);
# ``reinsurance`` is the fourth family -- reinsurance held, measured under GMM
# with the paragraphs 60-70 modifications, not a fourth model. Lowercase so they read
# as the public namespace path (``fcf.gmm`` / ``fcf.paa`` / ...) and the result
# repr (``gmm.Measurement``).
GMM = "gmm"
PAA = "paa"
VFA = "vfa"
REINSURANCE = "reinsurance"

#: Every family token, for callers that need to enumerate the measurements.
MODEL_TAGS = (GMM, PAA, VFA, REINSURANCE)


class ModelTagged(Protocol):
    """A per-model result that publishes which model produced it.

    ``model`` is a class-level tag, one of :data:`MODEL_TAGS`. The per-model
    measurement / reconciliation / report / aggregate types satisfy this
    structurally (each declares ``model: ClassVar[str]``). Cross-model
    CONTAINERS -- the portfolio-level results and the generic
    group-of-contracts settlement -- are deliberately NOT model-tagged: they
    span models and carry no single identity, so they are skipped by
    :func:`supported_model_tags` and fall back to their type name in
    :func:`model_tag`.
    """

    model: ClassVar[str]


def model_tag(obj) -> str:
    """The model tag of a result, accepting either an instance or a class.

    Reads ``obj.model`` -- declared as a ``ClassVar`` so it resolves on both an
    instance and the class object (some diagnostics name an EXPECTED class, not
    an instance). Anything untagged -- a cross-model container, or a non-result
    that reached a :func:`functools.singledispatch` default -- falls back to the
    type name, so a diagnostic building a ``TypeError`` never raises
    ``AttributeError`` first.
    """
    tag = getattr(obj, "model", None)
    if tag:
        return tag
    return obj.__name__ if isinstance(obj, type) else type(obj).__name__


def supported_model_tags(dispatcher) -> list[str]:
    """The sorted, de-duplicated model tags a singledispatch function handles.

    Derived from ``dispatcher.registry`` so a "supported models" message never
    goes stale as models are added or removed. Registry keys without a model
    tag (the ``object`` fallback and any cross-model container registered on the
    same generic function) are skipped; several registered types that share a
    model -- e.g. a measurement and its settlement movement -- collapse to one
    tag. Sorted because the registry's order is import-driven and differs
    between functions.
    """
    return sorted({tag for key in dispatcher.registry
                   if (tag := getattr(key, "model", None))})
