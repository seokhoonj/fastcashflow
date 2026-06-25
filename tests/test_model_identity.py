"""Model-identity layer (S5) -- the canonical tokens, the ``model_tag`` /
``supported_model_tags`` accessors, and the diagnostic messages derived from
them.

The golden parity net is numeric-only (it cannot see an error string), so the
identity surface is pinned here: the four-family vocabulary, the class-or-
instance accessor with its mandatory fallback, and the registry-derived
"supported" lists that several singledispatch defaults render. These guard the
S5.3 rename -- once the four result classes share the name ``Measurement``, a
site still reading ``type(x).__name__`` would silently collapse to
"Measurement"; routing through ``model_tag`` keeps the model identity.
"""
import pytest

import fastcashflow as fcf
from fastcashflow.reinsurance import QuotaShare
from fastcashflow._measurement.model import (
    MODEL_TAGS, model_tag, supported_model_tags,
)
from fastcashflow._measurement.gmm import Measurement as _GmmMeasurement
from fastcashflow._measurement.vfa import Measurement as _VfaMeasurement
from fastcashflow._measurement.paa import Measurement as _PaaMeasurement
from fastcashflow._measurement.reinsurance import Measurement as _ReinsuranceMeasurement
from fastcashflow.grouping import group, group_of_contracts
from fastcashflow.io import write_measurement
from fastcashflow.report import report
from fastcashflow._measurement.movement import roll_forward
from fastcashflow.disclosure import reconciliation_to_frame


def test_model_tags_canonical_order():
    # standard order: the three measurement models then reinsurance held
    assert MODEL_TAGS == ("gmm", "paa", "vfa", "reinsurance")


@pytest.mark.parametrize("cls,tag", [
    (_GmmMeasurement, "gmm"),
    (_PaaMeasurement, "paa"),
    (_VfaMeasurement, "vfa"),
    (_ReinsuranceMeasurement, "reinsurance"),
])
def test_model_tag_reads_classvar_off_the_class(cls, tag):
    # model_tag must accept a CLASS object (some diagnostics name an expected
    # type, not an instance) and read the ClassVar off it
    assert model_tag(cls) == tag


def test_model_tag_falls_back_to_type_name_when_untagged():
    # the fallback is mandatory -- write_measurement's singledispatch default can
    # receive a bare str, and model_tag must not raise before the TypeError
    assert model_tag("not a measurement") == "str"
    assert model_tag(object()) == "object"


@pytest.mark.parametrize("dispatcher", [
    group, group_of_contracts, write_measurement, report,
    roll_forward, reconciliation_to_frame,
])
def test_supported_model_tags_is_the_four_families(dispatcher):
    # every model dispatcher advertises exactly the four families, sorted and
    # deduplicated, despite registries polluted with object / containers /
    # settlement movements
    assert supported_model_tags(dispatcher) == ["gmm", "paa", "reinsurance", "vfa"]


# --- the rendered diagnostic messages (derived list + tag subject) -----------

def test_group_default_message_derives_supported_list():
    with pytest.raises(
        TypeError,
        match="group is not implemented for str; "
              "supported: gmm, paa, reinsurance, vfa.",
    ):
        group("not a measurement", "product")


def test_group_of_contracts_default_message_derives_supported_list():
    with pytest.raises(
        TypeError,
        match="group_of_contracts is not implemented for str; "
              "supported: gmm, paa, reinsurance, vfa.",
    ):
        group_of_contracts("not a measurement")


def test_write_measurement_default_message_derives_supported_list():
    with pytest.raises(
        TypeError,
        match="write_measurement does not handle str; "
              "pass a gmm / paa / reinsurance / vfa measurement "
              "or a portfolio measurement",
    ):
        write_measurement("not a measurement", "/tmp/x.parquet")


def test_report_default_message_derives_supported_list():
    with pytest.raises(
        TypeError,
        match=r"report\(\) expects one of gmm, paa, reinsurance, vfa, got str",
    ):
        report("not a measurement")


def test_roll_forward_default_message_uses_tag():
    with pytest.raises(TypeError, match="roll_forward does not handle str"):
        roll_forward("not a measurement")


def test_reconciliation_to_frame_default_message_uses_tag():
    with pytest.raises(
        TypeError,
        match="reconciliation_to_frame: no disclosure spec for str",
    ):
        reconciliation_to_frame("not a reconciliation")


# --- the per-model `Measurement` rename (S5.3) + alias retirement (S5.4) ------

def test_measurement_classes_share_the_name_but_stay_distinct():
    # each model owns a class literally named `Measurement` (so it reads as
    # `fcf.gmm.Measurement`); they are four distinct type objects, not one shared
    cls = (_GmmMeasurement, _VfaMeasurement, _PaaMeasurement, _ReinsuranceMeasurement)
    assert all(c.__name__ == "Measurement" for c in cls)
    assert len(set(cls)) == 4


def test_prefixed_names_are_retired():
    # the old prefixed names were removed -- the canonical name is `Measurement`
    # on each namespace, with no back-compat alias left
    for ns, old in ((fcf.gmm, "GMMMeasurement"), (fcf.vfa, "VFAMeasurement"),
                    (fcf.paa, "PAAMeasurement"),
                    (fcf.reinsurance, "ReinsuranceMeasurement")):
        assert not hasattr(ns, old)
        assert old not in ns.__all__


def test_namespace_facade_exposes_canonical_measurement():
    # the canonical short name is the model's own Measurement class
    assert fcf.gmm.Measurement is _GmmMeasurement
    assert fcf.vfa.Measurement is _VfaMeasurement
    assert fcf.paa.Measurement is _PaaMeasurement
    assert fcf.reinsurance.Measurement is _ReinsuranceMeasurement
    for ns in (fcf.gmm, fcf.vfa, fcf.paa, fcf.reinsurance):
        assert "Measurement" in ns.__all__


def test_repr_reads_the_model_namespace_path():
    # repr / str label come from self.model -> "<gmm.Measurement ...>", matching
    # the public path fcf.gmm.Measurement (not the old hardcoded "GMMMeasurement")
    mp = fcf.samples.model_points()
    router = fcf.samples.basis()
    g = fcf.gmm.measure(mp, router)
    assert repr(g).startswith("<gmm.Measurement: n_mp=")
    assert str(g).splitlines()[0] == "<gmm.Measurement -- 11 model points>"
    # reinsurance gained a compact repr here (it relied on the dataclass auto-repr
    # before S5.3, which the rename would have collapsed to a bare "Measurement(")
    r = fcf.reinsurance.measure(mp, router.resolve(("TERM_LIFE_A", "FC")),
                                treaty=QuotaShare(0.5))
    assert repr(r).startswith("<reinsurance.Measurement: n_mp=")
