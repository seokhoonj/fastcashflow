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

from fastcashflow._measurement_model import (
    MODEL_TAGS, model_tag, supported_model_tags,
)
from fastcashflow.grouping import group, group_of_contracts
from fastcashflow.io import write_measurement
from fastcashflow.report import report
from fastcashflow.movement import roll_forward
from fastcashflow.disclosure import reconciliation_to_frame
from fastcashflow._gmm import GMMMeasurement
from fastcashflow._paa import PAAMeasurement
from fastcashflow._vfa import VFAMeasurement
from fastcashflow._reinsurance import ReinsuranceMeasurement


def test_model_tags_canonical_order():
    # standard order: the three measurement models then reinsurance held
    assert MODEL_TAGS == ("gmm", "paa", "vfa", "reinsurance")


@pytest.mark.parametrize("cls,tag", [
    (GMMMeasurement, "gmm"),
    (PAAMeasurement, "paa"),
    (VFAMeasurement, "vfa"),
    (ReinsuranceMeasurement, "reinsurance"),
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
