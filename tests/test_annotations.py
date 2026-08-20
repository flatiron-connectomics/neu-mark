"""Encoding neuroglancer_annotations_v1 LINE annotations.

The layout numbers here are not read off the spec. They were established by decoding a
published synapse source that neuroglancer demonstrably renders, and by re-encoding one of
its annotations byte for byte with this module — which matched. That validation was done
once, interactively; what is kept here is the *arithmetic* it confirmed, checked offline.

Nothing in this suite contacts an external service, and no third-party data or schema is
reproduced. The property mix below is synthetic; only its shape (four 4-byte, two more
4-byte, a 2-byte signed, a 2-byte unsigned, nine 1-byte → 37 bytes → a 64-byte stride)
carries over, because that is what pins the padding rule and the ordering rule.
"""

import struct

import numpy as np
import pytest

from neu_mark import annotations as ann

#: A property mix whose byte layout matches the validated case: float32 x4, uint32 x2, int16,
#: uint16, uint8 x9 = 37 bytes, which pads a 24-byte LINE geometry to a 64-byte stride.
REFERENCE_PROPERTIES = (
    [{"id": f"f{i}", "type": "float32"} for i in range(4)]
    + [{"id": f"u{i}", "type": "uint32"} for i in range(2)]
    + [{"id": "signed_enum", "type": "int16"}, {"id": "small_index", "type": "uint16"}]
    + [{"id": f"flag{i}", "type": "uint8"} for i in range(9)]
)


# --------------------------------------------------------------------------- #
# the layout
# --------------------------------------------------------------------------- #
def test_record_size_reproduces_the_validated_stride():
    """24 geometry + 37 properties + 3 padding = 64."""
    assert ann.record_size(ann.LINE, REFERENCE_PROPERTIES) == 64


def test_a_group_blob_reproduces_the_validated_size_exactly():
    """8 + 10000*64 + 10000*8 = 720,008, which is what a real level-0 blob of this shape
    measured — the check that first confirmed the group encoding."""
    n = 10_000
    geometry = np.zeros((n, 6), "<f4")
    values = {p["id"]: np.zeros(n) for p in REFERENCE_PROPERTIES}
    blob = ann.encode_group(geometry, list(range(n)),
                            properties=REFERENCE_PROPERTIES, values=values)
    assert len(blob) == 8 + n * 64 + n * 8 == 720_008


def test_a_single_annotation_reproduces_the_validated_88_bytes():
    """64 + (4 + 8) + (4 + 8): two relationships, one id each."""
    values = {p["id"]: 0 for p in REFERENCE_PROPERTIES}
    blob = ann.encode_single([0, 0, 0, 0, 0, 0], properties=REFERENCE_PROPERTIES,
                             values=values, relationships=[[101], [202]])
    assert len(blob) == 88


def test_padding_bytes_are_zero():
    props = [{"id": "a", "type": "uint8"}]          # 24 + 1 -> 3 bytes of padding
    blob = ann.encode_group(np.zeros((1, 6), "<f4"), [7], properties=props,
                            values={"a": [1]})
    body = blob[8:8 + ann.record_size(ann.LINE, props)]
    assert len(body) == 28 and body[25:28] == b"\x00\x00\x00"


# --------------------------------------------------------------------------- #
# property order is a correctness rule
# --------------------------------------------------------------------------- #
def test_properties_are_sorted_descending_by_size():
    """A reader infers each offset from the declared order, so declaring uint8 before float32
    and writing in that order parses into garbage with no error anywhere."""
    mixed = [{"id": "small", "type": "uint8"}, {"id": "big", "type": "float32"},
             {"id": "mid", "type": "int16"}]
    assert [p["id"] for p in ann.sort_properties(mixed)] == ["big", "mid", "small"]


def test_sorting_is_stable_within_a_size_class():
    same = [{"id": "b", "type": "float32"}, {"id": "a", "type": "uint32"}]
    assert [p["id"] for p in ann.sort_properties(same)] == ["b", "a"]


def test_a_correctly_ordered_list_is_left_alone():
    """Descending-size input is already valid, so sorting must be a no-op on it."""
    assert ([p["id"] for p in ann.sort_properties(REFERENCE_PROPERTIES)]
            == [p["id"] for p in REFERENCE_PROPERTIES])


def test_uint64_is_refused_and_says_what_to_do_instead():
    """The reason a wide id is carried as a relationship, with a uint32 copy for shaders."""
    with pytest.raises(ValueError, match="NOT uint64"):
        ann.sort_properties([{"id": "body", "type": "uint64"}])


def test_build_info_sorts_the_properties_it_declares():
    info = ann.build_info(lower_bound=[0, 0, 0], upper_bound=[1, 1, 1],
                          voxel_size_xyz=[8, 8, 8],
                          properties=[{"id": "flag", "type": "uint8"},
                                      {"id": "conf", "type": "float32"}])
    assert [p["id"] for p in info["properties"]] == ["conf", "flag"]


# --------------------------------------------------------------------------- #
# values land where the layout says
# --------------------------------------------------------------------------- #
def test_a_group_round_trips_through_a_hand_written_decoder():
    props = [{"id": "conf", "type": "float32"}, {"id": "roi", "type": "int16"},
             {"id": "kind", "type": "uint8"}]
    n = 3
    geometry = np.arange(n * 6, dtype="<f4").reshape(n, 6)
    values = {"conf": [0.25, 0.5, 0.75], "roi": [1, 2, 3], "kind": [10, 20, 30]}
    ids = [11, 22, 33]
    blob = ann.encode_group(geometry, ids, properties=props, values=values)

    stride = ann.record_size(ann.LINE, props)       # 24 + 4 + 2 + 1 = 31 -> 32
    assert stride == 32
    count = struct.unpack("<Q", blob[:8])[0]
    assert count == n
    rows = np.frombuffer(blob[8:8 + n * stride], "u1").reshape(n, stride)
    assert np.array_equal(rows[:, :24].copy().view("<f4").reshape(n, 6), geometry)
    assert np.allclose(rows[:, 24:28].copy().view("<f4").ravel(), values["conf"])
    assert np.array_equal(rows[:, 28:30].copy().view("<i2").ravel(), values["roi"])
    assert np.array_equal(rows[:, 30], values["kind"])
    assert np.array_equal(np.frombuffer(blob[8 + n * stride:], "<u8"), ids)


def test_a_single_annotation_round_trips_including_its_relationships():
    props = [{"id": "conf", "type": "float32"}]
    blob = ann.encode_single([1, 2, 3, 4, 5, 6], properties=props,
                             values={"conf": 0.875},
                             relationships=[[100, 101], [200]])
    assert struct.unpack("<6f", blob[:24]) == (1, 2, 3, 4, 5, 6)
    assert struct.unpack("<f", blob[24:28])[0] == 0.875
    off = 28 + (-(28) % 4)                            # already aligned
    n1 = struct.unpack("<I", blob[off:off + 4])[0]
    assert n1 == 2
    assert struct.unpack("<2Q", blob[off + 4:off + 20]) == (100, 101)
    n2 = struct.unpack("<I", blob[off + 20:off + 24])[0]
    assert n2 == 1 and struct.unpack("<Q", blob[off + 24:off + 32])[0] == 200
    assert off + 32 == len(blob)


def test_the_decoder_agrees_with_the_hand_written_one():
    """`decode_group` is used to read a written source back, so it must not be the encoder's
    mirror image — the two tests above read the bytes independently, and this pins the decoder
    to that same reading rather than to `encode_group`'s internals."""
    props = [{"id": "conf", "type": "float32"}, {"id": "roi", "type": "int16"},
             {"id": "kind", "type": "uint8"}]
    geometry = np.arange(3 * 6, dtype="<f4").reshape(3, 6)
    values = {"conf": [0.25, 0.5, 0.75], "roi": [1, 2, 3], "kind": [10, 20, 30]}
    got = ann.decode_group(
        ann.encode_group(geometry, [11, 22, 33], properties=props, values=values),
        properties=props)
    assert np.array_equal(got["geometry"], geometry)
    assert np.array_equal(got["ids"], [11, 22, 33])
    assert np.allclose(got["values"]["conf"], values["conf"])
    assert np.array_equal(got["values"]["roi"], values["roi"])
    assert np.array_equal(got["values"]["kind"], values["kind"])


def test_a_single_record_decodes_back_to_its_relationships():
    props = [{"id": "conf", "type": "float32"}]
    got = ann.decode_single(
        ann.encode_single([1, 2, 3, 4, 5, 6], properties=props, values={"conf": 0.875},
                          relationships=[[100, 101], [200]]),
        properties=props, n_relationships=2)
    assert list(got["geometry"]) == [1, 2, 3, 4, 5, 6]
    assert got["values"]["conf"] == 0.875
    assert got["relationships"] == [[100, 101], [200]]


def test_an_empty_relationship_decodes_as_an_empty_list():
    """A line with one endpoint's body unknown declares a zero-length relationship, and the
    difference between that and 'no relationship member' is the whole record's shape."""
    got = ann.decode_single(
        ann.encode_single([0] * 6, relationships=[[7], []]), n_relationships=2)
    assert got["relationships"] == [[7], []]


def test_decoding_with_the_wrong_property_list_is_refused_not_misread():
    """The stride comes from the property list, so decoding with the wrong one would silently
    reinterpret every field. The length check is what turns that into an error."""
    blob = ann.encode_group(np.zeros((4, 6), "<f4"), [1, 2, 3, 4],
                            properties=[{"id": "c", "type": "float32"}],
                            values={"c": np.zeros(4)})
    with pytest.raises(ValueError, match="expected"):
        ann.decode_group(blob, properties=[{"id": "c", "type": "float32"},
                                           {"id": "d", "type": "float32"}])


def test_decoding_a_single_record_with_too_few_relationships_is_refused():
    blob = ann.encode_single([0] * 6, relationships=[[1], [2]])
    with pytest.raises(ValueError, match="trailing bytes"):
        ann.decode_single(blob, n_relationships=1)


def test_geometry_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"expected \(2, 6\)"):
        ann.encode_group(np.zeros((2, 3), "<f4"), [1, 2])


def test_a_missing_property_value_names_the_property():
    with pytest.raises(KeyError, match="conf"):
        ann.encode_group(np.zeros((1, 6), "<f4"), [1],
                         properties=[{"id": "conf", "type": "float32"}], values={})


def test_a_property_column_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="expected"):
        ann.encode_group(np.zeros((2, 6), "<f4"), [1, 2],
                         properties=[{"id": "c", "type": "float32"}], values={"c": [1.0]})


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
def test_info_declares_every_required_member():
    info = ann.build_info(lower_bound=[0, 0, 0], upper_bound=[10, 10, 10],
                          voxel_size_xyz=[8, 8, 8])
    for member in ("@type", "dimensions", "lower_bound", "upper_bound",
                   "annotation_type", "properties", "relationships", "by_id", "spatial"):
        assert member in info, member
    assert info["@type"] == "neuroglancer_annotations_v1"


def test_dimensions_are_metres_and_ordered_xyz():
    """8 nm is 8e-9 m, and the axis order fixes the geometry's order."""
    info = ann.build_info(lower_bound=[0, 0, 0], upper_bound=[1, 1, 1],
                          voxel_size_xyz=[8, 8, 8])
    assert list(info["dimensions"]) == ["x", "y", "z"]
    assert info["dimensions"]["x"] == [8e-09, "m"]


def test_annotation_type_is_lowercase():
    info = ann.build_info(lower_bound=[0, 0, 0], upper_bound=[1, 1, 1],
                          voxel_size_xyz=[8, 8, 8])
    assert info["annotation_type"] == "line"


def test_an_enum_property_carries_parallel_values_and_labels():
    """enum_labels is required exactly when enum_values is given. Index 0 meaning "none"
    is the established convention, which a shader tests for."""
    p = ann.enum_property("roi", ["<unspecified>", "ME(L)", "LO(R)"])
    assert p["enum_values"] == [0, 1, 2]
    assert len(p["enum_labels"]) == len(p["enum_values"])
