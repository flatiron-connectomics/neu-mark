"""Building the neuroglancer segment_properties document.

The format's constraints are the interesting part: exactly one `tags` property, tag values as
increasing indices, no spaces, case-insensitive matching. A document can satisfy all of that
and still be wrong in ways only a viewer shows, so what is testable is pinned here.
"""

import pandas as pd
import pytest

from em_annotation import segprops

BODIES = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4, 5, 6, 7], dtype="uint64"),
    "instance": pd.Series(["Tm2_A2(L)", "VLNP(R)_fragment", "LN_C5(L)_NCL",
                           "PGNG_CV", "irrelevant", "glia", None], dtype="string"),
    "type": pd.Series(["Tm2", None, "LN", None, None, None, None], dtype="string"),
})

COUNTS = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4, 6, 7], dtype="uint64"),
    "pre": [10, 0, 5, 1, 0, 0], "post": [90, 3, 20, 2, 1, 0],
    "syn": [100, 3, 25, 3, 1, 0],
})


def _built(**kw):
    return segprops.build(BODIES, **kw)


def _props(info):
    return {p["id"]: p for p in info["inline"]["properties"]}


# --------------------------------------------------------------------------- #
# the pd.NA trap, twice bitten
# --------------------------------------------------------------------------- #
def test_a_missing_type_produces_no_group_tag():
    """The bug this exists for: `pd.NA is not None` is True and `str(pd.NA)` is the truthy
    "<NA>", so a hand-rolled check tagged 16,606 real bodies `group-<na>` — and it read as
    100% coverage of a field that is populated on 5%."""
    info = _built()["info"]
    tags = _props(info)["tags"]["tags"]
    assert not [t for t in tags if "na" == t or "<na>" in t or "nan" in t]
    assert "group-tm2" in tags and "group-ln" in tags
    assert sum(1 for t in tags if t.startswith("group-")) == 2


def test_group_coverage_matches_the_populated_field():
    report = _built()["report"]
    # two of the five surviving bodies have a `type`
    assert report["coverage"]["group"]["bodies"] == 2


def test_normalize_tag_handles_every_flavour_of_missing():
    assert segprops.normalize_tag(pd.NA) is None
    assert segprops.normalize_tag(None) is None
    assert segprops.normalize_tag(float("nan")) is None
    assert segprops.normalize_tag("   ") is None


# --------------------------------------------------------------------------- #
# what the format requires
# --------------------------------------------------------------------------- #
def test_at_most_one_label_and_one_tags_property():
    props = _built()["info"]["inline"]["properties"]
    for kind in ("label", "tags", "description"):
        assert sum(1 for p in props if p["type"] == kind) <= 1, kind


def test_tag_values_are_increasing_indices_not_strings():
    info = _built()["info"]
    tags = _props(info)["tags"]
    for row in tags["values"]:
        assert all(isinstance(i, int) for i in row)
        assert row == sorted(row), row
        assert len(row) == len(set(row))
        assert all(0 <= i < len(tags["tags"]) for i in row)


def test_tag_descriptions_are_parallel_to_tags():
    tags = _props(_built()["info"])["tags"]
    assert len(tags["tag_descriptions"]) == len(tags["tags"])


def test_the_tags_property_carries_no_description_member():
    """The spec forbids it there, though it is allowed on the others."""
    tags = _props(_built()["info"])["tags"]
    assert "description" not in tags


def test_tags_have_no_spaces_and_no_leading_hash():
    tags = _props(_built()["info"])["tags"]["tags"]
    assert all(" " not in t and not t.startswith("#") for t in tags)


def test_tags_are_lowercased_because_matching_is_case_insensitive():
    """`Traced` and `traced` are the same tag; keeping both would put two
    indistinguishable chips in the viewer."""
    tags = _props(_built()["info"])["tags"]["tags"]
    assert all(t == t.lower() for t in tags)


def test_ids_are_base_ten_strings():
    info = _built()["info"]
    assert all(isinstance(i, str) and i.isdigit() for i in info["inline"]["ids"])


def test_the_at_type_is_what_link_subresources_checks_for():
    from em_volume_tools.ops.subresources import SUBRESOURCE_TYPES

    assert _built()["info"]["@type"] == SUBRESOURCE_TYPES["segment_properties"]


# --------------------------------------------------------------------------- #
# facets
# --------------------------------------------------------------------------- #
def test_label_is_the_raw_instance_string():
    """Which is what lets this ship before a cell-type parse is settled."""
    info = _built()["info"]
    labels = dict(zip(info["inline"]["ids"], _props(info)["instance"]["values"]))
    assert labels["1"] == "Tm2_A2(L)"
    assert labels["3"] == "LN_C5(L)_NCL"


def test_a_body_with_no_instance_falls_back_to_its_id():
    """A blank label is indistinguishable from a missing property in the viewer."""
    info = _built()["info"]
    labels = dict(zip(info["inline"]["ids"], _props(info)["instance"]["values"]))
    assert labels["7"] == "7"


def test_each_facet_lands_on_the_right_body():
    def tags_of(body):
        info = _built()["info"]
        tags = _props(info)["tags"]
        i = info["inline"]["ids"].index(str(body))
        return {tags["tags"][j] for j in tags["values"][i]}

    assert tags_of(1) == {"group-tm2", "side-l", "col-a2"}
    assert tags_of(2) == {"side-r", "fragment"}
    assert tags_of(3) == {"group-ln", "side-l", "col-c5", "nucleated"}
    assert tags_of(4) == {"cervical"}


def test_columns_are_multi_valued():
    frame = pd.DataFrame({"body": pd.Series([1], dtype="uint64"),
                          "instance": ["X_D2_E3(L)"]})
    info = segprops.build(frame)["info"]
    tags = _props(info)["tags"]
    assert {tags["tags"][i] for i in tags["values"][0]} >= {"col-d2", "col-e3"}


def test_noise_is_excluded_entirely_not_merely_untagged():
    result = _built()
    assert "5" not in result["info"]["inline"]["ids"]        # 'irrelevant'
    assert result["report"]["excluded"]["irrelevant"] == 1


def test_glia_are_kept_and_tagged_by_default():
    result = _built()
    assert "6" in result["info"]["inline"]["ids"]
    tags = _props(result["info"])["tags"]
    i = result["info"]["inline"]["ids"].index("6")
    assert "glia" in {tags["tags"][j] for j in tags["values"][i]}


def test_glia_can_be_dropped_on_request():
    result = _built(keep_glia=False)
    assert "6" not in result["info"]["inline"]["ids"]
    assert result["report"]["excluded"]["glia"] == 1


def test_a_doubt_mark_normalizes_to_the_plain_completeness_tag():
    frame = pd.DataFrame({"body": pd.Series([1], dtype="uint64"),
                          "instance": ["Tm(R)_truncated?"]})
    info = segprops.build(frame)["info"]
    tags = _props(info)["tags"]
    assert {tags["tags"][i] for i in tags["values"][0]} == {"side-r", "truncated"}


def test_a_like_suffix_stays_in_the_label_and_is_not_tagged():
    frame = pd.DataFrame({"body": pd.Series([1], dtype="uint64"),
                          "instance": ["DPM-like(L)"]})
    result = segprops.build(frame)
    info = result["info"]
    assert _props(info)["instance"]["values"] == ["DPM-like(L)"]
    tags = _props(info)["tags"]
    assert {tags["tags"][i] for i in tags["values"][0]} == {"side-l"}


def test_everything_excluded_is_an_error_not_an_empty_document():
    frame = pd.DataFrame({"body": pd.Series([1], dtype="uint64"),
                          "instance": ["irrelevant"]})
    with pytest.raises(ValueError, match="every body was excluded"):
        segprops.build(frame)


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def test_numbers_are_uint32_because_float32_would_round_them():
    """float32 is exact only to 2^24 = 16,777,216, and real bodies reach 86 million
    voxels — so a float32 `voxels` property silently rounds."""
    sizes = {1: 86_315_764, 2: 4_510_070}
    info = _built(counts=COUNTS, sizes=sizes)["info"]
    for name in ("pre", "post", "syn", "voxels"):
        p = _props(info)[name]
        assert p["type"] == "number" and p["data_type"] == "uint32", name
    voxels = dict(zip(info["inline"]["ids"], _props(info)["voxels"]["values"]))
    assert voxels["1"] == 86_315_764            # exact, not 86315768


def test_number_values_are_parallel_to_ids_and_ordered_the_same():
    info = _built(counts=COUNTS)["info"]
    ids, pre = info["inline"]["ids"], _props(info)["pre"]["values"]
    assert len(pre) == len(ids)
    assert dict(zip(ids, pre))["1"] == 10 and dict(zip(ids, pre))["3"] == 5


def test_a_body_missing_from_counts_gets_zero_not_a_hole():
    info = _built(counts=COUNTS.iloc[:1])["info"]
    assert len(_props(info)["pre"]["values"]) == len(info["inline"]["ids"])


def test_numbers_are_omitted_when_not_supplied():
    info = _built()["info"]
    assert [p["id"] for p in info["inline"]["properties"]] == ["instance", "tags"]


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def test_it_writes_where_the_info_key_will_point(tmp_path):
    from em_volume_tools.location import read_json

    dst = str(tmp_path / "vol")
    key = segprops.write(dst, _built()["info"])
    assert key == "segment_properties/info"
    assert read_json(dst, "segment_properties", "info")["@type"] == segprops.AT_TYPE


def test_the_module_never_opens_a_file_directly():
    """Every write goes through location, or an s3 destination silently writes nothing."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(segprops))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "open" not in called
