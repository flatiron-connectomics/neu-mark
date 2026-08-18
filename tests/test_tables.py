"""The table model: parsing, axis order, and the partner join.

Nothing here touches DVID. The element JSON shape is copied from real dvid.example.org
responses, including the string-typed `conf` and the ragged `Prop` keys.
"""

import numpy as np
import pandas as pd
import pytest

from em_annotation import tables


def el(pos_xyz, kind, *, rels=(), **prop):
    """One element, in DVID's own shape: Pos is XYZ."""
    return {"Pos": list(pos_xyz), "Kind": kind, "Tags": [],
            "Prop": {str(k): v for k, v in prop.items()},
            "Rels": [{"Rel": r, "To": list(t)} for r, t in rels]}


# A tbar at xyz (10, 20, 30) with two PSDs, one of which is on the same body.
TBAR = el((10, 20, 30), "PreSyn", conf="0.9",
          rels=[("PreSynTo", (11, 21, 31)), ("PreSynTo", (99, 99, 99))])
PSD = el((11, 21, 31), "PostSyn", conf="0.5", rels=[("PostSynTo", (10, 20, 30))])


# --------------------------------------------------------------------------- #
# axis order
# --------------------------------------------------------------------------- #
def test_dvid_xyz_becomes_named_zyx_columns():
    """Pos is [x, y, z]; the table's z/y/x must not be a transposed copy of it."""
    pts, _ = tables.elements_to_frames([TBAR], body=7)
    row = pts.iloc[0]
    assert (row["x"], row["y"], row["z"]) == (10, 20, 30)
    # column order is zyx, which is this package's in-memory convention
    assert list(pts.columns[:4]) == ["body", "z", "y", "x"]


def test_positions_helpers_are_the_only_order_decision():
    pts, _ = tables.elements_to_frames([TBAR, PSD], body=7)
    assert tables.positions_zyx(pts).tolist() == [[30, 20, 10], [31, 21, 11]]
    assert tables.positions_xyz(pts).tolist() == [[10, 20, 30], [11, 21, 31]]
    # and they are exact reverses of one another, which is the property that matters
    assert (tables.positions_zyx(pts) == tables.positions_xyz(pts)[:, ::-1]).all()


def test_positions_helpers_work_on_prefixed_columns():
    _, rels = tables.elements_to_frames([PSD], body=7)
    assert tables.positions_xyz(rels, "to_").tolist() == [[10, 20, 30]]


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_properties_become_columns_and_conf_is_numeric():
    """DVID stores conf as a STRING in Prop; a string confidence cannot be thresholded."""
    pts, _ = tables.elements_to_frames([TBAR], body=7)
    pts = tables.combine([(pts, pd.DataFrame())])[0]
    assert pts["conf"].dtype == np.float32
    assert pts["conf"].iloc[0] == pytest.approx(0.9)


def test_unknown_properties_are_kept_under_their_own_name():
    weird = el((1, 2, 3), "PreSyn", conf="1.0", user="jwu", annotation="ingested by jwu")
    pts, _ = tables.elements_to_frames([weird], body=7)
    assert pts["user"].iloc[0] == "jwu"
    assert pts["annotation"].iloc[0] == "ingested by jwu"


def test_body_comes_from_the_fetch_key_not_the_element():
    """An element carries no body; the /label/<id> request it came from is what knows."""
    pts, rels = tables.elements_to_frames([TBAR], body=13481220)
    assert pts["body"].tolist() == [13481220]
    assert rels["from_body"].tolist() == [13481220, 13481220]


def test_body_ids_stay_uint64():
    big = 2**63 + 12345
    pts, rels = tables.elements_to_frames([TBAR], body=big)
    pts, rels = tables.combine([(pts, rels)])
    assert pts["body"].dtype == np.uint64
    assert int(pts["body"].iloc[0]) == big


def test_a_relationship_with_a_null_target_is_kept_as_a_row():
    """A deleted partner leaves a dangling Rel. Dropping it would hide the count."""
    dangling = {"Pos": [1, 2, 3], "Kind": "PreSyn", "Rels": [{"Rel": "PreSynTo"}]}
    pts, rels = tables.elements_to_frames([dangling], body=7)
    pts, rels = tables.combine([(pts, rels)])
    assert len(rels) == 1
    assert pd.isna(rels["to_z"].iloc[0]) and pd.isna(rels["to_body"].iloc[0])


def test_a_malformed_position_is_an_error():
    with pytest.raises(ValueError, match="expected three coordinates"):
        tables.elements_to_frames([{"Pos": [1, 2], "Kind": "PreSyn"}], body=7)


def test_empty_response_gives_typed_empty_frames():
    pts, rels = tables.elements_to_frames([], body=7)
    assert len(pts) == 0 and len(rels) == 0
    assert pts["body"].dtype == np.uint64


# --------------------------------------------------------------------------- #
# the partner join
# --------------------------------------------------------------------------- #
def _two_bodies():
    """Body 1 holds the tbar, body 2 holds the matching PSD. One PSD is off-set."""
    return tables.combine([tables.elements_to_frames([TBAR], body=1),
                           tables.elements_to_frames([PSD], body=2)])


def test_partner_bodies_resolve_by_position_without_a_labelmap_lookup():
    _, rels = _two_bodies()
    pre = rels[(rels["rel"] == "PreSynTo") & (rels["to_x"] == 11)]
    assert pre["from_body"].iloc[0] == 1 and pre["to_body"].iloc[0] == 2
    post = rels[rels["rel"] == "PostSynTo"]
    assert post["from_body"].iloc[0] == 2 and post["to_body"].iloc[0] == 1


def test_an_unfetched_partner_leaves_to_body_null_not_zero():
    """A missing partner must be distinguishable from body 0."""
    _, rels = _two_bodies()
    outside = rels[rels["to_x"] == 99]
    assert len(outside) == 1 and pd.isna(outside["to_body"].iloc[0])


def test_to_body_is_nullable_uint64_never_float():
    """A body id that has been through float64 may have been rounded."""
    _, rels = _two_bodies()
    assert str(rels["to_body"].dtype) == "UInt64"


def test_duplicate_positions_are_refused_because_they_make_the_join_ambiguous():
    same = tables.combine([tables.elements_to_frames([TBAR], body=1)])[0]
    doubled = pd.concat([same, same], ignore_index=True)
    with pytest.raises(ValueError, match="share a position"):
        tables.position_index(doubled)


# --------------------------------------------------------------------------- #
# connections: both directions, de-duplicated
# --------------------------------------------------------------------------- #
def test_connections_dedupes_a_pair_seen_from_both_ends():
    """The tbar->psd edge appears as PreSynTo from body 1 and PostSynTo from body 2."""
    _, rels = _two_bodies()
    conns = tables.connections(rels)
    matched = conns[conns["post_x"] == 11]
    assert len(matched) == 1
    assert (matched["pre_body"].iloc[0], matched["post_body"].iloc[0]) == (1, 2)


def test_connections_keeps_edges_anchored_at_only_one_end():
    """The two directions are NOT redundant: each reaches partners the other cannot."""
    _, rels = _two_bodies()
    conns = tables.connections(rels)
    # the off-set PSD at xyz 99,99,99 is still an edge, with an unknown post body
    outside = conns[conns["post_x"] == 99]
    assert len(outside) == 1
    assert outside["pre_body"].iloc[0] == 1 and pd.isna(outside["post_body"].iloc[0])


def test_connections_orientation_comes_from_the_rel_name():
    """PostSynTo means the element is the PSD, whatever its Kind claims."""
    _, rels = _two_bodies()
    conns = tables.connections(rels)
    row = conns[conns["post_x"] == 11].iloc[0]
    # tbar is at xyz (10,20,30) -> pre_*, psd at (11,21,31) -> post_*
    assert (row["pre_z"], row["pre_y"], row["pre_x"]) == (30, 20, 10)
    assert (row["post_z"], row["post_y"], row["post_x"]) == (31, 21, 11)


def test_non_synaptic_relationships_are_ignored_by_connections_but_kept_raw():
    grouped = el((1, 1, 1), "Note", rels=[("GroupedWith", (2, 2, 2))])
    pts, rels = tables.combine([tables.elements_to_frames([grouped], body=5)])
    assert list(rels["rel"]) == ["GroupedWith"]
    assert len(tables.connections(rels)) == 0


def test_match_rate_reports_coverage_not_data_quality():
    _, rels = _two_bodies()
    m = tables.match_rate(tables.connections(rels))
    assert m["pairs"] == 2 and m["both_ends"] == 1 and m["one_end"] == 1
    assert m["fraction"] == 0.5


def test_match_rate_of_nothing_is_none_not_zero():
    m = tables.match_rate(tables.connections(tables._empty_rels()))
    assert m["pairs"] == 0 and m["fraction"] is None


# --------------------------------------------------------------------------- #
# body annotations
# --------------------------------------------------------------------------- #
RAGGED = {
    "13481220": {"bodyid": 13481220, "user": "kshinomiya", "instance": "CAm(L)",
                 "instance_user": "", "status": "Traced"},
    "18126428": {"status": "Traced", "bodyid": 18126428, "instance": "MT(L)"},
}


def test_ragged_body_records_normalise_to_the_union_of_fields():
    df = tables.keyvalues_to_frame(RAGGED)
    assert len(df) == 2
    assert set(["body", "status", "instance", "user", "json"]) <= set(df.columns)
    assert pd.isna(df.set_index("body").loc[18126428, "user"])


def test_instance_keeps_its_name_even_though_it_means_neuron_name():
    """Renaming it would break every join against neuclease and neuprint output."""
    df = tables.keyvalues_to_frame(RAGGED)
    assert df.set_index("body").loc[13481220, "instance"] == "CAm(L)"


def test_status_stays_a_string_so_an_unknown_value_cannot_fail_the_fetch():
    df = tables.keyvalues_to_frame(
        {"1": {"status": "Freshly Invented Status", "bodyid": 1}})
    assert str(df["status"].dtype) == "string"
    assert df["status"].iloc[0] == "Freshly Invented Status"


def test_non_body_keys_are_skipped():
    df = tables.keyvalues_to_frame({**RAGGED, "schema": {"version": 2}})
    assert len(df) == 2


def test_a_record_disagreeing_with_its_key_is_an_error():
    with pytest.raises(ValueError, match="says bodyid"):
        tables.keyvalues_to_frame({"5": {"bodyid": 6, "status": "Traced"}})


def test_the_raw_json_is_preserved_for_pushing_edits_back():
    df = tables.keyvalues_to_frame(RAGGED)
    assert df["json"].iloc[0] == RAGGED["13481220"]


def test_bodies_frame_is_sorted_by_body():
    df = tables.keyvalues_to_frame({"20": {"bodyid": 20}, "3": {"bodyid": 3}})
    assert df["body"].tolist() == [3, 20]
