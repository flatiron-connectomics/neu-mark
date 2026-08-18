"""Labelling each synapse with the neuropil it falls in."""

import pandas as pd
import pytest

from conftest import ROIS, URL, _open
from em_annotation import dvid as ad
from em_annotation import io, notebook as nb, ops, tables


def _src():
    from em_volume_tools.dvid import parse_url

    return {"backend": "dvid", **parse_url(URL)}


# --------------------------------------------------------------------------- #
# validating the ROI set before anything expensive
# --------------------------------------------------------------------------- #
def test_available_rois_lists_the_roi_instances(dvid_server):
    assert ad.available_rois(_open()) == sorted(ROIS)


def test_a_typo_is_caught_up_front_with_close_matches(dvid_server):
    """Checked before the combined volume is built, since that fetches every named ROI."""
    with pytest.raises(ValueError, match=r"named ME\(Q\)"):
        ad.resolve_roi_set(_open(), ["ME(L)", "ME(Q)"])


def test_an_empty_roi_set_explains_why_there_is_no_default(dvid_server):
    with pytest.raises(ValueError, match="deliberately no default"):
        ad.resolve_roi_set(_open(), [])


def test_a_repeated_roi_is_refused(dvid_server):
    with pytest.raises(ValueError, match=r"repeats ME\(L\)"):
        ad.resolve_roi_set(_open(), ["ME(L)", "ME(L)"])


def test_the_order_given_is_preserved(dvid_server):
    """It decides which ROI wins in an accepted overlap, so it must not be sorted."""
    assert ad.resolve_roi_set(_open(), ["OL(L)", "ME(L)"]) == ["OL(L)", "ME(L)"]


# --------------------------------------------------------------------------- #
# overlap
# --------------------------------------------------------------------------- #
def test_overlap_proceeds_by_default_and_is_measured_in_synapses(dvid_server, caplog):
    """Small intersections are expected here and there is no principled tie-break, so
    refusing is the wrong default. What matters is how many of YOUR synapses are affected —
    a voxel-overlap figure cannot answer "should I care"."""
    pts, _ = nb.points(_open(), [1, 2])
    with caplog.at_level("WARNING"):
        out = ad.label_point_rois(_open(), pts, ["ME(L)", "OL(L)"])
    assert out["overlapping"]
    assert "attributed by ROI ORDER alone" in caplog.text
    assert "ambiguous" in out


def test_the_ambiguous_count_comes_from_reversing_the_priority(dvid_server):
    """The measurement: unpack again with the order reversed and see which points change
    hands. Nearly free, because fetching the ROIs is the expensive half and is not repeated.

    In the stub, x=10 takes the FIRST roi and x=11 the LAST, so reversing swaps both — two
    points change and the competing pair is named."""
    pts, _ = nb.points(_open(), [1, 2])
    out = ad.label_point_rois(_open(), pts, ["ME(L)", "OL(L)"])
    assert out["ambiguous"] == 2
    assert out["ambiguous_pairs"] == {"ME(L) | OL(L)": 1, "OL(L) | ME(L)": 1}


def test_a_non_overlapping_set_does_no_second_unpack(dvid_server):
    pts, _ = nb.points(_open(), [1, 2])
    out = ad.label_point_rois(_open(), pts, ["ME(L)", "LO(L)"])
    assert out["overlapping"] == [] and out["ambiguous"] == 0


def test_strict_mode_refuses_and_points_at_the_subtracted_variants(dvid_server):
    pts, _ = nb.points(_open(), [1, 2])
    with pytest.raises(ValueError, match=r"INP\(-ATL\)\(L\)"):
        ad.label_point_rois(_open(), pts, ["ME(L)", "OL(L)"], on_overlap="error")


def test_an_unknown_overlap_policy_lists_the_valid_ones(dvid_server):
    pts, _ = nb.points(_open(), [1, 2])
    with pytest.raises(ValueError, match="on_overlap must be one of"):
        ad.label_point_rois(_open(), pts, ["ME(L)"], on_overlap="shrug")


def test_cli_strict_rois_refuses(tmp_path, dvid_server):
    from em_annotation.cli import main

    with pytest.raises(SystemExit, match="ROIs overlap"):
        main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
              "--rois", "ME(L),OL(L)", "--strict-rois"])


def test_cli_reports_the_ambiguity_by_default(tmp_path, dvid_server, capsys):
    from em_annotation.cli import main

    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
          "--rois", "ME(L),OL(L)"])
    printed = capsys.readouterr().out
    assert "1 ROI pairs intersect; 2 synapse(s)" in printed
    assert "attributed by ROI ORDER" in printed


# --------------------------------------------------------------------------- #
# the roi column
# --------------------------------------------------------------------------- #
def test_the_points_frame_gains_a_roi_column(dvid_server):
    pts, _ = nb.points(_open(), [1, 2], rois=["ME(L)", "LO(L)"])
    assert "roi" in pts.columns
    # body 1's element is at x=10 -> the first ROI; body 2's at x=11 -> the last
    by_body = pts.set_index("body")["roi"]
    assert by_body.loc[1] == "ME(L)" and by_body.loc[2] == "LO(L)"


def test_a_point_in_no_roi_is_null_not_a_region_called_unspecified(dvid_server):
    """`<unspecified>` is neuclease's name for label 0. "In no ROI" is missing data."""
    pts, _ = nb.points(_open(), [1, 2])
    out = ad.label_point_rois(_open(), pts.assign(x=999), ["ME(L)"])
    assert out["points"]["roi"].isna().all()
    assert ad.ROI_UNSPECIFIED not in set(out["points"]["roi"].dropna())
    assert out["unlabeled"] == len(pts) and out["labeled"] == 0


def test_the_roi_label_index_is_not_kept(dvid_server):
    """It is a position in the ROI list and meaningless without it; the list goes into the
    provenance record instead."""
    pts, _ = nb.points(_open(), [1, 2], rois=["ME(L)"])
    assert "roi_label" not in pts.columns


def test_the_callers_frame_is_not_mutated(dvid_server):
    """determine_point_rois works in place; that must not reach the caller's frame."""
    pts, _ = nb.points(_open(), [1, 2])
    before = list(pts.columns)
    ad.label_point_rois(_open(), pts, ["ME(L)"])
    assert list(pts.columns) == before


def test_counts_and_totals_are_reported(dvid_server):
    pts, _ = nb.points(_open(), [1, 2])
    out = ad.label_point_rois(_open(), pts, ["ME(L)", "LO(L)"])
    assert out["labeled"] + out["unlabeled"] == len(pts)
    assert out["counts"] == {"ME(L)": 1, "LO(L)": 1}
    assert out["rois"] == ["ME(L)", "LO(L)"]


def test_an_empty_points_frame_is_handled(dvid_server):
    pts, _ = nb.points(_open(), [999999])
    out = ad.label_point_rois(_open(), pts, ["ME(L)"])
    assert out["labeled"] == 0 and "roi" in out["points"].columns


def test_relationships_get_no_roi_column(dvid_server):
    """A relationship spans two points that may be in different neuropils, so one column
    would have to pick one; the join back to `points` answers either side."""
    _pts, rels = nb.points(_open(), [1, 2], rois=["ME(L)", "LO(L)"])
    assert "roi" not in rels.columns


# --------------------------------------------------------------------------- #
# the per-body aggregate
# --------------------------------------------------------------------------- #
def test_body_roi_counts_is_long_form(dvid_server):
    pts, _ = nb.points(_open(), [1, 2], rois=["ME(L)", "LO(L)"])
    out = tables.body_roi_counts(pts)
    assert set(out.columns) == {"body", "kind", "roi", "synapses"}
    assert int(out["synapses"].sum()) == 2


def test_body_roi_counts_without_a_roi_column_says_how_to_get_one(dvid_server):
    pts, _ = nb.points(_open(), [1, 2])
    with pytest.raises(KeyError, match="--rois"):
        tables.body_roi_counts(pts)


# --------------------------------------------------------------------------- #
# through ops and the CLI
# --------------------------------------------------------------------------- #
def test_ops_writes_the_roi_column_and_records_the_set(tmp_path, dvid_server):
    from em_volume_tools.location import read_json

    out = str(tmp_path / "syn")
    ops.fetch_points(_open(), out, [1, 2], rois=["ME(L)", "LO(L)"])
    assert io.read_table(out, "points")["roi"].notna().sum() == 2
    run = read_json(out, "provenance.json")["run"]
    assert run["rois"]["rois"] == ["ME(L)", "LO(L)"]
    assert run["rois"]["labeled"] == 2


def test_no_rois_means_no_column_and_no_record(tmp_path, dvid_server):
    from em_volume_tools.location import read_json

    out = str(tmp_path / "syn")
    ops.fetch_points(_open(), out, [1, 2])
    assert "roi" not in io.read_table(out, "points").columns
    assert read_json(out, "provenance.json")["run"].get("rois") is None


def test_cli_reports_how_many_synapses_landed_in_a_roi(tmp_path, dvid_server, capsys):
    from em_annotation.cli import main

    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
          "--rois", "ME(L),LO(L)"])
    printed = capsys.readouterr().out
    assert "rois: 2 of 2 synapses inside one of 2 ROIs" in printed


def test_cli_takes_a_roi_file(tmp_path, dvid_server, capsys):
    path = tmp_path / "rois.txt"
    path.write_text("# optic lobe\nME(L)\nLO(L)   # left only\n")
    from em_annotation.cli import main

    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
          "--rois", str(path)])
    assert "of 2 ROIs" in capsys.readouterr().out


def test_cli_resolves_a_config_roi_set(tmp_path, dvid_server, capsys, monkeypatch):
    from em_annotation import config
    from em_annotation.cli import main

    cfg = tmp_path / "c.toml"
    cfg.write_text('[dvid]\nserver = "s"\nuuid = "u"\n\n'
                   '[roi_sets]\noptic = ["ME(L)", "LO(L)"]\n')
    monkeypatch.setenv(config.ENV_VAR, str(cfg))
    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
          "--rois", "@optic"])
    printed = capsys.readouterr().out
    assert "--rois @optic  ->  2 ROIs" in printed


def test_an_unknown_config_roi_set_lists_the_known_ones(tmp_path, monkeypatch):
    from em_annotation import config

    cfg = tmp_path / "c.toml"
    cfg.write_text('[roi_sets]\noptic = ["ME(L)"]\n')
    monkeypatch.setenv(config.ENV_VAR, str(cfg))
    with pytest.raises(ValueError, match="Configured sets: optic"):
        nb.roi_set("@nope")


def test_a_comma_list_and_a_python_list_are_equivalent():
    assert nb.roi_set("ME(L), LO(L)") == ["ME(L)", "LO(L)"]
    assert nb.roi_set(["ME(L)", "LO(L)"]) == ["ME(L)", "LO(L)"]
