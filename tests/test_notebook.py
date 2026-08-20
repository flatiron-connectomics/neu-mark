"""The interactive API: same fetches as `ops`, returning DataFrames and writing nothing."""

import pandas as pd
import pytest

from conftest import KV_URL, SZ_URL, URL
from neu_mark import notebook as nb


def test_select_bodies_returns_a_frame_and_writes_nothing(tmp_path, dvid_server,
                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = nb.select_bodies(URL, min_synapses=10)
    assert isinstance(out, pd.DataFrame)
    assert out["body"].tolist() == [1, 2, 3]
    assert list(out.columns) == ["body", "pre", "post", "syn"]
    # the whole point: nothing landed on disk
    assert list(tmp_path.iterdir()) == []


def test_points_returns_two_frames(dvid_server, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pts, rels = nb.points(URL, [1, 2])
    assert len(pts) == 2 and len(rels) == 3
    assert list(tmp_path.iterdir()) == []


def test_points_can_also_return_connections(dvid_server):
    pts, rels, conns = nb.points(URL, [1, 2], connections=True)
    assert len(conns) == 2


def test_body_annotations_returns_one_row_per_body(dvid_server):
    out = nb.body_annotations(KV_URL, [1, 2])
    assert out["body"].tolist() == [1, 2]
    assert out.set_index("body").loc[1, "instance"] == "CAm(L)"


def test_synapse_counts_answers_for_a_given_list_ignoring_any_threshold(dvid_server):
    """select_bodies asks "which bodies are big"; this asks "how big are these"."""
    out = nb.synapse_counts(SZ_URL, [4]).set_index("body")
    assert out.loc[4, "syn"] == 5           # below every threshold, still reported


# --------------------------------------------------------------------------- #
# accepting bodies in whatever form you have them
# --------------------------------------------------------------------------- #
def test_a_dataframe_is_looked_up_by_column(dvid_server):
    """So select_bodies() output feeds straight in with no ["body"] in between."""
    sel = nb.select_bodies(URL, min_synapses=10)
    assert nb.body_ids(sel) == [1, 2, 3]
    assert nb.body_ids(sel.head(2)) == [1, 2]


def test_series_lists_and_arrays_all_work():
    import numpy as np

    assert nb.body_ids(pd.Series([3, 1])) == [3, 1]
    assert nb.body_ids([2, 1]) == [2, 1]
    assert nb.body_ids(np.array([5, 6], dtype="uint64")) == [5, 6]
    assert nb.body_ids((7,)) == [7]


def test_a_frame_with_no_body_column_says_what_it_has():
    with pytest.raises(KeyError, match="no body column"):
        nb.body_ids(pd.DataFrame({"x": [1]}))


def test_a_path_is_read_through_the_body_list_reader(tmp_path):
    path = tmp_path / "b.csv"
    pd.DataFrame({"body": [9, 4]}).to_csv(path, index=False)
    assert nb.body_ids(str(path)) == [4, 9]


def test_body_annotations_without_a_list_explains_why(dvid_server):
    with pytest.raises(ValueError, match="cannot cheaply enumerate"):
        nb.body_annotations(KV_URL)


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def test_source_pins_the_node_once(dvid_server):
    """So every later call in a notebook reads the same version rather than following a
    branch that moves under you."""
    src = nb.source(URL, locked=True)
    assert src["uuid"] == "846e3a" and src["node"]["locked"] is True


def test_an_opened_source_is_reused_not_reresolved(dvid_server):
    src = nb.source(URL, locked=True)
    pts, _ = nb.points(src, [1])
    assert len(pts) == 1
    # `locked` is not passed again, yet the locked node is still the one used
    assert nb.source(src, locked=False)["uuid"] == "846e3a"


def test_the_kind_selects_which_validation_applies(dvid_server):
    assert nb.source(KV_URL, kind="bodies")["instance"] == "labels_annotations"
    assert nb.source(URL, kind="counts")["instance"] == "synapses_labelsz"
    with pytest.raises(ValueError, match="expected annotation"):
        nb.source(KV_URL, kind="points")


def test_an_unknown_kind_lists_the_valid_ones(dvid_server):
    with pytest.raises(ValueError, match="kind must be one of"):
        nb.source(URL, kind="sideways")


def test_a_config_reference_resolves(dvid_server, tmp_path, monkeypatch):
    from neu_mark import config

    path = tmp_path / "c.toml"
    path.write_text('[dvid]\nserver = "dvid.example.org"\n'
                    'uuid = "93fdbc:main"\n\n[instances]\nsyn = "synapses"\n')
    monkeypatch.setenv(config.ENV_VAR, str(path))
    assert nb.source("@syn")["instance"] == "synapses"


def test_a_bare_name_without_a_config_says_what_to_do(monkeypatch, tmp_path):
    from neu_mark import config

    monkeypatch.delenv(config.ENV_VAR, raising=False)
    # No config anywhere above a fresh tmp dir, so the upward walk finds nothing.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "find", lambda start=None: None)
    with pytest.raises(ValueError, match="no config was found to build one from"):
        nb.source("synapses")


# --------------------------------------------------------------------------- #
# the lazy top-level exports
# --------------------------------------------------------------------------- #
def test_the_notebook_names_are_importable_from_the_package_root():
    import neu_mark

    for name in ("source", "select_bodies", "points", "body_annotations",
                 "synapse_counts", "body_ids", "rule", "RuleSet"):
        assert hasattr(neu_mark, name), name
        assert name in dir(neu_mark)


def test_importing_the_package_does_not_pull_pandas():
    """`cli` reads __version__ from __init__, so an eager import here would make
    `neu-mark --help` pay for pandas and neu-vol."""
    import subprocess
    import sys

    code = ("import sys, neu_mark; "
            "print('pandas' in sys.modules, 'neu_vol' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False False", out.stdout + out.stderr


def test_an_unknown_attribute_still_raises_attribute_error():
    import neu_mark

    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        neu_mark.nope
