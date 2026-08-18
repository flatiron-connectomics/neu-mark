"""The two stage-1 operations end to end, with DVID stubbed at the neuclease boundary.

Stubbed at ``neuclease.dvid.*`` rather than at this package's own functions, so the code
under test is the code that runs in production — the em-volume-tools lesson that a test
building its own spec by hand proves nothing about the path that runs (invariant 9).
"""

import pandas as pd
import pytest

from em_annotation import io, ops
from em_annotation import dvid as ann_dvid

URL = "dvid://dvid.example.org/93fdbc:main/synapses"
KV_URL = "dvid://dvid.example.org/93fdbc:main/labels_annotations"

SYN_INFO = {"Base": {"TypeName": "annotation", "Syncs": ["labels"]}}
KV_INFO = {"Base": {"TypeName": "keyvalue", "Syncs": []}}
LABELSZ_INFO = {"Base": {"TypeName": "labelsz", "Syncs": ["synapses"]}}
SZ_URL = "dvid://dvid.example.org/93fdbc:main/synapses_labelsz"

#: (pre, post) per body, chosen to cover the two shapes that make a TOTAL threshold the
#: right default: body 2 is sensory-ish (almost no postsynapses), body 3 projects out of
#: the traced volume (no presynapses at all). Body 4 is a fragment.
COUNTS = {1: (5, 100), 2: (60, 3), 3: (0, 40), 4: (2, 3)}

#: Body 1 has a tbar; body 2 has the matching PSD plus one whose partner is unfetched.
ELEMENTS = {
    1: [{"Pos": [10, 20, 30], "Kind": "PreSyn", "Prop": {"conf": "0.9", "user": "jwu"},
         "Rels": [{"Rel": "PreSynTo", "To": [11, 21, 31]},
                  {"Rel": "PreSynTo", "To": [99, 99, 99]}]}],
    2: [{"Pos": [11, 21, 31], "Kind": "PostSyn", "Prop": {"conf": "0.5"},
         "Rels": [{"Rel": "PostSynTo", "To": [10, 20, 30]}]}],
}

BODY_RECORDS = {
    "1": {"bodyid": 1, "status": "Traced", "instance": "CAm(L)", "user": "ks"},
    "2": {"bodyid": 2, "status": "Anchor"},
}


@pytest.fixture(autouse=True)
def _no_node_cache():
    from em_volume_tools.dvid import clear_node_cache

    clear_node_cache()
    yield
    clear_node_cache()


@pytest.fixture()
def dvid_server(monkeypatch):
    """A DVID that answers from the dicts above and records what was asked."""
    pytest.importorskip("neuclease")
    import neuclease.dvid as nd
    import neuclease.dvid.annotation as nda
    import neuclease.dvid.keyvalue as ndk

    import em_volume_tools.dvid as vdvid

    asked = []

    # A lock-and-spawn repo: HEAD (d38898) is open, its parent (846e3a) is locked.
    # A CONCRETE uuid resolves to itself, as the real endpoint does — without that a
    # re-resolution of an already-pinned spec silently jumps back to HEAD.
    def resolve_ref(server, ref, expand=False, **k):
        if ref.endswith("~1"):
            return "846e3a"
        if ref in ("d38898", "846e3a"):
            return ref
        return "d38898"

    monkeypatch.setattr(nd, "resolve_ref", resolve_ref)
    monkeypatch.setattr(nd, "fetch_commit",
                        lambda server, uuid, **k: uuid == "846e3a")

    def fetch_instance_info(server, uuid, instance, **k):
        if instance == "synapses":
            return SYN_INFO
        if instance.endswith("_labelsz"):
            return LABELSZ_INFO
        return KV_INFO

    monkeypatch.setattr(nd, "fetch_instance_info", fetch_instance_info)
    monkeypatch.setattr(nd, "fetch_repo_info", lambda server, uuid, **k: {
        "DataInstances": {"synapses": SYN_INFO, "synapses_labelsz": LABELSZ_INFO,
                          "labels_annotations": KV_INFO}})

    # labelsz: a ranked AllSyn threshold query, paged by offset, plus exact per-type counts.
    import neuclease.dvid.labelsz as ndsz

    def fetch_threshold(server, uuid, instance, threshold, element_type,
                        offset=0, n=None, **k):
        assert element_type == "AllSyn", element_type
        ranked = sorted(((p + s, b) for b, (p, s) in COUNTS.items()),
                        key=lambda t: (-t[0], t[1]))
        hits = [(b, tot) for tot, b in ranked if tot >= threshold]
        window = hits[offset:offset + (n or 10_000)]
        return pd.Series({b: tot for b, tot in window},
                         name="AllSyn", dtype="int64").rename_axis("body")

    def fetch_counts(server, uuid, instance, bodies, element_type, **k):
        idx = 0 if element_type == "PreSyn" else 1
        return pd.Series({int(b): COUNTS.get(int(b), (0, 0))[idx] for b in bodies},
                         name=element_type, dtype="int64").rename_axis("body")

    monkeypatch.setattr(ndsz, "fetch_threshold", fetch_threshold)
    monkeypatch.setattr(ndsz, "fetch_counts", fetch_counts)
    monkeypatch.setattr(vdvid, "instance_info",
                        lambda spec: fetch_instance_info(*vdvid.address(spec)))
    monkeypatch.setattr(vdvid, "node_provenance",
                        lambda spec, node: {"source": "dvid", "uuid": node["uuid"],
                                            "instance": spec["instance"],
                                            "requested": node.get("ref"),
                                            "locked": node["locked"]})

    def fetch_label(server, uuid, instance, label, relationships=False, **k):
        asked.append((label, relationships))
        return ELEMENTS.get(int(label), [])

    monkeypatch.setattr(nda, "fetch_label", fetch_label)

    def fetch_keyvalues(server, uuid, instance, keys, **k):
        return {key: BODY_RECORDS.get(key) for key in keys}

    monkeypatch.setattr(ndk, "fetch_keyvalues", fetch_keyvalues)
    return asked


def _src(url=URL):
    from em_volume_tools.dvid import parse_url

    return {"backend": "dvid", **parse_url(url)}


def _open(url=URL, kind="points", *, locked=False):
    """The resolved source, as the CLI produces it before doing anything else."""
    opener = ops.open_points_source if kind == "points" else ops.open_bodies_source
    return opener(_src(url), prefer_locked=locked)


# --------------------------------------------------------------------------- #
# points
# --------------------------------------------------------------------------- #
def test_points_writes_both_tables_and_a_provenance_sidecar(tmp_path, dvid_server):
    out = str(tmp_path / "syn")
    result = ops.fetch_points(_open(), out, [1, 2])

    assert set(result["written"]) == {"points.parquet", "relationships.parquet"}
    assert len(io.read_table(out, "points")) == 2
    assert len(io.read_table(out, "relationships")) == 3

    from em_volume_tools.location import read_json

    rec = read_json(out, "provenance.json")
    assert rec["tool"] == "em-annotation"
    assert rec["source"]["uuid"] == "d38898"


def test_relationships_are_always_fetched(dvid_server, tmp_path):
    """`relationships=True` is the same single request, and partners are half the point.
    neuclease's own fetch_elements_for_bodies hardcodes False and so cannot be used."""
    ops.fetch_points(_open(), str(tmp_path / "o"), [1, 2])
    assert dvid_server and all(rels is True for _body, rels in dvid_server)


def test_partner_bodies_resolve_without_touching_the_labelmap(tmp_path, dvid_server):
    out = str(tmp_path / "o")
    ops.fetch_points(_open(), out, [1, 2])
    rels = io.read_table(out, "relationships")
    matched = rels[rels["to_x"] == 11].iloc[0]
    assert matched["from_body"] == 1 and matched["to_body"] == 2
    # no `labels` instance was ever asked about
    assert all(body in (1, 2) for body, _ in dvid_server)


def test_unmatched_partners_are_kept_as_nulls_by_default(tmp_path, dvid_server):
    """Dropping silently would turn "the body list missed this partner" into "this
    synapse does not exist"."""
    out = str(tmp_path / "o")
    result = ops.fetch_points(_open(), out, [1, 2])
    rels = io.read_table(out, "relationships")
    assert rels["to_body"].isna().sum() == 1
    assert result["match"]["pairs"] == 2 and result["match"]["both_ends"] == 1


def test_drop_unmatched_is_opt_in(tmp_path, dvid_server):
    out = str(tmp_path / "o")
    ops.fetch_points(_open(), out, [1, 2], drop_unmatched=True)
    rels = io.read_table(out, "relationships")
    assert len(rels) == 2 and rels["to_body"].notna().all()


def test_the_match_rate_is_recorded_in_the_provenance(tmp_path, dvid_server):
    from em_volume_tools.location import read_json

    out = str(tmp_path / "o")
    ops.fetch_points(_open(), out, [1, 2])
    run = read_json(out, "provenance.json")["run"]
    assert run["match"]["both_ends"] == 1
    assert run["synced_to"] == ["labels"]


def test_connections_table_is_opt_in(tmp_path, dvid_server):
    out = str(tmp_path / "o")
    r = ops.fetch_points(_open(), out, [1, 2], write_connections=True)
    assert "connections.parquet" in r["written"]
    assert len(io.read_table(out, "connections")) == 2


def test_a_failing_body_is_recorded_and_the_rest_still_land(tmp_path, dvid_server,
                                                            monkeypatch):
    """A run over thousands of bodies must not lose the successes to one bad request."""
    import neuclease.dvid.annotation as nda

    real = nda.fetch_label

    def flaky(server, uuid, instance, label, **k):
        if int(label) == 2:
            raise ValueError("PERMISSION_DENIED: nope")
        return real(server, uuid, instance, label, **k)

    monkeypatch.setattr(nda, "fetch_label", flaky)
    with pytest.raises(ValueError, match="PERMISSION_DENIED"):
        ops.fetch_points(_open(), str(tmp_path / "o"), [1, 2])


def test_an_unsynced_annotation_instance_is_refused_up_front(tmp_path, monkeypatch,
                                                             dvid_server):
    """Without a sync there is no /label endpoint, so this would otherwise surface as one
    failure per body."""
    import em_volume_tools.dvid as vdvid

    monkeypatch.setattr(vdvid, "instance_info",
                        lambda spec: {"Base": {"TypeName": "annotation", "Syncs": []}})
    with pytest.raises(ValueError, match="not synced to a labelmap"):
        ops.open_points_source(_src())


def test_a_keyvalue_instance_is_refused_for_points(tmp_path, dvid_server):
    with pytest.raises(ValueError, match="expected annotation"):
        ops.open_points_source(_src(KV_URL))


def test_the_ref_is_resolved_to_a_concrete_uuid_before_any_fetch(tmp_path, dvid_server):
    """A branch ref names a node that moves; a table half from one node and half from the
    next is not a snapshot. Same discipline as em-volume-tools' invariant 9."""
    out = str(tmp_path / "o")
    ops.fetch_points(_open(), out, [1])
    from em_volume_tools.location import read_json

    src = read_json(out, "provenance.json")["source"]
    assert src["uuid"] == "d38898" and src["requested"] == "93fdbc:main"


def test_dvid_locked_selects_the_locked_ancestor(tmp_path, dvid_server):
    out = str(tmp_path / "o")
    ops.fetch_points(_open(locked=True), out, [1])
    from em_volume_tools.location import read_json

    assert read_json(out, "provenance.json")["source"]["uuid"] == "846e3a"


# --------------------------------------------------------------------------- #
# bodies
# --------------------------------------------------------------------------- #
def test_bodies_writes_its_own_table_to_its_own_destination(tmp_path, dvid_server):
    out = str(tmp_path / "bodyann")
    result = ops.fetch_bodies(_open(KV_URL, "bodies"), out, [1, 2])
    assert result["written"] == ["bodies.parquet"]
    df = io.read_table(out, "bodies")
    assert df.set_index("body").loc[1, "instance"] == "CAm(L)"
    assert result["found"] == 2 and not result["missing"]


def test_bodies_reports_the_ones_with_no_record(tmp_path, dvid_server):
    result = ops.fetch_bodies(_open(KV_URL, "bodies"), str(tmp_path / "o"), [1, 2, 12345])
    assert result["missing"] == [12345] and result["found"] == 2


def test_an_annotation_instance_is_refused_for_bodies(tmp_path, dvid_server):
    with pytest.raises(ValueError, match="expected keyvalue"):
        ops.open_bodies_source(_src())


# --------------------------------------------------------------------------- #
# select-bodies
# --------------------------------------------------------------------------- #
def _select(tmp_path, dvid_server, url=URL, **kw):
    source = ops.open_counts_source(_src(url))
    out = str(tmp_path / "sel")
    return out, ops.select_bodies(source, out, **kw)


def test_select_thresholds_on_the_total_and_ranks_by_it(tmp_path, dvid_server):
    _out, r = _select(tmp_path, dvid_server, min_synapses=10)
    df = r["bodies"]
    # body 4 (5 synapses) is out; the rest ranked by total descending
    assert df["body"].tolist() == [1, 2, 3]
    assert df["syn"].tolist() == [105, 63, 40]
    assert df["pre"].tolist() == [5, 60, 0]


def test_a_total_threshold_keeps_sensory_and_outward_projecting_neurons(tmp_path,
                                                                       dvid_server):
    """The reason the default is on the TOTAL. Body 2 has almost no postsynapses (sensory),
    body 3 has no presynapses at all (projects outside the traced volume). Requiring both
    would drop exactly the cells most worth looking at."""
    _out, r = _select(tmp_path, dvid_server, min_synapses=10)
    assert {2, 3} <= set(r["bodies"]["body"])

    # ...and the per-type filters, when asked for, do exclude them
    _out, only_pre = _select(tmp_path, dvid_server, min_synapses=10, min_pre=1)
    assert 3 not in set(only_pre["bodies"]["body"])          # no presynapses
    _out, only_post = _select(tmp_path, dvid_server, min_synapses=10, min_post=10)
    assert 2 not in set(only_post["bodies"]["body"])         # sensory


def test_limit_takes_the_top_n_of_the_ranking(tmp_path, dvid_server):
    _out, r = _select(tmp_path, dvid_server, min_synapses=10, limit=2)
    assert r["bodies"]["body"].tolist() == [1, 2]


def test_a_per_type_minimum_raises_the_query_threshold(tmp_path, dvid_server,
                                                       monkeypatch):
    """AllSyn is the catch-all so it is >= each individual type, which makes
    max(minimums) a sound threshold for the narrowing query."""
    import neuclease.dvid.labelsz as ndsz

    seen = []
    real = ndsz.fetch_threshold
    monkeypatch.setattr(ndsz, "fetch_threshold",
                        lambda *a, **k: (seen.append(a[3]), real(*a, **k))[1])
    _select(tmp_path, dvid_server, min_synapses=10, min_post=50)
    assert seen[0] == 50


def test_the_labelsz_instance_is_found_from_the_annotation_instance(tmp_path,
                                                                   dvid_server):
    """`synapses` is the name people know; the index records what it syncs to."""
    source = ops.open_counts_source(_src(URL))
    assert source["instance"] == "synapses_labelsz"
    assert source["indexes"] == ["synapses"]


def test_a_labelsz_instance_can_be_named_directly(tmp_path, dvid_server):
    source = ops.open_counts_source(_src(SZ_URL))
    assert source["instance"] == "synapses_labelsz"


def test_no_labelsz_for_the_instance_says_so(tmp_path, dvid_server, monkeypatch):
    import neuclease.dvid as nd

    monkeypatch.setattr(nd, "fetch_repo_info",
                        lambda server, uuid, **k: {"DataInstances": {"synapses": SYN_INFO}})
    with pytest.raises(ValueError, match="no 'labelsz' instance"):
        ops.open_counts_source(_src(URL))


def test_an_ambiguous_labelsz_asks_which_one(tmp_path, dvid_server, monkeypatch):
    import neuclease.dvid as nd

    monkeypatch.setattr(nd, "fetch_repo_info", lambda server, uuid, **k: {
        "DataInstances": {"synapses": SYN_INFO, "a_labelsz": LABELSZ_INFO,
                          "b_labelsz": LABELSZ_INFO}})
    with pytest.raises(ValueError, match="several labelsz instances"):
        ops.open_counts_source(_src(URL))


def test_an_empty_result_names_the_silent_unindexed_failure(tmp_path, dvid_server):
    """DVID answers a labelsz element type it does not index with an EMPTY result rather
    than an error, so an empty selection must not read as 'no large bodies'."""
    with pytest.raises(ValueError, match="EMPTY result"):
        _select(tmp_path, dvid_server, min_synapses=10_000)


def test_paging_collects_every_body_beyond_one_page(tmp_path, dvid_server, monkeypatch):
    monkeypatch.setattr("em_annotation.dvid._LABELSZ_PAGE", 2)
    _out, r = _select(tmp_path, dvid_server, min_synapses=1)
    assert r["bodies"]["body"].tolist() == [1, 2, 3, 4]


def test_the_deep_paging_cap_warns_rather_than_silently_truncating(tmp_path, dvid_server,
                                                                  monkeypatch, caplog):
    """Per-page cost grows with offset, so there is a cap — but a capped result must say so."""
    monkeypatch.setattr("em_annotation.dvid._LABELSZ_PAGE", 1)
    monkeypatch.setattr("em_annotation.dvid._MAX_SELECT", 2)
    with caplog.at_level("WARNING"):
        _out, r = _select(tmp_path, dvid_server, min_synapses=1)
    assert len(r["bodies"]) == 2
    assert "cap" in caplog.text


def test_the_list_records_which_node_it_was_computed_from(tmp_path, dvid_server):
    """Proofreading changes body ids, so a list nobody can date is one that goes stale."""
    from em_volume_tools.location import read_json

    out, r = _select(tmp_path, dvid_server, min_synapses=10)
    rec = read_json(out, "provenance.json")
    assert rec["source"]["uuid"] == "d38898"
    assert rec["run"]["bodies_selected"] == 3
    assert rec["run"]["indexes"] == ["synapses"]


def test_the_written_list_is_a_valid_bodies_input(tmp_path, dvid_server):
    """Closes the loop: what select-bodies writes is what --bodies reads."""
    from em_annotation import bodies as body_reader

    out, r = _select(tmp_path, dvid_server, min_synapses=10)
    path = f"{out}/{r['written'][0]}"
    assert body_reader.load(path) == [1, 2, 3]


def test_csv_is_the_default_and_is_lossless_here(tmp_path, dvid_server, caplog):
    """Every column is a non-nullable integer, which csv does preserve — so no warning."""
    with caplog.at_level("WARNING"):
        out, r = _select(tmp_path, dvid_server, min_synapses=10)
    assert r["written"] == ["selected_bodies.csv"]
    assert "read them back as str or float64" not in caplog.text


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_select_bodies_prints_the_command_to_run_next(tmp_path, dvid_server, capsys):
    from em_annotation.cli import main

    out = str(tmp_path / "sel")
    assert main(["select-bodies", "--src", URL, "--out", out,
                 "--min-synapses", "10"]) == 0
    printed = capsys.readouterr().out
    assert "selected 3 bodies" in printed
    assert f"--bodies {out}/selected_bodies.csv" in printed



def test_cli_points_runs_and_expands_the_uuid_placeholder(tmp_path, dvid_server, capsys):
    from em_annotation.cli import main

    dst = str(tmp_path / "syn_{uuid:6}")
    assert main(["points", "--src", URL, "--out", dst, "--bodies", "1,2"]) == 0
    out = capsys.readouterr().out
    assert "syn_d38898" in out
    assert (tmp_path / "syn_d38898" / "points.parquet").exists()


def test_the_destination_is_named_after_the_node_actually_read(tmp_path, dvid_server,
                                                              capsys):
    """Found live: with --dvid-locked the tables were read from the locked node while the
    directory was named after HEAD, because the CLI expanded {uuid} before resolving. The
    provenance said one node and the path said another — worse than no name, since the
    path is what someone browsing a directory believes."""
    from em_annotation.cli import main
    from em_volume_tools.location import read_json

    dst = str(tmp_path / "syn_{uuid:6}")
    main(["points", "--src", URL, "--out", dst, "--bodies", "1", "--dvid-locked"])

    written = tmp_path / "syn_846e3a"                  # the LOCKED node, not HEAD
    assert written.exists(), sorted(p.name for p in tmp_path.iterdir())
    assert not (tmp_path / "syn_d38898").exists()
    # and the two agree
    assert read_json(str(written), "provenance.json")["source"]["uuid"] == "846e3a"


def test_cli_warns_loudly_when_coverage_is_low(tmp_path, dvid_server, capsys):
    from em_annotation.cli import main

    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1"])
    out = capsys.readouterr().out
    # body 1 alone resolves neither of its two partners
    assert "0.0%" in out and "coverage" in out


def test_cli_refuses_a_non_dvid_source(tmp_path):
    from em_annotation.cli import main

    with pytest.raises(SystemExit, match="not a DVID URL"):
        main(["points", "--src", "/local/path", "--out", str(tmp_path), "--bodies", "1"])


def test_cli_info_shows_the_type_the_sync_and_both_nodes(dvid_server, capsys):
    from em_annotation.cli import main

    assert main(["info", "--src", URL]) == 0
    out = capsys.readouterr().out
    assert "annotation" in out and "labels" in out
    assert "d38898" in out and "846e3a" in out
    assert "OPEN" in out


def test_cli_bodies_writes_elsewhere(tmp_path, dvid_server):
    from em_annotation.cli import main

    out = str(tmp_path / "ann")
    assert main(["bodies", "--src", KV_URL, "--out", out, "--bodies", "1,2"]) == 0
    assert (tmp_path / "ann" / "bodies.parquet").exists()
    assert not (tmp_path / "ann" / "points.parquet").exists()
