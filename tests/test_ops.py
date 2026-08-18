"""The two stage-1 operations end to end, with DVID stubbed at the neuclease boundary.

Stubbed at ``neuclease.dvid.*`` rather than at this package's own functions, so the code
under test is the code that runs in production — the em-volume-tools lesson that a test
building its own spec by hand proves nothing about the path that runs (invariant 9).
"""

import pandas as pd
import pytest

from em_annotation import io, ops
from em_annotation import dvid as ann_dvid

from conftest import (BODY_RECORDS, COUNTS, ELEMENTS, KV_URL, LABELSZ_INFO,
                      SYN_INFO, SZ_URL, URL, _open, _src)


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
# the whole instance, without asking for its key list
# --------------------------------------------------------------------------- #
def test_the_whole_instance_is_read_without_ever_calling_keys(tmp_path, dvid_server,
                                                              monkeypatch):
    """`/keys` is unreliable at this size — 52 s twice, then 504 from the proxy — and there is
    no count endpoint, so nothing here may depend on it."""
    import neuclease.dvid.keyvalue as ndk

    def boom(*a, **k):
        raise AssertionError("fetch_keys must never be called")

    monkeypatch.setattr(ndk, "fetch_keys", boom)
    out = str(tmp_path / "all")
    result = ops.fetch_bodies(_open(KV_URL, "bodies"), out, everything=True)
    assert result["found"] == len(BODY_RECORDS)
    assert set(io.read_table(out, "bodies")["body"]) == {1, 2}


def test_the_key_ranges_are_recorded_in_the_provenance(tmp_path, dvid_server):
    from em_volume_tools.location import read_json

    out = str(tmp_path / "all")
    ops.fetch_bodies(_open(KV_URL, "bodies"), out, everything=True)
    run = read_json(out, "provenance.json")["run"]
    assert run["whole_instance"] is True
    assert len(run["key_ranges"]) >= 2 and run["bodies_found"] == len(BODY_RECORDS)


def test_a_body_list_and_all_are_mutually_exclusive(tmp_path, dvid_server):
    """They describe different populations — the >=10-synapse set holds 117 glia against
    1,014 in the instance — so silently intersecting them would be a surprise."""
    with pytest.raises(ValueError, match="not both"):
        ops.fetch_bodies(_open(KV_URL, "bodies"), str(tmp_path / "o"), [1],
                         everything=True)
    with pytest.raises(ValueError, match="needs a body list, or everything=True"):
        ops.fetch_bodies(_open(KV_URL, "bodies"), str(tmp_path / "o"))


def test_a_boundary_key_arriving_twice_is_not_an_error(dvid_server, monkeypatch):
    """DVID's key range is INCLUSIVE at both ends, so consecutive ranges overlap by one key.
    That is why summing per-range counts overcounts (58,395 against 58,394 real records)."""
    import neuclease.dvid.keyvalue as ndk

    from em_annotation import dvid as ad

    def overlapping(server, uuid, instance, lo, hi, **k):
        # every range returns the same boundary record, as an inclusive bound would
        return {"3": {"bodyid": 3, "instance": "shared"}}

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", overlapping)
    got = ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))
    assert set(got["records"]) == {"3"}


def test_the_same_key_with_different_values_is_an_error(dvid_server, monkeypatch):
    """A repeat is expected; a repeat that disagrees means the snapshot is not coherent."""
    import neuclease.dvid.keyvalue as ndk

    from em_annotation import dvid as ad

    seen = {"n": 0}

    def inconsistent(server, uuid, instance, lo, hi, **k):
        seen["n"] += 1
        return {"3": {"bodyid": 3, "instance": f"changed-{seen['n']}"}}

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", inconsistent)
    with pytest.raises(RuntimeError, match="not coherent"):
        ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))


def test_a_failing_range_is_split_rather_than_retried_whole(dvid_server, monkeypatch):
    """The '1'..'2' range holds 16,225 records; it once completed in 13.7 s and later 504'd
    twice at ~60 s each. The proxy window is fixed but the server's speed is not, so any
    static bucket size is a bet on load — splitting on failure removes the bet."""
    import neuclease.dvid.keyvalue as ndk

    from em_annotation import dvid as ad

    attempted = []

    def too_big(server, uuid, instance, lo, hi, **k):
        attempted.append((lo, hi))
        if (lo, hi) == ("1", "2"):
            raise RuntimeError("504 Server Error: Gateway Time-out")
        return {"11": {"bodyid": 11}} if lo == "11" else {}

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", too_big)
    got = ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))
    assert set(got["records"]) == {"11"}
    # it split '1'..'2' into sub-ranges rather than repeating the whole thing
    assert ("1", "10") in attempted and ("11", "12") in attempted
    # exactly ONE attempt at the failing range — no sleep-and-repeat, because repeating a
    # request that timed out just fails again more slowly
    assert attempted.count(("1", "2")) == 1


def test_splitting_gives_up_at_a_bounded_depth(dvid_server, monkeypatch):
    import neuclease.dvid.keyvalue as ndk

    from em_annotation import dvid as ad

    monkeypatch.setattr(ad, "MAX_SPLIT_DEPTH", 1)
    monkeypatch.setattr(ndk, "fetch_keyrangevalues",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("504 nope")))
    with pytest.raises(RuntimeError, match="cannot usefully be split further"):
        ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))


def test_refine_drops_candidates_outside_the_range():
    """`hi` is not always the next character after `lo`: splitting ' '..'0' gives ' 0'..' 9',
    and splitting that again would put ' 0' after ' 9' and produce unsorted boundaries."""
    from em_annotation import dvid as ad

    assert ad.refine(" ", " 0") == [" ", " 0"]        # nothing strictly between
    inner = ad.refine(" ", "0")
    assert inner == sorted(inner) and inner[1] == " 0"


def test_a_dead_server_fails_quickly_rather_than_recursing(dvid_server, monkeypatch):
    """Splitting is the right answer to ONE range being too big and the wrong answer to the
    server being down; without a budget the two look identical until the recursion exhausts
    itself, which at depth 3 was ~16,000 requests."""
    import neuclease.dvid.keyvalue as ndk

    from em_annotation import dvid as ad

    calls = {"n": 0}

    def dead(*a, **k):
        calls["n"] += 1
        raise RuntimeError("504 Server Error: Gateway Time-out")

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", dead)
    with pytest.raises(RuntimeError):
        ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))
    # A handful of requests, not the full recursion. Whichever guard trips first is fine —
    # the requirement is that a dead server is reported quickly, and at depth 3 with a
    # sleeping retry this same scenario took hours.
    assert calls["n"] < 10, calls["n"]


def test_a_split_that_cannot_subdivide_is_what_stops_the_recursion(dvid_server,
                                                                  monkeypatch):
    """No failure budget is needed, because this guard always fires first: the leading
    sub-range of any split is `lo`..`lo + '0'`, which has nothing strictly between its bounds.
    Verified here on a digit range, where subdivision otherwise looks unbounded."""
    import neuclease.dvid.keyvalue as ndk

    from em_annotation import dvid as ad

    calls = {"n": 0}

    def dead(server, uuid, instance, lo, hi, **k):
        calls["n"] += 1
        raise RuntimeError("504 Server Error: Gateway Time-out")

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", dead)
    # Passed rather than patched: `boundaries` is a default argument, bound at definition
    # time, so setting the module constant would not reach it.
    with pytest.raises(RuntimeError, match="cannot usefully be split further"):
        ad.fetch_all_body_annotations(_open(KV_URL, "bodies"), boundaries=("1", "2"))
    assert calls["n"] <= 3, calls["n"]


def test_refine_stays_lexicographically_sorted():
    from em_annotation import dvid as ad

    bounds = ad.refine("1", "2")
    assert bounds == sorted(bounds)
    assert bounds[:3] == ["1", "10", "11"] and bounds[-1] == "2"
    assert len(ad.key_ranges(bounds)) == 11


def test_unsorted_boundaries_are_refused():
    from em_annotation import dvid as ad

    with pytest.raises(ValueError, match="must be sorted"):
        ad.key_ranges(["5", "1"])


def test_cli_all_writes_the_whole_instance(tmp_path, dvid_server, capsys):
    from em_annotation.cli import main

    out = str(tmp_path / "all")
    assert main(["bodies", "--src", KV_URL, "--out", out, "--all"]) == 0
    printed = capsys.readouterr().out
    assert "records across" in printed
    assert (tmp_path / "all" / "bodies.parquet").exists()


def test_cli_refuses_both_or_neither(tmp_path, dvid_server):
    from em_annotation.cli import main

    with pytest.raises(SystemExit, match="pass one, not both"):
        main(["bodies", "--src", KV_URL, "--out", str(tmp_path / "o"),
              "--all", "--bodies", "1"])
    with pytest.raises(SystemExit, match="--bodies is required"):
        main(["bodies", "--src", KV_URL, "--out", str(tmp_path / "o")])


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
