"""Assembling a complete LINE annotation source from a connections table."""

import itertools
import struct

import numpy as np
import pandas as pd
import pytest

from neu_mark import annotations as ann
from neu_mark import annsource
from neu_vol.location import read_json

BOUNDS = ([0, 0, 0], [600, 800, 1000])       # xyz


def connections(n=200, *, partial=0, seed=0):
    """A frame shaped like `tables.connections()` output, with `partial` unresolved posts."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "pre_z": rng.integers(0, 1000, n), "pre_y": rng.integers(0, 800, n),
        "pre_x": rng.integers(0, 600, n),
        "post_z": rng.integers(0, 1000, n), "post_y": rng.integers(0, 800, n),
        "post_x": rng.integers(0, 600, n),
        "pre_body": pd.Series(rng.integers(1, 40, n), dtype="UInt64"),
        "post_body": pd.Series(rng.integers(1, 40, n), dtype="UInt64"),
        # as `tables.enrich_connections` produces them: joined from the elements
        "pre_conf": rng.random(n).astype("float32"),
        "post_conf": rng.random(n).astype("float32"),
    })
    if partial:
        df.loc[df.index[:partial], "post_body"] = pd.NA
    return df


def _built(**kw):
    kw.setdefault("per_cell", 50)
    return annsource.build(connections(**{k: kw.pop(k) for k in ("n", "partial", "seed")
                                          if k in kw}),
                           lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                           voxel_size_xyz=[8, 8, 8], **kw)


# --------------------------------------------------------------------------- #
# ids
# --------------------------------------------------------------------------- #
def test_ids_are_stable_across_runs():
    """The id is the by_id key and appears in every index, so counting would renumber the
    whole source on every export and invalidate any saved link into it."""
    df = connections(50)
    a = annsource.build(df, lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                        voxel_size_xyz=[8, 8, 8], per_cell=50)
    b = annsource.build(df, lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                        voxel_size_xyz=[8, 8, 8], per_cell=50)
    assert [i for i, _ in a["by_id"]] == [i for i, _ in b["by_id"]]


def test_ids_depend_only_on_the_endpoints():
    """So a line keeps its id when the body list grows and its partner becomes known."""
    pre = np.array([[1, 2, 3], [4, 5, 6]], dtype="<i4")
    post = np.array([[7, 8, 9], [10, 11, 12]], dtype="<i4")
    first = annsource.annotation_ids(pre, post)
    assert np.array_equal(first, annsource.annotation_ids(pre, post))
    # swapping the endpoints is a different synapse and must not collide
    assert not set(first) & set(annsource.annotation_ids(post, pre))


def test_ids_leave_the_top_bit_clear():
    """So a consumer reading them as signed cannot see a negative id."""
    ids = annsource.annotation_ids(np.arange(300, dtype="<i4").reshape(100, 3),
                                   np.arange(300, dtype="<i4").reshape(100, 3) + 7)
    assert (ids < 2 ** 63).all()


def test_mismatched_endpoint_arrays_are_refused():
    with pytest.raises(ValueError, match=r"both be \(n, 3\)"):
        annsource.annotation_ids(np.zeros((3, 3), "<i4"), np.zeros((2, 3), "<i4"))


# --------------------------------------------------------------------------- #
# half-resolved lines
# --------------------------------------------------------------------------- #
def test_a_line_with_one_unknown_body_is_still_written():
    """Geometry is always complete — the partner's coordinate comes from the relationship's
    `To` field — so what is missing is only the partner's body id."""
    built = _built(n=100, partial=30)
    assert built["report"]["lines"] == 100
    assert built["report"]["one_body"] == 30


def test_such_a_line_appears_in_only_the_known_relationship():
    built = _built(n=100, partial=30)
    ids_pre = {i for entries in [built["relationships"]["body_pre"]] for i, _ in entries}
    # every body that owns a line appears as a key; the count of lines in the post index is
    # short by exactly the unresolved ones
    def total(entries):
        return sum(struct.unpack("<Q", p[:8])[0] for _i, p in entries)
    assert total(built["relationships"]["body_pre"]) == 100
    assert total(built["relationships"]["body_post"]) == 70


def test_its_by_id_record_declares_an_empty_relationship():
    built = _built(n=10, partial=10)
    props = built["info"]["properties"]
    stride = ann.record_size(ann.LINE, props)
    _id, payload = built["by_id"][0]
    n_pre = struct.unpack("<I", payload[stride:stride + 4])[0]
    off = stride + 4 + 8 * n_pre
    n_post = struct.unpack("<I", payload[off:off + 4])[0]
    assert n_pre == 1 and n_post == 0
    assert off + 4 == len(payload)


def test_include_partial_false_drops_them():
    built = _built(n=100, partial=30, include_partial=False)
    assert built["report"]["lines"] == 70
    assert built["report"]["dropped_partial"] == 30
    assert built["report"]["one_body"] == 0


def test_dropping_everything_is_an_error():
    df = connections(20, partial=20)
    with pytest.raises(ValueError, match="every connection was dropped"):
        annsource.build(df, lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                        voxel_size_xyz=[8, 8, 8], include_partial=False)


def test_an_empty_table_is_refused():
    with pytest.raises(ValueError, match="no connections"):
        annsource.build(connections(0), lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                        voxel_size_xyz=[8, 8, 8])


# --------------------------------------------------------------------------- #
# the spatial index
# --------------------------------------------------------------------------- #
def test_every_annotation_is_emitted_exactly_once_across_levels():
    """The levels partition the annotations: a zoomed-out view draws a subsample and finer
    levels supply the rest, so a duplicate would double-draw and a miss would lose data."""
    built = _built(n=300)
    counts = [struct.unpack("<Q", p[:8])[0]
              for level in built["spatial"] for _c, p in level]
    assert sum(counts) == 300

    seen = []
    for level in built["spatial"]:
        for _cell, payload in level:
            n = struct.unpack("<Q", payload[:8])[0]
            stride = ann.record_size(ann.LINE, built["info"]["properties"])
            seen.extend(np.frombuffer(payload[8 + n * stride:], "<u8"))
    assert len(seen) == len(set(seen)) == 300


def test_no_level_exceeds_its_declared_limit():
    built = _built(n=400, per_cell=25)
    for level, entries in zip(built["info"]["spatial"], built["spatial"]):
        for _cell, payload in entries:
            assert struct.unpack("<Q", payload[:8])[0] <= level["limit"]


def test_no_level_is_declared_empty():
    """A level with no cells claims detail the source does not have. The schedule stops when
    nothing overflows, so this holds by construction — pinned because the earlier version
    planned the depth up front and could over-provision."""
    built = _built(n=60, per_cell=50)
    assert all(len(e) for e in built["spatial"])
    assert len(built["info"]["spatial"]) == len(built["spatial"])


def test_every_level_stays_near_per_cell():
    """`per_cell` bounds what a viewer downloads for one cell. It is a target rather than a hard
    cap, because emission is independent per annotation, so a cell lands NEAR it — but the
    finest level must not balloon, which is what a fixed-in-advance schedule caused."""
    built = _built(n=2_000, per_cell=100)
    limits = [level["limit"] for level in built["info"]["spatial"]]
    assert max(limits) < 200, limits


def test_the_declared_limit_is_the_largest_cell_actually_written():
    """A viewer reads `limit` as the bound on one cell's download, so it has to describe the
    file rather than restate the setting that generated it."""
    built = _built(n=800, per_cell=60)
    for level, entries in zip(built["info"]["spatial"], built["spatial"]):
        counts = [struct.unpack("<Q", p[:8])[0] for _c, p in entries]
        assert level["limit"] == max(counts)


def test_sparse_cells_are_not_drained_completely_at_a_coarse_level():
    """The bug this pins produced a file that was valid and rendered NOTHING when zoomed in.

    Capping each cell at `limit` fully empties any cell holding fewer than `limit` annotations,
    so sparse regions stop having cells partway down the pyramid — the occupied-cell count peaks
    and then FALLS toward the finest level. Thinning every cell by one per-level probability
    keeps a share of every region alive, so the count rises monotonically as cells subdivide.
    """
    built = _built(n=4_000, per_cell=50)
    cells = [len(e) for e in built["spatial"]]
    assert len(cells) > 3
    assert cells == sorted(cells), cells
    assert cells[-1] == max(cells), cells


def test_the_depth_used_may_be_shallower_than_the_uniform_guess():
    """Each level drains up to `per_cell` from *every* cell it touches, so the annotations run
    out sooner than the average-occupancy estimate suggests. The unused tail is dropped rather
    than declared empty."""
    built = _built(n=2_000, per_cell=100)
    guess = annsource.plan_spatial(2_000, *BOUNDS, per_cell=100)
    assert len(built["info"]["spatial"]) < len(guess)
    assert all(len(e) for e in built["spatial"])


def test_clustered_annotations_deepen_the_schedule_further_than_uniform_ones():
    """Clustering is exactly what the uniform estimate gets wrong — the average cell is within
    `per_cell` while dense cells are far over it — so it is the case the data-driven extension
    has to handle, and where the depth exceeds the guess."""
    n = 2_000
    spread = connections(n, seed=1)
    clustered = connections(n, seed=1)
    for axis in ("x", "y", "z"):
        clustered[f"pre_{axis}"] = clustered[f"pre_{axis}"] // 8

    def depth(df):
        return len(annsource.build(df, lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                                   voxel_size_xyz=[8, 8, 8],
                                   per_cell=100)["info"]["spatial"])

    assert depth(clustered) > depth(spread)
    assert depth(clustered) > len(annsource.plan_spatial(n, *BOUNDS, per_cell=100))


def test_the_terminal_level_takes_its_cells_whole_at_the_cap():
    """With the depth capped the last level cannot subsample — the annotations left have
    nowhere else to go, so it must exceed `per_cell` rather than silently drop them."""
    built = _built(n=2_000, per_cell=10, max_levels=3)
    assert len(built["info"]["spatial"]) == 3
    counts = [struct.unpack("<Q", p[:8])[0] for _c, p in built["spatial"][-1]]
    assert max(counts) > 10
    assert built["info"]["spatial"][-1]["limit"] == max(counts)
    total = sum(struct.unpack("<Q", p[:8])[0]
                for level in built["spatial"] for _c, p in level)
    assert total == 2_000        # nothing lost to the cap


def test_the_coarsest_level_is_a_single_cell():
    built = _built(n=300)
    assert built["info"]["spatial"][0]["grid_shape"] == [1, 1, 1]


def test_levels_halve_the_largest_axis_so_cells_get_more_cubic():
    """Level 0 is a single cell spanning the volume, so its aspect ratio is the volume's and
    no subdivision can change that. What the halving buys is that deeper levels approach
    cubic, which is what keeps a cell's contents spatially coherent."""
    levels = annsource.plan_spatial(10_000, [0, 0, 0], [100, 200, 400], per_cell=100)
    grids = [tuple(l["grid_shape"]) for l in levels]
    assert grids[0] == (1, 1, 1)
    assert grids[1] == (1, 1, 2)          # z is largest, so it is divided first

    def aspect(level):
        chunk = level["chunk_size"]
        return max(chunk) / min(chunk)

    assert aspect(levels[0]) == 4.0        # the volume's own shape
    assert aspect(levels[-1]) < aspect(levels[0])
    assert aspect(levels[-1]) <= 2.0 + 1e-9


def test_level_count_is_capped():
    levels = annsource.plan_spatial(10_000_000, [0, 0, 0], [1000, 1000, 1000],
                                    per_cell=10, max_levels=4)
    assert len(levels) == 4


# --------------------------------------------------------------------------- #
# info and properties
# --------------------------------------------------------------------------- #
def test_properties_are_the_documented_set_in_size_order():
    props = _built(n=20)["info"]["properties"]
    assert [p["id"] for p in props] == ["conf_pre", "conf_post",
                                        "body_pre_u32", "body_post_u32"]


def test_a_roi_enum_is_added_when_labels_are_given():
    df = connections(20)
    df["pre_roi_index"] = [i % 3 for i in range(20)]
    built = annsource.build(df, lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                            voxel_size_xyz=[8, 8, 8], per_cell=50,
                            roi_labels=["<unspecified>", "ME(L)", "LO(R)"])
    roi = [p for p in built["info"]["properties"] if p["id"] == "roi"][0]
    assert roi["type"] == "int16" and roi["enum_labels"][0] == "<unspecified>"
    # int16 sorts after the 4-byte properties
    assert [p["id"] for p in built["info"]["properties"]][-1] == "roi"


def test_roi_labels_without_the_index_column_are_refused():
    with pytest.raises(KeyError, match="pre_roi_index"):
        annsource.build(connections(5), lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                        voxel_size_xyz=[8, 8, 8], roi_labels=["<unspecified>", "ME(L)"])


def test_a_missing_conf_column_is_refused_rather_than_zero_filled():
    """A valid file showing 0 confidence for every synapse, with nothing to say so, is the
    silent failure this guard exists for — `connections` genuinely lacks conf, because it
    comes from the element rather than the relationship."""
    df = connections(10).drop(columns=["post_conf"])
    with pytest.raises(KeyError, match="post_conf"):
        annsource.build(df, lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                        voxel_size_xyz=[8, 8, 8])


def test_an_unknown_partner_confidence_stays_nan_not_zero():
    """Zero would claim the annotator had no confidence in a real synapse; NaN says unknown,
    and it fails every shader comparison so such a line stays visible under a threshold."""
    df = connections(4)
    df.loc[df.index[:2], "post_conf"] = np.nan
    built = annsource.build(df, lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                            voxel_size_xyz=[8, 8, 8], per_cell=50)
    _id, payload = built["by_id"][0]
    conf_post = struct.unpack("<f", payload[28:32])[0]
    assert np.isnan(conf_post)


def test_body_ids_are_truncated_to_32_bits_in_the_property_and_full_in_the_relationship():
    """A property cannot be uint64, so the shader-visible value is truncated — which is why
    it is named _u32 — while the relationship key keeps the whole id."""
    df = connections(1)
    wide = 2**40 + 12345
    df.loc[0, "pre_body"] = wide
    built = annsource.build(df, lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                            voxel_size_xyz=[8, 8, 8], per_cell=10)
    props = built["info"]["properties"]
    _id, payload = built["by_id"][0]
    offset = 24 + 4 + 4                      # geometry + conf_pre + conf_post
    assert struct.unpack("<I", payload[offset:offset + 4])[0] == wide & 0xFFFFFFFF
    assert [i for i, _ in built["relationships"]["body_pre"]] == [wide]


def test_each_index_is_sharded_for_its_own_size():
    built = _built(n=400, per_cell=25)
    assert built["info"]["by_id"]["sharding"]["@type"] == "neuroglancer_uint64_sharded_v1"
    for rel in built["info"]["relationships"]:
        assert "sharding" in rel
    for level in built["info"]["spatial"]:
        assert "sharding" in level


# --------------------------------------------------------------------------- #
# writing, and reading back through the sharded reader
# --------------------------------------------------------------------------- #
def test_a_written_source_reads_back_by_id(tmp_path):
    from neu_vol import sharded

    built = _built(n=150, partial=20)
    dst = str(tmp_path / "syn")
    written = annsource.write(dst, built)
    assert "info" in written and f"{annsource.BY_ID_KEY}/" in written

    ann_id, expected = built["by_id"][7]
    got = sharded.read_one(dst, built["info"]["by_id"]["sharding"], ann_id,
                           annsource.BY_ID_KEY)
    assert got == expected


def test_a_written_relationship_index_reads_back_by_body(tmp_path):
    from neu_vol import sharded

    built = _built(n=150)
    dst = str(tmp_path / "syn")
    annsource.write(dst, built)
    rel = [r for r in built["info"]["relationships"] if r["id"] == "body_pre"][0]
    body, expected = built["relationships"]["body_pre"][0]
    assert sharded.read_one(dst, rel["sharding"], body, rel["key"]) == expected


def test_the_info_lands_where_a_viewer_looks(tmp_path):
    from neu_vol.location import read_json

    built = _built(n=50)
    dst = str(tmp_path / "syn")
    annsource.write(dst, built)
    info = read_json(dst, "info")
    assert info["@type"] == "neuroglancer_annotations_v1"
    assert info["annotation_type"] == "line"


# --------------------------------------------------------------------------- #
# read-back against the source table: the only end-to-end evidence
# --------------------------------------------------------------------------- #
def _write(tmp_path, df, **kw):
    dst = str(tmp_path / "syn")
    annsource.write(dst, annsource.build(df, lower_bound=BOUNDS[0], upper_bound=BOUNDS[1],
                                         voxel_size_xyz=[8, 8, 8], **kw))
    return dst


def test_verify_finds_nothing_wrong_with_a_source_it_just_wrote(tmp_path):
    """Everything before `write` can be checked in memory; only a read-back proves the KEYS
    are right, and a wrong key gives a viewer nothing while every byte on the store is
    correct."""
    df = connections(150, partial=20)
    result = annsource.verify(_write(tmp_path, df, per_cell=50), df, sample=150)
    assert result["problems"] == []
    assert result["sampled"] == 150
    assert result["body_check"]["found"] == result["body_check"]["expected"] > 0


def test_verify_resolves_each_annotation_through_the_declared_sharding(tmp_path):
    """`read_annotation` goes through the info's own sharding spec rather than the one the
    build happened to use, which is what a viewer does."""
    df = connections(80)
    dst = _write(tmp_path, df, per_cell=30)
    pre = tables_positions(df, "pre_")
    ids = annsource.annotation_ids(pre, tables_positions(df, "post_"))
    got = annsource.read_annotation(dst, int(ids[5]))
    assert list(got["geometry"][:3]) == list(pre[5].astype(float))
    assert annsource.read_annotation(dst, 12345) is None      # absent, not an error


def test_a_spatial_cell_reads_back_at_the_key_a_viewer_computes(tmp_path):
    """The bug this pins wrote every object successfully and rendered almost nothing.

    A sharded spatial index keys a cell by its compressed Morton code. A row-major flattening
    agrees on any grid that subdivides at most one axis — so it survives small fixtures — and
    then disagrees on nearly every cell of a real multi-axis grid. The key here is recomputed
    from the grid POSITION, the way a reader does it, never from the writer's bookkeeping.
    """
    from neu_vol import sharded

    df = connections(1_500, seed=3)
    dst = _write(tmp_path, df, per_cell=40)
    info = read_json(dst, "info")
    lower = np.asarray(info["lower_bound"], dtype=float)

    # a level whose grid subdivides more than one axis is where the two encodings diverge
    depth, level = next((i, lvl) for i, lvl in enumerate(info["spatial"])
                        if sum(1 for g in lvl["grid_shape"] if g > 1) > 1)
    grid, chunk = level["grid_shape"], np.asarray(level["chunk_size"], dtype=float)

    found = 0
    for position in itertools.product(*(range(g) for g in grid)):
        code = sharded.compressed_morton_code(list(position), grid)
        raw = sharded.read_one(dst, level["sharding"], code, level["key"])
        if raw is None:
            continue
        found += 1
        group = ann.decode_group(raw, properties=info["properties"])
        pre = group["geometry"][:, :3]
        cell_lo = lower + np.asarray(position, dtype=float) * chunk
        assert np.all(pre >= cell_lo - 1e-6) and np.all(pre <= cell_lo + chunk + 1e-6), (
            f"level {depth} cell {position} holds annotations outside its own bounds")
    assert found, f"no cell of level {depth} (grid {grid}) was readable by Morton key"


def test_verify_checks_the_spatial_index_and_not_only_by_id(tmp_path):
    """`verify` passed a source whose spatial index was unreachable, because it only ever
    fetched by annotation id. A check that cannot see the index it is verifying is not one."""
    df = connections(1_500, seed=3)
    result = annsource.verify(_write(tmp_path, df, per_cell=40), df, sample=50)
    assert result["problems"] == []
    assert result["spatial"]["found"] > 0
    assert result["spatial"]["checked"] > result["levels"]


def test_verify_reports_a_confidence_that_does_not_match(tmp_path):
    """The check has to be able to fail, or a clean report says nothing."""
    df = connections(60)
    dst = _write(tmp_path, df, per_cell=30)
    tampered = df.copy()
    tampered.loc[tampered.index, "pre_conf"] = 0.0
    result = annsource.verify(dst, tampered, sample=60)
    assert result["problems"] and any("conf_pre" in p for p in result["problems"])


def test_verify_reports_a_relationship_that_does_not_match(tmp_path):
    df = connections(60, partial=10)
    dst = _write(tmp_path, df, per_cell=30)
    tampered = df.copy()
    tampered["post_body"] = pd.Series([999] * len(df), dtype="UInt64")
    result = annsource.verify(dst, tampered, sample=60)
    assert any("post_body" in p for p in result["problems"])


def tables_positions(df, prefix):
    from neu_mark import tables as _t

    return _t.positions_xyz(df, prefix).astype("<i4")
