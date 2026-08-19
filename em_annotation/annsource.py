"""Assembling a complete precomputed annotation source from a connections table.

One LINE per distinct (T-bar, PSD) coordinate pair. Writes the four things the format
requires: an ``info``, a ``by_id`` index, one index per relationship, and one or more spatial
levels — all sharded through :mod:`em_volume_tools.sharded`, so tensorstore owns the shard
format and this module owns only what goes in it.

## Annotation ids are hashed from the geometry, not counted

The id is the ``by_id`` key and it appears in every index, so it has to be stable. Hashing
both endpoints makes it a deterministic function of the synapse: re-running against the same
node produces identical ids, and a line's id does not change because the body list grew.
Sequential ids would renumber the entire source on every run, which would make two exports
impossible to compare and would invalidate any saved link into one.

## A pair with one unknown body is still a line

Every relationship carries both coordinates — the element's own and its partner's, from the
``To`` field — so geometry is always complete. What can be missing is the partner's *body
id*, because that body was not in the fetched list. Since elements are fetched per body, each
pair has at least one known body by construction.

Such a line renders, and filters correctly on the body that *is* known: it is the boundary of
the subgraph, a real synapse on a neuron of interest whose partner was not requested.
``include_partial=False`` drops them if a strictly intra-list connectome is wanted; measured
on the 20k list, that is the difference between 1,973,448 lines and 1,138,913 (57.7%).

## The spatial index emits each annotation at exactly one level

Coarse to fine, each level takes up to ``limit`` annotations per cell from what has not been
emitted yet; the remainder falls through. A zoomed-out view then fetches one small cell and
draws a representative scattering instead of two million lines.

The subsampling *policy* is ours — the format fixes the file layout, not how a writer chooses
what to put where — so the numbers here are a starting point to check in a viewer, not
something the spec dictates.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import annotations as ann

logger = logging.getLogger(__name__)

#: Index keys, matching the reference's naming so a reader of both is not surprised.
BY_ID_KEY = "by_id"
REL_PRE, REL_POST = "body_pre", "body_post"
REL_KEYS = {REL_PRE: "by_rel_body_pre", REL_POST: "by_rel_body_post"}
SPATIAL_KEY = "by_spatial_level_{level}"

#: The properties a synapse line carries, before sorting. `conf_*` are the element
#: confidences; the `_u32` pair exists because a property cannot be uint64, so the
#: shader-visible body id is truncated and named to say so.
def default_properties(roi_labels: Sequence[str] | None = None) -> list[dict[str, Any]]:
    props: list[dict[str, Any]] = [
        {"id": "conf_pre", "type": "float32",
         "description": "confidence of the presynaptic element"},
        {"id": "conf_post", "type": "float32",
         "description": "confidence of the postsynaptic element"},
        {"id": "body_pre_u32", "type": "uint32",
         "description": "presynaptic body id, TRUNCATED to 32 bits (properties cannot be "
                        "uint64); the full id is in the body_pre relationship"},
        {"id": "body_post_u32", "type": "uint32",
         "description": "postsynaptic body id, truncated to 32 bits"},
    ]
    if roi_labels:
        props.append(ann.enum_property(
            "roi", roi_labels, dtype="int16",
            description="neuropil containing the presynaptic site"))
    return props


def annotation_ids(pre_xyz: np.ndarray, post_xyz: np.ndarray) -> np.ndarray:
    """Stable uint64 ids, hashed from both endpoints.

    blake2b over the six little-endian int32 coordinates, truncated to 63 bits — the top bit
    is left clear so no id can be confused with a sign-extended value by a consumer that
    reads it as signed. Collisions at 63 bits over a few million keys are not a practical
    concern; a duplicate would show up as a lost annotation, which :func:`build` checks for.
    """
    pre = np.asarray(pre_xyz, dtype="<i4")
    post = np.asarray(post_xyz, dtype="<i4")
    if pre.shape != post.shape or pre.ndim != 2 or pre.shape[1] != 3:
        raise ValueError(f"endpoints must both be (n, 3); got {pre.shape} and {post.shape}")
    packed = np.hstack([pre, post]).tobytes()
    stride = 24
    out = np.empty(len(pre), dtype="<u8")
    for i in range(len(pre)):
        digest = hashlib.blake2b(packed[i * stride:(i + 1) * stride], digest_size=8).digest()
        out[i] = struct.unpack("<Q", digest)[0] >> 1
    return out


def _cells(points: np.ndarray, lower: np.ndarray, chunk: np.ndarray,
           grid: np.ndarray) -> np.ndarray:
    """Which cell each point falls in, as the **key a viewer will ask for**.

    That key is the cell's compressed Morton code, not a row-major flattening — see
    :func:`em_volume_tools.sharded.compressed_morton_code`, which explains why substituting one
    for the other produces a complete file that renders almost nothing.

    Points are xyz, ``grid``/``chunk`` are xyz, and the Morton code is defined over xyz with x
    varying fastest, so no axis order changes hands here.
    """
    from em_volume_tools.sharded import compressed_morton_code

    idx = np.floor((points - lower) / chunk).astype(np.int64)
    np.clip(idx, 0, grid - 1, out=idx)
    return compressed_morton_code(idx, [int(g) for g in grid])


def next_level(grid: Sequence[int], extent: Sequence[float]) -> dict[str, Any]:
    """One level finer: halve the axis whose cells are currently longest.

    That is what keeps cells roughly cubic, so a viewer's cell fetch covers a compact region
    whichever way it is looking.
    """
    grid = np.asarray(grid, dtype=np.int64).copy()
    extent = np.asarray(extent, dtype=float)
    grid[int(np.argmax(extent / grid))] *= 2
    return {"grid_shape": [int(g) for g in grid],
            "chunk_size": [float(e / g) for e, g in zip(extent, grid)]}


def plan_spatial(n: int, lower: Sequence[float], upper: Sequence[float], *,
                 per_cell: int = 4_000, max_levels: int = 16) -> list[dict[str, Any]]:
    """Grid shapes, coarsest first — the schedule assuming annotations spread evenly.

    This is a starting guess only, because annotations do not spread evenly: synapses
    concentrate in neuropil, so a grid whose *average* cell is within ``per_cell`` still has
    dense cells far over it. :func:`_assign_levels` extends the schedule from the real
    distribution, and that is what makes each level's ``limit`` mean what it says.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    extent = upper - lower
    levels = [{"grid_shape": [1, 1, 1], "chunk_size": [float(e) for e in extent]}]
    while n / int(np.prod(levels[-1]["grid_shape"])) > per_cell and len(levels) < max_levels:
        levels.append(next_level(levels[-1]["grid_shape"], extent))
    return levels


def _assign_levels(pre_xyz: np.ndarray, levels: list[dict[str, Any]],
                   lower: np.ndarray, extent: np.ndarray, limit_per_cell: int,
                   rng: np.random.Generator, max_levels: int
                   ) -> tuple[list[dict[int, np.ndarray]], list[int]]:
    """Which annotations each level emits, coarse to fine, each exactly once.

    This is the format's own generation algorithm, and the shape of it is load-bearing:

        maxCount(level) = the largest number of REMAINING annotations in any one cell
        p(level)        = min(1, limit / maxCount(level))
        emitted(cell)   = each remaining annotation in `cell`, independently, with prob. p

    The levels partition the annotations — what a cell emits is subtracted before its children
    are computed — and the schedule continues until nothing remains, which happens exactly when
    `maxCount <= limit` makes `p` equal 1.

    **The subsampling probability is ONE number per level, applied to every cell.** Capping each
    cell at `limit` instead looks equivalent and is not: a cell holding fewer than `limit`
    annotations gets drained *completely* at that level and contributes nothing to any finer
    one, so sparse regions run out of cells partway down the pyramid. Zoomed in far enough to
    reach those levels, a viewer then finds nothing there and the annotations vanish — which is
    a rendering bug with no trace in the file. Thinning every cell by the same probability keeps
    a share of every occupied region alive all the way to the finest level.

    ``levels`` is extended in place when the initial guess was too shallow. Placement uses the
    *presynaptic* endpoint, so a line belongs to one cell rather than straddling two.
    """
    remaining = np.arange(len(pre_xyz))
    per_level: list[dict[int, np.ndarray]] = []
    limits: list[int] = []
    depth = 0
    while True:
        if depth == len(levels):
            levels.append(next_level(levels[-1]["grid_shape"], extent))
        level = levels[depth]
        grid = np.asarray(level["grid_shape"], dtype=np.int64)
        chunk = np.asarray(level["chunk_size"], dtype=float)

        which = _cells(pre_xyz[remaining], lower, chunk, grid)
        order = np.argsort(which, kind="stable")
        groups = [g for g in np.split(order, np.flatnonzero(np.diff(which[order])) + 1)
                  if len(g)]
        max_count = max(len(g) for g in groups)

        # A level that cannot be followed by another must emit everything left, or the
        # annotations would simply be dropped.
        terminal = len(levels) >= max_levels and depth == len(levels) - 1
        p = 1.0 if terminal else min(1.0, limit_per_cell / max_count)

        cells: dict[int, np.ndarray] = {}
        keep_mask = np.zeros(len(remaining), dtype=bool)
        for members in groups:
            take = members if p >= 1.0 else members[rng.random(len(members)) < p]
            if not len(take):
                continue
            cells[int(which[members[0]])] = remaining[np.sort(take)]
            keep_mask[take] = True
        remaining = remaining[~keep_mask]

        per_level.append(cells)
        # The DECLARED limit is the largest cell actually written. The generation parameter is
        # only the target: emission is independent per annotation, so a cell's count lands near
        # `limit_per_cell` rather than on it, and a viewer reading this wants the true bound.
        limits.append(max(1, max((len(v) for v in cells.values()), default=1)))
        if not len(remaining):
            return per_level, limits
        if terminal:
            raise RuntimeError(
                f"{len(remaining)} annotations were not emitted at {max_levels} levels; the "
                f"terminal level emits everything left, so this is a bug, not a setting")
        depth += 1


def build(connections, *, lower_bound: Sequence[float], upper_bound: Sequence[float],
          voxel_size_xyz: Sequence[float], roi_labels: Sequence[str] | None = None,
          include_partial: bool = True, per_cell: int = 4_000,
          max_levels: int = 16, seed: int = 0) -> dict[str, Any]:
    """Everything needed to write a source, from a ``connections`` table.

    Returns ``{"info", "by_id", "relationships", "spatial", "report"}`` where the index values
    are ``{key: [(chunk_id, payload)]}`` ready for :func:`em_volume_tools.sharded.write_all`.
    Nothing is written here, so the caller can inspect or test it first.
    """
    import pandas as pd

    from em_volume_tools import sharded

    from . import tables

    frame = connections
    total = len(frame)
    if not total:
        raise ValueError("no connections to write")

    have_pre = frame["pre_body"].notna().to_numpy()
    have_post = frame["post_body"].notna().to_numpy()
    if not include_partial:
        frame = frame[have_pre & have_post].reset_index(drop=True)
        have_pre = have_post = np.ones(len(frame), dtype=bool)
    if not len(frame):
        raise ValueError("every connection was dropped; nothing to write")

    pre_xyz = tables.positions_xyz(frame, "pre_").astype("<i4")
    post_xyz = tables.positions_xyz(frame, "post_").astype("<i4")
    ids = annotation_ids(pre_xyz, post_xyz)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError(
            f"{len(ids) - len(np.unique(ids))} annotation ids collide; ids are hashed from "
            f"the endpoint pair, so a collision means two lines share both endpoints")

    props = ann.sort_properties(default_properties(roi_labels))
    pre_body = frame["pre_body"].fillna(0).astype("uint64").to_numpy()
    post_body = frame["post_body"].fillna(0).astype("uint64").to_numpy()

    # A declared property whose column is absent must NOT be quietly zero-filled: the file
    # would be valid, the viewer would show a confidence of 0 for every synapse, and nothing
    # would say so. `connections` genuinely lacks conf — it comes from the element, not the
    # relationship — so the caller must run `tables.enrich_connections` first.
    for column, hint in (("pre_conf", "tables.enrich_connections(conns, points)"),
                         ("post_conf", "tables.enrich_connections(conns, points)")):
        if column not in frame.columns:
            raise KeyError(
                f"the connections table has no {column!r} column, so the conf property "
                f"would be written as zeros for every annotation. Add it with {hint}.")

    values: dict[str, Any] = {
        "conf_pre": frame["pre_conf"].to_numpy(dtype="float32", na_value=np.float32("nan")),
        "conf_post": frame["post_conf"].to_numpy(dtype="float32", na_value=np.float32("nan")),
        "body_pre_u32": (pre_body & 0xFFFFFFFF).astype("uint32"),
        "body_post_u32": (post_body & 0xFFFFFFFF).astype("uint32"),
    }
    if roi_labels:
        if "pre_roi_index" not in frame.columns:
            raise KeyError(
                "roi_labels were given but the table has no 'pre_roi_index' column; map the "
                "joined 'pre_roi' names to their label indices first")
        values["roi"] = frame["pre_roi_index"].fillna(0).to_numpy(dtype="int16")

    geometry = np.hstack([pre_xyz, post_xyz]).astype("<f4")

    # ---- by_id: the single encoding, which is the only one carrying relationships ----
    by_id_entries = []
    for i in range(len(frame)):
        related = [[int(pre_body[i])] if have_pre[i] else [],
                   [int(post_body[i])] if have_post[i] else []]
        by_id_entries.append((int(ids[i]), ann.encode_single(
            geometry[i], properties=props,
            values={p["id"]: values[p["id"]][i] for p in props},
            relationships=related)))

    # ---- one index per relationship, keyed by body, holding that body's lines ----
    rel_entries: dict[str, list[tuple[int, bytes]]] = {}
    for rel, bodies, mask in ((REL_PRE, pre_body, have_pre),
                              (REL_POST, post_body, have_post)):
        entries = []
        present = np.flatnonzero(mask)
        order = present[np.argsort(bodies[present], kind="stable")]
        for group in np.split(order, np.flatnonzero(np.diff(bodies[order])) + 1):
            if not len(group):
                continue
            entries.append((int(bodies[group[0]]), ann.encode_group(
                geometry[group], ids[group], properties=props,
                values={p["id"]: values[p["id"]][group] for p in props})))
        rel_entries[rel] = entries

    # ---- spatial levels ----
    lower = np.asarray(lower_bound, dtype=float)
    extent = np.asarray(upper_bound, dtype=float) - lower
    levels = plan_spatial(len(frame), lower_bound, upper_bound,
                          per_cell=per_cell, max_levels=max_levels)
    assigned, limits = _assign_levels(pre_xyz.astype(float), levels, lower, extent, per_cell,
                                      np.random.default_rng(seed), max_levels)
    del levels[len(assigned):]      # the guess may have over-provisioned; keep what was used

    spatial_entries: list[list[tuple[int, bytes]]] = []
    for level, cells in zip(levels, assigned):
        entries = []
        for cell, members in cells.items():
            entries.append((int(cell), ann.encode_group(
                geometry[members], ids[members], properties=props,
                values={p["id"]: values[p["id"]][members] for p in props})))
        spatial_entries.append(entries)

    # ---- sharding, sized to each index rather than one setting for all ----
    by_id_sharding = sharded.plan_sharding(len(by_id_entries))
    rel_sharding = {rel: sharded.plan_sharding(max(1, len(e)))
                    for rel, e in rel_entries.items()}
    spatial_sharding = [sharded.plan_sharding(max(1, len(e))) for e in spatial_entries]

    info = ann.build_info(
        lower_bound=lower_bound, upper_bound=upper_bound, voxel_size_xyz=voxel_size_xyz,
        annotation_type=ann.LINE, properties=props,
        relationships=[{"id": rel, "key": REL_KEYS[rel], "sharding": rel_sharding[rel]}
                       for rel in (REL_PRE, REL_POST)],
        by_id={"key": BY_ID_KEY, "sharding": by_id_sharding},
        spatial=[{"key": SPATIAL_KEY.format(level=i), **level,
                  "limit": limits[i], "sharding": spatial_sharding[i]}
                 for i, level in enumerate(levels)])

    report = {
        "connections": total, "lines": len(frame),
        "dropped_partial": total - len(frame) if not include_partial else 0,
        "both_bodies": int((have_pre & have_post).sum()),
        "one_body": int((have_pre ^ have_post).sum()),
        "stride": ann.record_size(ann.LINE, props),
        "by_id_shards": 2 ** by_id_sharding["shard_bits"],
        "levels": [{"grid": lvl["grid_shape"], "cells": len(e), "limit": limits[i]}
                   for i, (lvl, e) in enumerate(zip(levels, spatial_entries))],
    }
    # `table` is the frame actually encoded — which is not the input when `include_partial` is
    # false — so a caller verifying the written source compares against the right rows.
    return {"info": info, "by_id": by_id_entries, "relationships": rel_entries,
            "spatial": spatial_entries, "report": report, "table": frame}


def write(dst: str, built: Mapping[str, Any]) -> list[str]:
    """Write the ``info`` and every index. Returns the keys written."""
    from em_volume_tools import sharded
    from em_volume_tools.location import write_json

    info = built["info"]
    write_json(dst, dict(info), "info")
    written = ["info"]

    sharded.write_all(dst, info["by_id"]["sharding"], built["by_id"], BY_ID_KEY)
    written.append(f"{BY_ID_KEY}/")
    logger.info("wrote %d annotations to %s/", len(built["by_id"]), BY_ID_KEY)

    for rel in info["relationships"]:
        entries = built["relationships"][rel["id"]]
        sharded.write_all(dst, rel["sharding"], entries, rel["key"])
        written.append(f"{rel['key']}/")
        logger.info("wrote %d bodies to %s/", len(entries), rel["key"])

    for level, entries in zip(info["spatial"], built["spatial"]):
        sharded.write_all(dst, level["sharding"], entries, level["key"])
        written.append(f"{level['key']}/")
        logger.info("wrote %d cells to %s/", len(entries), level["key"])
    return written


def read_annotation(src: str, annotation_id: int, *, info: Mapping[str, Any] | None = None
                    ) -> dict[str, Any] | None:
    """One annotation out of a written source's ``by_id`` index, decoded.

    ``None`` when the id is not there. This reads the source the way a viewer does — through
    the sharded index, using the ``info`` the source itself declares — so it is the check that
    the file is addressable and not merely present.
    """
    from em_volume_tools import sharded
    from em_volume_tools.location import read_json

    if info is None:
        info = read_json(src, "info")
        if info is None:
            raise FileNotFoundError(f"no annotation info at {str(src).rstrip('/')}")
    raw = sharded.read_one(src, info["by_id"]["sharding"], annotation_id, BY_ID_KEY)
    if raw is None:
        return None
    return ann.decode_single(raw, annotation_type=info["annotation_type"],
                             properties=info["properties"],
                             n_relationships=len(info["relationships"]))


def read_related(src: str, relationship: str, body: int, *,
                 info: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """One body's annotations from a relationship index, decoded — a viewer's keyed fetch."""
    from em_volume_tools import sharded
    from em_volume_tools.location import read_json

    if info is None:
        info = read_json(src, "info")
        if info is None:
            raise FileNotFoundError(f"no annotation info at {str(src).rstrip('/')}")
    rel = next((r for r in info["relationships"] if r["id"] == relationship), None)
    if rel is None:
        raise KeyError(f"no relationship {relationship!r}; this source declares "
                       + ", ".join(repr(r["id"]) for r in info["relationships"]))
    raw = sharded.read_one(src, rel["sharding"], body, rel["key"])
    if raw is None:
        return None
    return ann.decode_group(raw, annotation_type=info["annotation_type"],
                            properties=info["properties"])


def verify(src: str, connections, *, sample: int = 200, seed: int = 0) -> dict[str, Any]:
    """Read a written source back and check it against the table it was built from.

    Samples ``sample`` rows, fetches each by id, and compares endpoints, confidences and
    relationships. Then checks one body's relationship index in full. The point is that
    everything up to :func:`write` can be verified in memory, but only a read-back proves the
    **keys** are right — a wrong key gives a viewer nothing while every byte on the store is
    correct.
    """
    from em_volume_tools.location import read_json

    from . import tables

    info = read_json(src, "info")
    if info is None:
        raise FileNotFoundError(f"no annotation info at {str(src).rstrip('/')}")

    pre_xyz = tables.positions_xyz(connections, "pre_").astype("<i4")
    post_xyz = tables.positions_xyz(connections, "post_").astype("<i4")
    ids = annotation_ids(pre_xyz, post_xyz)

    rng = np.random.default_rng(seed)
    rows = rng.choice(len(ids), size=min(sample, len(ids)), replace=False)
    problems: list[str] = []
    for row in rows:
        got = read_annotation(src, int(ids[row]), info=info)
        if got is None:
            problems.append(f"id {int(ids[row])} (row {row}) is not in by_id")
            continue
        want = np.concatenate([pre_xyz[row], post_xyz[row]]).astype("<f4")
        if not np.array_equal(got["geometry"], want):
            problems.append(f"row {row}: geometry {got['geometry']} != {want}")
        for prop, column in (("conf_pre", "pre_conf"), ("conf_post", "post_conf")):
            if column not in connections.columns:
                continue
            a, b = float(got["values"][prop]), float(connections[column].iloc[row])
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                problems.append(f"row {row}: {prop} {a} != {b}")
        for rel_index, column in ((0, "pre_body"), (1, "post_body")):
            value = connections[column].iloc[row]
            want_rel = [] if pd_isna(value) else [int(value)]
            if got["relationships"][rel_index] != want_rel:
                problems.append(f"row {row}: {column} {got['relationships'][rel_index]} "
                                f"!= {want_rel}")

    # One body's index in full: the relationship index is what "this body's synapses" uses,
    # and a count that disagrees with the table means a viewer shows the wrong set.
    bodies = connections["pre_body"].dropna()
    body_check: dict[str, Any] = {}
    if len(bodies):
        body = int(bodies.value_counts().index[0])
        expected = int((connections["pre_body"] == body).sum())
        group = read_related(src, REL_PRE, body, info=info)
        found = 0 if group is None else len(group["ids"])
        body_check = {"body": body, "expected": expected, "found": found}
        if found != expected:
            problems.append(f"body {body}: pre index holds {found} lines, table says "
                            f"{expected}")

    spatial = verify_spatial(src, info)
    problems.extend(spatial["problems"])

    return {"sampled": len(rows), "problems": problems, "body_check": body_check,
            "levels": len(info["spatial"]), "spatial": spatial}


def verify_spatial(src: str, info: Mapping[str, Any], *, per_level: int = 3,
                   seed: int = 0) -> dict[str, Any]:
    """Fetch spatial cells by the key a viewer computes, and check what comes back.

    Two independent things, both of which were silently wrong once:

    - **The key.** A cell is addressed by its compressed Morton code. Computing the key here the
      way a *reader* does — from the grid position — rather than the way the writer did is the
      whole point: a row-major key writes every object successfully and a viewer finds almost
      none of them.
    - **The contents.** Every annotation in a cell must actually lie inside that cell's bounds.
      A right key over a wrong assignment renders annotations in the wrong place, which looks
      like a coordinate bug rather than an index one.
    """
    from em_volume_tools import sharded
    from em_volume_tools.location import read_json

    if info is None:
        info = read_json(src, "info")
    lower = np.asarray(info["lower_bound"], dtype=float)
    rng = np.random.default_rng(seed)
    problems: list[str] = []
    checked = found = 0

    for depth, level in enumerate(info["spatial"]):
        grid = np.asarray(level["grid_shape"], dtype=np.int64)
        chunk = np.asarray(level["chunk_size"], dtype=float)
        # Ask about cells chosen by POSITION, so a cell the writer never emitted simply reads
        # back empty rather than being skipped — an absent cell and a mis-keyed one look the
        # same from here, which is why the count is reported rather than asserted per cell.
        cells = np.stack([rng.integers(0, int(g), size=per_level) for g in grid], axis=1)
        for position in cells:
            checked += 1
            code = sharded.compressed_morton_code([int(v) for v in position],
                                                  [int(g) for g in grid])
            raw = sharded.read_one(src, level["sharding"], code, level["key"])
            if raw is None:
                continue
            found += 1
            group = ann.decode_group(raw, annotation_type=info["annotation_type"],
                                     properties=info["properties"])
            # Placement uses the presynaptic endpoint, which is the first half of the geometry.
            pre = group["geometry"][:, :3]
            cell_lo = lower + position * chunk
            inside = np.all((pre >= cell_lo - 1e-6)
                            & (pre <= cell_lo + chunk + 1e-6), axis=1)
            if not inside.all():
                problems.append(
                    f"level {depth} cell {tuple(int(v) for v in position)}: "
                    f"{int((~inside).sum())} of {len(inside)} annotations lie outside the "
                    f"cell's own bounds")
    if not found:
        problems.append(
            f"none of {checked} sampled spatial cells could be read; the spatial index is "
            f"present but not addressable at the keys a viewer computes")
    return {"checked": checked, "found": found, "problems": problems}


def pd_isna(value: Any) -> bool:
    """``pandas.isna`` without importing pandas at module scope."""
    import pandas as pd

    return bool(pd.isna(value))


def format_report(report: Mapping[str, Any]) -> list[str]:
    lines = [f"lines: {report['lines']:,} from {report['connections']:,} connections"
             f"   stride: {report['stride']} bytes"]
    if report.get("dropped_partial"):
        lines.append(f"  dropped {report['dropped_partial']:,} with an unresolved partner")
    lines.append(f"  both bodies known: {report['both_bodies']:,}"
                 f"   one only: {report['one_body']:,}")
    lines.append(f"  by_id: {report['by_id_shards']} shards")
    for i, lvl in enumerate(report["levels"]):
        lines.append(f"  level {i}: grid {lvl['grid']}  {lvl['cells']} cells  "
                     f"limit {lvl['limit']}")
    return lines
