"""The two stage-1 operations, as functions the CLI is a thin shell over.

Both take an **already-resolved source** — the dict :func:`open_source` returns, carrying a
concrete uuid rather than a ref. The caller resolves, once, and everything downstream
(including the destination name) derives from that one answer.

That split is not stylistic. When these functions resolved the ref themselves, the CLI had
to expand ``--dst {uuid:8}`` *before* calling them, so with ``--dvid-locked`` the tables
were correctly read from the locked node while the directory was named after HEAD. The
provenance said one node and the path said another, which is worse than having no name at
all: the name is what someone browsing a directory believes. Resolving in the caller makes
the ordering impossible to get wrong, and is the same discipline as em-volume-tools'
invariant 9.

Fetch for an explicit body list, write tables, write provenance **after** the tables.
Neither uses dask or a manifest — a fetch of 20k bodies is ~16 minutes in one process, and
a resumable distributed run would be more machinery than the job needs.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from . import dvid as _dvid
from . import io as _io

logger = logging.getLogger(__name__)

#: Table names written by `fetch_points`, and what each holds.
POINT_TABLES = ("points", "relationships")
BODY_TABLES = ("bodies",)


def open_points_source(src: Mapping[str, Any], *,
                       prefer_locked: bool = False) -> dict[str, Any]:
    """Resolve and validate a point-annotation source. Call this before anything else."""
    source = _dvid.open_source(src, expect=_dvid.POINT_INSTANCE,
                               prefer_locked=prefer_locked)
    _dvid.require_sync(source)
    return source


def open_bodies_source(src: Mapping[str, Any], *,
                       prefer_locked: bool = False) -> dict[str, Any]:
    """Resolve and validate a body-annotation (keyvalue) source."""
    return _dvid.open_source(src, expect=_dvid.BODY_INSTANCE,
                             prefer_locked=prefer_locked)


def open_counts_source(src: Mapping[str, Any], *,
                       prefer_locked: bool = False) -> dict[str, Any]:
    """Resolve a synapse-count source: a ``labelsz`` index, or the instance it indexes."""
    source = _dvid.open_source(
        src, expect=(_dvid.COUNT_INSTANCE, _dvid.POINT_INSTANCE),
        prefer_locked=prefer_locked)
    return _dvid.resolve_labelsz(source)


#: Written by `select_bodies`. A single file, and the name is stable so the next command
#: can be pointed at `<out>/selected_bodies.csv` without thinking.
SELECTED_TABLE = "selected_bodies"


def select_bodies(source: Mapping[str, Any], dst: str, *, min_synapses: int = 10,
                  min_pre: int = 0, min_post: int = 0, limit: int | None = None,
                  fmt: str = "csv") -> dict[str, Any]:
    """Choose which bodies are worth fetching annotations for, and write the list.

    Uses DVID's own ranked ``labelsz`` index, which costs seconds — against ~70 minutes to
    scan all ~80M label sizes, and unlike a size ranking it selects on what actually makes a
    body interesting in a connectome. A body with no synapses is a fragment.

    **The default threshold is on the TOTAL**, not on pre and post separately, and that is a
    domain constraint rather than a simplification: sensory neurons may have no postsynapses
    at all, and a neuron projecting outside the traced volume may have no presynapses.
    Requiring both would silently drop exactly the cells most worth looking at.
    ``min_pre`` / ``min_post`` exist for when that is what you want.

    Output goes to a directory so the list carries a provenance record naming the node it
    was computed from — a body list whose node nobody can identify is the thing that goes
    quietly stale, since proofreading changes body ids.
    """
    df = _dvid.fetch_synapse_counts(source, min_total=min_synapses, min_pre=min_pre,
                                    min_post=min_post, limit=limit)
    run = {"min_synapses": min_synapses, "min_pre": min_pre or None,
           "min_post": min_post or None, "limit": limit,
           "bodies_selected": int(len(df)),
           "indexes": source.get("indexes"),
           "synapses_total": int(df["syn"].sum()) if len(df) else 0}
    record = _record(source, dst, run=run)
    written = [_io.write_table(df, dst, SELECTED_TABLE, fmt=fmt, metadata=record)]
    _io.write_provenance(dst, record)
    return {"bodies": df, "written": written, "record": record}


def _record(source: Mapping[str, Any], dst: str, *, run: Mapping[str, Any]) -> dict:
    from em_volume_tools.ops.provenance import build_record

    record = build_record(src_spec={"backend": "dvid", **dict(source)}, dst=dst, **run)
    record["tool"] = "em-annotation"
    record["source"] = _dvid.node_record(source)
    return record


def fetch_points(source: Mapping[str, Any], dst: str, bodies: Sequence[int], *,
                 threads: int = _dvid.DEFAULT_THREADS, fmt: str = "parquet",
                 drop_unmatched: bool = False,
                 write_connections: bool = False) -> dict[str, Any]:
    """Point annotations for ``bodies`` -> ``points`` + ``relationships`` tables.

    ``source`` comes from :func:`open_points_source`; it carries a concrete uuid.

    ``drop_unmatched`` removes relationships whose partner body could not be resolved.
    **Off by default, deliberately.** An unresolved partner and a genuinely dangling
    reference are indistinguishable from the fetched data alone, so dropping silently
    turns "the body list did not cover this partner" into "this synapse does not exist" —
    and at low coverage that presents a mostly-incomplete connectome as a complete one.
    Keep the nulls, read the match rate, then drop once the rate says it is safe.
    """
    synced = _dvid.require_sync(source)
    logger.info("%s: %d bodies, %d threads (bodies from sync: %s)",
                source["instance"], len(bodies), threads, ", ".join(synced))

    result = _dvid.fetch_points(source, bodies, threads=threads)
    points, rels, match = result["points"], result["relationships"], result["match"]

    if drop_unmatched:
        before = len(rels)
        rels = rels[rels["to_body"].notna()].reset_index(drop=True)
        logger.info("dropped %d relationships with an unresolved partner",
                    before - len(rels))

    run = {"bodies_requested": len(bodies), "threads": threads,
           "elements": int(len(points)), "relationships": int(len(rels)),
           "match": match, "synced_to": synced,
           "drop_unmatched": drop_unmatched or None,
           "failures": result["failures"] or None}
    record = _record(source, dst, run=run)

    written = [_io.write_table(points, dst, "points", fmt=fmt, metadata=record),
               _io.write_table(rels, dst, "relationships", fmt=fmt, metadata=record)]
    if write_connections:
        written.append(_io.write_table(result["connections"], dst, "connections",
                                       fmt=fmt, metadata=record))
    _io.write_provenance(dst, record)

    return {**result, "relationships": rels, "written": written, "record": record}


def fetch_bodies(source: Mapping[str, Any], dst: str, bodies: Sequence[int], *,
                 fmt: str = "parquet") -> dict[str, Any]:
    """Body annotations for ``bodies`` -> a ``bodies`` table.

    ``source`` comes from :func:`open_bodies_source`; it carries a concrete uuid.

    A separate destination from the point tables on purpose: they come from a different
    instance, they are a different grain (one row per body, not per element), and nothing
    joins them at write time.
    """
    result = _dvid.fetch_body_annotations(source, bodies)
    frame = result["bodies"]

    run = {"bodies_requested": result["requested"], "bodies_found": result["found"],
           "bodies_missing": len(result["missing"]) or None}
    record = _record(source, dst, run=run)
    written = [_io.write_table(frame, dst, "bodies", fmt=fmt, metadata=record)]
    _io.write_provenance(dst, record)
    return {**result, "written": written, "record": record}
