"""Fetching annotations out of DVID, for a given set of bodies.

Addressing and version resolution are em-volume-tools' (``em_volume_tools.dvid``): a
``dvid://server/uuid/instance`` URL, a ref resolved **once** to a concrete node, and the
provenance record that names it. This module adds the two reads that matter here.

## Point annotations

``/label/<body>`` is the only endpoint that answers "which elements are in this body", and
it **exists only when the annotation instance is synced to a labelmap**. That is a property
of the instance, so it is checked up front — an unsynced instance answers with an error
per body rather than an empty list, which would otherwise surface as thousands of failures.

## Why the concurrency is ours

``neuclease.dvid.annotation.fetch_elements_for_bodies`` looks like exactly this function
and cannot be used. It builds ``partial(fetch_label, server, uuid, instance)`` and maps it
over bodies, so ``relationships`` keeps its default of **False** — it can never return
relationship data, which is half of what a synapse table is. Beyond that: its
``processes=0`` default is serial, it exposes only ``processes`` (multiprocessing, for a
purely I/O-bound job, pickling a ``requests.Session`` into each child), it concatenates
every body's frame at the end, and it has no retry and no per-body error isolation, so one
failed HTTP request loses the batch.

So: a thread pool, because this is I/O bound; ``with_retry`` per body, reusing
em-volume-tools' classifier, which already knows the ``requests``-flavoured transient
markers DVID produces; and failures collected per body rather than raised, because a run
over thousands of bodies should report the three that failed and keep the rest.

**Thread count is deliberately modest.** DVID is a shared service and answers overload
with 503. Measured against dvid.example.org: 400 bodies in 19.7 s at 8 threads, so ~16 min for
20k bodies — fast enough that pointing a dask fleet at a shared server buys nothing worth
the risk.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

# Imported as a MODULE, and every call below goes through it. A `from ... import name`
# creates a second binding, so patching `em_volume_tools.dvid.instance_info` would leave
# this module calling the real one — a test that stubs DVID would silently start needing
# the network. One module attribute is one patch point; the same rule holds in
# em-volume-tools' own backends/dvid.py, for the same reason.
from em_volume_tools import dvid as _vdvid
from em_volume_tools.dvid import MISSING
from em_volume_tools.retry import with_retry

from . import tables

logger = logging.getLogger(__name__)

#: Default pool size. See the module docstring — this is a shared-service courtesy limit,
#: not a throughput ceiling we measured against.
DEFAULT_THREADS = 8

#: DVID instance types this module can read, by what they hold.
POINT_INSTANCE = "annotation"
BODY_INSTANCE = "keyvalue"


def open_source(spec: Mapping[str, Any], *, expect: str,
                prefer_locked: bool = False) -> dict[str, Any]:
    """Resolve the ref, check the instance type, and return the pinned spec.

    The returned spec carries the **concrete uuid**, never the ref. Same discipline as
    em-volume-tools' invariant 9 and for the same reason: a nightly lock-and-spawn landing
    mid-run would otherwise move HEAD, and a table half from one node and half from the
    next is not a snapshot of anything.
    """
    node = _vdvid.resolve_node(spec, prefer_locked=prefer_locked)
    pinned = {**dict(spec), "uuid": node["uuid"],
              "requested_ref": spec.get("requested_ref", node["ref"]),
              "ancestors_walked": node.get("walked", 0)}
    info = _vdvid.instance_info(pinned)
    _vdvid.check_instance_type(info, pinned, expect)
    if not node["locked"]:
        logger.warning(
            "%s is an OPEN node, so it is still being written to and this pull is not "
            "reproducible. Pass --dvid-locked for the newest locked node.",
            node["uuid"])
    return {**pinned, "node": node, "instance_info": info}


def require_sync(source: Mapping[str, Any]) -> list[str]:
    """The labelmap(s) an annotation instance is synced to, or raise explaining why not.

    ``/label/<body>`` is built from the sync's own index; without it DVID cannot answer
    per-body at all. Checked once here rather than discovered as a per-body failure.
    """
    synced = _vdvid.synced_instances(source.get("instance_info") or {})
    if not synced:
        raise ValueError(
            f"{_vdvid.spec_url(source)} is not synced to a labelmap, so DVID cannot say "
            f"which elements belong to a body — the /label endpoint that every fetch "
            f"here uses does not exist for an unsynced instance. Sync it in DVID (its "
            f"labelsz/annotation sync), or fetch by region instead of by body.")
    return synced


def _fetch_one(source: Mapping[str, Any], body: int) -> list[dict]:
    try:
        from neuclease.dvid.annotation import fetch_label
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, instance = _vdvid.address(source)

    def go():
        return fetch_label(server, uuid, instance, body, relationships=True,
                           format="list")

    # The whole body is the retry unit: the request is idempotent, so a transient failure
    # costs one repeat and nothing is half-written.
    return with_retry(go, label=f"annotation body {body}") or []


def fetch_points(source: Mapping[str, Any], bodies: Sequence[int], *,
                 threads: int = DEFAULT_THREADS,
                 progress_every: int = 500) -> dict[str, Any]:
    """Fetch every element of every body, and join relationships to bodies.

    Returns ``{"points", "relationships", "connections", "match", "failures",
    "elements"}``. ``failures`` is ``{body: message}`` — a body that could not be fetched
    after retries is recorded and the run continues, because the alternative is losing
    thousands of successful fetches to one bad request.
    """
    results: list[tuple] = []
    failures: dict[int, str] = {}
    done = 0

    def work(body: int):
        return body, _fetch_one(source, body)

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        for body, elements in pool.map(work, bodies):
            done += 1
            try:
                results.append(tables.elements_to_frames(elements, body))
            except Exception as exc:                            # noqa: BLE001
                failures[body] = f"{type(exc).__name__}: {exc}"
            if progress_every and done % progress_every == 0:
                logger.info("fetched %d/%d bodies", done, len(bodies))

    points, rels = tables.combine(results)
    conns = tables.connections(rels)
    return {"points": points, "relationships": rels, "connections": conns,
            "match": tables.match_rate(conns), "failures": failures,
            "elements": int(len(points))}


def fetch_body_annotations(source: Mapping[str, Any], bodies: Sequence[int], *,
                           batch_size: int = 10_000) -> dict[str, Any]:
    """Fetch the per-body records from a keyvalue instance.

    Uses ``fetch_keyvalues`` directly rather than neuclease's ``fetch_body_annotations``,
    which coerces ``status`` to an ordered Categorical against a fixed Janelia vocabulary
    and **raises** on any value outside it. A newly-typed status is not a reason to fail a
    fetch, so statuses stay plain strings (see ``tables.keyvalues_to_frame``).

    Bodies with no record are simply absent from the result; the count is reported.
    """
    try:
        from neuclease.dvid.keyvalue import fetch_keyvalues
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, instance = _vdvid.address(source)
    keys = [str(b) for b in bodies]

    def go():
        return fetch_keyvalues(server, uuid, instance, keys, as_json=True,
                               batch_size=batch_size, show_progress=False)

    values = with_retry(go, label=f"body annotations for {len(keys)} bodies") or {}
    frame = tables.keyvalues_to_frame(values)
    return {"bodies": frame, "requested": len(keys), "found": int(len(frame)),
            "missing": [int(k) for k in keys if int(k) not in set(frame["body"].tolist())]}


def node_record(source: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """The provenance record for this source, with whatever the caller wants added."""
    node = source.get("node") or _vdvid.resolve_node(source)
    rec = _vdvid.node_provenance(
        source,
        {**node,
         "ref": source.get("requested_ref", node.get("ref")),
         "walked": source.get("ancestors_walked", 0)})
    info = source.get("instance_info") or {}
    rec["instance_type"] = info.get("Base", {}).get("TypeName")
    if synced := _vdvid.synced_instances(info):
        # Which labelmap the body attribution came from. Without this the table says
        # "body 13481220" without saying whose numbering that is.
        rec["synced_to"] = synced
    rec.update({k: v for k, v in extra.items() if v is not None})
    return rec
