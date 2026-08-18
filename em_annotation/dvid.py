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
with 503. Measured on our dataset: 400 bodies in 19.7 s at 8 threads, so ~16 min for
20k bodies — fast enough that pointing a dask fleet at a shared server buys nothing worth
the risk.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Mapping, Sequence

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
COUNT_INSTANCE = "labelsz"


def open_source(spec: Mapping[str, Any], *, expect: str | tuple[str, ...],
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
    expected = (expect,) if isinstance(expect, str) else tuple(expect)
    _vdvid.check_instance_type(info, pinned, *expected)
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


# --------------------------------------------------------------------------- #
# choosing which bodies to fetch, from DVID's own ranked synapse index
# --------------------------------------------------------------------------- #
#: DVID's page size for a `labelsz` threshold query. Not ours to choose — the endpoint
#: caps a response at 10,000 and pages by rank via `offset`.
_LABELSZ_PAGE = 10_000

#: A bound on how deep to page. Each page costs MORE than the last (DVID appears to
#: re-rank per request: measured 4.8 s at offset 0 against 11.5 s at offset 50,000), so an
#: unbounded walk down a low threshold is what makes this look like a hang rather than a
#: slow query. neuclease's own `fetch_threshold` defaults `n` to 1e12, i.e. exactly that.
_MAX_SELECT = 500_000


def resolve_labelsz(source: Mapping[str, Any]) -> dict[str, Any]:
    """Point a source at the ``labelsz`` instance to query, from whatever was given.

    Accepts either the ``labelsz`` instance itself or the ``annotation`` instance it
    indexes — the latter because that is the name people know (`synapses`), and the index's
    own ``Base.Syncs`` records which annotation instance it belongs to, so the mapping is
    discoverable rather than something to memorise.
    """
    info = source.get("instance_info") or _vdvid.instance_info(source)
    kind = _vdvid.instance_type(info)
    if kind == COUNT_INSTANCE:
        return {**dict(source), "indexes": _vdvid.synced_instances(info) or None}

    # An annotation instance: find the labelsz that syncs to it.
    try:
        from neuclease.dvid import fetch_repo_info
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, instance = _vdvid.address(source)
    repo = fetch_repo_info(server, uuid)
    matches = [name for name, d in (repo.get("DataInstances") or {}).items()
               if d.get("Base", {}).get("TypeName") == COUNT_INSTANCE
               and instance in (d.get("Base", {}).get("Syncs") or [])]
    if not matches:
        raise ValueError(
            f"no 'labelsz' instance on this node indexes {instance!r}, so DVID has no "
            f"per-body synapse counts to rank by. Point --src at a labelsz instance "
            f"directly, or have one created and synced to {instance!r} in DVID.")
    if len(matches) > 1:
        raise ValueError(
            f"several labelsz instances index {instance!r} ({', '.join(sorted(matches))}); "
            f"name the one you want in --src.")
    logger.info("using labelsz instance %r (it indexes %r)", matches[0], instance)
    return {**dict(source), "instance": matches[0], "indexes": [instance]}


def fetch_synapse_counts(source: Mapping[str, Any], *, min_total: int = 10,
                         min_pre: int = 0, min_post: int = 0,
                         limit: int | None = None):
    """Bodies with at least this many synapses, with their pre/post counts.

    Driven by one ranked ``threshold`` query on ``AllSyn``, then refined with exact
    per-element-type counts. The threshold used is ``max`` of the three minimums, which is
    sound because ``AllSyn`` is the catch-all and is therefore ``>=`` each individual type —
    verified against real bodies rather than assumed.

    Returns a DataFrame indexed 0..n with ``body``, ``pre``, ``post``, ``syn``, ranked by
    ``syn`` descending.
    """
    import pandas as pd

    from neuclease.dvid import labelsz

    server, uuid, instance = _vdvid.address(source)
    threshold = max(int(min_total), int(min_pre), int(min_post), 1)

    # Page explicitly rather than letting neuclease default n to 1e12: per-page cost grows
    # with offset, so an accidental deep walk is the difference between 20 s and minutes.
    pages, offset = [], 0
    while offset < _MAX_SELECT:
        want = min(_LABELSZ_PAGE, _MAX_SELECT - offset)
        page = with_retry(
            lambda o=offset, w=want: labelsz.fetch_threshold(
                server, uuid, instance, threshold, "AllSyn", offset=o, n=w),
            label=f"labelsz AllSyn>={threshold} offset={offset}")
        if page is None or not len(page):
            break
        pages.append(page)
        offset += len(page)
        if len(page) < want:
            break
        logger.info("selected %d bodies so far (AllSyn >= %d)", offset, threshold)
    else:
        logger.warning(
            "stopped at the %d-body cap with more bodies still above the threshold. "
            "Raise --min-synapses, or --limit what you need: paging deeper costs more per "
            "page than the last.", _MAX_SELECT)

    if not pages:
        # An empty result is the documented failure mode of asking a labelsz instance for
        # an element type it does not index, so say what was actually asked.
        raise ValueError(
            f"no bodies have AllSyn >= {threshold} in {_vdvid.spec_url(source)}. If that "
            f"is surprising, check that this labelsz instance is synced to an annotation "
            f"instance — DVID answers an unindexed element type with an EMPTY result "
            f"rather than an error.")

    syn = pd.concat(pages)
    bodies = [int(b) for b in syn.index]
    pre = with_retry(
        lambda: labelsz.fetch_counts(server, uuid, instance, bodies, "PreSyn"),
        label="labelsz PreSyn counts")
    post = with_retry(
        lambda: labelsz.fetch_counts(server, uuid, instance, bodies, "PostSyn"),
        label="labelsz PostSyn counts")

    df = pd.DataFrame({"syn": syn.astype("int64")})
    df["pre"] = pre.reindex(df.index).fillna(0).astype("int64")
    df["post"] = post.reindex(df.index).fillna(0).astype("int64")
    if min_pre:
        df = df[df["pre"] >= int(min_pre)]
    if min_post:
        df = df[df["post"] >= int(min_post)]
    df = df.sort_values(["syn", "pre"], ascending=False)
    if limit is not None:
        df = df.head(int(limit))
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "body"})
    out["body"] = out["body"].astype(tables.BODY_DTYPE)
    return out[["body", "pre", "post", "syn"]]


# --------------------------------------------------------------------------- #
# which neuropil each synapse is in
# --------------------------------------------------------------------------- #
#: What neuclease names label 0 — a point inside none of the given ROIs. Normalised to null
#: in our tables, because "in no ROI" is missing data, not a region called `<unspecified>`.
ROI_UNSPECIFIED = "<unspecified>"

ROI_INSTANCE = "roi"


def available_rois(source: Mapping[str, Any]) -> list[str]:
    """Every ``roi`` instance on this node."""
    try:
        from neuclease.dvid import fetch_repo_info
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, _instance = _vdvid.address(source)
    repo = fetch_repo_info(server, uuid)
    return sorted(name for name, d in (repo.get("DataInstances") or {}).items()
                  if d.get("Base", {}).get("TypeName") == ROI_INSTANCE)


def resolve_roi_set(source: Mapping[str, Any], rois: Sequence[str]) -> list[str]:
    """Validate an ROI name list against the node, before anything expensive happens.

    Checked up front because building the combined volume fetches every named ROI — on
    dvid.example.org that is ~1 s each — and a typo would otherwise surface as a failure or, worse,
    as a silently smaller set of labelled points.
    """
    wanted = [str(r).strip() for r in rois if str(r).strip()]
    if not wanted:
        raise ValueError(
            "no ROIs given. There is deliberately no default: the combined ROI volume is "
            "built by overwriting, so asking for every ROI on the node would label a point "
            "in ME(L) as whichever of ME(L) / OL(L) / all_neuropils was written last.")
    duplicated = sorted({r for r in wanted if wanted.count(r) > 1})
    if duplicated:
        raise ValueError(f"ROI list repeats {', '.join(duplicated)}")
    have = set(available_rois(source))
    missing = [r for r in wanted if r not in have]
    if missing:
        raise ValueError(
            f"no roi instance on this node named {', '.join(missing)}. "
            f"{len(have)} are available; the closest are "
            f"{', '.join(_closest(missing[0], have))}.")
    return wanted


def _closest(name: str, candidates: Iterable[str], n: int = 5) -> list[str]:
    import difflib

    return difflib.get_close_matches(name, sorted(candidates), n=n, cutoff=0.4) or \
        sorted(candidates)[:n]


#: What to do when the chosen ROIs intersect. Overlap is **expected to be small** in this
#: dataset, and there is no principled tie-break — so the default proceeds and quantifies it
#: rather than refusing. "error" is there for a caller who wants a strict partition.
ON_OVERLAP = ("warn", "error")


def label_point_rois(source: Mapping[str, Any], points, rois: Sequence[str], *,
                     on_overlap: str = "warn", processes: int = 0) -> dict[str, Any]:
    """Add a ``roi`` column to a points frame: which neuropil each synapse falls in.

    Wraps ``neuclease.dvid.roi.determine_point_rois``, which builds one combined label
    volume from the named ROIs and samples it at every point. Two properties worth knowing:

    - It works at **scale 5** while taking scale-0 coordinates, so this is cheap: one small
      volume, then a vectorised lookup for millions of points.
    - It wants ``x``/``y``/``z`` **columns**, which our tables have by name — so there is no
      flip here, and no opportunity for one. That is the payoff for never storing a bare
      positional coordinate array.

    ## Overlap

    The combined volume is built by writing each ROI in turn, so where two intersect the
    **later one wins** and the earlier is overwritten. There is no principled way to break
    that tie, and in this dataset the intersections are expected to be small — so refusing
    would be the wrong default.

    Instead the ambiguity is *measured, in the unit that matters*: the volume is unpacked a
    second time with the ROI order reversed and the two labellings compared, so the report
    says how many of **your synapses** would be attributed differently. That is nearly free
    — fetching the ROIs is the expensive step and it is not repeated; only the unpack is.
    A voxel-overlap figure cannot answer "should I care"; a count of affected synapses can.

    ``on_overlap="error"`` refuses instead, for a caller who needs a strict partition. This
    dataset has deliberately-subtracted variants (``INP(-ATL)(L)``, ``PENP(-AMMC)``,
    ``VLNP(-AOTU)(L)``) if one is wanted.
    """
    import pandas as pd

    try:
        from neuclease.dvid.roi import (determine_point_rois, fetch_roi_ranges_and_boxes,
                                        unpack_roi_ranges_to_combined_volume)
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    if on_overlap not in ON_OVERLAP:
        raise ValueError(f"on_overlap must be one of {', '.join(ON_OVERLAP)}; "
                         f"got {on_overlap!r}")

    names = resolve_roi_set(source, rois)
    server, uuid, _instance = _vdvid.address(source)

    logger.info("fetching %d ROIs", len(names))
    # The fetch is the expensive half and is done once; unpacking is cheap and is what gets
    # repeated below to measure the ambiguity.
    ranges, boxes = fetch_roi_ranges_and_boxes(server, uuid, names, processes=processes)
    volume, box, overlaps = unpack_roi_ranges_to_combined_volume(names, ranges, boxes)

    overlapping = [] if overlaps is None or not len(overlaps) else [
        (str(a), str(b), int(n)) for a, b, n in
        overlaps[["roi_a", "roi_b", "overlap"]].itertuples(index=False)]
    if overlapping and on_overlap == "error":
        raise ValueError(
            f"{len(overlapping)} of the given ROIs overlap, e.g. {overlapping[:4]}. The "
            f"combined volume is built by overwriting, so every point in an intersection "
            f"is attributed to whichever ROI was passed last. Choose a non-overlapping set "
            f"(this dataset has subtracted variants such as INP(-ATL)(L) and PENP(-AMMC) "
            f"for exactly this), or use on_overlap='warn' to proceed and have the affected "
            f"synapse count reported.")

    if not len(points):
        out = points.copy()
        out["roi"] = pd.Series(dtype="string")
        return {"points": out, "rois": names, "labeled": 0, "unlabeled": 0,
                "overlapping": overlapping, "ambiguous": 0, "counts": {}}

    # determine_point_rois mutates in place and needs a unique index; both are satisfied by
    # a copy with a fresh RangeIndex, and a copy keeps the caller's frame untouched.
    work = points.reset_index(drop=True).copy()
    determine_point_rois(server, uuid, names, work, combined_vol=volume, combined_box=box)
    roi = work["roi"].astype("string")
    roi = roi.where(roi != ROI_UNSPECIFIED, pd.NA)

    ambiguous, ambiguous_counts = 0, {}
    if overlapping:
        # Reverse the priority and see which points change hands. Same fetched ranges, same
        # box, so the only difference is which ROI wins an intersection.
        reversed_names = list(reversed(names))
        alt_vol, alt_box, _ = unpack_roi_ranges_to_combined_volume(
            reversed_names, ranges, boxes, box_zyx=box)
        alt = points.reset_index(drop=True).copy()
        determine_point_rois(server, uuid, reversed_names, alt,
                             combined_vol=alt_vol, combined_box=alt_box)
        other = alt["roi"].astype("string")
        other = other.where(other != ROI_UNSPECIFIED, pd.NA)
        differs = (roi.fillna("") != other.fillna(""))
        ambiguous = int(differs.sum())
        # Which way each ambiguous point could have gone, so the report names the competing
        # pair rather than just a count.
        pairs = pd.Series([f"{x} | {y}" for x, y in
                           zip(roi[differs].fillna("(none)"),
                               other[differs].fillna("(none)"))], dtype="object")
        ambiguous_counts = ({str(k): int(v) for k, v in pairs.value_counts().items()}
                            if len(pairs) else {})
        logger.warning(
            "%d ROI pairs intersect; %d of %d synapses (%.2f%%) sit in an intersection and "
            "are attributed by ROI ORDER alone: %s",
            len(overlapping), ambiguous, len(roi), 100 * ambiguous / max(len(roi), 1),
            ambiguous_counts or "none of yours")

    out = points.reset_index(drop=True).copy()
    out["roi"] = roi
    # `roi_label` is an index into the ROI list and is meaningless without it; the list goes
    # into the provenance record instead.
    unlabeled = int(roi.isna().sum())
    counts = {str(k): int(v) for k, v in roi.value_counts().items()}
    logger.info("%d of %d points fell inside an ROI (%d outside every one)",
                len(roi) - unlabeled, len(roi), unlabeled)
    return {"points": out, "rois": names, "labeled": int(len(roi) - unlabeled),
            "unlabeled": unlabeled, "overlapping": overlapping,
            "ambiguous": ambiguous, "ambiguous_pairs": ambiguous_counts,
            "counts": counts}


# --------------------------------------------------------------------------- #
# the whole keyvalue instance, without asking for its key list
# --------------------------------------------------------------------------- #
#: DVID's own full-range bounds for a keyvalue query, which is what neuclease defaults to.
_RANGE_LO = " "
_RANGE_HI = chr(ord("~") + 1)

#: Boundaries covering the printable key space in bounded requests. Every consecutive pair is
#: one request, and the union covers everything from ``" "`` to ``"~"`` — so **completeness
#: comes from the cover, not from knowing a total**. Body-id keys are decimal strings, so the
#: digit boundaries do the real work; the two outer ranges catch a non-numeric key (a schema
#: or config document) that would otherwise be silently missed.
DEFAULT_KEY_BOUNDARIES = (_RANGE_LO, *"0123456789", ":", _RANGE_HI)


def key_ranges(boundaries: Sequence[str] = DEFAULT_KEY_BOUNDARIES
               ) -> list[tuple[str, str]]:
    """Consecutive pairs of ``boundaries`` as ``(key1, key2)`` request bounds.

    **DVID's key range is INCLUSIVE at both ends**, not half-open — its own documentation
    calls ``key2`` the "maximal key" and notes that ``'a'``..``'z'`` catches a single ``'z'``.
    So consecutive ranges built this way *overlap by exactly one key* wherever a key equals a
    boundary, and a body literally numbered ``3`` comes back from both ``'2'``..``'3'`` and
    ``'3'``..``'4'``.

    That is harmless for completeness — an overlapping cover misses nothing, and collecting
    into a dict de-duplicates — but it does mean **summing the per-range counts overcounts**.
    Measured: ``fetch_keyrange`` totalled 58,395 keys across these ranges while
    ``fetch_keyrangevalues`` yielded 58,394 distinct records, the difference being boundary
    key ``'3'`` counted twice.
    """
    bounds = list(boundaries)
    if len(bounds) < 2:
        raise ValueError("need at least two boundaries to make a range")
    if sorted(bounds) != bounds:
        raise ValueError(f"boundaries must be sorted; got {bounds}")
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if a != b]


#: How many times a failing key range may be subdivided. Each level multiplies the request
#: count by eleven, so depth 2 allows ~121 sub-ranges per top-level range — about 134 records
#: each for the biggest bucket here, comfortably inside any plausible proxy window. Kept low
#: on purpose: depth 3 is ~16,000 requests in the worst case, which turns a broken server into
#: an hours-long failure instead of a quick one.
#: A dead server is reported after only a handful of requests without needing a failure
#: budget: the first sub-range of any split is ``lo``..``lo + '0'``, which has nothing
#: strictly between its bounds and so cannot be subdivided — see :func:`refine`. That guard
#: fires immediately, so total failure surfaces in under ten requests rather than exhausting
#: the recursion.
MAX_SPLIT_DEPTH = 2


def refine(lo: str, hi: str) -> list[str]:
    """Boundaries subdividing ``[lo, hi]`` by appending one digit to ``lo``.

    ``'1'``..``'2'`` becomes ``'1', '10', '11', … '19', '2'``, which works because these keys
    are decimal strings sorted lexicographically: ``'1' < '10' < … < '19' < '2'``. Each
    sub-range is roughly a tenth of the original.

    This is what makes the read self-tuning, and it is needed rather than optional. A fixed
    coarse grid is not enough: the ``'1'``..``'2'`` range holds 16,225 records and, having
    once completed in 13.7 s, later **504'd twice in a row** at about 60 s each — the proxy
    window is fixed but the server's speed is not, so any static bucket size is a bet on load.
    Splitting on failure removes the bet, and keeps the common case at a dozen requests
    because subdivision only happens where it is needed.

    Candidates outside ``(lo, hi)`` are dropped, which is not a detail: ``hi`` is not always
    the next character after ``lo``. Splitting ``' '``..``'0'`` yields ``' 0'``..``' 9'``, and
    splitting *that* again would put ``' 0'`` after ``' 9'`` and produce unsorted boundaries.
    Returns ``[lo, hi]`` unchanged when no candidate falls strictly between them — the caller
    reads that as "cannot subdivide further" rather than looping.
    """
    inner = [lo + d for d in "0123456789"]
    return [lo, *(m for m in inner if lo < m < hi), hi]


def fetch_all_body_annotations(source: Mapping[str, Any], *,
                               boundaries: Sequence[str] = DEFAULT_KEY_BOUNDARIES
                               ) -> dict[str, Any]:
    """Every record in a keyvalue instance, fetched as a series of bounded key ranges.

    **Deliberately never calls ``/keys``.** On a large instance that endpoint is unreliable
    rather than merely slow: measured on `labels_annotations` (58,394 keys) it took 52 s
    twice and then returned **504 Gateway Time-out** from the nginx proxy in front of DVID.
    A call that succeeds in testing and fails intermittently in production is the worst thing
    to build on, and there is no count endpoint to use instead — ``/keys`` always returns the
    whole list.

    So the instance is read as ``keyrangevalues`` over a cover of the key space. Three things
    follow, and all of them matter:

    - **No total is needed.** The ranges are exhaustive by construction, so every key is in at
      least one; there is nothing to compare a count against and nothing to stop early.
    - **Values are nearly free.** Measured 56.7 s for all 58,394 records (~4.9 MB) against
      54.3 s for the keys alone, worst single request 13.7 s. There is never a reason to
      fetch keys and then values separately.
    - **The ranges overlap by one key at each boundary**, because DVID's bounds are inclusive
      (see :func:`key_ranges`). Collecting into a dict de-duplicates, so the result is right —
      but a boundary key legitimately arrives twice and that is not an error.

    Pass finer ``boundaries`` (two-character prefixes, say) if an instance ever grows enough
    that one range starts timing out; the cover property holds for any sorted list.
    """
    try:
        from neuclease.dvid.keyvalue import fetch_keyrangevalues
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, instance = _vdvid.address(source)
    records: dict[str, Any] = {}
    per_range: list[dict] = []

    def fetch(lo: str, hi: str, depth: int) -> None:
        try:
            # ONE attempt, no backoff sleep: a 504 here means the request could not finish
            # inside the proxy's window, and repeating it unchanged just fails again more
            # slowly (measured: two 60 s attempts on the same range, both 504). Splitting is a
            # strictly better retry — a smaller request is more likely to succeed than the
            # same one again — so go straight to it rather than sleeping first.
            got = fetch_keyrangevalues(server, uuid, instance, lo, hi, as_json=True) or {}
        except Exception as exc:                                  # noqa: BLE001
            parts = key_ranges(refine(lo, hi))
            if depth >= MAX_SPLIT_DEPTH or len(parts) < 2:
                raise RuntimeError(
                    f"[{lo!r},{hi!r}] still fails and cannot usefully be split further "
                    f"(depth {depth} of {MAX_SPLIT_DEPTH}): "
                    f"{type(exc).__name__}: {exc}") from exc
            logger.warning("[%r,%r] failed (%s); splitting it and fetching the parts",
                           lo, hi, type(exc).__name__)
            for sub_lo, sub_hi in parts:
                fetch(sub_lo, sub_hi, depth + 1)
            return

        # Boundary keys arrive twice because DVID's bounds are inclusive, which is expected.
        # A repeat carrying a DIFFERENT value is not: that would mean the instance answered
        # inconsistently across two requests, and the dict update would silently pick one.
        inconsistent = [k for k in set(got) & set(records) if got[k] != records[k]]
        if inconsistent:
            raise RuntimeError(
                f"key(s) {sorted(inconsistent)[:3]} came back with different values from "
                f"two key ranges. The instance changed mid-read, or the server answered "
                f"inconsistently — either way this snapshot is not coherent.")
        records.update(got)
        per_range.append({"lo": lo, "hi": hi, "records": len(got), "depth": depth})
        logger.info("fetched %d records from [%r,%r] — %d so far",
                    len(got), lo, hi, len(records))

    for lo, hi in key_ranges(boundaries):
        fetch(lo, hi, 0)

    return {"records": records, "ranges": per_range}


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
