"""Fetches that return DataFrames and write nothing. The interactive entry point.

``em_annotation.ops`` is the other half: same fetches, but they write tables and a provenance
record to a destination. Both call the same functions in :mod:`em_annotation.dvid`, so what
you get in a notebook is what a run would write — the difference is only whether it lands on
disk.

Everything here is re-exported at package top level and imported lazily, so
``from em_annotation import select_bodies`` works without ``em-annot --help`` paying for
pandas:

    >>> from em_annotation import source, select_bodies, points, body_annotations
    >>> src   = source("dvid://dvid.example.org/93fdbc:main/synapses",
    ...               locked=True)
    >>> sel   = select_bodies(src, min_synapses=10)      # DataFrame: body, pre, post, syn
    >>> pts, rels = points(src, sel.head(50))           # DataFrames; nothing written
    >>> ann   = body_annotations("@bodies", sel.head(50))

Sources are accepted in four forms, so nothing has to be spelled twice: a ``dvid://`` URL, an
``@name`` config reference (see :mod:`em_annotation.config`), an already-opened source dict,
or a plain instance name when a config supplies the server and uuid.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from . import dvid as _dvid
from . import ops as _ops

#: Which `open_*_source` to use for each kind of read.
_OPENERS = {
    "points": _ops.open_points_source,
    "bodies": _ops.open_bodies_source,
    "counts": _ops.open_counts_source,
}


def _url(location: str, *, config=None) -> str:
    """Resolve an ``@name`` reference or a bare instance name into a full DVID URL."""
    from em_volume_tools.dvid import is_url

    from . import config as _config

    if is_url(location):
        return location
    cfg = config if config is not None else _config.load()
    if location.startswith(_config.REFERENCE_PREFIX):
        return cfg.resolve(location)
    # A bare name: only meaningful with a config, and worth a clear message otherwise.
    if not cfg.server:
        raise ValueError(
            f"{location!r} is not a dvid:// URL, and no config was found to build one "
            f"from. Either pass the full URL, or create a config — see "
            f"em_annotation.config for the search path and format.")
    return cfg.url(location)


def source(location: str | Mapping[str, Any], *, kind: str = "points",
           locked: bool = False, config=None) -> dict[str, Any]:
    """Resolve a location to an opened source: concrete uuid, validated instance type.

    ``kind`` selects which validation applies — ``"points"`` for an annotation instance,
    ``"bodies"`` for a keyvalue, ``"counts"`` for a labelsz index (or the annotation instance
    it indexes). The resolved node is pinned here, once, so every later call in the notebook
    reads the same version rather than following a branch that moves under you.
    """
    if kind not in _OPENERS:
        raise ValueError(f"kind must be one of {', '.join(_OPENERS)}; got {kind!r}")
    if isinstance(location, Mapping):
        # Already a spec or an opened source. Re-opening a pinned spec is idempotent and
        # cheap (the node resolution is memoized), and it validates the instance type.
        spec = dict(location)
    else:
        from em_volume_tools.dvid import parse_url

        spec = {"backend": "dvid", **parse_url(_url(location, config=config))}
    return _OPENERS[kind](spec, prefer_locked=locked)


def _as_source(location, kind: str, locked: bool, config) -> dict[str, Any]:
    if isinstance(location, Mapping) and location.get("node") is not None:
        return dict(location)                     # already opened; do not re-resolve
    return source(location, kind=kind, locked=locked, config=config)


def body_ids(bodies) -> list[int]:
    """Body ids out of whatever you have: a DataFrame, a Series, an array, or a list.

    A DataFrame is looked up by column, so the output of :func:`select_bodies` and a table
    read back off disk both work without a ``["body"]`` in between.
    """
    import pandas as pd

    from . import bodies as _bodies

    if isinstance(bodies, pd.DataFrame):
        for name in ("body", "bodyid", "body_id"):
            if name in bodies.columns:
                return [int(v) for v in bodies[name]]
        raise KeyError(
            f"no body column in the frame (has {', '.join(map(str, bodies.columns))})")
    if isinstance(bodies, pd.Series):
        return [int(v) for v in bodies.dropna()]
    if isinstance(bodies, str):
        return _bodies.load(bodies)
    if isinstance(bodies, Iterable):
        return [int(v) for v in bodies]
    return [int(bodies)]


def select_bodies(location="counts", *, min_synapses: int = 10, min_pre: int = 0,
                  min_post: int = 0, limit: int | None = None, locked: bool = False,
                  config=None):
    """Bodies ranked by synapse count. Returns ``body``, ``pre``, ``post``, ``syn``.

    The no-write half of ``em-annot select-bodies``. Same query, same ranked ``labelsz``
    index, nothing written.
    """
    src = _as_source(location, "counts", locked, config)
    return _dvid.fetch_synapse_counts(src, min_total=min_synapses, min_pre=min_pre,
                                      min_post=min_post, limit=limit)


def points(location, bodies, *, threads: int = _dvid.DEFAULT_THREADS,
           locked: bool = False, config=None, connections: bool = False,
           rois: Sequence[str] | str | None = None,
           on_roi_overlap: str = "warn"):
    """Point annotations for ``bodies``. Returns ``(points, relationships)``.

    With ``connections=True`` returns ``(points, relationships, connections)`` — the
    oriented, de-duplicated ``(tbar, psd)`` pairs. The match rate is not printed here; take
    it with ``em_annotation.tables.match_rate(connections)``, and remember it is a statement
    about how much of the connectome ``bodies`` covers.

    ``rois`` adds a ``roi`` column to the points frame: which neuropil each synapse is in.
    A list of instance names, or ``"@name"`` for a set from the config. Only the points
    frame gets it — a relationship spans two points that may be in different neuropils, so
    a single column on it would have to pick one, while a join back to ``points`` answers
    either side. ``em_annotation.tables.body_roi_counts`` aggregates it per body.
    """
    src = _as_source(location, "points", locked, config)
    result = _dvid.fetch_points(src, body_ids(bodies), threads=threads)
    if rois:
        labelled = _dvid.label_point_rois(
            src, result["points"], roi_set(rois, config=config),
            on_overlap=on_roi_overlap)
        result["points"] = labelled["points"]
        result["rois"] = labelled
    if connections:
        return result["points"], result["relationships"], result["connections"]
    return result["points"], result["relationships"]


def roi_set(rois: Sequence[str] | str, *, config=None) -> list[str]:
    """An ROI name list from a list, a comma-separated string, or an ``@name`` config set."""
    from . import config as _config

    if isinstance(rois, str):
        if rois.startswith(_config.REFERENCE_PREFIX):
            cfg = config if config is not None else _config.load()
            return cfg.roi_set(rois[len(_config.REFERENCE_PREFIX):])
        return [r.strip() for r in rois.split(",") if r.strip()]
    return [str(r).strip() for r in rois if str(r).strip()]


def body_annotations(location="bodies", bodies=None, *, locked: bool = False, config=None):
    """The per-body property records for ``bodies``, as one row per body."""
    if bodies is None:
        raise ValueError(
            "body_annotations needs a body list: DVID cannot cheaply enumerate the bodies "
            "worth asking about. Get one from select_bodies().")
    src = _as_source(location, "bodies", locked, config)
    return _dvid.fetch_body_annotations(src, body_ids(bodies))["bodies"]


def synapse_counts(location="counts", bodies=None, *, locked: bool = False, config=None):
    """Exact ``PreSyn``/``PostSyn`` counts for specific bodies, ignoring any threshold.

    :func:`select_bodies` answers "which bodies are big"; this answers "how big are these",
    which is the one a notebook usually wants when the body list came from somewhere else.
    """
    import pandas as pd

    if bodies is None:
        raise ValueError("synapse_counts needs a body list")
    src = _as_source(location, "counts", locked, config)
    ids = body_ids(bodies)
    from neuclease.dvid import labelsz

    from em_volume_tools.dvid import address

    server, uuid, instance = address(src)
    pre = labelsz.fetch_counts(server, uuid, instance, ids, "PreSyn")
    post = labelsz.fetch_counts(server, uuid, instance, ids, "PostSyn")
    out = pd.DataFrame({"pre": pre.reindex(ids).fillna(0).astype("int64"),
                        "post": post.reindex(ids).fillna(0).astype("int64")})
    out["syn"] = out["pre"] + out["post"]
    return out.rename_axis("body").reset_index()
