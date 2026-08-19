"""The table model: elements and relationships as DataFrames.

## Axis order

DVID stores a position as ``Pos: [x, y, z]`` and neuclease's parsers hand back ``x``,
``y``, ``z`` columns. em-libraries holds coordinates **zyx in memory** (CLAUDE.md
invariant 2), and the precomputed annotation format stores **xyz on disk**. So there are
three conventions in play and the mirroring bug they invite — a position reflected through
the ``z=x`` diagonal — produces a *valid* annotation in the wrong place, which no shape,
bounds or dtype check can see.

What defuses it here is that **every coordinate in every table is a named column**:
``z``, ``y``, ``x``, and ``to_z``/``to_y``/``to_x``. A named column cannot be silently
transposed; only a positional array can. So the rule is narrow and enforceable:

    :func:`positions_zyx` and :func:`positions_xyz` are the ONLY places a coordinate
    array is built from a table, and the only places the order is decided.

Anything needing an ``(N, 3)`` array — a labelmap read, an annotation encoder — goes
through one of them and says which order it wanted. Column *order* within the frame is
zyx to match the rest of the codebase, but nothing depends on it.

## Why relationships are stored raw

A DVID relationship points at a **coordinate, not a body**: ``{"Rel": "PostSynTo", "To":
[x, y, z]}``. The partner's body is recovered by joining that coordinate against the
elements table — which works precisely because elements were fetched *per body*, so every
row already carries the body it came from.

That join only resolves an edge when **both** endpoints were fetched, so its yield is a
function of how much of the connectome the body list covers. Measured on our dataset,
taking the top-N bodies by presynapse count: 20 bodies resolved 0.2% of pairs, 100 → 2.0%,
400 → 4.6%. This is why :func:`match_rate` exists and why the CLI reports it as a headline
number: a low rate means the body list is too small, and it looks exactly like a sparse
connectome if nobody says so.

Relationships are therefore kept **one row per relationship, unmodified**, with
``from_body``/``to_body`` filled in where known. :func:`connections` derives the oriented,
de-duplicated synapse view on top. Keeping the raw table lossless matters because the two
relationship directions are *not* redundant: ``PostSynTo`` from a fetched PSD reaches a
tbar outside the set, ``PreSynTo`` from a fetched tbar reaches PSDs outside it, and each
direction is anchored on the body list at one end only. Taking one direction would
discard half the edges.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: In-memory axis order for this package, and the column order of every table it writes.
AXES = ("z", "y", "x")

#: DVID's own order, which is what ``Pos`` and ``To`` are in.
DVID_AXES = ("x", "y", "z")

#: Coordinate dtype. int32 spans ±2.1e9 voxels — ample for any EM volume, and half the
#: width of int64 across tables that reach tens of millions of rows.
COORD_DTYPE = np.int32

#: Body ids are uint64 in DVID and must stay uint64 all the way to disk. See io.py: this
#: is the single reason parquet rather than csv is the default output format.
BODY_DTYPE = np.uint64

#: Relationship names that describe a chemical synapse, and which endpoint each puts
#: where. ``PostSynTo`` on an element means "this element is postsynaptic to the target",
#: so the element is the PSD and the target is the tbar; ``PreSynTo`` is the reverse.
#: Derived from the relationship name rather than the element's ``Kind`` so that a
#: mislabelled element cannot silently invert an edge.
SYNAPTIC_RELS = {"PostSynTo": ("post", "pre"), "PreSynTo": ("pre", "post")}


def _empty_points() -> pd.DataFrame:
    return pd.DataFrame({
        "body": pd.Series(dtype=BODY_DTYPE),
        **{a: pd.Series(dtype=COORD_DTYPE) for a in AXES},
        "kind": pd.Series(dtype="string"),
    })


def _empty_rels() -> pd.DataFrame:
    return pd.DataFrame({
        "rel": pd.Series(dtype="string"),
        **{a: pd.Series(dtype=COORD_DTYPE) for a in AXES},
        **{f"to_{a}": pd.Series(dtype=COORD_DTYPE) for a in AXES},
        "from_body": pd.Series(dtype=BODY_DTYPE),
        "to_body": pd.Series(dtype="UInt64"),
    })


def elements_to_frames(elements: Sequence[Mapping[str, Any]],
                       body: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One body's ``/label`` response -> (points, relationships).

    Parsed here rather than through ``neuclease.load_elements_as_dataframe`` because that
    returns relationships indexed by an xyz tuple and drops the element's body, and
    because properties need to arrive as columns without a second normalize pass. The
    element JSON shape is stable and simple: ``Pos``, ``Kind``, ``Tags``, ``Prop``,
    ``Rels``.
    """
    if not elements:
        return _empty_points(), _empty_rels()

    rows: list[dict] = []
    rel_rows: list[dict] = []
    for el in elements:
        pos = el.get("Pos")
        if pos is None or len(pos) != 3:
            raise ValueError(
                f"element in body {body} has Pos={pos!r}; expected three coordinates")
        # The one xyz -> zyx transcription, by name.
        x, y, z = (int(v) for v in pos)
        row: dict[str, Any] = {"body": int(body), "z": z, "y": y, "x": x,
                               "kind": el.get("Kind")}
        tags = el.get("Tags") or []
        # Kept as a delimited string: a list column cannot go to csv and round-trip, and
        # tags are a handful of short flags in practice.
        row["tags"] = ",".join(str(t) for t in tags) if tags else ""
        for key, value in (el.get("Prop") or {}).items():
            # `conf`/`user`/`annotation` are the ones seen in practice; anything else a
            # tool writes comes through under its own name rather than being dropped.
            row[str(key)] = value
        rows.append(row)

        for rel in el.get("Rels") or []:
            to = rel.get("To")
            if to is None or len(to) != 3:
                # DVID emits a relationship with a null target for a partner that was
                # deleted. Recorded as a row with no target rather than dropped, so the
                # count of dangling references stays visible.
                tx = ty = tz = None
            else:
                tx, ty, tz = (int(v) for v in to)
            rel_rows.append({"rel": rel.get("Rel"), "z": z, "y": y, "x": x,
                             "to_z": tz, "to_y": ty, "to_x": tx,
                             "from_body": int(body)})

    points = pd.DataFrame(rows)
    rels = pd.DataFrame(rel_rows) if rel_rows else _empty_rels()
    return points, rels


def _coerce_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = df["body"].astype(BODY_DTYPE)
    for a in AXES:
        df[a] = df[a].astype(COORD_DTYPE)
    if "kind" in df:
        df["kind"] = df["kind"].astype("string")
    if "conf" in df:
        # DVID stores confidence as a *string* in Prop; float32 is plenty and keeps a
        # 4M-row table small.
        df["conf"] = pd.to_numeric(df["conf"], errors="coerce").astype(np.float32)
    lead = ["body", *AXES, "kind", "tags"]
    rest = [c for c in df.columns if c not in lead]
    return df[[c for c in lead if c in df.columns] + rest]


def combine(frames: Iterable[tuple[pd.DataFrame, pd.DataFrame]]
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate per-body ``(points, rels)`` pairs and resolve partner bodies."""
    point_parts, rel_parts = [], []
    for pts, rels in frames:
        if len(pts):
            point_parts.append(pts)
        if len(rels):
            rel_parts.append(rels)

    points = (_coerce_points(pd.concat(point_parts, ignore_index=True))
              if point_parts else _empty_points())
    rels = (pd.concat(rel_parts, ignore_index=True) if rel_parts else _empty_rels())
    return points, resolve_partner_bodies(points, rels)


def position_index(points: pd.DataFrame) -> pd.Series:
    """``(z, y, x) -> body`` for every fetched element.

    A position identifies at most one element within an instance, which is checked rather
    than assumed: a duplicate would make the partner join ambiguous and silently pick one.
    """
    if not len(points):
        return pd.Series(dtype=BODY_DTYPE)
    dup = points.duplicated(subset=list(AXES)).sum()
    if dup:
        raise ValueError(
            f"{dup} elements share a position with another element. A position is the "
            f"join key for relationships, so duplicates would make partner resolution "
            f"ambiguous. This should not happen within one annotation instance.")
    return points.set_index(list(AXES))["body"]


def resolve_partner_bodies(points: pd.DataFrame, rels: pd.DataFrame) -> pd.DataFrame:
    """Fill ``to_body`` by joining each relationship's target against ``points``.

    Left unset (``NA``) where the target was not fetched. Nullable ``UInt64`` rather than
    float, because a body id that has been through float64 is a body id that may have
    been rounded.
    """
    rels = rels.copy()
    if not len(rels):
        rels["to_body"] = pd.Series(dtype="UInt64")
        return rels

    for col in ("from_body",):
        rels[col] = rels[col].astype(BODY_DTYPE)
    lut = position_index(points)
    keys = pd.MultiIndex.from_arrays(
        [rels[f"to_{a}"] for a in AXES], names=list(AXES))
    rels["to_body"] = pd.Series(lut.reindex(keys).to_numpy(), index=rels.index,
                                dtype="Float64").astype("UInt64")
    for a in AXES:
        rels[a] = rels[a].astype(COORD_DTYPE)
        # to_* stays nullable: a dangling relationship has no target coordinate.
        rels[f"to_{a}"] = rels[f"to_{a}"].astype("Int32")
    lead = ["rel", *AXES, *[f"to_{a}" for a in AXES], "from_body", "to_body"]
    return rels[[c for c in lead if c in rels.columns]
                + [c for c in rels.columns if c not in lead]]


def connections(rels: pd.DataFrame) -> pd.DataFrame:
    """The oriented, de-duplicated synapse view over the raw relationship table.

    One row per distinct ``(tbar, psd)`` coordinate pair, whichever direction it was seen
    from. Columns ``pre_z/pre_y/pre_x``, ``post_z/post_y/post_x``, ``pre_body``,
    ``post_body``.

    Both directions are used and then de-duplicated because each is anchored on the body
    list at one end only — see the module docstring. Non-synaptic relationship types are
    ignored here and remain in the raw table.
    """
    cols = {}
    frames = []
    for name, (here, there) in SYNAPTIC_RELS.items():
        part = rels[rels["rel"] == name]
        if not len(part):
            continue
        cols = {}
        for a in AXES:
            cols[f"{here}_{a}"] = part[a]
            cols[f"{there}_{a}"] = part[f"to_{a}"]
        cols[f"{here}_body"] = part["from_body"]
        cols[f"{there}_body"] = part["to_body"]
        frames.append(pd.DataFrame(cols))
    if not frames:
        return pd.DataFrame(columns=[f"{s}_{a}" for s in ("pre", "post") for a in AXES]
                            + ["pre_body", "post_body"])

    out = pd.concat(frames, ignore_index=True)
    order = [f"pre_{a}" for a in AXES] + [f"post_{a}" for a in AXES]
    out = out[order + ["pre_body", "post_body"]]
    # A pair seen from both ends appears twice; one of the two carries each body. Group so
    # the surviving row keeps whichever endpoint bodies are known.
    out = (out.groupby(order, dropna=False, as_index=False)
              .agg({"pre_body": "first", "post_body": "first"}))
    return out


def enrich_connections(conns: pd.DataFrame, points: pd.DataFrame,
                       columns: Sequence[str] = ("conf",)) -> pd.DataFrame:
    """Bring per-element columns onto a connections table as ``pre_<col>`` / ``post_<col>``.

    :func:`connections` carries only geometry and the two body ids, because that is all a
    relationship row knows. Anything recorded on the *element* — its confidence, the ROI it
    sits in — has to be joined back from ``points`` on the coordinate, which is safe because a
    position identifies at most one element (:func:`position_index` checks that).

    **A missing value stays NaN rather than becoming 0.** For a half-resolved line the partner
    element was never fetched, so its confidence is genuinely unknown; zero would claim the
    annotator had no confidence in a real synapse. NaN also fails every shader comparison, so
    ``if (prop_conf_post() < threshold) discard;`` leaves such a line visible — the permissive
    reading, which is the right default for data whose partner simply was not requested.
    """
    missing = [c for c in columns if c not in points.columns]
    if missing:
        raise KeyError(
            f"points has no column(s) {', '.join(missing)}; it has "
            f"{', '.join(map(str, points.columns))}")
    lut = points.drop_duplicates(subset=list(AXES)).set_index(list(AXES))[list(columns)]
    out = conns.copy()
    for side in ("pre", "post"):
        keys = pd.MultiIndex.from_arrays(
            [conns[f"{side}_{a}"] for a in AXES], names=list(AXES))
        got = lut.reindex(keys)
        for col in columns:
            out[f"{side}_{col}"] = got[col].to_numpy()
    return out


def match_rate(conns: pd.DataFrame) -> dict[str, Any]:
    """How much of the connectivity the body list actually resolved.

    Reported prominently because a low number is indistinguishable from a genuinely
    sparse connectome once the tables are written.
    """
    total = int(len(conns))
    if not total:
        return {"pairs": 0, "both_ends": 0, "one_end": 0, "fraction": None}
    both = int((conns["pre_body"].notna() & conns["post_body"].notna()).sum())
    one = int((conns["pre_body"].notna() ^ conns["post_body"].notna()).sum())
    return {"pairs": total, "both_ends": both, "one_end": one,
            "fraction": both / total}


def body_roi_counts(points: pd.DataFrame, *, kind_column: str = "kind") -> pd.DataFrame:
    """Per body and kind, how many synapses fall in each ROI. Long form, lossless.

    Not written as a table by any command — it is a group-by over ``points`` and would go
    stale beside it. Provided because it is the shape a tagging rule wants: "where are this
    body's presynapses" is ``pivot`` away, and where a neuron's *output* sits is often what
    identifies it.
    """
    if "roi" not in points.columns:
        raise KeyError(
            "no 'roi' column: fetch the points with a ROI set (em-annot points --rois, or "
            "notebook.points(..., rois=[...])) to label each synapse with its neuropil.")
    have = [c for c in ("body", kind_column, "roi") if c in points.columns]
    counts = (points[have].dropna(subset=["roi"])
              .value_counts(dropna=True).rename("synapses").reset_index())
    return counts.sort_values(["body", "synapses"], ascending=[True, False],
                              ignore_index=True)


def keyvalues_to_frame(values: Mapping[str, Any]) -> pd.DataFrame:
    """``{body_key: json}`` from a keyvalue instance -> one row per body.

    The records are **ragged** — on our dataset some bodies carry ``user`` and
    ``instance_user`` and others do not — so this normalizes to the union of keys, with
    missing fields left null.

    ``instance`` is kept under its own name even though it collides with DVID's *data
    instance*: in a body annotation it is the neuron's name (``CAm(L)``), and renaming it
    would break every join against neuclease and neuprint output. The raw record is kept
    in a ``json`` column so an edit can be pushed back without reconstructing it.
    """
    rows = []
    for key, value in values.items():
        if value is None:
            continue
        if not str(key).isdigit():
            # A keyvalue instance may hold non-body keys (schema documents, config).
            continue
        body = int(key)
        record = dict(value) if isinstance(value, Mapping) else {"value": value}
        stated = record.get("bodyid", record.get("body ID"))
        if stated is not None and int(stated) != body:
            raise ValueError(
                f"body annotation under key {key!r} says bodyid={stated!r}. The key is "
                f"authoritative here, and a mismatch means the instance is inconsistent "
                f"rather than that one of them should be preferred.")
        rows.append({"body": body,
                     **{k: v for k, v in record.items() if k != "bodyid"},
                     "json": value})
    if not rows:
        return pd.DataFrame({"body": pd.Series(dtype=BODY_DTYPE),
                             "json": pd.Series(dtype="object")})

    df = pd.DataFrame(rows)
    df["body"] = df["body"].astype(BODY_DTYPE)
    # `status` is left as a plain string on purpose. neuclease's own reader makes it an
    # ordered Categorical against a fixed Janelia vocabulary and *raises* on anything
    # unrecognised — so one newly-typed status would fail the whole fetch.
    for col in ("status", "instance", "user", "instance_user", "type"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    lead = ["body", "status", "instance", "type", "user", "instance_user"]
    rest = [c for c in df.columns if c not in lead and c != "json"]
    return df[[c for c in lead if c in df.columns] + rest + ["json"]].sort_values(
        "body", ignore_index=True)


# --------------------------------------------------------------------------- #
# the only coordinate-array builders
# --------------------------------------------------------------------------- #
def positions_zyx(df: pd.DataFrame, prefix: str = "") -> np.ndarray:
    """``(N, 3)`` array of positions in **zyx** order.

    With :func:`positions_xyz`, the only place a table's named coordinate columns become
    a positional array. Everything that needs one calls one of these and thereby states
    which order it meant.
    """
    return np.stack([df[f"{prefix}{a}"].to_numpy() for a in AXES], axis=-1)


def positions_xyz(df: pd.DataFrame, prefix: str = "") -> np.ndarray:
    """``(N, 3)`` array of positions in **xyz** order — DVID's and precomputed's."""
    return np.stack([df[f"{prefix}{a}"].to_numpy() for a in reversed(AXES)], axis=-1)
