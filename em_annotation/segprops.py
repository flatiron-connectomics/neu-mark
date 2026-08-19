"""Building a neuroglancer ``segment_properties`` source from the body annotations.

The format is one **inline** JSON document — there is no sharded form — so the whole thing
is a dict, and every write goes through ``em_volume_tools.location`` like the rest of this
package.

## What the spec allows, and what that forces

- **At most one property of type ``label``**, one ``description``, one ``tags``. Any number
  of ``number`` and ``string``. That single-``tags`` limit is the shape-defining constraint:
  every facet has to pool into one property, which is why tags carry a facet prefix
  (``side-l``, ``group-mi1``, ``col-c2``) rather than living in separate properties.
- ``tags`` values are **indices into that property's own ``tags`` array, in increasing
  order**. Not strings.
- A tag must contain **no spaces** and no leading ``#``, and matching is
  **case-insensitive** — so tags are lowercased and spaces hyphenated here, which also
  means two values differing only in case would collide and are folded deliberately.
- The ``description`` *member* must not appear on a ``tags`` property (it may on the
  others). Note ``description`` is both a member name and a property type; only the member
  is restricted.

## Why `label` is the raw instance string

Because it lets this ship before the ``cell_type`` question is settled. ``instance`` is
populated on 99.9% of annotated bodies and is the most informative single string available;
the curated ``type`` field, on only ~5% of the instance and ~13% of synapse-rich bodies,
goes in as a ``group-`` tag instead of competing to be the name. Nothing here parses a cell
type, so nothing here has to be revisited when that rule lands.

## Numbers, and one dtype that is not free to choose

``voxels`` **must be uint32**, not float32: float32 represents integers exactly only up to
2^24 (16,777,216), and real bodies here run to 86 million voxels, which would be silently
rounded. The synapse counts are small enough for either; they use uint32 too, for one rule.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: The subdirectory of a volume that holds this, and the `info` key naming it. Both are
#: fixed by the precomputed spec: the key's value is a subdirectory name.
SUBDIR = "segment_properties"

AT_TYPE = "neuroglancer_segment_properties"

#: Instance-string tokens that mean "not real data". A body whose instance carries one is
#: dropped from the output entirely rather than merely left untagged — it should not appear
#: in a viewer's segment list at all.
NOISE = ("irrelevant", "block", "chunk", "unknown")

#: Tracing-completeness markers. Kept and tagged, never used to exclude: they say how much
#: of a real neuron was traced, which is exactly what someone browsing wants to know.
COMPLETENESS = ("fragment", "truncated")

#: Single-token flags, mapped to the tag they produce. `nucleus` is the same thing as `NCL`
#: spelled differently; `CV` is the cervical connective.
FLAGS = {"ncl": "nucleated", "nucleus": "nucleated", "cv": "cervical", "glia": "glia"}

#: A column label: one or two uppercase letters then digits, optionally a trailing letter
#: (`C2`, `A1`, `MC1b`, `F9`). Multi-valued on purpose — some central complex neurons
#: innervate several, and tokens like `L8E8` name columns in different subregions.
COLUMN = re.compile(r"_([A-Z]{1,2}\d+[a-z]?)(?=$|[_(])")

SIDE = re.compile(r"\((L|R)\)")

#: Facet prefixes. Flags stay bare (`nucleated`, not `flag-nucleated`) because they read
#: better and cannot be confused with anything; everything with an open vocabulary is
#: prefixed so the tag list sorts into groups.
PREFIX = {"group": "group-", "side": "side-", "column": "col-"}

#: Shown in the viewer next to the tag, so a reader does not have to guess what `col-` is.
TAG_DESCRIPTIONS = {
    "group": "cell type, from the curated `type` field",
    "side": "hemisphere, from the (L)/(R) in the name",
    "column": "optic lobe / central complex column",
    "nucleated": "cell body (nucleus) is in the volume",
    "cervical": "passes through the cervical connective",
    "fragment": "a fragment, not a completely traced cell",
    "truncated": "traced but cut off by the volume boundary",
    "glia": "glia rather than a neuron",
}


def normalize_tag(value: Any) -> str | None:
    """A tag as the format requires: no spaces, no leading ``#``, lowercase.

    Lowercased because matching is case-insensitive, so ``Traced`` and ``traced`` are the
    same tag and keeping both would put two indistinguishable chips in the viewer.

    Missingness goes through :func:`em_annotation.explore.normalize`, which is the one place
    that knows ``pd.NA`` is neither ``None`` nor a float. Testing it by hand here is how
    16,606 bodies acquired a tag reading ``group-<na>``: ``pd.NA is not None`` is True and
    ``str(pd.NA)`` is the *truthy* string ``"<NA>"``.
    """
    from .explore import normalize

    text = normalize(value)
    if text is None:
        return None
    text = text.lstrip("#").strip()
    if not text:
        return None
    return re.sub(r"\s+", "-", text).lower()


def _tokens(instance: str | None) -> list[str]:
    from .explore import normalize

    return [t for t in re.split(r"[_)(]+", normalize(instance) or "") if t]


def is_noise(instance: str | None) -> str | None:
    """The noise token that disqualifies this body, if any."""
    lowered = [t.lower() for t in _tokens(instance)]
    return next((t for t in NOISE if t in lowered), None)


def facets(record: Mapping[str, Any]) -> dict[str, list[str]]:
    """Every facet for one body, as ``{facet: [values]}``, already tag-normalized.

    Deliberately does **not** derive a cell type from ``instance``: that is the open
    question, and this prototype avoids depending on its answer.
    """
    from .explore import normalize

    # `normalize`, not `str(...)`: a missing cell is `pd.NA`, which is not None and whose
    # str() is the truthy "<NA>". See normalize_tag.
    instance = normalize(record.get("instance"))
    text = instance or ""
    lowered = [t.lower() for t in _tokens(instance)]
    out: dict[str, list[str]] = {}

    if (m := SIDE.search(text)):
        out["side"] = [m.group(1)]
    if (kind := normalize(record.get("type"))):
        out["group"] = [kind]
    if (cols := COLUMN.findall(text)):
        out["column"] = list(dict.fromkeys(cols))
    for token in lowered:
        if token in FLAGS:
            out.setdefault("flag", []).append(FLAGS[token])
    for token in COMPLETENESS:
        if token in lowered:
            out.setdefault("completeness", []).append(token)

    cleaned: dict[str, list[str]] = {}
    for facet, values in out.items():
        prefix = PREFIX.get(facet, "")
        tags = [normalize_tag(prefix + str(v)) for v in values]
        kept = [t for t in dict.fromkeys(tags) if t]
        if kept:
            cleaned[facet] = kept
    return cleaned


def build(bodies, *, counts=None, sizes=None, keep_glia: bool = True) -> dict[str, Any]:
    """The ``segment_properties`` info document, plus a report of what went into it.

    ``bodies`` is a frame with ``body`` and whatever property fields exist; ``counts`` an
    optional frame with ``body``/``pre``/``post``/``syn``; ``sizes`` an optional
    ``{body: voxels}``.

    Returns ``{"info": …, "report": …}``. The report is not decoration: a facet that fired
    on 2% of bodies looks exactly like a facet the data does not have, and only a coverage
    number tells them apart.
    """
    import pandas as pd

    frame = bodies.copy()
    if "body" not in frame.columns:
        raise KeyError("bodies frame needs a 'body' column")

    from .explore import normalize

    excluded: dict[str, int] = {}
    rows = []
    for record in (r for _i, r in frame.iterrows()):
        instance = normalize(record.get("instance"))
        token = is_noise(instance)
        if token:
            excluded[token] = excluded.get(token, 0) + 1
            continue
        facet_values = facets(record)
        if not keep_glia and "glia" in facet_values.get("flag", []):
            excluded["glia"] = excluded.get("glia", 0) + 1
            continue
        rows.append((int(record["body"]), instance, facet_values))

    if not rows:
        raise ValueError("every body was excluded; nothing to write")

    ids = [str(body) for body, _inst, _f in rows]
    # `label` falls back to the id so that no segment is nameless in the viewer — a blank
    # label is indistinguishable from a missing property.
    labels = [str(inst) if inst is not None and str(inst).strip() else str(body)
              for body, inst, _f in rows]

    vocabulary: list[str] = []
    index_of: dict[str, int] = {}
    per_body: list[list[int]] = []
    coverage: dict[str, int] = {}
    for _body, _inst, facet_values in rows:
        indices = []
        for facet, values in facet_values.items():
            coverage[facet] = coverage.get(facet, 0) + 1
            for tag in values:
                if tag not in index_of:
                    index_of[tag] = len(vocabulary)
                    vocabulary.append(tag)
                indices.append(index_of[tag])
        # The spec requires the indices of one segment in increasing order.
        per_body.append(sorted(set(indices)))

    properties: list[dict[str, Any]] = [
        {"id": "instance", "type": "label", "values": labels},
        {"id": "tags", "type": "tags", "tags": vocabulary,
         # Parallel to `tags`, so every entry needs one; an unprefixed flag describes
         # itself, and a prefixed value takes its facet's description.
         "tag_descriptions": [_describe(tag) for tag in vocabulary],
         "values": per_body},
    ]

    numbers = _number_properties(ids, counts=counts, sizes=sizes)
    properties.extend(numbers)

    info = {"@type": AT_TYPE, "inline": {"ids": ids, "properties": properties}}
    report = {
        "bodies": len(ids),
        "excluded": excluded,
        "tags": len(vocabulary),
        "coverage": {facet: {"bodies": n, "fraction": n / len(ids)}
                     for facet, n in sorted(coverage.items())},
        "numbers": [p["id"] for p in numbers],
        "untagged": sum(1 for v in per_body if not v),
    }
    return {"info": info, "report": report}


def _describe(tag: str) -> str:
    for facet, prefix in PREFIX.items():
        if tag.startswith(prefix):
            return TAG_DESCRIPTIONS.get(facet, facet)
    return TAG_DESCRIPTIONS.get(tag, tag)


def _number_properties(ids: Sequence[str], *, counts=None, sizes=None) -> list[dict]:
    """`pre`/`post`/`syn`/`voxels` as uint32 number properties, for the ones supplied."""
    out: list[dict] = []
    order = [int(i) for i in ids]

    if counts is not None and len(counts):
        indexed = counts.set_index("body")
        for name, blurb in (("pre", "presynaptic sites (T-bars)"),
                            ("post", "postsynaptic sites (PSDs)"),
                            ("syn", "synapses in total, pre + post")):
            if name not in indexed.columns:
                continue
            series = indexed[name].reindex(order).fillna(0)
            out.append({"id": name, "type": "number", "data_type": "uint32",
                        "description": blurb,
                        "values": [int(v) for v in series]})

    if sizes:
        # uint32, NOT float32: float32 is exact only to 2^24 and real bodies here reach
        # 86 million voxels, which float32 would round.
        out.append({"id": "voxels", "type": "number", "data_type": "uint32",
                    "description": "size in voxels at full resolution",
                    "values": [int(sizes.get(b, 0)) for b in order]})
    return out


def write(dst: str, info: Mapping[str, Any], *, subdir: str = SUBDIR) -> str:
    """Write the document to ``<dst>/<subdir>/info`` through the kvstore."""
    from em_volume_tools.location import write_json

    write_json(dst, dict(info), subdir, "info")
    logger.info("wrote %s/%s/info (%d segments)", str(dst).rstrip("/"), subdir,
                len(info["inline"]["ids"]))
    return f"{subdir}/info"


def format_report(report: Mapping[str, Any]) -> list[str]:
    """The report as lines, for a CLI to print."""
    lines = [f"segments: {report['bodies']}   tags: {report['tags']}"]
    if report.get("excluded"):
        detail = ", ".join(f"{k} ({v})" for k, v in sorted(report["excluded"].items()))
        lines.append(f"excluded: {sum(report['excluded'].values())} — {detail}")
    for facet, stat in report.get("coverage", {}).items():
        lines.append(f"  {facet:<13} {stat['bodies']:>6} bodies ({stat['fraction']:.1%})")
    if report.get("numbers"):
        lines.append(f"  numbers: {', '.join(report['numbers'])}")
    if report.get("untagged"):
        lines.append(f"  {report['untagged']} segments carry no tag at all")
    return lines
