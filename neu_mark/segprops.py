"""Building a neuroglancer ``segment_properties`` source from the body annotations.

The format is one **inline** JSON document — there is no sharded form — so the whole thing
is a dict, and every write goes through ``neu_vol.location`` like the rest of this
package.

## What the spec allows, and what that forces

- **At most one property of type ``label``**, one ``description``, one ``tags``. Any number
  of ``number`` and ``string``. That single-``tags`` limit is the shape-defining constraint:
  every facet has to pool into one property, which is why tags carry a facet prefix
  (``side:L``, ``group:Mi1``) rather than living in separate properties.
  The separator is a **colon**, matching published sources, because it is what lets a
  reader tell a facet from a standalone flag — with a hyphen every tag becomes its own
  boolean and no facet can be grouped on.
- ``tags`` values are **indices into that property's own ``tags`` array, in increasing
  order**. Not strings.
- A tag must contain **no spaces** and no leading ``#``, and matching is
  **case-insensitive** — so spaces are hyphenated here. Case is **preserved**, because
  cell types are conventionally cased and folding them costs readability everywhere; the
  case-insensitive matching means two values differing only in case are one chip in the
  viewer, which :func:`case_collisions` reports rather than pre-empting.
- The ``description`` *member* must not appear on a ``tags`` property (it may on the
  others). Note ``description`` is both a member name and a property type; only the member
  is restricted.

## Where the facets come from

Every facet is a :class:`neu_mark.rules.Rule`. The defaults live in
:mod:`neu_mark.segprops_rules` and have no privileges: a dataset's own module is loaded
by path and **overrides them by name**, so ``side`` can be redefined, a ``glomerulus`` rule
added, and a ``drop`` rule can exclude a body outright. Nothing here parses a cell type —
that is dataset knowledge, and a rules module is where it goes.

## Why `label` is the raw instance string by default

Because it lets this ship before the ``cell_type`` question is settled. ``instance`` is
populated on 99.9% of annotated bodies and is the most informative single string available;
the curated ``type`` field, on only ~5% of the instance and ~13% of synapse-rich bodies,
goes in as a ``group:`` tag instead of competing to be the name. A rules module may declare
a reserved ``label`` rule (``tag=False``) to name segments itself; without one, the default
stands. The label is put through :func:`neu_mark.explore.normalize` and **not** through
:func:`normalize_tag` — hyphenating spaces is a *tag* constraint the format does not place
on a label, and applying it would rewrite ``LC10 anterior`` for no reason.

## The vocabulary guard

One ``tags`` property means one flat vocabulary array, **inline in a document every viewer
downloads whole**, so a rule that accidentally returns the entire instance string produces a
spec-legal source that is unusable and slow. ``max_tags`` takes either form — **below 1 a
fraction of the segment count, 1 or above an absolute ceiling** — and defaults to a fraction
because what counts as a lot depends on the dataset: 11,752 distinct ``type`` values over
~165k male-CNS bodies is 7%, and perfectly reasonable. See :func:`resolve_tag_limit`.

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

#: Joins a facet prefix to its value. **Load-bearing, not cosmetic.** It is what lets a
#: reader tell a *facet* (`side:L`, groupable) from a standalone flag (`fragment`, a
#: boolean); with a hyphen, `side-l` is indistinguishable from `fragment` and every tag
#: collapses into its own boolean — 483 of them on real data, with no `groupby` possible on
#: any facet. It also matches the convention published datasets use
#: (`superclass:ol_intrinsic`), so one reader handles both. A rule declaring ``prefix=""``
#: mints a bare tag, which is right for flags: they are not mutually exclusive — a body can
#: be nucleated *and* cervical — so there is no first value to prefer.
SEPARATOR = ":"

#: Cap on the tag vocabulary. **A fraction below 1, or an absolute count at 1 or above** —
#: see :func:`resolve_tag_limit`. The default is a fraction because the honest size depends
#: on the dataset: 11,752 distinct `type` values over ~165k male-CNS bodies is 7% and
#: entirely reasonable, while the same 11,752 over 21k bodies would mean a rule is echoing
#: the instance string back.
DEFAULT_MAX_TAGS = 0.10

#: Floor under the FRACTION branch only: on a short list every body can legitimately have
#: its own type, and 10% of 40 bodies is a meaningless threshold. An absolute count is a
#: statement rather than a ratio, so it is never raised to meet this.
MIN_TAG_ALLOWANCE = 100

#: Cap on distinct facets — the `side:`/`group:` namespaces, plus one for bare flags. This
#: is the *other* failure the fraction does not catch: the hyphen incident produced 483 tags
#: that were each their own field, which is a small vocabulary and an unusable layer.
MAX_FACETS = 24


def normalize_tag(value: Any) -> str | None:
    """A tag as the format requires: no spaces, no leading ``#``. **Case is preserved.**

    Cell types are conventionally cased (``Tm2``, ``LC10``, ``MBON``) and folding them to
    ``group:tm2`` costs readability in the viewer and in every group-by, for a collision
    that is rare and now *detected* rather than pre-empted: neuroglancer matches tags
    case-insensitively, so two values differing only in case would be one chip, and
    :func:`case_collisions` reports any before they are written. Published sources
    preserve case for the same reason.

    Missingness goes through :func:`neu_mark.explore.normalize`, which is the one place
    that knows ``pd.NA`` is neither ``None`` nor a float. Testing it by hand here is how
    16,606 bodies acquired a tag reading ``group:<na>``: ``pd.NA is not None`` is True and
    ``str(pd.NA)`` is the *truthy* string ``"<NA>"``.
    """
    from .explore import normalize

    text = normalize(value)
    if text is None:
        return None
    text = text.lstrip("#").strip()
    if not text:
        return None
    return re.sub(r"\s+", "-", text)


def case_collisions(vocabulary: Iterable[str]) -> dict[str, list[str]]:
    """``{lowercased: [tags]}`` for tags that differ only in case.

    Such tags are ONE chip in the viewer, because matching is case-insensitive — so a
    body carrying ``group:Tm2`` and another carrying ``group:TM2`` are indistinguishable
    there while remaining two values in the table. Reported rather than folded: folding
    is lossy and the collision is rare, so it is better to know which values did it.
    """
    seen: dict[str, list[str]] = {}
    for tag in vocabulary:
        seen.setdefault(tag.lower(), []).append(tag)
    return {low: tags for low, tags in seen.items() if len(tags) > 1}


def tag_for(rule, value: Any) -> str | None:
    """One rule's value as a tag: ``prefix:value``, or a bare tag where prefix is empty.

    The colon is added **here**, not by the rule, because ``Rule.prefix`` defaults to the
    rule's own name — so ``@rule def glomerulus`` yields ``glomerulus:VA1v`` with nothing
    declared, and an author who writes ``prefix="col:"`` out of habit does not get
    ``col::C2``.
    """
    prefix = getattr(rule, "prefix", "") or ""
    prefix = prefix.rstrip(SEPARATOR)
    text = f"{prefix}{SEPARATOR}{value}" if prefix else str(value)
    return normalize_tag(text)


def facets(record: Mapping[str, Any], ruleset=None) -> dict[str, list[str]]:
    """Every facet for one body, as ``{rule name: [tags]}``, already tag-normalized.

    Uses the built-in rules unless given a :class:`~neu_mark.rules.RuleSet`. Drop rules
    are **not** consulted here — exclusion is :func:`build`'s decision, and a function that
    silently returned ``{}`` for both "no facets" and "should not exist" would be the
    ambiguity this package keeps removing elsewhere.
    """
    from .explore import _plain
    from .rules import RESERVED

    if ruleset is None:
        # As-is when given, unlike `build`, which overrides the builtins with it: a caller
        # handing this one ruleset is asking what exactly that ruleset does.
        from .segprops_rules import BUILTIN

        ruleset = BUILTIN
    row = _plain(_as_series(record))
    out: dict[str, list[str]] = {}
    for r in ruleset.tagged:
        if r.name in RESERVED:
            continue
        tags = [tag_for(r, v) for v in r.values(row)]
        kept = [t for t in dict.fromkeys(tags) if t]
        if kept:
            out[r.name] = kept
    return out


def _as_series(record: Mapping[str, Any]):
    import pandas as pd

    return record if isinstance(record, pd.Series) else pd.Series(dict(record),
                                                                  dtype="object")


def _resolve_rules(rules=None, *, only: bool = False, keep_glia: bool = True):
    """The rules to run: builtins, or builtins **overridden by name**, or yours alone.

    Override rather than replace is the default because a module adding one glomerulus rule
    should not have to re-declare ``side`` and the flags to keep them. ``only=True`` is for
    a module that means to own the whole output.
    """
    from .rules import Rule, RuleSet
    from .segprops_rules import BUILTIN, DROP_GLIA

    if rules is None:
        resolved = BUILTIN
    elif only:
        resolved = rules if isinstance(rules, RuleSet) else RuleSet(rules)
    else:
        resolved = BUILTIN.override(rules)
    if not keep_glia:
        # An override, not a second exclusion path: after this, exactly one rule named
        # `glia` decides both whether a body is glia and what happens to it. Built from
        # WHATEVER rule currently answers that question — a module that redefined `glia`,
        # or replaced the builtins entirely, must not have --drop-glia quietly fall back to
        # this package's idea of what glia look like in its dataset.
        current = next((r for r in resolved if r.name == "glia"), None)
        dropper = DROP_GLIA if current is None else Rule(
            current.func, name="glia", drop=True, needs=current.needs,
            description=f"{current.description} (excluded)")
        resolved = resolved.override([dropper])
    return resolved


def build(bodies, *, rules=None, rules_only: bool = False, counts=None, sizes=None,
          keep_glia: bool = True, max_tags: float = DEFAULT_MAX_TAGS,
          max_facets: int = MAX_FACETS) -> dict[str, Any]:
    """The ``segment_properties`` info document, plus a report of what went into it.

    ``bodies`` is a frame with ``body`` and whatever property fields exist; ``counts`` an
    optional frame with ``body``/``pre``/``post``/``syn``; ``sizes`` an optional
    ``{body: voxels}``.

    ``rules`` is a :class:`~neu_mark.rules.RuleSet` (or anything it accepts) whose rules
    **override the builtins by name**; ``rules_only=True`` uses it alone. Evaluation order
    is: drop rules first, so an excluded body's tag rules never run and cannot skew the
    coverage numbers; then tag rules; then the reserved ``label``/``description`` rules;
    then any other ``tag=False`` rule, which is computed and reported but emits nothing.

    Returns ``{"info": …, "report": …}``. The report is not decoration: a facet that fired
    on 2% of bodies looks exactly like a facet the data does not have, and only a coverage
    number tells them apart.
    """
    import json

    from .explore import _plain, normalize
    from .rules import RESERVED

    frame = bodies.copy()
    if "body" not in frame.columns:
        raise KeyError("bodies frame needs a 'body' column")

    ruleset = _resolve_rules(rules, only=rules_only, keep_glia=keep_glia)

    # A declared column the records simply never carried is normal here, not an error:
    # `type` is populated on ~13% of bodies, so a body list where nobody has one has no
    # `type` column at all. Fill it and SAY so — `RuleSet.require` would refuse a
    # legitimate list, and leaving it out is the AttributeError `needs` exists to prevent.
    absent = sorted({c for r in ruleset for c in r.needs} - set(frame.columns))
    for column in absent:
        frame[column] = None
    if absent:
        logger.warning("the body records carry no %s column, so the rule(s) needing it "
                       "can never fire", ", ".join(absent))

    named = {r.name: r for r in ruleset}
    label_rule = named.get("label")
    description_rule = named.get("description")
    tag_rules = [r for r in ruleset.tagged if r.name not in RESERVED]
    quiet_rules = [r for r in ruleset
                   if not r.tag and not r.drop and r.name not in RESERVED]

    fired = {r.name: 0 for r in ruleset}
    distinct: dict[str, set] = {r.name: set() for r in ruleset}
    described: dict[str, str] = {}
    excluded: dict[str, int] = {}
    rows = []

    for _index, raw in frame.iterrows():
        row = _plain(raw)

        reason = None
        for r in ruleset.drops:
            got = r.scalar(row)
            if got:
                reason = got if isinstance(got, str) else r.name
                fired[r.name] += 1
                distinct[r.name].add(reason)
                break
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue

        tags: list[str] = []
        for r in tag_rules:
            minted = [t for t in dict.fromkeys(tag_for(r, v) for v in r.values(row)) if t]
            if not minted:
                continue
            fired[r.name] += 1
            distinct[r.name].update(minted)
            described.update(dict.fromkeys(minted, r.description))
            tags.extend(minted)

        for r in quiet_rules:
            values = r.values(row)
            if values:
                fired[r.name] += 1
                distinct[r.name].update(str(v) for v in values)

        # The label goes through `normalize` and NOT `normalize_tag`: hyphenating spaces is
        # a tag constraint, and a label is free of it.
        label = normalize(label_rule.scalar(row)) if label_rule is not None else None
        if label is not None:
            fired["label"] += 1
        else:
            label = normalize(row.get("instance"))

        description = (normalize(description_rule.scalar(row))
                       if description_rule is not None else None)
        if description is not None:
            fired["description"] += 1

        rows.append((int(raw["body"]), label, description, tags))

    if not rows:
        raise ValueError("every body was excluded; nothing to write")

    ids = [str(body) for body, _l, _d, _t in rows]
    # `label` falls back to the id so that no segment is nameless in the viewer — a blank
    # label is indistinguishable from a missing property.
    labels = [label if label else str(body) for body, label, _d, _t in rows]

    vocabulary: list[str] = []
    index_of: dict[str, int] = {}
    per_body: list[list[int]] = []
    for _body, _label, _desc, tags in rows:
        indices = []
        for tag in tags:
            if tag not in index_of:
                index_of[tag] = len(vocabulary)
                vocabulary.append(tag)
            indices.append(index_of[tag])
        # The spec requires the indices of one segment in increasing order.
        per_body.append(sorted(set(indices)))

    facet_names = sorted({t.split(SEPARATOR, 1)[0] if SEPARATOR in t else ""
                          for t in vocabulary})
    limit = check_vocabulary(vocabulary, distinct, segments=len(ids),
                             max_tags=max_tags, max_facets=max_facets,
                             facets=facet_names)

    properties: list[dict[str, Any]] = [
        {"id": "instance", "type": "label", "values": labels},
    ]
    if description_rule is not None:
        properties.append({"id": "description", "type": "description",
                           "values": [d or "" for _b, _l, d, _t in rows]})
    properties.append(
        {"id": "tags", "type": "tags", "tags": vocabulary,
         # Parallel to `tags`, so every entry needs one; it is the description of the rule
         # that minted it, which is why a bare flag can now describe itself properly.
         "tag_descriptions": [described.get(tag, tag) for tag in vocabulary],
         "values": per_body})

    numbers = _number_properties(ids, counts=counts, sizes=sizes)
    properties.extend(numbers)

    collisions = case_collisions(vocabulary)
    if collisions:
        logger.warning(
            "%d tag(s) differ only in case and will be ONE chip in the viewer: %s",
            len(collisions), sorted(v for vs in collisions.values() for v in vs)[:10])

    info = {"@type": AT_TYPE, "inline": {"ids": ids, "properties": properties}}
    report = {
        "bodies": len(ids),
        "excluded": excluded,
        "tags": len(vocabulary),
        # The RESOLVED ceiling and the setting that produced it. Both, because `0.1` and
        # `2111` are the same ceiling here and a reader should not have to re-derive which
        # branch ran — see `resolve_tag_limit`.
        "tag_limit": limit,
        "max_tags": max_tags,
        "facets": facet_names,
        # The whole document is inline and every viewer downloads all of it, so this is the
        # number that says what a big vocabulary actually costs the person opening the link.
        "bytes": len(json.dumps(info)),
        "case_collisions": {low: tags for low, tags in sorted(collisions.items())},
        "coverage": _coverage_report(ruleset, fired, distinct, segments=len(ids),
                                     total=len(frame)),
        "rules": {"source": ruleset.source, "names": ruleset.names},
        "absent_columns": absent,
        "numbers": [p["id"] for p in numbers],
        "untagged": sum(1 for v in per_body if not v),
    }
    return {"info": info, "report": report}


def _coverage_report(ruleset, fired: Mapping[str, int], distinct: Mapping[str, set], *,
                     segments: int, total: int) -> dict[str, dict]:
    """Per rule: how many bodies it fired on, and how many distinct values it produced.

    A **drop** rule's fraction is over the bodies that went IN, since the ones it fired on
    are by definition not among those that came out; every other fraction is over the
    segments written. Reporting both against one denominator is how a drop rule that fired
    on most of the input comes out at 250%.
    """
    out = {}
    for r in ruleset:
        n = fired.get(r.name, 0)
        denominator = total if r.drop else segments
        kind = "drop" if r.drop else "tag" if r.tag else "property"
        out[r.name] = {"bodies": n,
                       "fraction": (n / denominator) if denominator else 0.0,
                       "distinct": len(distinct.get(r.name, ())),
                       "kind": kind,
                       "description": r.description}
    return out


def resolve_tag_limit(max_tags: float, segments: int) -> tuple[int, str]:
    """``max_tags`` as a concrete tag ceiling: ``(limit, how)``.

    **Below 1 is a fraction of the segment count; 1 or above is an absolute count.** The
    branch is chosen by VALUE, never by whether the number was written with a decimal point:
    ``1`` and ``1.0`` are the same number to TOML and to JSON, so a setting that meant "all
    of them" would come back from a config or a provenance record meaning "one tag". Nothing
    is lost by the rule, since a fraction of 1.0 and an absolute count of ``segments`` are
    the same ceiling.

    :data:`MIN_TAG_ALLOWANCE` floors the **converted fraction only**. An absolute count is a
    statement rather than a ratio — ``max_tags=25`` means 25, and raising it to meet a floor
    would make the setting not do what it says.
    """
    if not max_tags > 0:
        raise ValueError(
            f"max_tags must be positive; got {max_tags!r}. Below 1 is a fraction of the "
            f"segment count, 1 or above an absolute number of tags.")
    if max_tags < 1:
        return max(MIN_TAG_ALLOWANCE, int(max_tags * segments)), "fraction"
    return int(max_tags), "count"


def check_vocabulary(vocabulary: Sequence[str], distinct: Mapping[str, set], *,
                     segments: int, max_tags: float = DEFAULT_MAX_TAGS,
                     max_facets: int = MAX_FACETS,
                     facets: Sequence[str] | None = None) -> int:
    """Refuse a vocabulary big enough to make the layer unusable. Returns the limit applied.

    Two different failures, so two checks. **Too many values** is a rule echoing the
    instance string back — spec-legal, and it produces a chip per body plus a document every
    viewer downloads whole. **Too many facets** is the hyphen incident: 483 tags that were
    each their own field, a small vocabulary and an unusable layer.

    See :func:`resolve_tag_limit` for how ``max_tags`` becomes a number.
    """
    if facets is None:
        facets = sorted({t.split(SEPARATOR, 1)[0] if SEPARATOR in t else ""
                         for t in vocabulary})
    if len(facets) > max_facets:
        raise ValueError(
            f"{len(facets)} distinct tag facets, over the {max_facets} allowed: "
            f"{', '.join(repr(f) for f in list(facets)[:12])}…. A facet per value means "
            f"nothing can be grouped on — check that the rules minting these meant to "
            f"declare prefix=\"\" for a bare flag rather than a namespace each.")

    limit, how = resolve_tag_limit(max_tags, segments)
    if len(vocabulary) > limit:
        worst, count = max(((n, len(v)) for n, v in distinct.items()),
                           key=lambda kv: kv[1], default=("", 0))
        # Say WHICH branch produced the number, so it is clear whether the floor was in play.
        applied = (f"{max_tags:.0%} of {segments}, floored at {MIN_TAG_ALLOWANCE}"
                   if how == "fraction" else f"an explicit count of {limit}")
        raise ValueError(
            f"{len(vocabulary)} tags for {segments} segments, over the limit of {limit} "
            f"({applied}). The rule producing most of them is {worst!r} with {count} "
            f"distinct values — if it is returning the whole instance string, that is the "
            f"bug; if the dataset really is this diverse, raise max_tags.")
    return limit


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
    from neu_vol.location import write_json

    write_json(dst, dict(info), subdir, "info")
    logger.info("wrote %s/%s/info (%d segments)", str(dst).rstrip("/"), subdir,
                len(info["inline"]["ids"]))
    return f"{subdir}/info"


def format_report(report: Mapping[str, Any]) -> list[str]:
    """The report as lines, for a CLI to print."""
    head = f"segments: {report['bodies']}   tags: {report['tags']}"
    if report.get("tag_limit"):
        head += f" of {report['tag_limit']} allowed"
    if report.get("bytes"):
        head += f"   document: {report['bytes'] / 1024:,.0f} KiB"
    lines = [head]
    if (report.get("rules") or {}).get("source"):
        lines.append(f"rules: {report['rules']['source']}")
    if report.get("absent_columns"):
        lines.append(f"absent from the records: {', '.join(report['absent_columns'])} — "
                     f"rules needing them cannot fire")
    if report.get("excluded"):
        detail = ", ".join(f"{k} ({v})" for k, v in sorted(report["excluded"].items()))
        lines.append(f"excluded: {sum(report['excluded'].values())} — {detail}")
    for name, stat in report.get("coverage", {}).items():
        kind = stat.get("kind", "tag")
        suffix = "" if kind == "tag" else f"   [{kind}]"
        lines.append(f"  {name:<13} {stat['bodies']:>7} bodies ({stat['fraction']:>6.1%})"
                     f"  {stat.get('distinct', 0):>5} distinct{suffix}")
    if report.get("numbers"):
        lines.append(f"  numbers: {', '.join(report['numbers'])}")
    if report.get("untagged"):
        lines.append(f"  {report['untagged']} segments carry no tag at all")
    return lines
