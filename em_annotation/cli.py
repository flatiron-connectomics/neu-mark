"""``em-annot`` — the command line over em_annotation.

Argparse, matching its siblings (`em-vol`, `em-morpho`) rather than introducing a second
CLI framework into the family. Heavy imports stay inside the subcommand that needs them so
`--help` is fast.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from . import __version__

#: Repeated verbatim in both fetch subcommands, since `--src` is where the URL structure
#: gets explained and it is the thing users are least likely to guess.
# Flowing prose, not an aligned table: argparse re-wraps `help=` whatever formatter the
# parser uses (RawDescriptionHelpFormatter only spares description/epilog), so the columns
# never survived to the terminal — and in the docs the unindented line after an indented
# block made docutils read it as a broken definition list, eight warnings' worth.
_SRC_HELP = (
    "DVID source as dvid://SERVER/UUID/INSTANCE, always three segments. "
    "SERVER may carry a port (emdata3:8900); a bare host gets http://, so use "
    "dvid+https://HOST/... for TLS. UUID may be a node uuid (93fdbc, abbreviated is "
    "fine) or a repo:branch ref (93fdbc:main), meaning HEAD of that branch. INSTANCE is "
    "the data instance name. A ':' in the server or the uuid adds no segment — this "
    "splits on '/' alone. Alternatively '@name' looks the instance up in your config "
    "(em_annotation.config), and the resolved URL is printed so the command still says "
    "what it read. "
)


def _add_common(q: argparse.ArgumentParser, *, instance_hint: str,
                bodies_required: bool = True) -> None:
    q.add_argument("--src", required=True, metavar="URL",
                   help=_SRC_HELP + f"For this command INSTANCE is {instance_hint}.")
    q.add_argument("--out", required=True, metavar="DST",
                   help="where the tables go — a local directory or s3://bucket/prefix. "
                        "May contain {uuid}, {uuid:N}, {uuid:full}, {branch} or "
                        "{instance}, expanded from the resolved node.")
    q.add_argument("--bodies", required=bodies_required, metavar="IDS", default=None,
                   help="body ids: inline (123,456), or a path to a .csv/.parquet/"
                        ".feather with a body column, or a text file with one id per "
                        "line." + ("" if bodies_required else " Omit it only with --all.")
                        + (" Required — DVID cannot cheaply enumerate the bodies worth "
                           "asking about." if bodies_required else ""))
    q.add_argument("--body-column", metavar="NAME", default=None,
                   help="which column holds the ids, when --bodies is a table with "
                        "several and none is named recognisably.")
    q.add_argument("--format", dest="fmt", default="parquet",
                   choices=("parquet", "csv", "feather"),
                   help="table format (default: parquet, which is the only one that "
                        "preserves uint64 body ids; csv warns about it).")
    q.add_argument("--dvid-locked", action="store_true",
                   help="read the newest LOCKED ancestor rather than what the ref points "
                        "at now. A branch HEAD is normally an open, still-changing node, "
                        "so a pull from it is not reproducible.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="em-annot",
        description="Fetch DVID annotations into tables, and publish them to "
                    "neuroglancer.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"em-annotation {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser(
        "points", help="point annotations (synapses) for a set of bodies",
        description="Fetch every element of every listed body from a DVID `annotation` "
                    "instance, and write two tables: `points` (one row per element) and "
                    "`relationships` (one row per relationship, with the partner's body "
                    "resolved where it was fetched).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(q, instance_hint="an annotation instance, e.g. 'synapses'")
    q.add_argument("--threads", type=int, default=None, metavar="N",
                   help="concurrent requests (default 8). DVID is a shared service and "
                        "answers overload with 503; raise this deliberately.")
    q.add_argument("--drop-unmatched", action="store_true",
                   help="drop relationships whose partner body was not resolved. Off by "
                        "default: an unresolved partner and a dangling reference look "
                        "identical, so read the reported match rate first.")
    q.add_argument("--connections", action="store_true",
                   help="also write a `connections` table: the oriented, de-duplicated "
                        "(tbar, psd) pairs derived from `relationships`.")
    q.add_argument("--rois", metavar="LIST", default=None,
                   help="label each synapse with the neuropil it falls in, adding a `roi` "
                        "column to the points table. A comma-separated list of DVID roi "
                        "instance names, a path to a file with one per line, or '@name' for "
                        "a set from your config. There is deliberately NO default: the "
                        "combined ROI volume is built by overwriting, so 'all of them' "
                        "would attribute a point to whichever containing ROI came last.")
    q.add_argument("--strict-rois", action="store_true",
                   help="refuse if the chosen ROIs overlap. By default overlap is allowed "
                        "and MEASURED: the later ROI wins, and the report says how many of "
                        "your synapses sit in an intersection and are therefore attributed "
                        "by ROI order alone. Use this if you need a strict partition — the "
                        "dataset has subtracted variants (INP(-ATL)(L), PENP(-AMMC)) "
                        "for that.")
    q.set_defaults(func=cmd_points)

    q = sub.add_parser(
        "bodies", help="per-body annotations (name, status), for a body list or all of them",
        description="Fetch the per-body records from a DVID `keyvalue` instance and write "
                    "a `bodies` table, one row per body. Records are ragged, so the table "
                    "is the union of the fields present.\n\n"
                    "--all reads the WHOLE instance, which is a different population and "
                    "often the one you want: a >=10-synapse selection holds 117 glia while "
                    "the instance holds 1,014, because most glia sit below any synapse "
                    "threshold. It reads a partition of the key space rather than asking "
                    "for the key list, which is unreliable at this size.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(q, instance_hint="a keyvalue instance, e.g. 'labels_annotations'",
                bodies_required=False)
    q.add_argument("--all", dest="everything", action="store_true",
                   help="every record in the instance, instead of a body list. Mutually "
                        "exclusive with --bodies.")
    q.set_defaults(func=cmd_bodies)

    q = sub.add_parser(
        "select-bodies",
        help="choose which bodies are worth fetching, by synapse count",
        description="Rank bodies by synapse count using DVID's own `labelsz` index and "
                    "write the list, for feeding to --bodies. Seconds, against ~70 "
                    "minutes to scan every label's voxel size — and synapse count is the "
                    "better criterion anyway: a body with no synapses is a fragment, and "
                    "many of the largest-by-volume bodies are glia.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--src", required=True, metavar="URL",
                   help=_SRC_HELP + "For this command INSTANCE is a labelsz instance "
                        "(e.g. 'synapses_labelsz') or the annotation instance it indexes "
                        "(e.g. 'synapses'), in which case its labelsz is found via Syncs.")
    q.add_argument("--out", required=True, metavar="DST",
                   help="destination directory for the list; writes selected_bodies.<fmt> "
                        "plus a provenance record naming the node it came from. Accepts "
                        "the same {uuid}/{branch}/{instance} placeholders.")
    q.add_argument("--min-synapses", type=int, default=10, metavar="N",
                   help="keep bodies with at least this many synapses in TOTAL "
                        "(default 10). Total rather than per-type on purpose: sensory "
                        "neurons may have no postsynapses, and a neuron leaving the "
                        "traced volume may have no presynapses.")
    q.add_argument("--min-pre", type=int, default=0, metavar="N",
                   help="additionally require at least N presynapses. Off by default — it "
                        "excludes neurons that only receive within the traced volume.")
    q.add_argument("--min-post", type=int, default=0, metavar="N",
                   help="additionally require at least N postsynapses. Off by default — it "
                        "excludes sensory neurons.")
    q.add_argument("--limit", type=int, default=None, metavar="N",
                   help="keep only the top N after filtering. Results are ranked by total "
                        "synapses, so this is how to ask for 'the biggest N'.")
    q.add_argument("--format", dest="fmt", default="csv",
                   choices=("csv", "parquet", "feather"),
                   help="list format (default: csv — a body list is meant to be read and "
                        "edited by hand, and every column here is a non-nullable integer, "
                        "which csv does preserve).")
    q.add_argument("--dvid-locked", action="store_true",
                   help="rank from the newest LOCKED ancestor. Worth doing: body ids "
                        "change with proofreading, so the list and the annotations should "
                        "come from the same node.")
    q.set_defaults(func=cmd_select_bodies)

    q = sub.add_parser(
        "segment-properties",
        help="a neuroglancer segment_properties source: names, tags and counts",
        description="Build a `segment_properties` source from the per-body records and "
                    "write it INTO a precomputed segmentation volume, so one neuroglancer "
                    "layer shows each body's name, its tags and its synapse counts.\n\n"
                    "`label` is the raw `instance` string — which is what lets this run "
                    "before a cell-type parse is settled. Tags carry a facet prefix "
                    "(group-, side-, col-) because the format allows only ONE tags "
                    "property, so every facet has to pool into it.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--src", required=True, metavar="URL",
                   help=_SRC_HELP + "For this command INSTANCE is the keyvalue instance "
                        "holding the per-body records, e.g. 'labels_annotations'.")
    q.add_argument("--dst", required=True, metavar="VOLUME",
                   help="the precomputed SEGMENTATION VOLUME to write into (local or "
                        "s3://…). The document lands at <dst>/segment_properties/info, and "
                        "the volume's own info gains the key pointing at it. Accepts the "
                        "same {uuid}/{branch}/{instance} placeholders.")
    q.add_argument("--bodies", required=True, metavar="IDS",
                   help="body ids: inline, or a .csv/.parquet/.feather with a body column, "
                        "or a text file with one per line.")
    q.add_argument("--body-column", metavar="NAME", default=None,
                   help="which column holds the ids, if the table has several.")
    q.add_argument("--counts", metavar="URL", default=None,
                   help="a labelsz instance (or the annotation instance it indexes) to add "
                        "`pre`, `post` and `syn` number properties. Cheap.")
    q.add_argument("--labelmap", metavar="URL", default=None,
                   help="a labelmap instance to add a `voxels` number property, from "
                        "DVID's own /sizes. ~1 minute for 20k bodies.")
    q.add_argument("--drop-glia", action="store_true",
                   help="exclude glia-labelled bodies. By default they are KEPT and carry "
                        "a `glia` tag — not connectome-relevant, but worth seeing.")
    q.add_argument("--no-link", dest="link", action="store_false",
                   help="write the source WITHOUT adding the key to the volume's info. Use "
                        "this to inspect the output before touching a published volume; "
                        "the linking step is what makes a viewer associate the two.")
    q.add_argument("--dvid-locked", action="store_true",
                   help="read the newest LOCKED ancestor rather than the branch HEAD.")
    q.set_defaults(func=cmd_segment_properties, link=True)

    q = sub.add_parser(
        "annotation-source",
        help="a neuroglancer precomputed annotation source: one LINE per connection",
        description="Write a `neuroglancer_annotations_v1` LINE source, one line from each "
                    "presynaptic site to each of its partners. Lines, not points, because a "
                    "connectome is edges — and one endpoint pair is one annotation, so a "
                    "T-bar with five partners contributes five.\n\n"
                    "Input is the tables an earlier `points` run wrote (`--tables`), which "
                    "is the cheap path: the tables are the durable artifact and a rebuild "
                    "with different bounds or sharding need not refetch. Give `--src` and "
                    "`--bodies` instead to fetch first.\n\n"
                    "This is a SEPARATE source from the segmentation — unlike mesh and "
                    "skeletons, an annotation layer cannot be named from a volume's info, so "
                    "a viewer adds it as its own layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--dst", required=True, metavar="PATH",
                   help="where the source goes (local or s3://…). Accepts the same "
                        "{uuid}/{branch}/{instance} placeholders when --src is given.")
    q.add_argument("--tables", metavar="DIR", default=None,
                   help="a directory an earlier `points` run wrote, holding "
                        "connections.parquet and points.parquet.")
    q.add_argument("--src", metavar="URL", default=None,
                   help=_SRC_HELP + "Only needed to fetch now instead of using --tables.")
    q.add_argument("--bodies", metavar="IDS", default=None,
                   help="body ids to fetch, when --src is given rather than --tables.")
    q.add_argument("--body-column", metavar="NAME", default=None,
                   help="which column holds the ids, if the table has several.")
    q.add_argument("--voxel-size", metavar="X,Y,Z", default="8,8,8",
                   help="nanometres per voxel of the frame the positions are in. Annotation "
                        "coordinates are stored in VOXELS, and this is what puts them in the "
                        "same physical space as the segmentation (default %(default)s).")
    q.add_argument("--bounds", metavar="X0,Y0,Z0,X1,Y1,Z1", default=None,
                   help="the source's lower and upper bound in voxels. Defaults to the "
                        "data's own extent, padded by one voxel.")
    q.add_argument("--drop-partial", dest="include_partial", action="store_false",
                   help="drop lines with only one endpoint's body resolved. By default they "
                        "are KEPT and undifferentiated: the partner is a real synapse whose "
                        "body simply was not in the fetched set, and the line is still where "
                        "the synapse is.")
    q.add_argument("--per-cell", type=int, default=4_000, metavar="N",
                   help="annotations a spatial cell aims to hold, which sets how many levels "
                        "the grid needs and how much a zoomed-out view downloads "
                        "(default %(default)s).")
    q.add_argument("--threads", type=int, default=None, metavar="N",
                   help="concurrent DVID requests, when fetching rather than using --tables.")
    q.add_argument("--dvid-locked", action="store_true",
                   help="read the newest LOCKED ancestor rather than the branch HEAD.")
    q.add_argument("--verify", type=int, nargs="?", const=200, default=None, metavar="N",
                   help="after writing, fetch N annotations back THROUGH the source's own "
                        "info and compare them with the table. Everything before the write "
                        "can be checked in memory; only a read-back proves the keys are "
                        "right, and a wrong key leaves a viewer with nothing while every "
                        "byte on the store is correct.")
    q.add_argument("--dry-run", action="store_true",
                   help="report the plan — lines, stride, levels, shards — write nothing.")
    q.set_defaults(func=cmd_annotation_source, include_partial=True)

    q = sub.add_parser(
        "info", help="what a DVID annotation source is, and which node you would get",
        description="One request per fact: the instance type, what it is synced to, and "
                    "both candidate nodes for the ref (what it points at now, and the "
                    "newest locked one).")
    q.add_argument("--src", required=True, metavar="URL", help=_SRC_HELP)
    q.set_defaults(func=cmd_info)

    return p


def _parse_args(argv=None):
    return build_parser().parse_args(argv)


def _resolve_src(value: str) -> str:
    """An ``@name`` config reference as a URL; anything else unchanged."""
    from . import config as _config

    if isinstance(value, str) and value.startswith(_config.REFERENCE_PREFIX):
        resolved = _config.load().resolve(value)
        print(f"{value}  ->  {resolved}")
        return resolved
    return value


def _src_spec(src: str) -> dict[str, Any]:
    """Resolve ``--src`` to a spec, expanding an ``@name`` config reference.

    A reference is **printed** when it resolves, for the same reason a ``{uuid}``
    placeholder in ``--out`` is: a command whose meaning depends on an uncommitted file
    should say what that file turned it into. The provenance record stores the resolved URL
    regardless, so a config cannot make an export less traceable.
    """
    from em_volume_tools.dvid import is_url, parse_url

    from . import config as _config

    if isinstance(src, str) and src.startswith(_config.REFERENCE_PREFIX):
        resolved = _config.load().resolve(src)
        print(f"--src {src}  ->  {resolved}")
        src = resolved

    if not is_url(src):
        raise SystemExit(
            f"--src {src!r} is not a DVID URL. This command reads annotations from DVID, "
            f"which is addressed as dvid://server/uuid/instance. A leading "
            f"{_config.REFERENCE_PREFIX!r} looks the name up in your config instead.")
    return {"backend": "dvid", **parse_url(src)}


def _expand_out(out: str, source: dict[str, Any]) -> str:
    """Resolve {uuid}/{branch}/{instance} in the destination.

    Takes the **already-resolved** source, so ``{uuid}`` is the node the data will
    actually be read from. Expanding against an unresolved spec would name the output
    after the branch HEAD even under ``--dvid-locked``, i.e. after a node the tables did
    not come from — and the path is what someone browsing a directory believes.
    """
    from em_volume_tools.ops.naming import expand, has_placeholder

    if not has_placeholder(out):
        return out
    resolved = expand(out, source)
    print(f"--out {out}  ->  {resolved}")
    return resolved


def _load_bodies(args) -> list[int]:
    from . import bodies as _bodies

    ids = _bodies.load(args.bodies, column=args.body_column)
    print(f"bodies: {_bodies.summarise(ids)}")
    return ids


def _roi_list(value: str | None) -> list[str] | None:
    """``--rois`` as a name list: a comma list, a file of names, or an ``@name`` config set."""
    import os

    if not value:
        return None
    from . import notebook as _nb

    if not value.startswith("@") and os.path.exists(value):
        with open(value) as fh:
            return [line.split("#", 1)[0].strip() for line in fh
                    if line.split("#", 1)[0].strip()]
    resolved = _nb.roi_set(value)
    if value.startswith("@"):
        print(f"--rois {value}  ->  {len(resolved)} ROIs")
    return resolved


def _report_rois(stats: dict | None) -> None:
    """How many synapses landed in a neuropil. A large unlabelled share means the wrong set."""
    if not stats:
        return
    total = stats["labeled"] + stats["unlabeled"]
    if not total:
        return
    print(f"rois: {stats['labeled']} of {total} synapses inside one of "
          f"{len(stats['rois'])} ROIs ({stats['labeled'] / total:.1%}), "
          f"{stats['unlabeled']} outside every one")
    top = sorted(stats["counts"].items(), key=lambda kv: -kv[1])[:5]
    if top:
        print("  " + ", ".join(f"{name} ({n})" for name, n in top))
    if stats.get("overlapping"):
        n = stats.get("ambiguous") or 0
        print(f"  {len(stats['overlapping'])} ROI pairs intersect; {n} synapse(s) "
              f"({n / total:.2%}) sit in an intersection and are attributed by ROI ORDER "
              f"alone" + (f" — {stats.get('ambiguous_pairs')}" if n else ""))
    if stats["unlabeled"] > stats["labeled"]:
        print("  NOTE: most synapses fell outside every ROI given. That is a statement "
              "about the ROI set, not the data — check it covers the traced volume.")


def _report_match(match: dict) -> None:
    """The headline number. A low fraction means the body list is too small."""
    if not match.get("pairs"):
        return
    frac = match["fraction"]
    print(f"connections: {match['pairs']} distinct (tbar, psd) pairs, "
          f"{match['both_ends']} with both bodies resolved ({frac:.1%}), "
          f"{match['one_end']} with one")
    if frac is not None and frac < 0.5:
        print(f"  NOTE: only {frac:.1%} of pairs have both endpoints in the body list. "
              f"A relationship resolves only when BOTH bodies were fetched, so this is "
              f"a statement about the body list's coverage, not about the data. Widen "
              f"--bodies for connectivity work.")


def cmd_points(args) -> int:
    from . import ops

    # Resolve the node FIRST: the destination name derives from it, and so does every
    # request. One resolution, one answer.
    source = ops.open_points_source(_src_spec(args.src),
                                    prefer_locked=bool(args.dvid_locked))
    out = _expand_out(args.out, source)
    ids = _load_bodies(args)

    kwargs = {} if args.threads is None else {"threads": args.threads}
    result = ops.fetch_points(source, out, ids, fmt=args.fmt,
                              drop_unmatched=bool(args.drop_unmatched),
                              rois=_roi_list(args.rois),
                              on_roi_overlap="error" if args.strict_rois else "warn",
                              write_connections=bool(args.connections), **kwargs)
    print(f"points: {len(result['points'])} elements, "
          f"{len(result['relationships'])} relationships")
    _report_rois(result.get("rois"))
    _report_match(result["match"])
    if result["failures"]:
        print(f"  {len(result['failures'])} bodies failed and were skipped; "
              f"first few: {list(result['failures'])[:5]}", file=sys.stderr)
    print(f"wrote {', '.join(result['written'])} to {out}")
    return 0


def cmd_bodies(args) -> int:
    from . import ops

    everything = bool(getattr(args, "everything", False))
    if everything and args.bodies:
        raise SystemExit(
            "--all and --bodies describe different populations; pass one, not both.")
    if not everything and not args.bodies:
        raise SystemExit(
            "--bodies is required, or pass --all to read the whole instance.")

    source = ops.open_bodies_source(_src_spec(args.src),
                                    prefer_locked=bool(args.dvid_locked))
    out = _expand_out(args.out, source)
    ids = None if everything else _load_bodies(args)

    result = ops.fetch_bodies(source, out, ids, fmt=args.fmt, everything=everything)
    if everything:
        print(f"bodies: {result['found']} records across "
              f"{len(result['ranges'])} key ranges")
    else:
        print(f"bodies: {result['found']} of {result['requested']} had a record")
        if result["missing"]:
            print(f"  {len(result['missing'])} had none (not an error — DVID simply has no "
                  f"annotation for them)")
    print(f"wrote {', '.join(result['written'])} to {out}")
    return 0


def cmd_select_bodies(args) -> int:
    from . import ops

    source = ops.open_counts_source(_src_spec(args.src),
                                    prefer_locked=bool(args.dvid_locked))
    out = _expand_out(args.out, source)
    result = ops.select_bodies(source, out, min_synapses=args.min_synapses,
                               min_pre=args.min_pre, min_post=args.min_post,
                               limit=args.limit, fmt=args.fmt)
    df = result["bodies"]
    print(f"selected {len(df)} bodies (>= {args.min_synapses} synapses"
          + (f", >= {args.min_pre} pre" if args.min_pre else "")
          + (f", >= {args.min_post} post" if args.min_post else "") + ")")
    if len(df):
        print(f"  synapses per body: {int(df['syn'].min())}..{int(df['syn'].max())}, "
              f"{int(df['syn'].sum())} total")
    listed = f"{out.rstrip('/')}/{result['written'][0]}"
    print(f"wrote {result['written'][0]} to {out}")
    print(f"  feed it onward with:  --bodies {listed}")
    return 0


def cmd_segment_properties(args) -> int:
    from . import ops, segprops

    bodies_src = ops.open_bodies_source(_src_spec(args.src),
                                        prefer_locked=bool(args.dvid_locked))
    counts_src = (ops.open_counts_source(_src_spec(args.counts),
                                         prefer_locked=bool(args.dvid_locked))
                  if args.counts else None)
    labelmap_src = None
    if args.labelmap:
        from em_volume_tools.dvid import parse_url
        labelmap_src = {"backend": "dvid", **parse_url(_resolve_src(args.labelmap))}
        labelmap_src["uuid"] = bodies_src["uuid"]      # the SAME node, always

    dst = _expand_out(args.dst, bodies_src)
    ids = _load_bodies(args)

    result = ops.segment_properties(
        bodies_src, dst, ids, counts_source=counts_src, labelmap_source=labelmap_src,
        keep_glia=not args.drop_glia, link=bool(args.link))

    for line in segprops.format_report(result["report"]):
        print(line)
    print(f"wrote {', '.join(result['written'])} to {dst}")
    if result["linked"]:
        print(f"linked into {dst}/info: {result['linked']}")
    else:
        print("NOT linked — the volume's info is untouched, so a viewer will not associate "
              "these with the labels layer. Re-run without --no-link when you are happy.")
    return 0


def _triple(value: str, name: str, n: int = 3) -> list[float]:
    parts = [p for p in value.replace(" ", "").split(",") if p]
    if len(parts) != n:
        raise SystemExit(f"{name} wants {n} comma-separated numbers, got {len(parts)}: "
                         f"{value!r}")
    return [float(p) for p in parts]


def cmd_annotation_source(args) -> int:
    from . import annsource, ops

    if bool(args.tables) == bool(args.src):
        raise SystemExit("pass --tables (reuse an earlier fetch) or --src plus --bodies "
                         "(fetch now), not both and not neither.")

    voxel_size = _triple(args.voxel_size, "--voxel-size")
    bounds = None
    if args.bounds:
        b = _triple(args.bounds, "--bounds", 6)
        bounds = (b[:3], b[3:])

    kwargs: dict[str, Any] = {}
    if args.tables:
        from . import io as _io

        conns, points = _io.read_tables(args.tables, ("connections", "points"))
        dst = args.dst
        kwargs.update(connections=conns, points=points)
        print(f"tables: {len(conns):,} connections, {len(points):,} elements "
              f"from {args.tables}")
    else:
        if not args.bodies:
            raise SystemExit("--src needs --bodies: DVID cannot cheaply enumerate the "
                             "bodies worth asking about.")
        source = ops.open_points_source(_src_spec(args.src),
                                        prefer_locked=bool(args.dvid_locked))
        dst = _expand_out(args.dst, source)
        kwargs.update(source=source, bodies=_load_bodies(args))
        if args.threads is not None:
            kwargs["threads"] = args.threads

    if args.dry_run:
        # Everything except the write, so the plan reported is the plan that would run.
        import numpy as np

        from . import tables as _t
        if "connections" not in kwargs:
            raise SystemExit("--dry-run needs --tables; there is nothing to plan before "
                             "the fetch has happened.")
        frame = _t.enrich_connections(kwargs["connections"], kwargs["points"])
        if bounds is None:
            corners = np.vstack([_t.positions_xyz(frame, "pre_"),
                                 _t.positions_xyz(frame, "post_")]).astype(float)
            bounds = (corners.min(axis=0), corners.max(axis=0) + 1.0)
        built = annsource.build(frame, lower_bound=bounds[0], upper_bound=bounds[1],
                                voxel_size_xyz=voxel_size, per_cell=args.per_cell,
                                include_partial=bool(args.include_partial))
        for line in annsource.format_report(built["report"]):
            print(line)
        print(f"would write to {dst} (nothing written)")
        return 0

    result = ops.annotation_source(dst, voxel_size_xyz=voxel_size, bounds=bounds,
                                   include_partial=bool(args.include_partial),
                                   per_cell=args.per_cell, **kwargs)
    for line in annsource.format_report(result["report"]):
        print(line)
    print(f"wrote {', '.join(result['written'])} to {dst}")

    if args.verify:
        check = annsource.verify(dst, result["table"], sample=args.verify)
        body = check["body_check"]
        print(f"verified {check['sampled']} annotations read back through the source's info")
        if body:
            print(f"  body {body['body']}: {body['found']} lines in the pre index, "
                  f"table says {body['expected']}")
        if check["problems"]:
            print(f"{len(check['problems'])} PROBLEMS — the source is not what the table "
                  f"says:", file=sys.stderr)
            for problem in check["problems"][:10]:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print("  no problems")

    print("add it in neuroglancer as its own layer — an annotation source cannot be named "
          "from the segmentation's info the way mesh and skeletons are.")
    return 0


def cmd_info(args) -> int:
    from em_volume_tools.dvid import (instance_info, instance_type, node_summary,
                                     synced_instances)

    src = _src_spec(args.src)
    info = instance_info(src)
    kind = instance_type(info)
    synced = synced_instances(info)
    print(f"instance : {src['instance']}  ({kind})")
    print(f"synced to: {', '.join(synced) if synced else '(nothing)'}")
    if kind == "annotation" and not synced:
        print("  WARNING: an unsynced annotation instance cannot be read per body — "
              "DVID's /label endpoint is built from the sync.")

    summary = node_summary(src)
    head = summary["head"]
    print(f"latest      : {head['uuid']}  "
          f"({'locked' if head['locked'] else 'OPEN — still being written'})")
    locked = summary["locked"]
    if locked is None:
        print(f"latest lock : unavailable ({summary['locked_error']})")
    elif locked["uuid"] == head["uuid"]:
        print("latest lock : same node")
    else:
        print(f"latest lock : {locked['uuid']}  "
              f"(walked back {locked['walked']}; --dvid-locked selects this)")
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
