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
_SRC_HELP = """\
DVID source, as dvid://SERVER/UUID/INSTANCE — always three segments:
  SERVER    host, optionally with a port (emdata3:8900). A bare host gets http://;
            use dvid+https://HOST/... for TLS.
  UUID      a node uuid (93fdbc, abbreviated is fine) OR a repo:branch ref
            (93fdbc:main), which means "HEAD of that branch".
  INSTANCE  the data instance name.
A ':' in the server (port) or the uuid (branch) adds no segment; this splits on '/'.
Or '@name' to look the instance up in your config (em_annotation.config); the
resolved URL is printed, so the command still says what it read.
"""


def _add_common(q: argparse.ArgumentParser, *, instance_hint: str) -> None:
    q.add_argument("--src", required=True, metavar="URL",
                   help=_SRC_HELP + f"For this command INSTANCE is {instance_hint}.")
    q.add_argument("--out", required=True, metavar="DST",
                   help="where the tables go — a local directory or s3://bucket/prefix. "
                        "May contain {uuid}, {uuid:N}, {uuid:full}, {branch} or "
                        "{instance}, expanded from the resolved node.")
    q.add_argument("--bodies", required=True, metavar="IDS",
                   help="body ids: inline (123,456), or a path to a .csv/.parquet/"
                        ".feather with a body column, or a text file with one id per "
                        "line. Required — DVID cannot cheaply enumerate the bodies "
                        "worth asking about.")
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
    q.set_defaults(func=cmd_points)

    q = sub.add_parser(
        "bodies", help="per-body annotations (name, status) for a set of bodies",
        description="Fetch the per-body records from a DVID `keyvalue` instance and write "
                    "a `bodies` table, one row per body. Records are ragged, so the "
                    "table is the union of the fields present.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(q, instance_hint="a keyvalue instance, e.g. 'labels_annotations'")
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
        "info", help="what a DVID annotation source is, and which node you would get",
        description="One request per fact: the instance type, what it is synced to, and "
                    "both candidate nodes for the ref (what it points at now, and the "
                    "newest locked one).")
    q.add_argument("--src", required=True, metavar="URL", help=_SRC_HELP)
    q.set_defaults(func=cmd_info)

    return p


def _parse_args(argv=None):
    return build_parser().parse_args(argv)


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
                              write_connections=bool(args.connections), **kwargs)
    print(f"points: {len(result['points'])} elements, "
          f"{len(result['relationships'])} relationships")
    _report_match(result["match"])
    if result["failures"]:
        print(f"  {len(result['failures'])} bodies failed and were skipped; "
              f"first few: {list(result['failures'])[:5]}", file=sys.stderr)
    print(f"wrote {', '.join(result['written'])} to {out}")
    return 0


def cmd_bodies(args) -> int:
    from . import ops

    source = ops.open_bodies_source(_src_spec(args.src),
                                    prefer_locked=bool(args.dvid_locked))
    out = _expand_out(args.out, source)
    ids = _load_bodies(args)

    result = ops.fetch_bodies(source, out, ids, fmt=args.fmt)
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
