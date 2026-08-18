"""Put `tmp_path` on tmpfs when one is available.

These suites are dominated by **fsync latency, not computation**. The default temp root
decides what that costs, and on a Flatiron workstation the default is the slow choice:
`/tmp` is RAID-backed xfs at ~8 ms per fsync, against ~0.004 ms on `/dev/shm`.

`PYTEST_DEBUG_TEMPROOT` is the knob rather than `--basetemp` on purpose: pytest reads it
inside `getbasetemp()`, so numbered per-run directories, the retention policy and the
sweep of old runs all keep working exactly as before — the tree just lands somewhere
faster. `--basetemp`, by contrast, `rm -rf`s the path it is given at session start and
retains nothing, so a failed run's artifacts are gone.

Escape hatches, in the order they win: an explicit `--basetemp` (pytest ignores the
temproot entirely), an inherited `PYTEST_DEBUG_TEMPROOT`, and `EM_TESTS_TMPFS=0` to force
the platform default.

The *code* below is duplicated in all four em-* repos, which are separate git repos and
must stay independently testable — a shared copy would mean a test-time import across the
layering. Keep the copies in step.

This suite writes little and needs no parallelism: it never talks to DVID (every test
stubs the transport) and its tables are a few dozen rows.
"""

import os
from pathlib import Path

# tmpfs spends real memory, so only use one with room to spare. A whole suite leaves
# well under a hundred MB, counting the runs pytest retains.
_MIN_FREE_BYTES = 2 * 1024**3

_CANDIDATES = ("/dev/shm", "/run/shm")


def _tmpfs_root() -> str | None:
    """A writable tmpfs directory with room to spare, or None to leave the default."""
    if os.environ.get("EM_TESTS_TMPFS") == "0":
        return None
    for candidate in _CANDIDATES:
        root = Path(candidate)
        if not root.is_dir() or not os.access(root, os.W_OK):
            continue
        try:
            st = os.statvfs(root)
        except OSError:
            continue
        if st.f_bavail * st.f_frsize < _MIN_FREE_BYTES:
            continue
        # Per-user, because tmpfs is node-wide and pytest insists on owning its root.
        mine = root / f"em-tests-{os.getuid()}"
        try:
            mine.mkdir(mode=0o700, exist_ok=True)
        except OSError:
            continue
        return str(mine)
    return None


_root = _tmpfs_root()
if _root is not None:
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", _root)


# --------------------------------------------------------------------------- #
# A stubbed DVID, shared by every test that reads one.
#
# Stubbed at the `neuclease.dvid.*` boundary rather than at this package's own functions,
# so the code under test is the code that runs in production — the em-volume-tools lesson
# that a test building its own spec by hand proves nothing about the path that runs.
# --------------------------------------------------------------------------- #
import pandas as pd
import pytest

from em_annotation import ops

URL = "dvid://dvid.example.org/93fdbc:main/synapses"
KV_URL = "dvid://dvid.example.org/93fdbc:main/labels_annotations"

SYN_INFO = {"Base": {"TypeName": "annotation", "Syncs": ["labels"]}}
KV_INFO = {"Base": {"TypeName": "keyvalue", "Syncs": []}}
LABELSZ_INFO = {"Base": {"TypeName": "labelsz", "Syncs": ["synapses"]}}
ROI_INFO = {"Base": {"TypeName": "roi", "Syncs": []}}
SZ_URL = "dvid://dvid.example.org/93fdbc:main/synapses_labelsz"

#: ROI instances the stub server offers. `OL(L)` contains `ME(L)`, as on the real one.
ROIS = ("ME(L)", "ME(R)", "OL(L)", "LO(L)")

#: (pre, post) per body, chosen to cover the two shapes that make a TOTAL threshold the
#: right default: body 2 is sensory-ish (almost no postsynapses), body 3 projects out of
#: the traced volume (no presynapses at all). Body 4 is a fragment.
COUNTS = {1: (5, 100), 2: (60, 3), 3: (0, 40), 4: (2, 3)}

#: Body 1 has a tbar; body 2 has the matching PSD plus one whose partner is unfetched.
ELEMENTS = {
    1: [{"Pos": [10, 20, 30], "Kind": "PreSyn", "Prop": {"conf": "0.9", "user": "jwu"},
         "Rels": [{"Rel": "PreSynTo", "To": [11, 21, 31]},
                  {"Rel": "PreSynTo", "To": [99, 99, 99]}]}],
    2: [{"Pos": [11, 21, 31], "Kind": "PostSyn", "Prop": {"conf": "0.5"},
         "Rels": [{"Rel": "PostSynTo", "To": [10, 20, 30]}]}],
}

BODY_RECORDS = {
    "1": {"bodyid": 1, "status": "Traced", "instance": "CAm(L)", "user": "ks"},
    "2": {"bodyid": 2, "status": "Anchor"},
}


@pytest.fixture(autouse=True)
def _no_node_cache():
    from em_volume_tools.dvid import clear_node_cache

    clear_node_cache()
    yield
    clear_node_cache()


@pytest.fixture()
def dvid_server(monkeypatch):
    """A DVID that answers from the dicts above and records what was asked."""
    pytest.importorskip("neuclease")
    import neuclease.dvid as nd
    import neuclease.dvid.annotation as nda
    import neuclease.dvid.keyvalue as ndk

    import em_volume_tools.dvid as vdvid

    asked = []

    # A lock-and-spawn repo: HEAD (d38898) is open, its parent (846e3a) is locked.
    # A CONCRETE uuid resolves to itself, as the real endpoint does — without that a
    # re-resolution of an already-pinned spec silently jumps back to HEAD.
    def resolve_ref(server, ref, expand=False, **k):
        if ref.endswith("~1"):
            return "846e3a"
        if ref in ("d38898", "846e3a"):
            return ref
        return "d38898"

    monkeypatch.setattr(nd, "resolve_ref", resolve_ref)
    monkeypatch.setattr(nd, "fetch_commit",
                        lambda server, uuid, **k: uuid == "846e3a")

    def fetch_instance_info(server, uuid, instance, **k):
        if instance == "synapses":
            return SYN_INFO
        if instance.endswith("_labelsz"):
            return LABELSZ_INFO
        return KV_INFO

    monkeypatch.setattr(nd, "fetch_instance_info", fetch_instance_info)
    monkeypatch.setattr(nd, "fetch_repo_info", lambda server, uuid, **k: {
        "DataInstances": {"synapses": SYN_INFO, "synapses_labelsz": LABELSZ_INFO,
                          "labels_annotations": KV_INFO,
                          **{name: ROI_INFO for name in ROIS}}})

    # ROIs. `OL(L)` contains `ME(L)` on the real server (measured: 214,883 voxels of
    # overlap at scale 5), which is the hierarchy the overlap guard exists for.
    import neuclease.dvid.roi as ndroi

    def fetch_roi_ranges_and_boxes(server, uuid, rois, **k):
        """The expensive half: one request per ROI. Stubbed as opaque per-name payloads."""
        names = list(rois)
        return {n: f"ranges:{n}" for n in names}, {n: f"box:{n}" for n in names}

    def unpack_roi_ranges_to_combined_volume(roi_labels, ranges, boxes, box_zyx=None,
                                             as_bool=False):
        """The cheap half, called twice (forward and reversed) to measure the ambiguity."""
        import numpy as np

        names = list(roi_labels)
        pairs = [(a_, b_, 214883) for i, a_ in enumerate(names)
                 for b_ in names[i + 1:] if {a_, b_} <= {"ME(L)", "OL(L)"}]
        overlaps = pd.DataFrame(pairs, columns=["roi_a", "roi_b", "overlap"])
        return (np.zeros((2, 2, 2), np.uint8),
                box_zyx if box_zyx is not None else np.array([[0, 0, 0], [2, 2, 2]]),
                overlaps)

    def determine_point_rois(server, uuid, rois, points_df, **k):
        """Label by x: 10 -> the first ROI, 11 -> the last, anything else unlabelled."""
        names = list(rois)
        labels, values = [], []
        for x in points_df["x"]:
            if int(x) == 10:
                labels.append(1); values.append(names[0])
            elif int(x) == 11:
                labels.append(len(names)); values.append(names[-1])
            else:
                labels.append(0); values.append("<unspecified>")
        points_df["roi_label"] = labels
        points_df["roi"] = values

    monkeypatch.setattr(ndroi, "fetch_roi_ranges_and_boxes", fetch_roi_ranges_and_boxes)
    monkeypatch.setattr(ndroi, "unpack_roi_ranges_to_combined_volume",
                        unpack_roi_ranges_to_combined_volume)
    monkeypatch.setattr(ndroi, "determine_point_rois", determine_point_rois)

    # labelsz: a ranked AllSyn threshold query, paged by offset, plus exact per-type counts.
    import neuclease.dvid.labelsz as ndsz

    def fetch_threshold(server, uuid, instance, threshold, element_type,
                        offset=0, n=None, **k):
        assert element_type == "AllSyn", element_type
        ranked = sorted(((p + s, b) for b, (p, s) in COUNTS.items()),
                        key=lambda t: (-t[0], t[1]))
        hits = [(b, tot) for tot, b in ranked if tot >= threshold]
        window = hits[offset:offset + (n or 10_000)]
        return pd.Series({b: tot for b, tot in window},
                         name="AllSyn", dtype="int64").rename_axis("body")

    def fetch_counts(server, uuid, instance, bodies, element_type, **k):
        idx = 0 if element_type == "PreSyn" else 1
        return pd.Series({int(b): COUNTS.get(int(b), (0, 0))[idx] for b in bodies},
                         name=element_type, dtype="int64").rename_axis("body")

    monkeypatch.setattr(ndsz, "fetch_threshold", fetch_threshold)
    monkeypatch.setattr(ndsz, "fetch_counts", fetch_counts)
    monkeypatch.setattr(vdvid, "instance_info",
                        lambda spec: fetch_instance_info(*vdvid.address(spec)))
    monkeypatch.setattr(vdvid, "node_provenance",
                        lambda spec, node: {"source": "dvid", "uuid": node["uuid"],
                                            "instance": spec["instance"],
                                            "requested": node.get("ref"),
                                            "locked": node["locked"]})

    def fetch_label(server, uuid, instance, label, relationships=False, **k):
        asked.append((label, relationships))
        return ELEMENTS.get(int(label), [])

    monkeypatch.setattr(nda, "fetch_label", fetch_label)

    def fetch_keyvalues(server, uuid, instance, keys, **k):
        return {key: BODY_RECORDS.get(key) for key in keys}

    def fetch_keyrangevalues(server, uuid, instance, key1, key2, **k):
        """Bounds INCLUSIVE at both ends, as DVID's really are — so consecutive ranges
        overlap by one key and the caller must tolerate a repeat."""
        return {key: v for key, v in BODY_RECORDS.items() if key1 <= key <= key2}

    monkeypatch.setattr(ndk, "fetch_keyvalues", fetch_keyvalues)
    monkeypatch.setattr(ndk, "fetch_keyrangevalues", fetch_keyrangevalues)

    def no_keys(*a, **k):
        raise AssertionError(
            "fetch_keys is unreliable on a large instance (52 s twice, then 504 from the "
            "proxy) and nothing may depend on it")

    monkeypatch.setattr(ndk, "fetch_keys", no_keys)
    return asked


def _src(url=URL):
    from em_volume_tools.dvid import parse_url

    return {"backend": "dvid", **parse_url(url)}


def _open(url=URL, kind="points", *, locked=False):
    """The resolved source, as the CLI produces it before doing anything else."""
    opener = ops.open_points_source if kind == "points" else ops.open_bodies_source
    return opener(_src(url), prefer_locked=locked)


