"""Encoding ``neuroglancer_annotations_v1`` LINE annotations.

One annotation per synaptic connection: a line from the T-bar to the PSD. That is what the
published male-CNS synapse dataset does, and it is strictly more useful than two POINT
sources, because the shader can render either endpoint alone —
``setLineWidth(0.0); setEndpointMarkerSize(pre, 0.0)`` draws just the T-bars. So LINE gives
both point views for free, and adds the connection.

## The byte layout, verified against a published file

Little-endian throughout. Per annotation:

1. geometry — for LINE, two ``float32`` triples: first endpoint then second
2. properties, in **descending size order**: 4-byte, then 2-byte, then 1-byte
3. 0-3 zero bytes of padding to a 4-byte boundary
4. *single-annotation encoding only*: per relationship, a ``uint32`` count then that many
   ``uint64`` ids

A group of annotations (the spatial and relationship indexes) instead writes a ``uint64``
count, then every annotation's geometry+properties+padding, then **all** the ids — and no
relationship trailers.

This was checked against ``gs://flyem-male-cns/v1.0/male-cns-v1.0-synapses-precomputed``
rather than read off the spec alone. Their records are 24 geometry + 37 properties + 3
padding = 64 bytes, and ``8 + 10000*64 + 10000*8 = 720,008`` is exactly the size of their
level-0 blob. One of their ``by_id`` entries decodes to 88 bytes = 64 + (4+8) + (4+8), the
trailer for two relationships with one id each.

## Property order is a correctness rule, not a style choice

The reader infers each property's offset from the declared order, so :func:`sort_properties`
puts them in descending size and the encoder writes them in that order. Declaring
``uint8`` before ``float32`` and writing them in that order produces a file that parses
into garbage — every value misaligned, no error anywhere.

**Properties cannot be uint64**, which is why the reference stores body ids as
``body_pre_u32`` / ``body_post_u32``: the shader-visible value is truncated to 32 bits, and
the suffix says so. The untruncated ids live in the *relationships*, which are uint64-keyed.
"""

from __future__ import annotations

import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

AT_TYPE = "neuroglancer_annotations_v1"

#: Lowercase, matching the published reference. The spec document writes "LINE"; the file
#: that demonstrably renders says "line", so follow the file.
LINE = "line"
POINT = "point"

#: Bytes per property type, and the numpy dtype used to encode it. ``rgb``/``rgba`` are in
#: the format but unused here.
PROPERTY_TYPES: dict[str, tuple[int, str]] = {
    "float32": (4, "<f4"), "uint32": (4, "<u4"), "int32": (4, "<i4"),
    "uint16": (2, "<u2"), "int16": (2, "<i2"),
    "uint8": (1, "u1"), "int8": (1, "i1"),
}

#: Geometry floats per annotation type.
GEOMETRY_FLOATS = {LINE: 6, POINT: 3}


def sort_properties(properties: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Properties in the order the format requires: descending size, order kept within a size.

    Stable within a size class so a caller's ordering of, say, several float32s survives.
    """
    known = [p for p in properties]
    for p in known:
        if p.get("type") not in PROPERTY_TYPES:
            raise ValueError(
                f"property {p.get('id')!r} has type {p.get('type')!r}; the format allows "
                f"{', '.join(sorted(PROPERTY_TYPES))} (notably NOT uint64 — store a wide id "
                f"as a relationship, and a truncated copy as uint32 if a shader needs it)")
    return sorted((dict(p) for p in known),
                  key=lambda p: -PROPERTY_TYPES[p["type"]][0])


def record_size(annotation_type: str, properties: Sequence[Mapping[str, Any]]) -> int:
    """Bytes per annotation in a group: geometry + properties + padding to 4 bytes."""
    geometry = 4 * GEOMETRY_FLOATS[annotation_type]
    props = sum(PROPERTY_TYPES[p["type"]][0] for p in properties)
    return geometry + props + (-(geometry + props) % 4)


def _property_columns(properties: Sequence[Mapping[str, Any]],
                      values: Mapping[str, Any], n: int) -> list[np.ndarray]:
    columns = []
    for p in properties:
        if p["id"] not in values:
            raise KeyError(f"no values supplied for property {p['id']!r}")
        _size, dtype = PROPERTY_TYPES[p["type"]]
        column = np.asarray(values[p["id"]])
        if column.shape != (n,):
            raise ValueError(
                f"property {p['id']!r} has {column.shape} values, expected ({n},)")
        columns.append(column.astype(dtype, copy=False))
    return columns


def encode_group(geometry: np.ndarray, ids: Sequence[int], *,
                 annotation_type: str = LINE,
                 properties: Sequence[Mapping[str, Any]] = (),
                 values: Mapping[str, Any] | None = None) -> bytes:
    """The multiple-annotation encoding: a spatial cell, or one relationship's annotations.

    ``geometry`` is ``(n, floats)`` in the **xyz** order the format stores — build it with
    ``tables.positions_xyz``, which is the only place that decides axis order.

    Note what this encoding does *not* carry: relationship ids. Those appear only in the
    single-annotation encoding, which is why :func:`encode_single` exists separately.
    """
    props = list(properties)
    n = len(ids)
    floats = GEOMETRY_FLOATS[annotation_type]
    geometry = np.asarray(geometry, dtype="<f4")
    if geometry.shape != (n, floats):
        raise ValueError(f"geometry is {geometry.shape}, expected ({n}, {floats}) for "
                         f"{annotation_type!r}")

    stride = record_size(annotation_type, props)
    body = np.zeros((n, stride), dtype="u1")
    body[:, :4 * floats] = geometry.reshape(n, -1).view("u1")

    offset = 4 * floats
    for p, column in zip(props, _property_columns(props, values or {}, n)):
        size = PROPERTY_TYPES[p["type"]][0]
        body[:, offset:offset + size] = column.view("u1").reshape(n, size)
        offset += size
    # Everything from `offset` to `stride` stays zero: that is the padding, and the reference
    # file's padding bytes are zero too.

    return (struct.pack("<Q", n) + body.tobytes()
            + np.asarray(ids, dtype="<u8").tobytes())


def encode_single(geometry: Sequence[float], *, annotation_type: str = LINE,
                  properties: Sequence[Mapping[str, Any]] = (),
                  values: Mapping[str, Any] | None = None,
                  relationships: Sequence[Sequence[int]] = ()) -> bytes:
    """The single-annotation encoding, for the ``by_id`` index.

    ``relationships`` is one sequence of segment ids per declared relationship, **in the
    declared order** — each written as a ``uint32`` count followed by ``uint64`` ids.
    """
    props = list(properties)
    floats = GEOMETRY_FLOATS[annotation_type]
    geom = np.asarray(geometry, dtype="<f4").reshape(-1)
    if geom.size != floats:
        raise ValueError(f"geometry has {geom.size} floats, expected {floats}")

    out = bytearray(geom.tobytes())
    for p in props:
        _size, dtype = PROPERTY_TYPES[p["type"]]
        if p["id"] not in (values or {}):
            raise KeyError(f"no value supplied for property {p['id']!r}")
        out += np.asarray((values or {})[p["id"]], dtype=dtype).tobytes()
    out += b"\x00" * (-len(out) % 4)

    for related in relationships:
        related = list(related)
        out += struct.pack("<I", len(related))
        out += np.asarray(related, dtype="<u8").tobytes()
    return bytes(out)


def build_info(*, lower_bound: Sequence[float], upper_bound: Sequence[float],
               voxel_size_xyz: Sequence[float], annotation_type: str = LINE,
               properties: Sequence[Mapping[str, Any]] = (),
               relationships: Sequence[Mapping[str, Any]] = (),
               by_id: Mapping[str, Any] | None = None,
               spatial: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The annotation source's ``info``.

    ``@type``, ``dimensions``, ``lower_bound``, ``upper_bound``, ``annotation_type``,
    ``properties``, ``relationships``, ``by_id`` and ``spatial`` are all **required** members
    — ``relationships`` and ``spatial`` may be empty arrays, but they must be present.

    ``dimensions`` names the axes and fixes their order, so it is written x, y, z to match
    the xyz the geometry is stored in. Sizes are metres, as the format requires: 8 nm is
    ``8e-9``.
    """
    props = sort_properties(properties)
    return {
        "@type": AT_TYPE,
        "dimensions": {axis: [float(size) * 1e-9, "m"]
                       for axis, size in zip("xyz", voxel_size_xyz)},
        "lower_bound": [float(v) for v in lower_bound],
        "upper_bound": [float(v) for v in upper_bound],
        "annotation_type": annotation_type,
        "properties": props,
        "relationships": [dict(r) for r in relationships],
        "by_id": dict(by_id or {"key": "by_id"}),
        "spatial": [dict(s) for s in spatial],
    }


def enum_property(prop_id: str, labels: Sequence[str], *, dtype: str = "int16",
                  description: str | None = None) -> dict[str, Any]:
    """A property whose integer values index ``labels``.

    ``enum_labels`` is required exactly when ``enum_values`` is given. The reference encodes
    its 238 neuropil ROIs this way, with ``<unspecified>`` at index 0 — so a value of 0
    meaning "none" is the established convention and a shader can test for it.
    """
    return {"id": prop_id, "type": dtype,
            **({"description": description} if description else {}),
            "enum_values": list(range(len(labels))),
            "enum_labels": [str(v) for v in labels]}
