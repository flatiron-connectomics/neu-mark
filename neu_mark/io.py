"""Writing tables out, and the provenance sidecar that makes them meaningful.

## Why parquet is the default

Because csv does not preserve a **nullable** integer column, and ``to_body`` is one: an
unresolved partner is null by design. csv has no types, so what comes back is whatever
``pandas.read_csv`` infers, and for a column mixing integers with blanks that is never an
integer dtype — measured on this pandas, ``UInt64`` with one null returns as ``str``; on
older ones it returns as ``float64``, which silently **rounds** any id above 2^53. Either
way a body id stops comparing equal to itself and a join against it quietly finds nothing.

(A *non*-nullable ``uint64`` column does survive csv — pandas infers ``uint64`` correctly,
including values above 2^63. So ``points.body`` would be fine and ``relationships.to_body``
would not, which is a worse failure mode than a uniform one: the corruption is confined to
the column people are least likely to check.)

Parquet stores the declared type, holds per-file key-value metadata (so the provenance
travels *inside* the file as well as beside it), and compresses a 4M-row point table to
something worth keeping.

csv remains available because eyeballing a few hundred rows in a terminal is a real need.
It warns, naming the columns that will not survive. Feather is offered for the
local-scratch case where write speed matters more than portability.

## Every write goes through the kvstore

``neu_vol.location.write_bytes``, never ``open()`` — so ``--out`` may be a local
path or ``s3://…`` with no branch in the caller. pyarrow can serialise to a buffer, which
is what makes this possible; a ``to_parquet(path)`` would quietly write nothing useful to
an object store. Same rule, and same reason, as neu-morpho's ``precomputed.py``.
"""

from __future__ import annotations

import io as _io
import json
import logging
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

FORMATS = ("parquet", "csv", "feather")

#: Extension per format.
_EXT = {"parquet": ".parquet", "csv": ".csv", "feather": ".feather"}

#: Key under which the provenance record is embedded in a parquet file's own metadata.
METADATA_KEY = b"neu_mark_provenance"


def table_name(name: str, fmt: str) -> str:
    if fmt not in _EXT:
        raise ValueError(f"unknown table format {fmt!r}; expected one of "
                         f"{', '.join(FORMATS)}")
    return f"{name}{_EXT[fmt]}"


def _to_bytes(df, fmt: str, metadata: Mapping[str, Any] | None) -> bytes:
    buf = _io.BytesIO()
    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        if metadata is not None:
            existing = table.schema.metadata or {}
            table = table.replace_schema_metadata(
                {**existing,
                 METADATA_KEY: json.dumps(metadata, default=str).encode()})
        pq.write_table(table, buf, compression="zstd")
    elif fmt == "feather":
        import pyarrow as pa
        import pyarrow.feather as feather

        feather.write_feather(pa.Table.from_pandas(df, preserve_index=False), buf)
    elif fmt == "csv":
        _warn_csv(df)
        buf.write(df.to_csv(index=False).encode())
    else:
        raise ValueError(f"unknown table format {fmt!r}")
    return buf.getvalue()


#: Pandas' nullable integer dtypes. These are the ones csv cannot round-trip: the null
#: forces the inferred type away from an integer, to `str` or `float64` depending on the
#: pandas version. A plain `uint64` column has no null and survives intact.
_NULLABLE_INT_DTYPES = ("UInt64", "Int64", "UInt32", "Int32", "UInt16", "Int16",
                        "UInt8", "Int8")


def _warn_csv(df) -> None:
    """Name the columns csv cannot faithfully round-trip."""
    at_risk = [c for c in df.columns
               if str(df[c].dtype) in _NULLABLE_INT_DTYPES and df[c].isna().any()]
    if at_risk:
        logger.warning(
            "writing csv: %s hold integers alongside nulls, and csv has no types — "
            "pandas will read them back as str or float64, not as integers, so ids stop "
            "comparing equal and any above 2^53 are rounded. Use --format parquet if "
            "these are going to be read programmatically.", ", ".join(sorted(at_risk)))


def write_table(df, dst: str, name: str, *, fmt: str = "parquet",
                metadata: Mapping[str, Any] | None = None) -> str:
    """Serialise ``df`` and write it under ``dst``. Returns the key written."""
    from neu_vol.location import write_bytes

    key = table_name(name, fmt)
    write_bytes(dst, _to_bytes(df, fmt, metadata), key)
    logger.info("wrote %s (%d rows) to %s", key, len(df), str(dst).rstrip("/"))
    return key


def read_table(src: str, name: str, *, fmt: str = "parquet"):
    """Read a table back, for tests and for the stage that consumes these files."""
    import pandas as pd

    from neu_vol.location import read_bytes

    raw = read_bytes(src, table_name(name, fmt))
    if raw is None:
        raise FileNotFoundError(f"no {table_name(name, fmt)} at {src}")
    buf = _io.BytesIO(raw)
    if fmt == "parquet":
        return pd.read_parquet(buf)
    if fmt == "feather":
        return pd.read_feather(buf)
    return pd.read_csv(buf)


#: Preference order when reading a table whose format was not stated. parquet first because
#: it is what `write_table` defaults to, and csv last because it is the one that cannot
#: round-trip a nullable uint64 body id.
READ_FORMATS = ("parquet", "feather", "csv")


def read_tables(src: str, names: Sequence[str]) -> tuple:
    """Read several tables out of one directory, detecting each one's format.

    The point is that a caller pointing at an earlier run's output should not have to know
    or restate which ``--format`` that run used.
    """
    from neu_vol.location import exists

    out = []
    for name in names:
        for fmt in READ_FORMATS:
            if exists(src, table_name(name, fmt)):
                break
        else:
            raise FileNotFoundError(
                f"no {name} table at {str(src).rstrip('/')} — looked for "
                + ", ".join(table_name(name, f) for f in READ_FORMATS))
        frame = read_table(src, name, fmt=fmt)
        if fmt == "csv":
            # Warn rather than refuse: for a points table csv is merely lossy at the edges,
            # but a body id that came back as float64 is silently wrong above 2**53.
            logger.warning("%s was read from csv; uint64 body ids do not survive that "
                           "round trip. Re-fetch with --format parquet if ids look odd.",
                           name)
        out.append(frame)
    return tuple(out)


def read_embedded_provenance(src: str, name: str) -> dict | None:
    """The provenance record stored inside a parquet file, if it is there."""
    import pyarrow.parquet as pq

    from neu_vol.location import read_bytes

    raw = read_bytes(src, table_name(name, "parquet"))
    if raw is None:
        return None
    meta = pq.read_schema(_io.BytesIO(raw)).metadata or {}
    blob = meta.get(METADATA_KEY)
    return json.loads(blob) if blob else None


def write_provenance(dst: str, record: Mapping[str, Any], *,
                     name: str = "provenance") -> None:
    """Write the sidecar. Never fails the run — the tables are the valuable part.

    A sidecar *as well as* the parquet metadata, because csv and feather have nowhere to
    put it and a record that only sometimes exists is one nobody learns to look for. Same
    argument as neu-vol' ``ops/provenance.py``, whose record shape this reuses.
    """
    from neu_vol.location import write_json

    try:
        write_json(dst, dict(record), f"{name}.json")
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("could not write %s.json at %s (%s: %s). The tables are fine; "
                       "only the record of where they came from is missing.",
                       name, dst, type(exc).__name__, exc)
