"""Writing tables out: dtype fidelity, the kvstore rule, and the provenance record."""

import numpy as np
import pandas as pd
import pytest

from neu_mark import io, tables


@pytest.fixture()
def frame():
    """A frame carrying every dtype that matters: uint64 ids and a nullable partner."""
    return pd.DataFrame({
        "body": pd.Series([2**63 + 7, 3], dtype="uint64"),
        "z": pd.Series([1, 2], dtype=np.int32),
        "to_body": pd.Series([2**63 + 9, None], dtype="UInt64"),
        "conf": pd.Series([0.5, 1.0], dtype=np.float32),
    })


def test_parquet_preserves_uint64_body_ids(tmp_path, frame):
    """The whole reason parquet is the default. csv cannot do this."""
    out = str(tmp_path / "t")
    io.write_table(frame, out, "points")
    back = io.read_table(out, "points")
    assert back["body"].dtype == np.uint64
    assert int(back["body"].iloc[0]) == 2**63 + 7
    assert str(back["to_body"].dtype) == "UInt64"
    assert int(back["to_body"].iloc[0]) == 2**63 + 9
    assert pd.isna(back["to_body"].iloc[1])


def test_csv_breaks_the_nullable_partner_column_and_says_so(tmp_path, frame, caplog):
    """The precise csv hazard, and it is narrower than "csv loses uint64".

    A plain uint64 column DOES survive csv — pandas infers uint64, even above 2^63. What
    does not survive is an integer column containing nulls, which `to_body` always is
    (an unresolved partner is null by design). That is the worse failure: the corruption
    is confined to the one column nobody thinks to check.
    """
    out = str(tmp_path / "t")
    with caplog.at_level("WARNING"):
        io.write_table(frame, out, "points", fmt="csv")
    # the warning names exactly the at-risk column: `to_body`, and NOT `body`
    named = caplog.text.split("writing csv: ")[1].split(" hold integers")[0]
    assert named == "to_body"

    back = io.read_table(out, "points", fmt="csv")
    assert back["body"].dtype == np.uint64                     # survives
    assert int(back["body"].iloc[0]) == 2**63 + 7
    # to_body does not: whatever it came back as, it is not an integer and no longer
    # compares equal to the value written.
    import pandas as pd

    assert not pd.api.types.is_integer_dtype(back["to_body"])
    assert not (back["to_body"].iloc[0] == 2**63 + 9)


def test_feather_also_preserves_dtypes(tmp_path, frame):
    out = str(tmp_path / "t")
    io.write_table(frame, out, "points", fmt="feather")
    back = io.read_table(out, "points", fmt="feather")
    assert back["body"].dtype == np.uint64


def test_provenance_travels_inside_the_parquet_file(tmp_path, frame):
    out = str(tmp_path / "t")
    record = {"source": {"uuid": "d38898ac", "instance": "synapses"}}
    io.write_table(frame, out, "points", metadata=record)
    assert io.read_embedded_provenance(out, "points") == record


def test_provenance_sidecar_is_written_for_every_format(tmp_path):
    from neu_vol.location import read_json

    out = str(tmp_path / "t")
    io.write_provenance(out, {"source": {"uuid": "abc"}})
    assert read_json(out, "provenance.json")["source"]["uuid"] == "abc"


def test_a_failed_provenance_write_does_not_lose_the_tables(tmp_path, frame, caplog):
    """The tables are the valuable part; the sidecar must never take a run down."""
    out = str(tmp_path / "t")
    io.write_table(frame, out, "points")
    with caplog.at_level("WARNING"):
        io.write_provenance("s3://definitely-not-a-real-bucket-xyz/p", {"a": 1})
    assert "only the record of where they came from is missing" in caplog.text
    assert len(io.read_table(out, "points")) == 2


def test_an_unknown_format_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown table format"):
        io.table_name("points", "xlsx")


def test_the_module_never_opens_a_file_directly():
    """Every write goes through location.write_bytes, or an s3 destination silently
    writes nothing. Same rule and same reason as neu-morpho's precomputed.py.

    Checked over the parsed AST, not the source text: the module docstring *discusses*
    `open()` and `to_parquet(path)`, and a text search cannot tell prose from a call.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(io))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "open" not in called

    attrs = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    # pandas' own writers take a path and would bypass the kvstore entirely
    assert not {"to_parquet", "to_feather"} & attrs


def test_round_trip_of_a_real_points_table(tmp_path):
    """End to end on frames the parser actually produces, not hand-built ones."""
    el = {"Pos": [10, 20, 30], "Kind": "PreSyn", "Prop": {"conf": "0.9"},
          "Rels": [{"Rel": "PreSynTo", "To": [11, 21, 31]}]}
    pts, rels = tables.combine([tables.elements_to_frames([el], body=13481220)])
    out = str(tmp_path / "t")
    io.write_table(pts, out, "points")
    io.write_table(rels, out, "relationships")
    assert io.read_table(out, "points")["conf"].iloc[0] == pytest.approx(0.9)
    assert io.read_table(out, "relationships")["rel"].iloc[0] == "PreSynTo"
