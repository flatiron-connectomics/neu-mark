"""Reading the body list — the required input to every command."""

import pandas as pd
import pytest

from em_annotation import bodies


def test_inline_list_accepts_commas_and_whitespace():
    assert bodies.load("3,1,2") == [1, 2, 3]
    assert bodies.load("3 1 2") == [1, 2, 3]
    assert bodies.load("10") == [10]


def test_ids_are_deduplicated_and_sorted():
    """Sorted so a run's order depends on the set requested, not how it was spelled;
    de-duplicated because asking twice costs a request and doubles the rows."""
    assert bodies.load("5,5,1,5") == [1, 5]


def test_an_iterable_is_accepted_directly():
    assert bodies.load([9, 8, 8]) == [8, 9]


def test_text_file_one_per_line_with_comments(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("# traced neurons\n13481220\n18126428  # MT(L)\n\n77263210\n")
    assert bodies.load(str(p)) == [13481220, 18126428, 77263210]


def test_csv_finds_a_recognisably_named_column(tmp_path):
    p = tmp_path / "b.csv"
    pd.DataFrame({"name": ["a", "b"], "body": [7, 3]}).to_csv(p, index=False)
    assert bodies.load(str(p)) == [3, 7]


def test_csv_single_column_needs_no_name(tmp_path):
    p = tmp_path / "b.csv"
    pd.DataFrame({"whatever": [7, 3]}).to_csv(p, index=False)
    assert bodies.load(str(p)) == [3, 7]


def test_parquet_round_trips_through_the_reader(tmp_path):
    p = tmp_path / "b.parquet"
    pd.DataFrame({"body": [2**63 + 7, 3]}, dtype="uint64").to_parquet(p)
    assert bodies.load(str(p)) == [3, 2**63 + 7]


def test_this_packages_own_bodies_table_is_a_valid_input(tmp_path):
    """So a body list from one node feeds the next fetch with no intermediate step."""
    from em_annotation import tables

    df = tables.keyvalues_to_frame({"5": {"bodyid": 5}, "9": {"bodyid": 9}})
    p = tmp_path / "bodies.parquet"
    df.drop(columns=["json"]).to_parquet(p)
    assert bodies.load(str(p)) == [5, 9]


def test_an_ambiguous_table_asks_for_the_column(tmp_path):
    p = tmp_path / "b.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="--body-column"):
        bodies.load(str(p))
    assert bodies.load(str(p), column="b") == [2]


def test_a_named_column_that_is_absent_lists_what_is_there(tmp_path):
    p = tmp_path / "b.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="has no column 'zzz'"):
        bodies.load(str(p), column="zzz")


def test_a_non_numeric_token_names_the_line(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("1\nnot-a-body\n")
    with pytest.raises(ValueError, match="line 2 has 'not-a-body'"):
        bodies.load(str(p))


def test_a_missing_file_that_looks_like_a_path_is_not_read_as_ids():
    with pytest.raises(FileNotFoundError):
        bodies.load("/no/such/file.csv")


def test_summarise_is_one_line():
    assert bodies.summarise([]) == "0 bodies"
    assert bodies.summarise([7]) == "1 body (7)"
    assert bodies.summarise([3, 9]) == "2 bodies (3..9)"
