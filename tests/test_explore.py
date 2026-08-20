"""The notebook-facing inspection functions.

The strings here are real ones from dvid.example.org, including the dirty ones: a trailing '?'
expressing doubt, trailing whitespace, and a body with no instance at all.
"""

import pandas as pd
import pytest

from neu_mark import explore as ex

BODIES = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4, 5, 6, 7], dtype="uint64"),
    "instance": pd.Series(["Tm2_A2(L)", "VLNP(R)_fragment", "Tm(R)_truncated?",
                           "irrelevant ", "LN_C5(L)_NCL", "PGNG_CV", None],
                          dtype="string"),
    "type": pd.Series(["Tm2", None, "Tm", None, "LN", None, None], dtype="string"),
})


def test_normalize_strips_whitespace_and_doubt_marks():
    assert ex.normalize(" Tm(R)_truncated? ") == "Tm(R)_truncated"
    assert ex.normalize("irrelevant ") == "irrelevant"
    assert ex.normalize(None) is None
    assert ex.normalize("   ") is None
    assert ex.normalize(float("nan")) is None


def test_instances_counts_bodies_and_drops_missing():
    out = ex.instances(BODIES)
    assert list(out.columns) == ["instance", "bodies"]
    # 6 distinct after normalization; the None row is not counted
    assert int(out["bodies"].sum()) == 6
    assert "Tm(R)_truncated" in set(out["instance"])       # the '?' is gone


def test_tokens_trailing_finds_the_suffix_vocabulary():
    out = ex.tokens(BODIES).set_index("token")["bodies"].to_dict()
    assert out["fragment"] == 1 and out["truncated"] == 1
    assert out["NCL"] == 1 and out["CV"] == 1


def test_dropping_the_side_stops_it_burying_the_suffixes():
    """Parens separate tokens, so a parenthesized side is the TRAILING token of any string
    ending in one — which puts L and R at the top of the histogram and hides the rest."""
    with_side = ex.tokens(BODIES).set_index("token")["bodies"].to_dict()
    assert with_side.get("L") == 1                       # 'Tm2_A2(L)' ends with (L)
    without = ex.tokens(BODIES, drop=r"\((L|R)\)").set_index("token")["bodies"].to_dict()
    assert "L" not in without and "R" not in without
    assert without["A2"] == 1                            # the real trailing token


def test_tokens_positions():
    lead = ex.tokens(BODIES, position="leading").set_index("token")["bodies"].to_dict()
    assert lead["VLNP"] == 1 and lead["PGNG"] == 1
    every = ex.tokens(BODIES, position="all").set_index("token")["bodies"].to_dict()
    assert every["Tm2"] == 1 and every["A2"] == 1        # both tokens of one string


def test_tokens_reports_distinct_strings_as_well_as_bodies():
    """A suffix on many distinct strings is vocabulary; on one string it is a name."""
    out = ex.tokens(BODIES).set_index("token")
    assert out.loc["fragment", "strings"] == 1


def test_an_unknown_position_is_refused():
    with pytest.raises(ValueError, match="trailing, leading or all"):
        ex.tokens(BODIES, position="sideways")


def test_a_missing_column_names_what_is_there():
    with pytest.raises(KeyError, match="no 'nope' column"):
        ex.instances(BODIES, "nope")


def test_variants_shows_what_normalization_repaired():
    """These vanish from every other view, which is why they get their own report: the
    token histogram shows a clean 'truncated' and `near` then finds nothing to fix."""
    out = ex.variants(BODIES)
    pairs = dict(zip(out["raw"], out["normalized"]))
    assert pairs["Tm(R)_truncated?"] == "Tm(R)_truncated"
    assert pairs["irrelevant "] == "irrelevant"
    # clean strings are absent
    assert "Tm2_A2(L)" not in pairs


def test_near_finds_a_semantic_misspelling_normalization_cannot():
    frame = pd.DataFrame({"instance": ["X_fragmnet", "Y_fragment", "Z_fragment"]})
    out = ex.near(ex.tokens(frame), "fragment")
    assert out["token"].tolist() == ["fragmnet"]
    assert out["bodies"].tolist() == [1]


def test_near_excludes_the_exact_token_unless_asked():
    frame = pd.DataFrame({"instance": ["Y_fragment"]})
    assert ex.near(ex.tokens(frame), "fragment").empty
    assert not ex.near(ex.tokens(frame), "fragment", include_exact=True).empty


# --------------------------------------------------------------------------- #
# the rule contract
# --------------------------------------------------------------------------- #
def test_a_rule_may_return_a_scalar_a_sequence_or_none():
    rules = {
        "one": lambda r: "x",
        "many": lambda r: ["a", "b"],
        "none": lambda r: None,
        "empty": lambda r: [],
    }
    out = ex.apply_rules(BODIES, rules)
    assert out["one"].iloc[0] == ("x",)
    assert out["many"].iloc[0] == ("a", "b")
    assert out["none"].iloc[0] == () and out["empty"].iloc[0] == ()


def test_multi_valued_facets_are_first_class():
    """Column labels genuinely are multi-valued — a central complex neuron can innervate
    more than one, and neuroglancer's `tags` property is a list per segment anyway."""
    out = ex.coverage(BODIES, {"col": lambda r: ["D2", "E3"]})
    assert out.loc[0, "multi_valued"] == len(BODIES)
    assert out.loc[0, "values"] == 2 * len(BODIES)


def test_missing_values_reach_a_rule_as_none_not_pd_na():
    """`pd.NA or ''` RAISES rather than being falsey, so the obvious rule body would blow
    up on the one body with no instance. Every rule author would hit it."""
    seen = []

    def rule(r):
        seen.append(r["instance"])
        return str(r["instance"] or "")[:2] or None

    ex.apply_rules(BODIES, {"first_two": rule})
    assert seen[-1] is None
    assert not any(x is pd.NA for x in seen)


def test_a_rule_that_raises_says_it_should_have_returned_none():
    with pytest.raises(RuntimeError, match="should return None, not raise"):
        ex.apply_rules(BODIES, {"bad": lambda r: 1 / 0})


def test_apply_rules_index_cannot_misalign_with_a_reindexed_frame():
    odd = BODIES.set_index(pd.Index(range(100, 107), name="weird"))
    out = ex.apply_rules(odd, {"x": lambda r: r["instance"]})
    assert list(out.index) == list(range(len(odd)))


def test_coverage_reports_the_number_to_watch():
    rules = {"side": lambda r: ("L" if "(L)" in str(r["instance"] or "") else None)}
    out = ex.coverage(BODIES, rules).iloc[0]
    assert out["bodies"] == 2                     # Tm2_A2(L) and LN_C5(L)_NCL
    assert out["coverage"] == pytest.approx(2 / 7)
    assert out["distinct"] == 1 and "L (2)" in out["top"]


def test_unparsed_ranks_what_to_fix_next_by_body_count():
    frame = pd.DataFrame({"instance": ["AGNG"] * 5 + ["VLNP(L)"] * 2 + ["Tm2(L)"]})
    rules = {"side": lambda r: ("L" if "(L)" in str(r["instance"] or "") else None)}
    out = ex.unparsed(frame, rules, "side")
    assert out.iloc[0]["instance"] == "AGNG" and out.iloc[0]["bodies"] == 5


def test_unparsed_of_a_rule_that_always_fires_is_empty():
    out = ex.unparsed(BODIES, {"all": lambda r: "x"}, "all")
    assert out.empty


def test_unparsed_names_the_rules_it_has():
    with pytest.raises(KeyError, match="no rule named 'zzz'"):
        ex.unparsed(BODIES, {"a": lambda r: None}, "zzz")


# --------------------------------------------------------------------------- #
# comparing a curated field against a parse
# --------------------------------------------------------------------------- #
def test_compare_classifies_the_relation_without_judging_it():
    """`type` being coarser than the parse is a difference of GRAIN, not an error — on
    dvid.example.org `type='LMC'` against `instance='L1/L3_D4(R)'` is deliberate. This reports
    the relation and leaves the decision to whoever knows the dataset."""
    core = lambda r: str(r["instance"] or "").split("(")[0] or None
    out = ex.compare(BODIES, "type", core)
    rel = dict(zip(out["body"], out["relation"]))
    assert rel[1] == "derived_is_longer"       # type=Tm2, derived=Tm2_A2
    assert rel[3] == "exact"                   # type=Tm, derived=Tm
    assert rel[2] == "field_missing"            # no type on VLNP(R)_fragment
    assert set(out["relation"]) <= set(ex.RELATIONS)


def test_compare_keeps_the_raw_string_for_reading_the_disagreements():
    core = lambda r: "ZZZ"
    out = ex.compare(BODIES, "type", core)
    row = out[out["body"] == 1].iloc[0]
    assert row["relation"] == "unrelated"
    assert row["instance"] == "Tm2_A2(L)"      # so it can be eyeballed
