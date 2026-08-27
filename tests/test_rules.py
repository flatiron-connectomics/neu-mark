"""The @rule framework: declaration, validation, and testing a rule in isolation."""

import re

import pandas as pd
import pytest

from neu_mark import rules
from neu_mark.rules import Rule, RuleSet, rule

BODIES = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4], dtype="uint64"),
    "instance": pd.Series(["Tm2_A2(L)", "VLNP(R)_fragment", "LN_C5(L)_NCL", None],
                          dtype="string"),
    "type": pd.Series(["Tm2", None, "LN", None], dtype="string"),
})

SIDE = re.compile(r"\((L|R)\)")


@rule
def side(r):
    """Hemisphere, from the parenthesized (L)/(R)."""
    m = SIDE.search(str(r["instance"] or ""))
    return m.group(1) if m else None


@rule(multi=True)
def column(r):
    """Column labels; genuinely multi-valued."""
    return re.findall(r"_([A-Z]\d+)(?=$|[_(])", str(r["instance"] or ""))


# --------------------------------------------------------------------------- #
# declaration
# --------------------------------------------------------------------------- #
def test_the_decorator_works_bare_and_called():
    assert isinstance(side, Rule) and isinstance(column, Rule)
    assert side.name == "side" and column.name == "column"
    assert side.multi is False and column.multi is True


def test_the_docstring_becomes_the_description():
    """Carried through to the coverage report and, later, to the segment property's own
    `description` member, so the viewer explains itself."""
    assert side.description == "Hemisphere, from the parenthesized (L)/(R)."


def test_a_rule_is_still_an_ordinary_callable():
    assert side(pd.Series({"instance": "Tm2_A2(L)"})) == "L"


def test_needs_defaults_to_instance_and_is_declarable():
    assert side.needs == ("instance",)

    @rule(needs=["instance", "type"])
    def both(r):
        return None

    assert both.needs == ("instance", "type")


# --------------------------------------------------------------------------- #
# testing a rule, which is the point
# --------------------------------------------------------------------------- #
def test_test_evaluates_one_string_positionally():
    assert side.test("Tm2_A2(L)") == "L"
    assert side.test("AGNG") is None
    assert column.test("L1/L3_D4(R)") == ["D4"]


def test_test_accepts_other_fields_by_keyword():
    @rule(needs=["instance", "type"])
    def curated(r):
        return r["type"]

    assert curated.test("anything", type="Tm2") == "Tm2"


def test_test_passes_missing_fields_as_none_not_pd_na():
    """So `r["instance"] or ""` behaves in a test exactly as it does in a run."""
    seen = {}

    @rule(needs=["instance", "type"])
    def spy(r):
        seen["type"] = r["type"]
        return str(r["instance"] or "")

    spy.test("X")
    assert seen["type"] is None


def test_test_of_a_missing_value_does_not_raise():
    assert side.test(None) is None


# --------------------------------------------------------------------------- #
# the multi declaration
# --------------------------------------------------------------------------- #
def test_a_single_valued_rule_returning_several_values_is_an_error():
    """The findall-instead-of-search slip. It looks fine until a string matches twice."""
    @rule
    def oops(r):
        return re.findall(r"([A-Z])", str(r["instance"] or ""))

    assert oops.values(pd.Series({"instance": "abc_D"})) == ("D",)   # one is fine
    with pytest.raises(ValueError, match="declared single-valued but returned 2"):
        oops.values(pd.Series({"instance": "A_B"}))


def test_a_multi_valued_rule_may_return_any_number():
    assert column.values(pd.Series({"instance": "X_A1_B2"})) == ("A1", "B2")
    assert column.values(pd.Series({"instance": "AGNG"})) == ()


# --------------------------------------------------------------------------- #
# RuleSet
# --------------------------------------------------------------------------- #
def test_a_ruleset_keeps_declaration_order():
    rs = RuleSet([side, column])
    assert rs.names == ["side", "column"] and len(rs) == 2
    assert rs["side"] is side


def test_duplicate_names_are_refused():
    with pytest.raises(ValueError, match="duplicate rule names: side"):
        RuleSet([side, side])


def test_an_undecorated_function_is_refused_with_the_fix():
    with pytest.raises(TypeError, match="decorate it with @rule"):
        RuleSet([lambda r: None])


def test_a_plain_mapping_is_accepted_for_the_first_five_minutes():
    rs = RuleSet({"x": lambda r: "v"})
    assert rs.names == ["x"] and rs.explain("anything") == {"x": "v"}


def test_reading_an_undeclared_column_names_the_rule_and_the_fix():
    """`require` can only check what a rule declares, so an undeclared read fails nowhere
    until the one row that lacks the column — as a bare KeyError naming neither the rule nor
    the fix, which is exactly what `needs` exists to prevent."""
    @rule
    def group(r):
        """reads `type` without saying so"""
        return r["type"]

    with pytest.raises(KeyError, match=r"does not declare"):
        group.test("Tm2_A2\\(L\\)")
    with pytest.raises(KeyError, match=r"needs=\['instance', 'type'\]"):
        RuleSet([group]).explain("Tm2")


def test_a_declared_column_that_is_missing_raises_the_original_error():
    """Only an UNDECLARED read is rewritten; anything else must surface as it is."""
    @rule(needs=["type"])
    def group(r):
        """declares it properly"""
        return r["nope"] if r["type"] is None else r["type"]

    with pytest.raises(KeyError) as caught:
        group.test(type=None)
    assert "does not declare" in str(caught.value)   # 'nope' is the undeclared one


def test_unmet_needs_are_reported_up_front_naming_the_rule():
    @rule(needs=["instance", "roi_post_top"])
    def needs_roi(r):
        return None

    with pytest.raises(KeyError, match="needs_roi needs roi_post_top"):
        RuleSet([needs_roi]).apply(BODIES)


def test_a_rules_repr_delimits_its_fields_rather_than_running_them_together():
    """Every value here is itself a word or a list, so space-separating them made
    `tag bare needs=instance consumes=…` one stream with nothing marking a field boundary."""
    text = repr(nucleated)
    assert text.startswith("<Rule nucleated | tag bare | needs=instance | "
                           "consumes=[ncl, nucleus] — ")
    assert text.endswith("— Cell body in the volume. Normalizes an ALIAS, so its value is "
                         "not its token.>")
    assert "| multi" in repr(column) and "multi" not in repr(side)


def test_a_ruleset_repr_aligns_its_rules_into_columns():
    """A listing is read DOWN a column — which are drop rules, which declare consumption —
    and ragged lines defeat that however well delimited each one is on its own."""
    text = repr(RuleSet([side, column, nucleated], source="wasp_rules.py"))
    header, *lines = text.splitlines()
    assert header == "<RuleSet 3 rules, from wasp_rules.py>"
    assert [line.split()[0] for line in lines] == ["side", "column", "nucleated"]
    # Every column starts at the same offset, the description included.
    assert len({line.index("needs=") for line in lines}) == 1
    assert len({line.index("—") for line in lines}) == 1


def test_alignment_survives_a_rule_with_nothing_in_a_column():
    """`side` declares no extras. Stripping its padding would start its description in a
    different column from its neighbours — the very thing the alignment is for."""
    lines = repr(RuleSet([side, nucleated])).splitlines()[1:]
    assert len({line.index("—") for line in lines}) == 1


def test_an_empty_ruleset_reprs_without_indexing_into_nothing():
    assert repr(RuleSet([])) == "<RuleSet 0 rules>"


def test_a_notebook_renders_the_table_and_a_terminal_the_reprs():
    """Via IPython's display protocol, so the OBJECT is the same everywhere and only its
    rendering differs. A method returning a frame in a notebook and text elsewhere would
    break the moment notebook code moved into a module — the path a rules module takes."""
    rs = RuleSet([side, column])
    html = rs._repr_html_()
    assert "<table" in html and "column" in html
    assert repr(rs).startswith("<RuleSet 2 rules")
    # The same call, the same type, wherever it runs.
    assert isinstance(rs.describe(), pd.DataFrame)


def test_describe_reports_what_every_rule_declared():
    """The no-data counterpart of `coverage`: what the rules ARE, not what they did."""
    @rule(drop=True)
    def noise(r):
        """junk"""
        return None

    frame = RuleSet([side, column, nucleated, noise]).describe()
    assert list(frame["rule"]) == ["side", "column", "nucleated", "noise"]
    assert list(frame["kind"]) == ["tag", "tag", "tag", "drop"]
    assert bool(frame.set_index("rule").loc["column", "multi"]) is True
    assert frame.set_index("rule").loc["side", "description"] == (
        "Hemisphere, from the parenthesized (L)/(R).")


def test_describe_says_whether_consumption_was_declared_or_derived():
    frame = RuleSet([side, nucleated]).describe().set_index("rule")
    assert frame.loc["side", "consumes"] == "derived"
    assert frame.loc["nucleated", "consumes"] == "ncl, nucleus"


def test_explain_stays_single_purpose():
    """It answers "what would this string produce". A no-argument call returning
    DESCRIPTIONS instead would make one method return two dicts of strings that no caller
    could tell apart by shape — `describe` is the separate question."""
    rs = RuleSet([side, column])
    assert rs.explain("Tm2_A2(L)") == {"side": "L", "column": ("A2",)}
    assert rs.explain() == {"side": None, "column": ()}


def test_explain_shows_every_facet_for_one_string():
    out = RuleSet([side, column]).explain("LN_C5(L)_NCL")
    assert out == {"side": "L", "column": ("C5",)}


def test_explain_reports_scalars_for_single_valued_rules():
    """So the result reads the way the facets are meant to, not uniformly as tuples."""
    out = RuleSet([side, column]).explain("AGNG")
    assert out["side"] is None and out["column"] == ()


def test_apply_and_coverage_carry_the_declared_facts():
    rs = RuleSet([side, column])
    applied = rs.apply(BODIES)
    assert applied["side"].tolist() == [("L",), ("R",), ("L",), ()]

    cov = rs.coverage(BODIES).set_index("rule")
    assert cov.loc["side", "bodies"] == 3
    assert bool(cov.loc["column", "multi"]) is True
    assert cov.loc["side", "description"].startswith("Hemisphere")


def test_unparsed_delegates_and_ranks_by_bodies():
    rs = RuleSet([side])
    out = rs.unparsed(BODIES, "side")
    assert out.empty or "instance" in out.columns


def test_keep_is_an_explicit_allowlist():
    """A drop-list would let a new field reach a viewer by being forgotten."""
    rs = RuleSet([side], keep=["instance", "type"])
    assert rs.keep == ("instance", "type")
    assert RuleSet([side]).keep == ()


# --------------------------------------------------------------------------- #
# loading a rules module by path
# --------------------------------------------------------------------------- #
MODULE = '''
import re
from neu_mark import rule

KEEP = ["instance", "type"]

@rule
def side(r):
    """Side."""
    m = re.search(r"\\((L|R)\\)", str(r["instance"] or ""))
    return m.group(1) if m else None

@rule(multi=True)
def column(r):
    """Columns."""
    return re.findall(r"_([A-Z]\\d+)(?=$|[_(])", str(r["instance"] or ""))
'''


def test_a_module_of_decorated_functions_needs_no_rules_variable(tmp_path):
    path = tmp_path / "r.py"
    path.write_text(MODULE)
    rs = rules.from_module(str(path))
    assert rs.names == ["side", "column"]
    assert rs.keep == ("instance", "type")
    assert rs.source == str(path)
    assert rs.explain("Tm2_A2(L)") == {"side": "L", "column": ("A2",)}


def test_an_explicit_rules_list_wins(tmp_path):
    path = tmp_path / "r.py"
    path.write_text(MODULE + "\nRULES = [column]\n")
    assert rules.from_module(str(path)).names == ["column"]


def test_a_module_with_no_rules_says_what_was_expected(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("X = 1\n")
    with pytest.raises(ValueError, match="defines no rules"):
        rules.from_module(str(path))


def test_a_missing_module_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no rules module at"):
        rules.from_module(str(tmp_path / "nope.py"))


def test_a_broken_module_does_not_linger_in_sys_modules(tmp_path):
    """So fixing the file and re-loading works, rather than resurrecting a half-built one."""
    import sys

    path = tmp_path / "broken.py"
    path.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(RuntimeError, match="boom"):
        rules.from_module(str(path))
    assert not [m for m in sys.modules if m.endswith("_broken")]

    path.write_text(MODULE)
    assert rules.from_module(str(path)).names == ["side", "column"]


# --------------------------------------------------------------------------- #
# drop rules
# --------------------------------------------------------------------------- #
def test_a_drop_rule_is_never_tagged():
    """The bodies it fires on are not in the output to carry a tag."""
    @rule(drop=True)
    def noise(r):
        """junk"""
        return "junk" if r["instance"] == "x" else None

    assert noise.drop is True and noise.tag is False


def test_a_multi_valued_drop_rule_is_refused():
    """A body is either excluded or not, so there is one reason, not a list of them."""
    with pytest.raises(ValueError, match="drop=True and multi=True"):
        @rule(drop=True, multi=True)
        def noise(r):
            """junk"""
            return ["a", "b"]


def test_drops_and_tagged_partition_the_set():
    @rule(drop=True)
    def noise(r):
        """junk"""
        return None

    rs = RuleSet([noise, side, column])
    assert [r.name for r in rs.drops] == ["noise"]
    assert [r.name for r in rs.tagged] == ["side", "column"]


# --------------------------------------------------------------------------- #
# consumed / remainder
# --------------------------------------------------------------------------- #
@rule(prefix="", consumes=["ncl", "nucleus"])
def nucleated(r):
    """Cell body in the volume. Normalizes an ALIAS, so its value is not its token."""
    lowered = {t.lower() for t in rules.explore.split_tokens(r["instance"])}
    return "nucleated" if lowered & {"ncl", "nucleus"} else None


def test_a_rule_reports_the_tokens_it_accounted_for():
    row = rules._row({"instance": "Tm2_A2(L)"}, needs=["instance"])
    assert side.consumed(row) == ("L",)
    assert column.consumed(row) == ("A2",)


def test_consumes_is_declared_only_where_the_value_is_not_the_token():
    """`column` returns `A2`, which IS the token, so it needs no declaration. `nucleated`
    returns `nucleated` from the token `NCL`, so without one the alias stays in the residue."""
    row = rules._row({"instance": "LN_C5(L)_NCL"}, needs=["instance"])
    assert nucleated.consumed(row) == ("NCL",)
    assert nucleated.remainder("LN_C5(L)_NCL") == "LN_C5_L"


def test_remainder_is_token_granular_not_substring_removal():
    """`side` returns "L". Deleting that SUBSTRING from LC10_C5(L) eats LC10's own L."""
    assert side.remainder("LC10_C5(L)") == "LC10_C5"
    assert side.remainder("L1/L3_D4(R)") == "L1/L3_D4"


def test_a_ruleset_remainder_composes_every_rule():
    rs = RuleSet([side, column, nucleated])
    assert rs.remainder("Tm2_A2(L)") == "Tm2"
    assert rs.remainder("LN_C5(L)_NCL") == "LN"
    assert rs.consumed("LN_C5(L)_NCL") == {
        "side": ("L",), "column": ("C5",), "nucleated": ("NCL",)}


def test_a_rule_that_does_not_read_the_string_consumes_nothing_from_it():
    """The trap that makes a naive version wrong: `group` returns the curated `type`, often
    the very token a cell-type remainder is trying to KEEP. It would eat its own answer."""
    @rule(needs=["type"])
    def group(r):
        """curated cell type"""
        return r["type"]

    row = rules._row({"instance": "Tm2_A2(L)", "type": "Tm2"},
                     needs=["instance", "type"])
    assert group.consumed(row) == ()
    assert RuleSet([side, column, group]).remainder(
        instance="Tm2_A2(L)", type="Tm2") == "Tm2"


def test_remainder_does_not_depend_on_rule_order():
    """`override` may put a replacement anywhere, and a reordering must not change this."""
    forward = RuleSet([side, column, nucleated])
    backward = RuleSet([nucleated, column, side])
    assert forward.remainder("LN_C5(L)_NCL") == backward.remainder("LN_C5(L)_NCL")


def test_a_fully_consumed_string_leaves_nothing():
    assert RuleSet([nucleated]).remainder("NCL") is None


def test_the_remainder_is_normalized_not_merely_shortened():
    """Runs of separators collapse and the margins are stripped, which falls out of
    rejoining the surviving tokens — note that also renders `(L)` as a plain separator."""
    assert RuleSet([column]).remainder("Tm2_A2(L)") == "Tm2_L"
    assert RuleSet([side]).remainder("_Tm2__(L)_") == "Tm2"


def test_consumes_accepts_a_callable_for_anything_irregular():
    @rule(consumes=lambda r: ["weird"] if "x" in (r["instance"] or "") else [])
    def odd(r):
        """fires on x, accounts for `weird`"""
        return "x" if "x" in (r["instance"] or "") else None

    assert odd.remainder("x_weird_Tm2") == "x_Tm2"


# --------------------------------------------------------------------------- #
# override
# --------------------------------------------------------------------------- #
def test_override_replaces_by_name_and_keeps_the_position():
    """Position matters because a later rule may read what an earlier one produced."""
    @rule(name="side")
    def other(r):
        """lowercased side"""
        return "l"

    merged = RuleSet([side, column]).override([other])
    assert merged.names == ["side", "column"]
    assert merged["side"].description == "lowercased side"


def test_override_appends_rules_that_are_new():
    @rule
    def glomerulus(r):
        """antennal lobe glomerulus"""
        return None

    assert RuleSet([side]).override([glomerulus]).names == ["side", "glomerulus"]


def test_override_leaves_both_operands_alone():
    """It returns a new set: a builtin RuleSet is module state, and mutating it would make
    one run's plugin leak into the next."""
    base = RuleSet([side, column])
    base.override([rule(lambda r: None, name="side")])
    assert base["side"] is side


def test_a_duplicate_inside_one_declaration_is_still_an_error():
    """Two rules with one name in a single module is a mistake; a module redefining a
    builtin is the point of loading one. Only the second is an override."""
    with pytest.raises(ValueError, match="duplicate rule names"):
        RuleSet([side, rule(lambda r: None, name="side")])


# --------------------------------------------------------------------------- #
# reserved names
# --------------------------------------------------------------------------- #
def test_a_reserved_name_must_declare_tag_false():
    """Left tagged, a rule named `label` mints `label:…` tags and never sets the property —
    a valid document that does nothing anyone asked for."""
    with pytest.raises(ValueError, match="reserved rule name"):
        rule(lambda r: "x", name="label")
    assert rule(lambda r: "x", name="label", tag=False).name == "label"


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def test_provenance_embeds_the_source_so_the_module_can_be_RECOVERED(tmp_path):
    """A hash identifies a file you already have; it cannot give you the file back. The
    text is what makes the record answer "what produced this" a month later."""
    import hashlib

    path = tmp_path / "r.py"
    path.write_text(MODULE)
    record = rules.provenance(rules.from_module(str(path)))

    assert record["names"] == ["side", "column"]
    assert [f["path"] for f in record["files"]] == [str(path.resolve())]
    entry = record["files"][0]
    assert entry["text"] == MODULE
    assert entry["sha256"] == hashlib.sha256(MODULE.encode()).hexdigest()


def test_provenance_records_what_each_rule_declared(tmp_path):
    path = tmp_path / "r.py"
    path.write_text(MODULE)
    declared = {r["name"]: r for r in rules.provenance(rules.from_module(str(path)))["rules"]}
    assert declared["column"]["multi"] is True
    assert declared["column"]["description"] == "Columns."
    assert declared["side"]["needs"] == ["instance"]


def test_provenance_skips_the_builtins_because_they_ship_with_the_package(tmp_path):
    """Embedding neu_mark's own source in every sidecar would say nothing."""
    from neu_mark.segprops_rules import BUILTIN

    assert rules.provenance(BUILTIN)["files"] == []


def test_provenance_can_be_asked_not_to_embed(tmp_path):
    path = tmp_path / "r.py"
    path.write_text(MODULE)
    entry = rules.provenance(rules.from_module(str(path)), embed=False)["files"][0]
    assert "text" not in entry and entry["sha256"]


def test_a_module_too_large_to_embed_says_so_rather_than_silently_omitting(tmp_path):
    path = tmp_path / "r.py"
    path.write_text(MODULE + "\n#" + "x" * 5000 + "\n")
    entry = rules.provenance(rules.from_module(str(path)), max_embed_bytes=100)["files"][0]
    assert "text" not in entry
    assert "max_embed_bytes=100" in entry["text_omitted"]
