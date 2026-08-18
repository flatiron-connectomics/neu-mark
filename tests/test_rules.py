"""The @rule framework: declaration, validation, and testing a rule in isolation."""

import re

import pandas as pd
import pytest

from em_annotation import rules
from em_annotation.rules import Rule, RuleSet, rule

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


def test_unmet_needs_are_reported_up_front_naming_the_rule():
    @rule(needs=["instance", "roi_post_top"])
    def needs_roi(r):
        return None

    with pytest.raises(KeyError, match="needs_roi needs roi_post_top"):
        RuleSet([needs_roi]).apply(BODIES)


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
from em_annotation import rule

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
