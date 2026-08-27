"""Building the neuroglancer segment_properties document.

The format's constraints are the interesting part: exactly one `tags` property, tag values as
increasing indices, no spaces, case-insensitive matching. A document can satisfy all of that
and still be wrong in ways only a viewer shows, so what is testable is pinned here.
"""

import re

import pandas as pd
import pytest

from neu_mark import rule, segprops

BODIES = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4, 5, 6, 7], dtype="uint64"),
    "instance": pd.Series(["Tm2_A2(L)", "VLNP(R)_fragment", "LN_C5(L)_NCL",
                           "PGNG_CV", "irrelevant", "glia", None], dtype="string"),
    "type": pd.Series(["Tm2", None, "LN", None, None, None, None], dtype="string"),
})

COUNTS = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4, 6, 7], dtype="uint64"),
    "pre": [10, 0, 5, 1, 0, 0], "post": [90, 3, 20, 2, 1, 0],
    "syn": [100, 3, 25, 3, 1, 0],
})


def _built(**kw):
    return segprops.build(BODIES, **kw)


def _props(info):
    return {p["id"]: p for p in info["inline"]["properties"]}


# --------------------------------------------------------------------------- #
# the pd.NA trap, twice bitten
# --------------------------------------------------------------------------- #
def test_a_missing_type_produces_no_group_tag():
    """The bug this exists for: `pd.NA is not None` is True and `str(pd.NA)` is the truthy
    "<NA>", so a hand-rolled check tagged 16,606 real bodies `group-<na>` — and it read as
    100% coverage of a field that is populated on 5%."""
    info = _built()["info"]
    tags = _props(info)["tags"]["tags"]
    assert not [t for t in tags if "na" == t or "<na>" in t or "nan" in t]
    assert "group:Tm2" in tags and "group:LN" in tags
    assert sum(1 for t in tags if t.startswith("group:")) == 2


def test_group_coverage_matches_the_populated_field():
    report = _built()["report"]
    # two of the five surviving bodies have a `type`
    assert report["coverage"]["group"]["bodies"] == 2


def test_normalize_tag_handles_every_flavour_of_missing():
    assert segprops.normalize_tag(pd.NA) is None
    assert segprops.normalize_tag(None) is None
    assert segprops.normalize_tag(float("nan")) is None
    assert segprops.normalize_tag("   ") is None


# --------------------------------------------------------------------------- #
# what the format requires
# --------------------------------------------------------------------------- #
def test_at_most_one_label_and_one_tags_property():
    props = _built()["info"]["inline"]["properties"]
    for kind in ("label", "tags", "description"):
        assert sum(1 for p in props if p["type"] == kind) <= 1, kind


def test_tag_values_are_increasing_indices_not_strings():
    info = _built()["info"]
    tags = _props(info)["tags"]
    for row in tags["values"]:
        assert all(isinstance(i, int) for i in row)
        assert row == sorted(row), row
        assert len(row) == len(set(row))
        assert all(0 <= i < len(tags["tags"]) for i in row)


def test_tag_descriptions_are_parallel_to_tags():
    tags = _props(_built()["info"])["tags"]
    assert len(tags["tag_descriptions"]) == len(tags["tags"])


def test_the_tags_property_carries_no_description_member():
    """The spec forbids it there, though it is allowed on the others."""
    tags = _props(_built()["info"])["tags"]
    assert "description" not in tags


def test_tags_have_no_spaces_and_no_leading_hash():
    tags = _props(_built()["info"])["tags"]["tags"]
    assert all(" " not in t and not t.startswith("#") for t in tags)


def test_tag_case_is_preserved():
    """Cell types are conventionally cased, and folding `Tm2` to `tm2` costs readability
    in the viewer and in every group-by. The case-insensitive matching that motivated
    folding is handled by REPORTING collisions instead — see below."""
    tags = _props(_built()["info"])["tags"]["tags"]
    assert any(t != t.lower() for t in tags), tags
    assert "group:Tm2" in tags


def test_facet_tags_use_a_colon_so_a_reader_can_group_on_them():
    """With a hyphen, `side-l` is indistinguishable from a standalone flag like
    `fragment`, so every tag becomes its own boolean and no facet can be grouped on.
    Flags stay bare on purpose: they are not mutually exclusive."""
    tags = _props(_built()["info"])["tags"]["tags"]
    faceted = [t for t in tags if ":" in t]
    assert faceted, tags
    assert {t.split(":", 1)[0] for t in faceted} == {"group", "side"}
    for flag in ("fragment", "truncated", "nucleated", "cervical", "glia"):
        assert all(":" not in t for t in tags if t == flag)


def test_case_only_collisions_are_reported_not_folded():
    """Two tags differing only in case are ONE chip in the viewer, because matching is
    case-insensitive. Folding them would be lossy, so they are surfaced instead."""
    from neu_mark import segprops

    assert segprops.case_collisions(["group:Tm2", "group:TM2", "side:L"]) == {
        "group:tm2": ["group:Tm2", "group:TM2"]}
    assert segprops.case_collisions(["group:Tm2", "side:L"]) == {}


def test_ids_are_base_ten_strings():
    info = _built()["info"]
    assert all(isinstance(i, str) and i.isdigit() for i in info["inline"]["ids"])


def test_the_at_type_is_what_link_subresources_checks_for():
    from neu_vol.ops.subresources import SUBRESOURCE_TYPES

    assert _built()["info"]["@type"] == SUBRESOURCE_TYPES["segment_properties"]


# --------------------------------------------------------------------------- #
# facets
# --------------------------------------------------------------------------- #
def test_label_is_the_raw_instance_string():
    """Which is what lets this ship before a cell-type parse is settled."""
    info = _built()["info"]
    labels = dict(zip(info["inline"]["ids"], _props(info)["instance"]["values"]))
    assert labels["1"] == "Tm2_A2(L)"
    assert labels["3"] == "LN_C5(L)_NCL"


def test_a_body_with_no_instance_falls_back_to_its_id():
    """A blank label is indistinguishable from a missing property in the viewer."""
    info = _built()["info"]
    labels = dict(zip(info["inline"]["ids"], _props(info)["instance"]["values"]))
    assert labels["7"] == "7"


def test_each_facet_lands_on_the_right_body():
    def tags_of(body):
        info = _built()["info"]
        tags = _props(info)["tags"]
        i = info["inline"]["ids"].index(str(body))
        return {tags["tags"][j] for j in tags["values"][i]}

    assert tags_of(1) == {"group:Tm2", "side:L"}
    assert tags_of(2) == {"side:R", "fragment"}
    assert tags_of(3) == {"group:LN", "side:L", "nucleated"}
    assert tags_of(4) == {"cervical"}


def test_no_builtin_parses_a_column_out_of_the_instance():
    """The removed one. There is no generically correct column parser — the pattern that
    finds `_A2` also eats part of a cell type, and which is which is dataset knowledge. It
    lives in a rules module now, so the builtins must not quietly still do it."""
    tags = _props(_built()["info"])["tags"]["tags"]
    assert not [t for t in tags if t.startswith("col")]
    assert "column" not in segprops.build(BODIES)["report"]["coverage"]


def test_noise_is_excluded_entirely_not_merely_untagged():
    result = _built()
    assert "5" not in result["info"]["inline"]["ids"]        # 'irrelevant'
    assert result["report"]["excluded"]["irrelevant"] == 1


def test_glia_are_kept_and_tagged_by_default():
    result = _built()
    assert "6" in result["info"]["inline"]["ids"]
    tags = _props(result["info"])["tags"]
    i = result["info"]["inline"]["ids"].index("6")
    assert "glia" in {tags["tags"][j] for j in tags["values"][i]}


def test_glia_can_be_dropped_on_request():
    result = _built(keep_glia=False)
    assert "6" not in result["info"]["inline"]["ids"]
    assert result["report"]["excluded"]["glia"] == 1


def test_a_doubt_mark_normalizes_to_the_plain_completeness_tag():
    frame = pd.DataFrame({"body": pd.Series([1], dtype="uint64"),
                          "instance": ["Tm(R)_truncated?"]})
    info = segprops.build(frame)["info"]
    tags = _props(info)["tags"]
    assert {tags["tags"][i] for i in tags["values"][0]} == {"side:R", "truncated"}


def test_a_like_suffix_stays_in_the_label_and_is_not_tagged():
    frame = pd.DataFrame({"body": pd.Series([1], dtype="uint64"),
                          "instance": ["DPM-like(L)"]})
    result = segprops.build(frame)
    info = result["info"]
    assert _props(info)["instance"]["values"] == ["DPM-like(L)"]
    tags = _props(info)["tags"]
    assert {tags["tags"][i] for i in tags["values"][0]} == {"side:L"}


def test_everything_excluded_is_an_error_not_an_empty_document():
    frame = pd.DataFrame({"body": pd.Series([1], dtype="uint64"),
                          "instance": ["irrelevant"]})
    with pytest.raises(ValueError, match="every body was excluded"):
        segprops.build(frame)


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def test_numbers_are_uint32_because_float32_would_round_them():
    """float32 is exact only to 2^24 = 16,777,216, and real bodies reach 86 million
    voxels — so a float32 `voxels` property silently rounds."""
    sizes = {1: 86_315_764, 2: 4_510_070}
    info = _built(counts=COUNTS, sizes=sizes)["info"]
    for name in ("pre", "post", "syn", "voxels"):
        p = _props(info)[name]
        assert p["type"] == "number" and p["data_type"] == "uint32", name
    voxels = dict(zip(info["inline"]["ids"], _props(info)["voxels"]["values"]))
    assert voxels["1"] == 86_315_764            # exact, not 86315768


def test_number_values_are_parallel_to_ids_and_ordered_the_same():
    info = _built(counts=COUNTS)["info"]
    ids, pre = info["inline"]["ids"], _props(info)["pre"]["values"]
    assert len(pre) == len(ids)
    assert dict(zip(ids, pre))["1"] == 10 and dict(zip(ids, pre))["3"] == 5


def test_a_body_missing_from_counts_gets_zero_not_a_hole():
    info = _built(counts=COUNTS.iloc[:1])["info"]
    assert len(_props(info)["pre"]["values"]) == len(info["inline"]["ids"])


def test_numbers_are_omitted_when_not_supplied():
    info = _built()["info"]
    assert [p["id"] for p in info["inline"]["properties"]] == ["instance", "tags"]


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def test_it_writes_where_the_info_key_will_point(tmp_path):
    from neu_vol.location import read_json

    dst = str(tmp_path / "vol")
    key = segprops.write(dst, _built()["info"])
    assert key == "segment_properties/info"
    assert read_json(dst, "segment_properties", "info")["@type"] == segprops.AT_TYPE


def test_the_module_never_opens_a_file_directly():
    """Every write goes through location, or an s3 destination silently writes nothing."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(segprops))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "open" not in called


# --------------------------------------------------------------------------- #
# rules: adding, overriding, replacing
# --------------------------------------------------------------------------- #
def _tags_of(result, body):
    tags = _props(result["info"])["tags"]
    i = result["info"]["inline"]["ids"].index(str(body))
    return {tags["tags"][j] for j in tags["values"][i]}


def test_a_plugin_rule_is_added_alongside_the_builtins():
    """The column parser, put back where it belongs — as dataset content."""
    @rule(multi=True)
    def column(r):
        """optic lobe column"""
        return re.findall(r"_([A-Z]\d+)(?=$|[_(])", r["instance"] or "")

    result = segprops.build(BODIES, rules=[column])
    assert _tags_of(result, 1) == {"group:Tm2", "side:L", "column:A2"}
    assert "side" in result["report"]["coverage"]        # the builtins are still there


def test_a_plugin_rule_overrides_a_builtin_by_name_and_keeps_its_position():
    """Otherwise adding one glomerulus rule means redeclaring everything you did not want
    to touch. Position matters because a later rule may read what an earlier one produced."""
    @rule
    def side(r):
        """hemisphere, lowercased"""
        m = re.search(r"\((L|R)\)", r["instance"] or "")
        return m.group(1).lower() if m else None

    result = segprops.build(BODIES, rules=[side])
    assert _tags_of(result, 1) == {"group:Tm2", "side:l"}
    names = result["report"]["rules"]["names"]
    assert names.index("side") == 1                      # where the builtin sat
    assert names.count("side") == 1


def test_rules_only_drops_the_builtins():
    @rule
    def side(r):
        """hemisphere"""
        return "L" if "(L)" in (r["instance"] or "") else None

    result = segprops.build(BODIES, rules=[side], rules_only=True)
    assert result["report"]["rules"]["names"] == ["side"]
    # Nothing excluded either: `noise` is a builtin drop rule and went with the rest.
    assert result["report"]["excluded"] == {}
    assert "5" in result["info"]["inline"]["ids"]


def test_the_prefix_colon_is_added_once_however_the_rule_spells_it():
    """`Rule.prefix` is bare and defaults to the rule name, but the old constant carried
    the colon — so an author writing prefix="col:" out of habit must not get `col::A2`."""
    @rule(prefix="col:", name="col")
    def col(r):
        """column"""
        return "A2" if "A2" in (r["instance"] or "") else None

    assert "col:A2" in _tags_of(segprops.build(BODIES, rules=[col]), 1)


def test_a_drop_rule_excludes_the_body_and_counts_its_reason():
    @rule(drop=True)
    def tiny(r):
        """bodies the dataset marks as specks"""
        return "speck" if "PGNG" in (r["instance"] or "") else None

    result = segprops.build(BODIES, rules=[tiny])
    assert "4" not in result["info"]["inline"]["ids"]
    assert result["report"]["excluded"]["speck"] == 1
    assert result["report"]["coverage"]["tiny"]["kind"] == "drop"


def test_a_dropped_body_never_runs_the_tag_rules():
    """Or its facets would count toward coverage for a segment that is not in the output,
    and every percentage in the report would be against the wrong denominator."""
    seen = []

    @rule
    def watcher(r):
        """records which bodies reached the tag rules"""
        seen.append(r["instance"])
        return None

    segprops.build(BODIES, rules=[watcher])
    assert "irrelevant" not in seen                      # dropped by the `noise` builtin
    assert "Tm2_A2(L)" in seen


def test_drop_glia_is_an_override_not_a_second_exclusion_path():
    """One rule named `glia` decides both whether a body is glia and what happens to it."""
    kept = segprops.build(BODIES)
    dropped = segprops.build(BODIES, keep_glia=False)
    assert kept["report"]["coverage"]["glia"]["kind"] == "tag"
    assert dropped["report"]["coverage"]["glia"]["kind"] == "drop"
    assert dropped["report"]["excluded"]["glia"] == 1


def test_drop_glia_drops_whatever_the_CURRENT_glia_rule_matches():
    """A dataset that spells glia differently redefines the rule; --drop-glia must then
    exclude what THAT rule finds, not fall back to this package's vocabulary."""
    @rule(prefix="")
    def glia(r):
        """glia, as this dataset spells it"""
        return "glia" if "PGNG" in (r["instance"] or "") else None

    dropped = segprops.build(BODIES, rules=[glia], keep_glia=False)
    assert "4" not in dropped["info"]["inline"]["ids"]    # PGNG_CV, the redefined match
    assert "6" in dropped["info"]["inline"]["ids"]        # 'glia', which no rule now matches


# --------------------------------------------------------------------------- #
# the reserved rules, and the tag=False seam
# --------------------------------------------------------------------------- #
def test_a_label_rule_names_the_segments():
    @rule(name="label", tag=False)
    def label(r):
        """the name a viewer shows"""
        return (r["type"] or None) and f"cell {r['type']}"

    info = segprops.build(BODIES, rules=[label])["info"]
    labels = dict(zip(info["inline"]["ids"], _props(info)["instance"]["values"]))
    assert labels["1"] == "cell Tm2"
    # Falls back to the instance where the rule did not fire, and to the id where there is
    # no instance either — so no segment is ever nameless.
    assert labels["2"] == "VLNP(R)_fragment"
    assert labels["7"] == "7"


def test_a_label_keeps_its_spaces_because_it_is_not_a_tag():
    """`normalize_tag` hyphenates spaces to satisfy the tag format; a label is under no
    such constraint, and running one through it would rewrite `LC10 anterior` for nothing."""
    @rule(name="label", tag=False)
    def label(r):
        """the name a viewer shows"""
        return "LC10 anterior"

    info = segprops.build(BODIES, rules=[label])["info"]
    assert _props(info)["instance"]["values"][0] == "LC10 anterior"


def test_a_tagged_rule_named_label_is_refused():
    """It would mint `label:…` tags and silently never set the property — a valid document
    that does nothing anyone asked for."""
    with pytest.raises(ValueError, match="reserved"):
        @rule(name="label")
        def label(r):
            """oops"""
            return "x"


def test_a_quiet_rule_is_computed_and_reported_but_emits_nothing():
    """`tag=False` is the seam for a future number/string property. Evaluating and
    reporting it now is what keeps the seam real rather than a field nobody has ever set."""
    @rule(tag=False)
    def depth(r):
        """soma depth, one day a number property"""
        return 7 if "(L)" in (r["instance"] or "") else None

    result = segprops.build(BODIES, rules=[depth])
    assert result["report"]["coverage"]["depth"] == {
        "bodies": 2, "fraction": 2 / 6, "distinct": 1, "kind": "property",
        "description": "soma depth, one day a number property"}
    assert not [t for t in _props(result["info"])["tags"]["tags"] if "depth" in t]
    assert [p["id"] for p in result["info"]["inline"]["properties"]] == ["instance", "tags"]


def test_tag_descriptions_come_from_the_rule_that_minted_the_tag():
    """Which is what lets a BARE flag describe itself. The old prefix-scan could only find
    a description for a prefixed tag, so standalone flags needed a second lookup table."""
    info = _built()["info"]
    tags = _props(info)["tags"]
    described = dict(zip(tags["tags"], tags["tag_descriptions"]))
    assert described["fragment"] == "a fragment, not a completely traced cell"
    assert described["side:L"] == "hemisphere, from the (L)/(R) in the name"


def test_a_rule_needing_an_absent_column_warns_rather_than_raising():
    """A body list where nobody has a `type` has no `type` column at all, which is normal.
    Refusing it would reject a legitimate list; not saying so would look like a broken rule."""
    frame = BODIES.drop(columns=["type"])
    result = segprops.build(frame)
    assert result["report"]["absent_columns"] == ["type"]
    assert result["report"]["coverage"]["group"]["bodies"] == 0


# --------------------------------------------------------------------------- #
# the vocabulary guard
# --------------------------------------------------------------------------- #
def _many(n):
    """n bodies with n distinct instances, and nothing any builtin rule fires on — so the
    tag count under `echo` is exactly n and the limit is the only thing under test."""
    return pd.DataFrame({"body": pd.Series(range(1, n + 1), dtype="uint64"),
                         "instance": pd.Series([f"cell{i}" for i in range(n)],
                                               dtype="string")})


@rule
def echo(r):
    """a rule that hands the whole instance string back"""
    return r["instance"]


@rule
def coarse(r):
    """diverse but not unique: two bodies per value, so a fraction below 1 can admit it"""
    return "g" + str(int(r["instance"].removeprefix("cell")) // 2)


def test_a_rule_echoing_the_instance_string_is_refused():
    """Spec-legal and unusable: a chip per body, in a document every viewer downloads
    whole. The message has to name the offending rule, or there is nothing to act on."""
    with pytest.raises(ValueError, match=r"'echo' with 2000 distinct"):
        segprops.build(_many(2000), rules=[echo])


def test_raising_the_fraction_admits_a_genuinely_diverse_dataset():
    """11,752 distinct male-CNS `type` values over ~165k bodies is 7% and entirely
    reasonable; the same count over 21k bodies would mean a rule is echoing. `coarse` is
    50% here, so it fails the 10% default and passes at 60%."""
    with pytest.raises(ValueError, match="'coarse' with 1000 distinct"):
        segprops.build(_many(2000), rules=[coarse])
    result = segprops.build(_many(2000), rules=[coarse], max_tags=0.6)
    assert (result["report"]["tags"], result["report"]["tag_limit"]) == (1000, 1200)


def test_a_short_list_is_not_judged_by_the_fraction():
    """10% of 40 bodies is a meaningless threshold — on a short list every body can
    legitimately have its own type."""
    result = segprops.build(_many(40), rules=[echo])
    assert result["report"]["tags"] == 40
    assert result["report"]["tag_limit"] == segprops.MIN_TAG_ALLOWANCE


# --------------------------------------------------------------------------- #
# max_tags: fraction below 1, absolute count at 1 and above
# --------------------------------------------------------------------------- #
def test_a_value_below_one_is_a_fraction_and_one_or_above_is_a_count():
    """Chosen by VALUE, never by whether it was written with a decimal point: `1` and `1.0`
    are the same number to TOML and to JSON, so a setting meaning "all of them" would come
    back from a config meaning "one tag"."""
    assert segprops.resolve_tag_limit(0.10, 21116) == (2111, "fraction")
    assert segprops.resolve_tag_limit(2111, 21116) == (2111, "count")
    assert segprops.resolve_tag_limit(1, 21116) == (1, "count")
    assert segprops.resolve_tag_limit(1.0, 21116) == (1, "count")


def test_an_explicit_count_is_respected_and_never_raised_to_the_floor():
    """A count is a statement, not a ratio. Flooring it to MIN_TAG_ALLOWANCE would make the
    setting not do what it says — only the CONVERTED fraction gets the floor."""
    assert segprops.MIN_TAG_ALLOWANCE > 25
    assert segprops.resolve_tag_limit(25, 21116) == (25, "count")
    assert segprops.resolve_tag_limit(0.001, 21116)[0] == segprops.MIN_TAG_ALLOWANCE


def test_an_absolute_count_caps_the_vocabulary():
    with pytest.raises(ValueError, match="an explicit count of 25"):
        segprops.build(_many(2000), rules=[echo], max_tags=25)
    assert segprops.build(_many(2000), rules=[echo], max_tags=5000)["report"]["tags"] == 2000


def test_the_error_says_which_branch_produced_the_number():
    """Otherwise it is unclear whether the floor was in play."""
    with pytest.raises(ValueError, match=r"10% of 2000, floored at 100"):
        segprops.build(_many(2000), rules=[echo])


def test_a_non_positive_limit_is_refused(tmp_path):
    for bad in (0, -1, -0.5):
        with pytest.raises(ValueError, match="must be positive"):
            segprops.resolve_tag_limit(bad, 100)


def test_the_report_carries_both_the_setting_and_the_resolved_ceiling():
    """`0.1` and `2111` are the same ceiling on this list; a reader should not have to
    re-derive which branch ran."""
    report = segprops.build(_many(2000), rules=[coarse], max_tags=0.6)["report"]
    assert (report["max_tags"], report["tag_limit"]) == (0.6, 1200)


def test_too_many_facets_is_refused_even_though_the_vocabulary_is_tiny():
    """The hyphen incident: 483 tags that were each their own field. A small vocabulary,
    and nothing in the layer can be grouped on."""
    rules = [rule(lambda r, _i=i: "x", name=f"facet{i}") for i in range(segprops.MAX_FACETS + 2)]
    with pytest.raises(ValueError, match="distinct tag facets"):
        segprops.build(BODIES, rules=rules)


def test_the_report_carries_the_document_size():
    """The vocabulary's real cost is bytes every viewer downloads, so say what it is."""
    report = _built()["report"]
    assert report["bytes"] > 0
    assert report["facets"] == ["", "group", "side"]
