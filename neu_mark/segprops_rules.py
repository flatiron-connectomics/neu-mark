"""The built-in facet rules, which are ordinary :mod:`neu_mark.rules` rules.

There is nothing privileged here. These are declared with the same ``@rule`` decorator a
dataset's own module uses, they are evaluated by the same code, and any of them can be
replaced by name — which is the point: **read this file as the worked example of what a
rules module looks like**, then write yours the same way.

What is deliberately absent is a **cell type parse**. An earlier version carried a `column`
rule here, matching things like ``_A2`` and ``_C5``; it was removed because there is no
generically correct column parser — the same pattern that finds a real optic-lobe column
also swallows part of a cell type, and which is which is a fact about the dataset, not
about the string. That knowledge belongs in a rules module next to the body lists. The
seam it left is the whole reason plugins exist:

    # wasp_rules.py
    from neu_mark import rule

    @rule(multi=True)
    def column(r):
        \"\"\"Optic lobe column, e.g. C2 or A1.\"\"\"
        return re.findall(r"_([A-Z]\\d+)(?=$|[_(])", r["instance"] or "")

The vocabulary each one recognizes is a **fact about Megaphragma sample3**, measured over
21,021 annotated bodies, not a general truth: `fragment` 7,109 bodies, `truncated` 6,625,
`CV` 795, `NCL` 196, `glia` 117, and the four noise tokens 394 between them. A different
dataset will want different rules, which is why these are overridable rather than baked in.
"""

from __future__ import annotations

import re

from .explore import normalize, split_tokens
from .rules import Rule, RuleSet, rule

#: Instance-string tokens that mean "not real data". A body whose instance carries one is
#: dropped from the output entirely rather than merely left untagged — it should not appear
#: in a viewer's segment list at all.
NOISE = ("irrelevant", "block", "chunk", "unknown")

SIDE = re.compile(r"\((L|R)\)")


@rule(drop=True)
def noise(r):
    """not real data: an `irrelevant`, `block`, `chunk` or `unknown` body"""
    lowered = [t.lower() for t in split_tokens(r["instance"])]
    return next((t for t in NOISE if t in lowered), None)


@rule
def side(r):
    """hemisphere, from the (L)/(R) in the name"""
    m = SIDE.search(normalize(r["instance"]) or "")
    return m.group(1) if m else None


@rule(needs=["type"])
def group(r):
    """cell type, from the curated `type` field"""
    return normalize(r["type"])


#: The two rules below normalize an ALIAS, so the tag they produce is not the token they
#: matched — `NCL` becomes `nucleated`, `CV` becomes `cervical`. Everywhere else the value
#: *is* the token, which is why `consumes` is derived by default and declared only here.
#: Without it, `RuleSet.remainder` would leave `NCL` and `CV` sitting in the residue.
NUCLEUS_TOKENS = ("ncl", "nucleus")
CERVICAL_TOKENS = ("cv",)


@rule(prefix="", consumes=NUCLEUS_TOKENS)
def nucleated(r):
    """cell body (nucleus) is in the volume"""
    # `NCL` and `nucleus` are the same thing spelled two ways.
    lowered = {t.lower() for t in split_tokens(r["instance"])}
    return "nucleated" if lowered & set(NUCLEUS_TOKENS) else None


@rule(prefix="", consumes=CERVICAL_TOKENS)
def cervical(r):
    """passes through the cervical connective"""
    lowered = {t.lower() for t in split_tokens(r["instance"])}
    return "cervical" if lowered & set(CERVICAL_TOKENS) else None


@rule(prefix="")
def glia(r):
    """glia rather than a neuron"""
    return "glia" if "glia" in {t.lower() for t in split_tokens(r["instance"])} else None


@rule(prefix="")
def fragment(r):
    """a fragment, not a completely traced cell"""
    lowered = {t.lower() for t in split_tokens(r["instance"])}
    return "fragment" if "fragment" in lowered else None


@rule(prefix="")
def truncated(r):
    """traced but cut off by the volume boundary"""
    lowered = {t.lower() for t in split_tokens(r["instance"])}
    return "truncated" if "truncated" in lowered else None


#: What ``--drop-glia`` swaps in for the ``glia`` rule above. A drop rule and a tag rule
#: share one namespace, so this is an *override* rather than a second mechanism: after it,
#: asking "is this body glia" has exactly one answer and one place that decides what to do
#: about it. Glia are KEPT by default — not connectome-relevant, but worth seeing.
DROP_GLIA = Rule(lambda r: glia.func(r), name="glia", drop=True,
                 description="glia rather than a neuron (excluded)")

#: Evaluated in declaration order, drops first. Override by name; see :meth:`RuleSet.override`.
BUILTIN = RuleSet([noise, side, group, nucleated, cervical, glia, fragment, truncated],
                  source="neu_mark.segprops_rules.BUILTIN")
