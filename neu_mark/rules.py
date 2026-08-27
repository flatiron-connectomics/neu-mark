"""The ``@rule`` decorator and ``RuleSet``: declaring how annotation strings become facets.

## The word "rule" means two things, and the difference is the whole design

- a **rule** (lower case) is any callable taking one row and returning a value, a sequence,
  or ``None``. That is the contract :mod:`neu_mark.explore` works to, and it is what you
  write in a notebook in the first five minutes: ``{"side": lambda r: ...}``.
- a :class:`Rule` (this module) is that same callable **plus its declarations** — what
  columns it reads, whether it is multi-valued, what it produces, what it is for.

Every ``Rule`` is a rule, so anything in ``explore`` accepts one. The reverse is not true,
and the things that need the declarations are the things that *write output*:
:func:`neu_mark.segprops.build` takes a :class:`RuleSet`, never a bare dict, because it
has to know which rules mint tags, which exclude a body, and what to call each facet in
the document it publishes. **Explore with the loose form; publish with the declared one.**

A rules module is **dataset content, not code** — it encodes which prefixes are neuropil
names rather than cell types, which suffixes are completeness markers, and which misspellings
mean what. It lives with the body lists, not in this repo, and is loaded by path with
:func:`from_module`.

## What the declarations buy

- **``needs``** — the columns the rule reads. Checked once, up front, so a missing column is
  an error naming the rule instead of an ``AttributeError`` eight thousand rows in.
- **``multi``** — whether the facet is genuinely multi-valued. Column labels are (a central
  complex neuron can innervate several); side is not. Declaring it catches the common slip
  of using ``re.findall`` where ``re.search`` was meant, which otherwise silently produces a
  one-element list that looks fine until it produces a two-element one.
- **``prefix``** — the facet namespace, defaulting to the rule's own name, so ``glomerulus``
  yields ``glomerulus:VA1v`` with nothing declared. ``prefix=""`` mints a bare tag, which is
  what a non-exclusive flag wants. The colon is added by the consumer, never by you.
- **``drop``** — the rule EXCLUDES the body it fires on, returning the reason. One mechanism
  for "this should not be in the output at all", rather than a special case per vocabulary.
- **``consumes``** — which tokens of the source string this rule accounts for, enabling
  :meth:`RuleSet.remainder`: the string with everything the rules recognized taken out. The
  default derives them from the values returned, so declare it only where a rule normalizes
  an **alias** and the two differ (``NCL`` -> ``nucleated``). See below.
- **``tag=False``** — the rule produces something other than a tag. Two names are reserved
  (:data:`RESERVED`): ``label`` sets what a viewer calls the segment, ``description`` its
  description. Any other ``tag=False`` rule is still evaluated and coverage-reported and
  simply emits nothing — the seam for a ``number`` or ``string`` property later.
- **the docstring** — becomes the ``description``, carried to the coverage report and to the
  ``tag_descriptions`` of every tag the rule mints, so the viewer explains itself.
- **declaration order** — a rule may read a column an earlier rule produced.

## Overriding, not redeclaring

:meth:`RuleSet.override` merges by name: same name replaces (in place, keeping position),
new names are appended. That is what lets a dataset module add one glomerulus rule without
restating the builtins it was happy with. Note the asymmetry with ``__init__``, which
*refuses* duplicate names — two rules with one name inside a single module is a mistake,
while a module redefining a builtin is the entire point of loading one.

## What is left over

``RuleSet.remainder`` is the string with every recognized token removed — the raw material
for a cell-type parse that does not carry its own copy of every vocabulary the rules already
encode:

    >>> RULES.remainder("LN_C5(L)_NCL")
    'LN'
    >>> RULES.consumed("LN_C5(L)_NCL")
    {'side': ('L',), 'nucleated': ('NCL',), 'column': ('C5',)}

Three decisions in there, each avoiding a way this goes quietly wrong:

- **Token-granular, not substring removal.** ``side`` returns ``"L"``, and deleting that
  substring from ``LC10_C5(L)`` eats the ``L`` of ``LC10``. Tokens are the natural unit
  because these strings are ``_``/parenthesis-delimited. Rejoining the survivors with a
  single ``_`` is also what collapses separator runs and strips the margins.
- **A rule that does not read the string consumes nothing from it**, which falls out of
  ``needs`` rather than needing its own declaration. Without that, ``group`` — which returns
  the curated ``type``, often exactly the token a cell-type remainder wants to *keep* —
  would eat its own answer.
- **One union, not chained subtraction**, so the result cannot depend on rule order.
  :meth:`RuleSet.override` may put a replacement anywhere.

**What comes back is a residue, not a cell type.** On this dataset roughly three quarters of
non-empty remainders are a bare neuropil name (``AGNG``, ``VLNP``) — a body nobody has
identified — so turning one into a facet needs a vocabulary that can tell those apart. That
is a rule of its own, and a dataset's to write.

## Testing a rule

The method you will use most:

    >>> @rule
    ... def side(r):
    ...     '''Hemisphere, from the parenthesized (L)/(R).'''
    ...     m = re.search(r"\\((L|R)\\)", r["instance"] or "")
    ...     return m.group(1) if m else None
    >>> side.test("Tm2_A2(L)")
    'L'
    >>> side.test("AGNG")                     # did not fire -> None
    >>> RuleSet([side]).explain("LN_C5(L)_NCL")
    {'side': 'L'}

``test`` builds a one-row frame from keyword arguments and runs the rule through exactly the
path a real row takes — including missing values arriving as ``None`` rather than ``pd.NA``,
which is the footgun that makes ``r["instance"] or ""`` raise if you bypass it.
:meth:`RuleSet.explain` does the same for every rule at once, which is how you see what one
string would produce end to end.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from . import explore

logger = logging.getLogger(__name__)

#: Default column a rule reads, and the one `test` fills when given a bare string.
DEFAULT_COLUMN = "instance"

#: Reserved rule names, which produce a **property** rather than a tag. Declared
#: ``tag=False``; see :mod:`neu_mark.segprops` for what each one becomes. Any other
#: ``tag=False`` rule is still evaluated and coverage-reported, it just emits nothing —
#: which is what keeps the seam for `number`/`string` properties real rather than notional.
RESERVED = ("label", "description")


class Rule:
    """One named facet derived from a row. Callable exactly like the function it wraps."""

    def __init__(self, func: Callable, *, name: str | None = None,
                 needs: Sequence[str] | None = None, multi: bool = False,
                 tag: bool = True, prefix: str | None = None,
                 description: str | None = None, drop: bool = False,
                 consumes: Iterable[str] | Callable | None = None):
        self.func = func
        self.name = name or getattr(func, "__name__", "rule")
        self.needs = tuple(needs) if needs else (DEFAULT_COLUMN,)
        self.multi = bool(multi)
        #: Which tokens of the source string this rule ACCOUNTS FOR, for :meth:`remainder`.
        #: ``None`` derives them from the values the rule returns, which is right whenever
        #: the value is the token — declare it only where the two differ, i.e. where the
        #: rule normalizes an alias (``NCL`` -> ``nucleated``). A callable takes the row.
        self._consumes = consumes
        #: Whether this rule EXCLUDES the body it fires on. Its return value is the reason,
        #: counted in the build report — so a drop rule reads like any other rule and its
        #: coverage number is "how many bodies did this remove".
        self.drop = bool(drop)
        if self.drop and self.multi:
            raise ValueError(
                f"rule {self.name!r} is declared drop=True and multi=True. A body is either "
                f"excluded or not, so a drop rule returns one reason or None.")
        #: Whether this facet should become neuroglancer tags. Only ONE `tags` property is
        #: allowed per source, so every tagged facet pools into it under `prefix`. A drop
        #: rule is never tagged: the bodies it fires on are not in the output to carry one.
        self.tag = False if self.drop else bool(tag)
        if self.name in RESERVED and self.tag:
            raise ValueError(
                f"{self.name!r} is a reserved rule name: it sets the segment's {self.name} "
                f"PROPERTY, not a tag. Declare it @rule(name={self.name!r}, tag=False). "
                f"Left tagged it would quietly mint {self.name}:… tags instead, which is a "
                f"valid document that simply never sets the property.")
        self.prefix = prefix if prefix is not None else self.name
        self.description = description or (func.__doc__ or "").strip().split("\n")[0]
        self.__doc__ = func.__doc__
        self.__name__ = self.name

    def _parts(self) -> tuple[str, str, str, str, str]:
        """``(name, kind, needs, extras, description)``, the columns of a rule's repr.

        Shared with :meth:`RuleSet.__repr__` so a listing can align them without either
        rebuilding the other's format — one place decides what a rule's fields are.
        """
        kind = "drop" if self.drop else "tag" if self.tag else "property"
        if self.tag:
            kind += f" {self.prefix}:" if self.prefix else " bare"
        extras = ["multi"] if self.multi else []
        if self._consumes is not None:
            extras.append("consumes=" + ("callable" if callable(self._consumes) else
                                         "[" + ", ".join(str(c)
                                                         for c in self._consumes) + "]"))
        return (self.name, kind, "needs=" + "+".join(self.needs), " ".join(extras),
                self.description)

    def __repr__(self) -> str:
        """Everything the rule declares, on one line, ending with what it is FOR.

        Fields are separated by ``|`` rather than spaces: every value here is itself a word
        or a list, so space-delimiting them ran ``tag bare needs=instance consumes=…``
        together as one stream with nothing to say where a field ended.

        Worth the effort because this is what shows up in a traceback, in a debugger and
        when a terminal echoes a bare rule — all places where "which rule is this and what
        does it do" is the only question being asked.
        """
        name, kind, needs, extras, description = self._parts()
        fields = " | ".join(p for p in (name, kind, needs, extras) if p)
        return f"<Rule {fields} — {description}>" if description else f"<Rule {fields}>"

    def __call__(self, row) -> Any:
        return self.func(row)

    def _invoke(self, row) -> Any:
        """Call the function, turning an UNDECLARED column read into a legible error.

        ``needs`` is checked up front by :meth:`RuleSet.require`, but that can only check
        what a rule declares. A rule reading a column it never mentioned fails nowhere until
        the one row that lacks it — as a bare ``KeyError: 'type'``, naming neither the rule
        nor the fix, which is the failure ``needs`` exists to prevent.
        """
        try:
            return self.func(row)
        except KeyError as exc:
            column = exc.args[0] if exc.args else None
            if not isinstance(column, str) or column in self.needs:
                raise
            raise KeyError(
                f"rule {self.name!r} read the column {column!r}, which it does not "
                f"declare — so nothing could check for it up front. Write "
                f"@rule(needs={[*self.needs, column]!r}). Currently declared: "
                f"{', '.join(self.needs)}.") from exc

    def values(self, row) -> tuple:
        """The rule's output as a tuple, validated against ``multi``."""
        got = explore._as_tuple(self._invoke(row))
        if not self.multi and len(got) > 1:
            raise ValueError(
                f"rule {self.name!r} is declared single-valued but returned {len(got)} "
                f"values {got!r}. Either it should use re.search rather than re.findall, "
                f"or the facet really is multi-valued — in which case declare "
                f"@rule(multi=True), which is free: neuroglancer's `tags` property is a "
                f"list per segment anyway.")
        return got

    def scalar(self, row) -> Any:
        """The rule's single value, or ``None``. For the single-valued kinds: ``drop``
        reasons and the reserved ``label`` rule, where a tuple would only be unwrapped."""
        got = self.values(row)
        return got[0] if got else None

    def consumed(self, row) -> tuple[str, ...]:
        """The tokens of the source string this rule accounts for.

        **A rule that does not read the source string consumes nothing from it**, and that
        falls out of ``needs`` rather than needing its own declaration. It is also the trap
        that makes a naive version wrong: ``group`` returns the curated ``type``, often
        ``Tm2``, which is *exactly* the token a cell-type remainder is trying to keep. Left
        to consume, it would eat its own answer and leave an empty core.
        """
        if DEFAULT_COLUMN not in self.needs:
            return ()
        if self._consumes is None:
            candidates = self.values(row)
        elif callable(self._consumes):
            candidates = explore._as_tuple(self._consumes(row))
        else:
            candidates = tuple(self._consumes)
        wanted = {str(c).lower() for c in candidates}
        return tuple(t for t in explore.split_tokens(row[DEFAULT_COLUMN])
                     if t.lower() in wanted)

    def remainder(self, _value: str | None = None, /, **fields) -> str | None:
        """The source string with this rule's tokens removed. ``rule.remainder(s)`` reads
        like the question "what is left after this rule has had its say?".

        Consumption is **token-granular** — the strings are ``_``/parenthesis-delimited, so
        a token is the natural unit and it sidesteps the trap that sinks substring removal:
        ``side`` returns ``"L"``, and deleting that substring from ``LC10_C5(L)`` eats the
        ``L`` of ``LC10``. Rejoining the survivors with a single ``_`` is what collapses runs
        of separators and strips them from the margins; note it also renders ``(L)`` as a
        plain token, so the result is normalized rather than merely shortened.
        """
        row = _row({DEFAULT_COLUMN: _value, **fields} if _value is not None else fields,
                   needs=self.needs)
        drop = {t.lower() for t in self.consumed(row)}
        kept = [t for t in explore.split_tokens(row[DEFAULT_COLUMN])
                if t.lower() not in drop]
        return "_".join(kept) or None

    def test(self, _value: str | None = None, /, **fields) -> Any:
        """Evaluate on one synthetic row. ``rule.test("Tm2_A2(L)")`` fills ``instance``.

        Returns the rule's own return value, not a tuple, so it reads like the function —
        but it goes through the real row-building path, so missing fields arrive as ``None``
        exactly as they would in a run.
        """
        if _value is not None:
            fields = {DEFAULT_COLUMN: _value, **fields}
        row = _row(fields, needs=self.needs)
        return self._invoke(row)


def rule(func: Callable | None = None, /, *, needs: Sequence[str] | None = None,
         multi: bool = False, tag: bool = True, prefix: str | None = None,
         name: str | None = None, description: str | None = None,
         drop: bool = False, consumes: Iterable[str] | Callable | None = None):
    """Declare a function as a rule. Usable bare (``@rule``) or called (``@rule(...)``)."""
    def wrap(f: Callable) -> Rule:
        return Rule(f, name=name, needs=needs, multi=multi, tag=tag, prefix=prefix,
                    description=description, drop=drop, consumes=consumes)

    return wrap(func) if func is not None else wrap


def _row(fields: Mapping[str, Any], *, needs: Iterable[str] = ()) -> pd.Series:
    """A single row as a rule sees it: declared columns present, missing ones ``None``."""
    data = {str(k): v for k, v in fields.items()}
    for column in needs:
        data.setdefault(column, None)
    return pd.Series(data, dtype="object")


class RuleSet:
    """An ordered collection of rules, plus the reports for iterating on them."""

    def __init__(self, rules: Iterable[Rule] | Mapping[str, Callable],
                 *, keep: Sequence[str] | None = None, source: str | None = None):
        if isinstance(rules, Mapping):
            items = [r if isinstance(r, Rule) else Rule(r, name=n)
                     for n, r in rules.items()]
        else:
            items = list(rules)
        for r in items:
            if not isinstance(r, Rule):
                raise TypeError(f"{r!r} is not a Rule; decorate it with @rule")
        names = [r.name for r in items]
        duplicated = {n for n in names if names.count(n) > 1}
        if duplicated:
            raise ValueError(f"duplicate rule names: {', '.join(sorted(duplicated))}")
        self.rules = items
        #: Source columns that survive into a published layer. Explicit rather than a
        #: drop-list so nothing unwanted reaches a viewer by being forgotten.
        self.keep = tuple(keep) if keep is not None else ()
        self.source = source

    def __repr__(self) -> str:
        """The rules in evaluation order, one per line, **aligned into columns**.

        A listing is read down a column — "which of these are drop rules", "which declare
        consumption" — and ragged single-line reprs defeat that however well delimited each
        one is on its own. So the fields come from :meth:`Rule._parts` and are padded to the
        widest in the set; the ``<Rule …>`` wrapper is dropped because inside a RuleSet
        listing it is the same six characters on every line.

        Echoing a bare ``RULES`` therefore answers "what is in here and what does each one
        do" without calling anything. :meth:`describe` is the same information as a frame,
        for when you want to sort or filter it.
        """
        where = f", from {self.source}" if self.source else ""
        header = f"<RuleSet {len(self.rules)} rules{where}>"
        rows = [r._parts() for r in self.rules]
        if not rows:
            return header
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        lines = []
        for name, kind, needs, extras, description in rows:
            fields = "  ".join(text.ljust(width) for text, width in
                               zip((name, kind, needs, extras), widths))
            # Strip the padding only when nothing follows it — otherwise a rule with no
            # `extras` would start its description in a different column from its neighbours,
            # which is the alignment this method exists to provide.
            lines.append(f"  {fields}  — {description}" if description
                         else f"  {fields}".rstrip())
        return "\n".join([header, *lines])

    def _repr_html_(self) -> str:
        """Render as :meth:`describe`'s table in a notebook, via IPython's display protocol.

        So a bare ``RULES`` is a frame where a frame reads better and the plain reprs in a
        terminal — **without any method returning a different type depending on where it
        runs**. That distinction is the whole point: a value whose type depends on the
        frontend breaks the moment notebook code is moved into a module, which is exactly
        the path a rules module takes. Here the object is the same everywhere and only its
        *rendering* differs, which is what this protocol is for.
        """
        return self.describe().to_html(index=False)

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __getitem__(self, name: str) -> Rule:
        for r in self.rules:
            if r.name == name:
                return r
        raise KeyError(f"no rule named {name!r}; have {', '.join(self.names)}")

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.rules]

    @property
    def tagged(self) -> list[Rule]:
        """The rules that mint tags, in declaration order."""
        return [r for r in self.rules if r.tag]

    @property
    def drops(self) -> list[Rule]:
        """The rules that exclude a body. Evaluated FIRST, so an excluded body's tag rules
        never run and cannot skew the coverage numbers for the bodies that survive."""
        return [r for r in self.rules if r.drop]

    def override(self, other: "RuleSet | Iterable[Rule] | Mapping[str, Callable]",
                 ) -> "RuleSet":
        """This set with ``other``'s rules replacing same-named ones, others appended.

        Distinct from the duplicate check in ``__init__``, and deliberately so: two rules
        with one name *inside a single declaration* is a mistake, while a rules module
        redefining ``side`` is the entire point of loading one. Without this, adding a
        glomerulus rule would mean re-declaring every builtin you did not want to touch.

        A replaced rule keeps the **position** of the one it replaced, so the declaration
        order a later rule may depend on is not reshuffled by an override.
        """
        incoming = other if isinstance(other, RuleSet) else RuleSet(other)
        by_name = {r.name: r for r in incoming.rules}
        replaced = [n for n in self.names if n in by_name]
        merged = [by_name.pop(r.name, r) for r in self.rules]
        merged.extend(by_name.values())
        if replaced:
            logger.info("rules module overrides the builtin rule(s): %s",
                        ", ".join(replaced))
        return RuleSet(merged, keep=incoming.keep or self.keep,
                       source=incoming.source or self.source)

    def as_mapping(self) -> dict[str, Callable]:
        """``{name: callable}``, for the :mod:`neu_mark.explore` functions."""
        return {r.name: r.values for r in self.rules}

    def require(self, frame: pd.DataFrame) -> None:
        """Check every rule's ``needs`` against a frame, naming the rule that is unmet."""
        missing = {r.name: [c for c in r.needs if c not in frame.columns]
                   for r in self.rules}
        missing = {k: v for k, v in missing.items() if v}
        if missing:
            detail = "; ".join(f"{k} needs {', '.join(v)}" for k, v in missing.items())
            raise KeyError(
                f"the frame is missing columns these rules declare: {detail}. It has "
                f"{', '.join(map(str, frame.columns))}.")

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Every rule over every row: one column per rule, values as tuples."""
        self.require(frame)
        return explore.apply_rules(frame, self.as_mapping())

    def describe(self) -> pd.DataFrame:
        """What every rule DECLARES — the counterpart to :meth:`coverage`, which is what
        they did to your data. One row per rule, in evaluation order.

        Deliberately not folded into :meth:`explain`. That answers "what would this string
        produce", and a no-argument call returning descriptions instead would make one
        method return two ``dict``\\ s of strings that are indistinguishable by shape: is
        ``{'side': 'L'}`` a value or a description? Worse, a loop passing a string that
        happens to be ``None`` would switch silently between the two.
        """
        return pd.DataFrame([{
            "rule": r.name,
            "kind": "drop" if r.drop else "tag" if r.tag else "property",
            "prefix": r.prefix,
            "multi": r.multi,
            "needs": ", ".join(r.needs),
            "consumes": ("derived" if r._consumes is None else
                         "callable" if callable(r._consumes) else
                         ", ".join(str(c) for c in r._consumes)),
            "description": r.description,
        } for r in self.rules])

    def coverage(self, frame: pd.DataFrame, *, top: int = 3) -> pd.DataFrame:
        """Per-rule coverage, with each rule's declared facts alongside."""
        self.require(frame)
        out = explore.coverage(frame, self.as_mapping(), top=top)
        meta = pd.DataFrame({"rule": self.names,
                             "multi": [r.multi for r in self.rules],
                             "tag": [r.tag for r in self.rules],
                             "drop": [r.drop for r in self.rules],
                             "description": [r.description for r in self.rules]})
        return out.merge(meta, on="rule", how="left")

    def unparsed(self, frame: pd.DataFrame, name: str, *,
                 column: str = DEFAULT_COLUMN) -> pd.DataFrame:
        """The strings one rule did not fire on, ranked by bodies. The iteration loop."""
        self.require(frame)
        return explore.unparsed(frame, self.as_mapping(), name, column=column)

    def consumed(self, _value: str | None = None, /, **fields) -> dict[str, tuple]:
        """``{rule: tokens}`` for every rule that accounted for part of the string."""
        row = _row({DEFAULT_COLUMN: _value, **fields} if _value is not None else fields,
                   needs={c for r in self.rules for c in r.needs})
        return {r.name: got for r in self.rules if (got := r.consumed(row))}

    def remainder(self, _value: str | None = None, /, **fields) -> str | None:
        """What no rule accounted for — the composite of every rule's :meth:`Rule.remainder`.

        This is the point of declaring consumption at all. The alternative is a hand-written
        parser carrying its own copy of every vocabulary the rules already encode, which is
        what the notebook did: add a flag to a rule, forget to add it there, and the flag
        starts appearing inside cell-type names with nothing to signal it.

        Removal is computed as one union rather than by chaining subtractions, so the result
        does not depend on rule order — which matters, since :meth:`override` may put a
        replacement anywhere and a reordering must not silently change the answer.

        **What comes back is a residue, not a cell type.** On this dataset roughly three
        quarters of non-empty remainders are a bare neuropil name (``AGNG``, ``VLNP``), i.e.
        a body nobody has identified. Deciding what that means is a rule of its own.
        """
        fields = {DEFAULT_COLUMN: _value, **fields} if _value is not None else fields
        row = _row(fields, needs={c for r in self.rules for c in r.needs})
        drop = {t.lower() for r in self.rules for t in r.consumed(row)}
        kept = [t for t in explore.split_tokens(row[DEFAULT_COLUMN])
                if t.lower() not in drop]
        return "_".join(kept) or None

    def explain(self, _value: str | None = None, /, **fields) -> dict:
        """Every rule's output for one synthetic row — "what would this string produce?"

        Single-valued rules report their scalar and multi-valued ones their tuple, so the
        result reads the way the facets are meant to.
        """
        if _value is not None:
            fields = {DEFAULT_COLUMN: _value, **fields}
        needs = {c for r in self.rules for c in r.needs}
        row = _row(fields, needs=needs)
        out = {}
        for r in self.rules:
            got = r.values(row)
            out[r.name] = got if r.multi else (got[0] if got else None)
        return out


#: Ceiling on the source text embedded in a provenance record. A rules module is a few KB;
#: past this it is something else, and a sidecar is not an archive.
MAX_EMBED_BYTES = 256 * 1024


def provenance(ruleset: RuleSet, *, embed: bool = True,
               max_embed_bytes: int = MAX_EMBED_BYTES) -> dict:
    """What produced this output, recorded so it can be answered a month later.

    Once the rules are external they — not this package — decide what the source says, so
    the version of the code that ran has to be pinned somewhere. A **hash alone cannot do
    that**: it identifies a file you already have, and lets you verify a candidate, but it
    cannot give you the file back. So the source text is embedded verbatim; that is what
    makes the record self-contained, with no dependency on the module still existing, on it
    ever having been committed, or on any repository being reachable.

    Git details are recorded **opportunistically** — a commit and remote if the module
    happens to sit in a work tree, along with whether it was dirty at the time. Nothing is
    committed on your behalf: a publishing run is the wrong moment to mutate a repository,
    and the module may well not be in one.

    Files are collected from the rules themselves (``__code__.co_filename``), not just from
    the loaded path, so a module importing helpers from a sibling pins those too. Anything
    inside the installed ``neu_mark`` package is skipped — the builtins are versioned with
    the code, and embedding them in every sidecar would say nothing.
    """
    import hashlib

    package = Path(__file__).resolve().parent
    paths: list[Path] = []
    if ruleset.source and Path(ruleset.source).is_file():
        paths.append(Path(ruleset.source).resolve())
    for r in ruleset.rules:
        code = getattr(r.func, "__code__", None)
        name = getattr(code, "co_filename", None)
        if not name:
            continue
        candidate = Path(name)
        if not candidate.is_file():
            continue
        candidate = candidate.resolve()
        if package in candidate.parents or candidate == package:
            continue
        if candidate not in paths:
            paths.append(candidate)

    files = []
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError as exc:                                    # pragma: no cover
            files.append({"path": str(path), "error": str(exc)})
            continue
        entry = {"path": str(path), "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}
        if embed and len(data) <= max_embed_bytes:
            entry["text"] = data.decode("utf-8", errors="replace")
        elif embed:
            entry["text_omitted"] = (
                f"{len(data)} bytes exceeds max_embed_bytes={max_embed_bytes}")
        git = _git_details(path)
        if git:
            entry["git"] = git
        files.append(entry)

    return {
        "source": ruleset.source,
        "names": ruleset.names,
        "keep": list(ruleset.keep),
        "rules": [{"name": r.name, "description": r.description, "needs": list(r.needs),
                   "multi": r.multi, "tag": r.tag, "drop": r.drop, "prefix": r.prefix}
                  for r in ruleset.rules],
        "files": files,
    }


def _git_details(path: Path) -> dict | None:
    """Commit, remote and dirty flag for a file that happens to be in a work tree.

    Best-effort by design: git may be absent, the file may be outside any repository, and
    neither is a reason to fail a run that is otherwise fully recorded by the embedded text.
    """
    import subprocess

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(("git", "-C", str(path.parent), *args),
                                  capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):             # pragma: no cover
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if not commit:
        return None
    out = {"commit": commit}
    for key, args in (("remote", ("config", "--get", "remote.origin.url")),
                      ("path", ("ls-files", "--full-name", "--", str(path)))):
        value = run(*args)
        if value:
            out[key] = value
    status = run("status", "--porcelain", "--", str(path))
    # `status` is "" for a clean file and non-empty for a modified or untracked one, so the
    # flag says whether the recorded commit actually describes the bytes that ran.
    out["dirty"] = bool(status)
    return out


def from_module(path: str | Path, *, variable: str = "RULES") -> RuleSet:
    """Load a rules module by file path.

    Loaded by path rather than imported by name because a rules module is dataset content
    that lives next to the body lists, not something installed. If it defines ``RULES`` that
    is used; otherwise every ``@rule`` found at module level is collected in declaration
    order, so a module can be nothing but decorated functions.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no rules module at {path}")
    spec = importlib.util.spec_from_file_location(f"neu_mark_rules_{path.stem}", path)
    if spec is None or spec.loader is None:                       # pragma: no cover
        raise ImportError(f"cannot load a module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered so dataclasses/pickle inside the module behave, and removed on failure so a
    # broken edit does not leave a half-initialised module shadowing the next attempt.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    declared = getattr(module, variable, None)
    if isinstance(declared, RuleSet):
        declared.source = str(path)
        return declared
    if declared is not None:
        return RuleSet(declared, keep=getattr(module, "KEEP", None), source=str(path))
    found = [v for v in vars(module).values() if isinstance(v, Rule)]
    if not found:
        raise ValueError(
            f"{path} defines no rules: expected a `{variable}` list/RuleSet, or functions "
            f"decorated with @rule at module level.")
    return RuleSet(found, keep=getattr(module, "KEEP", None), source=str(path))
