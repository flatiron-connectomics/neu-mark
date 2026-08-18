"""The ``@rule`` decorator and ``RuleSet``: declaring how annotation strings become facets.

A rules module is **dataset content, not code** — it encodes which prefixes are neuropil
names rather than cell types, which suffixes are completeness markers, and which misspellings
mean what. It lives with the body lists, not in this repo, and is loaded by path.

## Why a decorator rather than a plain dict of functions

``em_annotation.explore`` accepts a bare ``{name: callable}`` and always will, because that
is what you reach for in the first five minutes. The decorator adds the four things that
matter once rules are being maintained rather than tried out:

- **``needs``** — the columns the rule reads. Checked once, up front, so a missing column is
  an error naming the rule instead of an ``AttributeError`` eight thousand rows in.
- **``multi``** — whether the facet is genuinely multi-valued. Column labels are (a central
  complex neuron can innervate several); side is not. Declaring it catches the common slip
  of using ``re.findall`` where ``re.search`` was meant, which otherwise silently produces a
  one-element list that looks fine until it produces a two-element one.
- **the docstring** — carried through to the coverage report and, later, to the
  ``description`` member of the corresponding segment property, so the viewer explains
  itself.
- **declaration order** — a rule may read a column an earlier rule produced.

## Testing a rule

The point of the decorator you will use most:

    >>> @rule
    ... def side(r):
    ...     m = re.search(r"\\((L|R)\\)", r["instance"] or "")
    ...     return m.group(1) if m else None
    >>> side.test(instance="Tm2_A2(L)")
    'L'
    >>> side.test(instance="AGNG")            # did not fire
    >>> RULES.explain(instance="LN_C5(L)_NCL")
    {'side': 'L', 'flag': ('nucleated',), 'cell_type': 'LN_C5', ...}

``test`` builds a one-row frame from keyword arguments and runs the rule through exactly the
path a real row takes — including missing values arriving as ``None`` rather than ``pd.NA``,
which is the footgun that makes ``r["instance"] or ""`` raise if you bypass it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from . import explore

#: Default column a rule reads, and the one `test` fills when given a bare string.
DEFAULT_COLUMN = "instance"


class Rule:
    """One named facet derived from a row. Callable exactly like the function it wraps."""

    def __init__(self, func: Callable, *, name: str | None = None,
                 needs: Sequence[str] | None = None, multi: bool = False,
                 tag: bool = True, prefix: str | None = None,
                 description: str | None = None):
        self.func = func
        self.name = name or getattr(func, "__name__", "rule")
        self.needs = tuple(needs) if needs else (DEFAULT_COLUMN,)
        self.multi = bool(multi)
        #: Whether this facet should become neuroglancer tags. Only ONE `tags` property is
        #: allowed per source, so every tagged facet pools into it under `prefix`.
        self.tag = bool(tag)
        self.prefix = prefix if prefix is not None else self.name
        self.description = description or (func.__doc__ or "").strip().split("\n")[0]
        self.__doc__ = func.__doc__
        self.__name__ = self.name

    def __repr__(self) -> str:                                    # pragma: no cover
        return (f"<Rule {self.name} needs={list(self.needs)} multi={self.multi} "
                f"tag={self.tag}>")

    def __call__(self, row) -> Any:
        return self.func(row)

    def values(self, row) -> tuple:
        """The rule's output as a tuple, validated against ``multi``."""
        got = explore._as_tuple(self.func(row))
        if not self.multi and len(got) > 1:
            raise ValueError(
                f"rule {self.name!r} is declared single-valued but returned {len(got)} "
                f"values {got!r}. Either it should use re.search rather than re.findall, "
                f"or the facet really is multi-valued — in which case declare "
                f"@rule(multi=True), which is free: neuroglancer's `tags` property is a "
                f"list per segment anyway.")
        return got

    def test(self, _value: str | None = None, /, **fields) -> Any:
        """Evaluate on one synthetic row. ``rule.test("Tm2_A2(L)")`` fills ``instance``.

        Returns the rule's own return value, not a tuple, so it reads like the function —
        but it goes through the real row-building path, so missing fields arrive as ``None``
        exactly as they would in a run.
        """
        if _value is not None:
            fields = {DEFAULT_COLUMN: _value, **fields}
        row = _row(fields, needs=self.needs)
        return self.func(row)


def rule(func: Callable | None = None, /, *, needs: Sequence[str] | None = None,
         multi: bool = False, tag: bool = True, prefix: str | None = None,
         name: str | None = None, description: str | None = None):
    """Declare a function as a rule. Usable bare (``@rule``) or called (``@rule(...)``)."""
    def wrap(f: Callable) -> Rule:
        return Rule(f, name=name, needs=needs, multi=multi, tag=tag, prefix=prefix,
                    description=description)

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

    def as_mapping(self) -> dict[str, Callable]:
        """``{name: callable}``, for the :mod:`em_annotation.explore` functions."""
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

    def coverage(self, frame: pd.DataFrame, *, top: int = 3) -> pd.DataFrame:
        """Per-rule coverage, with each rule's declared facts alongside."""
        self.require(frame)
        out = explore.coverage(frame, self.as_mapping(), top=top)
        meta = pd.DataFrame({"rule": self.names,
                             "multi": [r.multi for r in self.rules],
                             "tag": [r.tag for r in self.rules],
                             "description": [r.description for r in self.rules]})
        return out.merge(meta, on="rule", how="left")

    def unparsed(self, frame: pd.DataFrame, name: str, *,
                 column: str = DEFAULT_COLUMN) -> pd.DataFrame:
        """The strings one rule did not fire on, ranked by bodies. The iteration loop."""
        self.require(frame)
        return explore.unparsed(frame, self.as_mapping(), name, column=column)

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
    spec = importlib.util.spec_from_file_location(f"em_annotation_rules_{path.stem}", path)
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
