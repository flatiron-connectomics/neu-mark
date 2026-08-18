"""Looking at annotation strings before writing rules to parse them.

Every function here takes a DataFrame and returns a DataFrame. Nothing prints, nothing
reads or writes, and nothing needs a DVID connection — so this is usable straight from a
notebook, which is where the work actually happens: you look at the vocabulary, write a
rule, see what it missed, and go round again. ``em-annot instance-report`` is a thin
terminal wrapper over the same calls.

Named ``explore`` rather than ``inspect`` to avoid shadowing the standard library module of
that name inside this package.

A rule receives one row as a **pandas Series**, so both ``r["instance"]`` and ``r.instance``
work. Prefer the subscript form: attribute access goes through the Series' own namespace, so
a column called ``name``, ``size`` or ``dtype`` would silently return the Series attribute
instead of the data.

**Counts are by body, not by distinct string**, because the two disagree in ways that
matter: ``irrelevant`` appears as 237 whole strings but 376 bodies once it is also counted
as a trailing token. A vocabulary decision should be weighted by how much data it touches.

## The rule contract these functions assume

A rule is any callable taking one row and returning:

- a scalar (``"L"``, ``7``, ``True``) — one value for that body,
- a **sequence** — several values, for facets that are genuinely multi-valued: a central
  complex neuron may innervate more than one column, so ``column`` returns a list. This
  costs nothing downstream because neuroglancer's ``tags`` property is already a list per
  segment,
- ``None`` or an empty sequence — did not fire, which is not an error and must stay visible.

That is deliberately looser than a decorator so these functions are useful before any rule
framework exists: a plain ``dict`` of ``{name: function}`` works today.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

#: Junk seen on real strings: trailing/leading whitespace and a trailing '?' expressing
#: the annotator's doubt (`truncated?`). Removed before any vocabulary comparison, so that
#: `truncated?` and `truncated ` do not each become their own token.
_DOUBT = re.compile(r"[?!]+$")

#: What separates tokens in an instance string. ``)`` counts because the side is
#: parenthesized mid-string (`LN_C5(L)_NCL`), so ``)`` ends a token as surely as ``_``.
_SPLIT = re.compile(r"[_)(]+")


def normalize(value: Any) -> str | None:
    """Strip whitespace and trailing doubt marks. ``None`` for anything missing.

    The missing-value test goes through ``pd.isna``, not ``isinstance(value, float)``: a
    ``string``-dtype column holds ``pd.NA``, which is neither ``None`` nor a float, so a
    narrower check lets it through to ``str()`` and every report grows a literal ``"<NA>"``
    row. Guarded by ``is_scalar`` because ``pd.isna`` of a list returns an array.
    """
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    text = _DOUBT.sub("", str(value).strip()).strip()
    return text or None


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"no {column!r} column; frame has "
                       f"{', '.join(map(str, frame.columns))}")
    return frame[column].map(normalize)


def instances(frame: pd.DataFrame, column: str = "instance") -> pd.DataFrame:
    """Distinct strings and how many bodies carry each, most common first.

    The first thing to look at. On dvid.example.org this is ~1,700 rows for 21k bodies — a
    reviewable vocabulary rather than an open set, which is what makes rule-writing
    tractable at all.
    """
    counts = _series(frame, column).value_counts(dropna=True)
    return (counts.rename("bodies").rename_axis(column).reset_index())


def tokens(frame: pd.DataFrame, column: str = "instance", *,
           position: str = "trailing", drop: str | None = None) -> pd.DataFrame:
    """Token frequency, for hunting controlled vocabulary and its misspellings.

    ``position`` is ``"trailing"`` (the last token — where completeness markers, flags and
    region suffixes live), ``"leading"``, or ``"all"``. A string with no separator yields
    itself, so a bare ``AGNG`` shows up under ``trailing`` too; that is intentional, since
    those placeholder names are part of the vocabulary being catalogued.

    ``drop`` is a regex removed before tokenizing, for a facet you have already accounted
    for and do not want in the histogram. Parentheses separate tokens, so a parenthesized
    side makes ``L`` and ``R`` the *trailing* token of thousands of strings and buries the
    actual suffixes — ``drop=r"\\((L|R)\\)"`` is the usual first move.

    Returns ``token``, ``bodies``, ``strings`` — the second being how many *distinct*
    strings the token appears in, which separates a widespread suffix from one long tail.
    """
    if position not in ("trailing", "leading", "all"):
        raise ValueError(f"position must be trailing, leading or all; got {position!r}")
    pattern = re.compile(drop) if drop else None

    body_counts: dict[str, int] = {}
    string_counts: dict[str, int] = {}
    for text, n in _series(frame, column).value_counts(dropna=True).items():
        if pattern is not None:
            text = pattern.sub("_", text)
        parts = [p for p in _SPLIT.split(text) if p]
        if not parts:
            continue
        picked = (parts[-1:] if position == "trailing"
                  else parts[:1] if position == "leading" else parts)
        for token in dict.fromkeys(picked):          # de-dup within one string
            body_counts[token] = body_counts.get(token, 0) + int(n)
            string_counts[token] = string_counts.get(token, 0) + 1

    out = pd.DataFrame({"token": list(body_counts),
                        "bodies": list(body_counts.values()),
                        "strings": [string_counts[t] for t in body_counts]})
    return out.sort_values(["bodies", "token"], ascending=[False, True],
                          ignore_index=True)


def near(token_frame: pd.DataFrame, target: str, *, cutoff: float = 0.72,
         include_exact: bool = False) -> pd.DataFrame:
    """Tokens close enough to ``target`` to be misspellings of it, with their weight.

    The mechanism for the "catch the misspellings" problem. Fuzzy rather than exact
    because the real variants are unguessable in advance — on our dataset this surfaces
    ``truncated?`` (8 bodies) and a trailing-space ``truncated `` (3) that a controlled
    vocabulary would otherwise silently treat as unknown tokens.
    """
    rows = []
    for token, bodies, strings in token_frame[["token", "bodies", "strings"]].itertuples(
            index=False):
        if not include_exact and str(token).lower() == target.lower():
            continue
        ratio = difflib.SequenceMatcher(None, str(token).lower(), target.lower()).ratio()
        if ratio >= cutoff:
            rows.append({"token": token, "bodies": bodies, "strings": strings,
                         "ratio": round(ratio, 3)})
    return pd.DataFrame(rows, columns=["token", "bodies", "strings", "ratio"]).sort_values(
        ["bodies", "ratio"], ascending=False, ignore_index=True)


def variants(frame: pd.DataFrame, column: str = "instance") -> pd.DataFrame:
    """Raw strings that :func:`normalize` repaired, and what they became.

    Separate from :func:`near` because the two catch different things and only one of them
    is still visible afterwards. Whitespace and trailing ``?`` are *silently repaired* — so
    ``truncated?`` never reaches a token histogram as its own entry, and asking ``near`` to
    find it comes back empty. That repair is worth seeing: it is the annotator expressing
    doubt, and if the count grows, the vocabulary is drifting. ``near`` remains the tool for
    genuine misspellings (``fragmnet``), which normalization cannot fix.

    Returns ``raw``, ``normalized``, ``bodies`` for every string that changed.
    """
    if column not in frame.columns:
        raise KeyError(f"no {column!r} column")
    rows = []
    for raw, n in frame[column].value_counts(dropna=True).items():
        clean = normalize(raw)
        if clean != raw:
            rows.append({"raw": raw, "normalized": clean, "bodies": int(n)})
    return pd.DataFrame(rows, columns=["raw", "normalized", "bodies"]).sort_values(
        "bodies", ascending=False, ignore_index=True)


def _plain(row: pd.Series) -> pd.Series:
    """A row with pandas' missing values replaced by ``None``.

    Not a nicety. With a ``string`` dtype a missing cell is ``pd.NA``, and ``pd.NA or ""``
    *raises* rather than being falsey — so the obvious rule body ``str(r["instance"] or "")``
    blows up on the first body with no instance. Every rule author would hit that, so the
    framework hands over rows where missing means ``None``.
    """
    return row.where(row.notna(), None)


def _as_tuple(value: Any) -> tuple:
    """A rule's return value as a tuple of values, empty when it did not fire."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        # Scalars, including numpy scalars and bools. NaN counts as "did not fire".
        if isinstance(value, float) and pd.isna(value):
            return ()
        return (value,)
    return tuple(v for v in value if v is not None)


def apply_rules(frame: pd.DataFrame, rules: Mapping[str, Callable],
                *, key: str = "body") -> pd.DataFrame:
    """Run every rule over every row. One column per rule, values as tuples.

    Tuples rather than scalars because a facet may be multi-valued — column labels are,
    since a central complex neuron can innervate several. An empty tuple means the rule did
    not fire, which is distinct from a rule returning an empty string.
    """
    rows = [_plain(row) for _index, row in frame.iterrows()]
    out = {}
    if key in frame.columns:
        out[key] = frame[key].to_numpy()
    for name, rule in rules.items():
        try:
            out[name] = [_as_tuple(rule(row)) for row in rows]
        except Exception as exc:                                    # noqa: BLE001
            raise RuntimeError(
                f"rule {name!r} raised on some row: {type(exc).__name__}: {exc}. A rule "
                f"that cannot parse a value should return None, not raise — the whole "
                f"point is that unparsed values stay visible.") from exc
    # A positional index, so a caller's own index cannot misalign these against `frame`.
    return pd.DataFrame(out).reset_index(drop=True)


def coverage(frame: pd.DataFrame, rules: Mapping[str, Callable],
             *, top: int = 3) -> pd.DataFrame:
    """Per-rule: how often it fired, how many distinct values, and the commonest.

    The number to watch. A rule written against a dirty vocabulary under-fires silently,
    and low coverage is indistinguishable from "the data does not have this" unless it is
    put in front of you — the same failure class as the partner match rate in
    :mod:`em_annotation.tables`.
    """
    applied = apply_rules(frame, rules)
    total = len(frame)
    rows = []
    for name in rules:
        col = applied[name]
        fired = col.map(bool)
        values = [v for tup in col for v in tup]
        counts = pd.Series(values, dtype="object").value_counts() if values else pd.Series(
            dtype="int64")
        rows.append({
            "rule": name,
            "bodies": int(fired.sum()),
            "coverage": (float(fired.sum()) / total) if total else None,
            "values": len(values),
            "distinct": int(counts.size),
            "multi_valued": int((col.map(len) > 1).sum()),
            "top": ", ".join(f"{v} ({n})" for v, n in counts.head(top).items()),
        })
    return pd.DataFrame(rows)


def unparsed(frame: pd.DataFrame, rules: Mapping[str, Callable], rule: str, *,
             column: str = "instance") -> pd.DataFrame:
    """The strings a given rule did not fire on, ranked by how many bodies they cover.

    What you read between iterations: the top of this list is the next rule to write, or
    the vocabulary entry that is missing.
    """
    if rule not in rules:
        raise KeyError(f"no rule named {rule!r}; have {', '.join(rules)}")
    applied = apply_rules(frame, {rule: rules[rule]})
    missed = frame[~applied[rule].map(bool).to_numpy()]
    return instances(missed, column) if len(missed) else pd.DataFrame(
        columns=[column, "bodies"])


#: How a candidate value relates to a curated field. Ordered from strongest agreement to
#: outright conflict; `derived_is_longer` is the interesting one for column labels, since
#: it means the parse found something the curated field does not carry.
RELATIONS = ("exact", "case_only", "derived_is_longer", "field_is_longer",
             "substring", "unrelated", "field_missing", "derived_missing")


def _relation(field: str | None, derived: str | None) -> str:
    if field is None and derived is None:
        return "field_missing"
    if field is None:
        return "field_missing"
    if derived is None:
        return "derived_missing"
    if field == derived:
        return "exact"
    if field.lower() == derived.lower():
        return "case_only"
    if derived.startswith(field + "_") or derived.startswith(field + "("):
        return "derived_is_longer"
    if field.startswith(derived + "_") or field.startswith(derived + "("):
        return "field_is_longer"
    if field in derived or derived in field:
        return "substring"
    return "unrelated"


def compare(frame: pd.DataFrame, field: str, against: Callable, *,
            key: str = "body", column: str = "instance") -> pd.DataFrame:
    """Row-by-row: how a curated field relates to what a rule derives.

    Tidy output rather than a summary, so a notebook can do both — ``.relation
    .value_counts()`` for the table, ``[df.relation == "unrelated"]`` for the cases worth
    reading. Two fields disagreeing is not necessarily an error: on our dataset ``type``
    is often the *coarser* label (``LMC`` against an instance of ``L1/L3_D4(R)``), which is
    a difference of grain, not a mistake. Deciding which to trust is a judgement about the
    dataset and this function exists to inform it, not to make it.
    """
    rows = []
    for _index, raw_row in frame.iterrows():
        row = _plain(raw_row)
        got = _as_tuple(against(row))
        one = normalize(got[0]) if got else None
        raw_field = normalize(row.get(field))
        rows.append({
            key: row.get(key),
            field: raw_field,
            "derived": one,
            column: row.get(column),
            "relation": _relation(raw_field, one),
        })
    return pd.DataFrame(rows, columns=[key, field, "derived", column, "relation"])
