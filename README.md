# neu-mark

Annotations out of DVID, into columnar tables, and on into neuroglancer.

Two kinds of annotation:

- **point annotations** — synapses, and anything else in a DVID `annotation` instance
  (`labels_todo`, `bookmark_annotations`).
- **body annotations** — the per-body records in a `keyvalue` instance
  (`labels_annotations`): neuron name, status, who traced it.

Tables come first and are the durable artifact: analysis reads them, and the
neuroglancer layers are built from them, so a fetch is never repeated just to change how
something is displayed.

## Install

`neuclease` is a hard requirement for anything that talks to DVID and **cannot be a pip
dependency** — it needs `libdvid-cpp`, `vigra` and `dvidutils`, all conda-only on
flyem-forge:

```bash
mamba install -n neu-env -c flyem-forge -c conda-forge neuclease
pip install --no-deps -e ./neu-mark
```

## Usage

Every fetch takes an explicit `--bodies` list, because the dataset has ~80M label ids and
the great majority are single-voxel fragments nobody cares about. `select-bodies` is how
you build that list.

### Choosing the bodies

`select-bodies` ranks by **synapse count**, using the `labelsz` index DVID already
maintains — `AllSyn >= 10` returns 21,116 bodies in ~20 s on our dataset. Point `--src` at
either the `labelsz` instance or the annotation instance it indexes (`synapses`), and its
`Base.Syncs` is used to find the index.

Synapse count rather than voxel size, for three reasons: a body with no synapses is a
fragment; many of the largest-by-volume bodies are glia; and scanning every label's size is
~70 minutes (~80M labels at ~19k/s via `fetch_listlabels`, which pages serially by
`start = last_id + 1`).

**The threshold is on the total, not on pre and post separately.** That is a domain
constraint, not a simplification — sensory neurons may have no postsynapses, and a neuron
projecting outside the traced volume may have no presynapses, so requiring both drops
exactly the cells most worth looking at. `--min-pre` / `--min-post` exist when you do want
that; both default to off.

Rank from a **locked** node. Body ids change with proofreading, so the list and the
annotations should come from the same node — which is also why `select-bodies` writes a
provenance record naming it, rather than a bare list of ids.

```bash
# what am I pointing at, and which node would I get?
neu-mark info --src dvid://dvid.example.org/93fdbc:main/synapses

# build the body list itself, from DVID's ranked synapse index (~20 s)
neu-mark select-bodies \
    --src dvid://dvid.example.org/93fdbc:main/synapses \
    --min-synapses 10 --out 'bodies_{uuid:8}/' --dvid-locked

# synapses for a body list -> points.parquet + relationships.parquet
neu-mark points \
    --src dvid://dvid.example.org/93fdbc:main/synapses \
    --bodies traced_neurons.csv \
    --out synapses_{uuid:8}/ --dvid-locked

# per-body annotations -> bodies.parquet  (a SEPARATE destination)
neu-mark bodies \
    --src dvid://dvid.example.org/93fdbc:main/labels_annotations \
    --bodies traced_neurons.csv \
    --out body_annotations_{uuid:8}/ --dvid-locked
```

`--out` may be a local path or `s3://…`, and may carry `{uuid}` / `{uuid:N}` /
`{uuid:full}` / `{branch}` / `{instance}`, expanded from the resolved node. Naming an
export after the node it came from matters because a branch ref means a different node
tomorrow.

## Tables

| table | grain | key columns |
| --- | --- | --- |
| `selected_bodies` | one row per selected body | `body`, `pre`, `post`, `syn` |
| `points` | one row per element | `body`, `z`, `y`, `x`, `kind`, `conf`, `user`, `roi` |
| `relationships` | one row per relationship | `rel`, `z/y/x`, `to_z/to_y/to_x`, `from_body`, `to_body` |
| `connections` (`--connections`) | one row per distinct (tbar, psd) pair | `pre_*`, `post_*`, `pre_body`, `post_body` |
| `bodies` | one row per body | `body`, `status`, `instance` (= neuron name), `user`, `json` |

Coordinates are always **named** columns in `z, y, x` order. `tables.positions_zyx` and
`tables.positions_xyz` are the only places a positional coordinate array is built, and
therefore the only places the order is decided — a mirrored annotation is a valid
annotation in the wrong place, and nothing else would catch it.

`parquet` is the default format because csv cannot preserve an integer column that
contains **nulls**, and `to_body` is one by design — an unresolved partner is null. csv
has no types, so pandas infers on read, and a column mixing integers with blanks never
infers as an integer: measured here it returns as `str`, on older pandas as `float64`,
which silently rounds ids above 2^53. A *non*-nullable `uint64` column does survive csv
intact, which makes this the worse failure mode — the corruption is confined to the one
column nobody thinks to check. `--format csv` is available and warns, naming the columns
that will not survive.

### Which neuropil each synapse is in

`--rois` (or `points(..., rois=[...])`) adds a `roi` column to the points table, via
`neuclease`'s `determine_point_rois`. It samples a combined ROI volume at scale 5, so it is
cheap, and it takes `x`/`y`/`z` **columns** — which our tables have by name, so there is no
flip and no chance of one.

```bash
neu-mark points --src @synapses --bodies traced.csv --out syn/ \
    --rois 'ME(L),ME(R),LO(L),LO(R),AGNG,PGNG,SNP(L),SNP(R)'     # or --rois @neuropils
```

Only the **points** table gets it. A relationship spans two points that may be in different
neuropils, so one column on it would have to pick a side; a join back to `points` answers
either. `tables.body_roi_counts(points)` aggregates per body and kind.

**The ROI set is required — there is no "all".** The combined volume is built by writing each
ROI in turn, so passing every ROI on the node would attribute a point in `ME(L)` to whichever
of `ME(L)` / `OL(L)` / `all_neuropils` was written last. A name that is not an ROI instance
on the node is caught before any fetch, with close matches suggested.

Where ROIs do intersect the later one wins, and there is no principled tie-break — so this
**proceeds and measures it**, in the unit that decides whether to care: the volume is
unpacked a second time with the order reversed (the ROI *fetch* is not repeated, so this is
nearly free) and the two labellings compared. The report says how many of *your* synapses
change hands:

```
rois: 989 of 1919 synapses inside one of 3 ROIs (51.5%), 930 outside every one
  OL(L) (919), AGNG (70)
  1 ROI pairs intersect; 919 synapse(s) (47.89%) sit in an intersection and are
  attributed by ROI ORDER alone — {'OL(L) | ME(L)': 919}
```

`--strict-rois` refuses instead, for a strict partition; the dataset has deliberately
subtracted variants (`INP(-ATL)(L)`, `PENP(-AMMC)`, `VLNP(-AOTU)(L)`) for that.

### The partner-resolution caveat

A DVID relationship points at a **coordinate, not a body**. The partner's body is
recovered by joining that coordinate against the elements table — which works because
elements are fetched *per body*, so every row already knows its own body. But an edge only
resolves when **both** endpoints were fetched, so the yield depends on how much of the
connectome `--bodies` covers. Measured on our dataset, top-N bodies by presynapse
count:

| bodies | points | distinct pairs | both ends resolved |
| --- | --- | --- | --- |
| 20 | 6,148 | 23,652 | 0.2% |
| 100 | 26,748 | 75,901 | 2.0% |
| 400 | 80,479 | 226,334 | 4.6% |

So the commands report the match rate as a headline number, and unresolved partners are
kept as nulls rather than dropped. `--drop-unmatched` exists for once the rate is high:
dropping earlier turns "the body list did not cover this partner" into "this synapse does
not exist", which at low coverage presents a mostly-incomplete connectome as a complete
one.

## Reading the whole instance

```bash
neu-mark bodies --src @bodies --all --out 'all_bodies_{uuid:8}/' --dvid-locked
```

This **never calls `/keys`**, which is unreliable at this size — 58,394 keys took 52 s twice
and then returned `504 Gateway Time-out` from the proxy in front of DVID, and there is no
count endpoint to use instead. It reads a *cover of the key space* as bounded
`keyrangevalues` requests, so completeness comes from the cover being exhaustive rather than
from knowing a total. Values cost essentially nothing over keys (56.7 s against 54.3 s).

Two properties of DVID worth knowing if you use the underlying functions directly: its key
ranges are **inclusive at both ends**, so consecutive ranges overlap by one key and summing
per-range counts overcounts; and a range that is too big to finish inside the proxy's window
is **subdivided rather than retried**, because repeating it unchanged just fails again more
slowly.

## In a notebook

The same fetches, returning DataFrames and writing nothing:

```python
from neu_mark import source, select_bodies, points, body_annotations

src = source("@synapses", locked=True)        # node pinned once
sel = select_bodies("@counts", min_synapses=10, locked=True)   # body, pre, post, syn
pts, rels = points(src, sel.head(50))          # bodies: frame, Series, list or a path
ann = body_annotations("@bodies", sel.head(50), locked=True)
```

`neu_mark.ops` is the other half — same fetches, but writing tables and a provenance
record to a destination. Both go through the same functions, so what you see in a notebook
is what a run would write.

### Site defaults (optional)

An uncommitted TOML file supplies the server, uuid and instance names, so `@synapses` stands
in for a full URL. It is a **URL builder, not a fallback**: nothing consults it implicitly,
`@name` is required so a saved command visibly depends on it, and both the CLI and the
library print what a reference resolved to.

It is found as `neu-mark.toml` in the working directory **or any parent**, so one file at
the workspace root serves every repo and every notebook subdirectory beneath it — the way
`pyproject.toml` is found. `$NEU_MARK_CONFIG` overrides. There is deliberately **no
machine-wide location**: a hidden `~/.config` file applies to every shell on the host and is
the kind of invisible state that makes one command behave differently for two people.

```toml
# neu-mark.toml, at the workspace root
[dvid]
server = "dvid.example.org"
uuid   = "93fdbc:main"
locked = true

[instances]
synapses = "synapses"
bodies   = "labels_annotations"
counts   = "synapses_labelsz"

[roi_sets]
neuropils = ["ME(L)", "ME(R)", "LO(L)", "LO(R)", "AGNG", "PGNG"]
```

```console
$ neu-mark select-bodies --src @counts --min-synapses 400 --out 'sel/{uuid:8}' --dvid-locked
--src @counts  ->  dvid://dvid.example.org/93fdbc:main/synapses_labelsz
--out sel/{uuid:8}  ->  sel/821d68d2
```

## Inspecting the annotation strings

`instance` is the field that carries the information and it is dirty in bounded ways.
`neu_mark.explore` is a set of functions for looking at it — DataFrame in, DataFrame
out, no printing and no I/O, so it is meant for a notebook:

```python
from neu_mark import explore as ex

ex.instances(bodies)                        # distinct strings -> body counts
ex.tokens(bodies, drop=r"\((L|R)\)")        # the suffix vocabulary, side removed
ex.variants(bodies)                         # what normalization repaired ('truncated?')
ex.near(ex.tokens(bodies), "fragment")      # fuzzy candidates for misspellings
```

`near` and `variants` catch different things and you want both: whitespace and a trailing
`?` are repaired by `normalize`, so `truncated?` never appears in the token histogram and
`near` finds nothing left to fix. `variants` is what shows you the repair happened.

Once you have a guess about the vocabulary, try it out. Here a **rule** is nothing more
than a function of one row, and `rules` is a plain dict of them:

```python
guesses = {"cell_type": lambda r: (r["instance"] or "").split("_")[0] or None}

ex.apply_rules(bodies, guesses)                  # one column per rule
ex.coverage(bodies, guesses)                     # bodies, coverage, distinct, top
ex.unparsed(bodies, guesses, "cell_type")        # what it missed, ranked by bodies
ex.compare(bodies, "type", guesses["cell_type"]) # curated field vs your parse, row by row
```

A rule takes one row (a Series, with missing values as `None`) and returns a scalar, a
**sequence** for genuinely multi-valued facets — column labels are, since some central
complex neurons innervate several — or `None` for "did not fire", which stays visible
rather than becoming an error.

## Writing rules

A dict of lambdas is right for the first five minutes and wrong for anything you publish
from. `neu_mark.rules` is the same contract with the declarations added, and a
dataset-specific `RuleSet` is loaded by path from a module living with the dataset:

```python
# wasp_rules.py
import re
from neu_mark import explore as ex, rule

KEEP = ["instance", "type"]          # explicit; nothing else reaches a viewer

@rule
def side(r):
    """Hemisphere, from the parenthesized (L)/(R)."""
    m = re.search(r"\((L|R)\)", str(r["instance"] or ""))
    return m.group(1) if m else None

@rule(multi=True)
def column(r):
    """Column labels; genuinely multi-valued."""
    return re.findall(r"_([A-Z]\d+)(?=$|[_(])", str(r["instance"] or ""))

@rule(drop=True)                     # `noise` is a builtin, so this REPLACES its
def noise(r):                        # vocabulary rather than adding to it — name it
    """Bodies this dataset marks as not real data."""   # something else to have both
    tokens = {t.lower() for t in ex.split_tokens(r["instance"])}
    return next((t for t in ("irrelevant", "block") if t in tokens), None)
```

```python
from neu_mark import rules
RS = rules.from_module("wasp_rules.py")
```

What a declaration buys you, beyond the bare function:

| | |
|---|---|
| `needs` | the columns it reads, checked up front — a missing one is an error naming the rule, not an `AttributeError` 8,000 rows in |
| `multi` | checked, not decorative: a single-valued rule returning two values raises, which catches `re.findall` where `re.search` was meant |
| `prefix` | the facet namespace, defaulting to the rule's name — `glomerulus` gives `glomerulus:VA1v` with nothing declared. `prefix=""` mints a bare tag, right for a flag. The colon is added for you |
| `drop` | the rule **excludes** the body it fires on and returns the reason, which is counted in the report |
| `consumes` | which tokens of the instance this rule accounts for, so `RuleSet.remainder` can hand you what no rule recognized. Derived from the returned values by default — declare it only where a rule normalizes an alias (`NCL` → `nucleated`) |
| `tag=False` | produces something other than a tag. `label` and `description` are reserved and set those properties; anything else is computed and reported but emits nothing |
| the docstring | becomes the description — in the coverage report, and on every tag the rule mints, so the viewer explains itself |

Testing one rule, or all of them, on one string:

```python
RS["side"].test("Tm2_A2(L)")        # 'L'
RS["side"].test("AGNG")             # None — did not fire
RS.explain("LN_C5(L)_NCL")          # {'side': 'L', 'column': ('C5',), ...}

print(RS)                           # each rule's repr: declarations, then what it is for
RS.describe()                       # the same as a frame, to sort or filter
RS.coverage(bodies)                 # what they did to your data, plus what they declare
RS.unparsed(bodies, "column")       # what to fix next, ranked by bodies
```

`test` and `explain` build a synthetic row through the real path, so missing fields arrive as
`None` rather than `pd.NA` — the difference that makes `r["instance"] or ""` work.

### What no rule recognized

```python
RS.remainder("LN_C5(L)_NCL")     # 'LN'
RS.consumed("LN_C5(L)_NCL")      # {'side': ('L',), 'nucleated': ('NCL',), 'column': ('C5',)}
```

`remainder` strips every token the rules accounted for and rejoins the rest, which collapses
runs of separators and strips them from the margins. It is the raw material for a cell-type
parse that does **not** carry its own copy of every vocabulary the rules already encode —
the failure mode being that you add a flag to a rule, forget to add it to the parser, and
the flag starts appearing inside cell-type names.

Removal is token-granular rather than substring-based, because `side` returns `"L"` and
deleting that substring from `LC10_C5(L)` would eat the `L` of `LC10`. Rules whose value
*is* the matched token need no `consumes`; only the alias-normalizing ones do.

Be aware what comes back is a **residue, not a cell type**: on this dataset roughly three
quarters of non-empty remainders are a bare neuropil name (`AGNG`, `VLNP`) — a body nobody
has identified — so making a facet out of one needs a vocabulary that can tell those apart.

## Publishing rules as segment properties

`segment-properties` runs a built-in set of rules — side, the curated `type`, and the
flags. Your module's rules **override the built-in ones by name** and any others are added,
so adding one glomerulus rule does not mean restating the rest; `--rules-only` uses yours
alone. There is deliberately no built-in column or cell-type parse: the pattern that finds
`_A2` also eats part of a cell type, and which is which is a fact about your dataset.

```console
$ neu-mark segment-properties --src @bodies --dst ./seg --bodies sel/bodies.csv \
      --rules wasp_rules.py --dry-run
segments: 21116   tags: …   of 2111 allowed   document: … KiB
rules: wasp_rules.py
excluded: 394 — block (7), chunk (4), irrelevant (380), unknown (3)
  noise             394 bodies (  1.8%)      4 distinct   [drop]
  side            16238 bodies ( 76.9%)      2 distinct
  fragment         7109 bodies ( 33.8%)      1 distinct
  truncated        6625 bodies ( 31.5%)      1 distinct
  column               …                                          <- yours
DRY RUN — nothing written. ./seg is untouched.
```

(Counts are from the ≥10-synapse body list; the elided ones depend on your rules.)

Run `--dry-run` first: it builds and reports without writing anything, which `--no-link`
does not — that still writes the document. A rule that fires on 2% of bodies looks exactly
like a facet the data does not have, and only the coverage number tells them apart.

Two guards, because a bad rule produces a *valid* source that no viewer can use.

`--max-tags` caps the vocabulary and takes either form: **below 1 it is a fraction of the
segments written, 1 or above an absolute count.** The default is `0.10`, because the honest
size is dataset-dependent — 11,752 distinct `type` values over ~165k male-CNS bodies is 7%
and entirely reasonable, while the same number over 21k bodies means a rule is handing back
the instance string. The error names the rule with the most distinct values.

The branch is chosen by value, never by whether you wrote a decimal point: `1` and `1.0` are
the same number to TOML and to JSON, so a setting meaning "all of them" would come back from
a config meaning "one tag". A converted fraction is floored at 100 tags so a short list is
not judged by a ratio; an explicit count is a statement and is respected as given, floor or
no floor.

Distinct facets are capped separately, which catches the other failure: tags that are each
their own field can be a small vocabulary and still leave nothing to group on.

Because the rules now decide what the source says, the provenance sidecar records them —
the resolved path, a sha256, git details if the module happens to be in a work tree, and
**the module source verbatim**. A hash lets you verify a file you already have; the text is
what answers "what produced this" when nobody can find the module a month later.

## Layering

Sits above neu-vol (`location` for every write, `dvid` for node resolution,
`ops.provenance` and `ops.naming`) and alongside neu-morpho, from which it imports
nothing.
