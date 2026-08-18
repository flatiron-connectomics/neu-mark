# em-annotation

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
mamba install -n em-lib -c flyem-forge -c conda-forge neuclease
pip install --no-deps -e ./em-annotation
```

## Usage

Every fetch takes an explicit `--bodies` list, because the dataset has ~80M label ids and
the great majority are single-voxel fragments nobody cares about. `select-bodies` is how
you build that list.

### Choosing the bodies

`select-bodies` ranks by **synapse count**, using the `labelsz` index DVID already
maintains — `AllSyn >= 10` returns 21,116 bodies in ~20 s on dvid.example.org. Point `--src` at
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
em-annot info --src dvid://dvid.example.org/93fdbc:main/synapses

# build the body list itself, from DVID's ranked synapse index (~20 s)
em-annot select-bodies \
    --src dvid://dvid.example.org/93fdbc:main/synapses \
    --min-synapses 10 --out 'bodies_{uuid:8}/' --dvid-locked

# synapses for a body list -> points.parquet + relationships.parquet
em-annot points \
    --src dvid://dvid.example.org/93fdbc:main/synapses \
    --bodies traced_neurons.csv \
    --out synapses_{uuid:8}/ --dvid-locked

# per-body annotations -> bodies.parquet  (a SEPARATE destination)
em-annot bodies \
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
| `points` | one row per element | `body`, `z`, `y`, `x`, `kind`, `conf`, `user`, … |
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

### The partner-resolution caveat

A DVID relationship points at a **coordinate, not a body**. The partner's body is
recovered by joining that coordinate against the elements table — which works because
elements are fetched *per body*, so every row already knows its own body. But an edge only
resolves when **both** endpoints were fetched, so the yield depends on how much of the
connectome `--bodies` covers. Measured against dvid.example.org, top-N bodies by presynapse
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

## Layering

Sits above em-volume-tools (`location` for every write, `dvid` for node resolution,
`ops.provenance` and `ops.naming`) and alongside em-seg-morpho, from which it imports
nothing.
