"""neu-mark: annotations out of DVID, into tables and neuroglancer layers.

Two kinds of annotation, one package:

- **point annotations** — synapses (``synapses``), and anything else stored in a DVID
  ``annotation`` instance (``labels_todo``, ``bookmark_annotations``).
- **body annotations** — the per-body records in a ``keyvalue`` instance
  (``labels_annotations``): neuron name, status, who traced it.

Both land as columnar tables first (:mod:`neu_mark.tables`, :mod:`neu_mark.io`)
and are published to neuroglancer second. The tables are the durable artifact: they are
what analysis reads, and they are what a precomputed layer is built from, so a fetch is
never repeated to change how something is displayed.

**Everything is driven by an explicit body list.** The dataset has a huge number of label
ids of which the overwhelming majority are fragments nobody cares about, and DVID offers
no cheap way to enumerate "the interesting ones" — the label-index and
whole-instance-threshold routes are both O(all bodies) and were measured as unusable. So a
body list is a required input rather than an optimization; see :mod:`neu_mark.bodies`.

Layering: this sits above neu-vol (``location`` for every write, ``dvid`` for node
resolution, ``ops.provenance`` for the record) and alongside neu-morpho. It imports
nothing from neu-morpho and nothing above.
"""

__version__ = "0.1.0"

#: Re-exported from `notebook` (fetches that return DataFrames and write nothing) and from
#: `rules`. Resolved through a module `__getattr__` (PEP 562) rather than imported here,
#: because `cli` reads `__version__` from this module and an eager import would make
#: `neu-mark --help` pay for pandas and neu-vol. Same reason and same mechanism as
#: neu-vol' lazy `start_dask`.
_LAZY = {
    "source": "notebook",
    "select_bodies": "notebook",
    "points": "notebook",
    "body_annotations": "notebook",
    "all_body_annotations": "notebook",
    "synapse_counts": "notebook",
    "body_ids": "notebook",
    "rule": "rules",
    "Rule": "rules",
    "RuleSet": "rules",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value          # cached, so this runs once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_LAZY})
