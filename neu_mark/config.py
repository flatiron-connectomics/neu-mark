"""Site defaults for building DVID URLs, from an uncommitted TOML file.

The tension this has to resolve: implicit defaults are *good* for interactive exploration
and *bad* for anything that produces an artifact. A run whose source depended on a file
nobody else has is a run nobody can reproduce, and the whole point of the provenance record
is to say exactly which node an export came from.

So this is a **URL builder, not a fallback**. ``cfg.url("synapses")`` returns a complete
``dvid://server/uuid/instance`` string that you can print, paste and put in a saved command.
Nothing consults the config implicitly: the CLI takes a config reference only when it is
written as ``@name``, and it prints what that resolved to, exactly as it does for a
``{uuid}`` placeholder in ``--out``. The provenance record stores the resolved URL either
way, so a config can never make an export less traceable.

Search order (first hit wins):

1. ``$NEU_MARK_CONFIG``
2. ``neu-mark.toml`` in the current directory **or any parent** — so one file at the
   workspace root serves every repo and every notebook subdirectory under it, the way
   ``pyproject.toml`` and ``.git`` are found.

There is deliberately **no machine-wide location**. A config in the tree is one you trip over
while working, next to the body lists and the notes it goes with; a hidden one in
``~/.config`` applies to every shell on the host and is exactly the kind of invisible state
that makes a command behave differently for two people running it.

The upward walk is what makes a config in the tree usable: notebooks live a couple of
directories down from where the file naturally sits, and requiring the cwd to match exactly
would mean it silently stopped being found depending on where Jupyter was started.

.. code-block:: toml

   # neu-mark.toml, at the workspace root
   [dvid]
   server = "dvid.example.org"
   uuid    = "93fdbc:main"      # a branch ref is fine; it is resolved per run
   locked  = true               # prefer the newest locked node

   [instances]
   synapses = "synapses"
   bodies   = "labels_annotations"
   labels   = "labels"
   counts   = "synapses_labelsz"
   todo     = "labels_todo"

   # Named ROI lists, for `--rois @neuropils`. The set must be a chosen, non-overlapping
   # partition — see `dvid.label_point_rois` for why there is no "all".
   [roi_sets]
   neuropils = ["ME(L)", "ME(R)", "LO(L)", "LO(R)", "AL(L)", "LH(L)", "LH(R)"]

   # Named rules modules, for `--rules @wasp`. A path, relative to this file — so the
   # reference means the same thing from any directory the config is found from.
   [rule_sets]
   wasp = "rules/wasp_rules.py"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

#: The filename looked for in the cwd and every parent directory. The only place a config is
#: ever found, other than $NEU_MARK_CONFIG — see the module docstring for why there is
#: no machine-wide location.
FILENAME = "neu-mark.toml"

#: The environment variable that overrides the search entirely.
ENV_VAR = "NEU_MARK_CONFIG"

#: Marks a ``--src`` value as a config lookup rather than a URL. Required, so that a saved
#: command line visibly says "this depended on my config" instead of looking self-contained.
REFERENCE_PREFIX = "@"


class Config:
    """Resolved site defaults. Immutable; build URLs from it."""

    def __init__(self, data: Mapping[str, Any], path: str | None = None):
        self._data = dict(data)
        self.path = path
        dvid = self._data.get("dvid") or {}
        self.server = dvid.get("server")
        self.uuid = dvid.get("uuid")
        self.locked = bool(dvid.get("locked", False))
        self.scheme = dvid.get("scheme", "dvid")
        self.instances = dict(self._data.get("instances") or {})
        #: Named ROI lists. The set has to be chosen (a non-overlapping partition), and it is
        #: long, so naming one is the difference between a usable command and a paragraph.
        self.roi_sets = {k: list(v) for k, v in
                         (self._data.get("roi_sets") or {}).items()}
        #: Named rules modules. Same argument as `roi_sets`: the thing being named is a
        #: path nobody wants to retype, and `@name` keeps the dependency visible.
        self.rule_sets = {k: str(v) for k, v in
                          (self._data.get("rule_sets") or {}).items()}

    def rule_set(self, name: str) -> str:
        """The path of a named rules module from ``[rule_sets]``.

        Resolved **relative to the config file**, not the working directory, so a notebook
        two directories down and a shell at the workspace root get the same module. A
        relative path interpreted against the cwd is how the same command loads different
        rules depending on where it was run.
        """
        if name not in self.rule_sets:
            known = ", ".join(sorted(self.rule_sets)) or "(none configured)"
            raise ValueError(
                f"no rules module named {name!r} in {self._where()}. Configured: {known}.")
        path = Path(self.rule_sets[name]).expanduser()
        if not path.is_absolute() and self.path:
            path = Path(self.path).parent / path
        return str(path)

    def roi_set(self, name: str) -> list[str]:
        """A named ROI list from ``[roi_sets]``."""
        if name not in self.roi_sets:
            known = ", ".join(sorted(self.roi_sets)) or "(none configured)"
            raise ValueError(
                f"no ROI set named {name!r} in {self._where()}. Configured sets: {known}.")
        return list(self.roi_sets[name])

    def __repr__(self) -> str:                                    # pragma: no cover
        return (f"Config(path={self.path!r}, server={self.server!r}, "
                f"uuid={self.uuid!r}, locked={self.locked}, "
                f"instances={sorted(self.instances)})")

    def url(self, name: str, *, uuid: str | None = None) -> str:
        """A complete ``dvid://`` URL for a configured instance name.

        ``name`` may be a key under ``[instances]`` or a literal DVID instance name — a
        literal is allowed so a one-off instance needs no config edit.
        """
        if not self.server:
            raise ValueError(
                f"{self._where()} sets no dvid.server, so no URL can be built from it.")
        if not (uuid or self.uuid):
            raise ValueError(
                f"{self._where()} sets no dvid.uuid, so no URL can be built from it.")
        instance = self.instances.get(name, name)
        return f"{self.scheme}://{self.server}/{uuid or self.uuid}/{instance}"

    def _where(self) -> str:
        return f"the config at {self.path}" if self.path else "the (empty) default config"

    def resolve(self, value: str) -> str:
        """Turn ``@name`` into a URL; leave anything else untouched.

        The one entry point the CLI uses, so the ``@`` requirement lives in one place.
        """
        if not isinstance(value, str) or not value.startswith(REFERENCE_PREFIX):
            return value
        name = value[len(REFERENCE_PREFIX):]
        if not name:
            raise ValueError(f"{value!r} names nothing after {REFERENCE_PREFIX!r}")
        if name not in self.instances and not self.server:
            raise ValueError(
                f"--src {value!r} is a config reference, but no config was found. Looked at "
                f"${ENV_VAR}, then for {FILENAME} in the working directory and every parent.")
        if name not in self.instances:
            # Allowed, but say so: a typo would otherwise become a request for a
            # nonexistent DVID instance and fail much further along.
            known = ", ".join(sorted(self.instances)) or "(none configured)"
            raise ValueError(
                f"--src {value!r}: {name!r} is not in [instances] of {self._where()}. "
                f"Configured names: {known}. Use a full dvid:// URL for anything else.")
        return self.url(name)


def _read(path: Path) -> dict:
    import tomllib

    with open(path, "rb") as fh:
        return tomllib.load(fh)


def find(start: str | Path | None = None) -> str | None:
    """The config file that would be used, or ``None``.

    Walks up from ``start`` (the cwd by default) looking for :data:`FILENAME`, then falls
    back to :data:`SEARCH`.
    """
    explicit = os.environ.get(ENV_VAR)
    if explicit:
        # An explicit path that does not exist is an error, not something to fall through:
        # the caller said which file they meant.
        if not Path(explicit).expanduser().is_file():
            raise FileNotFoundError(
                f"${ENV_VAR} points at {explicit!r}, which is not a file")
        return str(Path(explicit).expanduser())

    here = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    if here.is_file():
        here = here.parent
    # `here` then all parents, nearest first — so a repo-local file beats a workspace one.
    for directory in (here, *here.parents):
        candidate = directory / FILENAME
        if candidate.is_file():
            return str(candidate)
    return None


def load(path: str | None = None) -> Config:
    """Load the config, or an empty one if there is none.

    Empty rather than an error, so importing this package never depends on a file existing.
    Asking an empty config for a URL is what raises, and it names where it looked.
    """
    chosen = str(Path(path).expanduser()) if path else find()
    if chosen is None:
        return Config({}, None)
    return Config(_read(Path(chosen)), chosen)
