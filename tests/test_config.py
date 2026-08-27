"""Site defaults: a URL builder, never a silent fallback."""

import pytest

from neu_mark import config

TOML = """
[dvid]
server = "dvid.example.org"
uuid = "93fdbc:main"
locked = true

[instances]
synapses = "synapses"
bodies = "labels_annotations"
counts = "synapses_labelsz"

[rule_sets]
wasp = "rules/wasp_rules.py"
absolute = "/opt/shared/rules.py"
"""


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "neu-mark.toml"
    path.write_text(TOML)
    monkeypatch.setenv(config.ENV_VAR, str(path))
    return config.load()


def test_it_builds_a_complete_url(cfg):
    """A complete URL, so it can be printed, pasted and put in a saved command — the whole
    reason this is a builder rather than a default consulted behind your back."""
    assert cfg.url("synapses") == (
        "dvid://dvid.example.org/93fdbc:main/synapses")
    assert cfg.url("counts").endswith("/synapses_labelsz")


def test_a_literal_instance_name_needs_no_config_entry(cfg):
    assert cfg.url("labels_todo").endswith("/labels_todo")


def test_an_explicit_uuid_overrides_the_default(cfg):
    assert cfg.url("synapses", uuid="846e3a").endswith("846e3a/synapses")


def test_locked_is_carried_through(cfg):
    assert cfg.locked is True


def test_resolve_only_touches_the_at_prefix(cfg):
    """The prefix is required so a saved command line visibly depends on a config."""
    assert cfg.resolve("@synapses").endswith("/synapses")
    url = "dvid://other/abc/labels"
    assert cfg.resolve(url) == url
    assert cfg.resolve("/some/path") == "/some/path"


def test_an_unknown_reference_lists_what_is_configured(cfg):
    with pytest.raises(ValueError, match="Configured names: bodies, counts, synapses"):
        cfg.resolve("@nope")


def test_an_empty_reference_is_refused(cfg):
    with pytest.raises(ValueError, match="names nothing after"):
        cfg.resolve("@")


def test_no_config_loads_empty_rather_than_raising(tmp_path, monkeypatch):
    """Importing this package must never depend on a file existing."""
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    empty = config.load()
    empty = config.Config({}, None) if empty.path else empty
    assert empty.path is None and empty.server is None


def test_asking_an_empty_config_for_a_url_says_where_it_looked():
    with pytest.raises(ValueError, match="sets no dvid.server"):
        config.Config({}, None).url("synapses")


def test_a_reference_with_no_config_at_all_names_the_search(monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="in the working directory and every parent"):
        config.Config({}, None).resolve("@synapses")


def test_an_explicit_env_path_that_does_not_exist_is_an_error(tmp_path, monkeypatch):
    """Falling through would silently use a different file than the one named."""
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "missing.toml"))
    with pytest.raises(FileNotFoundError, match="is not a file"):
        config.load()


# --------------------------------------------------------------------------- #
# finding it: an upward walk, and nothing machine-wide
# --------------------------------------------------------------------------- #
def test_it_is_found_by_walking_up_from_a_subdirectory(tmp_path, monkeypatch):
    """The case that matters: notebooks sit a couple of directories below where the file
    naturally lives, and requiring an exact cwd would make it silently stop being found."""
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    (tmp_path / config.FILENAME).write_text(TOML)
    deep = tmp_path / "neu-mark" / "notebooks"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert config.find() == str(tmp_path / config.FILENAME)
    assert config.load().server == "dvid.example.org"


def test_the_nearest_config_wins(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    (tmp_path / config.FILENAME).write_text(TOML)
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / config.FILENAME).write_text('[dvid]\nserver = "nearer"\n')
    monkeypatch.chdir(inner)
    assert config.load().server == "nearer"


def test_there_is_no_machine_wide_location(tmp_path, monkeypatch):
    """A hidden ~/.config file applies to every shell on the host and is the kind of
    invisible state that makes one command behave differently for two people."""
    assert not hasattr(config, "SEARCH")
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)          # nothing above tmp_path has one
    assert config.find(tmp_path) is None


def test_find_accepts_an_explicit_starting_point(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    (tmp_path / config.FILENAME).write_text(TOML)
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert config.find(deep) == str(tmp_path / config.FILENAME)


def test_a_named_rules_module_resolves_against_the_CONFIG_not_the_cwd(cfg, tmp_path,
                                                                     monkeypatch):
    """A notebook two directories down and a shell at the workspace root must load the same
    module. Resolved against the cwd, the same command loads different rules depending on
    where it was run — which is precisely the invisible state this file exists to avoid."""
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert cfg.rule_set("wasp") == str(tmp_path / "rules" / "wasp_rules.py")


def test_an_absolute_rules_path_is_left_alone(cfg):
    assert cfg.rule_set("absolute") == "/opt/shared/rules.py"


def test_an_unknown_rules_name_lists_what_is_configured(cfg):
    with pytest.raises(ValueError, match="absolute, wasp"):
        cfg.rule_set("nope")


def test_load_accepts_an_explicit_path(tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text(TOML)
    assert config.load(str(path)).uuid == "93fdbc:main"


def test_a_config_with_no_uuid_says_so(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[dvid]\nserver = "host"\n')
    with pytest.raises(ValueError, match="sets no dvid.uuid"):
        config.load(str(path)).url("synapses")
