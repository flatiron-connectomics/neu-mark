"""Site defaults: a URL builder, never a silent fallback."""

import pytest

from em_annotation import config

TOML = """
[dvid]
server = "dvid.example.org"
uuid = "93fdbc:main"
locked = true

[instances]
synapses = "synapses"
bodies = "labels_annotations"
counts = "synapses_labelsz"
"""


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "em-annotation.toml"
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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "SEARCH", (str(tmp_path / "nope.toml"),))
    empty = config.load()
    assert empty.path is None and empty.server is None


def test_asking_an_empty_config_for_a_url_says_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.setattr(config, "SEARCH", (str(tmp_path / "nope.toml"),))
    with pytest.raises(ValueError, match="sets no dvid.server"):
        config.load().url("synapses")


def test_a_reference_with_no_config_at_all_names_the_search_path(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.setattr(config, "SEARCH", (str(tmp_path / "nope.toml"),))
    with pytest.raises(ValueError, match="no config was found"):
        config.load().resolve("@synapses")


def test_an_explicit_env_path_that_does_not_exist_is_an_error(tmp_path, monkeypatch):
    """Falling through would silently use a different file than the one named."""
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "missing.toml"))
    with pytest.raises(FileNotFoundError, match="is not a file"):
        config.load()


def test_the_cwd_file_is_found_before_the_home_one(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    (tmp_path / "em-annotation.toml").write_text(TOML)
    monkeypatch.chdir(tmp_path)
    assert config.load().server == "dvid.example.org"


def test_load_accepts_an_explicit_path(tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text(TOML)
    assert config.load(str(path)).uuid == "93fdbc:main"


def test_a_config_with_no_uuid_says_so(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[dvid]\nserver = "host"\n')
    with pytest.raises(ValueError, match="sets no dvid.uuid"):
        config.load(str(path)).url("synapses")
