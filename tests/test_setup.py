"""Preflight checks + managed-tool install lifecycle."""

from __future__ import annotations

from assistant.config import Settings
from assistant.setup.manage import (
    FEATURES,
    check_paths,
    check_tools,
    find_uv,
    install_command,
    perform_install,
    preflight,
)


def test_check_tools_covers_every_feature():
    tools = check_tools()
    assert {t["feature"] for t in tools} == set(FEATURES)
    for t in tools:
        assert isinstance(t["installed"], bool)
        assert t["package"] and t["label"]


def test_check_paths_reports_existing_dirs(tmp_path):
    settings = Settings(home_dir=tmp_path, models_dir=tmp_path / "models")
    (tmp_path / "models").mkdir()
    paths = {p["name"]: p for p in check_paths(settings)}
    assert paths["models"]["exists"] is True
    assert paths["skills"]["exists"] is False  # never created


def test_preflight_shape(tmp_path):
    # hf_cache=False isolates the count from any real HuggingFace cache on the machine.
    settings = Settings(models_dir=tmp_path / "models", hf_cache=False)
    report = preflight(settings)
    assert report["venv"] and report["python"]
    assert report["models"]["count"] == 0  # empty dir, no cache scan
    assert {"paths", "tools", "config_path", "download_dir"} <= report.keys()


def test_install_command_prefers_uv():
    uv_cmd = install_command("mlx", uv="/opt/homebrew/bin/uv")
    assert uv_cmd[:3] == ["/opt/homebrew/bin/uv", "pip", "install"] and "mlx-lm" in uv_cmd
    pip_cmd = install_command("mlx", uv=None)
    assert pip_cmd[1:4] == ["-m", "pip", "install"] and "mlx-lm" in pip_cmd


def test_install_command_upgrade_adds_flag():
    # The GUI "更新套件" / Update package path passes upgrade=True → pip/uv pull latest.
    uv_cmd = install_command("mlx", uv="/opt/homebrew/bin/uv", upgrade=True)
    assert "--upgrade" in uv_cmd and uv_cmd[-1] == "mlx-lm"
    pip_cmd = install_command("mlx", uv=None, upgrade=True)
    assert "--upgrade" in pip_cmd and pip_cmd[-1] == "mlx-lm"
    # A first install (default) must NOT upgrade.
    assert "--upgrade" not in install_command("mlx", uv=None)


def test_install_command_source_override_forces_reinstall():
    # A patched build (e.g. mlx-lm git ref): update must re-pull the source, not PyPI
    # --upgrade. Force a clean reinstall so a moving branch/PR ref doesn't no-op (N11).
    spec = "git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1192/head"
    uv_cmd = install_command("mlx", uv="/opt/homebrew/bin/uv", upgrade=True, source=spec)
    assert uv_cmd[-1] == spec
    assert "--reinstall" in uv_cmd and "--upgrade" not in uv_cmd
    pip_cmd = install_command("mlx", uv=None, upgrade=True, source=spec)
    assert pip_cmd[-1] == spec and "--force-reinstall" in pip_cmd
    # A first install from source: target is the spec, no forced reinstall.
    first = install_command("mlx", uv=None, source=spec)
    assert first[-1] == spec and "--force-reinstall" not in first


def test_check_tools_surfaces_source_and_new_keys():
    settings = Settings(
        managed_tool_sources={"mlx": "git+https://x/mlx-lm.git@main"}
    )
    tools = {t["feature"]: t for t in check_tools(settings)}
    assert tools["mlx"]["source"] == "git+https://x/mlx-lm.git@main"
    assert tools["images"]["source"] is None
    for t in tools.values():
        assert {"version", "latest", "source", "update_available"} <= t.keys()


def test_check_tools_update_available_needs_newer_pypi():
    # Pretend the embeddings package has a sky-high latest; update only fires when the
    # tool is actually installed (and strictly older). Uninstalled → never an update.
    from assistant.setup.manage import FEATURES, _installed

    settings = Settings()
    latest = {FEATURES["embeddings"]["package"]: "9999.0.0"}
    tools = {t["feature"]: t for t in check_tools(settings, latest=latest)}
    expected = _installed(FEATURES["embeddings"]["module"])  # env-dependent install state
    assert tools["embeddings"]["update_available"] is expected


def test_is_newer_semantics():
    from assistant.setup.manage import _is_newer

    assert _is_newer("1.2.0", "1.1.9") is True
    assert _is_newer("1.1.0", "1.1.0") is False
    assert _is_newer(None, "1.0.0") is False
    assert _is_newer("1.0.0", None) is False


def test_find_uv_prefers_env_override(tmp_path, monkeypatch):
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\n")
    monkeypatch.setenv("ASSISTANT_UV", str(fake_uv))
    assert find_uv() == str(fake_uv)


def test_find_uv_ignores_nonexistent_override(tmp_path, monkeypatch):
    # A bogus override must never be returned; discovery falls through to PATH/candidates.
    monkeypatch.setenv("ASSISTANT_UV", str(tmp_path / "missing-uv"))
    monkeypatch.delenv("UV", raising=False)
    assert find_uv() != str(tmp_path / "missing-uv")


async def test_perform_install_marks_done():
    state: dict = {}
    await perform_install(state, "audio", lambda _f: None)
    assert state["audio"]["status"] == "done"
    assert state["audio"]["package"] == "mlx-audio"


async def test_perform_install_records_error():
    state: dict = {}

    def boom(_feature):
        raise RuntimeError("no network")

    await perform_install(state, "video", boom)
    assert state["video"]["status"] == "error"
    assert "no network" in state["video"]["error"]
