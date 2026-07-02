from assistant.config import Settings


def test_defaults():
    s = Settings()
    assert s.backend_port == 9981
    assert s.omlx_base_url == "http://127.0.0.1:8000"
    assert s.approval_required is True
    # 0 = no count cap: residency is governed by the memory ceiling (RAM×0.9 default), so
    # models that fit stay resident instead of thrashing reloads on every switch.
    assert s.max_loaded_models == 0


def test_env_override(monkeypatch):
    monkeypatch.setenv("ASSISTANT_BACKEND_PORT", "9999")
    monkeypatch.setenv("ASSISTANT_OMLX_PORT", "1234")
    monkeypatch.setenv("ASSISTANT_OMLX_AUTOSTART", "false")
    s = Settings()
    assert s.backend_port == 9999
    assert s.omlx_base_url == "http://127.0.0.1:1234"
    assert s.omlx_autostart is False
