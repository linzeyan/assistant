"""Model download lifecycle (downloader injected — no network)."""

from __future__ import annotations

from assistant.api.routes_downloads import perform_download


async def test_download_marks_done_on_success():
    state: dict = {}

    def downloader(repo_id):
        return f"/cache/{repo_id}"

    await perform_download(state, "org/model", downloader)
    assert state["org/model"]["status"] == "done"
    assert state["org/model"]["error"] is None


async def test_download_records_error_on_failure():
    state: dict = {}

    def downloader(repo_id):
        raise RuntimeError("network down")

    await perform_download(state, "org/model", downloader)
    assert state["org/model"]["status"] == "error"
    assert "network down" in state["org/model"]["error"]
