"""CRUD + import over /skills (GUI-driven skill authoring)."""

from __future__ import annotations

import pytest

from assistant.config import Settings
from assistant.main import create_app
from fastapi.testclient import TestClient


def _client(tmp_path):
    # `with TestClient` runs the lifespan, which populates app.state (skills store).
    return TestClient(create_app(Settings(skills_dir=tmp_path / "skills")))


def test_create_view_update_delete_cycle(tmp_path):
    with _client(tmp_path) as client:
        r = client.post(
            "/skills", json={"name": "My Skill", "description": "d", "body": "step one"}
        )
        assert r.status_code == 200, r.text
        slug = r.json()["name"]
        assert slug == "my-skill"

        entry = next(
            s for s in client.get("/skills").json()["skills"] if s["name"] == slug
        )
        assert entry["editable"] is True

        view = client.get(f"/skills/{slug}").json()
        assert "step one" in view["body"] and view["editable"] is True

        upd = client.put(f"/skills/{slug}", json={"description": "d2", "body": "step two"})
        assert upd.status_code == 200
        assert "step two" in client.get(f"/skills/{slug}").json()["body"]

        assert client.delete(f"/skills/{slug}").status_code == 200
        assert all(s["name"] != slug for s in client.get("/skills").json()["skills"])


def test_create_conflict_returns_409(tmp_path):
    with _client(tmp_path) as client:
        client.post("/skills", json={"name": "dup", "body": "x"})
        assert client.post("/skills", json={"name": "dup", "body": "y"}).status_code == 409


def test_create_requires_name_and_body(tmp_path):
    with _client(tmp_path) as client:
        assert client.post("/skills", json={"name": "", "body": "x"}).status_code == 400
        assert client.post("/skills", json={"name": "n", "body": " "}).status_code == 400


def test_bundled_skill_edit_is_copy_on_write(tmp_path):
    with _client(tmp_path) as client:
        names = [s["name"] for s in client.get("/skills").json()["skills"]]
        if "summarize-python" not in names:
            pytest.skip("bundled summarize-python not present")
        # Editing a bundled skill is allowed — it writes a user-dir shadow.
        r = client.put("/skills/summarize-python", json={"body": "my custom steps"})
        assert r.status_code == 200, r.text
        view = client.get("/skills/summarize-python").json()
        assert "my custom steps" in view["body"]
        assert view["editable"] is True  # now backed by a user copy
        # And the shadow can be removed (reverting to the bundled original).
        assert client.delete("/skills/summarize-python").status_code == 200


def test_pure_bundled_skill_cannot_be_deleted(tmp_path):
    with _client(tmp_path) as client:
        names = [s["name"] for s in client.get("/skills").json()["skills"]]
        if "summarize-python" not in names:
            pytest.skip("bundled summarize-python not present")
        # Without a user shadow, the read-only bundle can't be deleted.
        assert client.delete("/skills/summarize-python").status_code == 403


def test_import_parses_frontmatter(tmp_path):
    md = "---\nname: Imported One\ndescription: imp\n---\n\n# Body\nDo it.\n"
    with _client(tmp_path) as client:
        r = client.post("/skills/import", json={"content": md})
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "imported-one"
        assert "Do it." in client.get("/skills/imported-one").json()["body"]


def test_import_without_name_rejected(tmp_path):
    with _client(tmp_path) as client:
        assert (
            client.post("/skills/import", json={"content": "no frontmatter"}).status_code
            == 400
        )
