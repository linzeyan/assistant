from pathlib import Path

from assistant.skills.discovery import SkillStore, normalize_slug, scan_skills
from assistant.skills.manage import create_skill, delete_skill


def _write_skill(d: Path, name: str, desc: str, body: str = "do things") -> None:
    sd = d / name
    sd.mkdir(parents=True)
    (sd / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n{body}\n")


def test_scan_and_index(tmp_path):
    _write_skill(tmp_path, "alpha", "A skill")
    idx = scan_skills([tmp_path])
    assert "alpha" in idx and idx["alpha"].description == "A skill"


def test_bundled_cognitive_skills_are_discoverable():
    # Spring4 SB.1 / SA.2: the engine was complete but the shelf was empty. The bundled cognitive
    # skills must ship at repo-root/skills (where main.py points _BUNDLED_SKILLS_DIR), each with a
    # frontmatter that parses and a non-empty trigger-y description, or pull-and-follow (SB.3 /
    # Lane C kill-gate) can't even be measured. All passed their kill-gate on the pinned model.
    bundled = Path(__file__).resolve().parent.parent / "skills"
    idx = scan_skills([bundled])
    for name in ("investigate", "spec", "decide", "review"):
        assert name in idx, f"bundled skill '{name}' missing"
        assert len(idx[name].description) > 30, f"'{name}' description too thin to trigger on"
        # Hard ≤60-line cap (lazy-load + small-model context budget); SKILL.md = frontmatter+body.
        assert idx[name].path.read_text().count("\n") <= 60, f"'{name}' SKILL.md over 60 lines"
    # The Iron Law must survive investigate's 30-50x distillation from gstack.
    assert "no root cause, no fix" in idx["investigate"].path.read_text().lower()


def test_store_reload_and_read_body(tmp_path):
    bundled, user = tmp_path / "bundled", tmp_path / "user"
    bundled.mkdir()
    user.mkdir()
    _write_skill(bundled, "alpha", "A")
    store = SkillStore(dirs=[bundled, user], user_dir=user)
    store.scan()
    assert set(store.index()) == {"alpha"}

    ok, _ = create_skill(store, "Beta Skill", "B desc", "steps here")
    assert ok
    assert "beta-skill" in store.index()  # reload picked it up
    assert "steps here" in store.read_body("beta-skill")


def test_user_overrides_bundled(tmp_path):
    bundled, user = tmp_path / "bundled", tmp_path / "user"
    bundled.mkdir()
    user.mkdir()
    _write_skill(bundled, "shared", "bundled version")
    _write_skill(user, "shared", "user version")
    store = SkillStore(dirs=[bundled, user], user_dir=user)
    store.scan()
    assert store.get("shared").description == "user version"


def test_delete_archives_user_skill(tmp_path):
    user = tmp_path / "user"
    user.mkdir()
    store = SkillStore(dirs=[user], user_dir=user)
    store.scan()
    create_skill(store, "temp", "t", "x")
    assert "temp" in store.index()

    ok, _ = delete_skill(store, "temp")
    assert ok
    assert "temp" not in store.index()
    assert (user / "_archive" / "temp" / "SKILL.md").is_file()  # recoverable


def test_cannot_delete_bundled_skill(tmp_path):
    bundled, user = tmp_path / "bundled", tmp_path / "user"
    bundled.mkdir()
    user.mkdir()
    _write_skill(bundled, "core", "c")
    store = SkillStore(dirs=[bundled, user], user_dir=user)
    store.scan()
    ok, msg = delete_skill(store, "core")
    assert ok is False and "bundled" in msg


def test_normalize_slug():
    assert normalize_slug("My Cool Skill") == "my-cool-skill"
    assert normalize_slug("a__b  c") == "a-b-c"
