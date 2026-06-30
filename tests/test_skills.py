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


def _write_skill_fm(d: Path, name: str, extra: str, body: str = "do things") -> None:
    # Like _write_skill but lets a test inject extra frontmatter lines (tags/requires).
    sd = d / name
    sd.mkdir(parents=True)
    (sd / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} desc\n{extra}\n---\n{body}\n"
    )


def test_frontmatter_tags_and_requires_parsed(tmp_path):
    # E2: tags + requires parse from frontmatter, tolerating list and comma-scalar shapes.
    _write_skill_fm(tmp_path, "a", "tags: [x, y]\nrequires: [video]")
    _write_skill_fm(tmp_path, "b", "tags: one, two\nrequires: Vision")  # scalar + case
    idx = scan_skills([tmp_path])
    assert idx["a"].tags == ("x", "y") and idx["a"].requires == ("video",)
    assert idx["b"].tags == ("one", "two") and idx["b"].requires == ("vision",)  # lowercased


def test_requires_gate_hides_skill_when_capability_absent(tmp_path):
    # E2: a skill requiring an unavailable capability is omitted from the index the model sees,
    # mirroring the tool schema gate (G). Tags surface on the line for skills that do show.
    _write_skill_fm(tmp_path, "make-video", "requires: video")
    _write_skill_fm(tmp_path, "summarize", "tags: [text]")
    store = SkillStore(dirs=[tmp_path], user_dir=tmp_path, capabilities={"vision"})  # no video
    store.scan()
    text = store.index_text()
    assert "make-video" not in text  # gated out: required backend not installed
    assert "summarize" in text and "[tags: text]" in text
    # All scanned skills remain retrievable by name even if gated from the index.
    assert store.get("make-video") is not None
    # With the capability present, the skill reappears.
    store.set_capabilities({"vision", "video"})
    assert "make-video" in store.index_text()


def test_requires_gate_off_when_capabilities_unset(tmp_path):
    # capabilities=None (the default / unit-test path) => no gating, every skill shown.
    _write_skill_fm(tmp_path, "make-video", "requires: video")
    store = SkillStore(dirs=[tmp_path], user_dir=tmp_path)
    store.scan()
    assert "make-video" in store.index_text()


def test_unknown_requires_token_is_fail_open(tmp_path):
    # An unrecognised requires token must not silently hide a user's skill (fail-open).
    _write_skill_fm(tmp_path, "weird", "requires: quantum-flux")
    store = SkillStore(dirs=[tmp_path], user_dir=tmp_path, capabilities=set())
    store.scan()
    assert "weird" in store.index_text()


def test_project_dir_shadows_user_and_is_audited(tmp_path):
    # E1: a third (project) dir scanned last wins the slug, and the override is recorded.
    bundled, user, project = tmp_path / "b", tmp_path / "u", tmp_path / "p"
    for d in (bundled, user, project):
        d.mkdir()
    _write_skill(bundled, "shared", "bundled version")
    _write_skill(project, "shared", "project version")
    store = SkillStore(dirs=[bundled, user, project], user_dir=user)
    report = store.reload()
    assert store.get("shared").description == "project version"  # project wins
    assert "shared" in report["shadowed"]
    assert any(str(bundled) in src for src in store.shadows()["shared"])
