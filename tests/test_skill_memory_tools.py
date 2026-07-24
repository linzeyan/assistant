from assistant.memory.file_provider import FileMemoryProvider
from assistant.skills.discovery import SkillStore
from assistant.tools import build_registry
from assistant.tools.base import ToolContext


def _ctx(tmp_path):
    user = tmp_path / "skills"
    user.mkdir()
    store = SkillStore(dirs=[user], user_dir=user)
    store.scan()
    memory = FileMemoryProvider(tmp_path / "memory")
    return ToolContext(cwd=tmp_path, skills=store, memory=memory), store


async def test_skill_manage_create_then_view_and_list(tmp_path):
    reg = build_registry()
    ctx, store = _ctx(tmp_path)
    res = await reg.get("skill_manage").handler(
        {"action": "create", "name": "My Skill", "description": "d", "body": "steps here"},
        ctx,
    )
    assert res.ok and "my-skill" in store.index()

    view = await reg.get("skill_view").handler({"name": "my-skill"}, ctx)
    assert "steps here" in view.content
    listing = await reg.get("skills_list").handler({}, ctx)
    assert "my-skill" in listing.content


async def test_skill_manage_requires_body():
    reg = build_registry()
    # missing body -> a clear error, not a crash
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        store = SkillStore(dirs=[Path(d)], user_dir=Path(d))
        store.scan()
        ctx = ToolContext(cwd=Path(d), skills=store)
        res = await reg.get("skill_manage").handler(
            {"action": "create", "name": "x", "description": "d"}, ctx
        )
        assert res.ok is False and "body" in res.content


def test_self_modifying_tools_need_approval():
    reg = build_registry()
    assert reg.get("skill_manage").needs_approval is True
    assert reg.get("memory_write").needs_approval is False  # learning facts is low-risk


async def test_memory_tools_roundtrip(tmp_path):
    reg = build_registry()
    ctx, _ = _ctx(tmp_path)
    w = await reg.get("memory_write").handler(
        {"content": "likes dark mode", "tags": ["ui"]}, ctx
    )
    assert w.ok
    s = await reg.get("memory_search").handler({"query": "dark mode"}, ctx)
    assert "dark mode" in s.content


async def test_tools_degrade_without_context(tmp_path):
    reg = build_registry()
    bare = ToolContext(cwd=tmp_path)  # no skills / memory attached
    assert (await reg.get("skills_list").handler({}, bare)).ok is False
    assert (await reg.get("memory_write").handler({"content": "x"}, bare)).ok is False
    assert (await reg.get("memory_forget").handler({"id": "x"}, bare)).ok is False


async def test_memory_forget_retires_stale_entry(tmp_path):
    # The model must be able to resolve a contradiction ("I switched to pnpm") by
    # deleting the superseded fact — otherwise old and new memories are recalled
    # together forever. Deleting user memory is destructive, hence approval-gated.
    reg = build_registry()
    assert reg.get("memory_forget").needs_approval is True
    ctx, _ = _ctx(tmp_path)
    await reg.get("memory_write").handler({"content": "uses npm"}, ctx)
    s = await reg.get("memory_search").handler({"query": "npm"}, ctx)
    entry_id = s.content.split("[", 1)[1].split("]", 1)[0]  # ids ride the results
    res = await reg.get("memory_forget").handler({"id": entry_id}, ctx)
    assert res.ok
    gone = await reg.get("memory_search").handler({"query": "npm"}, ctx)
    assert "npm" not in gone.content
    # unknown id -> a clear error, not a silent success
    assert (await reg.get("memory_forget").handler({"id": entry_id}, ctx)).ok is False
