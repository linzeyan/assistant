"""Skill self-authoring: the mechanism behind "the agent learns new skills".

New/updated skills are written into the user skills dir (never overwriting bundled
skills). Deletion archives rather than removes — borrowing hermes-agent's
archive-not-delete safety so a mistaken removal is always recoverable.
"""

from __future__ import annotations

import shutil

from .discovery import SkillStore, normalize_slug


def render_skill_md(name: str, description: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body.strip()}\n"


def create_skill(
    store: SkillStore,
    name: str,
    description: str,
    body: str,
    *,
    overwrite: bool = False,
) -> tuple[bool, str]:
    slug = normalize_slug(name)
    skill_dir = store.user_dir / slug
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists() and not overwrite:
        return False, f"skill '{slug}' already exists (use action='update' to replace)"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md.write_text(render_skill_md(slug, description, body))
    store.reload()
    return True, f"{'updated' if overwrite else 'created'} skill '{slug}' at {skill_md}"


def delete_skill(store: SkillStore, name: str) -> tuple[bool, str]:
    slug = normalize_slug(name)
    meta = store.get(slug)
    if meta is None:
        return False, f"unknown skill '{slug}'"
    # Only user-authored skills can be removed; bundled skills are read-only.
    try:
        meta.directory.relative_to(store.user_dir)
    except ValueError:
        return False, f"'{slug}' is a bundled skill and cannot be deleted"
    archive = store.user_dir / "_archive"
    archive.mkdir(parents=True, exist_ok=True)
    dest = archive / slug
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(meta.directory), str(dest))
    store.reload()
    return True, f"archived skill '{slug}' to {dest}"
