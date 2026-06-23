"""Skill discovery: scan directories for ``<skill>/SKILL.md`` files.

Ported in spirit from hermes-agent's ``scan_skill_commands``: parse frontmatter for
name + description, normalise the name to a slug, and present a flat index. Skills
are NOT auto-injected in full — only the index (name + description) goes into the
system prompt; the model reads a skill's body on demand via ``skill_view``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .frontmatter import parse_frontmatter


def normalize_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "skill"


@dataclass(frozen=True)
class SkillMeta:
    name: str  # normalised slug
    description: str
    path: Path  # the SKILL.md file
    directory: Path  # the skill's folder


def _first_nonempty_line(body: str) -> str:
    for line in body.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line
    return ""


def scan_skills(dirs: list[Path]) -> dict[str, SkillMeta]:
    """Scan dirs in order; later dirs win on slug collisions (user overrides bundled)."""
    index: dict[str, SkillMeta] = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for skill_md in sorted(d.glob("*/SKILL.md")):
            try:
                text = skill_md.read_text(errors="replace")
            except OSError:
                continue
            meta, body = parse_frontmatter(text)
            name = normalize_slug(str(meta.get("name") or skill_md.parent.name))
            description = str(meta.get("description") or _first_nonempty_line(body))
            index[name] = SkillMeta(
                name=name,
                description=description,
                path=skill_md,
                directory=skill_md.parent,
            )
    return index


class SkillStore:
    """Holds the current skill index and supports reloading.

    ``dirs`` are scanned in order (bundled first, user dir last so user skills win).
    ``user_dir`` is where self-authored skills are written.
    """

    def __init__(self, dirs: list[Path], user_dir: Path):
        self._dirs = [Path(d) for d in dirs]
        self._user_dir = Path(user_dir)
        self._index: dict[str, SkillMeta] = {}

    @property
    def user_dir(self) -> Path:
        return self._user_dir

    def scan(self) -> dict[str, SkillMeta]:
        self._index = scan_skills(self._dirs)
        return self._index

    def index(self) -> dict[str, SkillMeta]:
        return self._index

    def reload(self) -> dict:
        before = set(self._index)
        self.scan()
        after = set(self._index)
        return {
            "added": sorted(after - before),
            "removed": sorted(before - after),
            "unchanged": sorted(before & after),
            "total": len(after),
        }

    def get(self, name: str) -> SkillMeta | None:
        return self._index.get(normalize_slug(name))

    def read_body(self, name: str) -> str | None:
        meta = self.get(name)
        if meta is None:
            return None
        _, body = parse_frontmatter(meta.path.read_text(errors="replace"))
        return body

    def index_text(self) -> str:
        if not self._index:
            return "(no skills available)"
        return "\n".join(
            f"- {m.name}: {m.description}"
            for m in sorted(self._index.values(), key=lambda m: m.name)
        )
