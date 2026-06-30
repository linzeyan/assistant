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


# Capability keys a skill may declare under ``requires`` that the index gate understands (E2/S13):
# the optional media backends, whose availability is process-stable (an installed dep doesn't come
# and go at runtime), so gating on them keeps the system prompt a byte-stable cacheable prefix.
# An unknown ``requires`` token is IGNORED (fail-open) — we won't silently hide a user's skill over a
# capability we can't reason about.
KNOWN_CAPABILITIES = frozenset({"vision", "audio", "video", "images"})


@dataclass(frozen=True)
class SkillMeta:
    name: str  # normalised slug
    description: str
    path: Path  # the SKILL.md file
    directory: Path  # the skill's folder
    tags: tuple[str, ...] = ()  # free-form labels surfaced in the index (E2)
    requires: tuple[str, ...] = ()  # capability gate; lowercased (E2), see KNOWN_CAPABILITIES


def _str_tuple(value) -> tuple[str, ...]:
    """Normalise a frontmatter list/scalar into a tuple of trimmed, non-empty strings.

    Tolerant of the two shapes a hand-written SKILL.md uses: a YAML list (``[a, b]``) or a
    comma-separated scalar (``a, b``). Anything else degrades to empty rather than failing the scan."""
    if value is None:
        return ()
    items = value if isinstance(value, list) else str(value).split(",")
    return tuple(s for s in (str(i).strip() for i in items) if s)


def _first_nonempty_line(body: str) -> str:
    for line in body.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line
    return ""


def scan_skills(dirs: list[Path]) -> dict[str, SkillMeta]:
    """Scan dirs in order; later dirs win on slug collisions (user overrides bundled)."""
    return scan_skills_with_shadows(dirs)[0]


def scan_skills_with_shadows(
    dirs: list[Path],
) -> tuple[dict[str, SkillMeta], dict[str, list[str]]]:
    """Like ``scan_skills`` but also reports shadows (E1 shadow audit): later dirs win on a slug
    collision, and each overridden (lower-priority) source dir is recorded under that slug — so a
    project skill silently masking a user/bundled one of the same name is visible, not a mystery."""
    index: dict[str, SkillMeta] = {}
    shadows: dict[str, list[str]] = {}
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
            if name in index:  # a later (higher-priority) dir is about to shadow an earlier one
                shadows.setdefault(name, []).append(str(index[name].directory))
            index[name] = SkillMeta(
                name=name,
                description=description,
                path=skill_md,
                directory=skill_md.parent,
                tags=_str_tuple(meta.get("tags")),
                requires=tuple(r.lower() for r in _str_tuple(meta.get("requires"))),
            )
    return index, shadows


class SkillStore:
    """Holds the current skill index and supports reloading.

    ``dirs`` are scanned in order (bundled first, user dir last so user skills win).
    ``user_dir`` is where self-authored skills are written.
    """

    def __init__(
        self,
        dirs: list[Path],
        user_dir: Path,
        capabilities: set[str] | None = None,
    ):
        self._dirs = [Path(d) for d in dirs]
        self._user_dir = Path(user_dir)
        self._index: dict[str, SkillMeta] = {}
        self._shadows: dict[str, list[str]] = {}
        # Available capability keys for the ``requires`` gate; None = no gating (every skill shown).
        # Set once at startup (set_capabilities) so the gated index stays a stable cacheable prefix.
        self._capabilities = capabilities

    @property
    def user_dir(self) -> Path:
        return self._user_dir

    def set_capabilities(self, capabilities: set[str] | None) -> None:
        """Set the capability set the ``requires`` gate filters against. Called at startup once the
        media backends are known — process-stable, so the resulting index doesn't perturb the prefix."""
        self._capabilities = capabilities

    def scan(self) -> dict[str, SkillMeta]:
        self._index, self._shadows = scan_skills_with_shadows(self._dirs)
        return self._index

    def shadows(self) -> dict[str, list[str]]:
        """slug -> overridden (lower-priority) source dirs, from the last scan (E1 shadow audit)."""
        return dict(self._shadows)

    def index(self) -> dict[str, SkillMeta]:
        return self._index

    def _meets_requires(self, meta: SkillMeta) -> bool:
        """A skill is offered only when every KNOWN capability it requires is available. Unknown
        requirements are ignored (fail-open); no gate at all when capabilities weren't configured."""
        if self._capabilities is None:
            return True
        return not any(
            r in KNOWN_CAPABILITIES and r not in self._capabilities for r in meta.requires
        )

    def reload(self) -> dict:
        before = set(self._index)
        self.scan()
        after = set(self._index)
        return {
            "added": sorted(after - before),
            "removed": sorted(before - after),
            "unchanged": sorted(before & after),
            "shadowed": sorted(self._shadows),
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
        # Skills whose required capability is missing are omitted (E2) — the model shouldn't be told
        # about a workflow it can't run (e.g. a video skill when no video backend is installed),
        # mirroring the tool schema gate (G). Tags ride the line for lightweight discoverability.
        metas = sorted(
            (m for m in self._index.values() if self._meets_requires(m)),
            key=lambda m: m.name,
        )
        if not metas:
            return "(no skills available)"
        lines = []
        for m in metas:
            line = f"- {m.name}: {m.description}"
            if m.tags:
                line += f" [tags: {', '.join(m.tags)}]"
            lines.append(line)
        return "\n".join(lines)
