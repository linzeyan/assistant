from __future__ import annotations

import yaml


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a SKILL.md into (frontmatter dict, body).

    Frontmatter is a leading YAML block delimited by ``---`` lines, mirroring
    hermes-agent's SKILL.md format. A missing/invalid block yields ``({}, text)`` so
    a malformed skill degrades to "body only" rather than failing the whole scan.
    """
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            body = parts[2].lstrip("\n")
            return (meta if isinstance(meta, dict) else {}), body
    return {}, text
