"""Path helpers for CC tmux-TUI transcripts.

Derives ~/.claude/projects/<slug>/<session_id>.jsonl from (cwd, session_id)
using the slug rule probed against real CC 2.1.114 on 2026-04-19:
replace every non-[a-zA-Z0-9-] character with `-`.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

_SESSION_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_CODEX_SESSION_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


def cwd_to_slug(cwd: str | Path) -> str:
    """Derive the ~/.claude/projects/<slug>/ slug from a cwd path.

    Rule: replace every character that is NOT [a-zA-Z0-9-] with `-`.
    Verified against real CC 2.1.114 on 2026-04-19.
    """
    return re.sub(r"[^a-zA-Z0-9-]", "-", str(cwd))


def transcript_path(cwd: str | Path, session_id: str, home: Path | None = None) -> Path:
    """Full path to the CC transcript jsonl for (cwd, session_id).

    Defensive check on session_id format: must match a strict UUID4 shape
    `XXXXXXXX-XXXX-4XXX-[89ab]XXX-XXXXXXXXXXXX` (version 4, variant 10xx).
    This is a trust-boundary marker — if session_id ever arrives from an
    untrusted source, the check blocks path-traversal attempts like
    `../../../etc/passwd` AND degenerate shapes like 36 dashes.

    Uses an explicit `if / raise ValueError` instead of `assert` because
    running Python with `-O` silently strips asserts, disabling the guard.
    """
    home = home or Path.home()
    return transcript_path_for_config_dir(cwd, session_id, home / ".claude")


def transcript_path_for_config_dir(
    cwd: str | Path,
    session_id: str,
    config_dir: str | Path,
) -> Path:
    """Derive a transcript path for an explicit Runtime-owned Claude config."""
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")
    root = Path(config_dir)
    if not root.is_absolute() or root == Path("/") or ".." in root.parts:
        raise ValueError(f"Invalid Claude config directory: {str(root)!r}")
    return root / "projects" / cwd_to_slug(cwd) / f"{session_id}.jsonl"


def generate_session_uuid() -> str:
    """UUID for `claude --session-id <uuid>`. Must be a valid UUID string."""
    return str(uuid.uuid4())
