"""Validated bridge metadata published by the external provider Runtime."""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path

from telegram_bot.core.tui.paths import transcript_path_for_config_dir

logger = logging.getLogger(__name__)

_METADATA_NAME = "provider.runtime.json"
_MAX_METADATA_BYTES = 16 * 1024


def claude_runtime_transcript_path(
    *,
    session_dir: str | Path,
    cwd: str,
    session_id: str,
) -> Path | None:
    """Read a private Runtime sidecar and derive the exact Claude transcript."""
    path = Path(session_dir) / _METADATA_NAME
    try:
        metadata_stat = path.lstat()
        if not stat.S_ISREG(metadata_stat.st_mode) or path.is_symlink():
            raise ValueError("metadata is not a regular file")
        if metadata_stat.st_uid != os.getuid() or stat.S_IMODE(metadata_stat.st_mode) & 0o077:
            raise ValueError("metadata ownership or permissions are unsafe")
        if metadata_stat.st_size > _MAX_METADATA_BYTES:
            raise ValueError("metadata is too large")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("metadata is not an object")
        if payload.get("schema_version") != 1 or payload.get("provider") != "claude_code":
            raise ValueError("metadata schema or provider is unsupported")
        if payload.get("session_id") != session_id:
            raise ValueError("metadata belongs to a different session")
        config_dir = payload.get("config_dir")
        if not isinstance(config_dir, str):
            raise ValueError("metadata has no config directory")
        return transcript_path_for_config_dir(cwd, session_id, config_dir)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        logger.warning("Ignoring invalid provider Runtime metadata at %s", path, exc_info=True)
        return None
