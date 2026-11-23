from __future__ import annotations

import json
from pathlib import Path

from telegram_bot.core.services.tail_runner import _last_codex_turn_boundary_offset


def _record(record_type: str, **payload: str) -> bytes:
    return (json.dumps({"type": record_type, "payload": payload}) + "\n").encode()


def test_last_codex_turn_boundary_skips_large_history(tmp_path: Path) -> None:
    transcript = tmp_path / "rollout.jsonl"
    large_history = b"x" * (8 * 1024 * 1024) + b"\n"
    old_turn = _record("turn_context", turn_id="old-turn")
    latest_turn = _record("turn_context", turn_id="latest-turn")
    latest_message = _record("response_item", id="message")
    transcript.write_bytes(large_history + old_turn + latest_turn + latest_message)
    expected = len(large_history) + len(old_turn)

    offset = _last_codex_turn_boundary_offset(transcript, transcript.stat().st_size)

    assert offset == expected


def test_last_codex_turn_boundary_respects_durable_checkpoint(tmp_path: Path) -> None:
    transcript = tmp_path / "rollout.jsonl"
    first_turn = _record("turn_context", turn_id="first-turn")
    checkpoint = len(first_turn)
    later_turn = _record("turn_context", turn_id="later-turn")
    transcript.write_bytes(first_turn + later_turn)

    offset = _last_codex_turn_boundary_offset(transcript, checkpoint)

    assert offset == 0
