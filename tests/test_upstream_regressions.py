from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from telegram_bot.core.services import tmux_manager as tmux_manager_module
from telegram_bot.core.services.providers import CODEX_ADAPTER, CodexTranscriptSnapshot
from telegram_bot.core.services.tmux_manager import TmuxManager
from telegram_bot.core.services.tmux_state import TmuxSessionState


@pytest.mark.asyncio
async def test_fresh_codex_late_acceptance_continues_from_discovered_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first prompt accepted after the pane retry window must still be tailed."""
    manager = TmuxManager(tmp_path / "sessions")
    channel_key = (100, 200)
    state = TmuxSessionState(
        session_name="cc-100-200",
        session_dir=str(tmp_path / "sessions" / "cc-100-200"),
        session_id=None,
        mode="acceptEdits",
        cwd=str(tmp_path),
        mcp_config="",
        chat_id=100,
        offset=0,
        runner_version="codex-tui-v1",
        provider="codex",
        model=None,
        transcript_path=None,
        base_mcp_config=None,
    )
    manager._sessions[channel_key] = state

    snapshot = CodexTranscriptSnapshot(root=tmp_path / "codex-sessions", files={})
    monkeypatch.setattr(CODEX_ADAPTER, "capture_tui_transcript_snapshot", lambda: snapshot)
    monkeypatch.setattr(CODEX_ADAPTER, "is_modal_present", lambda _pane: False)
    monkeypatch.setattr(manager, "_codex_delivery_visible", lambda *_args: True)
    monkeypatch.setattr(manager, "_codex_prompt_still_in_input_bar", lambda *_args: True)
    monkeypatch.setattr(tmux_manager_module, "_CODEX_PASTE_POLL_STEP_SEC", 0.0)
    monkeypatch.setattr(tmux_manager_module, "_ENTER_RETRY_SETTLE_SEC", 0.0)

    capture_pane = AsyncMock(return_value="pane with prompt still visible")
    send_text = AsyncMock()
    send_enter = AsyncMock()
    alert = AsyncMock()
    monkeypatch.setattr(tmux_manager_module, "capture_pane", capture_pane)
    monkeypatch.setattr(tmux_manager_module, "send_text_to_tmux", send_text)
    monkeypatch.setattr(tmux_manager_module, "send_enter", send_enter)
    monkeypatch.setattr(manager, "_send_modal_alert", alert)

    async def discover_transcript(
        discovered_state: TmuxSessionState,
        *,
        snapshot: CodexTranscriptSnapshot,
        prompt: str,
    ) -> None:
        assert snapshot.root == tmp_path / "codex-sessions"
        assert prompt == "first prompt"
        discovered_state.session_id = "00000000-0000-4000-8000-000000000001"
        discovered_state.transcript_path = str(tmp_path / "turn.jsonl")

    locate = AsyncMock(side_effect=discover_transcript)
    tail = AsyncMock(return_value=("", None))
    monkeypatch.setattr(manager, "_locate_codex_transcript_after_send", locate)
    monkeypatch.setattr(manager, "_tail_until_done", tail)

    await manager.send_stream(channel_key, "first prompt", AsyncMock())

    assert send_enter.await_count == tmux_manager_module._ENTER_RETRY_LIMIT
    locate.assert_awaited_once()
    tail.assert_awaited_once()
    alert.assert_not_awaited()
