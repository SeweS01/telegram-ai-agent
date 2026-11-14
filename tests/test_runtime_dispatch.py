import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import pytest

from telegram_bot.core.handlers.streaming import ensure_exec_mode_ready
from telegram_bot.core.services.runtime_dispatch import (
    DispatchDecision,
    DispatchResult,
    RuntimeDispatchError,
    RuntimeDispatchGuard,
)
from telegram_bot.core.services.tmux_manager import TmuxManager, TmuxSessionState


@pytest.mark.asyncio
async def test_runtime_dispatch_bypasses_channels_outside_managed_chat(tmp_path: Path) -> None:
    guard = RuntimeDispatchGuard(
        socket_path=tmp_path / "missing.sock",
        agent_instance_id="agent-1",
        managed_chat_id=-1001,
        wait_timeout_seconds=10,
    )

    result = await guard.prepare((-1002, 42))

    assert result.decision is DispatchDecision.ALLOW


@pytest.mark.asyncio
async def test_runtime_dispatch_decodes_recycle_and_rejects_deny(tmp_path: Path) -> None:
    guard = RuntimeDispatchGuard(
        socket_path=tmp_path / "runtime.sock",
        agent_instance_id="agent-1",
        managed_chat_id=-1001,
        wait_timeout_seconds=10,
    )
    guard._request = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "ok": True,
            "decision": "RECYCLE_REQUIRED",
            "desired": {"provider": "codex", "model": "gpt-new"},
        }
    )
    result = await guard.prepare((-1001, 42))
    assert result.decision is DispatchDecision.RECYCLE_REQUIRED
    assert result.desired_provider == "codex"
    assert result.desired_model == "gpt-new"

    guard._request = AsyncMock(  # type: ignore[method-assign]
        return_value={"ok": True, "decision": "DENY", "reason": "unsafe"}
    )
    with pytest.raises(RuntimeDispatchError, match="unsafe"):
        await guard.prepare((-1001, 42))


@pytest.mark.asyncio
async def test_runtime_dispatch_maps_claude_code_to_upstream_engine(tmp_path: Path) -> None:
    guard = RuntimeDispatchGuard(
        socket_path=tmp_path / "runtime.sock",
        agent_instance_id="agent-1",
        managed_chat_id=-1001,
        wait_timeout_seconds=10,
    )
    guard._request = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "ok": True,
            "decision": "START_REQUIRED",
            "desired": {"provider": "claude_code", "model": "deepseek-flash"},
        }
    )

    result = await guard.prepare((-1001, 42))

    assert result.desired_provider == "claude"
    assert result.desired_model == "deepseek-flash"


@pytest.mark.asyncio
async def test_ensure_exec_mode_waits_then_recycles_before_allowing_prompt() -> None:
    manager = SimpleNamespace(
        prepare_runtime_dispatch=AsyncMock(
            side_effect=[
                DispatchResult(DispatchDecision.WAIT, retry_after_seconds=0.001),
                DispatchResult(
                    DispatchDecision.RECYCLE_REQUIRED,
                    desired_provider="codex",
                    desired_model="gpt-new",
                ),
            ]
        ),
        runtime_dispatch_wait_timeout_seconds=10.0,
        recycle=AsyncMock(return_value=True),
        has_session_state=lambda _key: True,
        is_active=lambda _key: True,
    )
    source = SimpleNamespace(message_id=7, answer=AsyncMock())

    ready = await ensure_exec_mode_ready(
        (-1001, 42),
        SimpleNamespace(),
        manager,
        SimpleNamespace(),
        source,
    )

    assert ready is True
    manager.recycle.assert_awaited_once_with(
        (-1001, 42),
        ANY,
        model_override="gpt-new",
        provider_override="codex",
    )
    source.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_exec_mode_resumes_retired_saved_session() -> None:
    manager = SimpleNamespace(
        prepare_runtime_dispatch=AsyncMock(
            return_value=DispatchResult(
                DispatchDecision.START_REQUIRED,
                desired_provider="codex",
                desired_model="gpt-new",
            )
        ),
        runtime_dispatch_wait_timeout_seconds=10.0,
        recycle=AsyncMock(return_value=True),
        has_session_state=lambda _key: True,
        is_active=lambda _key: True,
    )
    source = SimpleNamespace(message_id=8, answer=AsyncMock())

    ready = await ensure_exec_mode_ready(
        (-1001, 42),
        SimpleNamespace(),
        manager,
        SimpleNamespace(),
        source,
    )

    assert ready is True
    manager.recycle.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_recycle_switches_provider_without_reusing_foreign_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TmuxManager(tmp_path / "sessions")
    channel_key = (-1001, 42)
    old_session_id = "01a14985-de09-7190-bc15-4c4c258f60b0"
    state = TmuxSessionState(
        session_name="cc-n1001-42",
        session_dir=str(tmp_path / "sessions" / "cc-n1001-42"),
        session_id=old_session_id,
        mode="free",
        cwd=str(tmp_path),
        mcp_config=str(tmp_path / "old-mcp.json"),
        chat_id=-1001,
        offset=123,
        runner_version="codex-tui-v1",
        provider="codex",
        model="gpt-old",
        transcript_path=str(tmp_path / "codex-rollout.jsonl"),
        base_mcp_config=str(tmp_path / "base-mcp.json"),
    )
    manager._sessions[channel_key] = state
    monkeypatch.setattr(manager, "close_buffer", AsyncMock())
    monkeypatch.setattr(manager, "_spawn_tmux", AsyncMock())
    monkeypatch.setattr(
        manager,
        "_ensure_runtime_mcp_config",
        Mock(return_value=str(tmp_path / "runtime-mcp.json")),
    )
    session_manager = SimpleNamespace(
        build_tmux_startup_args=Mock(return_value=["claude", "--session-id", "new"]),
        clear_provider_session=AsyncMock(),
    )

    recycled = await manager.recycle(
        channel_key,
        session_manager,
        provider_override="claude",
        model_override="deepseek-flash",
    )

    assert recycled is True
    current = manager._sessions[channel_key]
    assert current.provider == "claude"
    assert current.model == "deepseek-flash"
    assert current.runner_version == "claude-tui-v1"
    assert current.session_id is not None
    assert current.session_id != old_session_id
    assert current.transcript_path is None
    session_manager.build_tmux_startup_args.assert_called_once_with(
        mode="free",
        mcp_config=str(tmp_path / "runtime-mcp.json"),
        session_id_new=current.session_id,
        model="deepseek-flash",
    )
    manager._spawn_tmux.assert_awaited_once_with(
        name="cc-n1001-42",
        session_dir=tmp_path / "sessions" / "cc-n1001-42",
        cwd=str(tmp_path),
        startup_cmd=["claude", "--session-id", "new"],
        channel_key=channel_key,
        provider="claude",
    )
    session_manager.clear_provider_session.assert_awaited_once_with(channel_key)


def test_claude_runtime_metadata_selects_isolated_transcript_root(tmp_path: Path) -> None:
    manager = TmuxManager(tmp_path / "sessions")
    session_id = "01a14985-de09-4190-bc15-4c4c258f60b0"
    session_dir = tmp_path / "sessions" / "cc-n1001-42"
    session_dir.mkdir(parents=True)
    config_dir = tmp_path / "runtime" / "claude-config" / "deepseek_default"
    metadata = session_dir / "provider.runtime.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "claude_code",
                "session_id": session_id,
                "config_dir": str(config_dir),
            }
        ),
        encoding="utf-8",
    )
    metadata.chmod(0o600)
    state = TmuxSessionState(
        session_name="cc-n1001-42",
        session_dir=str(session_dir),
        session_id=session_id,
        mode="free",
        cwd="/home/devops/workspace/repos/foto_bot",
        mcp_config="",
        chat_id=-1001,
        provider="claude",
        transcript_path=None,
    )

    path = manager._transcript_path_for_state(state)

    assert path == (
        config_dir / "projects" / "-home-devops-workspace-repos-foto-bot" / f"{session_id}.jsonl"
    )
    assert state.transcript_path == str(path)


def test_claude_runtime_metadata_rejects_wrong_session(tmp_path: Path) -> None:
    manager = TmuxManager(tmp_path / "sessions")
    session_id = "01a14985-de09-4190-bc15-4c4c258f60b0"
    session_dir = tmp_path / "sessions" / "cc-n1001-42"
    session_dir.mkdir(parents=True)
    metadata = session_dir / "provider.runtime.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "claude_code",
                "session_id": "11a14985-de09-4190-bc15-4c4c258f60b0",
                "config_dir": str(tmp_path / "untrusted"),
            }
        ),
        encoding="utf-8",
    )
    metadata.chmod(0o600)
    known_path = tmp_path / "known.jsonl"
    state = TmuxSessionState(
        session_name="cc-n1001-42",
        session_dir=str(session_dir),
        session_id=session_id,
        mode="free",
        cwd=str(tmp_path),
        mcp_config="",
        chat_id=-1001,
        provider="claude",
        transcript_path=str(known_path),
    )

    assert manager._transcript_path_for_state(state) == known_path
