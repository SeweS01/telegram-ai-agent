from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from telegram_bot.core.handlers.streaming import ensure_exec_mode_ready
from telegram_bot.core.services.runtime_dispatch import (
    DispatchDecision,
    DispatchResult,
    RuntimeDispatchError,
    RuntimeDispatchGuard,
)


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
    assert result.desired_model == "gpt-new"

    guard._request = AsyncMock(  # type: ignore[method-assign]
        return_value={"ok": True, "decision": "DENY", "reason": "unsafe"}
    )
    with pytest.raises(RuntimeDispatchError, match="unsafe"):
        await guard.prepare((-1001, 42))


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
