"""Optional fail-closed dispatch adapter for ai-agent-runtime.

The adapter sends channel identity only. User prompts never leave the normal
Telegram Agent -> tmux path through this protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import struct
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from telegram_bot.core.types import ChannelKey

_HEADER = struct.Struct("!I")
_MAX_FRAME_SIZE = 1024 * 1024


class DispatchDecision(StrEnum):
    ALLOW = "ALLOW"
    START_REQUIRED = "START_REQUIRED"
    RECYCLE_REQUIRED = "RECYCLE_REQUIRED"
    WAIT = "WAIT"


class RuntimeDispatchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DispatchResult:
    decision: DispatchDecision
    retry_after_seconds: float = 0.5
    reason: str | None = None
    desired_provider: str | None = None
    desired_model: str | None = None


class RuntimeDispatchGuard:
    """Version-qualified local socket client for pre-dispatch authorization."""

    def __init__(
        self,
        *,
        socket_path: Path,
        agent_instance_id: str,
        managed_chat_id: int,
        wait_timeout_seconds: float,
    ) -> None:
        if not socket_path.is_absolute():
            raise ValueError("Runtime dispatch socket path must be absolute")
        if not agent_instance_id:
            raise ValueError("Runtime AgentInstance id is required")
        if managed_chat_id >= 0:
            raise ValueError("Runtime managed chat id must be a Telegram supergroup id")
        if wait_timeout_seconds <= 0:
            raise ValueError("Runtime dispatch wait timeout must be positive")
        self.socket_path = socket_path
        self.agent_instance_id = agent_instance_id
        self.managed_chat_id = managed_chat_id
        self.wait_timeout_seconds = wait_timeout_seconds

    def manages(self, channel_key: ChannelKey) -> bool:
        return channel_key[0] == self.managed_chat_id and channel_key[1] is not None

    async def prepare(self, channel_key: ChannelKey) -> DispatchResult:
        if not self.manages(channel_key):
            return DispatchResult(DispatchDecision.ALLOW)
        thread_id = channel_key[1]
        if thread_id is None:
            return DispatchResult(DispatchDecision.ALLOW)
        payload = {
            "protocol_major": 1,
            "protocol_minor": 1,
            "action": "prepare_dispatch",
            "request_id": str(uuid.uuid4()),
            "agent_instance_id": self.agent_instance_id,
            "chat_id": channel_key[0],
            "thread_id": thread_id,
        }
        response = await self._request(payload)
        if response.get("ok") is not True:
            error = response.get("error")
            detail = error.get("detail") if isinstance(error, dict) else None
            raise RuntimeDispatchError(
                str(detail) if isinstance(detail, str) else "Runtime rejected dispatch"
            )
        raw_decision = response.get("decision")
        if raw_decision == "DENY":
            reason = response.get("reason")
            raise RuntimeDispatchError(
                str(reason) if isinstance(reason, str) else "Runtime denied dispatch"
            )
        try:
            decision = DispatchDecision(str(raw_decision))
        except ValueError as exc:
            raise RuntimeDispatchError("Runtime returned an invalid dispatch decision") from exc
        retry_ms = response.get("retry_after_ms", 500)
        if not isinstance(retry_ms, int) or isinstance(retry_ms, bool) or retry_ms < 50:
            retry_ms = 500
        reason = response.get("reason")
        desired = response.get("desired")
        desired_provider = desired.get("provider") if isinstance(desired, dict) else None
        desired_model = desired.get("model") if isinstance(desired, dict) else None
        if not isinstance(desired_provider, str) or not desired_provider:
            raise RuntimeDispatchError("Runtime returned an invalid desired provider")
        if not isinstance(desired_model, str) or not desired_model:
            raise RuntimeDispatchError("Runtime returned an invalid desired model")
        return DispatchResult(
            decision=decision,
            retry_after_seconds=min(float(retry_ms) / 1000.0, 5.0),
            reason=str(reason) if isinstance(reason, str) else None,
            desired_provider=desired_provider,
            desired_model=desired_model,
        )

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        if not encoded or len(encoded) > _MAX_FRAME_SIZE:
            raise RuntimeDispatchError("Runtime dispatch request is too large")
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path),
                timeout=5.0,
            )
        except (OSError, TimeoutError) as exc:
            raise RuntimeDispatchError("Runtime dispatch socket is unavailable") from exc
        try:
            writer.write(_HEADER.pack(len(encoded)))
            writer.write(encoded)
            await asyncio.wait_for(writer.drain(), timeout=5.0)
            header = await asyncio.wait_for(reader.readexactly(_HEADER.size), timeout=10.0)
            (size,) = _HEADER.unpack(header)
            if size == 0 or size > _MAX_FRAME_SIZE:
                raise RuntimeDispatchError("Runtime returned an invalid frame")
            raw = await asyncio.wait_for(reader.readexactly(size), timeout=10.0)
            decoded = json.loads(raw)
            if not isinstance(decoded, dict) or decoded.get("protocol_major") != 1:
                raise RuntimeDispatchError("Runtime returned an invalid protocol response")
            return decoded
        except (OSError, TimeoutError, asyncio.IncompleteReadError, json.JSONDecodeError) as exc:
            raise RuntimeDispatchError("Runtime dispatch request failed") from exc
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
