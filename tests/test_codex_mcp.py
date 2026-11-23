from __future__ import annotations

import json
from pathlib import Path

from telegram_bot.core.services.codex_mcp import build_codex_mcp_config_args


def test_inherited_mcp_servers_are_reset_atomically() -> None:
    args = build_codex_mcp_config_args(
        None,
        inherited_server_names={"project_memory", "user_docs"},
    )

    assert args == ["-c", "mcp_servers={}"]


def test_topic_mcp_is_added_after_inherited_table_reset(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "bot": {
                        "command": "python",
                        "args": ["-m", "example_bot_mcp"],
                        "env": {"BOT_TOKEN": "kept-out-of-argv"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    args = build_codex_mcp_config_args(
        str(config),
        inherited_server_names={"project_memory"},
    )

    assert args[:2] == ["-c", "mcp_servers={}"]
    assert "mcp_servers.project_memory.enabled=false" not in args
    assert "mcp_servers.bot.enabled=true" in args
    assert all("kept-out-of-argv" not in value for value in args)
