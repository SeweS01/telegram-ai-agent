"""Pane capture helpers: TUI state detectors, HTML escaping, readiness wait.

State detectors are substring-matching predicates on `tmux capture-pane -p`
output. `escape_pane_for_html` prepares a snapshot for Telegram `<pre>`
blocks — it HTML-escapes `&<>` and strips C0 control bytes (while keeping
`\\t` and `\\n`) to avoid Telegram rejecting the message.

`await_prompt_ready` is a net-new async polling wrapper used by the tmux
runner at session spawn. It completes known first-run and safety dialogs,
then polls until the prompt marker appears. On timeout it sends one fallback
Enter, polls for another 5s, then kills the session and returns False.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import re
import subprocess
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Real CC 2.1.114 trust-dialog text observed in PoC on 2026-04-19.
TRUST_DIALOG_SUBSTRINGS = (
    "trust this folder",
    "Do you trust the contents of this directory",  # Codex / older CC, fallback
)

# Claude Code 2.1.325 first-run screens observed with a fresh isolated
# CLAUDE_CONFIG_DIR. They precede the ordinary per-project trust dialog.
THEME_DIALOG_SUBSTRING = "Choose the text style that looks best with your terminal"
SECURITY_NOTES_SUBSTRINGS = ("Security notes:", "Press Enter to continue")
DANGEROUS_MODE_DIALOG_SUBSTRINGS = (
    "Claude Code running in Bypass Permissions mode",
    "Yes, I accept",
)

# Prompt-ready markers in CC TUI:
#   "❯ " — idle prompt, ready for input
#   "start a new conversation" — welcome screen
#   "/help" — mentioned in the welcome shortcut list
READINESS_MARKERS = (
    "❯",
    "start a new conversation",
    "/help",
)

_C0_STRIP_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_PROMPT_LINE_RE = re.compile(r"(?m)^[❯>]\s")

_POLL_INTERVAL_SEC = 0.5
_FALLBACK_POLL_BUDGET_SEC = 5.0
_DIALOG_SELECTION_SETTLE_SEC = 0.5


def is_trust_dialog(pane_text: str) -> bool:
    return any(s in pane_text for s in TRUST_DIALOG_SUBSTRINGS)


def first_run_dialog(pane_text: str) -> str | None:
    """Return the known Claude first-run dialog currently shown, if any."""
    if THEME_DIALOG_SUBSTRING in pane_text:
        return "theme"
    if all(marker in pane_text for marker in SECURITY_NOTES_SUBSTRINGS):
        return "security_notes"
    if all(marker in pane_text for marker in DANGEROUS_MODE_DIALOG_SUBSTRINGS):
        return "dangerous_mode"
    return None


def is_prompt_ready(pane_text: str) -> bool:
    if _PROMPT_LINE_RE.search(pane_text):
        return True
    return any(m in pane_text for m in ("start a new conversation", "/help"))


def escape_pane_for_html(pane_text: str) -> str:
    """Prepare a raw `tmux capture-pane -p` snapshot for Telegram `<pre>`.

    HTML-escape `&<>` (quote=False keeps `'` and `"` as-is since they are
    safe inside `<pre>`), then strip C0 control bytes `[\\x00-\\x08\\x0b-\\x1f]`
    plus DEL `\\x7f` which Telegram rejects. `\\t` (0x09) and `\\n` (0x0a)
    are preserved.
    """
    escaped = html.escape(pane_text, quote=False)
    return _C0_STRIP_RE.sub("", escaped)


# Same rationale as send_keys._TMUX_CMD_TIMEOUT_SEC: a hung tmux server must
# not block callers (often holding the per-channel lifecycle lock) forever.
_TMUX_CMD_TIMEOUT_SEC = 10.0


def _capture_pane(session_name: str) -> str:
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", f"={session_name}:", "-p", "-S", "-200"],
        check=True,
        capture_output=True,
        text=True,
        timeout=_TMUX_CMD_TIMEOUT_SEC,
    )
    return result.stdout


def _send_enter(session_name: str) -> None:
    subprocess.run(
        ["tmux", "send-keys", "-t", f"={session_name}:", "Enter"],
        check=True,
        timeout=_TMUX_CMD_TIMEOUT_SEC,
    )


def _send_down(session_name: str) -> None:
    subprocess.run(
        ["tmux", "send-keys", "-t", f"={session_name}:", "Down"],
        check=True,
        timeout=_TMUX_CMD_TIMEOUT_SEC,
    )


async def _confirm_dialog(session_name: str, *, select_yes: bool) -> bool:
    """Confirm one dialog, moving off a fail-safe No default if needed."""
    try:
        if select_yes:
            await asyncio.to_thread(_send_down, session_name)
            # Claude's TUI must render the selection change before Enter.
            # Sending Down+Enter together accepts the old No selection on
            # 2.1.325 and exits the process.
            await asyncio.sleep(_DIALOG_SELECTION_SETTLE_SEC)
        await asyncio.to_thread(_send_enter, session_name)
    except subprocess.SubprocessError:
        return False
    return True


def _kill_session(session_name: str) -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", f"={session_name}"],
        check=False,
        timeout=_TMUX_CMD_TIMEOUT_SEC,
    )


async def await_prompt_ready(
    session_name: str,
    timeout: float = 30.0,
    clock: Callable[[], float] | None = None,
) -> bool:
    """Poll a tmux session until the Claude CLI prompt is ready.

    Polls `tmux capture-pane` every 500ms. Exit conditions:
      (a) known first-run/safety dialog detected → confirm it once and keep
          polling; dialogs with a fail-safe `No, exit` default first receive
          Down, a render delay, then Enter.
      (b) trust-dialog detected + not yet handled → select the affirmative
          choice when required, confirm it, and keep polling.
      (c) prompt-ready detected → return True.
      (d) deadline reached → send one fallback Enter, poll for 5 more
          seconds; if still not ready → `tmux kill-session` + return False.

    `clock` is injectable for tests (default `time.monotonic`). Allows
    sharing a deadline with `_spawn_tmux` (Wave 2, Decision 7 shared budget).
    Any `subprocess.CalledProcessError` from capture-pane means tmux died
    — return False immediately.

    Due to the 5s fallback window, the minimum wall-time to a False result
    is ~5s even if `timeout` is smaller.
    """
    clock = clock or time.monotonic
    deadline = clock() + timeout
    trust_handled = False
    handled_first_run_dialogs: set[str] = set()
    pane = ""  # captured in loop; initialized so the timeout log never UnboundLocalError.

    while clock() < deadline:
        try:
            pane = await asyncio.to_thread(_capture_pane, session_name)
        except subprocess.SubprocessError:
            return False

        dialog = first_run_dialog(pane)
        if dialog is not None and dialog not in handled_first_run_dialogs:
            select_yes = dialog == "dangerous_mode" and "❯ No, exit" in pane
            if not await _confirm_dialog(session_name, select_yes=select_yes):
                return False
            handled_first_run_dialogs.add(dialog)
            await asyncio.sleep(_POLL_INTERVAL_SEC)
            continue

        if is_trust_dialog(pane) and not trust_handled:
            select_yes = "❯ No, exit" in pane and "Yes, I trust this folder" in pane
            if not await _confirm_dialog(session_name, select_yes=select_yes):
                return False
            trust_handled = True
            await asyncio.sleep(_POLL_INTERVAL_SEC)
            continue

        if is_prompt_ready(pane):
            return True

        await asyncio.sleep(_POLL_INTERVAL_SEC)

    # Main loop exhausted — one fallback Enter, then 5s grace window.
    try:
        await asyncio.to_thread(_send_enter, session_name)
    except subprocess.SubprocessError:
        return False

    fallback_deadline = clock() + _FALLBACK_POLL_BUDGET_SEC
    while clock() < fallback_deadline:
        try:
            pane = await asyncio.to_thread(_capture_pane, session_name)
        except subprocess.SubprocessError:
            return False
        if is_prompt_ready(pane):
            return True
        await asyncio.sleep(_POLL_INTERVAL_SEC)

    # Elevated to WARNING with last pane snippet — session won't start, user
    # sees generic error; this log is the only clue what CC was stuck on.
    last_lines = "\n".join(pane.splitlines()[-10:]) if pane else "<empty>"
    logger.warning(
        "TUI_IO: readiness timeout for session=%s after %.1fs, killing. Last pane lines:\n%s",
        session_name,
        timeout,
        last_lines,
    )
    with contextlib.suppress(subprocess.SubprocessError):
        await asyncio.to_thread(_kill_session, session_name)
    return False
