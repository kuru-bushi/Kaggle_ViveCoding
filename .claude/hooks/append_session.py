#!/usr/bin/env python3
"""Claude Code session logger hook.

Invoked by `.claude/settings.json` on:
- UserPromptSubmit -> append the user's prompt to session/YYYY-MM-DD_live.md
- Stop             -> rewrite session/YYYY-MM-DD_<sid>.md from the full transcript

Idempotent: Stop overwrites the per-session markdown each turn so the latest
state is always saved (good for crash recovery).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

MAX_TOOL_INPUT_CHARS = 800
MAX_TOOL_RESULT_CHARS = 600


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n... <truncated {len(s) - n} chars>"


def _format_user_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for c in content:
        ctype = c.get("type", "")
        if ctype == "text":
            parts.append(c.get("text", ""))
        elif ctype == "tool_result":
            tc = c.get("content", "")
            if isinstance(tc, list):
                tc = "\n".join(p.get("text", "") for p in tc if p.get("type") == "text")
            parts.append(f"<tool_result>\n{_truncate(str(tc), MAX_TOOL_RESULT_CHARS)}\n</tool_result>")
        else:
            parts.append(f"<{ctype or 'unknown'}>")
    return "\n".join(parts)


def _format_assistant_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for c in content:
        ctype = c.get("type", "")
        if ctype == "text":
            parts.append(c.get("text", ""))
        elif ctype == "tool_use":
            name = c.get("name", "?")
            inp = json.dumps(c.get("input", {}), ensure_ascii=False)
            parts.append(f"**Tool: {name}**\n```json\n{_truncate(inp, MAX_TOOL_INPUT_CHARS)}\n```")
        elif ctype == "thinking":
            continue
    return "\n\n".join(p for p in parts if p)


def main() -> int:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    event = data.get("hook_event_name", "")
    session_id = data.get("session_id", "unknown")
    cwd = data.get("cwd", os.getcwd())
    session_dir = Path(cwd) / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if event == "UserPromptSubmit":
        prompt = data.get("prompt", "")
        live = session_dir / f"{date}_live.md"
        with live.open("a", encoding="utf-8") as f:
            f.write(f"\n## {time_str} — User (session {session_id[:8]})\n\n{prompt}\n")
        return 0

    if event == "Stop":
        transcript_path = data.get("transcript_path", "")
        if not transcript_path or not os.path.isfile(transcript_path):
            return 0
        out = session_dir / f"{date}_{session_id[:8]}.md"
        lines: list[str] = [
            f"# Session `{session_id}`",
            f"Generated: {now.isoformat(timespec='seconds')}",
            f"Transcript: `{transcript_path}`",
            "",
        ]
        with open(transcript_path, encoding="utf-8") as f:
            for raw_line in f:
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                etype = entry.get("type")
                ts = entry.get("timestamp", "")
                msg = entry.get("message", {})
                content = msg.get("content", "")
                if etype == "user":
                    body = _format_user_content(content)
                    if body.strip():
                        lines.append(f"## {ts} — User\n\n{body}\n")
                elif etype == "assistant":
                    body = _format_assistant_content(content)
                    if body.strip():
                        lines.append(f"## {ts} — Assistant\n\n{body}\n")
        out.write_text("\n".join(lines), encoding="utf-8")
        return 0

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Never let hook failures block Claude Code
        sys.exit(0)
