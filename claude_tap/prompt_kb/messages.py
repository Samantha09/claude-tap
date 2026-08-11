"""Extract user messages from trace records for semantic session search.

Only genuine user-authored text is kept: tool results, harness-injected
pseudo-user messages (<system-reminder>, command envelopes), empty text,
and binary attachments are dropped. Provider parsing reuses the
normalization helpers in claude_tap.prompt_snapshot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from claude_tap.prompt_kb.chunk import MAX_SECTION_CHARS, _split_long
from claude_tap.prompt_snapshot import _content_text, _request_body, infer_provider

_HARNES_PREFIXES = ("<system-reminder", "<command-message", "<local-command")


@dataclass(frozen=True)
class UserMessage:
    record_index: int
    message_index: int
    timestamp: str
    text: str


def message_content_hash(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_user_messages(records: list[dict[str, Any]]) -> list[UserMessage]:
    out: list[UserMessage] = []
    for record_index, record in enumerate(records):
        body = _request_body(record)
        if not body:
            continue
        provider = infer_provider(record)
        timestamp = str(record.get("timestamp") or "")
        message_index = 0
        for text in _user_texts(provider, body):
            for piece in _split_message(text):
                out.append(
                    UserMessage(
                        record_index=record_index,
                        message_index=message_index,
                        timestamp=timestamp,
                        text=piece,
                    )
                )
                message_index += 1
    return out


def _split_message(text: str) -> list[str]:
    if len(text) <= MAX_SECTION_CHARS:
        return [text]
    return [piece for _title, piece in _split_long("", text)]


def _user_texts(provider: str, body: dict[str, Any]) -> list[str]:
    if provider == "anthropic":
        return _anthropic_user_texts(body)
    if provider == "openai":
        return _openai_user_texts(body)
    if provider == "gemini":
        return _gemini_user_texts(body)
    return []


def _keep_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return not any(stripped.startswith(prefix) for prefix in _HARNES_PREFIXES)


def _anthropic_user_texts(body: dict[str, Any]) -> list[str]:
    out: list[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return out
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # Tool results travel as role=user; keep only real text blocks.
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
            ]
            text = "\n\n".join(t.strip() for t in texts if t.strip())
        else:
            text = _content_text(content)
        if _keep_text(text):
            out.append(text.strip())
    return out


def _openai_user_texts(body: dict[str, Any]) -> list[str]:
    out: list[str] = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = _content_text(msg.get("content"))
                if _keep_text(text):
                    out.append(text.strip())
    input_value = body.get("input")
    if isinstance(input_value, str):
        if _keep_text(input_value):
            out.append(input_value.strip())
    elif isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in (None, "message") or item.get("role") not in (None, "user"):
                continue
            text = _content_text(item.get("content"))
            if _keep_text(text):
                out.append(text.strip())
    prompt = body.get("prompt")
    if isinstance(prompt, str) and _keep_text(prompt):
        out.append(prompt.strip())
    return out


def _gemini_user_texts(body: dict[str, Any]) -> list[str]:
    out: list[str] = []
    contents = body.get("contents")
    if not isinstance(contents, list):
        return out
    for item in contents:
        if not isinstance(item, dict):
            continue
        if (item.get("role") or "user") != "user":
            continue
        parts = item.get("parts")
        if not isinstance(parts, list):
            continue
        texts = [
            part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]  # functionResponse parts carry no "text" key and are dropped implicitly
        text = "\n\n".join(t.strip() for t in texts if t.strip())
        if _keep_text(text):
            out.append(text.strip())
    return out
