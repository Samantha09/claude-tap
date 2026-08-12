"""Extract user and assistant messages from trace records for semantic session search.

Only genuine user-authored text is kept: tool results, harness-injected
pseudo-user messages (<system-reminder>, command envelopes), empty text,
and binary attachments are dropped. Provider parsing reuses the
normalization helpers in claude_tap.prompt_snapshot.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from claude_tap.prompt_kb.chunk import MAX_SECTION_CHARS, _split_long
from claude_tap.prompt_snapshot import _content_text, _request_body, infer_provider

_HARNESS_PREFIXES = ("<system-reminder", "<command-message", "<local-command")

MIN_ASSISTANT_CHARS = 20  # short acknowledgements carry no search value

# Drop rules ported from claude_tap.viewer._clean_session_user_text: harness
# boilerplate injected by Codex/Gemini CLIs as role=user messages. Only the
# viewer's drop rules are ported, not its extraction rules.
_DROP_FIRST_TAGS = {
    "artifacts",
    "codex_internal_context",
    "environment_context",
    "local-command-caveat",
    "session_context",
    "skills",
    "slash_commands",
    "subagents",
    "system-reminder",
    "user_information",
}

_DROP_STARTSWITH = (
    "# AGENTS.md instructions",
    "<INSTRUCTIONS>",
    "# Files mentioned by the user:",
)

_DROP_PATTERNS = (
    re.compile(r"^</?image(_input)?(\s+[^>]*)?>$", flags=re.IGNORECASE),
    re.compile(r"^\[SUGGESTION MODE:", flags=re.IGNORECASE),
    re.compile(r"^(web page content|page content|网页内容)\s*[:：]", flags=re.IGNORECASE),
    re.compile(r"^\[Image:\s*source:", flags=re.IGNORECASE),
)

_FIRST_TAG_RE = re.compile(r"^<([A-Za-z_-]+)")


@dataclass(frozen=True)
class UserMessage:
    record_index: int
    message_index: int
    timestamp: str
    text: str


@dataclass(frozen=True)
class AssistantMessage:
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


def extract_assistant_messages(records: list[dict[str, Any]]) -> list[AssistantMessage]:
    """Extract assistant reply text from response bodies.

    Only visible text is kept: thinking blocks, tool calls, and replies
    shorter than MIN_ASSISTANT_CHARS are dropped. Malformed/missing response
    bodies are skipped silently (pure reads; nothing transient to retry).
    """
    out: list[AssistantMessage] = []
    for record_index, record in enumerate(records):
        body = _response_body(record)
        if not body:
            continue
        text = _assistant_text(infer_provider(record), body).strip()
        if len(text) < MIN_ASSISTANT_CHARS:
            continue
        for message_index, piece in enumerate(_split_message(text)):
            out.append(
                AssistantMessage(
                    record_index=record_index,
                    message_index=message_index,
                    timestamp=str(record.get("timestamp") or ""),
                    text=piece,
                )
            )
    return out


def _response_body(record: dict[str, Any]) -> dict[str, Any]:
    resp = record.get("response") if isinstance(record.get("response"), dict) else {}
    body = resp.get("body")
    return body if isinstance(body, dict) else {}


def _assistant_text(provider: str, body: dict[str, Any]) -> str:
    if provider == "anthropic":
        content = body.get("content")
        if not isinstance(content, list):
            return ""
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        return "\n\n".join(t.strip() for t in texts if t.strip())
    if provider == "openai":
        return _openai_assistant_text(body)
    if provider == "gemini":
        candidates = body.get("candidates")
        if not isinstance(candidates, list):
            return ""
        texts: list[str] = []
        for cand in candidates:
            content = cand.get("content") if isinstance(cand, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            texts.extend(
                part["text"]
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
            )
        return "\n\n".join(t.strip() for t in texts if t.strip())
    return ""


def _openai_assistant_text(body: dict[str, Any]) -> str:
    texts: list[str] = []
    choices = body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                texts.append(_content_text(message.get("content")))
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str)
                )
    return "\n\n".join(t.strip() for t in texts if t.strip())


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
    if any(stripped.startswith(prefix) for prefix in _HARNESS_PREFIXES):
        return False
    first_tag = _FIRST_TAG_RE.match(stripped)
    if first_tag and first_tag.group(1).lower() in _DROP_FIRST_TAGS:
        return False
    if stripped.startswith(_DROP_STARTSWITH):
        return False
    return not any(pattern.match(stripped) for pattern in _DROP_PATTERNS)


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
