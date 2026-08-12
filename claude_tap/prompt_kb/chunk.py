"""Split prompt snapshots into embeddable chunks and compute content hashes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from claude_tap.prompt_snapshot import PromptSnapshot, PromptTool

MAX_SECTION_CHARS = 2000
MIN_SECTION_CHARS = 200

# Harness-injected template sections: present in nearly every snapshot, low
# information value, and their git/shell/CLI vocabulary poisons similarity.
BOILERPLATE_TITLES = frozenset({"environment", "context management", "harness", "session-specific guidance"})

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    kind: str  # "prompt_section" | "tool"
    title: str
    text: str


def chunk_snapshot(snapshot: PromptSnapshot) -> list[Chunk]:
    chunks: list[Chunk] = []
    for label, prompt in (("system", snapshot.system_prompt), ("developer", snapshot.developer_prompt)):
        if prompt and prompt.strip():
            chunks.extend(_split_prompt(prompt))
    chunks.extend(_tool_chunk(tool) for tool in snapshot.tools)
    return chunks


def _split_prompt(text: str) -> list[Chunk]:
    # Drop boilerplate-titled sections before merging, so a short boilerplate
    # block can't be folded into a neighbor and escape the title check.
    sections = [(t, b) for t, b in _heading_sections(text.strip()) if t.strip().lower() not in BOILERPLATE_TITLES]
    merged = _merge_small(sections)
    chunks: list[Chunk] = []
    for title, body in merged:
        if title.strip().lower() in BOILERPLATE_TITLES:
            continue
        for piece in _split_long(title, body):
            chunks.append(Chunk(kind="prompt_section", title=piece[0], text=piece[1]))
    return chunks


def _heading_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown headings; text before the first heading is one section."""
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current_lines or current_title:
                sections.append((current_title, current_lines))
            current_title = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines or current_title:
        sections.append((current_title, current_lines))
    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def _merge_small(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge sections smaller than MIN_SECTION_CHARS into an adjacent large
    section (preferring the previous one). Small sections with no large
    neighbor (e.g. the whole prompt is tiny) are kept as-is, so headings
    still produce distinct chunks."""
    merged: list[tuple[str, str]] = []
    pending: list[tuple[str, str]] = []  # leading small sections awaiting a large successor
    for title, body in sections:
        if len(body) < MIN_SECTION_CHARS:
            if merged:
                prev_title, prev_body = merged[-1]
                merged[-1] = (prev_title, prev_body + "\n\n" + body)
            else:
                pending.append((title, body))
        else:
            if pending:
                body = "\n\n".join([b for _, b in pending] + [body])
                pending = []
            merged.append((title, body))
    merged.extend(pending)
    return merged


def _split_long(title: str, body: str) -> list[tuple[str, str]]:
    if len(body) <= MAX_SECTION_CHARS:
        return [(title, body)]
    pieces: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0
    for para in re.split(r"\n\s*\n", body):
        if current and current_len + len(para) + 2 > MAX_SECTION_CHARS:
            pieces.append((title, "\n\n".join(current)))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para) + 2
    if current:
        pieces.append((title, "\n\n".join(current)))
    # Hard-split any paragraph that alone exceeds the limit.
    result: list[tuple[str, str]] = []
    for piece_title, piece in pieces:
        while len(piece) > MAX_SECTION_CHARS:
            result.append((piece_title, piece[:MAX_SECTION_CHARS]))
            piece = piece[MAX_SECTION_CHARS:]
        if piece:
            result.append((piece_title, piece))
    return result


def _tool_chunk(tool: PromptTool) -> Chunk:
    params = _tool_param_names(tool.schema)
    text = tool.name
    if tool.description:
        text += "\n" + tool.description
    if params:
        text += "\n参数: " + ", ".join(params)
    return Chunk(kind="tool", title=tool.name, text=text)


def _tool_param_names(schema: dict[str, Any]) -> list[str]:
    for key in ("input_schema", "parameters"):
        wrapper = schema.get(key)
        if isinstance(wrapper, dict):
            props = wrapper.get("properties")
            if isinstance(props, dict):
                return sorted(str(name) for name in props)
    props = schema.get("properties")
    if isinstance(props, dict):
        return sorted(str(name) for name in props)
    return []


def content_hash(client: str, model: str, snapshot: PromptSnapshot) -> str:
    payload = {
        "client": client,
        "model": model,
        "system": _normalize_text(snapshot.system_prompt),
        "developer": _normalize_text(snapshot.developer_prompt),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "params": _tool_param_names(tool.schema),
                # Canonicalized schema internals so a schema change with an
                # unchanged description still produces a new version hash.
                "schema": json.dumps(tool.schema, ensure_ascii=False, sort_keys=True),
            }
            for tool in sorted(snapshot.tools, key=lambda t: t.name)
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").splitlines()).strip()
