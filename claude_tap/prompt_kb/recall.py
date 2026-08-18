"""Memory-oriented retrieval behind the kb_recall / kb_recent MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from claude_tap.prompt_kb.search import search_messages

if TYPE_CHECKING:
    from claude_tap.prompt_kb.embed import Embedder
    from claude_tap.prompt_kb.rerank import Reranker
    from claude_tap.prompt_kb.store import KbStore

RECALL_NOTE = (
    "Memories from local past sessions, ranked by relevance; "
    "they may be outdated — verify against the current codebase before relying on them."
)
RECENT_NOTE = (
    "Recent sessions newest-first, independent of the current input — a plain timeline for picking up previous work."
)
EMPTY_NOTE = "The knowledge base is empty — no past sessions have been indexed yet."

_FIRST_MESSAGE_CHARS = 200
_EXCHANGE_CHARS = 300


def truncate(text: str, max_chars: int) -> str:
    """Hard-crop text to max_chars, marking truncation with an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def format_attribution(timestamp: str, client: str, session_id: str) -> str:
    """Human/model-readable provenance: '2026-08-15 14:32 · claude-code · session 7f3a9c'."""
    minute = timestamp[:16].replace("T", " ")
    return f"{minute} · {client} · session {session_id[:6]}"


def _fmt_minute(iso_ts: str) -> str:
    return iso_ts[:16].replace("T", " ")


def recent_overview(
    store: "KbStore",
    *,
    client: str | None,
    sessions: int,
    messages_per_session: int,
) -> dict[str, Any]:
    """Timeline of recent sessions for kb_recent; pure SQL, no embedder needed."""
    rows = store.recent_sessions(client, sessions)
    if not rows:
        return {"sessions": [], "note": EMPTY_NOTE}
    out: list[dict[str, Any]] = []
    for row in rows:
        first = store.session_first_user_message(row["session_id"])
        exchanges = store.session_last_messages(row["session_id"], messages_per_session)
        out.append(
            {
                "session_id": row["session_id"],
                "client": row["client"],
                "time_range": f"{_fmt_minute(row['first_ts'])} → {_fmt_minute(row['last_ts'])[-5:]}",
                "first_user_message": truncate(first["text"], _FIRST_MESSAGE_CHARS) if first else "",
                "recent_exchanges": [
                    {"role": m["role"], "text": truncate(m["text"], _EXCHANGE_CHARS), "timestamp": m["seen_at"]}
                    for m in exchanges
                ],
            }
        )
    return {"sessions": out, "note": RECENT_NOTE}


def recall_memories(
    store: "KbStore",
    embedder: "Embedder",
    query: str,
    *,
    client: str | None,
    limit: int,
    min_score: float | None,
    reranker: "Reranker | None",
    rrf_weights: tuple[float, float, float],
) -> dict[str, Any]:
    """Flattened relevance-ranked message hits for kb_recall (messages corpus only)."""
    groups, reranked = search_messages(
        store,
        embedder,
        query,
        client=client,
        limit=limit,
        min_score=min_score,
        reranker=reranker,
        rrf_weights=rrf_weights,
    )
    hits = [(hit, group) for group in groups for hit in group.hits]
    hits.sort(key=lambda pair: pair[0].score, reverse=True)
    memories = []
    for hit, group in hits[:limit]:
        memories.append(
            {
                "text": hit.text,
                "role": hit.role,
                "score": hit.score,
                "attribution": format_attribution(hit.timestamp, group.client, group.session_id),
                "session_id": group.session_id,
                "client": group.client,
                "timestamp": hit.timestamp,
            }
        )
    return {"memories": memories, "note": RECALL_NOTE, "reranked": reranked}
