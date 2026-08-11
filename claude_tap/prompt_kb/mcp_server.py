"""MCP stdio server: expose the prompt KB to agents as read-only tools."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, Literal

from claude_tap.prompt_kb.embed import EmbedderUnavailable, create_embedder, load_config
from claude_tap.prompt_kb.index import index_pending
from claude_tap.prompt_kb.search import ReindexRequired, search, search_messages
from claude_tap.prompt_kb.store import KbStore

if TYPE_CHECKING:
    from claude_tap.prompt_kb.embed import Embedder

_ctx: tuple[KbStore, Embedder] | None = None


def _get_ctx() -> tuple[KbStore, Embedder]:
    """Lazily open the KB store and build the embedder.

    The embedding model loads on first tool call, not at server startup.
    """
    global _ctx
    if _ctx is None:
        _ctx = (KbStore.default(), create_embedder(load_config()))
    return _ctx


def kb_search(
    query: str,
    client: str | None = None,
    kind: Literal["tool", "prompt_section"] | None = None,
    limit: int = 10,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Search the local prompt knowledge base (prompts, tool definitions, user messages).

    Args:
        query: Natural-language query, e.g. "which CLI has a sandboxed shell tool".
        client: Optional client filter, e.g. "claude-code" or "codex".
        kind: Optional chunk-kind filter for the chunks section.
        limit: Max groups per section.
        min_score: Minimum cosine score (0-1) for a hit to be included.

    Returns:
        {"chunks": [...], "messages": [...]} grouped by snapshot / session.
    """
    try:
        store, embedder = _get_ctx()
    except EmbedderUnavailable as exc:
        return {"error": f"embedder unavailable: {exc}", "chunks": [], "messages": []}
    try:
        index_pending(store, embedder)
    except sqlite3.OperationalError:
        pass  # dashboard's lazy indexer holds the write lock; search the stale index
    try:
        chunk_groups = search(store, embedder, query, client=client, kind=kind, limit=limit, min_score=min_score)
        message_groups = search_messages(store, embedder, query, client=client, limit=limit, min_score=min_score)
    except ReindexRequired as exc:
        return {"error": str(exc), "chunks": [], "messages": []}
    return {
        "chunks": [
            {
                "client": g.client,
                "model": g.model,
                "first_seen": g.first_seen,
                "last_seen": g.last_seen,
                "session_count": g.session_count,
                "hits": [{"kind": h.kind, "title": h.title, "text": h.text, "score": h.score} for h in g.hits],
            }
            for g in chunk_groups
        ],
        "messages": [
            {
                "session_id": g.session_id,
                "client": g.client,
                "model": g.model,
                "hits": [{"text": h.text, "timestamp": h.timestamp, "score": h.score} for h in g.hits],
            }
            for g in message_groups
        ],
    }


def kb_status() -> dict[str, Any]:
    """Report knowledge-base index size and embedder identity.

    Returns:
        store.stats() keys (snapshots/chunks/pending/failed/indexed/messages)
        plus "embedder" (indexed embedder name, or "none").
    """
    store = KbStore.default()
    return {**store.stats(), "embedder": store.get_meta("embedder_name") or "none"}
