"""Cosine-similarity search over indexed chunks and user messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from claude_tap.prompt_kb.embed import Embedder, EmbedderUnavailable
from claude_tap.prompt_kb.store import KbStore


class ReindexRequired(Exception):
    """Stored embeddings were produced by a different embedder."""


@dataclass(frozen=True)
class SearchHit:
    kind: str
    title: str
    text: str
    score: float


@dataclass
class SnapshotResult:
    snapshot_id: int
    client: str
    model: str
    first_seen: str
    last_seen: str
    session_count: int
    hits: list[SearchHit] = field(default_factory=list)


@dataclass(frozen=True)
class MessageHit:
    text: str
    timestamp: str
    score: float
    role: str


@dataclass
class SessionResult:
    session_id: str
    client: str
    model: str
    hits: list[MessageHit] = field(default_factory=list)


def _cosine_scores(matrix, query_vec):
    import numpy as np

    q_norm = np.linalg.norm(query_vec) or 1.0
    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms[m_norms == 0] = 1.0
    return (matrix @ query_vec) / (m_norms * q_norm)


def _check_embedder_meta(store: KbStore, embedder: Embedder) -> None:
    name = store.get_meta("embedder_name")
    dim = store.get_meta("embedding_dim")
    if name is None:
        return  # nothing indexed yet
    if name != embedder.name or (dim and embedder.dimension and dim != str(embedder.dimension)):
        raise ReindexRequired(
            f"indexed with {name} (dim {dim}), current embedder is "
            f"{embedder.name} (dim {embedder.dimension}); run `claude-tap kb reindex`"
        )


def _dedup_across_snapshots(scored: list[tuple[Any, float]]) -> list[tuple[Any, float, int]]:
    """Collapse (kind, title, text) duplicates that recur across snapshots.

    Keeps the highest-scoring occurrence (ties: newest last_seen); dropped
    snapshots' session counts are summed into a bonus so the surviving
    group's session_count still reflects total exposure.
    """
    best: dict[tuple[str, str, str], list] = {}
    for row, score in scored:
        key = (row["kind"], row["title"] or "", row["text"])
        entry = best.setdefault(key, [row, score, 0])
        if (score, row["last_seen"]) > (entry[1], entry[0]["last_seen"]):
            entry[2] += int(entry[0]["session_count"])  # demote previous winner
            entry[0], entry[1] = row, score
        elif entry[0] is not row:
            entry[2] += int(row["session_count"])  # drop the challenger
    return [(entry[0], entry[1], entry[2]) for entry in best.values()]


def search(
    store: KbStore,
    embedder: Embedder,
    query: str,
    *,
    client: str | None = None,
    kind: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    rel_delta: float = 0.05,
) -> list[SnapshotResult]:
    _check_embedder_meta(store, embedder)
    rows = store.indexed_chunks()
    if client:
        rows = [row for row in rows if row["client"] == client]
    if kind:
        rows = [row for row in rows if row["kind"] == kind]
    if not rows:
        return []
    try:
        import numpy as np
    except ImportError as exc:
        raise EmbedderUnavailable(
            "numpy is not installed; install the optional dependency: pip install 'claude-tap[rag]'"
        ) from exc
    matrix = np.array([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
    embed_query = getattr(embedder, "embed_query", None) or embedder.embed
    query_vec = np.array(embed_query([query])[0], dtype=np.float32)
    scores = _cosine_scores(matrix, query_vec)

    scored = [(row, float(score)) for row, score in zip(rows, scores) if score > min_score]
    if not scored:
        return []
    top = max(score for _, score in scored)
    # Relative floor: keep hits close to the best score (absolute scores are
    # not calibrated; rel_delta=1.0 disables the floor entirely).
    kept = [(row, score) for row, score in scored if score > top - rel_delta or score == top]
    groups: dict[int, SnapshotResult] = {}
    for row, score, bonus_sessions in _dedup_across_snapshots(kept):
        group = groups.setdefault(
            row["snapshot_id"],
            SnapshotResult(
                snapshot_id=row["snapshot_id"],
                client=row["client"],
                model=row["model"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                session_count=row["session_count"],
            ),
        )
        group.session_count += bonus_sessions
        group.hits.append(
            SearchHit(kind=row["kind"], title=row["title"] or "", text=row["text"], score=score)
        )
    ordered = sorted(
        groups.values(),
        key=lambda g: max((h.score for h in g.hits), default=0.0),
        reverse=True,
    )
    for group in ordered:
        group.hits = sorted(group.hits, key=lambda h: h.score, reverse=True)[:3]
    return ordered[:limit]


def search_messages(
    store: KbStore,
    embedder: Embedder,
    query: str,
    *,
    client: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    rel_delta: float = 0.05,
) -> list[SessionResult]:
    """Cosine-similarity search over indexed user messages, grouped by session."""
    _check_embedder_meta(store, embedder)
    rows = store.indexed_messages()
    if client:
        rows = [row for row in rows if row["client"] == client]
    if not rows:
        return []
    try:
        import numpy as np
    except ImportError as exc:
        raise EmbedderUnavailable(
            "numpy is not installed; install the optional dependency: pip install 'claude-tap[rag]'"
        ) from exc
    matrix = np.array([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
    embed_query = getattr(embedder, "embed_query", None) or embedder.embed
    query_vec = np.array(embed_query([query])[0], dtype=np.float32)
    scores = _cosine_scores(matrix, query_vec)

    scored = [(row, float(score)) for row, score in zip(rows, scores) if score > min_score]
    if not scored:
        return []
    top = max(score for _, score in scored)
    # No cross-snapshot dedup here: messages are already unique by content_hash.
    kept = [(row, score) for row, score in scored if score > top - rel_delta or score == top]
    groups: dict[str, SessionResult] = {}
    for row, score in kept:
        group = groups.setdefault(
            row["session_id"],
            SessionResult(
                session_id=row["session_id"],
                client=row["client"],
                model=row["model"],
            ),
        )
        group.hits.append(
            MessageHit(text=row["text"], timestamp=row["timestamp"], score=float(score), role=row["role"])
        )
    ordered = sorted(
        groups.values(),
        key=lambda g: max((h.score for h in g.hits), default=0.0),
        reverse=True,
    )
    for group in ordered:
        group.hits = sorted(group.hits, key=lambda h: h.score, reverse=True)[:3]
    return ordered[:limit]
