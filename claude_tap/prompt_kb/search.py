"""Cosine-similarity search over indexed chunks, grouped by snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field

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


def search(
    store: KbStore,
    embedder: Embedder,
    query: str,
    *,
    client: str | None = None,
    kind: str | None = None,
    limit: int = 10,
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
    query_vec = np.array(embedder.embed([query])[0], dtype=np.float32)
    q_norm = np.linalg.norm(query_vec) or 1.0
    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms[m_norms == 0] = 1.0
    scores = (matrix @ query_vec) / (m_norms * q_norm)

    groups: dict[int, SnapshotResult] = {}
    for row, score in zip(rows, scores):
        if score <= 0:
            continue  # no token/vector overlap with the query
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
        group.hits.append(
            SearchHit(
                kind=row["kind"],
                title=row["title"] or "",
                text=row["text"],
                score=float(score),
            )
        )
    ordered = sorted(
        groups.values(),
        key=lambda g: max((h.score for h in g.hits), default=0.0),
        reverse=True,
    )
    for group in ordered:
        group.hits = sorted(group.hits, key=lambda h: h.score, reverse=True)[:3]
    return ordered[:limit]
