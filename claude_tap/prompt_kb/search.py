"""Hybrid search: vector + FTS keyword channels, RRF-fused, reranked."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from claude_tap.prompt_kb.embed import Embedder, EmbedderUnavailable
from claude_tap.prompt_kb.rerank import Reranker
from claude_tap.prompt_kb.store import KbStore
from claude_tap.prompt_kb.tokenize import segment


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


_RRF_K = 60
_WORD_RE = re.compile(r"[A-Za-z0-9_一-鿿]+")


def _match_query(text: str) -> str:
    """Sanitize raw text into an FTS5 MATCH query (OR of word tokens)."""
    return " OR ".join(_WORD_RE.findall(text))


def _rrf_fuse(rankings: list[list[tuple[int, float]]], limit: int) -> list[int]:
    """Reciprocal-rank fusion over (rowid, score) rankings → fused rowids, best first."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (rowid, _score) in enumerate(ranking, 1):
            fused[rowid] = fused.get(rowid, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(fused, key=fused.__getitem__, reverse=True)[:limit]


def _fts_channels(store: KbStore, entity: str, query: str, recall: int) -> list[list[tuple[int, float]]]:
    """Keyword rankings: trigram on raw terms, jieba-table on segmented terms."""
    channels = []
    tri_match = _match_query(query)
    if tri_match:
        channels.append(store.fts_rank(entity, "tri", tri_match, recall))
    jieba_match = _match_query(segment(query))
    if jieba_match:
        channels.append(store.fts_rank(entity, "jieba", jieba_match, recall))
    return channels


def _final_scores(
    reranker: Reranker | None, query: str, texts: list[str], fallback: list[float]
) -> tuple[list[float], bool]:
    """Reranker (sigmoid-calibrated) scores when available, else cosine fallback.

    A reranker that raises at runtime degrades to the fallback: search must
    never fail because an enhancement failed.
    """
    if reranker is None:
        return fallback, False
    if not texts:
        return fallback, True
    try:
        return [float(score) for score in reranker.rerank(query, texts)], True
    except Exception:  # noqa: BLE001 - a broken reranker must not break search
        return fallback, False


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
    recall: int = 20,
    reranker: Reranker | None = None,
) -> tuple[list[SnapshotResult], bool]:
    """Hybrid search over chunks; returns (groups, reranked).

    Three channels (vector cosine, trigram FTS, jieba FTS) are RRF-fused into
    candidates; the reranker rescores them when available. Scores are reranker
    scores when reranked=True (calibrated, rel_delta ignored), else cosine
    fallback scores (rel_delta applies as before).
    """
    _check_embedder_meta(store, embedder)
    rows = store.indexed_chunks()
    if client:
        rows = [row for row in rows if row["client"] == client]
    if kind:
        rows = [row for row in rows if row["kind"] == kind]
    if not rows:
        return [], reranker is not None
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
    cosine_by_id = {int(row["id"]): float(score) for row, score in zip(rows, scores)}
    by_id = {int(row["id"]): row for row in rows}
    # Vector channel: positive-overlap rows only, so zero-cosine tail rows
    # cannot ride the fusion into the candidate pool.
    vector_ranking = sorted(
        ((row_id, score) for row_id, score in cosine_by_id.items() if score > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )[:recall]
    candidate_ids = [
        cid
        for cid in _rrf_fuse([vector_ranking, *_fts_channels(store, "chunks", query, recall)], recall)
        if cid in by_id
    ]
    if not candidate_ids:
        return [], reranker is not None
    # Cross-snapshot dedup before reranking (no rerank compute wasted on dups);
    # cosine breaks ties exactly as it did for the pre-hybrid pipeline.
    deduped = _dedup_across_snapshots([(by_id[cid], cosine_by_id[cid]) for cid in candidate_ids])
    final, reranked = _final_scores(
        reranker,
        query,
        [row["text"] for row, _score, _bonus in deduped],
        [cosine_by_id[int(row["id"])] for row, _score, _bonus in deduped],
    )
    if reranked:
        kept = [
            (row, score, bonus)
            for (row, _cos, bonus), score in zip(deduped, final)
            if score > min_score
        ]
    else:
        kept = [
            (row, score, bonus)
            for (row, _cos, bonus), score in zip(deduped, final)
            if score >= min_score
        ]
        if kept:
            top = max(score for _, score, _ in kept)
            # Relative floor for uncalibrated cosine scores (rel_delta=1.0 disables).
            kept = [(row, score, bonus) for row, score, bonus in kept if score > top - rel_delta or score == top]
    groups: dict[int, SnapshotResult] = {}
    for row, score, bonus_sessions in kept:
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
    return ordered[:limit], reranked


def search_messages(
    store: KbStore,
    embedder: Embedder,
    query: str,
    *,
    client: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    rel_delta: float = 0.05,
    recall: int = 20,
    reranker: Reranker | None = None,
) -> tuple[list[SessionResult], bool]:
    """Hybrid search over indexed chat messages (user + assistant), grouped by session."""
    _check_embedder_meta(store, embedder)
    rows = store.indexed_messages()
    if client:
        rows = [row for row in rows if row["client"] == client]
    if not rows:
        return [], reranker is not None
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
    cosine_by_id = {int(row["id"]): float(score) for row, score in zip(rows, scores)}
    by_id = {int(row["id"]): row for row in rows}
    vector_ranking = sorted(
        ((row_id, score) for row_id, score in cosine_by_id.items() if score > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )[:recall]
    candidate_ids = [
        cid
        for cid in _rrf_fuse([vector_ranking, *_fts_channels(store, "messages", query, recall)], recall)
        if cid in by_id
    ]
    if not candidate_ids:
        return [], reranker is not None
    candidates = [by_id[cid] for cid in candidate_ids]
    final, reranked = _final_scores(
        reranker,
        query,
        [row["text"] for row in candidates],
        [cosine_by_id[int(row["id"])] for row in candidates],
    )
    if reranked:
        kept = [(row, score) for row, score in zip(candidates, final) if score > min_score]
    else:
        kept = [(row, score) for row, score in zip(candidates, final) if score >= min_score]
        if kept:
            top = max(score for _, score in kept)
            kept = [(row, score) for row, score in kept if score > top - rel_delta or score == top]
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
    return ordered[:limit], reranked
