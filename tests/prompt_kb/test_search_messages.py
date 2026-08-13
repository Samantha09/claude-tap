"""Semantic search over indexed user messages."""

import pytest

from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending
from claude_tap.prompt_kb.search import ReindexRequired, search_messages
from claude_tap.prompt_kb.store import KbStore
from tests.prompt_kb.fake_embedder import FakeEmbedder
from tests.prompt_kb.fake_reranker import FakeReranker


@pytest.fixture()
def seeded(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    for i, (sid, text) in enumerate(
        [
            ("s1", "how do I fix the race condition in the worker pool"),
            ("s1", "the lock ordering was wrong"),
            ("s2", "recipe for tomato soup"),
        ]
    ):
        store.upsert_message(
            session_id=sid,
            record_index=i,
            message_index=0,
            client="claude" if sid == "s1" else "codex",
            model="k3",
            timestamp=f"2026-08-0{i + 1}T00:00:00Z",
            content_hash=f"h{i}",
            text=text,
            seen_at="t",
        )
    index_pending(store, embedder)
    return store, embedder


def test_search_groups_by_session(seeded):
    store, embedder = seeded
    results, _ = search_messages(store, embedder, "race condition lock")
    assert results
    top = results[0]
    assert top.session_id == "s1"
    assert top.client == "claude"
    assert all(h.text for h in top.hits)
    assert all(h.score > 0 for h in top.hits)
    assert top.hits[0].timestamp  # timestamp carried through


def test_search_client_filter(seeded):
    store, embedder = seeded
    # "recipe" hashes to bucket 1; no s1 token lands in bucket 1, so the
    # claude messages have zero overlap with the query. ("tomato"/"soup"
    # collide with s1 tokens in buckets 14/0 and would leak through.)
    results, _ = search_messages(store, embedder, "recipe", client="codex")
    assert [r.session_id for r in results] == ["s2"]
    results, _ = search_messages(store, embedder, "recipe", client="claude")
    assert results == []


def test_search_min_score(seeded):
    store, embedder = seeded
    results, _ = search_messages(store, embedder, "race condition", min_score=0.99)
    assert results == []


def test_message_hit_carries_role(seeded):
    store, embedder = seeded
    store.upsert_message(
        session_id="s1", record_index=9, message_index=0, client="claude",
        model="k3", timestamp="2026-08-02T00:00:00Z", content_hash="h9",
        text="the race condition fix is a strict lock ordering protocol",
        seen_at="t", role="assistant",
    )
    index_pending(store, embedder)
    results, _ = search_messages(store, embedder, "race condition lock", rel_delta=1.0)
    roles = {h.role for g in results for h in g.hits}
    assert "assistant" in roles and "user" in roles


def test_messages_reranked_flag(seeded):
    store, embedder = seeded
    # FakeReranker: query tokens {race, condition, lock}; both s1 messages
    # overlap (2/3 and 1/3), the soup message scores 0.0 and is dropped.
    results, reranked = search_messages(store, embedder, "race condition lock", reranker=FakeReranker())
    assert reranked is True
    assert [g.session_id for g in results] == ["s1"]
    assert all(h.score > 0 for g in results for h in g.hits)


def test_messages_not_reranked_without_reranker(seeded):
    store, embedder = seeded
    results, reranked = search_messages(store, embedder, "race condition lock")
    assert reranked is False
    assert results


def test_reindex_required_on_embedder_mismatch(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    store.set_meta("embedder_name", "other")
    store.set_meta("embedding_dim", "99")
    with pytest.raises(ReindexRequired):
        search_messages(store, FakeEmbedder(), "q")


def test_perf_20k_chunks_under_200ms(tmp_path):
    import time

    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    rows = [
        (f"s{i % 100}", i, 0, "claude", "k3", "t", f"h{i}", f"message {i} about topic {i % 50}", "t")
        for i in range(20_000)
    ]
    with store._connect() as conn:
        conn.executemany(
            """INSERT INTO kb_messages
               (session_id, record_index, message_index, client, model,
                timestamp, content_hash, text, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    index_pending(store, embedder)
    start = time.perf_counter()
    search_messages(store, embedder, "topic 42 message")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2
