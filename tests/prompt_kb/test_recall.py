"""Tests for recall.py: formatting helpers + kb_recent/kb_recall logic."""

import pytest

pytest.importorskip("numpy")  # recall_memories depends on the [rag] extra

from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending  # noqa: E402
from claude_tap.prompt_kb.recall import (  # noqa: E402
    EMPTY_NOTE,
    RECALL_NOTE,
    RECENT_NOTE,
    format_attribution,
    recall_memories,
    recent_overview,
    truncate,
)
from claude_tap.prompt_kb.store import KbStore  # noqa: E402
from tests.prompt_kb.fake_embedder import FakeEmbedder  # noqa: E402


def _msg(store, *, session, ts, text, role="user", hash_, msg_idx=0, client="claude-code"):
    return store.upsert_message(
        session_id=session,
        record_index=0,
        message_index=msg_idx,
        client=client,
        model="m",
        timestamp=ts,
        content_hash=hash_,
        text=text,
        role=role,
        seen_at=ts,
    )


def test_truncate_short_text_untouched():
    assert truncate("hello", 10) == "hello"
    assert truncate("x" * 10, 10) == "x" * 10  # exactly at limit: no marker


def test_truncate_appends_ellipsis():
    out = truncate("x" * 500, 200)
    assert out == "x" * 199 + "…"
    assert len(out) == 200


def test_format_attribution():
    assert (
        format_attribution("2026-08-15T14:32:47Z", "claude-code", "7f3a9c12-abcd")
        == "2026-08-15 14:32 · claude-code · session 7f3a9c"
    )


def test_recent_overview_empty_store(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    result = recent_overview(store, client=None, sessions=5, messages_per_session=3)
    assert result == {"sessions": [], "note": EMPTY_NOTE}


def test_recent_overview_structure_and_order(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _msg(store, session="s1", ts="2026-08-01T09:00:00Z", text="task one", hash_="a")
    _msg(store, session="s1", ts="2026-08-01T09:05:00Z", text="doing it", hash_="b", role="assistant", msg_idx=1)
    _msg(store, session="s2", ts="2026-08-02T10:00:00Z", text="task two", hash_="c")
    result = recent_overview(store, client=None, sessions=5, messages_per_session=3)
    assert result["note"] == RECENT_NOTE
    sessions = result["sessions"]
    assert [s["session_id"] for s in sessions] == ["s2", "s1"]
    s1 = sessions[1]
    assert s1["time_range"] == "2026-08-01 09:00 → 09:05"
    assert s1["first_user_message"] == "task one"
    assert [(m["role"], m["text"]) for m in s1["recent_exchanges"]] == [
        ("user", "task one"),
        ("assistant", "doing it"),
    ]
    assert all(m["timestamp"] for m in s1["recent_exchanges"])


def test_recent_overview_limits_and_truncation(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _msg(store, session="s1", ts="2026-08-01T09:00:00Z", text="t" * 500, hash_="a")
    for i in range(4):
        _msg(
            store,
            session="s1",
            ts=f"2026-08-01T10:0{i}:00Z",
            text="x" * 400,
            hash_=f"b{i}",
            role="assistant",
            msg_idx=i + 1,
        )
    result = recent_overview(store, client=None, sessions=5, messages_per_session=2)
    s1 = result["sessions"][0]
    assert len(s1["first_user_message"]) == 200  # 199 chars + ellipsis
    assert len(s1["recent_exchanges"]) == 2
    assert all(len(m["text"]) == 300 for m in s1["recent_exchanges"])


def test_recall_memories_flattens_and_attributes(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    _msg(
        store,
        session="s1",
        ts="2026-08-01T09:00:00Z",
        text="how do I fix the race condition in the worker pool",
        hash_="r1",
    )
    index_pending(store, embedder)
    result = recall_memories(
        store,
        embedder,
        "race condition lock",
        client=None,
        limit=5,
        min_score=None,
        reranker=None,
        rrf_weights=(1.0, 1.0, 1.0),
    )
    assert result["note"] == RECALL_NOTE
    assert result["reranked"] is False
    assert len(result["memories"]) == 1
    mem = result["memories"][0]
    assert "race condition" in mem["text"]
    assert mem["role"] == "user"
    assert mem["session_id"] == "s1"
    assert mem["client"] == "claude-code"
    assert mem["timestamp"] == "2026-08-01T09:00:00Z"
    assert mem["attribution"] == "2026-08-01 09:00 · claude-code · session s1"
    assert mem["score"] > 0


def test_recall_memories_empty_store(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    result = recall_memories(
        store,
        embedder,
        "anything",
        client=None,
        limit=5,
        min_score=None,
        reranker=None,
        rrf_weights=(1.0, 1.0, 1.0),
    )
    assert result == {"memories": [], "note": RECALL_NOTE, "reranked": False}


def test_recall_memories_respects_limit(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    for i in range(4):
        _msg(
            store,
            session=f"s{i}",
            ts=f"2026-08-0{i + 1}T09:00:00Z",
            text=f"shared topic worker pool variant {i}",
            hash_=f"l{i}",
        )
    index_pending(store, embedder)
    result = recall_memories(
        store,
        embedder,
        "worker pool",
        client=None,
        limit=2,
        min_score=0.0,
        reranker=None,
        rrf_weights=(1.0, 1.0, 1.0),
    )
    assert len(result["memories"]) == 2
