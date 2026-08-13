"""kb_messages storage: migration, dedup upsert, index state machine."""

import sqlite3

from claude_tap.prompt_kb.store import KbStore


def _msg(**kw):
    base = dict(
        session_id="s1",
        record_index=0,
        message_index=0,
        client="claude",
        model="k3",
        timestamp="2026-08-10T01:00:00Z",
        content_hash="h1",
        text="how do I fix the race condition",
        seen_at="2026-08-10T01:00:00Z",
    )
    base.update(kw)
    return base


def test_table_created_idempotently(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    KbStore(tmp_path / "kb.sqlite3")  # second open must not fail
    assert store.stats()["messages"] == 0


def test_upsert_dedup_same_hash_same_client(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    mid1, created1 = store.upsert_message(**_msg())
    mid2, created2 = store.upsert_message(**_msg(session_id="s2", seen_at="2026-08-10T02:00:00Z"))
    assert created1 is True
    assert created2 is False
    assert mid1 == mid2
    rows = store.indexed_messages()
    assert len(rows) == 0  # not indexed yet; check via pending
    pending = store.pending_messages(10)
    assert len(pending) == 1
    assert pending[0]["session_id"] == "s1"  # first-seen session kept


def test_upsert_dedup_updates_last_seen(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    store.upsert_message(**_msg())
    store.upsert_message(**_msg(session_id="s2", seen_at="2026-08-11T00:00:00Z"))
    row = store.pending_messages(10)[0]
    assert row["last_seen"] == "2026-08-11T00:00:00Z"


def test_same_text_different_client_not_deduped(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _, c1 = store.upsert_message(**_msg())
    _, c2 = store.upsert_message(**_msg(client="codex"))
    assert c1 is True and c2 is True
    assert len(store.pending_messages(10)) == 2


def test_index_state_machine(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    mid, _ = store.upsert_message(**_msg())
    store.mark_message_failed(mid)
    store.mark_message_failed(mid)
    assert store.requeue_failed_messages() == 1  # attempts=2 < MAX_ATTEMPTS
    store.mark_message_failed(mid)
    store.mark_message_failed(mid)
    store.mark_message_failed(mid)  # attempts hits 3
    assert store.requeue_failed_messages() == 0
    store.mark_message_indexed(mid, b"\x00" * 16)
    rows = store.indexed_messages()
    assert len(rows) == 1
    assert rows[0]["text"] == "how do I fix the race condition"
    assert rows[0]["session_id"] == "s1"


def test_reset_message_embeddings(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    mid, _ = store.upsert_message(**_msg())
    store.mark_message_indexed(mid, b"\x00" * 16)
    assert store.reset_message_embeddings() == 1
    assert len(store.indexed_messages()) == 0
    assert len(store.pending_messages(10)) == 1


def test_delete_messages_for_session(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    store.upsert_message(**_msg())
    store.upsert_message(**_msg(content_hash="h2", text="other", session_id="s2"))
    assert store.delete_messages_for_session("s1") == 1
    assert store.stats()["messages"] == 1


def test_stats_counts_messages(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    store.upsert_message(**_msg())
    stats = store.stats()
    assert stats["messages"] == 1
    assert stats["chunks"] == 0  # existing keys untouched


def test_same_text_user_and_assistant_both_stored(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _, c1 = store.upsert_message(**_msg())
    _, c2 = store.upsert_message(**_msg(role="assistant", record_index=1))
    assert c1 is True and c2 is True
    assert len(store.pending_messages(10)) == 2


def test_upsert_dedup_within_same_role(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _, c1 = store.upsert_message(**_msg(role="assistant"))
    _, c2 = store.upsert_message(**_msg(role="assistant", session_id="s2"))
    assert c1 is True and c2 is False
    assert len(store.pending_messages(10)) == 1


def test_stats_split_by_role(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    store.upsert_message(**_msg())
    store.upsert_message(**_msg(role="assistant", content_hash="h2", text="a longer assistant reply text"))
    stats = store.stats()
    assert stats["messages"] == 2
    assert stats["messages_user"] == 1
    assert stats["messages_assistant"] == 1


def test_migrate_old_db_adds_role_and_resets_backfill(tmp_path):
    """旧 schema 库（无 role 列、旧去重索引）打开后：role 迁移、索引重建、messages_done 重置。"""
    db = tmp_path / "kb.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE kb_messages (
          id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, record_index INTEGER NOT NULL,
          message_index INTEGER NOT NULL, client TEXT NOT NULL, model TEXT NOT NULL,
          timestamp TEXT NOT NULL, content_hash TEXT NOT NULL, text TEXT NOT NULL,
          last_seen TEXT NOT NULL, embedding BLOB,
          index_state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0
        );
        CREATE UNIQUE INDEX idx_kb_messages_dedup ON kb_messages(content_hash, client);
        CREATE TABLE kb_sources (
          session_id TEXT PRIMARY KEY, snapshot_id INTEGER,
          processed_at TEXT NOT NULL, messages_done INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO kb_messages (session_id, record_index, message_index, client, model,
          timestamp, content_hash, text, last_seen)
        VALUES ('s1', 0, 0, 'claude', 'k3', 't', 'h1', 'old row text', 't');
        INSERT INTO kb_sources (session_id, snapshot_id, processed_at, messages_done)
        VALUES ('s1', NULL, 't', 1);
        """
    )
    conn.close()
    store = KbStore(db)  # 触发迁移
    rows = store.pending_messages(10)
    assert rows[0]["role"] == "user"  # 存量行默认 user
    with sqlite3.connect(db) as check:
        cols = [r[1] for r in check.execute("PRAGMA table_info(kb_messages)").fetchall()]
        idx_cols = [r[2] for r in check.execute("PRAGMA index_info(idx_kb_messages_dedup)").fetchall()]
        done = check.execute("SELECT messages_done FROM kb_sources WHERE session_id='s1'").fetchone()[0]
    assert "role" in cols
    assert idx_cols == ["content_hash", "client", "role"]
    assert done == 0  # 回填被触发


def test_fts_synced_on_message_insert_and_session_delete(trace_db):
    store = KbStore.default()
    store.upsert_message(
        session_id="s1",
        record_index=0,
        message_index=0,
        client="codex",
        model="gpt-5",
        timestamp="2026-08-01T00:00:00Z",
        content_hash="m1",
        text="how to cancel a scheduled cron job",
        seen_at="t",
    )
    assert len(store.fts_rank("messages", "tri", "cron", 10)) == 1
    store.delete_messages_for_session("s1")
    assert store.fts_rank("messages", "tri", "cron", 10) == []


def test_fts_not_written_on_dedup_hit(trace_db):
    store = KbStore.default()
    kwargs = dict(
        session_id="s1",
        record_index=0,
        message_index=0,
        client="codex",
        model="gpt-5",
        timestamp="2026-08-01T00:00:00Z",
        content_hash="m1",
        text="unique phrase about reticulating splines",
        seen_at="t",
    )
    store.upsert_message(**kwargs)
    store.upsert_message(**{**kwargs, "session_id": "s2", "seen_at": "t2"})  # dedup hit
    ranked = store.fts_rank("messages", "tri", "reticulating", 10)
    assert len(ranked) == 1
