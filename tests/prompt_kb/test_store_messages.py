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


def _occurrences(db, message_id=None):
    sql = "SELECT message_id, session_id, seen_at FROM kb_message_occurrences"
    params = ()
    if message_id is not None:
        sql += " WHERE message_id=?"
        params = (message_id,)
    with sqlite3.connect(db) as conn:
        return {(r[0], r[1]): r[2] for r in conn.execute(sql, params).fetchall()}


def test_upsert_creates_occurrence(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    mid, _ = store.upsert_message(**_msg(seen_at="2026-08-10T01:00:00Z"))
    assert _occurrences(db, mid) == {(mid, "s1"): "2026-08-10T01:00:00Z"}


def test_dedup_hit_adds_occurrence_for_second_session(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    mid, _ = store.upsert_message(**_msg())
    store.upsert_message(**_msg(session_id="s2", seen_at="2026-08-11T00:00:00Z"))
    assert _occurrences(db, mid) == {
        (mid, "s1"): "2026-08-10T01:00:00Z",
        (mid, "s2"): "2026-08-11T00:00:00Z",
    }


def test_delete_session_keeps_shared_message_and_reassigns_representative(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    mid, _ = store.upsert_message(**_msg(seen_at="2026-08-10T01:00:00Z"))
    store.upsert_message(**_msg(session_id="s2", seen_at="2026-08-11T00:00:00Z"))
    store.upsert_message(**_msg(session_id="s3", seen_at="2026-08-09T00:00:00Z"))  # earliest surviving
    assert store.delete_messages_for_session("s1") == 0  # no message rows deleted
    assert store.stats()["messages"] == 1  # shared row survives
    assert _occurrences(db, mid) == {
        (mid, "s2"): "2026-08-11T00:00:00Z",
        (mid, "s3"): "2026-08-09T00:00:00Z",
    }
    row = store.pending_messages(10)[0]
    assert row["session_id"] == "s3"  # representative reassigned to earliest surviving occurrence


def test_delete_last_occurrence_removes_message_and_fts(trace_db, tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    mid, _ = store.upsert_message(**_msg(text="unique phrase about zebra crossings"))
    store.upsert_message(**_msg(session_id="s2", text="unique phrase about zebra crossings", seen_at="t2"))
    store.delete_messages_for_session("s1")
    assert store.stats()["messages"] == 1
    store.delete_messages_for_session("s2")  # last occurrence
    assert store.stats()["messages"] == 0
    assert store.fts_rank("messages", "tri", "zebra", 10) == []
    assert _occurrences(db, mid) == {}


def test_migrate_backfills_occurrences_for_existing_rows(tmp_path):
    """存量库（有 kb_messages、无 occurrences 表）打开后回填，且幂等。"""
    db = tmp_path / "kb.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE kb_messages (
          id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, record_index INTEGER NOT NULL,
          message_index INTEGER NOT NULL, client TEXT NOT NULL, model TEXT NOT NULL,
          timestamp TEXT NOT NULL, content_hash TEXT NOT NULL, text TEXT NOT NULL,
          last_seen TEXT NOT NULL, embedding BLOB,
          index_state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          role TEXT NOT NULL DEFAULT 'user'
        );
        CREATE TABLE kb_sources (
          session_id TEXT PRIMARY KEY, snapshot_id INTEGER,
          processed_at TEXT NOT NULL, messages_done INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO kb_messages (session_id, record_index, message_index, client, model,
          timestamp, content_hash, text, last_seen, role)
        VALUES ('s1', 0, 0, 'claude', 'k3', 't', 'h1', 'old row text', '2026-08-01T00:00:00Z', 'user');
        """
    )
    conn.close()
    KbStore(db)
    occ = _occurrences(db)
    assert len(occ) == 1
    assert list(occ.values()) == ["2026-08-01T00:00:00Z"]  # seen_at seeded from last_seen
    KbStore(db)  # second open: no duplicates, no failure
    assert len(_occurrences(db)) == 1


def test_migrate_resets_messages_done_to_heal_lost_content(tmp_path):
    """老库升级时把所有已处理会话重置为 messages_done=0，让 lazy 循环重新
    提取、补回因首现会话删除而丢失的内容（并补全多会话 occurrences）。"""
    db = tmp_path / "kb.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE kb_messages (
          id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, record_index INTEGER NOT NULL,
          message_index INTEGER NOT NULL, client TEXT NOT NULL, model TEXT NOT NULL,
          timestamp TEXT NOT NULL, content_hash TEXT NOT NULL, text TEXT NOT NULL,
          last_seen TEXT NOT NULL, embedding BLOB,
          index_state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          role TEXT NOT NULL DEFAULT 'user'
        );
        CREATE TABLE kb_sources (
          session_id TEXT PRIMARY KEY, snapshot_id INTEGER,
          processed_at TEXT NOT NULL, messages_done INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO kb_sources (session_id, snapshot_id, processed_at, messages_done)
        VALUES ('s1', NULL, 't', 1), ('s2', NULL, 't', 1);
        """
    )
    conn.close()
    KbStore(db)
    with sqlite3.connect(db) as check:
        rows = check.execute("SELECT session_id, messages_done FROM kb_sources ORDER BY session_id").fetchall()
    assert rows == [("s1", 0), ("s2", 0)]
    KbStore(db)  # idempotent: second open does not reset again after reprocessing
    with sqlite3.connect(db) as check:
        check.execute("UPDATE kb_sources SET messages_done = 1")
    KbStore(db)
    with sqlite3.connect(db) as check:
        assert check.execute("SELECT MIN(messages_done) FROM kb_sources").fetchone()[0] == 1


def _tombstones(db):
    with sqlite3.connect(db) as conn:
        return set(conn.execute("SELECT content_hash, client, role FROM kb_purged").fetchall())


def test_purge_removes_row_occurrences_fts_and_writes_tombstone(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    mid, _ = store.upsert_message(**_msg(text="secret phrase about purple bananas"))
    store.upsert_message(**_msg(session_id="s2", text="secret phrase about purple bananas", seen_at="t2"))
    assert store.purge_message("h1", "claude", "user") == 1
    assert store.stats()["messages"] == 0
    assert _occurrences(db, mid) == {}
    assert store.fts_rank("messages", "tri", "purple", 10) == []
    assert _tombstones(db) == {("h1", "claude", "user")}


def test_purged_content_is_never_reupserted(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    store.upsert_message(**_msg())
    store.purge_message("h1", "claude", "user")
    mid, created = store.upsert_message(**_msg(session_id="s9"))
    assert (mid, created) == (0, False)
    assert store.stats()["messages"] == 0
    assert _occurrences(db) == {}


def test_purge_undo_removes_tombstone_and_allows_reupsert(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    store.upsert_message(**_msg())
    store.purge_message("h1", "claude", "user")
    assert store.unpurge_message("h1", "claude", "user") == 1
    assert _tombstones(db) == set()
    mid, created = store.upsert_message(**_msg(session_id="s9"))
    assert created is True and mid > 0


def test_purge_nonexistent_writes_tombstone_but_deletes_nothing(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    assert store.purge_message("nope", "claude", "user") == 0  # no row to delete
    assert _tombstones(db) == {("nope", "claude", "user")}  # preemptive tombstone
    assert store.unpurge_message("nope", "claude", "user") == 1  # tombstone removed
    assert store.unpurge_message("nope", "claude", "user") == 0  # already gone


def test_purge_content_purges_every_client_role_variant(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    store.upsert_message(**_msg())
    store.upsert_message(**_msg(client="codex"))
    store.upsert_message(**_msg(role="assistant"))
    assert store.purge_content("h1") == 3
    assert store.stats()["messages"] == 0
    assert _tombstones(db) == {
        ("h1", "claude", "user"),
        ("h1", "codex", "user"),
        ("h1", "claude", "assistant"),
    }


def test_purge_content_with_filters_scopes_variants(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    store.upsert_message(**_msg())
    store.upsert_message(**_msg(client="codex"))
    assert store.purge_content("h1", client="codex") == 1
    assert store.stats()["messages"] == 1
    assert _tombstones(db) == {("h1", "codex", "user")}


def test_purge_content_preemptive_tombstone_needs_full_key(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    assert store.purge_content("future-hash") == 0
    assert _tombstones(db) == set()  # no row, no full key → nothing written
    assert store.purge_content("future-hash", client="claude", role="user") == 0
    assert _tombstones(db) == {("future-hash", "claude", "user")}  # preemptive


def test_unpurge_content_removes_matching_tombstones(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    store.upsert_message(**_msg())
    store.upsert_message(**_msg(client="codex"))
    store.purge_content("h1")
    assert store.unpurge_content("h1", client="codex") == 1
    assert _tombstones(db) == {("h1", "claude", "user")}
    assert store.unpurge_content("h1") == 1
    assert _tombstones(db) == set()
