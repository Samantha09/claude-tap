import sqlite3

from claude_tap.prompt_kb.store import KbStore


def _upsert(store, *, content_hash="h1", client="codex", model="gpt-5", seen_at="2026-08-01T00:00:00Z"):
    return store.upsert_snapshot(
        content_hash=content_hash,
        client=client,
        provider="openai",
        model=model,
        system_prompt="sys",
        developer_prompt="",
        tools_json="[]",
        seen_at=seen_at,
    )


def test_upsert_dedups_by_client_model_hash(trace_db):
    store = KbStore.default()
    first_id, created = _upsert(store)
    assert created is True
    again_id, created = _upsert(store, seen_at="2026-08-02T00:00:00Z")
    assert created is False
    assert again_id == first_id
    row = store.get_snapshot(first_id)
    assert row["session_count"] == 2
    assert row["last_seen"] == "2026-08-02T00:00:00Z"
    assert row["first_seen"] == "2026-08-01T00:00:00Z"


def test_timeline_orders_by_first_seen(trace_db):
    store = KbStore.default()
    _upsert(store, content_hash="h2", seen_at="2026-08-02T00:00:00Z")
    _upsert(store, content_hash="h1", seen_at="2026-08-01T00:00:00Z")
    versions = store.timeline("codex", "gpt-5")
    assert [v["content_hash"] for v in versions] == ["h1", "h2"]


def test_chunks_lifecycle(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("tool", "shell", "shell tool desc"), ("prompt_section", "Rules", "rule text")])
    pending = store.pending_chunks(10)
    assert len(pending) == 2
    store.mark_chunk_indexed(pending[0]["id"], b"\x00" * 64)
    store.mark_chunk_failed(pending[1]["id"])
    assert store.stats() == {
        "snapshots": 1,
        "chunks": 2,
        "pending": 0,
        "failed": 1,
        "indexed": 1,
        "messages": 0,
        "messages_user": 0,
        "messages_assistant": 0,
    }
    assert len(store.indexed_chunks()) == 1
    assert store.reset_embeddings() == 2
    assert store.stats()["pending"] == 2


def test_sources_mark_processed(trace_db):
    store = KbStore.default()
    assert store.is_source_processed("s1") is False
    store.record_source("s1", None, "2026-08-06T00:00:00Z")
    assert store.is_source_processed("s1") is True


def test_migrate_adds_messages_done_to_legacy_db(tmp_path):
    db = tmp_path / "kb.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE kb_sources (session_id TEXT PRIMARY KEY, snapshot_id INTEGER, processed_at TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO kb_sources VALUES ('legacy-1', NULL, '2026-08-01T00:00:00Z')")
    store = KbStore(db)
    # Legacy rows default to messages_done=0 (not yet backfilled).
    assert store.sources_missing_messages() == ["legacy-1"]
    # Reopening the migrated database is a no-op (idempotent migration).
    KbStore(db)
    assert store.sources_missing_messages() == ["legacy-1"]


def test_sources_missing_messages_and_mark_done(trace_db):
    store = KbStore.default()
    store.record_source("s1", None, "2026-08-06T00:00:00Z")
    store.record_source("s2", None, "2026-08-06T00:00:00Z", messages_done=True)
    assert store.sources_missing_messages() == ["s1"]
    store.mark_source_messages_done("s1")
    assert store.sources_missing_messages() == []


def test_sources_missing_messages_respects_limit(trace_db):
    store = KbStore.default()
    for i in range(5):
        store.record_source(f"s{i}", None, "2026-08-06T00:00:00Z")
    assert len(store.sources_missing_messages(limit=3)) == 3


def test_meta_roundtrip(trace_db):
    store = KbStore.default()
    assert store.get_meta("embedder_name") is None
    store.set_meta("embedder_name", "fake")
    assert store.get_meta("embedder_name") == "fake"


def test_connect_sets_busy_timeout(trace_db):
    """Connections must wait for a concurrent writer instead of failing reads instantly."""
    store = KbStore.default()
    conn = store._connect()
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 2000
    finally:
        conn.close()


def test_boilerplate_chunks_purged_once(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    sid, _ = store.upsert_snapshot(
        content_hash="h1", client="claude", provider="anthropic", model="k3",
        system_prompt="s", developer_prompt="", tools_json="[]", seen_at="t",
    )
    store.replace_chunks(sid, [("prompt_section", "Environment", "boilerplate"),
                               ("prompt_section", "Style", "real content")])
    store.set_meta("boilerplate_purged", "0")  # 模拟未清除状态
    KbStore(db)  # 重开触发一次性清除（标记非 "1" 即执行）
    rows = store.indexed_chunks()
    assert [r["title"] for r in rows] == []  # 未索引不在 indexed_chunks
    with sqlite3.connect(db) as conn:
        titles = [r[0] for r in conn.execute("SELECT title FROM kb_chunks").fetchall()]
        purged = conn.execute("SELECT value FROM kb_meta WHERE key='boilerplate_purged'").fetchone()[0]
    assert titles == ["Style"]
    assert purged == "1"


def test_fts_synced_on_replace_chunks(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("tool", "shell", "CronDelete cancels scheduled cron jobs")])
    ranked = store.fts_rank("chunks", "tri", "CronDelete", 10)
    assert len(ranked) == 1 and ranked[0][0] == 1
    # jieba table got the segmented copy of the same text.
    ranked_jieba = store.fts_rank("chunks", "jieba", "CronDelete", 10)
    assert len(ranked_jieba) == 1 and ranked_jieba[0][0] == 1


def test_fts_chinese_word_match_via_jieba_channel(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("prompt_section", "指南", "取消定时任务的正确方法")])
    # 2-char Chinese word: trigram cannot match it, the jieba channel can.
    assert store.fts_rank("chunks", "tri", "定时", 10) == []
    ranked = store.fts_rank("chunks", "jieba", "定时", 10)
    assert len(ranked) == 1


def test_fts_updated_on_replace_chunks(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("tool", "shell", "alpha bravo charlie")])
    store.replace_chunks(snap_id, [("tool", "shell", "delta echo foxtrot")])
    assert store.fts_rank("chunks", "tri", "alpha", 10) == []
    assert len(store.fts_rank("chunks", "tri", "delta", 10)) == 1


def test_fts_rank_rejects_unknown_channel(trace_db):
    store = KbStore.default()
    import pytest

    with pytest.raises(ValueError):
        store.fts_rank("chunks", "bogus", "x", 10)
    with pytest.raises(ValueError):
        store.fts_rank("bogus", "tri", "x", 10)


def _build_legacy_db_without_fts(path):
    """A pre-hybrid-schema DB: main tables with data, no FTS tables, no meta flag."""
    import sqlite3 as _sq

    conn = _sq.connect(path)
    conn.execute(
        "CREATE TABLE kb_chunks (id INTEGER PRIMARY KEY, snapshot_id INTEGER NOT NULL,"
        " kind TEXT NOT NULL, title TEXT, text TEXT NOT NULL, embedding BLOB,"
        " index_state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute("INSERT INTO kb_chunks (snapshot_id, kind, title, text) VALUES (1, 'tool', 'shell', 'legacy cron tooling')")
    conn.execute(
        "CREATE TABLE kb_messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,"
        " record_index INTEGER NOT NULL, message_index INTEGER NOT NULL, client TEXT NOT NULL,"
        " model TEXT NOT NULL, timestamp TEXT NOT NULL, content_hash TEXT NOT NULL, text TEXT NOT NULL,"
        " last_seen TEXT NOT NULL, embedding BLOB, index_state TEXT NOT NULL DEFAULT 'pending',"
        " attempts INTEGER NOT NULL DEFAULT 0, role TEXT NOT NULL DEFAULT 'user')"
    )
    conn.execute(
        "INSERT INTO kb_messages (session_id, record_index, message_index, client, model,"
        " timestamp, content_hash, text, last_seen) VALUES ('s1', 0, 0, 'c', 'm', 't', 'h', 'legacy message text', 't')"
    )
    conn.execute("CREATE TABLE kb_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()


def test_fts_backfilled_on_open_for_legacy_db(trace_db, tmp_path):
    path = tmp_path / "legacy_kb.sqlite3"
    _build_legacy_db_without_fts(path)
    store = KbStore(path)  # migration runs on open
    assert len(store.fts_rank("chunks", "tri", "legacy", 10)) == 1
    assert len(store.fts_rank("messages", "tri", "legacy", 10)) == 1
    assert store.get_meta("fts_backfilled") == "1"
    store2 = KbStore(path)  # second open: no duplicate FTS rows
    assert len(store2.fts_rank("chunks", "tri", "legacy", 10)) == 1


def test_rebuild_fts_clears_and_reindexes(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("tool", "shell", "rebuild me please")])
    assert store.rebuild_fts() == 1
    assert len(store.fts_rank("chunks", "tri", "rebuild", 10)) == 1
    assert store.rebuild_fts() == 1  # idempotent, no duplicates
    assert len(store.fts_rank("chunks", "tri", "rebuild", 10)) == 1
