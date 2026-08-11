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
    assert store.stats() == {"snapshots": 1, "chunks": 2, "pending": 0, "failed": 1, "indexed": 1, "messages": 0}
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
