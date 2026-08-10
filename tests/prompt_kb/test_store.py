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


def test_meta_roundtrip(trace_db):
    store = KbStore.default()
    assert store.get_meta("embedder_name") is None
    store.set_meta("embedder_name", "fake")
    assert store.get_meta("embedder_name") == "fake"
