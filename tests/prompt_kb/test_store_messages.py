"""kb_messages storage: migration, dedup upsert, index state machine."""

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
