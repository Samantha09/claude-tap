"""Timeline query methods on KbStore (back kb_recent)."""

from claude_tap.prompt_kb.store import KbStore


def _msg(store, *, session, ts, text, role="user", hash_, record=0, msg_idx=0, client="claude-code"):
    return store.upsert_message(
        session_id=session,
        record_index=record,
        message_index=msg_idx,
        client=client,
        model="m",
        timestamp=ts,
        content_hash=hash_,
        text=text,
        role=role,
        seen_at=ts,
    )


def test_recent_sessions_ordered_by_last_activity(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _msg(store, session="old", ts="2026-08-01T10:00:00Z", text="a", hash_="h1")
    _msg(store, session="new", ts="2026-08-03T10:00:00Z", text="b", hash_="h2")
    _msg(store, session="old", ts="2026-08-02T10:00:00Z", text="c", hash_="h3")
    rows = store.recent_sessions(client=None, limit=10)
    assert [r["session_id"] for r in rows] == ["new", "old"]
    old = rows[1]
    assert old["first_ts"] == "2026-08-01T10:00:00Z"
    assert old["last_ts"] == "2026-08-02T10:00:00Z"
    assert old["client"] == "claude-code"


def test_recent_sessions_client_filter_and_limit(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _msg(store, session="s1", ts="2026-08-01T10:00:00Z", text="a", hash_="h1", client="codex")
    _msg(store, session="s2", ts="2026-08-02T10:00:00Z", text="b", hash_="h2")
    rows = store.recent_sessions(client="codex", limit=10)
    assert [r["session_id"] for r in rows] == ["s1"]
    assert len(store.recent_sessions(client=None, limit=1)) == 1


def test_first_user_message_skips_assistant_and_picks_earliest(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _msg(store, session="s", ts="2026-08-01T10:01:00Z", text="reply", hash_="h1", role="assistant")
    _msg(store, session="s", ts="2026-08-01T10:00:00Z", text="original task", hash_="h2")
    row = store.session_first_user_message("s")
    assert row is not None
    assert row["text"] == "original task"
    assert row["seen_at"] == "2026-08-01T10:00:00Z"
    assert store.session_first_user_message("nonexistent") is None


def test_last_messages_chronological_and_capped(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    for i in range(5):
        _msg(
            store,
            session="s",
            ts=f"2026-08-01T10:0{i}:00Z",
            text=f"m{i}",
            hash_=f"h{i}",
            role="user" if i % 2 == 0 else "assistant",
            msg_idx=i,
        )
    rows = store.session_last_messages("s", 2)
    assert [r["text"] for r in rows] == ["m3", "m4"]  # ascending seen_at
    assert rows[0]["role"] == "assistant"


def test_shared_message_appears_in_both_sessions(tmp_path):
    """Deduped message (same content_hash) belongs to both sessions via occurrences."""
    store = KbStore(tmp_path / "kb.sqlite3")
    _msg(store, session="s1", ts="2026-08-01T10:00:00Z", text="shared", hash_="same")
    _msg(store, session="s2", ts="2026-08-02T10:00:00Z", text="shared", hash_="same")
    assert store.session_first_user_message("s2")["text"] == "shared"
    assert store.session_first_user_message("s2")["seen_at"] == "2026-08-02T10:00:00Z"
    assert [r["session_id"] for r in store.recent_sessions(client=None, limit=10)] == ["s2", "s1"]
