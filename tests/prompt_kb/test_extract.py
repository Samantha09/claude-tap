from claude_tap.prompt_kb import extract
from claude_tap.prompt_kb.extract import extract_messages, extract_session, extract_unprocessed
from claude_tap.prompt_kb.store import KbStore
from claude_tap.trace_store import get_trace_store


def _anthropic_record(turn: int = 1) -> dict:
    return {
        "timestamp": f"2026-08-01T00:00:0{turn}Z",
        "turn": turn,
        "request": {
            "method": "POST",
            "path": "/v1/messages",
            "body": {
                "model": "claude-test",
                "system": "# Rules\nbe helpful",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"name": "shell", "description": "run commands", "input_schema": {"properties": {"cmd": {}}}}
                ],
            },
        },
        "response": {"status": 200, "body": "", "sse_events": []},
    }


def test_extract_session_stores_snapshot_and_chunks(trace_db):
    store = KbStore.default()
    snap_id = extract_session(
        store,
        session_id="s1",
        client="claude-code",
        records=[_anthropic_record()],
        processed_at="2026-08-01T00:00:10Z",
    )
    assert snap_id is not None
    row = store.get_snapshot(snap_id)
    assert row["client"] == "claude-code"
    assert row["model"] == "claude-test"
    assert "Rules" in row["system_prompt"]
    assert store.stats()["pending"] == 2  # 1 prompt section + 1 tool
    assert store.is_source_processed("s1") is True


def test_extract_session_marks_empty_when_no_prompt(trace_db):
    store = KbStore.default()
    snap_id = extract_session(
        store,
        session_id="s2",
        client="codex",
        records=[{"request": {"path": "/health", "body": {}}}],
        processed_at="2026-08-01T00:00:10Z",
    )
    assert snap_id is None
    assert store.is_source_processed("s2") is True


def test_extract_session_also_stores_user_messages(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    records = [
        {
            "timestamp": "2026-08-10T01:00:00Z",
            "request": {
                "method": "POST",
                "path": "/v1/messages",
                "body": {
                    "model": "k3",
                    "system": "sys",
                    "messages": [
                        {"role": "user", "content": "fix the flaky test"},
                        {"role": "user", "content": "fix the flaky test"},  # dup -> dedup
                    ],
                },
            },
            "response": {"status": 200, "body": {}},
        }
    ]
    snap_id = extract_session(
        store,
        session_id="sess-1",
        client="claude",
        records=records,
        processed_at="2026-08-10T02:00:00Z",
    )
    assert snap_id is not None
    pending = store.pending_messages(10)
    assert len(pending) == 1  # duplicate text deduped
    row = store.indexed_messages()  # not indexed yet
    assert row == []


def test_extract_messages_model_from_body(tmp_path):
    import sqlite3

    store = KbStore(tmp_path / "kb.sqlite3")
    records = [
        {
            "timestamp": "2026-08-10T01:00:00Z",
            "request": {
                "method": "POST",
                "path": "/v1/messages",
                "body": {"model": "k3-256k", "messages": [{"role": "user", "content": "hello"}]},
            },
            "response": {"status": 200},
        }
    ]
    created = extract_messages(store, session_id="s1", client="claude", records=records)
    assert created["user"] == 1
    with sqlite3.connect(tmp_path / "kb.sqlite3") as conn:
        row = conn.execute("SELECT model, client, session_id FROM kb_messages").fetchone()
    assert row == ("k3-256k", "claude", "s1")


def test_extract_messages_stores_assistant_replies(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    records = [
        {
            "timestamp": "2026-08-10T01:00:00Z",
            "request": {
                "method": "POST",
                "path": "/v1/messages",
                "body": {
                    "model": "k3-256k",
                    "messages": [{"role": "user", "content": "how do I fix the race condition"}],
                },
            },
            "response": {
                "status": 200,
                "body": {
                    "content": [
                        {"type": "thinking", "thinking": "reasoning here"},
                        {"type": "text", "text": "use a lock ordering protocol to fix the race condition"},
                    ]
                },
            },
        }
    ]
    created = extract_messages(store, session_id="s1", client="claude", records=records)
    assert created == {"user": 1, "assistant": 1}
    rows = {(row["role"], row["text"]) for row in store.pending_messages(10)}
    assert ("user", "how do I fix the race condition") in rows
    assert ("assistant", "use a lock ordering protocol to fix the race condition") in rows
    # 重抽幂等：去重键 (hash, client, role) 挡住重复
    assert extract_messages(store, session_id="s1", client="claude", records=records) == {"user": 0, "assistant": 0}


def test_extract_unprocessed_walks_trace_store(trace_db):
    trace = get_trace_store()
    session_id = trace.create_session(client="claude-code", proxy_mode="reverse")
    trace.append_record(session_id, _anthropic_record())
    store = KbStore.default()
    result = extract_unprocessed(store, trace)
    assert result == {"processed": 1, "snapshots": 1, "skipped": 0, "messages_backfilled": 0}
    assert extract_unprocessed(store, trace) == {"processed": 0, "snapshots": 0, "skipped": 0, "messages_backfilled": 0}


def test_extract_unprocessed_continues_after_failure(trace_db, monkeypatch):
    trace = get_trace_store()
    good = trace.create_session(client="claude-code", proxy_mode="reverse")
    trace.append_record(good, _anthropic_record())
    bad = trace.create_session(client="claude-code", proxy_mode="reverse")
    trace.append_record(bad, _anthropic_record(turn=2))
    store = KbStore.default()

    real_extract_session = extract.extract_session

    def flaky_extract_session(store, *, session_id, **kwargs):
        if session_id == bad:
            raise RuntimeError("boom")
        return real_extract_session(store, session_id=session_id, **kwargs)

    monkeypatch.setattr(extract, "extract_session", flaky_extract_session)
    result = extract.extract_unprocessed(store, trace)
    assert result == {"processed": 1, "snapshots": 1, "skipped": 1, "messages_backfilled": 0}
    # The failed session is not recorded, so it is retried on the next pass.
    assert store.is_source_processed(bad) is False
    assert store.is_source_processed(good) is True


def test_extract_unprocessed_marks_empty_session_messages_done(trace_db):
    trace = get_trace_store()
    trace.create_session(client="claude-code", proxy_mode="reverse")  # no records
    store = KbStore.default()
    result = extract_unprocessed(store, trace)
    assert result["skipped"] == 1
    assert store.sources_missing_messages() == []


def test_extract_unprocessed_backfills_messages_for_legacy_sources(trace_db):
    trace = get_trace_store()
    session_id = trace.create_session(client="claude-code", proxy_mode="reverse")
    trace.append_record(session_id, _anthropic_record())
    store = KbStore.default()
    # Simulate the pre-messages build: session processed, messages never extracted.
    store.record_source(session_id, None, "2026-08-09T00:00:00Z")
    result = extract_unprocessed(store, trace)
    assert result == {"processed": 0, "snapshots": 0, "skipped": 0, "messages_backfilled": 1}
    assert [row["text"] for row in store.pending_messages(10)] == ["hi"]
    assert store.sources_missing_messages() == []


def test_extract_unprocessed_backfill_marks_deleted_session_done(trace_db):
    store = KbStore.default()
    store.record_source("gone-session", None, "2026-08-09T00:00:00Z")
    result = extract_unprocessed(store, get_trace_store())
    assert result["messages_backfilled"] == 0
    assert store.sources_missing_messages() == []


def test_extract_unprocessed_backfill_keeps_failed_session_retriable(trace_db, monkeypatch):
    trace = get_trace_store()
    session_id = trace.create_session(client="claude-code", proxy_mode="reverse")
    trace.append_record(session_id, _anthropic_record())
    store = KbStore.default()
    store.record_source(session_id, None, "2026-08-09T00:00:00Z")

    def boom(store, *, session_id, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(extract, "extract_messages", boom)
    result = extract.extract_unprocessed(store, trace)
    assert result["messages_backfilled"] == 0
    # Not marked done: the next pass retries the backfill.
    assert store.sources_missing_messages() == [session_id]


def test_extract_session_extracts_messages_even_without_snapshot(tmp_path, monkeypatch):
    store = KbStore(tmp_path / "kb.sqlite3")
    records = [_anthropic_record()]

    def no_snapshot(records):
        raise ValueError("no prompt-bearing request found in trace")

    monkeypatch.setattr(extract, "snapshot_from_records", no_snapshot)
    snap_id = extract_session(
        store,
        session_id="s-ns",
        client="claude",
        records=records,
        processed_at="2026-08-10T00:00:00Z",
    )
    assert snap_id is None
    assert [row["text"] for row in store.pending_messages(10)] == ["hi"]
    assert store.is_source_processed("s-ns") is True
    assert store.sources_missing_messages() == []
