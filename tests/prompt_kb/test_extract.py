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
        store, session_id="sess-1", client="claude",
        records=records, processed_at="2026-08-10T02:00:00Z",
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
                "method": "POST", "path": "/v1/messages",
                "body": {"model": "k3-256k",
                         "messages": [{"role": "user", "content": "hello"}]},
            },
            "response": {"status": 200},
        }
    ]
    created = extract_messages(store, session_id="s1", client="claude", records=records)
    assert created == 1
    with sqlite3.connect(tmp_path / "kb.sqlite3") as conn:
        row = conn.execute("SELECT model, client, session_id FROM kb_messages").fetchone()
    assert row == ("k3-256k", "claude", "s1")


def test_extract_unprocessed_walks_trace_store(trace_db):
    trace = get_trace_store()
    session_id = trace.create_session(client="claude-code", proxy_mode="reverse")
    trace.append_record(session_id, _anthropic_record())
    store = KbStore.default()
    result = extract_unprocessed(store, trace)
    assert result == {"processed": 1, "snapshots": 1, "skipped": 0}
    assert extract_unprocessed(store, trace) == {"processed": 0, "snapshots": 0, "skipped": 0}


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
    assert result == {"processed": 1, "snapshots": 1, "skipped": 1}
    # The failed session is not recorded, so it is retried on the next pass.
    assert store.is_source_processed(bad) is False
    assert store.is_source_processed(good) is True
