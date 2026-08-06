"""TraceStore.get_stats aggregates sessions for the dashboard stats view."""

from __future__ import annotations

from datetime import datetime, timezone

from claude_tap.trace_store import TraceStore


def _record(turn: int, *, model: str, usage: dict, status: int = 200) -> dict:
    return {
        "timestamp": f"2026-08-04T08:00:{turn:02d}+00:00",
        "turn": turn,
        "request": {"method": "POST", "path": "/v1/messages", "body": {"model": model}},
        "response": {"status": status, "body": {"usage": usage}},
    }


def _seed(store: TraceStore) -> None:
    # Session A: claude, project X, Aug 4, 2 calls on opus (one with cache)
    a = store.create_session(client="claude", cwd="/proj/x", started_at=datetime(2026, 8, 4, 8, tzinfo=timezone.utc))
    store.append_record(a, _record(1, model="opus", usage={"input_tokens": 100, "output_tokens": 50}))
    store.append_record(
        a,
        _record(
            2,
            model="opus",
            usage={"input_tokens": 200, "output_tokens": 60, "cache_read_input_tokens": 1000},
        ),
    )
    # Session B: kimi, project Y, Aug 5, 1 call on k2 with an error
    b = store.create_session(client="kimi", cwd="/proj/y", started_at=datetime(2026, 8, 5, 9, tzinfo=timezone.utc))
    store.append_record(b, _record(1, model="k2", usage={"input_tokens": 300, "output_tokens": 70}, status=500))
    # Session C: claude, no cwd, Aug 5, no records
    store.create_session(client="claude", started_at=datetime(2026, 8, 5, 10, tzinfo=timezone.utc))


def test_get_stats_totals(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    _seed(store)
    totals = store.get_stats()["totals"]
    assert totals["sessions"] == 3
    assert totals["records"] == 3
    assert totals["input_tokens"] == 600
    assert totals["output_tokens"] == 180
    assert totals["cache_read_tokens"] == 1000
    assert totals["cache_create_tokens"] == 0
    assert totals["errors"] == 1


def test_get_stats_daily_and_groupings(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    _seed(store)
    stats = store.get_stats()

    daily = {d["date"]: d for d in stats["daily"]}
    assert daily["2026-08-04"]["sessions"] == 1
    assert daily["2026-08-04"]["tokens"] == 100 + 50 + 200 + 60 + 1000
    assert daily["2026-08-05"]["sessions"] == 2
    assert daily["2026-08-05"]["tokens"] == 370

    by_client = {c["client"]: c for c in stats["by_client"]}
    assert by_client["claude"]["sessions"] == 2
    assert by_client["claude"]["tokens"] == 1410
    assert by_client["kimi"]["sessions"] == 1

    by_project = {p["cwd"]: p for p in stats["by_project"]}
    assert by_project["/proj/x"]["tokens"] == 1410
    assert by_project["/proj/y"]["tokens"] == 370
    assert by_project[""]["sessions"] == 1  # unknown project bucket

    by_model = {m["model"]: m for m in stats["by_model"]}
    assert by_model["opus"]["sessions"] == 1
    assert by_model["opus"]["tokens"] == 1410
    assert by_model["k2"]["sessions"] == 1
    assert by_model["k2"]["tokens"] == 370


def test_get_stats_filters_date_range(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    _seed(store)
    only_4th = store.get_stats(date_from="2026-08-04", date_to="2026-08-04")
    assert only_4th["totals"]["sessions"] == 1
    from_5th = store.get_stats(date_from="2026-08-05")
    assert from_5th["totals"]["sessions"] == 2
    until_4th = store.get_stats(date_to="2026-08-04")
    assert until_4th["totals"]["sessions"] == 1


def test_get_stats_empty_store_returns_zeros(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    stats = store.get_stats()
    assert stats["totals"]["sessions"] == 0
    assert stats["totals"]["input_tokens"] == 0
    assert stats["daily"] == []
    assert stats["by_client"] == []
    assert stats["by_model"] == []
    assert stats["by_project"] == []
