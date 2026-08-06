# Dashboard Stats View — Design Spec

## Overview

Add a statistics view to the existing shared dashboard (port 19527, `LiveViewerServer` dashboard mode). A view toggle on the session-list page switches between "会话列表" and "统计". The stats view aggregates all trace sessions in the local SQLite store — no new services, no new dependencies.

Goal: answer "全部统计" in one screen — token consumption, time trends, client/model distribution, project distribution.

Non-goals: RAG / semantic search / LLM-generated insights (explicitly deferred by user); per-session detail changes.

## Data Layer

### Schema v6: `sessions.cwd` + notes-era convergence

- Upstream schema is v4. Add `cwd TEXT NOT NULL DEFAULT ''` to the `sessions` CREATE TABLE statement and provide an idempotent v4→v5 migration (`PRAGMA table_info` guard before `ALTER TABLE`, hardcoded `PRAGMA user_version = 5`).
- v5→v6 is a stamp-only migration: the abandoned notes app already stamped the production database `user_version = 6` (cwd + orphaned tables `tags`/`notes`/`summaries`/FTS). Stamping 6 lets notes-era, fresh, and v4-upgraded databases converge on one accepted version; the orphaned tables are left untouched and unused.
- `create_session(...)` gains `cwd: str = ""`; `create_trace_writer` captures `os.getcwd()` by default (explicit param wins).
- Legacy rows keep `cwd = ''` and aggregate into an "未知项目" bucket.

## API

`GET /api/stats?from=YYYY-MM-DD&to=YYYY-MM-DD` on the dashboard server. Both params optional (default: all time). Reuse the existing date validation style: invalid format or inverted range → 400 with a Chinese error message.

Response:

```json
{
  "totals": {
    "sessions": 42, "records": 380,
    "input_tokens": 1200000, "output_tokens": 300000,
    "cache_read_tokens": 800000, "cache_create_tokens": 90000,
    "errors": 2
  },
  "daily": [
    { "date": "2026-08-04", "sessions": 5, "tokens": 150000 }
  ],
  "by_client": [
    { "client": "claude", "sessions": 30, "tokens": 900000 }
  ],
  "by_model": [
    { "model": "claude-opus-4-8", "sessions": 25, "tokens": 700000 }
  ],
  "by_project": [
    { "cwd": "/usr/local/data/apps/claude-tap-main", "sessions": 20, "tokens": 500000 }
  ]
}
```

- Token totals come from `summary_json` fields (`input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_create_tokens`); total tokens for breakdowns = sum of the four token fields. `records` is the summed `record_count`; `errors` counts sessions with `status = 'error'` or a summary error flag.
- `summary_json` exists in two formats (live-writer final summary with `models_used` dict, and dashboard incremental summary with a single top `model`); `by_model` accepts both and aggregates per-session top model as `{model, sessions, tokens}`. Sessions with no/unknown model are skipped.
- `daily` uses `date_key` (local-time session date).
- `by_project` groups by `cwd`; empty cwd renders as "未知项目" in the UI.
- Empty database returns zeros and empty arrays (no error).

Implementation: `TraceStore.get_stats(date_from: str = "", date_to: str = "") -> dict` doing one pass over `sessions`; handler in `live.py` mirrors the existing `_handle_*` style.

## UI

In `claude_tap/dashboard.html` (vanilla JS, zero dependencies — matches existing style):

- View toggle `会话列表 | 统计` in the dashboard title row.
- Stats view sections:
  1. Overview cards: sessions, records, errors, total tokens with input/output/cache-read/cache-create breakdown.
  2. Daily trend bar chart (CSS bars, hover tooltip with exact numbers).
  3. Client distribution (bar + percentage).
  4. Model distribution (bar + percentage).
  5. Project distribution (bar + percentage; empty cwd shown as 未知项目).
- Pure CSS bars (div widths in %), no chart library.
- Bilingual labels via the existing `data-i18n` mechanism (en + zh-CN entries).
- Date-range selector reuses the existing date filter options.

## Error Handling

- Stats endpoint failure surfaces as the existing dashboard error style; empty/zero data renders "暂无数据" placeholders, not broken charts.
- Migration is idempotent and leaves legacy rows untouched.

## Testing

- Migration: v4 DB upgrades through v5 to v6; legacy rows keep `cwd=''`; notes-era v6 databases (with orphaned personal tables) are accepted unchanged; new sessions record cwd; `create_trace_writer` defaults to process cwd, honors explicit override.
- `get_stats`: aggregation correctness (token breakdown, per-session top model, grouping by client/cwd/date), date-range filtering (both sides, single side), empty DB → zeros.
- HTTP: `/api/stats` 200 shape; 400 on invalid/inverted dates (Chinese messages).
- UI: existing dashboard test pattern (aiohttp server + fetch HTML) asserts the view toggle and stats container exist.

## Commit Plan

One concern per commit, Chinese messages: (1) schema v6 (cwd + notes-era convergence); (2) stats store + API; (3) stats UI. User reviews before committing (standing instruction: do not commit without confirmation).
