# KB 记忆工具（kb_recall + kb_recent）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 claude-tap-kb MCP server 新增两个只读记忆工具：`kb_recall`（相似度召回历史消息，带出处）与 `kb_recent`（纯 SQL 近期会话时间线）。

**Architecture:** 新模块 `claude_tap/prompt_kb/recall.py` 承载两工具的核心逻辑；`KbStore` 新增 3 个时间线查询方法；`mcp_server.py` 只加薄包装与注册。`search.py` 零改动（`kb_recall` 复用 `search_messages`）；proxy / extract 管线不动，KB 只读。

**Tech Stack:** Python 3.11+, sqlite3, numpy（`[rag]` extra）, mcp 1.x/2.x 双版本兼容, pytest。

**Spec:** `docs/superpowers/specs/2026-08-18-kb-memory-tools-design.md`

## Global Constraints

- Commit message 一律中文（含 type/scope 前缀）；代码、注释、docstring 仅英文。
- 每次 commit 前跑 gate：`uv run ruff check .`、`uv run ruff format --check .`、`uv run pytest tests/ -x --timeout=60`。
- 每个 commit 只处理一个关注点。
- MCP 工具永不抛栈：所有异常进返回值 `{"error": ...}`。
- 时间戳一律 ISO 格式字符串（`2026-08-01T00:00:00Z`），与现有 `kb_messages.timestamp` 一致。
- 公共文档必须中英双份（`README.md` + `README_zh.md`）。

---

### Task 1: KbStore 时间线查询方法

**Files:**
- Modify: `claude_tap/prompt_kb/store.py`（在 `upsert_message` 之后、`pending_messages` 之前插入）
- Test: `tests/prompt_kb/test_store_timeline.py`（新建）

**Interfaces:**
- Consumes: 现有 schema `kb_messages` ⋈ `kb_message_occurrences`；`KbStore.upsert_message(...)`（测试播种用，签名见 store.py:317）
- Produces（后续任务依赖这三个方法，签名必须完全一致）:
  - `KbStore.recent_sessions(client: str | None, limit: int) -> list[sqlite3.Row]` — 每行含 `session_id, client, first_ts, last_ts`，按 `last_ts` 倒序
  - `KbStore.session_first_user_message(session_id: str) -> sqlite3.Row | None` — 行含 `text, seen_at, client`
  - `KbStore.session_last_messages(session_id: str, n: int) -> list[sqlite3.Row]` — 每行含 `role, text, seen_at`，按 `seen_at` 升序（时间正序）

注意：`kb_messages.session_id` 是首现会话；跨会话共享行经 occurrences 归属多个会话，所以时间线查询**必须走 occurrences join**，按 `o.seen_at` 排序（该消息在**这个**会话出现的时间），不能用 `kb_messages.record_index`（那是首现会话的序号）。

- [ ] **Step 1: 写失败测试**

创建 `tests/prompt_kb/test_store_timeline.py`：

```python
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
        _msg(store, session="s", ts=f"2026-08-01T10:0{i}:00Z", text=f"m{i}", hash_=f"h{i}",
             role="user" if i % 2 == 0 else "assistant", msg_idx=i)
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_store_timeline.py -v`
Expected: FAIL — `AttributeError: 'KbStore' object has no attribute 'recent_sessions'`

- [ ] **Step 3: 实现三个查询方法**

在 `claude_tap/prompt_kb/store.py` 的 `upsert_message` 方法之后插入：

```python
    def recent_sessions(self, client: str | None, limit: int) -> list[sqlite3.Row]:
        """Sessions ordered by last activity (MAX occurrence seen_at desc)."""
        sql = """
            SELECT o.session_id AS session_id,
                   MAX(m.client) AS client,
                   MIN(o.seen_at) AS first_ts,
                   MAX(o.seen_at) AS last_ts
            FROM kb_message_occurrences o
            JOIN kb_messages m ON m.id = o.message_id
        """
        params: list[object] = []
        if client:
            sql += " WHERE m.client = ?"
            params.append(client)
        sql += " GROUP BY o.session_id ORDER BY last_ts DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def session_first_user_message(self, session_id: str) -> sqlite3.Row | None:
        """Earliest role='user' occurrence in the session (the session's task opener)."""
        with self._connect() as conn:
            return conn.execute(
                """SELECT m.text AS text, o.seen_at AS seen_at, m.client AS client
                   FROM kb_message_occurrences o
                   JOIN kb_messages m ON m.id = o.message_id
                   WHERE o.session_id = ? AND m.role = 'user'
                   ORDER BY o.seen_at ASC LIMIT 1""",
                (session_id,),
            ).fetchone()

    def session_last_messages(self, session_id: str, n: int) -> list[sqlite3.Row]:
        """Last n occurrences in the session, both roles, ascending seen_at."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT m.role AS role, m.text AS text, o.seen_at AS seen_at
                   FROM kb_message_occurrences o
                   JOIN kb_messages m ON m.id = o.message_id
                   WHERE o.session_id = ?
                   ORDER BY o.seen_at DESC LIMIT ?""",
                (session_id, n),
            ).fetchall()
        return list(reversed(rows))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_store_timeline.py -v`
Expected: 5 PASS

- [ ] **Step 5: gate + commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -x --timeout=60
git add claude_tap/prompt_kb/store.py tests/prompt_kb/test_store_timeline.py
git commit -m "feat(prompt_kb): KbStore 新增时间线查询——recent_sessions/session_first_user_message/session_last_messages"
```

---

### Task 2: recall.py — 格式化助手 + kb_recent 时间线逻辑

**Files:**
- Create: `claude_tap/prompt_kb/recall.py`
- Test: `tests/prompt_kb/test_recall.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `recent_sessions` / `session_first_user_message` / `session_last_messages`
- Produces:
  - `RECENT_NOTE: str`、`RECALL_NOTE: str`、`EMPTY_NOTE: str`（模块常量，Task 4 的 MCP 层透传）
  - `truncate(text: str, max_chars: int) -> str` — 超长截断并追加 `"…"`
  - `format_attribution(timestamp: str, client: str, session_id: str) -> str` — `"2026-08-15 14:32 · claude-code · session 7f3a9c"`（ISO 截到分钟；session_id 取前 6 字符）
  - `recent_overview(store: KbStore, *, client: str | None, sessions: int, messages_per_session: int) -> dict` — 返回 `{"sessions": [...], "note": RECENT_NOTE}`；空库返回 `{"sessions": [], "note": EMPTY_NOTE}`

- [ ] **Step 1: 写失败测试**

创建 `tests/prompt_kb/test_recall.py`：

```python
"""Tests for recall.py: formatting helpers + kb_recent overview logic."""

from claude_tap.prompt_kb.recall import (
    EMPTY_NOTE,
    RECENT_NOTE,
    format_attribution,
    recent_overview,
    truncate,
)
from claude_tap.prompt_kb.store import KbStore


def _msg(store, *, session, ts, text, role="user", hash_, msg_idx=0, client="claude-code"):
    return store.upsert_message(
        session_id=session, record_index=0, message_index=msg_idx, client=client,
        model="m", timestamp=ts, content_hash=hash_, text=text, role=role, seen_at=ts,
    )


def test_truncate_short_text_untouched():
    assert truncate("hello", 10) == "hello"
    assert truncate("x" * 10, 10) == "x" * 10  # exactly at limit: no marker


def test_truncate_appends_ellipsis():
    out = truncate("x" * 500, 200)
    assert out == "x" * 199 + "…"
    assert len(out) == 200


def test_format_attribution():
    assert (
        format_attribution("2026-08-15T14:32:47Z", "claude-code", "7f3a9c12-abcd")
        == "2026-08-15 14:32 · claude-code · session 7f3a9c"
    )


def test_recent_overview_empty_store(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    result = recent_overview(store, client=None, sessions=5, messages_per_session=3)
    assert result == {"sessions": [], "note": EMPTY_NOTE}


def test_recent_overview_structure_and_order(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _msg(store, session="s1", ts="2026-08-01T09:00:00Z", text="task one", hash_="a")
    _msg(store, session="s1", ts="2026-08-01T09:05:00Z", text="doing it", hash_="b", role="assistant", msg_idx=1)
    _msg(store, session="s2", ts="2026-08-02T10:00:00Z", text="task two", hash_="c")
    result = recent_overview(store, client=None, sessions=5, messages_per_session=3)
    assert result["note"] == RECENT_NOTE
    sessions = result["sessions"]
    assert [s["session_id"] for s in sessions] == ["s2", "s1"]
    s1 = sessions[1]
    assert s1["time_range"] == "2026-08-01 09:00 → 09:05"
    assert s1["first_user_message"] == "task one"
    assert [(m["role"], m["text"]) for m in s1["recent_exchanges"]] == [
        ("user", "task one"),
        ("assistant", "doing it"),
    ]
    assert all(m["timestamp"] for m in s1["recent_exchanges"])


def test_recent_overview_limits_and_truncation(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _msg(store, session="s1", ts="2026-08-01T09:00:00Z", text="t" * 500, hash_="a")
    for i in range(4):
        _msg(store, session="s1", ts=f"2026-08-01T10:0{i}:00Z", text="x" * 400,
             hash_=f"b{i}", role="assistant", msg_idx=i + 1)
    result = recent_overview(store, client=None, sessions=5, messages_per_session=2)
    s1 = result["sessions"][0]
    assert len(s1["first_user_message"]) == 200  # 199 chars + ellipsis
    assert len(s1["recent_exchanges"]) == 2
    assert all(len(m["text"]) == 300 for m in s1["recent_exchanges"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_recall.py -v`
Expected: FAIL — `ModuleNotFoundError: claude_tap.prompt_kb.recall`

- [ ] **Step 3: 实现 recall.py**

创建 `claude_tap/prompt_kb/recall.py`：

```python
"""Memory-oriented retrieval behind the kb_recall / kb_recent MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_tap.prompt_kb.store import KbStore

RECALL_NOTE = (
    "Memories from local past sessions, ranked by relevance; "
    "they may be outdated — verify against the current codebase before relying on them."
)
RECENT_NOTE = (
    "Recent sessions newest-first, independent of the current input — "
    "a plain timeline for picking up previous work."
)
EMPTY_NOTE = "The knowledge base is empty — no past sessions have been indexed yet."

_FIRST_MESSAGE_CHARS = 200
_EXCHANGE_CHARS = 300


def truncate(text: str, max_chars: int) -> str:
    """Hard-crop text to max_chars, marking truncation with an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def format_attribution(timestamp: str, client: str, session_id: str) -> str:
    """Human/model-readable provenance: '2026-08-15 14:32 · claude-code · session 7f3a9c'."""
    minute = timestamp[:16].replace("T", " ")
    return f"{minute} · {client} · session {session_id[:6]}"


def _fmt_minute(iso_ts: str) -> str:
    return iso_ts[:16].replace("T", " ")


def recent_overview(
    store: "KbStore",
    *,
    client: str | None,
    sessions: int,
    messages_per_session: int,
) -> dict[str, Any]:
    """Timeline of recent sessions for kb_recent; pure SQL, no embedder needed."""
    rows = store.recent_sessions(client, sessions)
    if not rows:
        return {"sessions": [], "note": EMPTY_NOTE}
    out: list[dict[str, Any]] = []
    for row in rows:
        first = store.session_first_user_message(row["session_id"])
        exchanges = store.session_last_messages(row["session_id"], messages_per_session)
        out.append(
            {
                "session_id": row["session_id"],
                "client": row["client"],
                "time_range": f"{_fmt_minute(row['first_ts'])} → {_fmt_minute(row['last_ts'])[-5:]}",
                "first_user_message": truncate(first["text"], _FIRST_MESSAGE_CHARS) if first else "",
                "recent_exchanges": [
                    {"role": m["role"], "text": truncate(m["text"], _EXCHANGE_CHARS), "timestamp": m["seen_at"]}
                    for m in exchanges
                ],
            }
        )
    return {"sessions": out, "note": RECENT_NOTE}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_recall.py -v`
Expected: 6 PASS

- [ ] **Step 5: gate + commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -x --timeout=60
git add claude_tap/prompt_kb/recall.py tests/prompt_kb/test_recall.py
git commit -m "feat(prompt_kb): recall 模块——时间线 overview 与出处/截断格式化"
```

---

### Task 3: recall.py — kb_recall 相似度召回逻辑

**Files:**
- Modify: `claude_tap/prompt_kb/recall.py`
- Test: `tests/prompt_kb/test_recall.py`（追加）

**Interfaces:**
- Consumes: `search_messages(store, embedder, query, *, client, limit, min_score, rel_delta, reranker, rrf_weights) -> tuple[list[SessionResult], bool]`（search.py:283，零改动直接复用）；SessionResult 有 `.session_id/.client/.model/.hits`，MessageHit 有 `.text/.timestamp/.score/.role/.content_hash`
- Produces: `recall_memories(store, embedder, query, *, client: str | None, limit: int, min_score: float | None, reranker, rrf_weights) -> dict` — 返回 `{"memories": [...], "note": RECALL_NOTE, "reranked": bool}`；空结果返回 `{"memories": [], "note": RECALL_NOTE, "reranked": bool}`

- [ ] **Step 1: 写失败测试（追加到 test_recall.py）**

文件**顶部** import 区改为（保持所有 import 在文件头部，避免 E402）：

```python
"""Tests for recall.py: formatting helpers + kb_recent/kb_recall logic."""

import pytest

pytest.importorskip("numpy")  # recall_memories depends on the [rag] extra

from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending  # noqa: E402
from claude_tap.prompt_kb.recall import (  # noqa: E402
    EMPTY_NOTE,
    RECALL_NOTE,
    RECENT_NOTE,
    format_attribution,
    recall_memories,
    recent_overview,
    truncate,
)
from claude_tap.prompt_kb.store import KbStore  # noqa: E402
from tests.prompt_kb.fake_embedder import FakeEmbedder  # noqa: E402
```

文件**末尾**追加三个测试：

```python
def test_recall_memories_flattens_and_attributes(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    _msg(store, session="s1", ts="2026-08-01T09:00:00Z",
         text="how do I fix the race condition in the worker pool", hash_="r1")
    index_pending(store, embedder)
    result = recall_memories(
        store, embedder, "race condition lock",
        client=None, limit=5, min_score=None, reranker=None,
        rrf_weights=(1.0, 1.0, 1.0),
    )
    assert result["note"] == RECALL_NOTE
    assert result["reranked"] is False
    assert len(result["memories"]) == 1
    mem = result["memories"][0]
    assert "race condition" in mem["text"]
    assert mem["role"] == "user"
    assert mem["session_id"] == "s1"
    assert mem["client"] == "claude-code"
    assert mem["timestamp"] == "2026-08-01T09:00:00Z"
    assert mem["attribution"] == "2026-08-01 09:00 · claude-code · session s1"
    assert mem["score"] > 0


def test_recall_memories_empty_store(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    result = recall_memories(
        store, embedder, "anything",
        client=None, limit=5, min_score=None, reranker=None,
        rrf_weights=(1.0, 1.0, 1.0),
    )
    assert result == {"memories": [], "note": RECALL_NOTE, "reranked": False}


def test_recall_memories_respects_limit(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    for i in range(4):
        _msg(store, session=f"s{i}", ts=f"2026-08-0{i + 1}T09:00:00Z",
             text=f"shared topic worker pool variant {i}", hash_=f"l{i}")
    index_pending(store, embedder)
    result = recall_memories(
        store, embedder, "worker pool",
        client=None, limit=2, min_score=0.0, reranker=None,
        rrf_weights=(1.0, 1.0, 1.0),
    )
    assert len(result["memories"]) == 2
```

注意：第三个测试传 `min_score=0.0`、`rel_delta` 用默认值——fake embedder 是词袋哈希，相似文本分数接近，传 0.0 关掉下限才能保证多条命中进入断言。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_recall.py -k recall_memories -v`
Expected: FAIL — `ImportError: cannot import name 'recall_memories'`

- [ ] **Step 3: 实现 recall_memories（追加到 recall.py）**

文件头部 import 区追加：

```python
from claude_tap.prompt_kb.search import search_messages

if TYPE_CHECKING:
    from claude_tap.prompt_kb.embed import Embedder
    from claude_tap.prompt_kb.rerank import Reranker
```

函数实现：

```python
def recall_memories(
    store: "KbStore",
    embedder: "Embedder",
    query: str,
    *,
    client: str | None,
    limit: int,
    min_score: float | None,
    reranker: "Reranker | None",
    rrf_weights: tuple[float, float, float],
) -> dict[str, Any]:
    """Flattened relevance-ranked message hits for kb_recall (messages corpus only)."""
    groups, reranked = search_messages(
        store,
        embedder,
        query,
        client=client,
        limit=limit,
        min_score=min_score,
        reranker=reranker,
        rrf_weights=rrf_weights,
    )
    hits = [
        (hit, group) for group in groups for hit in group.hits
    ]
    hits.sort(key=lambda pair: pair[0].score, reverse=True)
    memories = []
    for hit, group in hits[:limit]:
        memories.append(
            {
                "text": hit.text,
                "role": hit.role,
                "score": hit.score,
                "attribution": format_attribution(hit.timestamp, group.client, group.session_id),
                "session_id": group.session_id,
                "client": group.client,
                "timestamp": hit.timestamp,
            }
        )
    return {"memories": memories, "note": RECALL_NOTE, "reranked": reranked}
```

注意：MessageHit 不带 session_id，必须从分组结构里携带 `(hit, group)` 对——不要靠 content_hash 或 dataclass 相等性反查（两条内容相同的命中会认错 session）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_recall.py -v`
Expected: 全部 PASS（含 Task 2 的 6 个）

- [ ] **Step 5: gate + commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -x --timeout=60
git add claude_tap/prompt_kb/recall.py tests/prompt_kb/test_recall.py
git commit -m "feat(prompt_kb): recall_memories——search_messages 拍平为带出处记忆列表"
```

---

### Task 4: MCP 工具注册（kb_recall + kb_recent）

**Files:**
- Modify: `claude_tap/prompt_kb/mcp_server.py`（imports、`kb_status` 之后加两个工具函数、`main()` 注册）
- Test: `tests/prompt_kb/test_mcp_server.py`（追加单测）、`tests/prompt_kb/test_mcp_stdio.py`（扩展冒烟）

**Interfaces:**
- Consumes: Task 2 的 `recent_overview` / `RECENT_NOTE` / `EMPTY_NOTE`，Task 3 的 `recall_memories` / `RECALL_NOTE`
- Produces: MCP 工具 `kb_recall(query, client=None, limit=5, min_score=None) -> dict` 与 `kb_recent(client=None, sessions=5, messages_per_session=3) -> dict`，注册进 `claude-tap mcp` stdio server

- [ ] **Step 1: 写失败单测（追加到 tests/prompt_kb/test_mcp_server.py）**

复用文件头部的 `ctx` fixture 与 `_seed`（`ctx` 已 monkeypatch `_get_ctx` 和 `KbStore.default`）：

```python
def test_kb_recall_returns_attributed_memories(ctx):
    store, _ = ctx
    _seed(store)
    result = mcp_server.kb_recall("race condition lock")
    assert result["note"] and result["reranked"] is False
    mem = result["memories"][0]
    assert "race condition" in mem["text"]
    assert mem["attribution"].endswith("· codex · session s1")
    assert set(result) == {"memories", "note", "reranked"}


def test_kb_recent_returns_timeline(ctx):
    store, _ = ctx
    _seed(store)
    result = mcp_server.kb_recent()
    assert result["note"]
    session = result["sessions"][0]
    assert session["session_id"] == "s1"
    assert session["first_user_message"].startswith("how do I fix the race condition")
    assert session["recent_exchanges"][0]["role"] == "user"


def test_kb_recall_never_raises(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "_get_ctx", lambda: (_ for _ in ()).throw(EmbedderUnavailable("no model"))
    )
    result = mcp_server.kb_recall("anything")
    assert result["memories"] == [] and "no model" in result["error"]


def test_kb_recent_never_raises(monkeypatch):
    class BrokenStore:
        def recent_sessions(self, client, limit):
            raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(mcp_server.KbStore, "default", classmethod(lambda cls: BrokenStore()))
    result = mcp_server.kb_recent()
    assert result["sessions"] == [] and "locked" in result["error"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_mcp_server.py -v`
Expected: FAIL — `AttributeError: module 'claude_tap.prompt_kb.mcp_server' has no attribute 'kb_recall'`

- [ ] **Step 3: 实现 MCP 工具**

`mcp_server.py` import 区追加：

```python
from claude_tap.prompt_kb.recall import recall_memories, recent_overview
```

`kb_status` 之后、`main()` 之前追加：

```python
def kb_recall(
    query: str,
    client: str | None = None,
    limit: int = 5,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Recall what past sessions said about a topic (chat-message corpus, user + assistant).

    Call this BEFORE answering when the user's question may relate to earlier work,
    a stated preference, or a similar past problem. Each memory carries an
    "attribution" line (when · which client · which session) so its origin is
    always visible. Prompt/tool-definition search is kb_search's job, not this tool's.

    Args:
        query: Natural-language topic, e.g. "DLP proxy TLS fix".
        client: Optional client filter, e.g. "claude-code" or "codex".
        limit: Max memories (default 5 — memories are few and precise, not many).
        min_score: Minimum score (0-1); unset applies the reranker neutral-band floor.

    Returns:
        {"memories": [...], "note": str, "reranked": bool}; on failure
        {"error": str, "memories": [], "reranked": false}.
    """
    try:
        store, embedder, reranker = _get_ctx()
    except EmbedderUnavailable as exc:
        return {"error": f"embedder unavailable: {exc}", "memories": [], "reranked": False}
    try:
        index_pending(store, embedder)
    except sqlite3.OperationalError:
        pass  # dashboard's lazy indexer holds the write lock; search the stale index
    try:
        return recall_memories(
            store,
            embedder,
            query,
            client=client,
            limit=limit,
            min_score=min_score,
            reranker=reranker,
            rrf_weights=load_config().rrf_weights,
        )
    except ReindexRequired as exc:
        return {"error": str(exc), "memories": [], "reranked": False}
    except EmbedderUnavailable as exc:
        return {"error": f"embedder unavailable: {exc}", "memories": [], "reranked": False}
    except Exception as exc:  # noqa: BLE001 - never throw a stack at the MCP client
        return {"error": f"kb_recall failed: {exc}", "memories": [], "reranked": False}


def kb_recent(
    client: str | None = None,
    sessions: int = 5,
    messages_per_session: int = 3,
) -> dict[str, Any]:
    """Timeline of recent sessions — what was being worked on, newest first.

    Call this when the user asks to "continue previous work" or "what were we
    doing". Pure recency, no similarity ranking and no embedder needed. Each
    session shows its time range, opening user message, and last exchanges.

    Args:
        client: Optional client filter, e.g. "claude-code" or "codex".
        sessions: Max sessions (default 5).
        messages_per_session: Trailing exchanges per session (default 3).

    Returns:
        {"sessions": [...], "note": str}; on failure {"error": str, "sessions": []}.
    """
    try:
        store = KbStore.default()
        return recent_overview(
            store, client=client, sessions=sessions, messages_per_session=messages_per_session
        )
    except Exception as exc:  # noqa: BLE001 - never throw a stack at the MCP client
        return {"error": f"kb_recent failed: {exc}", "sessions": []}
```

`main()` 注册区改为：

```python
    server = FastMCP("claude-tap-kb")
    server.tool()(kb_search)
    server.tool()(kb_status)
    server.tool()(kb_recall)
    server.tool()(kb_recent)
    server.run()  # stdio transport
```

注意 `test_kb_recent_returns_timeline` 依赖 ctx fixture 里 `KbStore.default` 已被 monkeypatch 到临时 store——`kb_recent` 不经过 `_get_ctx`，所以那行 monkeypatch 是它唯一的注入点，测试与实现都以此为准。

- [ ] **Step 4: 跑单测确认通过**

Run: `uv run pytest tests/prompt_kb/test_mcp_server.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 扩展 stdio 冒烟测试**

`tests/prompt_kb/test_mcp_stdio.py` 的工具断言改为覆盖新工具：

```python
            tools = await session.list_tools()
            assert {"kb_search", "kb_status", "kb_recall", "kb_recent"} <= {t.name for t in tools.tools}
            result = await session.call_tool("kb_status", {})
```

并在其后追加对 `kb_recent` 的真实调用（空库，验证不报错且结构正确）：

```python
            recent = await session.call_tool("kb_recent", {})
            assert not getattr(recent, "is_error", getattr(recent, "isError", False))
            payload = json.loads(recent.content[0].text)
            assert set(payload) >= {"sessions", "note"}
```

- [ ] **Step 6: 跑冒烟 + gate + commit**

```bash
uv run pytest tests/prompt_kb/test_mcp_stdio.py tests/prompt_kb/test_mcp_server.py -v
uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -x --timeout=60
git add claude_tap/prompt_kb/mcp_server.py tests/prompt_kb/test_mcp_server.py tests/prompt_kb/test_mcp_stdio.py
git commit -m "feat(prompt_kb): MCP 新增 kb_recall/kb_recent 记忆工具——模型可调用跨会话记忆"
git push
```

---

### Task 5: README 双语文档更新

**Files:**
- Modify: `README.md`（MCP 段落，约 638 行）
- Modify: `README_zh.md`（对应段落，约 633 行）

**Interfaces:**
- Consumes: Task 4 的工具签名与 description
- Produces: 对外文档与实际工具集一致

- [ ] **Step 1: 更新 README.md**

找到 "This registers a `claude-tap-kb` server with two read-only tools" 一句，改为四个工具的描述：

```markdown
This registers a `claude-tap-kb` server with four read-only tools: `kb_search` (semantic search over prompts, tool definitions and user messages), `kb_status` (index stats), `kb_recall` (relevance-ranked memories from past chat messages, each with a visible when/client/session attribution), and `kb_recent` (a recency-ordered timeline of recent sessions for picking up previous work).
```

- [ ] **Step 2: 更新 README_zh.md**

找到 "提供两个只读工具" 一句，对应改为：

```markdown
即可注册 `claude-tap-kb` server，提供四个只读工具：`kb_search`（对 prompt、工具定义与用户消息做语义检索）、`kb_status`（索引统计）、`kb_recall`（按相关度召回历史会话消息，每条命中带时间/client/会话出处）与 `kb_recent`（按时间倒序的近期会话时间线，用于接续之前的工作）。
```

- [ ] **Step 3: gate + commit + push**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -x --timeout=60
git add README.md README_zh.md
git commit -m "docs(readme): MCP 工具集更新为四个——新增 kb_recall/kb_recent 说明"
git push
```

---

## 手动验证（实施完成后由用户执行）

1. 本机 Claude Code 新开会话，问「上次我们做了什么」——观察模型是否主动调 `kb_recent`，出处是否可读
2. 问「我之前怎么处理 DLP 代理的」——观察是否调 `kb_recall`，命中是否相关
3. 若模型不主动调用，迭代两个工具的 description 措辞（这是预期内的调参环节）
