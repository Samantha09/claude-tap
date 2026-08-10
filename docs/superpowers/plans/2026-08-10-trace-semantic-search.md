# Trace 语义检索(方向 A)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 trace 会话中的用户消息索引进 prompt 知识库,KB 页新增"会话"分区,支持自然语言搜索历史会话并跳转详情。

**Architecture:** 新增 `kb_messages` 表(与 `kb_chunks` 平行的双路管线)。新模块 `prompt_kb/messages.py` 复用 `prompt_snapshot.py` 的 provider 归一化层抽取用户消息;`extract.py`/`index.py`/`search.py` 镜像扩展;API 在现有 `results` 键外新增 `messages` 键;dashboard KB 页渲染会话命中卡片。

**Tech Stack:** Python 3.11+ / aiohttp / SQLite / numpy(可选 [rag])/ pytest / Playwright。

**Spec:** `docs/superpowers/specs/2026-08-10-trace-semantic-search-design.md`

## Global Constraints

- 每个 commit 前运行 gate:`uv run ruff check .`、`uv run ruff format --check .`、`uv run pytest tests/ -x --timeout=60`(注:本机经验用 `.venv/bin/python -m pytest` 亦可,见 f9092fb 的先例)
- 代码、注释仅英文;commit message 一律中文(含 type/scope 前缀)
- 每个 commit 只处理一个关注点
- 未装 `[rag]` 依赖时一切行为与现状一致(优雅降级)
- 用户消息 chunk 上限复用 `MAX_SECTION_CHARS = 2000`(`claude_tap/prompt_kb/chunk.py`)
- KB 库是派生数据,schema 变更走 `KbStore._migrate()` 幂等路径

## 接口总览(跨任务契约)

```python
# claude_tap/prompt_kb/messages.py (Task 2 新建)
@dataclass(frozen=True)
class UserMessage:
    record_index: int
    message_index: int
    timestamp: str
    text: str

def extract_user_messages(records: list[dict]) -> list[UserMessage]: ...
def message_content_hash(text: str) -> str: ...

# claude_tap/prompt_kb/store.py (Task 1 扩展)
def upsert_message(self, *, session_id: str, record_index: int, message_index: int,
                   client: str, model: str, timestamp: str, content_hash: str,
                   text: str, seen_at: str) -> tuple[int, bool]: ...
def pending_messages(self, limit: int) -> list[sqlite3.Row]: ...
def mark_message_indexed(self, message_id: int, embedding: bytes) -> None: ...
def mark_message_failed(self, message_id: int) -> None: ...
def requeue_failed_messages(self) -> int: ...
def indexed_messages(self) -> list[sqlite3.Row]: ...
def reset_message_embeddings(self) -> int: ...
def delete_messages_for_session(self, session_id: str) -> int: ...

# claude_tap/prompt_kb/search.py (Task 5 扩展)
@dataclass(frozen=True)
class MessageHit:
    text: str
    timestamp: str
    score: float

@dataclass
class SessionResult:
    session_id: str
    client: str
    model: str
    hits: list[MessageHit]

def search_messages(store, embedder, query, *, client=None, limit=10, min_score=0.0) -> list[SessionResult]: ...
```

---

### Task 1: `kb_messages` 存储层

**Files:**
- Modify: `claude_tap/prompt_kb/store.py`
- Test: `tests/prompt_kb/test_store_messages.py`(新建)

**Interfaces:**
- Consumes: 现有 `KbStore._connect()`、`MAX_ATTEMPTS` 模式
- Produces: 上方"接口总览"中 store.py 的全部方法;`stats()` 新增 `messages` 键

- [ ] **Step 1: 写失败测试**

新建 `tests/prompt_kb/test_store_messages.py`:

```python
"""kb_messages storage: migration, dedup upsert, index state machine."""

from claude_tap.prompt_kb.store import KbStore


def _msg(**kw):
    base = dict(
        session_id="s1", record_index=0, message_index=0,
        client="claude", model="k3", timestamp="2026-08-10T01:00:00Z",
        content_hash="h1", text="how do I fix the race condition",
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
    mid2, created2 = store.upsert_message(**_msg(
        session_id="s2", seen_at="2026-08-10T02:00:00Z"))
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_store_messages.py -x -q`
Expected: FAIL(`upsert_message` 不存在)

- [ ] **Step 3: 实现**

`claude_tap/prompt_kb/store.py` 的 `SCHEMA` 追加:

```sql
CREATE TABLE IF NOT EXISTS kb_messages (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  record_index INTEGER NOT NULL,
  message_index INTEGER NOT NULL,
  client TEXT NOT NULL,
  model TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  text TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  embedding BLOB,
  index_state TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kb_messages_state ON kb_messages(index_state);
CREATE INDEX IF NOT EXISTS idx_kb_messages_session ON kb_messages(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_messages_dedup ON kb_messages(content_hash, client);
```

`KbStore` 新增方法(全部用 `with self._connect() as conn`,与现有方法同风格):

```python
def upsert_message(
    self, *, session_id: str, record_index: int, message_index: int,
    client: str, model: str, timestamp: str, content_hash: str,
    text: str, seen_at: str,
) -> tuple[int, bool]:
    """Insert a user-message chunk; dedup on (content_hash, client).

    Returns (message_id, created). On dedup hit only last_seen is updated
    and the first-seen session_id is kept.
    """
    with self._connect() as conn:
        row = conn.execute(
            "SELECT id FROM kb_messages WHERE content_hash=? AND client=?",
            (content_hash, client),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE kb_messages SET last_seen=? WHERE id=?",
                (seen_at, row["id"]),
            )
            return int(row["id"]), False
        cur = conn.execute(
            """INSERT INTO kb_messages
               (session_id, record_index, message_index, client, model,
                timestamp, content_hash, text, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, record_index, message_index, client, model,
             timestamp, content_hash, text, seen_at),
        )
        return int(cur.lastrowid), True

def pending_messages(self, limit: int) -> list[sqlite3.Row]:
    with self._connect() as conn:
        return conn.execute(
            "SELECT id, text FROM kb_messages WHERE index_state='pending' ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()

def mark_message_indexed(self, message_id: int, embedding: bytes) -> None:
    with self._connect() as conn:
        conn.execute(
            "UPDATE kb_messages SET embedding=?, index_state='indexed' WHERE id=?",
            (embedding, message_id),
        )

def mark_message_failed(self, message_id: int) -> None:
    with self._connect() as conn:
        conn.execute(
            "UPDATE kb_messages SET index_state='failed', attempts=attempts+1 WHERE id=?",
            (message_id,),
        )

def requeue_failed_messages(self) -> int:
    with self._connect() as conn:
        cur = conn.execute(
            "UPDATE kb_messages SET index_state='pending' WHERE index_state='failed' AND attempts < ?",
            (self.MAX_ATTEMPTS,),
        )
        return cur.rowcount

def indexed_messages(self) -> list[sqlite3.Row]:
    with self._connect() as conn:
        return conn.execute(
            """SELECT id, session_id, client, model, timestamp, text, embedding
               FROM kb_messages WHERE index_state='indexed'"""
        ).fetchall()

def reset_message_embeddings(self) -> int:
    with self._connect() as conn:
        cur = conn.execute(
            "UPDATE kb_messages SET embedding=NULL, index_state='pending', attempts=0"
        )
        return cur.rowcount

def delete_messages_for_session(self, session_id: str) -> int:
    with self._connect() as conn:
        cur = conn.execute("DELETE FROM kb_messages WHERE session_id=?", (session_id,))
        return cur.rowcount
```

`stats()` 的 return dict 增加一行(放在 `"indexed"` 之后):

```python
"messages": int(
    conn.execute("SELECT COUNT(*) c FROM kb_messages").fetchone()["c"]
),
```

注意:旧库升级无需 `_migrate` 加列——`CREATE TABLE IF NOT EXISTS` 已幂等覆盖;
`_migrate()` 只处理 `kb_chunks` 的历史加列,不动。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_store_messages.py tests/prompt_kb/test_store.py -x -q`
Expected: PASS(新旧测试全绿)

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/store.py tests/prompt_kb/test_store_messages.py
git commit -m "feat(prompt_kb): 新增 kb_messages 表与去重/状态机存储方法"
```

---

### Task 2: 用户消息抽取器 `messages.py`

**Files:**
- Create: `claude_tap/prompt_kb/messages.py`
- Test: `tests/prompt_kb/test_messages.py`(新建)

**Interfaces:**
- Consumes: `claude_tap.prompt_snapshot` 的 `infer_provider()`、`_request_body()`、`_content_text()`(同项目内复用私有归一化层,避免四种格式的解析逻辑写两份);`chunk.MAX_SECTION_CHARS`、`_split_long`
- Produces: `UserMessage` dataclass、`extract_user_messages()`、`message_content_hash()`(见接口总览)

- [ ] **Step 1: 写失败测试**

新建 `tests/prompt_kb/test_messages.py`:

```python
"""User-message extraction from trace records across provider formats."""

import hashlib

from claude_tap.prompt_kb.messages import (
    extract_user_messages,
    message_content_hash,
)


def _record(body, path="/v1/messages", timestamp="2026-08-10T01:00:00Z"):
    return {
        "timestamp": timestamp,
        "request": {"method": "POST", "path": path, "body": body},
        "response": {"status": 200},
    }


def test_anthropic_user_messages():
    records = [
        _record({
            "model": "k3",
            "messages": [
                {"role": "user", "content": "how do I fix the race condition"},
                {"role": "assistant", "content": "use a lock"},
                {"role": "user", "content": [
                    {"type": "text", "text": "it still hangs"},
                    {"type": "image", "source": {"data": "..."}},
                ]},
            ],
        }),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == [
        "how do I fix the race condition",
        "it still hangs",
    ]
    assert msgs[0].record_index == 0
    assert msgs[1].message_index == 1
    assert msgs[0].timestamp == "2026-08-10T01:00:00Z"


def test_anthropic_tool_result_blocks_skipped():
    records = [
        _record({
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": "file contents here"},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t2",
                     "content": "output"},
                    {"type": "text", "text": "now fix it"},
                ]},
            ],
        }),
    ]
    msgs = extract_user_messages(records)
    # First message is pure tool_result -> dropped; second keeps only text part
    assert [m.text for m in msgs] == ["now fix it"]


def test_openai_chat_completions():
    records = [
        _record({
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": "dev"},
                {"role": "user", "content": "refactor the parser"},
                {"role": "tool", "tool_call_id": "c1", "content": "result"},
                {"role": "user", "content": [
                    {"type": "text", "text": "and add tests"},
                ]},
            ],
        }, path="/v1/chat/completions"),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["refactor the parser", "and add tests"]


def test_openai_responses_input():
    records = [
        _record({
            "model": "gpt-5",
            "instructions": "dev",
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "explain this repo"}]},
                {"type": "function_call_output", "call_id": "c1", "output": "x"},
            ],
        }, path="/v1/responses"),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["explain this repo"]


def test_gemini_contents():
    records = [
        _record({
            "contents": [
                {"role": "user", "parts": [{"text": "write a haiku"}]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "f", "response": {}}},
                ]},
                {"role": "model", "parts": [{"text": "ok"}]},
            ],
        }, path="/v1beta/models/gemini-3:generateContent"),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["write a haiku"]


def test_harness_injected_messages_filtered():
    records = [
        _record({
            "messages": [
                {"role": "user", "content": "<system-reminder>secret</system-reminder>"},
                {"role": "user", "content": "<command-message>/clear</command-message>"},
                {"role": "user", "content": "   "},
                {"role": "user", "content": "real question"},
            ],
        }),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["real question"]


def test_long_message_split():
    long_text = ("paragraph one. " * 200) + "\n\n" + ("paragraph two. " * 200)
    records = [_record({"messages": [{"role": "user", "content": long_text}]})]
    msgs = extract_user_messages(records)
    assert len(msgs) > 1
    assert all(len(m.text) <= 2000 for m in msgs)
    # split pieces share record_index, message_index increments
    assert msgs[0].record_index == msgs[1].record_index


def test_content_hash_normalizes():
    h1 = message_content_hash("hello world  \n")
    h2 = message_content_hash("hello world")
    assert h1 == h2
    assert h1 == hashlib.sha256(b"hello world").hexdigest()


def test_unknown_provider_and_empty_body_skipped():
    records = [
        {"timestamp": "t", "request": {"path": "/health", "body": {}}},
        _record({"messages": [{"role": "assistant", "content": "hi"}]}),
    ]
    assert extract_user_messages(records) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_messages.py -x -q`
Expected: FAIL(`ModuleNotFoundError: claude_tap.prompt_kb.messages`)

- [ ] **Step 3: 实现**

新建 `claude_tap/prompt_kb/messages.py`:

```python
"""Extract user messages from trace records for semantic session search.

Only genuine user-authored text is kept: tool results, harness-injected
pseudo-user messages (<system-reminder>, command envelopes), empty text,
and binary attachments are dropped. Provider parsing reuses the
normalization helpers in claude_tap.prompt_snapshot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from claude_tap.prompt_kb.chunk import MAX_SECTION_CHARS, _split_long
from claude_tap.prompt_snapshot import _content_text, _request_body, infer_provider

_HARNES_PREFIXES = ("<system-reminder", "<command-message", "<local-command")


@dataclass(frozen=True)
class UserMessage:
    record_index: int
    message_index: int
    timestamp: str
    text: str


def message_content_hash(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_user_messages(records: list[dict[str, Any]]) -> list[UserMessage]:
    out: list[UserMessage] = []
    for record_index, record in enumerate(records):
        body = _request_body(record)
        if not body:
            continue
        provider = infer_provider(record)
        timestamp = str(record.get("timestamp") or "")
        message_index = 0
        for text in _user_texts(provider, body):
            for piece in _split_message(text):
                out.append(UserMessage(
                    record_index=record_index,
                    message_index=message_index,
                    timestamp=timestamp,
                    text=piece,
                ))
                message_index += 1
    return out


def _split_message(text: str) -> list[str]:
    if len(text) <= MAX_SECTION_CHARS:
        return [text]
    return [piece for _title, piece in _split_long("", text)]


def _user_texts(provider: str, body: dict[str, Any]) -> list[str]:
    if provider == "anthropic":
        return _anthropic_user_texts(body)
    if provider == "openai":
        return _openai_user_texts(body)
    if provider == "gemini":
        return _gemini_user_texts(body)
    return []


def _keep_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return not any(stripped.startswith(prefix) for prefix in _HARNES_PREFIXES)


def _anthropic_user_texts(body: dict[str, Any]) -> list[str]:
    out: list[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return out
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # Tool results travel as role=user; keep only real text blocks.
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            text = "\n\n".join(t.strip() for t in texts if t.strip())
        else:
            text = _content_text(content)
        if _keep_text(text):
            out.append(text.strip())
    return out


def _openai_user_texts(body: dict[str, Any]) -> list[str]:
    out: list[str] = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = _content_text(msg.get("content"))
                if _keep_text(text):
                    out.append(text.strip())
    input_value = body.get("input")
    if isinstance(input_value, str):
        if _keep_text(input_value):
            out.append(input_value.strip())
    elif isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in (None, "message") or item.get("role") not in (None, "user"):
                continue
            text = _content_text(item.get("content"))
            if _keep_text(text):
                out.append(text.strip())
    prompt = body.get("prompt")
    if isinstance(prompt, str) and _keep_text(prompt):
        out.append(prompt.strip())
    return out


def _gemini_user_texts(body: dict[str, Any]) -> list[str]:
    out: list[str] = []
    contents = body.get("contents")
    if not isinstance(contents, list):
        return out
    for item in contents:
        if not isinstance(item, dict):
            continue
        if (item.get("role") or "user") != "user":
            continue
        parts = item.get("parts")
        if not isinstance(parts, list):
            continue
        texts = [
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]  # functionResponse parts carry no "text" key and are dropped implicitly
        text = "\n\n".join(t.strip() for t in texts if t.strip())
        if _keep_text(text):
            out.append(text.strip())
    return out
```

注意 `test_gemini_contents` 中 functionResponse part 没有 `text` 键所以被自然过滤——
注释里已说明,无需特判。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_messages.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/messages.py tests/prompt_kb/test_messages.py
git commit -m "feat(prompt_kb): 新增四种 provider 格式的用户消息抽取器"
```

---

### Task 3: `extract.py` 接入消息抽取

**Files:**
- Modify: `claude_tap/prompt_kb/extract.py`
- Test: `tests/prompt_kb/test_extract.py`(扩展)

**Interfaces:**
- Consumes: Task 1 的 `upsert_message()`、Task 2 的 `extract_user_messages()` / `message_content_hash()`
- Produces: `extract_session()` 除返回 snapshot_id 外,副作用新增 kb_messages 行;签名变为 `extract_session(store, *, session_id, client, records, processed_at) -> int | None`(不变),新增独立函数 `extract_messages(store, *, session_id, client, records) -> int`(返回新建消息数)

- [ ] **Step 1: 写失败测试**

先看 `tests/prompt_kb/test_extract.py` 现有 fixture 风格,追加:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_extract.py -x -q`
Expected: FAIL(`extract_messages` 不存在 / 消息未入库)

- [ ] **Step 3: 实现**

`claude_tap/prompt_kb/extract.py` 顶部 import 追加:

```python
from claude_tap.prompt_kb.messages import extract_user_messages, message_content_hash
```

新增函数并接入 `extract_session`:

```python
def extract_messages(store: KbStore, *, session_id: str, client: str, records: list[dict[str, Any]]) -> int:
    """Store user messages from a session's records into kb_messages.

    Returns the number of newly created (non-deduped) message chunks.
    """
    created = 0
    for msg in extract_user_messages(records):
        body = _record_model(records[msg.record_index])
        _id, was_created = store.upsert_message(
            session_id=session_id,
            record_index=msg.record_index,
            message_index=msg.message_index,
            client=client,
            model=body,
            timestamp=msg.timestamp,
            content_hash=message_content_hash(msg.text),
            text=msg.text,
            seen_at=msg.timestamp or datetime.now(timezone.utc).isoformat(),
        )
        if was_created:
            created += 1
    return created


def _record_model(record: dict[str, Any]) -> str:
    req = record.get("request") if isinstance(record.get("request"), dict) else {}
    body = req.get("body") if isinstance(req.get("body"), dict) else {}
    return str(body.get("model") or "")
```

在 `extract_session()` 的 `store.record_source(session_id, snapshot_id, processed_at)` 之前插入:

```python
    extract_messages(store, session_id=session_id, client=client, records=records)
```

注意放在 `record_source` 之前:消息入库失败会抛异常,会话不会被标记已处理,
下轮重试(与 snapshot 失败语义一致)。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_extract.py -x -q`
Expected: PASS(含既有测试——既有 fixture 的 records 若无用户消息则行为不变)

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/extract.py tests/prompt_kb/test_extract.py
git commit -m "feat(prompt_kb): 会话抽取管线接入用户消息入库"
```

---

### Task 4: `index.py` 双表懒索引

**Files:**
- Modify: `claude_tap/prompt_kb/index.py`
- Test: `tests/prompt_kb/test_index.py`(扩展)

**Interfaces:**
- Consumes: Task 1 的 `pending_messages/mark_message_indexed/mark_message_failed/requeue_failed_messages/reset_message_embeddings`
- Produces: `index_pending()` 返回 dict 新增 `messages_indexed`/`messages_failed` 键;`rebuild_index()` 同时重置两表

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_index.py` 追加(参考现有 `index_pending` 测试的 fixture):

```python
def test_index_pending_embeds_messages(tmp_path):
    from tests.prompt_kb.fake_embedder import FakeEmbedder
    store = KbStore(tmp_path / "kb.sqlite3")
    store.upsert_message(
        session_id="s1", record_index=0, message_index=0,
        client="claude", model="k3", timestamp="t",
        content_hash="h1", text="how to fix race condition",
        seen_at="t",
    )
    embedder = FakeEmbedder()
    result = index_pending(store, embedder)
    assert result["messages_indexed"] == 1
    assert result["messages_failed"] == 0
    assert len(store.indexed_messages()) == 1


def test_index_pending_message_batch_failure_marks_failed(tmp_path):
    class FailingEmbedder:
        name = "fail"
        dimension = 16
        def embed(self, texts):
            raise RuntimeError("boom")
    store = KbStore(tmp_path / "kb.sqlite3")
    store.upsert_message(
        session_id="s1", record_index=0, message_index=0,
        client="claude", model="k3", timestamp="t",
        content_hash="h1", text="x", seen_at="t",
    )
    result = index_pending(store, FailingEmbedder())
    assert result["messages_failed"] == 1


def test_rebuild_resets_both_tables(tmp_path):
    from tests.prompt_kb.fake_embedder import FakeEmbedder
    store = KbStore(tmp_path / "kb.sqlite3")
    snap_id, _ = store.upsert_snapshot(
        content_hash="h", client="c", provider="p", model="m",
        system_prompt="s", developer_prompt="", tools_json="[]", seen_at="t",
    )
    store.replace_chunks(snap_id, [("tool", "t", "text")])
    store.upsert_message(
        session_id="s1", record_index=0, message_index=0,
        client="c", model="m", timestamp="t",
        content_hash="h1", text="msg", seen_at="t",
    )
    embedder = FakeEmbedder()
    index_pending(store, embedder)
    result = rebuild_index(store, embedder)
    assert result["indexed"] == 1 and result["messages_indexed"] == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_index.py -x -q`
Expected: FAIL(`messages_indexed` 键不存在)

- [ ] **Step 3: 实现**

`claude_tap/prompt_kb/index.py` 的 `index_pending` 末尾(`return` 之前)追加消息索引,
并把 return 改为:

```python
    messages_indexed, messages_failed = _index_pending_messages(store, embedder, batch_size)
    return {
        "indexed": indexed,
        "failed": failed,
        "messages_indexed": messages_indexed,
        "messages_failed": messages_failed,
        "remaining": store.stats()["pending"],
    }
```

新增私有函数(与 chunk 循环同构):

```python
def _index_pending_messages(store: KbStore, embedder: Embedder, batch_size: int) -> tuple[int, int]:
    indexed = failed = 0
    store.requeue_failed_messages()
    while True:
        batch = store.pending_messages(batch_size)
        if not batch:
            break
        try:
            vectors = embedder.embed([row["text"] for row in batch])
        except Exception:  # noqa: BLE001 - one bad batch must not stop indexing
            logger.warning("message embedding batch failed", exc_info=True)
            for row in batch:
                store.mark_message_failed(row["id"])
            failed += len(batch)
            continue
        for row, blob in zip(batch, vectors_to_blob(vectors)):
            store.mark_message_indexed(row["id"], blob)
        indexed += len(batch)
    return indexed, failed
```

`rebuild_index` 改为:

```python
def rebuild_index(store: KbStore, embedder: Embedder) -> dict:
    store.reset_embeddings()
    store.reset_message_embeddings()
    return index_pending(store, embedder)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_index.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/index.py tests/prompt_kb/test_index.py
git commit -m "feat(prompt_kb): 懒索引循环扩展为 chunks/messages 双表索引"
```

---

### Task 5: `search.py` 会话消息检索

**Files:**
- Modify: `claude_tap/prompt_kb/search.py`
- Test: `tests/prompt_kb/test_search_messages.py`(新建)

**Interfaces:**
- Consumes: Task 1 的 `indexed_messages()`;现有 `_check_embedder_meta`、`ReindexRequired`
- Produces: `MessageHit` / `SessionResult` / `search_messages()`(见接口总览);私有 `_cosine_scores()` 供两路复用

- [ ] **Step 1: 写失败测试**

新建 `tests/prompt_kb/test_search_messages.py`:

```python
"""Semantic search over indexed user messages."""

import pytest

from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending
from claude_tap.prompt_kb.search import ReindexRequired, search_messages
from claude_tap.prompt_kb.store import KbStore
from tests.prompt_kb.fake_embedder import FakeEmbedder


@pytest.fixture()
def seeded(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    for i, (sid, text) in enumerate([
        ("s1", "how do I fix the race condition in the worker pool"),
        ("s1", "the lock ordering was wrong"),
        ("s2", "recipe for tomato soup"),
    ]):
        store.upsert_message(
            session_id=sid, record_index=i, message_index=0,
            client="claude" if sid == "s1" else "codex",
            model="k3", timestamp=f"2026-08-0{i+1}T00:00:00Z",
            content_hash=f"h{i}", text=text, seen_at="t",
        )
    index_pending(store, embedder)
    return store, embedder


def test_search_groups_by_session(seeded):
    store, embedder = seeded
    results = search_messages(store, embedder, "race condition lock")
    assert results
    top = results[0]
    assert top.session_id == "s1"
    assert top.client == "claude"
    assert all(h.text for h in top.hits)
    assert all(h.score > 0 for h in top.hits)
    assert top.hits[0].timestamp  # timestamp carried through


def test_search_client_filter(seeded):
    store, embedder = seeded
    results = search_messages(store, embedder, "tomato soup", client="codex")
    assert [r.session_id for r in results] == ["s2"]
    results = search_messages(store, embedder, "tomato soup", client="claude")
    assert results == []


def test_search_min_score(seeded):
    store, embedder = seeded
    results = search_messages(store, embedder, "race condition", min_score=0.99)
    assert results == []


def test_reindex_required_on_embedder_mismatch(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    store.set_meta("embedder_name", "other")
    store.set_meta("embedding_dim", "99")
    with pytest.raises(ReindexRequired):
        search_messages(store, FakeEmbedder(), "q")


def test_perf_20k_chunks_under_200ms(tmp_path):
    import time
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    rows = [
        (f"s{i % 100}", i, 0, "claude", "k3", "t", f"h{i}",
         f"message {i} about topic {i % 50}", "t")
        for i in range(20_000)
    ]
    with store._connect() as conn:
        conn.executemany(
            """INSERT INTO kb_messages
               (session_id, record_index, message_index, client, model,
                timestamp, content_hash, text, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    index_pending(store, embedder)
    start = time.perf_counter()
    search_messages(store, embedder, "topic 42 message")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2
```

(性能测试用 FakeEmbedder 16 维;真实 e5 为 384 维,matmul 复杂度同阶,
20k×384 仍在毫秒级,阈值 200ms 留足余量。)

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_search_messages.py -x -q`
Expected: FAIL(`search_messages` 不存在)

- [ ] **Step 3: 实现**

`claude_tap/prompt_kb/search.py` 追加(并把现有内联余弦计算抽成 `_cosine_scores`
供两处复用):

```python
@dataclass(frozen=True)
class MessageHit:
    text: str
    timestamp: str
    score: float


@dataclass
class SessionResult:
    session_id: str
    client: str
    model: str
    hits: list[MessageHit] = field(default_factory=list)


def _cosine_scores(matrix, query_vec):
    import numpy as np
    q_norm = np.linalg.norm(query_vec) or 1.0
    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms[m_norms == 0] = 1.0
    return (matrix @ query_vec) / (m_norms * q_norm)


def search_messages(
    store: KbStore,
    embedder: Embedder,
    query: str,
    *,
    client: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
) -> list[SessionResult]:
    _check_embedder_meta(store, embedder)
    rows = store.indexed_messages()
    if client:
        rows = [row for row in rows if row["client"] == client]
    if not rows:
        return []
    try:
        import numpy as np
    except ImportError as exc:
        raise EmbedderUnavailable(
            "numpy is not installed; install the optional dependency: pip install 'claude-tap[rag]'"
        ) from exc
    matrix = np.array([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
    embed_query = getattr(embedder, "embed_query", None) or embedder.embed
    query_vec = np.array(embed_query([query])[0], dtype=np.float32)
    scores = _cosine_scores(matrix, query_vec)

    groups: dict[str, SessionResult] = {}
    for row, score in zip(rows, scores):
        if score <= min_score:
            continue
        group = groups.setdefault(
            row["session_id"],
            SessionResult(
                session_id=row["session_id"],
                client=row["client"],
                model=row["model"],
            ),
        )
        group.hits.append(
            MessageHit(text=row["text"], timestamp=row["timestamp"], score=float(score))
        )
    ordered = sorted(
        groups.values(),
        key=lambda g: max((h.score for h in g.hits), default=0.0),
        reverse=True,
    )
    for group in ordered:
        group.hits = sorted(group.hits, key=lambda h: h.score, reverse=True)[:3]
    return ordered[:limit]
```

同时把现有 `search()` 里的内联余弦段替换为 `scores = _cosine_scores(matrix, query_vec)`
(纯重构,同一 commit 内,行为不变由既有 test_search.py 保证)。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_search_messages.py tests/prompt_kb/test_search.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/search.py tests/prompt_kb/test_search_messages.py
git commit -m "feat(prompt_kb): 新增用户消息语义检索 search_messages"
```

---

### Task 6: API——search 双分区 + 删除级联

**Files:**
- Modify: `claude_tap/live.py`(`_handle_kb_search` 在 581-628 行附近;`_handle_delete_session` 820 行附近;`_handle_delete_sessions` 836 行附近)
- Test: `tests/prompt_kb/test_kb_api.py`(扩展)

**Interfaces:**
- Consumes: Task 5 的 `search_messages`;Task 1 的 `delete_messages_for_session`
- Produces: `/api/kb/search` 响应新增 `"messages": [{session_id, client, model, hits: [{text, timestamp, score}]}]`;DELETE 会话端点级联清理 kb_messages

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_kb_api.py` 追加:

```python
@pytest.fixture()
def seeded_kb_messages(trace_db, monkeypatch):
    store = KbStore.default()
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    store.upsert_message(
        session_id="sess-abc", record_index=0, message_index=0,
        client="claude", model="k3", timestamp="2026-08-10T01:00:00Z",
        content_hash="h1", text="how to fix the race condition",
        seen_at="2026-08-10T01:00:00Z",
    )
    index_pending(store, embedder)
    monkeypatch.setattr("claude_tap.live.create_embedder", lambda config: embedder)
    return embedder


async def test_kb_search_includes_messages(trace_db, seeded_kb_messages, tmp_path):
    server = LiveViewerServer(port=0, migrate_from=tmp_path, dashboard_mode=True)
    port = await server.start()
    try:
        status, payload = await _get_json(port, "/api/kb/search?q=race+condition")
        assert status == 200
        assert "messages" in payload
        assert payload["messages"][0]["session_id"] == "sess-abc"
        assert payload["messages"][0]["hits"][0]["text"]
        assert payload["results"] == []  # prompt partition still present
    finally:
        await server.stop()


async def test_delete_session_cascades_kb_messages(trace_db, seeded_kb_messages, tmp_path):
    store = KbStore.default()
    assert store.stats()["messages"] == 1
    server = LiveViewerServer(port=0, migrate_from=tmp_path, dashboard_mode=True)
    port = await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"http://127.0.0.1:{port}/api/sessions/sess-abc"
            ) as resp:
                assert resp.status in (200, 404)  # 404 if trace session absent; cascade must still run
        assert store.stats()["messages"] == 0
    finally:
        await server.stop()
```

注意:级联测试里 trace 库中可能不存在 `sess-abc` 会话——级联清理必须无论
trace 删除结果如何都执行(见下方实现的位置)。

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_kb_api.py -x -q`
Expected: FAIL(`messages` 键不存在 / 级联未发生)

- [ ] **Step 3: 实现**

`claude_tap/live.py` 顶部 import 区追加:

```python
from claude_tap.prompt_kb.search import search_messages as kb_search_messages
```

`_handle_kb_search` 中,`results = kb_search(...)` 之后、return 之前插入
(放在同一个 try 块内,共享 ReindexRequired/EmbedderUnavailable 处理):

```python
        messages = kb_search_messages(
            KbStore.default(),
            embedder,
            query,
            client=request.query.get("client") or None,
            limit=int(request.query.get("limit", "10")),
            min_score=min_score,
        )
```

return 的 dict 增加:

```python
                "messages": [
                    {
                        "session_id": group.session_id,
                        "client": group.client,
                        "model": group.model,
                        "hits": [
                            {"text": h.text, "timestamp": h.timestamp, "score": h.score}
                            for h in group.hits
                        ],
                    }
                    for group in messages
                ],
```

级联删除——新增模块级私有辅助:

```python
def _delete_kb_messages_quietly(session_ids: list[str]) -> None:
    """Cascade-delete KB message rows; KB failures must never block trace deletion."""
    try:
        store = KbStore.default()
    except Exception:  # noqa: BLE001
        logger.exception("kb store unavailable during session cascade delete")
        return
    for sid in session_ids:
        try:
            store.delete_messages_for_session(sid)
        except Exception:  # noqa: BLE001
            logger.exception("kb message cascade delete failed for session %s", sid)
```

在 `_handle_delete_session` 中 `store.delete_session(session_id)` 调用之后
(无论其返回结果如何)插入 `_delete_kb_messages_quietly([session_id])`;
在 `_handle_delete_sessions` 成功收集到 `deletable_ids` 之后插入
`_delete_kb_messages_quietly(deletable_ids)`。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_kb_api.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_tap/live.py tests/prompt_kb/test_kb_api.py
git commit -m "feat(prompt_kb): kb search API 新增会话消息分区并级联删除"
```

---

### Task 7: CLI 输出会话小节

**Files:**
- Modify: `claude_tap/prompt_kb/cli.py`
- Test: `tests/prompt_kb/test_cli.py`(扩展)

**Interfaces:**
- Consumes: Task 5 的 `search_messages`
- Produces: `kb search` 在 prompt 结果后打印 `messages:` 小节;`kb status` 因 `stats()` 新增键自动显示 `messages=N`(无需改码)

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_cli.py` 追加(参考现有 search 测试的 capsys 用法):

```python
def test_kb_search_prints_message_section(tmp_path, monkeypatch, capsys):
    store = KbStore.default()
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    store.upsert_message(
        session_id="sess-1", record_index=0, message_index=0,
        client="claude", model="k3", timestamp="2026-08-10T01:00:00Z",
        content_hash="h1", text="how to fix the race condition",
        seen_at="t",
    )
    index_pending(store, embedder)
    monkeypatch.setattr("claude_tap.prompt_kb.cli.create_embedder", lambda config: embedder)
    rc = kb_main(["search", "race condition"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "messages:" in out
    assert "sess-1" in out
```

(测试前确认 test_cli.py 现有 fixture 如何重定向 KB 库——沿用同一模式。)

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_cli.py -x -q`
Expected: FAIL(输出无 `messages:`)

- [ ] **Step 3: 实现**

`claude_tap/prompt_kb/cli.py` import 追加 `search_messages`,`kb_main` 的
search 分支在现有 for 循环打印后追加:

```python
    message_results = search_messages(
        store, embedder, args.query, client=args.client, limit=args.limit
    )
    if message_results:
        print("messages:")
        for rank, group in enumerate(message_results, 1):
            print(f"[{rank}] session {group.session_id} ({group.client} / {group.model})")
            for hit in group.hits:
                print(f"    score={hit.score:.3f} {hit.timestamp}")
                print(f"    {hit.text[:200]}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_cli.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/cli.py tests/prompt_kb/test_cli.py
git commit -m "feat(prompt_kb): kb CLI 搜索输出新增会话消息小节"
```

---

### Task 8: dashboard KB 页"会话"分区

**Files:**
- Modify: `claude_tap/dashboard.html`(JS 函数区 2493-2700 附近;i18n 字典 en ~1506 行、zh ~1645 行;`kbSearch` 2493 行;`kbRenderFiltered` 2550 行)
- Test: `tests/prompt_kb/test_kb_render_logic.py`(扩展 Python 镜像)、`tests/prompt_kb/test_kb_render_browser.py`(扩展 Playwright)

**Interfaces:**
- Consumes: Task 6 的 API `"messages"` 键;现有 `kbScoreClass`/`kbEmptyState`/`t()`/`fmtTime()`
- Produces: JS 函数 `kbRenderMessageCard(group)`、`kbFilterMessageGroups(groups, minScore)`(纯函数,可在 Python 镜像测试);i18n 键 `kb_messages_section`、`kb_view_session`

- [ ] **Step 1: 写失败测试(Layer 1 Python 镜像)**

`tests/prompt_kb/test_kb_render_logic.py` 追加镜像函数与测试:

```python
def filter_message_groups(groups, min_score):
    return [
        {**g, "hits": [h for h in g.get("hits", []) if h["score"] >= min_score]}
        for g in groups
        if any(h["score"] >= min_score for h in g.get("hits", []))
    ]


def test_filter_message_groups():
    groups = [
        {"session_id": "s1", "hits": [{"score": 0.9}, {"score": 0.3}]},
        {"session_id": "s2", "hits": [{"score": 0.2}]},
    ]
    filtered = filter_message_groups(groups, 0.5)
    assert [g["session_id"] for g in filtered] == ["s1"]
    assert len(filtered[0]["hits"]) == 1
```

以及模板同步测试(防止 JS 与镜像漂移,参考现有 test_kb_render_logic 中
对 dashboard.html 的断言模式):

```python
def test_template_contains_message_rendering():
    from claude_tap.dashboard import read_dashboard_template
    html = read_dashboard_template()
    assert "kbRenderMessageCard" in html
    assert "kb_messages_section" in html
    assert "kb_view_session" in html
    assert "kbFilterMessageGroups" in html
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_kb_render_logic.py -x -q`
Expected: FAIL(模板断言失败)

- [ ] **Step 3: 实现 JS 与 i18n**

`dashboard.html` i18n 字典 en 块(`kb_no_results_title` 附近)加:

```js
    kb_messages_section: "Sessions",
    kb_view_session: "View session →",
```

zh 块对应位置加:

```js
    kb_messages_section: "会话",
    kb_view_session: "查看会话 →",
```

`kbSearch` 中 `renderKbResults(data.results || [])` 改为
`renderKbResults(data.results || [], data.messages || [])`;
501/409 分支里同时清空 `kbLastMessages = []`。

JS 全局状态与纯函数(放在 `kbLastGroups` 声明附近):

```js
let kbLastMessages = null;

function kbFilterMessageGroups(groups, minScore) {
  return (groups || [])
    .map(g => ({ ...g, hits: (g.hits || []).filter(h => h.score >= minScore) }))
    .filter(g => g.hits.length);
}
```

`renderKbResults(results, messages)`:

```js
function renderKbResults(results, messages = []) {
  kbLastGroups = results || [];
  kbLastMessages = messages || [];
  kbRenderFiltered();
}
```

`kbRenderFiltered()` 末尾(folded prompt 卡片渲染完之后)追加:

```js
  const msgGroups = kbFilterMessageGroups(kbLastMessages || [], minScore);
  if (msgGroups.length) {
    const section = document.createElement("div");
    section.className = "kb-section-title";
    section.textContent = t("kb_messages_section");
    container.appendChild(section);
    for (const g of msgGroups) container.appendChild(kbRenderMessageCard(g));
  }
```

注意空态逻辑微调:`kbLastGroups.length === 0` 的早退条件需改为
"prompt 与 messages 都为空才显示 📭 空态"——把第 2559 行的判断改为:

```js
  if (!kbLastGroups.length && !(kbLastMessages || []).length) {
```

以及 `filtered.length === 0` 分支前计算好 msgGroups,只有两者皆空才走
filtered-by-score 空态。

`kbRenderMessageCard(group)`:

```js
function kbRenderMessageCard(group) {
  const card = document.createElement("div");
  card.className = "kb-group kb-message-group";
  const header = document.createElement("div");
  header.className = "kb-group-header";
  const title = document.createElement("span");
  title.className = "kb-group-title";
  title.textContent = `${group.client} / ${group.model}`;
  header.appendChild(title);
  const latest = (group.hits || []).reduce(
    (acc, h) => (h.timestamp > acc ? h.timestamp : acc), "");
  if (latest) {
    const meta = document.createElement("span");
    meta.className = "kb-group-meta";
    meta.textContent = fmtTime(latest);
    header.appendChild(meta);
  }
  card.appendChild(header);
  for (const hit of group.hits) card.appendChild(kbRenderHit({
    kind: "user_message",
    title: "",
    text: hit.text,
    score: hit.score,
  }));
  const link = document.createElement("a");
  link.className = "kb-session-link";
  link.href = `/dashboard/session/${encodeURIComponent(group.session_id)}`;
  link.textContent = t("kb_view_session");
  card.appendChild(link);
  return card;
}
```

`kbRenderHit` 的 badge 分支(2645-2646 行)需兼容 `kind === "user_message"`:

```js
  badge.className = `kb-hit-kind ${hit.kind === "tool" ? "tool" : hit.kind === "user_message" ? "message" : "prompt"}`;
  badge.textContent = hit.kind === "tool" ? t("kb_kind_tool") : hit.kind === "user_message" ? t("kb_kind_message") : t("kb_kind_prompt");
```

同时加 i18n 键 `kb_kind_message`(en: "Message",zh: "消息")与 CSS 类
`.kb-hit-kind.message`(参照 `.kb-hit-kind.tool` 的既有样式换一个区分色),
`.kb-section-title`(分区标题,沿用 kb-summary 的字号/间距变量),
`.kb-session-link`(按钮式链接,复用 `.kb-btn-secondary` 视觉)。

- [ ] **Step 4: 写 Layer 2 Playwright 测试并运行**

`tests/prompt_kb/test_kb_render_browser.py` 追加(沿用现有注入模式):

```python
MESSAGES = [
    {
        "session_id": "sess-abc", "client": "claude", "model": "k3-256k",
        "hits": [
            {"text": "how do I fix the race condition in the worker pool",
             "timestamp": "2026-08-09T10:00:00Z", "score": 0.87},
        ],
    },
]


def test_kb_message_section_rendered(page, tmp_path):
    html_path = tmp_path / "dashboard.html"
    html_path.write_text(read_dashboard_template(), encoding="utf-8")
    page.goto(f"file://{html_path}")
    page.evaluate(
        """([groups, messages]) => {
            kbLastGroups = groups;
            renderKbResults(groups, messages);
        }""",
        [GROUPS, MESSAGES],
    )
    section = page.locator(".kb-section-title", has_text="Sessions")
    assert section.count() == 1
    card = page.locator(".kb-message-group")
    assert card.count() == 1
    link = card.locator("a.kb-session-link")
    assert link.get_attribute("href") == "/dashboard/session/sess-abc"
    assert "race condition" in card.locator(".kb-hit-text").inner_text()
```

Run: `uv run pytest tests/prompt_kb/test_kb_render_browser.py tests/prompt_kb/test_kb_render_logic.py -x -q`
Expected: PASS(zh 语言块再断言一次 `kb_messages_section` 中文文案,参考现有双语断言)

- [ ] **Step 5: Commit**

```bash
git add claude_tap/dashboard.html tests/prompt_kb/test_kb_render_logic.py tests/prompt_kb/test_kb_render_browser.py
git commit -m "feat(prompt_kb): 知识库页新增会话消息分区(卡片/过滤/跳转/i18n)"
```

---

### Task 9: 全量 gate + 真实数据验证

**Files:**
- 无新增代码;验证与文档收尾

- [ ] **Step 1: 全量 gate**

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -x --timeout=60
```

- [ ] **Step 2: 真实 trace 验证(证据)**

用本机真实 trace 库(`~/.local/share/claude-tap` 或 `CLOUDTAP_DB` 指向的库):

```bash
claude-tap kb reindex   # 触发双表重建,观察 indexed/messages_indexed 进度
claude-tap kb status    # 确认 messages > 0
claude-tap kb search "一个你真实问过的问题"   # 确认 messages: 小节命中
```

再开 dashboard(`http://127.0.0.1:19527`),KB 页搜索同一问题,确认"会话"
分区卡片渲染、点击跳转会话详情。按仓库截图政策留存真实 trace 截图证据
(`.traces/` 数据,禁止合成 mock)。

- [ ] **Step 3: 更新 spec 状态并收尾**

把 spec 文档状态行改为"已实现",commit:

```bash
git commit -m "docs(prompt_kb): 方向 A 设计文档状态更新为已实现"
```

---

## Self-Review 记录

- Spec 覆盖:数据模型→Task 1;抽取分块→Task 2/3;索引→Task 4;检索→Task 5;
  API→Task 6;CLI→Task 7;UI→Task 8;验收标准 1/2/3/4/5→Task 8/6/1/既有降级测试/Task 5 性能测试。无缺口。
- 占位符:无;所有步骤均含完整代码与断言。
- 类型一致:`upsert_message` 参数名在 Task 1/3/5/6/7 一致;`search_messages`
  返回 `SessionResult.hits: list[MessageHit]`,API 序列化字段与 UI 消费字段
  (`text/timestamp/score`)一致;`kbRenderHit` 复用时 `kind="user_message"`
  在 Task 8 的 badge 分支有对应处理。
- 已知让步:Task 8 的 `kbRenderFiltered` 空态微调涉及对既有代码的两处条件修改,
  已在步骤中给出改法;执行者需注意不要把 prompt 空态逻辑改坏
  (既有 Playwright 测试会兜底)。
