# 知识库聊天内容检索（用户输入 + 模型回复为主要依据）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 prompt 知识库以聊天内容（用户输入 + 模型回复正文）为主要检索依据：assistant 回复入库可检索、prompt 区样板降噪、跨快照去重、相对阈值截断长尾。

**Architecture:** 演进式扩展——`kb_messages` 加 `role` 列（存量迁移 + 全量回填），assistant 正文从 trace response 体按 provider 解析；`chunk.py` 索引期剔除样板 section（存量打开时一次性清除）；`search.py` 增加跨快照内容折叠与 `rel_delta` 相对阈值；检索输出 messages 分区优先并带角色标注。

**Tech Stack:** Python 3 / SQLite / numpy 暴力余弦 / aiohttp(dashboard) / FastMCP(stdio)；前端为 dashboard.html 内联 JS（两层测试：Python 镜像 + Playwright）。

**Spec:** `docs/superpowers/specs/2026-08-12-chat-content-search-design.md`（决策均已与用户确认）

## Global Constraints

- Commit message 一律中文（`type(scope):` 前缀保留英文），代码/注释英文
- 测试运行器：`.venv/bin/pytest`（本机 DLP 拦截外网，不要用 `uv run` 触发联网解析）
- search 相关测试以 `numpy = pytest.importorskip("numpy")` 开头（[rag] 可选依赖）
- Python 文件带 `from __future__ import annotations`；ruff 风格（120 列）
- mcp 依赖钉 `<2`（mcp 2.0 API 不兼容），不得改动
- 知识库只做本地存储与检索，不新增任何外发路径
- FakeEmbedder（`tests/prompt_kb/fake_embedder.py`）：16 维 bag-of-words，token 经 sha256 分桶；已验证分数：query `alpha beta gamma delta epsilon zeta` 对 "全量 token 文本"≈0.938、对 "仅含 alpha 文本"≈0.510、对 `write elegant prose`≈0.667

---

### Task 1: store.py——`role` 列迁移、去重键扩展、stats 拆分

**Files:**
- Modify: `claude_tap/prompt_kb/store.py`
- Test: `tests/prompt_kb/test_store_messages.py`

**Interfaces:**
- Consumes: 现有 `KbStore`/`SCHEMA`/`_migrate` 幂等模式
- Produces: `upsert_message(..., role: str = "user")`；`indexed_messages()` 行含 `role` 键；`stats()` 新增 `messages_user`/`messages_assistant`；迁移副作用：旧库 `kb_sources.messages_done` 全量重置为 0（Task 3 的回填依赖此触发）

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_store_messages.py` 追加：

```python
"""追加：role 列迁移与新去重键。"""

import sqlite3


def test_same_text_user_and_assistant_both_stored(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _, c1 = store.upsert_message(**_msg())
    _, c2 = store.upsert_message(**_msg(role="assistant", record_index=1))
    assert c1 is True and c2 is True
    assert len(store.pending_messages(10)) == 2


def test_upsert_dedup_within_same_role(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    _, c1 = store.upsert_message(**_msg(role="assistant"))
    _, c2 = store.upsert_message(**_msg(role="assistant", session_id="s2"))
    assert c1 is True and c2 is False
    assert len(store.pending_messages(10)) == 1


def test_stats_split_by_role(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    store.upsert_message(**_msg())
    store.upsert_message(**_msg(role="assistant", content_hash="h2", text="a longer assistant reply text"))
    stats = store.stats()
    assert stats["messages"] == 2
    assert stats["messages_user"] == 1
    assert stats["messages_assistant"] == 1


def test_migrate_old_db_adds_role_and_resets_backfill(tmp_path):
    """旧 schema 库（无 role 列、旧去重索引）打开后：role 迁移、索引重建、messages_done 重置。"""
    db = tmp_path / "kb.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE kb_messages (
          id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, record_index INTEGER NOT NULL,
          message_index INTEGER NOT NULL, client TEXT NOT NULL, model TEXT NOT NULL,
          timestamp TEXT NOT NULL, content_hash TEXT NOT NULL, text TEXT NOT NULL,
          last_seen TEXT NOT NULL, embedding BLOB,
          index_state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0
        );
        CREATE UNIQUE INDEX idx_kb_messages_dedup ON kb_messages(content_hash, client);
        CREATE TABLE kb_sources (
          session_id TEXT PRIMARY KEY, snapshot_id INTEGER,
          processed_at TEXT NOT NULL, messages_done INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO kb_messages (session_id, record_index, message_index, client, model,
          timestamp, content_hash, text, last_seen)
        VALUES ('s1', 0, 0, 'claude', 'k3', 't', 'h1', 'old row text', 't');
        INSERT INTO kb_sources (session_id, snapshot_id, processed_at, messages_done)
        VALUES ('s1', NULL, 't', 1);
        """
    )
    conn.close()
    store = KbStore(db)  # 触发迁移
    rows = store.pending_messages(10)
    assert rows[0]["role"] == "user"  # 存量行默认 user
    with sqlite3.connect(db) as check:
        cols = [r[1] for r in check.execute("PRAGMA index_info(kb_messages)").fetchall()]
        idx_cols = [r[2] for r in check.execute("PRAGMA index_info(idx_kb_messages_dedup)").fetchall()]
        done = check.execute("SELECT messages_done FROM kb_sources WHERE session_id='s1'").fetchone()[0]
    assert "role" in cols
    assert idx_cols == ["content_hash", "client", "role"]
    assert done == 0  # 回填被触发
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/prompt_kb/test_store_messages.py -x -q`
Expected: FAIL（`upsert_message() got an unexpected keyword argument 'role'` / 无 role 列）

- [ ] **Step 3: 实现**

`store.py` 修改：

1. `SCHEMA` 中 `kb_messages` 定义在 `attempts` 后加一行 `role TEXT NOT NULL DEFAULT 'user'`；去重索引改为：

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_messages_dedup ON kb_messages(content_hash, client, role);
```

2. `_migrate()` 末尾追加：

```python
        message_columns = {row["name"] for row in conn.execute("PRAGMA table_info(kb_messages)")}
        if "role" not in message_columns:
            conn.execute("ALTER TABLE kb_messages ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
            conn.execute("DROP INDEX IF EXISTS idx_kb_messages_dedup")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_messages_dedup ON kb_messages(content_hash, client, role)"
            )
            # Trigger a full message backfill: assistant replies are net-new,
            # user messages dedup idempotently under the new (hash, client, role) key.
            conn.execute("UPDATE kb_sources SET messages_done = 0")
```

3. `upsert_message()`：签名加 `role: str = "user"`（放在 `seen_at` 之前，保持 keyword-only）；去重查询改为
   `SELECT id FROM kb_messages WHERE content_hash=? AND client=? AND role=?`（参数加 `role`）；
   INSERT 列清单与 VALUES 加 `role`。docstring 改为 "dedup on (content_hash, client, role)"。

4. `indexed_messages()`：SELECT 列表加 `role`。

5. `stats()`：`"messages"` 保留总数，追加：

```python
            by_role = {
                row["role"]: row["c"]
                for row in conn.execute("SELECT role, COUNT(*) c FROM kb_messages GROUP BY role")
            }
```

   返回 dict 增加 `"messages_user": int(by_role.get("user", 0))` 和 `"messages_assistant": int(by_role.get("assistant", 0))`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/prompt_kb/test_store_messages.py tests/prompt_kb/test_store.py -q`
Expected: 全部 PASS（含既有测试）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/store.py tests/prompt_kb/test_store_messages.py
git commit -m "feat(prompt_kb): kb_messages 新增 role 列并扩展去重键，迁移触发全量回填"
```

---

### Task 2: messages.py——assistant 回复正文抽取

**Files:**
- Modify: `claude_tap/prompt_kb/messages.py`
- Test: `tests/prompt_kb/test_messages.py`

**Interfaces:**
- Consumes: `infer_provider(record)`（按 request path 判定 provider）、`_content_text`、`_split_message`（均已存在）
- Produces: `MIN_ASSISTANT_CHARS = 20`；`AssistantMessage(record_index: int, message_index: int, timestamp: str, text: str)`；`extract_assistant_messages(records: list[dict]) -> list[AssistantMessage]`（Task 3 调用）

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_messages.py` 追加：

```python
from claude_tap.prompt_kb.messages import (
    MIN_ASSISTANT_CHARS,
    extract_assistant_messages,
)


def _assistant_record(resp_body, path="/v1/messages", timestamp="2026-08-10T01:00:00Z"):
    return {
        "timestamp": timestamp,
        "request": {"method": "POST", "path": path, "body": {}},
        "response": {"status": 200, "body": resp_body},
    }


LONG = "this is a sufficiently long assistant reply explaining the fix"


def test_anthropic_assistant_text_only():
    records = [
        _assistant_record(
            {
                "content": [
                    {"type": "thinking", "thinking": "let me think about this problem"},
                    {"type": "text", "text": LONG},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ]
            }
        )
    ]
    msgs = extract_assistant_messages(records)
    assert [m.text for m in msgs] == [LONG]
    assert msgs[0].record_index == 0 and msgs[0].message_index == 0
    assert msgs[0].timestamp == "2026-08-10T01:00:00Z"


def test_openai_chat_assistant_text():
    records = [
        _assistant_record(
            {"choices": [{"message": {"role": "assistant", "content": LONG}}]},
            path="/v1/chat/completions",
        )
    ]
    assert [m.text for m in extract_assistant_messages(records)] == [LONG]


def test_openai_responses_assistant_text():
    records = [
        _assistant_record(
            {
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": LONG}],
                    },
                    {"type": "function_call", "name": "shell", "arguments": "{}"},
                ]
            },
            path="/v1/responses",
        )
    ]
    assert [m.text for m in extract_assistant_messages(records)] == [LONG]


def test_gemini_assistant_text_skips_thought_parts():
    records = [
        _assistant_record(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "hidden reasoning", "thought": True},
                                {"text": LONG},
                            ]
                        }
                    }
                ]
            },
            path="/v1beta/models/gemini-2.0-flash:generateContent",
        )
    ]
    assert [m.text for m in extract_assistant_messages(records)] == [LONG]


def test_short_and_empty_replies_dropped():
    records = [
        _assistant_record({"content": [{"type": "text", "text": "好的"}]}),
        _assistant_record({"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}),
        {"timestamp": "t", "request": {"path": "/v1/messages"}, "response": {"status": 200}},  # no body
    ]
    assert extract_assistant_messages(records) == []
    assert MIN_ASSISTANT_CHARS == 20


def test_long_reply_split_into_pieces():
    piece = "paragraph with enough words to pass the minimum length filter. "
    records = [_assistant_record({"content": [{"type": "text", "text": (piece * 60)}]})]
    msgs = extract_assistant_messages(records)
    assert len(msgs) > 1
    assert [m.message_index for m in msgs] == list(range(len(msgs)))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/prompt_kb/test_messages.py -x -q`
Expected: FAIL（`cannot import name 'MIN_ASSISTANT_CHARS'`）

- [ ] **Step 3: 实现**

`messages.py` 追加（模块 docstring 首行改为 "Extract user and assistant messages from trace records for semantic session search."）：

```python
MIN_ASSISTANT_CHARS = 20  # short acknowledgements carry no search value


@dataclass(frozen=True)
class AssistantMessage:
    record_index: int
    message_index: int
    timestamp: str
    text: str


def extract_assistant_messages(records: list[dict[str, Any]]) -> list[AssistantMessage]:
    """Extract assistant reply text from response bodies.

    Only visible text is kept: thinking blocks, tool calls, and replies
    shorter than MIN_ASSISTANT_CHARS are dropped. Malformed/missing response
    bodies are skipped silently (pure reads; nothing transient to retry).
    """
    out: list[AssistantMessage] = []
    for record_index, record in enumerate(records):
        body = _response_body(record)
        if not body:
            continue
        text = _assistant_text(infer_provider(record), body).strip()
        if len(text) < MIN_ASSISTANT_CHARS:
            continue
        for message_index, piece in enumerate(_split_message(text)):
            out.append(
                AssistantMessage(
                    record_index=record_index,
                    message_index=message_index,
                    timestamp=str(record.get("timestamp") or ""),
                    text=piece,
                )
            )
    return out


def _response_body(record: dict[str, Any]) -> dict[str, Any]:
    resp = record.get("response") if isinstance(record.get("response"), dict) else {}
    body = resp.get("body")
    return body if isinstance(body, dict) else {}


def _assistant_text(provider: str, body: dict[str, Any]) -> str:
    if provider == "anthropic":
        content = body.get("content")
        if not isinstance(content, list):
            return ""
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        return "\n\n".join(t.strip() for t in texts if t.strip())
    if provider == "openai":
        return _openai_assistant_text(body)
    if provider == "gemini":
        candidates = body.get("candidates")
        if not isinstance(candidates, list):
            return ""
        texts: list[str] = []
        for cand in candidates:
            content = cand.get("content") if isinstance(cand, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            texts.extend(
                part["text"]
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
            )
        return "\n\n".join(t.strip() for t in texts if t.strip())
    return ""


def _openai_assistant_text(body: dict[str, Any]) -> str:
    texts: list[str] = []
    choices = body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                texts.append(_content_text(message.get("content")))
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str)
                )
    return "\n\n".join(t.strip() for t in texts if t.strip())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/prompt_kb/test_messages.py -q`
Expected: 全部 PASS（含既有 user 消息测试）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/messages.py tests/prompt_kb/test_messages.py
git commit -m "feat(prompt_kb): 新增 assistant 回复正文抽取（四 provider，过滤 thinking/工具调用/短回复）"
```

---

### Task 3: extract.py——抽取管线接入 assistant 回复

**Files:**
- Modify: `claude_tap/prompt_kb/extract.py`
- Test: `tests/prompt_kb/test_extract.py`

**Interfaces:**
- Consumes: Task 2 的 `extract_assistant_messages`；Task 1 的 `upsert_message(role=...)`
- Produces: `extract_messages(...) -> dict[str, int]`（`{"user": n, "assistant": m}`，替代原 `-> int`）；回填与懒索引机制不变（Task 1 的 messages_done 重置已触发全量重抽）

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_extract.py` 追加：

```python
def test_extract_messages_stores_assistant_replies(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    records = [
        {
            "timestamp": "2026-08-10T01:00:00Z",
            "request": {
                "method": "POST", "path": "/v1/messages",
                "body": {"model": "k3-256k",
                         "messages": [{"role": "user", "content": "how do I fix the race condition"}]},
            },
            "response": {
                "status": 200,
                "body": {"content": [
                    {"type": "thinking", "thinking": "reasoning here"},
                    {"type": "text", "text": "use a lock ordering protocol to fix the race condition"},
                ]},
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
```

同时把既有 `test_extract_messages_model_from_body` 中的 `assert created == 1` 改为
`assert created["user"] == 1`（返回类型从 int 变为 dict）。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/prompt_kb/test_extract.py -x -q`
Expected: FAIL（新测试 `created == {...}` 不成立 / 旧断言类型不符）

- [ ] **Step 3: 实现**

`extract.py`：import 加 `extract_assistant_messages`；`extract_messages` 改为：

```python
def extract_messages(store: KbStore, *, session_id: str, client: str, records: list[dict[str, Any]]) -> dict[str, int]:
    """Store user messages and assistant replies into kb_messages.

    Returns {"user": n, "assistant": m} counts of newly created
    (non-deduped) message chunks per role.
    """
    created = {"user": 0, "assistant": 0}
    for role, messages in (
        ("user", extract_user_messages(records)),
        ("assistant", extract_assistant_messages(records)),
    ):
        for msg in messages:
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
                role=role,
            )
            if was_created:
                created[role] += 1
    return created
```

（`upsert_message` 调用加 `role=role`，其余参数不变；`UserMessage`/`AssistantMessage` 字段同名可直接共用循环。）

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/prompt_kb/test_extract.py tests/prompt_kb/test_index.py -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/extract.py tests/prompt_kb/test_extract.py
git commit -m "feat(prompt_kb): 会话抽取管线接入 assistant 回复入库"
```

---

### Task 4: chunk.py 样板黑名单 + store 存量清除

**Files:**
- Modify: `claude_tap/prompt_kb/chunk.py`、`claude_tap/prompt_kb/store.py`
- Test: `tests/prompt_kb/test_chunk.py`、`tests/prompt_kb/test_store.py`

**Interfaces:**
- Consumes: `_heading_sections`/`_merge_small` 的 (title, body) 结构
- Produces: `BOILERPLATE_TITLES: frozenset[str]`（chunk.py，store.py 顶层 import 它做存量清除）；`kb_meta` 键 `boilerplate_purged`（一次性标记）

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_chunk.py` 追加：

```python
def test_boilerplate_sections_skipped():
    from claude_tap.prompt_kb.chunk import BOILERPLATE_TITLES
    env = "# Environment\n" + ("working directory and platform details. " * 20)
    style = "# Style Guide\n" + ("write elegant prose with care. " * 20)
    snapshot = SimpleNamespace(
        system_prompt=f"{env}\n\n{style}", developer_prompt="", tools=[]
    )
    chunks = chunk_snapshot(snapshot)
    assert chunks and all(c.title != "Environment" for c in chunks)
    assert any(c.title == "Style Guide" for c in chunks)
    assert "environment" in BOILERPLATE_TITLES
```

（文件顶部按需 `from types import SimpleNamespace`，并确认已 import `chunk_snapshot`。）

`tests/prompt_kb/test_store.py` 追加：

```python
def test_boilerplate_chunks_purged_once(tmp_path):
    db = tmp_path / "kb.sqlite3"
    store = KbStore(db)
    sid, _ = store.upsert_snapshot(
        content_hash="h1", client="claude", provider="anthropic", model="k3",
        system_prompt="s", developer_prompt="", tools_json="[]", seen_at="t",
    )
    store.replace_chunks(sid, [("prompt_section", "Environment", "boilerplate"),
                               ("prompt_section", "Style", "real content")])
    store.set_meta("boilerplate_purged", "0")  # 模拟未清除状态
    KbStore(db)  # 重开触发一次性清除（标记非 "1" 即执行）
    rows = store.indexed_chunks()
    assert [r["title"] for r in rows] == []  # 未索引不在 indexed_chunks
    with sqlite3.connect(db) as conn:
        titles = [r[0] for r in conn.execute("SELECT title FROM kb_chunks").fetchall()]
        purged = conn.execute("SELECT value FROM kb_meta WHERE key='boilerplate_purged'").fetchone()[0]
    assert titles == ["Style"]
    assert purged == "1"
```

（文件顶部确认已 import `sqlite3`。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/prompt_kb/test_chunk.py tests/prompt_kb/test_store.py -x -q`
Expected: FAIL（`BOILERPLATE_TITLES` 不存在 / 清除未发生）

- [ ] **Step 3: 实现**

`chunk.py` 顶部（常量区）追加：

```python
# Harness-injected template sections: present in nearly every snapshot, low
# information value, and their git/shell/CLI vocabulary poisons similarity.
BOILERPLATE_TITLES = frozenset({"environment", "context management", "harness", "session-specific guidance"})
```

`_split_prompt()` 的循环改为：

```python
    for title, body in merged:
        if title.strip().lower() in BOILERPLATE_TITLES:
            continue
        for piece in _split_long(title, body):
            chunks.append(Chunk(kind="prompt_section", title=piece[0], text=piece[1]))
```

`store.py` 顶部 import 加 `from claude_tap.prompt_kb.chunk import BOILERPLATE_TITLES`（chunk.py 只依赖 prompt_snapshot，无循环引用）；`_migrate()` 末尾追加：

```python
        purged = conn.execute("SELECT value FROM kb_meta WHERE key='boilerplate_purged'").fetchone()
        if purged is None or purged["value"] != "1":
            placeholders = ",".join("?" for _ in BOILERPLATE_TITLES)
            conn.execute(
                f"DELETE FROM kb_chunks WHERE lower(title) IN ({placeholders})",
                sorted(BOILERPLATE_TITLES),
            )
            conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('boilerplate_purged', '1')")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/prompt_kb/test_chunk.py tests/prompt_kb/test_store.py -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/chunk.py claude_tap/prompt_kb/store.py tests/prompt_kb/test_chunk.py tests/prompt_kb/test_store.py
git commit -m "feat(prompt_kb): 索引期剔除 harness 样板 section 并一次性清除存量 chunk"
```

---

### Task 5: search.py——跨快照去重、rel_delta 相对阈值、role 透传

**Files:**
- Modify: `claude_tap/prompt_kb/search.py`
- Test: `tests/prompt_kb/test_search.py`、`tests/prompt_kb/test_search_messages.py`

**Interfaces:**
- Consumes: Task 1 的 `indexed_messages()` 行 `role` 键
- Produces: `search(..., rel_delta: float = 0.05)`、`search_messages(..., rel_delta: float = 0.05)`；`MessageHit` 新增 `role: str` 字段（Task 6 的 live/mcp/cli 依赖）；`SnapshotResult.session_count` 在去重折叠时累加被折叠快照的计数

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_search.py` 追加：

```python
def _seed_overlap(store: KbStore) -> None:
    """三个快照：A 与 query 全量重叠（top），B/C 部分重叠（长尾）。"""
    for client, model, seen, chunks in [
        ("codex", "gpt-5", "2026-08-01T00:00:00Z",
         [("prompt_section", "Guide", "alpha beta gamma delta epsilon zeta runs fast")]),
        ("claude-code", "claude", "2026-08-02T00:00:00Z",
         [("prompt_section", "Notes", "alpha only shares one token here")]),
        ("claude-code", "claude", "2026-08-03T00:00:00Z",
         [("prompt_section", "Style", "write elegant prose")]),
    ]:
        sid, _ = store.upsert_snapshot(
            content_hash=f"h-{client}-{seen}", client=client, provider="p", model=model,
            system_prompt="s", developer_prompt="", tools_json="[]", seen_at=seen,
        )
        store.replace_chunks(sid, chunks)


def test_rel_delta_trims_long_tail(trace_db):
    store = KbStore.default()
    _seed_overlap(store)
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    results = search(store, embedder, "alpha beta gamma delta epsilon zeta")
    assert len(results) == 1  # top=0.938，0.667/0.510 被 rel_delta=0.05 截断
    assert results[0].client == "codex"
    everything = search(store, embedder, "alpha beta gamma delta epsilon zeta", rel_delta=1.0)
    assert len(everything) == 3


def test_identical_chunks_folded_across_snapshots(trace_db):
    store = KbStore.default()
    for client, seen in [("codex", "2026-08-01T00:00:00Z"), ("claude-code", "2026-08-02T00:00:00Z")]:
        sid, _ = store.upsert_snapshot(
            content_hash=f"h-{client}", client=client, provider="p", model="m",
            system_prompt="s", developer_prompt="", tools_json="[]", seen_at=seen,
        )
        store.replace_chunks(sid, [("tool", "shell", "sandbox shell command runner")])
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    results = search(store, embedder, "shell sandbox", rel_delta=1.0)
    assert len(results) == 1  # 同分平局归最新 last_seen 的快照
    assert results[0].client == "claude-code"
    assert results[0].session_count == 2  # 被折叠快照的计数累加
```

`tests/prompt_kb/test_search_messages.py` 追加：

```python
def test_message_hit_carries_role(seeded):
    store, embedder = seeded
    store.upsert_message(
        session_id="s1", record_index=9, message_index=0, client="claude",
        model="k3", timestamp="2026-08-02T00:00:00Z", content_hash="h9",
        text="the race condition fix is a strict lock ordering protocol",
        seen_at="t", role="assistant",
    )
    index_pending(store, embedder)
    results = search_messages(store, embedder, "race condition lock", rel_delta=1.0)
    roles = {h.role for g in results for h in g.hits}
    assert "assistant" in roles and "user" in roles
```

（`test_search_messages.py` 既有 fixture 名为 `seeded`，沿用；其 seeded 查询返回多组时把 `rel_delta=1.0` 传入以避免相对阈值干扰角色断言。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/prompt_kb/test_search.py tests/prompt_kb/test_search_messages.py -x -q`
Expected: FAIL（`rel_delta` 参数不存在 / `MessageHit` 无 role）

- [ ] **Step 3: 实现**

`search.py` 修改：

1. `MessageHit` dataclass 加字段 `role: str`。

2. `search()` 签名加 `rel_delta: float = 0.05`；打分后的过滤/分组段替换为：

```python
    scored = [(row, float(score)) for row, score in zip(rows, scores) if score > min_score]
    if not scored:
        return []
    top = max(score for _, score in scored)
    # Relative floor: keep hits close to the best score (absolute scores are
    # not calibrated; rel_delta=1.0 disables the floor entirely).
    kept = [(row, score) for row, score in scored if score > top - rel_delta or score == top]
    groups: dict[int, SnapshotResult] = {}
    for row, score, bonus_sessions in _dedup_across_snapshots(kept):
        group = groups.setdefault(
            row["snapshot_id"],
            SnapshotResult(
                snapshot_id=row["snapshot_id"],
                client=row["client"],
                model=row["model"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                session_count=row["session_count"],
            ),
        )
        group.session_count += bonus_sessions
        group.hits.append(
            SearchHit(kind=row["kind"], title=row["title"] or "", text=row["text"], score=score)
        )
```

（分组排序与每组 top-3 截断逻辑不变。）

3. 新增私有函数：

```python
def _dedup_across_snapshots(scored: list[tuple[Any, float]]) -> list[tuple[Any, float, int]]:
    """Collapse (kind, title, text) duplicates that recur across snapshots.

    Keeps the highest-scoring occurrence (ties: newest last_seen); dropped
    snapshots' session counts are summed into a bonus so the surviving
    group's session_count still reflects total exposure.
    """
    best: dict[tuple[str, str, str], list] = {}
    for row, score in scored:
        key = (row["kind"], row["title"] or "", row["text"])
        entry = best.setdefault(key, [row, score, 0])
        if (score, row["last_seen"]) > (entry[1], entry[0]["last_seen"]):
            entry[2] += int(entry[0]["session_count"])  # demote previous winner
            entry[0], entry[1] = row, score
        elif entry[0] is not row:
            entry[2] += int(row["session_count"])  # drop the challenger
    return [(entry[0], entry[1], entry[2]) for entry in best.values()]
```

注意：同一 row 首次进入时 `entry[0] is row` 为 True，不加分——上述 `elif` 保证不重复计数。

4. `search_messages()` 签名加 `rel_delta: float = 0.05`；过滤段同样改为 scored/top/kept 三步（不做跨快照去重——消息按 content_hash 已天然去重）；`MessageHit` 构造改为：

```python
        group.hits.append(MessageHit(text=row["text"], timestamp=row["timestamp"], score=float(score), role=row["role"]))
```

5. 文件顶部 `from typing import Any`（如尚无）。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/prompt_kb/test_search.py tests/prompt_kb/test_search_messages.py -q`
Expected: 全部 PASS（含既有测试——既有断言不受 rel_delta 默认值影响）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/search.py tests/prompt_kb/test_search.py tests/prompt_kb/test_search_messages.py
git commit -m "feat(prompt_kb): 检索新增跨快照去重、rel_delta 相对阈值与消息角色透传"
```

---

### Task 6: live.py / mcp_server.py / cli.py——role 与 rel_delta 贯通

**Files:**
- Modify: `claude_tap/live.py:650-660`（messages JSON 区）、`claude_tap/prompt_kb/mcp_server.py`、`claude_tap/prompt_kb/cli.py`
- Test: `tests/prompt_kb/test_kb_api.py`、`tests/prompt_kb/test_mcp_server.py`、`tests/prompt_kb/test_cli.py`

**Interfaces:**
- Consumes: Task 5 的 `MessageHit.role`、`search(..., rel_delta=)`、`search_messages(..., rel_delta=)`
- Produces: HTTP `/api/kb/search` messages hits 含 `role`；MCP `kb_search(query, client, kind, limit, min_score, rel_delta)` 响应 messages 分区在前且 hits 含 `role`；CLI `claude-tap kb search --rel-delta`，messages 小节先打印且带 `[user]`/`[assistant]` 前缀

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_kb_api.py`：在既有 messages 断言的测试中追加（或新增等价测试）：

```python
    assert all("role" in hit for g in data["messages"] for hit in g["hits"])
```

`tests/prompt_kb/test_mcp_server.py` 追加：

```python
def test_kb_search_messages_first_and_roles(monkeypatch):
    """kb_search 响应 messages 分区在前，hit 带 role，rel_delta 透传。"""
    calls = {}

    class FakeGroup:
        session_id = "s1"
        client = "claude"
        model = "k3"
        hits = [type("H", (), {"text": "t", "timestamp": "ts", "score": 0.9, "role": "assistant"})()]

    def fake_search_messages(store, embedder, query, **kw):
        calls.update(kw)
        return [FakeGroup()]

    monkeypatch.setattr("claude_tap.prompt_kb.mcp_server.search", lambda *a, **kw: [])
    monkeypatch.setattr("claude_tap.prompt_kb.mcp_server.search_messages", fake_search_messages)
    monkeypatch.setattr("claude_tap.prompt_kb.mcp_server.index_pending", lambda *a, **kw: None)
    monkeypatch.setattr("claude_tap.prompt_kb.mcp_server._get_ctx", lambda: (object(), object()))
    from claude_tap.prompt_kb.mcp_server import kb_search

    result = kb_search("q", rel_delta=0.1)
    assert list(result.keys())[0] == "messages"
    assert result["messages"][0]["hits"][0]["role"] == "assistant"
    assert calls["rel_delta"] == 0.1
```

`tests/prompt_kb/test_cli.py`：把既有 search 输出断言调整为 messages 小节在前——新增：

```python
def test_cli_search_prints_messages_first_with_roles(tmp_path, capsys, monkeypatch):
    ...  # 复用既有 CLI 测试的 store/embedder monkeypatch 模式，
    # 断言 stdout 中 "messages:" 出现在首个 "[1]" 快照组之前，
    # 且消息行含 "[user]" 前缀
```

（实现时参照该文件既有 fixture；若既有测试已断言顺序则直接改断言。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/prompt_kb/test_kb_api.py tests/prompt_kb/test_mcp_server.py tests/prompt_kb/test_cli.py -x -q`
Expected: FAIL（role 键缺失 / 顺序不符 / rel_delta 未透传）

- [ ] **Step 3: 实现**

`live.py` `_handle_kb_search`：messages JSON 的 hits 推导改为：

```python
                            "hits": [
                                {"text": h.text, "timestamp": h.timestamp, "score": h.score, "role": h.role}
                                for h in group.hits
                            ],
```

（响应 dict 的 `"messages"` 键移到 `"results"` 之前。）

`mcp_server.py` `kb_search`：签名加 `rel_delta: float = 0.05`（docstring Args 加一行：`rel_delta: Relative score floor; hits below top_score - rel_delta are dropped. 1.0 disables.`）；两个 search 调用都传 `rel_delta=rel_delta`；返回 dict 改为 messages 在前：

```python
    return {
        "messages": [
            {
                "session_id": g.session_id,
                "client": g.client,
                "model": g.model,
                "hits": [{"text": h.text, "timestamp": h.timestamp, "score": h.score, "role": h.role} for h in g.hits],
            }
            for g in message_groups
        ],
        "chunks": [
            {
                "client": g.client,
                "model": g.model,
                "first_seen": g.first_seen,
                "last_seen": g.last_seen,
                "session_count": g.session_count,
                "hits": [{"kind": h.kind, "title": h.title, "text": h.text, "score": h.score} for h in g.hits],
            }
            for g in chunk_groups
        ],
    }
```

docstring 的 Returns 说明同步改为 `{"messages": [...], "chunks": [...]}`，并注明 messages 为主要依据、排在前面。

`cli.py`：search 子命令加 `search_parser.add_argument("--rel-delta", type=float, default=0.05)`；`kb_main` 中两个 search 调用传 `rel_delta=args.rel_delta`；**messages 打印块整体移到快照组循环之前**，命中行改为：

```python
                print(f"    [{hit.role}] score={hit.score:.3f} {hit.timestamp}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/prompt_kb/test_kb_api.py tests/prompt_kb/test_mcp_server.py tests/prompt_kb/test_cli.py tests/prompt_kb/test_mcp_stdio.py -q`
Expected: 全部 PASS（含 stdio 冒烟回归）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/live.py claude_tap/prompt_kb/mcp_server.py claude_tap/prompt_kb/cli.py tests/prompt_kb/test_kb_api.py tests/prompt_kb/test_mcp_server.py tests/prompt_kb/test_cli.py
git commit -m "feat(prompt_kb): API/MCP/CLI 贯通 role 标注与 rel_delta，messages 分区前置"
```

---

### Task 7: dashboard.html——会话分区前置 + 角色徽标

**Files:**
- Modify: `claude_tap/dashboard.html`（i18n en ~1495 行 / zh ~1637 行；`kbRenderFiltered` 2573-2618 行；`kbRenderMessageCard` 2672-2702 行；`kbRenderHit` 2704-2711 行）
- Test: `tests/prompt_kb/test_kb_render_logic.py`（Python 镜像）、`tests/prompt_kb/test_kb_render_browser.py`（Playwright）

**Interfaces:**
- Consumes: Task 6 的 `/api/kb/search` messages hits `role` 字段
- Produces: i18n 键 `kb_role_user`（en "Question" / zh "提问"）、`kb_role_assistant`（en "Answer" / zh "回答"）、`kb_chunks_section`（en "Prompt snapshots" / zh "Prompt 快照"）；纯函数 `kbHitBadge(hit) -> [className, i18nKey]`（供镜像测试）

- [ ] **Step 1: 写失败测试**

`tests/prompt_kb/test_kb_render_logic.py` 追加镜像与测试：

```python
def hit_badge(hit):
    kind = hit.get("kind", "")
    if kind == "tool":
        return ("tool", "kb_kind_tool")
    if kind == "user_message":
        return ("message", "kb_role_user")
    if kind == "assistant_message":
        return ("message", "kb_role_assistant")
    return ("prompt", "kb_kind_prompt")


def test_hit_badge_roles():
    assert hit_badge({"kind": "user_message"}) == ("message", "kb_role_user")
    assert hit_badge({"kind": "assistant_message"}) == ("message", "kb_role_assistant")
    assert hit_badge({"kind": "tool"}) == ("tool", "kb_kind_tool")
```

`tests/prompt_kb/test_kb_render_browser.py` 追加：

```python
def test_messages_section_renders_before_chunks(page):
    page.evaluate(
        "renderKbResults(window.__KB_TEST_GROUPS, window.__KB_TEST_MESSAGES)"
    )
    children = page.eval_on_selector_all(
        "#kb-results > *", "els => els.map(e => e.className)"
    )
    assert any("kb-section-title" in c for c in children)
    first_section = next(c for c in children if "kb-section-title" in c)
    idx_msg = children.index(first_section)
    idx_summary = next((i for i, c in enumerate(children) if "kb-summary" in c), len(children))
    assert idx_msg < idx_summary  # 会话分区在快照分区之前

def test_role_badges_rendered(page):
    page.evaluate(
        "renderKbResults(window.__KB_TEST_GROUPS, window.__KB_TEST_MESSAGES)"
    )
    badges = page.eval_on_selector_all(
        ".kb-message-group .kb-hit-kind", "els => els.map(e => e.textContent)"
    )
    assert "提问" in badges and "回答" in badges
```

同时在 `window.__KB_TEST_MESSAGES` 注入 fixture（若无）：
`[{session_id: "s1", client: "claude", model: "k3", hits: [{text: "q", timestamp: "t1", score: 0.9, role: "user"}, {text: "a", timestamp: "t2", score: 0.85, role: "assistant"}]}]`。
（browser 测试语言环境为中文；若页面默认 en 则断言 "Question"/"Answer"。实现时先跑一次看实际语言再定断言文本。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/prompt_kb/test_kb_render_logic.py tests/prompt_kb/test_kb_render_browser.py -x -q`
Expected: FAIL（`kb_role_user` 未定义 / 顺序不符）

- [ ] **Step 3: 实现**

`dashboard.html` 修改：

1. i18n en 字典（~1495 行 `kb_kind_tool` 附近）加：
   `kb_role_user: "Question", kb_role_assistant: "Answer", kb_chunks_section: "Prompt snapshots",`
   zh 字典（~1637 行对应位置）加：
   `kb_role_user: "提问", kb_role_assistant: "回答", kb_chunks_section: "Prompt 快照",`
   （注意给前一行末尾补逗号。）

2. 新增纯函数（`kbFilterMessageGroups` 旁）：

```javascript
function kbHitBadge(hit) {
  if (hit.kind === "tool") return ["tool", "kb_kind_tool"];
  if (hit.kind === "user_message") return ["message", "kb_role_user"];
  if (hit.kind === "assistant_message") return ["message", "kb_role_assistant"];
  return ["prompt", "kb_kind_prompt"];
}
```

3. `kbRenderHit` 的 badge 两行改为：

```javascript
  const [badgeClass, badgeKey] = kbHitBadge(hit);
  badge.className = `kb-hit-kind ${badgeClass}`;
  badge.textContent = t(badgeKey);
```

4. `kbRenderMessageCard` 的 hits 渲染改为传 role：

```javascript
  for (const hit of group.hits) card.appendChild(kbRenderHit({
    kind: hit.role === "assistant" ? "assistant_message" : "user_message",
    title: "",
    text: hit.text,
    score: hit.score,
  }));
```

5. `kbRenderFiltered`：`if (msgGroups.length) {...}` 块整体移到 `if (filtered.length) {...}` 块之前；`if (filtered.length)` 块内 summary 之前插入快照分区标题：

```javascript
    const chunkSection = document.createElement("div");
    chunkSection.className = "kb-section-title";
    chunkSection.textContent = t("kb_chunks_section");
    container.appendChild(chunkSection);
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/prompt_kb/test_kb_render_logic.py tests/prompt_kb/test_kb_render_browser.py tests/prompt_kb/test_kb_page.py -q`
Expected: 全部 PASS（含既有锚点测试）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/dashboard.html tests/prompt_kb/test_kb_render_logic.py tests/prompt_kb/test_kb_render_browser.py
git commit -m "feat(prompt_kb): 知识库页会话分区前置并新增提问/回答角色徽标"
```

---

### Task 8: 全量回归 + 真实数据前后对比验证

**Files:**
- 无代码改动（验证任务；发现问题回到对应任务修复）

**Interfaces:**
- Consumes: Task 1-7 全部产出

- [ ] **Step 1: 全量回归**

Run: `.venv/bin/pytest tests/ --ignore=tests/test_e2e.py -q`
Expected: 全部 PASS

Run: `.venv/bin/pytest tests/test_e2e.py -v --timeout=120`
Expected: 全部 PASS

- [ ] **Step 2: 真实库迁移与回填**

```bash
.venv/bin/python -c "
from claude_tap.prompt_kb.store import KbStore
s = KbStore.default()
print(s.stats())
"
```

Expected: 输出含 `messages_user=488 messages_assistant=0`（回填尚未跑）；无迁移异常。

随后触发回填与索引（后台渐进，或一次性）：

```bash
.venv/bin/claude-tap kb reindex  # 或启动 dashboard 让懒索引线程跑
```

等待后再次 `kb status`，Expected: `messages_assistant` 显著大于 0、`pending=0`。

- [ ] **Step 3: 前后对比——复跑 2026-08-12 实测的 7 组查询**

用 MCP `kb_search`（或 `.venv/bin/claude-tap kb search`）复跑：

1. `哪个 CLI 有沙箱 shell 工具`
2. `which CLI has a sandboxed shell tool`
3. `怎么写 commit message`
4. `前端页面截图验证`
5. `取消定时任务 cron`（kind=tool）
6. `沙箱 sandbox 执行命令`（min_score=0.86）
7. `Playwright 浏览器截图验证`（kind=prompt_section）

验收标准：
- 查询 1/2：chunks 区不再出现 `Environment`/`Context management`/`Harness` 标题的命中；`Bash` 工具排名上升
- 全部查询：无跨快照重复 group（同内容只出现一次）
- 查询 5：CronDelete 仍居首（回归保护）
- messages 区命中带 role，且能搜到 assistant 的回答内容（用查询 4 验证：命中 Playwright 测试会话的回答）
- 长尾噪声减少（每组 hits 数 ≤3 且低分长尾被截断）

- [ ] **Step 4: 结果记录与提交**

将对比结论追加到 spec 文档末尾（`## 实施验证` 一节：日期、验收结果、遗留问题），然后：

```bash
git add docs/superpowers/specs/2026-08-12-chat-content-search-design.md
git commit -m "docs(spec): 聊天内容检索实施验证结果（7 组查询前后对比）"
```

---

## Self-Review 记录

- **Spec 覆盖**：§1→Task 1；§2→Task 2/3；§3→Task 4；§4→Task 5/6/7；§5→Task 1（messages_done 重置）/Task 2（容错注释）/Task 8（回填验证）；§6→各任务测试 + Task 8。全覆盖。
- **类型一致性**：`extract_assistant_messages`/`AssistantMessage`/`MIN_ASSISTANT_CHARS`（Task 2 产出，Task 3 消费）；`role`（Task 1→3→5→6→7）；`rel_delta`（Task 5→6）；`BOILERPLATE_TITLES`（Task 4 chunk 产出、store 消费）；`kbHitBadge`（Task 7 JS 与 Python 镜像同名）。一致。
- **既有测试兼容**：`test_search_respects_min_score`（min_score=0.999 → scored 为空 → 返回 []，断言仍成立）；`test_extract.py` 旧断言在 Task 3 Step 1 中同步修改；dashboard 既有锚点测试在 Task 7 Step 4 回归。
