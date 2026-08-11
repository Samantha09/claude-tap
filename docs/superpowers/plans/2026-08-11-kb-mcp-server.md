# 知识库 MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `claude-tap mcp` stdio MCP server，把 prompt 知识库以 `kb_search` / `kb_status` 两个只读工具暴露给 MCP 客户端（如 Claude Code）。

**Architecture:** 新增单文件模块 `claude_tap/prompt_kb/mcp_server.py`，用官方 `mcp` SDK 的 FastMCP 做协议层；工具函数是 `index_pending()` + `search()` + `search_messages()` 的薄包装，不含新检索逻辑。`cli.py` 的 `main_entry` 仿照 `kb` 分支新增 `mcp` 分发。

**Tech Stack:** Python 3.11+、`mcp>=1.0`（FastMCP，新 extra `claude-tap[mcp]`）、现有 `[rag]` 栈（sentence-transformers + numpy）、SQLite、pytest + pytest-asyncio（asyncio_mode=auto）。

**Spec:** `docs/superpowers/specs/2026-08-11-kb-mcp-server-design.md`

## Global Constraints

- Commit message 一律中文（含 type/scope 前缀），代码/注释用英文。
- 测试命令：`uv run --extra dev pytest tests/ -x --timeout=60`（单文件同理换路径）。
- Lint：`uv run --extra dev ruff check .` 与 `uv run --extra dev ruff format --check .` 必须通过；行宽 120。
- 未装 `[mcp]` 或 `[rag]` 时，trace 录制、dashboard、kb CLI 完全不受影响（优雅降级原则）。
- MCP 工具函数签名只允许出现 JSON 可序列化参数（暴露即 MCP schema）；store/embedder 一律走模块级 `_get_ctx()`，不进签名。
- 不新增检索逻辑；`search()` / `search_messages()` / `index_pending()` / `store.stats()` 原样复用。

---

### Task 1: `mcp` optional extra 打包声明

**Files:**
- Modify: `pyproject.toml:61-64`（`rag = [...]` extra 之后）
- Test: `tests/prompt_kb/test_packaging.py`

**Interfaces:**
- Consumes: 无
- Produces: `pyproject.toml` 中 `project.optional-dependencies.mcp = ["mcp>=1.0"]`，供后续任务与用户安装 `claude-tap[mcp,rag]` 使用。

- [ ] **Step 1: 写失败测试**

在 `tests/prompt_kb/test_packaging.py` 末尾追加：

```python
def test_mcp_extra_declared():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert "mcp" in extras
    assert any(dep.startswith("mcp") for dep in extras["mcp"])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run --extra dev pytest tests/prompt_kb/test_packaging.py::test_mcp_extra_declared -v`
Expected: FAIL（`KeyError: 'mcp'` 或 assert 失败）

- [ ] **Step 3: 在 pyproject.toml 声明 extra**

在 `rag = [...]` 块之后插入：

```toml
mcp = [
    "mcp>=1.0",
]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run --extra dev pytest tests/prompt_kb/test_packaging.py -v`
Expected: PASS（全部 4 条）

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/prompt_kb/test_packaging.py
git commit -m "feat(prompt_kb): 新增 mcp optional extra 声明"
```

---

### Task 2: `mcp_server.py` 核心工具（kb_search / kb_status happy path）

**Files:**
- Create: `claude_tap/prompt_kb/mcp_server.py`
- Test: `tests/prompt_kb/test_mcp_server.py`

**Interfaces:**
- Consumes:
  - `claude_tap.prompt_kb.embed.create_embedder(load_config()) -> Embedder`（`Embedder` 协议：`name: str`、`dimension: int`、`embed(texts)`、`embed_query(texts)`）
  - `claude_tap.prompt_kb.index.index_pending(store, embedder, *, batch_size=32) -> dict`
  - `claude_tap.prompt_kb.search.search(store, embedder, query, *, client=None, kind=None, limit=10, min_score=0.0) -> list[SnapshotResult]`；`SnapshotResult` 字段：`client/model/first_seen/last_seen/session_count`，`hits: list[SearchHit]`（`SearchHit.kind/title/text/score`）
  - `claude_tap.prompt_kb.search.search_messages(store, embedder, query, *, client=None, limit=10, min_score=0.0) -> list[SessionResult]`；`SessionResult` 字段：`session_id/client/model`，`hits: list[MessageHit]`（`MessageHit.text/timestamp/score`）
  - `claude_tap.prompt_kb.store.KbStore.default() -> KbStore`；`store.stats() -> dict`（键：`snapshots/chunks/pending/failed/indexed/messages`）；`store.get_meta("embedder_name") -> str | None`
- Produces（后续任务与 MCP 客户端依赖的确切签名）：
  - `kb_search(query: str, client: str | None = None, kind: Literal["tool", "prompt_section"] | None = None, limit: int = 10, min_score: float = 0.0) -> dict[str, Any]`，返回 `{"chunks": [...], "messages": [...]}`（错误时另有 `"error"` 键，Task 3 加入）
  - `kb_status() -> dict[str, Any]`，返回 `{**store.stats(), "embedder": str}`
  - `_get_ctx() -> tuple[KbStore, Embedder]`（模块级惰性单例，测试可 monkeypatch）

- [ ] **Step 1: 写失败测试**

创建 `tests/prompt_kb/test_mcp_server.py`：

```python
"""Unit tests for the MCP server tools (no stdio, no real model)."""

import pytest

pytest.importorskip("numpy")  # search depends on the [rag] extra

from claude_tap.prompt_kb import mcp_server  # noqa: E402
from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending  # noqa: E402
from claude_tap.prompt_kb.store import KbStore  # noqa: E402
from tests.prompt_kb.fake_embedder import FakeEmbedder  # noqa: E402


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    """Isolated KB store + fake embedder injected into the module-level context."""
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    monkeypatch.setattr(mcp_server, "_get_ctx", lambda: (store, embedder))
    monkeypatch.setattr(mcp_server.KbStore, "default", classmethod(lambda cls: store))
    return store, embedder


def _seed(store: KbStore) -> None:
    snap_id, _ = store.upsert_snapshot(
        content_hash="ha",
        client="codex",
        provider="openai",
        model="gpt-5",
        system_prompt="s",
        developer_prompt="",
        tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(snap_id, [("tool", "shell", "sandbox shell command runner")])
    store.upsert_message(
        session_id="s1",
        record_index=0,
        message_index=0,
        client="codex",
        model="gpt-5",
        timestamp="2026-08-01T00:00:00Z",
        content_hash="m0",
        text="how do I fix the race condition in the worker pool",
        seen_at="t",
    )
    index_pending(store, FakeEmbedder())


def test_kb_search_returns_chunks_section(ctx):
    store, _ = ctx
    _seed(store)
    result = mcp_server.kb_search("shell sandbox")
    assert result["chunks"][0]["client"] == "codex"
    assert result["chunks"][0]["model"] == "gpt-5"
    hit = result["chunks"][0]["hits"][0]
    assert hit["kind"] == "tool" and hit["title"] == "shell"
    assert hit["score"] > 0.5
    assert set(result) == {"chunks", "messages"}


def test_kb_search_returns_messages_section(ctx):
    store, _ = ctx
    _seed(store)
    result = mcp_server.kb_search("race condition lock")
    assert result["messages"][0]["session_id"] == "s1"
    hit = result["messages"][0]["hits"][0]
    assert "race condition" in hit["text"]
    assert hit["timestamp"] and hit["score"] > 0


def test_kb_search_indexes_pending_first(ctx):
    """New traces must be searchable without an explicit reindex."""
    store, _ = ctx
    _seed(store)
    # Arrives after the initial index: stays pending until kb_search runs.
    store.upsert_message(
        session_id="s2",
        record_index=1,
        message_index=0,
        client="codex",
        model="gpt-5",
        timestamp="2026-08-02T00:00:00Z",
        content_hash="m1",
        text="kubectl restart the deadlock pod",
        seen_at="t",
    )
    result = mcp_server.kb_search("deadlock pod")
    texts = [h["text"] for g in result["messages"] for h in g["hits"]]
    assert any("deadlock" in t for t in texts)


def test_kb_status(ctx):
    store, _ = ctx
    _seed(store)
    status = mcp_server.kb_status()
    assert status["embedder"] == "fake"
    for key in ("snapshots", "chunks", "pending", "failed", "indexed", "messages"):
        assert key in status
    assert status["snapshots"] == 1 and status["messages"] == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run --extra dev pytest tests/prompt_kb/test_mcp_server.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'claude_tap.prompt_kb.mcp_server'`）

- [ ] **Step 3: 实现 mcp_server.py（happy path，错误处理在 Task 3 加）**

创建 `claude_tap/prompt_kb/mcp_server.py`：

```python
"""MCP stdio server: expose the prompt KB to agents as read-only tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from claude_tap.prompt_kb.embed import create_embedder, load_config
from claude_tap.prompt_kb.index import index_pending
from claude_tap.prompt_kb.search import search, search_messages
from claude_tap.prompt_kb.store import KbStore

if TYPE_CHECKING:
    from claude_tap.prompt_kb.embed import Embedder

_ctx: tuple[KbStore, Embedder] | None = None


def _get_ctx() -> tuple[KbStore, Embedder]:
    """Lazily open the KB store and build the embedder.

    The embedding model loads on first tool call, not at server startup.
    """
    global _ctx
    if _ctx is None:
        _ctx = (KbStore.default(), create_embedder(load_config()))
    return _ctx


def kb_search(
    query: str,
    client: str | None = None,
    kind: Literal["tool", "prompt_section"] | None = None,
    limit: int = 10,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Search the local prompt knowledge base (prompts, tool definitions, user messages).

    Args:
        query: Natural-language query, e.g. "which CLI has a sandboxed shell tool".
        client: Optional client filter, e.g. "claude-code" or "codex".
        kind: Optional chunk-kind filter for the chunks section.
        limit: Max groups per section.
        min_score: Minimum cosine score (0-1) for a hit to be included.

    Returns:
        {"chunks": [...], "messages": [...]} grouped by snapshot / session.
    """
    store, embedder = _get_ctx()
    index_pending(store, embedder)
    chunk_groups = search(store, embedder, query, client=client, kind=kind, limit=limit, min_score=min_score)
    message_groups = search_messages(store, embedder, query, client=client, limit=limit, min_score=min_score)
    return {
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
        "messages": [
            {
                "session_id": g.session_id,
                "client": g.client,
                "model": g.model,
                "hits": [{"text": h.text, "timestamp": h.timestamp, "score": h.score} for h in g.hits],
            }
            for g in message_groups
        ],
    }


def kb_status() -> dict[str, Any]:
    """Report knowledge-base index size and embedder identity.

    Returns:
        store.stats() keys (snapshots/chunks/pending/failed/indexed/messages)
        plus "embedder" (indexed embedder name, or "none").
    """
    store = KbStore.default()
    return {**store.stats(), "embedder": store.get_meta("embedder_name") or "none"}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run --extra dev pytest tests/prompt_kb/test_mcp_server.py -v`
Expected: PASS（4 条）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/mcp_server.py tests/prompt_kb/test_mcp_server.py
git commit -m "feat(prompt_kb): 新增 MCP server 核心检索工具 kb_search/kb_status"
```

---

### Task 3: kb_search 错误处理与优雅降级

**Files:**
- Modify: `claude_tap/prompt_kb/mcp_server.py`（`kb_search` 函数与文件头部 import）
- Test: `tests/prompt_kb/test_mcp_server.py`

**Interfaces:**
- Consumes: Task 2 的 `kb_search` / `_get_ctx`；`claude_tap.prompt_kb.embed.EmbedderUnavailable`；`claude_tap.prompt_kb.search.ReindexRequired`
- Produces: `kb_search` 错误时返回 `{"error": str, "chunks": [], "messages": []}`，绝不向 MCP 客户端抛异常栈（`EmbedderUnavailable`、`ReindexRequired`、`sqlite3.OperationalError` 三种路径）

- [ ] **Step 1: 写失败测试**

在 `tests/prompt_kb/test_mcp_server.py` 追加（文件头部补 `import sqlite3` 和 `from claude_tap.prompt_kb.embed import EmbedderUnavailable`）：

```python
def test_kb_search_embedder_unavailable(monkeypatch):
    def _raise():
        raise EmbedderUnavailable("sentence-transformers is not installed")

    monkeypatch.setattr(mcp_server, "_get_ctx", _raise)
    result = mcp_server.kb_search("anything")
    assert "sentence-transformers" in result["error"]
    assert result["chunks"] == [] and result["messages"] == []


def test_kb_search_reindex_required(ctx, monkeypatch):
    store, _ = ctx
    _seed(store)
    other = FakeEmbedder()
    other.name = "other"  # instance attribute shadows the class attribute
    monkeypatch.setattr(mcp_server, "_get_ctx", lambda: (store, other))
    result = mcp_server.kb_search("shell sandbox")
    assert "reindex" in result["error"]
    assert result["chunks"] == [] and result["messages"] == []


def test_kb_search_survives_index_lock(ctx, monkeypatch):
    """A locked DB (dashboard lazy indexer) must not fail the search."""
    store, _ = ctx
    _seed(store)

    def _locked(store, embedder, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(mcp_server, "index_pending", _locked)
    result = mcp_server.kb_search("shell sandbox")
    assert "error" not in result
    assert result["chunks"][0]["hits"][0]["title"] == "shell"
```

注意：`FakeEmbedder.name` 是类属性，`other.name = "other"` 设实例属性遮蔽它，单测内有效。

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run --extra dev pytest tests/prompt_kb/test_mcp_server.py -v`
Expected: 3 条新测试 FAIL（`EmbedderUnavailable` / `ReindexRequired` / `OperationalError` 异常直接抛出，而非返回 error dict）

- [ ] **Step 3: 给 kb_search 加错误处理**

`claude_tap/prompt_kb/mcp_server.py` 头部 import 改为：

```python
import sqlite3
from typing import TYPE_CHECKING, Any, Literal

from claude_tap.prompt_kb.embed import EmbedderUnavailable, create_embedder, load_config
from claude_tap.prompt_kb.index import index_pending
from claude_tap.prompt_kb.search import ReindexRequired, search, search_messages
```

`kb_search` 函数体改为：

```python
    try:
        store, embedder = _get_ctx()
    except EmbedderUnavailable as exc:
        return {"error": f"embedder unavailable: {exc}", "chunks": [], "messages": []}
    try:
        index_pending(store, embedder)
    except sqlite3.OperationalError:
        pass  # dashboard's lazy indexer holds the write lock; search the stale index
    try:
        chunk_groups = search(store, embedder, query, client=client, kind=kind, limit=limit, min_score=min_score)
        message_groups = search_messages(store, embedder, query, client=client, limit=limit, min_score=min_score)
    except ReindexRequired as exc:
        return {"error": str(exc), "chunks": [], "messages": []}
```

（return dict 部分不变。）

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run --extra dev pytest tests/prompt_kb/test_mcp_server.py -v`
Expected: PASS（全部 7 条）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/mcp_server.py tests/prompt_kb/test_mcp_server.py
git commit -m "feat(prompt_kb): kb_search 错误路径降级为 error dict"
```

---

### Task 4: `main()` 入口与 `claude-tap mcp` CLI 分发

**Files:**
- Modify: `claude_tap/prompt_kb/mcp_server.py`（追加 `INSTALL_HINT` 常量与 `main()`）
- Modify: `claude_tap/cli.py:1085-1088`（`kb` 分发块之后）
- Test: `tests/prompt_kb/test_mcp_server.py`

**Interfaces:**
- Consumes: Task 2/3 的 `kb_search` / `kb_status`；`mcp.server.fastmcp.FastMCP`（可选 import）
- Produces:
  - `main() -> int`：缺 `[mcp]` 依赖时打印安装提示到 stderr 并返回 2；否则注册两个工具并跑 stdio server，正常退出返回 0
  - `INSTALL_HINT: str` 常量
  - CLI：`claude-tap mcp`（及 `python -m claude_tap mcp`）分发到 `mcp_server.main()`

- [ ] **Step 1: 写失败测试**

在 `tests/prompt_kb/test_mcp_server.py` 追加：

```python
def test_main_without_mcp_extra(capsys, monkeypatch):
    monkeypatch.setattr(mcp_server, "FastMCP", None)
    assert mcp_server.main() == 2
    assert "claude-tap[mcp,rag]" in capsys.readouterr().err


def test_main_registers_tools_and_runs_stdio(monkeypatch):
    class _FakeFastMCP:
        instances = []

        def __init__(self, name):
            self.name = name
            self.tools = []
            self.ran = False
            _FakeFastMCP.instances.append(self)

        def tool(self):
            def decorate(fn):
                self.tools.append(fn.__name__)
                return fn

            return decorate

        def run(self):
            self.ran = True

    monkeypatch.setattr(mcp_server, "FastMCP", _FakeFastMCP)
    assert mcp_server.main() == 0
    server = _FakeFastMCP.instances[0]
    assert server.tools == ["kb_search", "kb_status"]
    assert server.ran


def test_cli_dispatch_mcp(monkeypatch):
    import sys

    import claude_tap.cli as cli

    monkeypatch.setattr(sys, "argv", ["claude-tap", "mcp"])
    monkeypatch.setattr(mcp_server, "main", lambda: 0)
    with pytest.raises(SystemExit) as exc_info:
        cli.main_entry()
    assert exc_info.value.code == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run --extra dev pytest tests/prompt_kb/test_mcp_server.py -v`
Expected: 3 条新测试 FAIL（`mcp_server` 无 `FastMCP`/`main` 属性；`main_entry` 不识别 `mcp` 子命令）

- [ ] **Step 3: 实现 main() 与 CLI 分发**

`claude_tap/prompt_kb/mcp_server.py`：头部 `import sqlite3` 后补 `import sys`，并在 `_ctx` 定义前插入：

```python
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore[assignment]

INSTALL_HINT = "MCP support is not installed; run: pip install 'claude-tap[mcp,rag]'"
```

文件末尾追加：

```python
def main() -> int:
    """Run the MCP stdio server; degrade to an install hint without [mcp]."""
    if FastMCP is None:
        print(INSTALL_HINT, file=sys.stderr)
        return 2
    server = FastMCP("claude-tap-kb")
    server.tool()(kb_search)
    server.tool()(kb_status)
    server.run()  # stdio transport
    return 0
```

`claude_tap/cli.py` 在 `kb` 分发块（1085-1088 行）之后插入：

```python
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        from claude_tap.prompt_kb.mcp_server import main as mcp_main

        sys.exit(mcp_main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run --extra dev pytest tests/prompt_kb/test_mcp_server.py -v`
Expected: PASS（全部 10 条）

- [ ] **Step 5: 手工验证入口可用**

Run: `uv run --extra dev python -m claude_tap mcp </dev/null; echo "exit=$?"`
Expected: 若本环境已装 `mcp` 包——server 启动后因 stdin EOF 退出（exit=0 或 1 均可，但不能是 ModuleNotFoundError）；若未装——stderr 打印 `MCP support is not installed; run: pip install 'claude-tap[mcp,rag]'`，exit=2

- [ ] **Step 6: Commit**

```bash
git add claude_tap/prompt_kb/mcp_server.py claude_tap/cli.py tests/prompt_kb/test_mcp_server.py
git commit -m "feat(prompt_kb): 新增 claude-tap mcp 子命令与 stdio server 入口"
```

---

### Task 5: stdio 冒烟测试

**Files:**
- Test: `tests/prompt_kb/test_mcp_stdio.py`（新建）

**Interfaces:**
- Consumes: Task 4 的 `claude-tap mcp` 入口（经 `python -m claude_tap mcp`）；`mcp` 客户端 SDK（`ClientSession` / `StdioServerParameters` / `stdio_client`）
- Produces: 一条端到端测试，证明真实 MCP 客户端能 initialize → list_tools → call `kb_status`

- [ ] **Step 1: 写冒烟测试**

创建 `tests/prompt_kb/test_mcp_stdio.py`：

```python
"""End-to-end smoke test: real MCP client over stdio (skipped without [mcp])."""

import json
import os
import sys

import pytest

pytest.importorskip("mcp")  # requires the [mcp] extra

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


async def test_stdio_roundtrip(tmp_path):
    env = {**os.environ, "CLOUDTAP_DB": str(tmp_path / "traces.sqlite3")}
    params = StdioServerParameters(command=sys.executable, args=["-m", "claude_tap", "mcp"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {"kb_search", "kb_status"} <= {t.name for t in tools.tools}
            result = await session.call_tool("kb_status", {})
            assert not result.isError
            payload = json.loads(result.content[0].text)
            for key in ("snapshots", "chunks", "pending", "failed", "indexed", "messages", "embedder"):
                assert key in payload
```

说明：pytest-asyncio 已配置 `asyncio_mode = "auto"`（pyproject.toml:76），async 测试无需 marker。`CLOUDTAP_DB` 指向 tmp 路径使子进程的 `KbStore.default()` 落在隔离目录。`kb_status` 不依赖 embedder，因此无需 `[rag]` 的模型下载即可跑通。

- [ ] **Step 2: 运行冒烟测试**

Run: `uv run --extra dev --extra mcp pytest tests/prompt_kb/test_mcp_stdio.py -v`
Expected: 已装 `mcp` 时 PASS；未装时 SKIPPED（`pytest.importorskip`）。若 `uv run --extra mcp` 解析失败（如镜像缺包），改用 `uv run --extra dev --with mcp pytest ...`

- [ ] **Step 3: Commit**

```bash
git add tests/prompt_kb/test_mcp_stdio.py
git commit -m "test(prompt_kb): 新增 MCP stdio 端到端冒烟测试"
```

---

### Task 6: README 文档与全量回归

**Files:**
- Modify: `README.md:636`（"## Prompt Knowledge Base (optional)" 小节段落之后）
- Modify: `README_zh.md:631`（"## Prompt 知识库（可选）"小节段落之后）

**Interfaces:**
- Consumes: Task 1-5 全部成果
- Produces: 面向用户的安装与 `claude mcp add` 接入说明

- [ ] **Step 1: README.md 追加英文说明**

在 README.md 的 "Prompt Knowledge Base (optional)" 小节末尾（`claude-tap kb status` 那句之后）追加一段：

```markdown
The knowledge base can also be exposed to agents over MCP (stdio): after `pip install 'claude-tap[mcp,rag]'`, run `claude mcp add claude-tap-kb -- claude-tap mcp`. This registers a `claude-tap-kb` server with two read-only tools: `kb_search` (semantic search over prompts, tool definitions and user messages) and `kb_status` (index stats). Each `kb_search` call first indexes newly captured traces, so results stay fresh even when the dashboard is not running.
```

- [ ] **Step 2: README_zh.md 追加中文说明**

在 README_zh.md 的「Prompt 知识库（可选）」小节末尾（`claude-tap kb status`。那句之后）追加一段：

```markdown
知识库也可通过 MCP（stdio）暴露给 agent：安装 `pip install 'claude-tap[mcp,rag]'` 后执行 `claude mcp add claude-tap-kb -- claude-tap mcp`，即可注册 `claude-tap-kb` server，提供两个只读工具：`kb_search`（对 prompt、工具定义与用户消息做语义检索）与 `kb_status`（索引统计）。每次 `kb_search` 会先索引新采集的 trace，dashboard 未运行时结果依然新鲜。
```

- [ ] **Step 3: 全量回归 + lint**

Run: `uv run --extra dev pytest tests/ -x --timeout=60`
Expected: PASS（无新增失败；`test_mcp_stdio.py` 在无 `mcp` 依赖时为 SKIPPED）

Run: `uv run --extra dev ruff check . && uv run --extra dev ruff format --check .`
Expected: 无输出（通过）

- [ ] **Step 4: Commit**

```bash
git add README.md README_zh.md
git commit -m "docs(prompt_kb): README 新增 MCP server 安装与接入说明"
```

---

## Self-Review 记录

- **Spec 覆盖**：暴露两工具（Task 2）、stdio 传输 + `claude mcp add` 接入（Task 4/5/6）、查前补索引（Task 2 `test_kb_search_indexes_pending_first`）、FastMCP + extra 打包（Task 1/4）、四类错误处理（Task 3/4）、降级测试与冒烟测试（Task 3/4/5）、README 接入说明（Task 6）。spec 验收标准 1-4 均有对应任务。
- **类型一致性**：`kb_search`/`kb_status`/`main`/`_get_ctx`/`INSTALL_HINT` 在 Task 2-5 间签名一致；测试 monkeypatch 的名字（`_get_ctx`、`FastMCP`、`index_pending`、`main`）与实现逐一核对。
- **已知取舍**（与 spec 对齐，不改）：`kb_status` 对从未用过 RAG 的用户会创建空 KB 库文件（沿用 `KbStore.default()` 既有行为）；`kind` 过滤只作用于 chunks 分区（`search_messages` 无 kind 参数）。
