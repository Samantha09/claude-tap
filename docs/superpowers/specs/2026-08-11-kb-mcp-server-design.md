# 知识库 MCP Server 设计文档

日期：2026-08-11
状态：已确认需求，待实现

## 背景

prompt 知识库（方向 D）与会话消息检索（方向 A）已实现，检索入口有两处：
dashboard KB 页（人用）与 `claude-tap kb search`（终端用）。方向 D spec 预留了
第三个入口：**MCP server**——`search.py` 的 `search()` 接口边界干净，MCP 只是
包它一层。

接入 MCP 后，Claude Code 等 agent 可以在工作时自己检索本机知识库
（"哪家的工具定义适合某任务"、"上次我是怎么解决 X 的"），知识库从"人去查"
变成"agent 自助查"。这也是方向 C（注入增强代理）的前置：C 的注入逻辑同样需要
一套对外的程序化检索入口。

### 已确认的决策（与用户逐项确认）

| 决策点 | 结论 |
|--------|------|
| 暴露工具 | 只读两个：`kb_search` + `kb_status`；不暴露 reindex 等管理操作 |
| 传输方式 | stdio：`claude-tap mcp` 子命令跑 stdio server，`claude mcp add` 接入 |
| 索引新鲜度 | 每次 `kb_search` 先跑一轮 `index_pending` 补增量再检索 |
| 实现方式 | 官方 `mcp` SDK（FastMCP），新 extra `claude-tap[mcp]`；否决手写 JSON-RPC（方案 2）与独立入口（方案 3） |

补索引不会频繁触发重 embedding：`index_pending` 只处理 pending 状态的增量条目，
已有内容靠 content hash 去重；pending 为空时退化为一次毫秒级 SQL 查询。

## 目标与非目标

### 目标（预期效果）

1. **agent 可检索**：MCP 客户端调用 `kb_search` 拿到结构化 JSON 结果
   （prompt/工具块 + 会话消息两路），含来源信息（client/model/时间/session_id）
2. **agent 可自查状态**：`kb_status` 返回索引规模、embedder 名称、pending 数，
   agent 可先判断知识库有没有货
3. **结果新鲜**：dashboard 没跑时积累的 trace，`kb_search` 查前补一轮即可搜到
4. **一行接入**：`claude mcp add claude-tap-kb -- claude-tap mcp` 完成注册
5. **优雅降级**：未装 `[mcp]` 或 `[rag]` 依赖时，`claude-tap mcp` 打印安装提示
   退出；trace 录制、dashboard、kb CLI 完全不受影响

### 非目标（YAGNI）

- 不暴露管理/写入工具（reindex、删除）；管理仍走 `claude-tap kb` CLI
- 不做 HTTP/SSE 传输（dashboard 挂载点后续有需要再加）
- 不在 MCP 进程内跑懒索引循环（进程可能短命，查前补一轮已覆盖）
- 不新增索引内容；检索能力完全复用现有 `search()` / `search_messages()`

### 验收标准

1. `claude mcp add` 注册后，MCP 客户端 tools/list 可见 `kb_search` 与 `kb_status`，
   tools/call 返回结构化 JSON
2. dashboard 停止状态下新录一批 trace，首次 `kb_search` 能命中新内容（补索引生效），
   后续查询响应 < 200ms（索引规模同方向 A 验收）
3. 未装 `[mcp]` 时 `claude-tap mcp` 打印安装提示并以非零码退出，其余功能不受影响
4. 索引模型与当前 embedder 不一致时，`kb_search` 返回引导用户跑
   `claude-tap kb reindex` 的提示文本，不抛异常栈

## 架构

新增单文件模块，复用全部现有设施：

```
claude_tap/prompt_kb/mcp_server.py   # 新增（约 150 行）
├── main()                # claude-tap mcp 入口：依赖检查 → 构建 FastMCP server → run stdio
├── _get_ctx()            # 惰性单例：KbStore.default() + create_embedder(load_config())
│                         #   首次工具调用才加载 embedding 模型，server 启动秒级
├── kb_search(...)        # MCP 工具：补索引 → 两路检索 → 结构化 dict
└── kb_status()           # MCP 工具：store.stats() + embedder 元信息
```

`cli.py` 的 `main_entry` 仿照现有 `kb` 分发（`sys.argv[1] == "kb"` 模式）新增
`mcp` 分支，转发到 `mcp_server.main()`。

### 边界原则

- MCP 层只做**协议适配**：不新增检索逻辑，`kb_search` 是
  `index_pending()` + `search()` + `search_messages()` 的薄包装
- 工具函数与协议层可分离测试：单测直接调用工具函数（注入假 Embedder），
  不起 stdio 进程

### 打包与依赖

`pyproject.toml` 新增：

```toml
mcp = [
    "mcp>=1.0",
]
```

用户安装 `pip install 'claude-tap[mcp,rag]'`（mcp extra 不含 rag 依赖，
两个 extra 独立组合）。`mcp_server.py` 顶部：

```python
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None
```

`main()` 在 `FastMCP is None` 时打印安装提示并以退出码 2 退出。

## 数据流

### kb_search(query, client=None, kind=None, limit=10, min_score=0.0)

1. `_get_ctx()`：惰性构建 `(KbStore, Embedder)` 单例；embedder 不可用时
   返回安装提示 dict（见错误处理）
2. `index_pending(store, embedder)` 补一轮增量；SQLite 并发写冲突时跳过，
   继续用现有索引（见错误处理）
3. `search(store, embedder, query, client=client, kind=kind, limit=limit, min_score=min_score)`
   得到 prompt/工具块分组结果
4. `search_messages(store, embedder, query, client=client, limit=limit)`
   得到会话消息分组结果（kind 过滤只作用于第一路）
5. 合并为结构化 dict 返回（FastMCP 自动走 structured content）：

```json
{
  "chunks": [
    {"client": "...", "model": "...", "first_seen": "...", "last_seen": "...",
     "session_count": 3,
     "hits": [{"kind": "tool", "title": "...", "text": "...", "score": 0.83}]}
  ],
  "messages": [
    {"session_id": "...", "client": "...", "model": "...",
     "hits": [{"text": "...", "score": 0.79, "timestamp": "..."}]}
  ]
}
```

### kb_status()

复用 `store.stats()` 与 `store.get_meta("embedder_name")`（同 `kb status` CLI），
返回 dict（字段固定为 `store.stats()` 现有键名 + embedder 元信息）：

```json
{"snapshots": 12, "chunks": 1234, "pending": 0, "failed": 0, "indexed": 1234,
 "messages": 5678, "embedder": "local:intfloat/multilingual-e5-small"}
```

## 错误处理

| 场景 | 行为 |
|------|------|
| 未装 `[mcp]` | `claude-tap mcp` 打印 `pip install 'claude-tap[mcp,rag]'` 提示，退出码 2 |
| 未装 `[rag]` / embedding 不可用 | `kb_search` 返回 `{"error": "..."}` 提示 dict，不抛栈给 agent；`kb_status` 不受影响（不依赖 embedder） |
| `ReindexRequired`（索引模型不一致） | `kb_search` 返回 `{"error": "... run `claude-tap kb reindex`"}` 提示 dict |
| `index_pending` 撞 SQLite 写锁（dashboard 懒索引并发） | 捕获后跳过补索引，用现有索引继续检索；当次结果可能略旧但不失败 |
| 个别 chunk 索引失败 | 沿用方向 D 既有行为：标记 failed 可重试，不影响检索 |

## 测试

- **工具函数单测**：注入确定性假 Embedder（复用现有 search 测试的 fake），
  直接调用 `kb_search` / `kb_status`，断言结构化输出的分组、字段、过滤参数生效
- **查前补索引测试**：预置 pending 数据 → 调 `kb_search` → 断言新内容可命中；
  模拟 `index_pending` 抛锁错误 → 断言检索仍返回旧索引结果
- **降级测试**：mock `FastMCP = None` 时 `main()` 的提示与退出码；
  mock `EmbedderUnavailable` 时 `kb_search` 返回错误提示 dict
- **stdio 冒烟测试**：用 `mcp` 客户端 SDK 起子进程 `claude-tap mcp`，
  完成 initialize → tools/list → tools/call `kb_status` 一轮
  （标记为需要 `[mcp]` 依赖，CI 无依赖时 skip）

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 与 dashboard 懒索引并发写同一 SQLite | 查前补索引捕获锁错误降级为只读检索；SQLite WAL/超时沿用现有 store 配置 |
| MCP 进程每次启动加载模型慢 | 惰性加载：server 启动不加载，首次 `kb_search` 才初始化 embedder |
| `mcp` SDK 版本演进导致 API 变动 | extra 固定 `mcp>=1.0` 下限；协议交互全部经由 FastMCP 高层 API，不碰底层 |

## 文档与接入说明

README（英文 + 中文）新增一小节：

```bash
pip install 'claude-tap[mcp,rag]'
claude mcp add claude-tap-kb -- claude-tap mcp
```
