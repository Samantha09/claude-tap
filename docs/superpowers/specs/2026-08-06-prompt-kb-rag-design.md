# Prompt 知识库（RAG 方向 D）设计文档

日期：2026-08-06
状态：已确认需求，待实现

## 背景

claude-tap 已能从 15+ 家 AI coding CLI 的 trace 中提取归一化的 `PromptSnapshot`
（`prompt_snapshot.py`：provider、model、system_prompt、developer_prompt、tools）。
下游 Phistory 已证明 prompt 快照有归档/对比价值。

本设计把快照能力升级为**本地可检索的 prompt/工具生态知识库**：对本机被动采集的
trace 快照做细粒度切块 + embedding 索引，回答两类典型问题：

1. "哪家的工具定义适合某任务"（语义搜索 tool 定义）
2. "哪版 prompt 加了 X 规则"（块级命中 + 版本时间线）

### 已确认的决策（与用户逐项确认）

| 决策点 | 结论 |
|--------|------|
| 语料来源 | 只索引本机被动采集的 trace，不主动批量采集、不导入外部数据 |
| 查询入口 | dashboard（19527）加"Prompt 知识库"页优先；`search()` 接口边界干净，预留 MCP |
| Embedding | 混合：默认本地 sentence-transformers 小模型，可配置切 API；可选依赖 `claude-tap[rag]` |
| 检索粒度 | 细粒度多文档：prompt 按段切块 + 每个 tool 单独一条 |
| 索引时机 | dashboard 长驻进程内懒索引（模型只加载一次）+ `claude-tap kb` 手动命令 |
| 存储架构 | SQLite + numpy 暴力余弦（数据量千级 chunk，无需向量索引库） |
| 版本管理 | 内容 hash 去重；相同 client+model+内容只存一个版本，记录 first_seen/last_seen/session_count |

其他 RAG 方向（A trace 语义检索 / B 语料输出 / C 注入增强代理）已记录暂缓，不属于本设计范围。

## 目标与非目标

### 目标（预期效果）

1. **语义搜索**：dashboard 知识库页输入自然语言，返回命中的 tool 块 / prompt 段落块，
   显示来源（client、model、版本时间）、片段高亮，可展开 tool 完整 schema
2. **版本时间线**：某 client+model 的 prompt 版本按 first_seen 排列；块命中能定位
   "该内容最早出现于哪个版本"
3. **零操作积累**：日常使用 CLI 无需任何额外动作，dashboard 后台懒索引自动完成
4. **CLI 可查**：`claude-tap kb search "<query>"` 终端出结果
5. **优雅降级**：未装 `[rag]` 依赖、embedding 不可用、索引失败时，trace 录制与
   dashboard 其他功能完全不受影响

### 非目标（YAGNI）

- 不做 LLM 生成式问答（检索出原文片段即结束）
- 不索引对话正文和工具调用记录（那是方向 A）
- 不做跨机器同步、不发布公网
- 本期不做 MCP server，只保证接口边界

### 验收标准

1. ≥3 个不同 client 的 trace 存在时，搜索结果正确区分来源（client/model/版本时间）
2. 同一 prompt 跑 10 个会话，`kb_snapshots` 只 1 条版本记录且 `session_count=10`
3. 未装 `[rag]` 依赖时：dashboard 其他功能不受影响，知识库页显示安装提示而非报错
4. embedding 不可用时 trace 录制与 dashboard 照常工作，失败 chunk 标记 failed 可重试
5. 千级 chunk 规模下搜索响应 < 200ms

## 架构

新增独立包 `claude_tap/prompt_kb/`：

```
claude_tap/prompt_kb/
├── __init__.py        # 公开接口：search(), index_pending(), rebuild()
├── extract.py         # trace session 记录 → PromptSnapshot（复用 prompt_snapshot.py）
├── chunk.py           # PromptSnapshot → 切块（prompt 段落块 + tool 块）
├── store.py           # SQLite 读写：版本表、chunk 表、来源表、元信息表
├── embed.py           # Embedder 协议 + LocalEmbedder / ApiEmbedder + 工厂
├── index.py           # 懒索引循环 + 全量重建（dashboard 与 CLI 共用）
└── search.py          # 查询 → embed → numpy 余弦 → 按快照分组排序
```

### 边界原则

- `search.py` 的 `search(query, filters) -> list[SearchResult]` 是唯一对外检索入口；
  将来的 MCP server 只是包它一层
- `embed.py` 的 Embedder 是协议（`embed(texts: list[str]) -> list[list[float]]`），
  测试注入确定性假 Embedder，不碰真模型
- `prompt_kb` 不 import dashboard 代码；dashboard 单向调用它
- 提取层复用现有 `prompt_snapshot.py`，不重写归一化逻辑

### 对现有代码的改动点（最小化）

- `dashboard.py`：新增 `/api/kb/search`、`/api/kb/status`、`/api/kb/reindex` 路由；
  启动时 spawn 懒索引后台线程
- `dashboard.html`：新增"Prompt 知识库"标签页（搜索框 + 过滤器 + 结果列表 + 版本时间线视图）。
  项目界面默认语言已为简体中文（2026-08-06 起），本页面 i18n 条目以中文为源语言编写
- `cli.py`：新增 `claude-tap kb search` / `claude-tap kb reindex` 子命令
- `pyproject.toml`：新增可选依赖组 `rag = ["sentence-transformers", "numpy"]`

## 数据模型

单个 SQLite 文件（跟随 trace 存储目录，如 `~/.claude-tap/prompt_kb.db`）。该库是
**派生数据**：损坏或换模型后可从 trace 全量重建。

```sql
-- 快照版本：内容 hash 去重，天然形成版本时间线
kb_snapshots (
  id INTEGER PRIMARY KEY,
  content_hash TEXT NOT NULL,          -- client+model+prompt+tools 归一化后的 hash
  client TEXT NOT NULL,                -- claude-code / codex / kimi ...
  provider TEXT NOT NULL,              -- anthropic / openai / gemini
  model TEXT NOT NULL,
  system_prompt TEXT,
  developer_prompt TEXT,
  tools_json TEXT,                     -- 序列化的 PromptTool 列表
  first_seen TEXT NOT NULL,            -- 首次采集时间（= 版本诞生时间）
  last_seen TEXT NOT NULL,
  session_count INTEGER DEFAULT 1,
  UNIQUE(client, model, content_hash)
)

-- 细粒度索引单元
kb_chunks (
  id INTEGER PRIMARY KEY,
  snapshot_id INTEGER NOT NULL REFERENCES kb_snapshots(id),
  kind TEXT NOT NULL,                  -- 'prompt_section' | 'tool'
  title TEXT,                          -- 段落标题或工具名
  text TEXT NOT NULL,                  -- 切块正文
  embedding BLOB,                      -- float32 向量；NULL = 待索引
  index_state TEXT NOT NULL DEFAULT 'pending'  -- pending | indexed | failed
)

-- 已提取过的 trace 会话（避免重复提取）
kb_sources (
  session_id TEXT PRIMARY KEY,
  snapshot_id INTEGER REFERENCES kb_snapshots(id),  -- NULL = 该会话无 prompt 内容
  processed_at TEXT NOT NULL
)

-- 元信息：换 embedding 模型时检测不匹配 → 提示全量重建
kb_meta (
  key TEXT PRIMARY KEY,
  value TEXT
)  -- embedder_name, embedding_dim, schema_version
```

### 切块规则（chunk.py）

- system_prompt / developer_prompt：按 Markdown 标题切分；无标题时按段落合并，
  上限约 500 token；保留标题层级路径进 `title` 字段
- 每个 tool 一条 chunk：`text = name + "\n" + description + "\n参数: " + 参数名列表`；
  schema 全文不进向量，命中后从 `kb_snapshots.tools_json` 取原文展示
- `content_hash` 输入：client + model + 归一化后的 system/developer prompt + tools
  规范序列化（排序、去空白差异），保证同内容必同 hash

## 数据流

### 索引流（懒索引，dashboard 长驻进程内）

1. trace 照常录制（proxy 进程零改动、零负担）
2. dashboard 启动时 spawn 索引线程：
   a. 扫描 trace 会话，跳过 `kb_sources` 已处理的
   b. 对新会话调用提取层得 `PromptSnapshot`；无 prompt 内容的会话记
      `kb_sources(snapshot_id=NULL)` 后跳过
   c. 按 content_hash upsert `kb_snapshots`：已存在则 `session_count+1`、更新
      `last_seen`；不存在则插入新版本并生成 pending chunks
   d. 加载 embedding 模型一次（长驻进程内保持热），批量 embed pending chunks，
      置 indexed
   e. 休眠后轮询新 pending
3. `claude-tap kb reindex`：清空全部 embedding、chunks 重置 pending，重建索引
   （换 embedding 模型后使用）

### 检索流

1. dashboard `/api/kb/search?q=...&client=...&kind=...` 或 `claude-tap kb search`
2. embed 查询文本 → 读取全部 indexed 向量 → numpy 余弦相似度
3. 按相似度排序，按 snapshot 分组（同一快照的多个命中聚合展示）
4. 返回：chunk 文本、kind、title、所属快照的 client/model/first_seen/版本信息

### 版本时间线查询

`SELECT ... FROM kb_snapshots WHERE client=? AND model=? ORDER BY first_seen`
即版本史；块命中的"最早出现于" = 该 chunk 所属快照的 first_seen。

## Embedding 配置

配置文件（沿用 claude-tap 现有配置位置）新增 `[prompt_kb]` 段：

```toml
[prompt_kb]
embedder = "local"                    # local | api
local_model = "intfloat/multilingual-e5-small"  # 默认本地多语言小模型（~470MB，首次自动下载）；语料为英文 prompt + 中文查询，必须多语言模型
# api_base = "https://..."            # embedder=api 时：OpenAI 兼容 embedding 端点
# api_model = "text-embedding-3-small"
# api_key_env = "OPENAI_API_KEY"      # 从环境变量读 key，不存明文
```

- 默认 local；未装 `[rag]` 依赖时一切功能降级而非报错
- `kb_meta` 记录当前 embedder_name + embedding_dim；配置切换导致不匹配时，
  search/status 接口返回"需要 reindex"提示，不静默给出错误结果

## 错误处理

| 场景 | 行为 |
|------|------|
| 未装 `sentence-transformers` | 知识库页显示"安装 claude-tap[rag] 以启用"；API 返回 501 + 提示；索引线程不启动 |
| 模型首次下载失败（网络/DLP 拦截） | 索引线程记录错误并退避重试；`/api/kb/status` 显示 `embedder_unavailable`；trace/dashboard 其他功能不受影响 |
| 单条 chunk embed 失败 | 标记 `failed`，下轮重试（带次数上限，超限保持 failed 并在 status 暴露计数） |
| API embedder 失败（key/网络） | 同上：failed + status 可见，不影响本地功能 |
| 切换 embedding 模型 | kb_meta 不匹配 → search 返回"需 reindex"提示 |
| prompt_kb.db 损坏 | 派生数据：删除后 reindex 从 trace 全量重建 |
| 会话提取失败（异常 trace） | 记 `kb_sources(snapshot_id=NULL)` 跳过并记日志，不阻塞后续会话 |

## 测试策略

遵循仓库现有 pytest 模式（`tests/`）：

- **chunk.py**：标题切分、无标题段落合并、超长段落二次切分、tool chunk 文本格式
- **store.py**：content_hash 去重（同 hash → session_count 递增、last_seen 更新）、
  版本时间线排序、kb_sources 跳过逻辑
- **search.py**：注入确定性假 Embedder（词频哈希向量），验证排序、分组、过滤器
  （client/kind）、空库行为
- **index.py**：用 fixture trace 记录端到端走通 提取→切块→索引→可搜索；
  embedder 抛错时 chunk 标记 failed 且不中断批次
- **dashboard API**：`/api/kb/*` 路由测试；未装 rag 依赖时返回 501 提示而非 500
- **降级**：模拟 import 失败，验证 dashboard 启动与既有路由回归正常

embedding 维度在测试中用小的固定值（如 8 维假向量），不依赖真实模型。

## 里程碑（供实现计划参考）

1. `prompt_kb` 核心包：store + chunk + embed 抽象 + search（假 Embedder 可测）
2. index.py + extract 集成 + `claude-tap kb` CLI
3. dashboard API + 知识库页面
4. 降级路径、reindex、文档与可选依赖打包
