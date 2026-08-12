# 知识库聊天内容检索（用户输入 + 模型回复为主要依据）设计文档

日期：2026-08-12
状态：已确认需求与设计，待实现

## 背景

prompt 知识库（见 `2026-08-06-prompt-kb-rag-design.md`、
`2026-08-10-trace-semantic-search-design.md`、`2026-08-11-kb-mcp-server-design.md`）
当前索引两类内容：prompt 快照切块（system prompt 段落 + 工具定义）和用户消息。
2026-08-12 对 MCP `kb_search` 的实测（7 组中英文查询）暴露了四个问题：

1. **模型回复完全未索引**——聊天检索只有"问"没有"答"，而回答往往才是经验沉淀的主体
2. **样板 section 霸榜**：`Environment`/`Context management`/`Harness` 等 harness 固定模板
   出现在几乎每个快照，自带 git/shell/CLI 词汇，泛查询时噪声压过信号
   （搜"哪个 CLI 有沙箱 shell 工具"时 `getDiagnostics` 排在 `Bash` 之前）
3. **跨快照重复**：同一 chunk 存在于 ~10 个快照，limit=10 实际只返回 ~3 份不同内容
4. **绝对分数不可校准**：e5-small 对结构化短文本的余弦基线高，相关/无关全挤在
   0.79–0.90 窄带，`score>0` 默认过滤形同虚设，绝对阈值又无法拍通用值

用户拍板：检索应以**用户输入和模型返回的具体聊天内容**为主要依据；
prompt 快照/工具定义区**保留但降噪**，聊天内容排序优先。

### 已确认的决策（与用户逐项确认）

| 决策点 | 结论 |
|--------|------|
| 模型回复索引范围 | 仅正文 text；thinking 块、tool_use/函数调用一律跳过 |
| prompt 区去留 | 保留索引与检索能力，但样板 section 剔除 + 搜索结果聊天优先 |
| 回复数据源 | trace record 的 response 体（SSE 重组后的完整 JSON，一条 record 一条回复） |
| 问答绑定 | 不做 turn 级 Q/A 拼接索引（方案 2，复杂度不成比例），session 分组 + 时间戳缓解；后续数据量上来再评估 |
| 实现路线 | 演进式：`kb_messages` 加 `role` 列扩展，不新建表 |

## 目标与非目标

### 目标

1. 助手回复正文入库可检索，与用户消息同一 session 分组、带角色标注
2. 泛查询时工具定义/真实 prompt 段落排在样板模板之前
3. limit 预算不再被跨快照重复内容消耗
4. 长尾噪声被相对阈值自动截断，无需用户猜绝对分数
5. 存量数据全自动迁移与回填，用户零操作

### 非目标

- 不索引 thinking、工具调用参数、工具返回内容
- 不做问答对（turn）绑定索引
- 不做 ANN 向量索引、reranker
- 不改动 prompt 快照的采集与版本时间线语义
- 长消息代码 dump 的向量污染本轮不单独处理（切段 + 相对阈值兜底）

## 设计

### §1 Schema 迁移与存储（store.py）

`kb_messages` 表：

```sql
ALTER TABLE kb_messages ADD COLUMN role TEXT NOT NULL DEFAULT 'user';
DROP INDEX idx_kb_messages_dedup;
CREATE UNIQUE INDEX idx_kb_messages_dedup ON kb_messages(content_hash, client, role);
```

- 迁移走 `_migrate()` 幂等模式（参照已有 `attempts`/`messages_done` 两个 ALTER 先例）；
  存量行自动落入 `role='user'`，语义不变
- `upsert_message()` 增加 `role` 参数；`pending_messages()`/`indexed_messages()` 按需加 role 过滤
- `stats()` 的 messages 计数拆为 `messages_user`/`messages_assistant`（保留 `messages` 总数兼容）
- `kb_sources.messages_done` 全量重置为 0 触发回填（见 §5）

### §2 Assistant 回复抽取（messages.py / extract.py）

`messages.py` 新增：

```python
MIN_ASSISTANT_CHARS = 20

@dataclass(frozen=True)
class AssistantMessage:
    record_index: int
    message_index: int   # 超长切段后的序号
    timestamp: str
    text: str

def extract_assistant_messages(records: list[dict]) -> list[AssistantMessage]: ...
```

数据源为 `record["response"]["body"]`，按 provider 取正文：

| provider | 路径 |
|---|---|
| anthropic | `content[]` 中 `type=="text"` 块，`\n\n` 连接 |
| openai chat | `choices[].message.content`（字符串或 parts） |
| openai responses | `output[]` 中 `type=="message"` 项的 `output_text` |
| gemini | `candidates[].content.parts[].text` |

过滤规则：

- 空文本跳过；无 text 块的回复（纯工具调用）不入库
- `len(text.strip()) < MIN_ASSISTANT_CHARS` 跳过——助手短客气话（"好的，我来处理"）
  量大无检索价值。用户消息侧**不**加此过滤（短问题有检索价值）
- 用户消息的 `_DROP_*` harness 规则不适用 assistant 路径

超长回复复用 `_split_message()`（MAX_SECTION_CHARS 不变）切段，`message_index` 递增。

`extract.py` 的 `extract_messages()` 扩展为同时抽取两类，分别
`upsert_message(role="user"/"assistant")`，返回 `{"user": n, "assistant": m}` 计数。

### §3 Prompt 区降噪

**① 索引期样板黑名单（chunk.py）**：

```python
BOILERPLATE_TITLES = {"environment", "context management", "harness", "session-specific guidance"}
```

`chunk_snapshot()` 切 section 时跳过（大小写不敏感）。工具定义与真实 prompt 段落不受影响。
存量清理：`KbStore` 打开时幂等执行
`DELETE FROM kb_chunks WHERE lower(title) IN (...)`，embedding 随行删除，无需重索引。

**② 检索期跨快照去重（search.py）**：打分后、分组前按 `(kind, title, text)` 折叠，
保留分数最高的出现，归入 `last_seen` 最新的快照组，`session_count` 累加。
快照实体与时间线语义不变，仅搜索输出不重复。

### §4 检索层：聊天优先 + 相对阈值

- `kb_search` 返回 `{"messages": [...], "chunks": [...]}`，messages 分区提为第一位；
  两分区各自独立 limit（现状不变）
- `MessageHit` 增加 `role` 字段，透传到 MCP 响应、CLI 输出、dashboard 卡片
- 相对阈值（`search()`/`search_messages()` 各分区独立）：

  ```python
  cutoff = max(min_score, top_score - rel_delta)   # rel_delta 默认 0.05
  ```

  `min_score` 参数保留（向后兼容），新增 `rel_delta` 参数；top hit 永远通过，
  截断的是长尾。传 `rel_delta=1.0` 可完全关闭
- Dashboard：会话分区卡片渲染在快照分区之前；命中条目加角色徽标
  （i18n：`kb_role_user`="提问"、`kb_role_assistant`="回答"）；滑块语义不变（与 rel_delta 叠加取更严者）
- CLI `claude-tap kb search`：messages 小节先打印，每条带 `[user]`/`[assistant]` 前缀
- MCP `kb_search` docstring 更新：messages 优先、role 字段、rel_delta 含义

### §5 错误处理与回填

- 迁移时 `kb_sources.messages_done` 重置为 0，现有 `_backfill_missing_messages`
  触发全量重抽；user 消息按 `(content_hash, client, role)` 去重幂等，assistant 为净新增
- response 体缺失/非 JSON/结构不符 → 跳过该 record，不影响 session 标记
  （纯读取无可重试瞬态错误，与 user 路径的 session 级重试语义不同）
- 抽取/入库抛异常 → session 不标 `messages_done`，下轮重试（现有语义）
- MCP `kb_search` 错误契约不变，`role` 为纯增量字段
- 成本预估：消息量 488 → ~1.5k，本地 e5-small 批量 embedding + 暴力检索无压力
  （5 万 chunk 阈值结论不变）

### §6 测试策略

| 层 | 新增/扩展 |
|---|---|
| `tests/prompt_kb/test_messages.py` | 4 种 provider response 解析；thinking/tool_use 跳过；<20 字符过滤；超长切段 |
| `tests/prompt_kb/test_store_messages.py` | 老库迁移后 role='user'；同文本 user/assistant 各存一行 |
| `tests/prompt_kb/test_chunk.py` | 黑名单标题不产 chunk |
| `tests/prompt_kb/test_search.py` | 跨快照同内容折叠为一组；rel_delta 截断长尾、top hit 必过 |
| 迁移测试 | 构造旧 schema 库 → 新代码打开 → 验证 ALTER、样板 chunk 清除、messages_done 重置 |
| CLI/dashboard | 锚点测试扩展：角色徽标、分区顺序 |
| 端到端 | 现有 MCP stdio 冒烟测试回归；手动跑实测的 7 组查询做前后对比 |

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 样板黑名单误删有价值 section | 只按标题精确匹配 4 个 harness 固定模板；工具定义与正文不受影响；常量可配置扩展 |
| rel_delta=0.05 在新分数分布下不合适 | 参数化暴露，dashboard 滑块可叠加绝对门槛；实测对比后可调默认值 |
| 回复正文含敏感信息 | 与 trace 库同级本地存储，不新增外发路径（沿用既有结论） |
| 回填期间搜索结果新旧混杂 | 渐进索引 pending 计数可见；回填幂等可重入 |

## 实施验证

**日期**：2026-08-12（Task 8，分支 dev/trace-semantic-search）

### 回归

- 全量单测：`.venv/bin/pytest tests/ --ignore=tests/test_e2e.py -q` → **1156 passed, 26 skipped**（82s）
- e2e：`.venv/bin/pytest tests/test_e2e.py -v --timeout=120` → **68 passed**（23s）

### 迁移与回填（before → after）

执行 Task 8 时真实库（`~/.local/share/claude-tap/prompt_kb.sqlite3`）的迁移已在 Task 1–7 开发/验证期间应用，
故无法复现 `messages_assistant=0` 的迁移前状态；改为验证迁移幂等性与全量 reindex 正确性。

| 指标 | 基线（2026-08-12 实测） | Task 8 时 |
|---|---|---|
| snapshots | 10 | 10 |
| chunks | 383 | **334**（样板 section 一次性清除 −49） |
| messages_user | 488 | **1079**（当日新增会话累积；user 抽取逻辑未变） |
| messages_assistant | 0 | **858** |
| pending / failed | — | 0 / 0 |

全量 reindex（`claude-tap kb reindex`，uv tool 环境）：`indexed=334 failed=0 messages_indexed=1937 messages_failed=0`，
耗时约 71s（e5-small 本地缓存）。注：repo `.venv` 未装 `claude-tap[rag]`，验证使用 uv tool editable 安装
（同一份代码 + sentence-transformers 5.7.0）。

### 7 组查询验收

| # | 查询 | 判定 | 关键结果 |
|---|---|---|---|
| 1 | 哪个 CLI 有沙箱 shell 工具 | ✅ | 无 Environment/Context management/Harness 命中；Bash 进入 top-3 组（0.826/0.825）；getDiagnostics 仍居首（0.850，见遗留） |
| 2 | which CLI has a sandboxed shell tool | ✅ | 无样板命中；**Bash 升为 top 组首命中（0.849）**，此前仅 ~第 3（0.826） |
| 3 | 怎么写 commit message | ✅ | 无样板/无同内容跨快照重复；messages 区命中 commit 相关 assistant/user |
| 4 | 前端页面截图验证 | ✅ | messages 区 top 命中 assistant 回答（"好，启动 dashboard 并截图验证"，0.896），带 role 标注 |
| 5 | 取消定时任务 cron（kind=tool） | ✅ | **CronDelete 仍居首（0.902）**，单一 group，回归保护成立 |
| 6 | 沙箱 sandbox 执行命令（min_score=0.86） | ✅（带遗留） | chunks 仍为空（新分布下无 chunk ≥0.86）；messages 区 top 变为 assistant 内容（0.887），不再只有代码 dump |
| 7 | Playwright 浏览器截图验证（kind=prompt_section） | ✅ | messages 区 top 命中 assistant "playwright + 本机 Chrome 验证通过"（0.924） |

全查询公共验收：

- **无跨快照同内容重复**：以 (kind, title, text) 全键校验 6 组有 chunk 结果的查询，同一内容均只出现一次 ✅
- **长尾截断**：每组 hits ≤3（代码强制）；rel_delta 相对阈值生效（Q4→2 组、Q5→1 组、Q6→0 chunk）✅
- **role 透传**：所有查询 messages 区命中均带 role，user/assistant 均可被检索 ✅

### 遗留问题

1. 中文查询（Q1）下 `mcp__ide__getDiagnostics`（0.850）仍排在 `Bash`（0.826）之前——e5-small 对
   中文短查询的区分度限制，非回归；英文查询已修正。后续可考虑更强多语模型或查询侧改写。
2. 绝对分阈值（min_score=0.86）在新分数分布下仍会全灭 chunks 区，且 messages 区中段仍混入代码 dump
   （0.869–0.880）。绝对分数未校准，属 spec 风险表已收录项；建议下游默认用 rel_delta 而非绝对阈值。
