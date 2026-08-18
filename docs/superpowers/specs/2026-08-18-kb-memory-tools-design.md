# KB 记忆工具（方向 C phase 1：模型可调用记忆）

**日期**：2026-08-18
**状态**：设计已确认，待实施
**前置**：2026-08-14-rag-next-steps.md 项 5（方向 C：RAG 增强代理）

## 背景与定位

方向 C 的完整形态是「proxy 转发前从历史 trace 检索片段注入上下文」。经 brainstorming
拍板：**phase 1 只做模型可调用记忆（不自动注入、不碰 proxy 转发路径）**，用真实流量
验证「模型会不会主动查、查了有没有用」；自动注入（首轮 + 话题转换）降级为 phase 2
可选项，其中话题转换检测因噪声大长期搁置。

目标场景（三选全）：

1. **相似问题召回**——新问题的措辞与历史会话相似时，召回当时的问答片段
2. **用户事实/偏好记忆**——沉淀的偏好与环境事实（如「commit 用中文」「DLP 代理」）
3. **项目连续性记忆**——回答「上次做到哪了 / 最近在做什么」

关键洞察：场景 1/2 是**相似度问题**，场景 3 是**时间线问题**（「上次做到哪」与上次
工作内容字面上可能零重叠，相似度检索注定召回差）。因此拆成两个工具，各扛一种语义。

## 架构与组件

```
Claude Code / Codex CLI
        │ MCP stdio
        ▼
prompt_kb/mcp_server.py   ← 新增两个工具注册（薄层：参数透传 + 错误兜底）
        │
prompt_kb/recall.py       ← 新模块：kb_recall 拍平/排序/出处格式化，kb_recent 纯 SQL 时间线
        │
prompt_kb/search.py       ← kb_recall 复用 search_messages（零改动）
prompt_kb/store.py        ← kb_recent 直接 SQL 查询（kb_messages ⋈ kb_message_occurrences）
```

- **proxy / global_inject / extract 管线全部不动**。phase 1 对转发路径零接触，KB 只读。
- 埋点侧已无缺口：`extract.py` 同时抽 user/assistant 两种角色进 `kb_messages`。
- `kb_search`（prompt 快照 + 消息混合检索）原样保留，与新工具各司其职。

## 工具规格

### `kb_recall(query, client=None, limit=5, min_score=None)` —— 相似度召回

返回扁平命中列表（不按 session 分组，模型要的是 top-N 片段本身）：

```json
{
  "memories": [
    {
      "text": "……",
      "role": "user",
      "score": 0.83,
      "attribution": "2026-08-15 14:32 · claude-code · session 7f3a9c",
      "session_id": "…", "client": "…", "timestamp": "…"
    }
  ],
  "note": "以下来自本机历史会话，按相关度排序；内容可能过时，引用前请对照当前代码核实。",
  "reranked": true
}
```

- 调用链：`_get_ctx()`（复用现有惰性加载）→ `index_pending` 惰性补索引（沿用
  `kb_search` 的 `OperationalError` 容忍）→ `search_messages(...)` → 拍平 → 按
  score 排序 → 截到 `limit`（默认 5，比 `kb_search` 的 10 小，记忆贵精不贵多）
- `min_score` 缺省沿用 reranked 路径的 `DEFAULT_MIN_SCORE=0.505` 中性带下限
- **只搜消息语料，不搜 prompt 快照**（后者是 `kb_search` 的职责）
- `attribution` 是给模型和人共读的出处行；`note` 是防过时固定提醒——两者共同落地
  「每次检索都显示来源说明」的需求

description（决定模型主动性，需认真措辞）：
「回忆历史会话中与某话题相关的讨论。当用户的问题可能涉及之前做过的工作、提过的
偏好或类似问题时，先调用再回答。」

### `kb_recent(client=None, sessions=5, messages_per_session=3)` —— 纯时间线

不做相似度、不需要 embedding，全 SQL（预期 <50ms）：

```json
{
  "sessions": [
    {
      "session_id": "…", "client": "claude-code",
      "time_range": "2026-08-17 09:12 → 11:40",
      "first_user_message": "帮我看看 RRF 权重怎么调……（截断 200 字）",
      "recent_exchanges": [
        {"role": "user", "text": "…", "timestamp": "…"},
        {"role": "assistant", "text": "…", "timestamp": "…"}
      ]
    }
  ],
  "note": "近期会话按时间倒序，与当前输入无关——纯时间线，用于接续之前的工作。"
}
```

- 会话排序：`kb_message_occurrences` ⋈ `kb_messages` 按 `session_id` 聚合，
  `MAX(timestamp)` 倒序（按最后活动时间）
- `first_user_message`：session 内 `role='user'` 且 `(record_index, message_index)`
  最小的一条，截断 200 字符，当「任务标题」用
- `recent_exchanges`：session 内按 `(record_index, message_index)` 取最后 N 条
  （两种角色混排），每条截断 300 字符
- 响应体积控制：单条截断 + 省略号标记；要看全文可拿 `session_id` 去 `kb_search` 查

description：「按时间线查看最近在哪些会话里做了什么。当用户说『继续之前的』
『上次做到哪』时调用。」

## 错误处理

沿用现有 MCP 约定：工具永不抛栈，错误进返回值。

- `kb_recall`：`EmbedderUnavailable` / `ReindexRequired` / 其他异常 →
  `{"error": "…", "memories": [], "reranked": false}`；KB 为空不算错误 →
  `{"memories": [], "note": "知识库为空——还没有索引任何历史会话"}`
- `kb_recent`：空 KB → `{"sessions": [], "note": "…"}`；sqlite 异常 →
  `{"error": "…", "sessions": []}`

## 已知局限（v1 接受）

- `kb_recent` 的「任务标题」取自首条 user 消息；resume 的会话标题是续聊内容而非
  原始任务。不做会话合并。
- 无项目（cwd）过滤：cwd 在 trace 库 `sessions` 表，KB 抽取时未带入。按拍板结果
  v1 全局召回，项目维度留待后续（需补元数据管道：抽取时冗余或跨库 join）。

## 测试方案

- 新增 `tests/test_kb_recall.py`：tmp 目录 fixture KB 库，手工插入
  messages + occurrences 行
  - `kb_recent`：会话时间倒序、first_user_message 选取、最后 N 条往返、截断与
    省略号、跨会话共享行（occurrences）不重复出现、空库
  - `kb_recall`：拍平排序与 limit、attribution 格式、空库 note；embedder 用假
    向量 stub（沿用现有搜索测试的 stub 模式，不起真模型）
  - 错误路径：embedder 不可用 → error dict 而非异常
- MCP stdio 冒烟：照 913b262 双版本兼容模式，确认两个新工具在 MCP 1.x/2.x 下
  都能列出和调用
- 手动验证：本机 Claude Code 新会话问「上次我们做了什么」「我之前怎么处理 DLP
  代理的」，观察模型是否主动调用、出处是否可读

## 后续阶段（不在本 spec 范围）

- **phase 2（可选）**：proxy 首轮自动注入项目连续性摘要。是否启动取决于 phase 1
  实测：模型主动调用的频率与召回质量。
- **话题转换检测**：embedding 相似度判断话题漂移噪声大，长期搁置。
- **记忆蒸馏层（方向 C 远期）**：定期用 LLM 把历史会话蒸馏成结构化「事实/偏好/
  决策」条目存新表，检索直接查蒸馏结果。质量最高但要引入蒸馏管道（LLM 成本、
  调度、失败重试）。
- **项目（cwd）过滤**：补元数据管道后给两个工具加 `project` 参数。
