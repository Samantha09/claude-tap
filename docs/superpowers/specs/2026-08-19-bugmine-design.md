# bugmine：AI bug 复盘聚合（检测器 → bug_events → dashboard）

**日期**：2026-08-19
**状态**：设计已确认，待实施
**背景**：用户需要「快速把 AI 开发中写出来的 bug 整理出来」，拍板用途为**复盘聚合统计**
（非实时兜底）。分析维度四项全选：信号类型分布与趋势、按项目对比、按 client/模型对比、自愈率。

## 定位与边界

- 数据源为 claude-tap 已采集的 trace（records 全量请求/响应），零 LLM 成本，全部确定性启发式。
- **v1 三个信号**：`test_failure`、`fix_commit`、`user_correction`。
  `review_finding`（reviewer 输出的 Critical/Important）因格式不固定、噪声大**不进 v1**，留作 future。
- 不做 proxy 实时钩子，转发路径零接触；懒挖掘触发。
- 出口为 dashboard 新 tab；CLI `claude-tap bugs mine` 为副产品。
- 方向 C 的知识库（prompt_kb）与本功能完全解耦：bugmine 不用 embedding、不写 KB 库。

## 架构

```
trace records（已有）
   │
   ▼  claude_tap/bugmine/（新包）
   ├─ detectors.py    信号检测器：扫一个 session 的 records → bug_event 列表
   └─ mine.py         增量编排：仿 prompt_kb/extract.py 的 extract_unprocessed
   │
   ▼  traces DB（trace_store v5 迁移新增两表）
bug_events(id, session_id, ts, client, model, cwd,
           signal_type, title, excerpt, record_index, healed)
bugmine_sources(session_id, processed_at)
   │
   ▼  dashboard.html 新 tab「Bugs」+ live.py /api/bugs/* 路由
```

### 关键决策

- **`bug_events` 放 traces DB 而非 KB 库**：cwd/client 在同库 `sessions` 表，聚合同库
  join；`ON DELETE CASCADE` 使删会话时事件自动清除；bugmine 无 embedding，与 KB 解耦。
- **cwd/client/model 冗余进 `bug_events`**：提取时从 session 抄入，聚合查询免 join。
- **表结构进 trace_store 的 v5 迁移**（现有 `_create_v3/v4_tables` + `migration_state`
  机制），新库直建、旧库升级。
- **懒挖掘**：`/api/bugs/*` 路由进入时先跑 `mine_unprocessed(limit=50)`（同 KB 懒索引
  模式）；CLI `claude-tap bugs mine` 手动触发。
- **`record_index` 入事件**：明细行可跳回 `/dashboard/session/{session_id}` 的 trace 现场。

### bug_events 字段语义

| 字段 | 说明 |
|---|---|
| `session_id` | 所属会话（外键随会话删除级联） |
| `ts` | 事件发生时间（取所在 record 的 timestamp，ISO） |
| `client` / `model` / `cwd` | 从 session 冗余的聚合维度 |
| `signal_type` | `test_failure` / `fix_commit` / `user_correction` |
| `title` | 人类可读短标题（见检测器细则） |
| `excerpt` | 证据摘录，截断 300 字符 |
| `record_index` | 事件所在 record 序号（跳转现场用） |
| `healed` | 1/0/NULL，仅 `test_failure` 有定义（见下） |

去重：同一 session 重复挖掘不产生重复事件——`bugmine_sources` 处理后即跳过；
重挖（规则升级后）由 CLI `mine --reprocess` 删旧事件重来（v1 可先不做 --reprocess，
列为可选）。

## 检测器细则（detectors.py）

检测器按序遍历一个 session 的消息流（records → request body messages），
维护「最近的 Bash tool_use 命令」状态，产出事件列表。

### `test_failure`

- 触发：`tool_result` 块 `is_error=true` 且内容匹配 `/(\d+ failed|FAILED|ERROR|Traceback)/i`
- title：关联的最近 Bash 命令（截断 120）；excerpt：第一个 FAILED/Traceback 行起截断 300
- `healed`：同会话**其后**出现 `is_error` 为假且内容匹配 `/\d+ passed/i` 的
  `tool_result` → 1；到会话结束未出现 → 0
- 边界：非测试类的 is_error（如网络错误）被内容关键词挡住；同一命令反复失败产生
  多个事件（每次失败都是一个独立的 bug 现场，不合并）

### `fix_commit`

- 触发：assistant `tool_use` name=Bash，`input.command` 含 `git commit`，提取的
  `-m` 首行匹配 `/^fix[(:/\s]/i`
- title：commit message 首行；excerpt：完整命令截断 300
- `healed`：NULL（fix 本身是修复动作，无自愈概念）
- 边界：`git commit` 出现在 heredoc/引号嵌套里尽力提取首行，提取不到则用命令前
  120 字符兜底；不匹配 fix 前缀的提交不产生事件

### `user_correction`

- 触发：role=user 的**纯文本**消息（排除 tool_result 块）匹配
  `/(不对|错了|搞错|有 ?bug|不对劲|wrong|incorrect|broken|that's not)/i`
- title：消息前 60 字符；excerpt：前 300 字符
- `healed`：NULL
- 已知噪声：模式表偏宽（如「有 bug 吗」的疑问句会命中）——v1 接受，复盘场景
  宁可多报；模式表集中在 detectors.py 顶部常量，后续好调

### 自愈率口径

自愈率 = `test_failure` 事件中 `healed=1` 的占比。其余信号 `healed` 恒 NULL，
统计时排除。

## 增量编排（mine.py）

仿 `prompt_kb/extract.py:96` 的 `extract_unprocessed`：

```
mine_unprocessed(trace, *, limit=50):
    遍历最近 sessions（over-fetch）
    跳过：bugmine_sources 已处理 / record_count=0（直接标记）
    try: records = load_records → detectors → 批量插 bug_events → record_source
    except: 不标记，下一轮重试（同 KB 语义）
```

## Dashboard 页与 API

- `dashboard.html` 新增「Bugs」tab，四个聚合面板 + 事件明细列表：
  1. 信号类型分布计数 + 按天趋势
  2. 按项目（cwd）对比表
  3. 按 client × 模型对比表
  4. 自愈率（overall + 按天趋势）
  5. 明细列表行点击跳 `/dashboard/session/{session_id}`（已有路由）
- API（live.py）：
  - `GET /api/bugs/stats?days=30` → 四个聚合的预计算 JSON（服务端 SQL GROUP BY，
    前端只渲染）
  - `GET /api/bugs/events?days=&signal_type=&cwd=&limit=100` → 明细列表
  - 两路由进入先跑 `mine_unprocessed(limit=50)` 懒挖掘
- CLI：`claude-tap bugs mine`（手动触发，输出处理计数）

## 错误处理

- 单 session 挖掘失败不阻塞其他 session，不标记、下轮重试（同 KB）。
- `/api/bugs/*` 在库不存在/为空时返回空聚合（`{"by_signal": [], ...}`），不报错页。
- 检测器对畸形 record（缺字段、body 非 dict）跳过该 record，不抛。

## 测试方案

- 检测器单测：合成 record fixture（tool_use/tool_result 配对），三信号各自的
  命中/不命中/边界（healed 判定、非测试 is_error 不命中、fix 前缀边界、
  user_correction 排除 tool_result）
- mine.py：fixture trace 库的增量跳过、失败重试语义、空会话标记
- v5 迁移：旧 schema 库升级后两表存在
- API handler 测试：聚合形状、过滤参数、空库
- dashboard JS：`js-in-html-testing` 两层模式（Python 单测 + Playwright）
- **UI 证据**（仓库硬规则）：真实 trace 支撑的 Bugs tab 截图进 PR

## Future（不在本 spec）

- `review_finding` 信号（需先稳定 reviewer 输出格式）
- `claude-tap bugs mine --reprocess`（检测规则升级后的重挖）
- LLM 蒸馏精化 `user_correction` 软信号
- bug 与方向 C 记忆工具的联动（「这类 bug 以前怎么修的」）
