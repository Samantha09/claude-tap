# Prompt 知识库页面重构设计文档

日期：2026-08-07
状态：已确认需求，待实现

## 背景

Prompt 知识库页（dashboard `kb-view`）目前是纯文本堆叠：结果卡片无边距层次、
分数只有括号数字、命中预览截断后无法看全文、同一 client+model 的多个历史快照
各自渲染成内容雷同的卡片，搜索无结果时只有一句"No matching results"。
用户反馈"太丑""搜出来看不懂"。

本次做前端重构 + 功能增强，不改后端 API。

## 已确认的决策（与用户逐项确认）

| 决策点 | 结论 |
|--------|------|
| 改造幅度 | 重构 + 功能增强（非纯 CSS 美化，也不加新 API） |
| 最低分过滤 | 前端滑块，对已有搜索结果即时重渲染，不重新请求 |
| 雷同快照 | 按 client+model 归并，最新版为主卡片，其余折叠为"历史快照" |
| 命中文本 | 默认 200 字符预览，可展开全文/折叠 |
| 兼容约束 | 保留现有测试锚点（id、timeline toggle、placeholder 文案等） |

## 目标与非目标

### 目标

1. 结果卡片结构化：类型徽章上色（工具=蓝 / Prompt 段落=紫）、分数进度条 + 数字、
   分档着色（≥0.9 绿 / 0.7–0.9 蓝 / <0.7 灰）
2. 雷同快照归并折叠，结果顶部显示摘要（"找到 N 组结果，其中 M 组已折叠"）
3. 最低分数滑块前端过滤；低分被滤光时给出提示
4. 空结果状态带引导文案（说明知识库收录范围，建议搜工具名或调低分数）
5. 时间线美化（竖线 + 圆点，当前快照高亮），保留现有 toggle 行为
6. 暗色主题自动适配（全部走现有 CSS 变量）

### 非目标（YAGNI）

- 不改 `/api/kb/search`、`/api/kb/timeline` 等后端接口
- 不做服务端最低分过滤、不做分页
- 不预先获取每张卡片的版本数（版本历史按钮保持点击时才请求）
- 不动 trace 列表/详情/统计页

## 设计

### 搜索控制区

- 输入框 `#kb-query`、类型筛选 `#kb-kind`、搜索按钮一行排布，复用现有输入框样式
- 新增最低分数滑块 `#kb-min-score`（range 0.0–1.0，步进 0.05，默认 0.0），
  旁显当前值；`input` 事件触发前端重渲染（用缓存的上次搜索结果，不重新 fetch）
- "重建索引"按钮改为次要样式（描边按钮，区别于主按钮）
- 状态行改为徽章组：嵌入模型 / 已索引数 / 快照数（复用 `kbEmbedderLabel`）

### 结果渲染（`renderKbResults` 重写）

数据流：`/api/kb/search` 返回 groups → 缓存到 `state.kbLastResults` →
前端管线 `归并(client+model) → 分数过滤 → 渲染`。

- **归并**：同一 `client+model` 的 group 合并为一组；`first_seen` 最新者为
  主卡片（标"最新版"），其余按 first_seen 倒序收进 `▸ N 个历史快照` 折叠区，
  点击展开为完整卡片（同样的 hit 结构）
- **摘要行**：`找到 N 组结果`；有折叠时追加 `，其中 M 组已折叠为历史快照`
- **hit 行**：徽章（`.kb-hit-kind.tool` 蓝 / `.prompt` 紫）+ 标题 +
  分数条（`.kb-score-bar`，宽度 = score×100%）+ 分数数字；分档着色类
  `.score-high`（≥0.9 绿）/ `.score-mid`（0.7–0.9 蓝）/ `.score-low`（<0.7 灰）
- **展开全文**：预览超过 200 字符时显示 `▸ 展开全文` 按钮，切换显示完整
  `hit.text`（等宽字体块）/ 收起
- **分数过滤**：hit.score < 滑块值时整条不渲染；某组 hits 全被滤掉则整组不渲染；
  全部被滤掉时显示 `N 条结果被最低分数过滤掉了` + 调低分数建议
- **空结果**：`📭 没有命中` + 引导文案（知识库只收录 system prompt 和工具定义，
  不含聊天内容；试试搜工具名或调低最低分数）

### 时间线

- 保留 `kbLoadTimeline` 的 toggle 行为与 `ul.kb-timeline` 结构
- CSS 美化：左竖线边框 + 圆点列表项，当前快照（`v.id === group.snapshot_id`）
  高亮加粗

### i18n 新增词条（en + zh-CN 双语，跟随现有字典结构）

| key | en | zh-CN |
|-----|----|-------|
| `kb_min_score` | Min score | 最低分数 |
| `kb_expand` | Show full text | 展开全文 |
| `kb_collapse` | Collapse | 收起 |
| `kb_latest` | Latest | 最新版 |
| `kb_history_snapshots` | {n} history snapshot(s) | {n} 个历史快照 |
| `kb_summary` | {n} result group(s) | 找到 {n} 组结果 |
| `kb_summary_folded` | , {m} folded as history | ，其中 {m} 组已折叠为历史快照 |
| `kb_no_results_hint` | The KB only indexes system prompts and tool definitions, not chat content. Try a tool name or lower the min score. | 知识库只收录 system prompt 和工具定义，不含聊天内容。试试搜工具名，或调低最低分数 |
| `kb_filtered_by_score` | {n} hit(s) hidden by min score | {n} 条结果被最低分数过滤掉了 |

`t()` 已支持 `{var}` 插值（沿用现有调用方式）。

### CSS

全部新增样式挂在 `.kb-*` 类下，颜色一律用现有变量（`--blue`/`--green`/
`--purple`/`--border`/`--bg-card`/`--text-secondary` 等），暗色主题零额外工作。
新增类：`.kb-status-chips`、`.kb-chip`、`.kb-group-header`、`.kb-latest-badge`、
`.kb-score`、`.kb-score-bar`、`.score-high/mid/low`、`.kb-hit-text`、
`.kb-expand-btn`、`.kb-history-fold`、`.kb-summary`、`.kb-empty`、
`.kb-timeline` 美化规则。

### 兼容锚点（现有测试锁定，必须保留）

- `id="kb-view"` / `kb-query` / `kb-results` / `kb-status`、`data-view="kb"`
- placeholder 文案 `搜索 prompt 规则或工具定义`
- `card.querySelector("ul.kb-timeline")` toggle、`list.className = "kb-timeline"`、
  `fmtTime(v.first_seen)`
- `kbEmbedderLabel(data.embedder)`、`fmtTime(group.first_seen)`
- view 切换守卫三段判断（`state.view !== "kb"` 等）

## 错误处理

- `/api/kb/search` 返回 501/409：保持现状（显示 hint/error 文本），套上 `.kb-empty` 样式
- 时间线请求失败：在卡片内显示错误文案，不弹窗
- `state.kbLastResults` 为空时拖滑块：无操作

## 测试

1. 更新 `tests/prompt_kb/test_kb_page.py`：保留现有锚点断言，新增——
   滑块元素存在、新 i18n key 双语存在、归并/过滤/展开函数名存在
2. 用 `js-in-html-testing` 技能为新渲染逻辑写 JS 测试：
   归并（同 client+model 多组 → 1 主 + N 折叠）、分数过滤（滑块值滤 hit/整组/全部）、
   展开全文 toggle、空结果引导文案
3. 跑 `e2e-test` 技能做端到端验证（知识库页搜索 → 渲染 → 折叠展开 → 时间线）

## 验收标准

1. 同一 client+model 的 3 个快照搜索结果只显示 1 张主卡片 + "2 个历史快照"折叠
2. 拖最低分滑块到 0.85，截图中的 0.82 分 hit 立即消失，不发起新请求
3. 搜索无结果时显示 📭 引导文案而非一行小字
4. 暗色/亮色主题下徽章、分数条、卡片均正常显示
5. 现有 `test_kb_page.py` 锚点断言全部保持通过
