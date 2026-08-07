# Prompt 知识库页面重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 dashboard Prompt 知识库页：结构化结果卡片（徽章/分数条/展开全文）、同 client+model 快照折叠、最低分滑块前端过滤、空状态引导、时间线美化。

**Architecture:** 纯前端改造，只动 `claude_tap/dashboard.html`（CSS + JS 渲染管线）+ i18n 词条 + 测试。搜索 API 返回后缓存到 `kbLastGroups`，前端管线 `过滤(min-score) → 归并(client+model) → 渲染` 重渲染不重新请求。测试用 js-in-html-testing 双层策略：Python 复制算法单测（Layer 1）+ Playwright 浏览器测试（Layer 2）。

**Tech Stack:** aiohttp dashboard（单文件 HTML 模板）、原生 JS/CSS 变量、pytest、Playwright。

## Global Constraints

- 只改 `claude_tap/dashboard.html`；不改 `/api/kb/*` 后端接口
- 颜色一律用现有 CSS 变量（`--blue`/`--green`/`--purple`/`--border`/`--bg-card`/`--bg-hover`/`--text-secondary`/`--text-tertiary` 等），暗色主题零额外工作
- commit message 一律中文（含 type/scope 前缀），代码/注释英文
- 现有测试锚点必须保留：
  - `id="kb-view"` / `kb-query` / `kb-results` / `kb-status`、`data-view="kb"`
  - placeholder 文案 `搜索 prompt 规则或工具定义`
  - `card.querySelector("ul.kb-timeline")`、`list.className = "kb-timeline"`、`fmtTime(v.first_seen)`
  - `kbEmbedderLabel(data.embedder)`、`fmtTime(group.first_seen)`
  - view 切换守卫三段 `state.view !== "kb"` 判断
- API 返回 group 结构：`{snapshot_id, client, model, first_seen, last_seen, session_count, hits: [{kind, title, text, score}]}`；`first_seen` 为 ISO 字符串（`new Date()` 可解析，同格式字典序=时间序）
- `t(key, params)` 支持 `{var}` 插值

---

### Task 1: i18n 词条（en + zh-CN）

**Files:**
- Modify: `claude_tap/dashboard.html`（en 字典 1412-1419 行、zh-CN 字典 1542-1549 行附近）
- Test: `tests/prompt_kb/test_kb_page.py`

**Interfaces:**
- Produces: i18n key `kb_min_score` / `kb_expand` / `kb_collapse` / `kb_latest` / `kb_history_snapshots` / `kb_summary` / `kb_summary_folded` / `kb_no_results_title` / `kb_no_results_hint` / `kb_filtered_by_score`（后续 Task 的 JS 通过 `t("...")` 使用）

- [ ] **Step 1: 写失败测试**

在 `tests/prompt_kb/test_kb_page.py` 追加：

```python
def test_kb_new_i18n_entries_bilingual():
    html = read_dashboard_template()
    for key in (
        "kb_min_score", "kb_expand", "kb_collapse", "kb_latest",
        "kb_history_snapshots", "kb_summary", "kb_summary_folded",
        "kb_no_results_title", "kb_no_results_hint", "kb_filtered_by_score",
    ):
        assert f"{key}:" in html, key
    # zh-CN
    for text in ("最低分数", "展开全文", "收起", "最新版", "个历史快照",
                 "找到 {n} 组结果", "知识库只收录"):
        assert text in html, text
    # en
    for text in ("Min score", "Show full text", "Collapse", "Latest",
                 "history snapshot", "result group"):
        assert text in html, text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_kb_page.py::test_kb_new_i18n_entries_bilingual -v`
Expected: FAIL（`kb_min_score:` not in html）

- [ ] **Step 3: 加 i18n 词条**

en 字典（1419 行 `kb_unavailable: "Knowledge base unavailable: "` 后追加逗号并新增）：

```javascript
    kb_unavailable: "Knowledge base unavailable: ",
    kb_min_score: "Min score",
    kb_expand: "Show full text",
    kb_collapse: "Collapse",
    kb_latest: "Latest",
    kb_history_snapshots: "{n} history snapshot(s)",
    kb_summary: "{n} result group(s)",
    kb_summary_folded: ", {m} folded as history",
    kb_no_results_title: "No matching results",
    kb_no_results_hint: "The KB only indexes system prompts and tool definitions, not chat content. Try a tool name or lower the min score.",
    kb_filtered_by_score: "{n} hit(s) hidden by the min-score filter"
```

zh-CN 字典（1549 行 `kb_unavailable: "知识库不可用："` 同样处理）：

```javascript
    kb_unavailable: "知识库不可用：",
    kb_min_score: "最低分数",
    kb_expand: "展开全文",
    kb_collapse: "收起",
    kb_latest: "最新版",
    kb_history_snapshots: "{n} 个历史快照",
    kb_summary: "找到 {n} 组结果",
    kb_summary_folded: "，其中 {m} 组已折叠为历史快照",
    kb_no_results_title: "没有命中的结果",
    kb_no_results_hint: "知识库只收录 system prompt 和工具定义，不含聊天内容。试试搜工具名，或调低最低分数。",
    kb_filtered_by_score: "{n} 条结果被最低分数过滤掉了"
```

注意：改动前 `kb_unavailable` 行末尾没有逗号，新增时先给该行补逗号。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_kb_page.py -v`
Expected: 全部 PASS（含既有锚点测试）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/dashboard.html tests/prompt_kb/test_kb_page.py
git commit -m "feat(prompt_kb): 知识库页新增重构所需 i18n 词条"
```

---

### Task 2: 控制区 HTML + 全部新 CSS

**Files:**
- Modify: `claude_tap/dashboard.html`（CSS 998-1003 行替换、HTML 1185-1198 行替换）
- Test: `tests/prompt_kb/test_kb_page.py`

**Interfaces:**
- Produces: DOM 元素 `#kb-min-score` / `#kb-min-score-val`；CSS 类 `.kb-status-chips` `.kb-chip` `.kb-summary` `.kb-empty` `.kb-group-header` `.kb-latest-badge` `.kb-group-meta` `.kb-hit-head` `.kb-hit-title` `.kb-hit-kind.tool/.prompt` `.kb-score` `.kb-score-bar` `.kb-score-fill` `.kb-score-num` `.score-high/.score-mid/.score-low` `.kb-hit-text(.full)` `.kb-expand-btn` `.kb-history-fold` `.kb-timeline li.kb-timeline-current` `.kb-btn-primary` `.kb-btn-secondary` `.kb-timeline-btn`（Task 4 的 JS 渲染依赖这些类名）

- [ ] **Step 1: 写失败测试**

在 `tests/prompt_kb/test_kb_page.py` 追加：

```python
def test_kb_controls_have_min_score_slider():
    html = read_dashboard_template()
    assert 'id="kb-min-score"' in html
    assert 'type="range"' in html
    assert 'id="kb-min-score-val"' in html


def test_kb_new_css_classes_present():
    html = read_dashboard_template()
    for cls in (
        ".kb-status-chips", ".kb-chip", ".kb-summary", ".kb-empty",
        ".kb-group-header", ".kb-latest-badge", ".kb-group-meta",
        ".kb-hit-head", ".kb-hit-kind.tool", ".kb-hit-kind.prompt",
        ".kb-score-bar", ".kb-score-fill", ".score-high", ".score-mid",
        ".score-low", ".kb-hit-text", ".kb-expand-btn", ".kb-history-fold",
        ".kb-timeline-current", ".kb-btn-primary", ".kb-btn-secondary",
    ):
        assert cls in html, cls
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_kb_page.py::test_kb_controls_have_min_score_slider tests/prompt_kb/test_kb_page.py::test_kb_new_css_classes_present -v`
Expected: FAIL

- [ ] **Step 3a: 替换 CSS（998-1003 行的 4 条 `.kb-*` 规则整体替换为）**

```css
.kb-view { display: flex; flex-direction: column; gap: 12px; }
.kb-controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.kb-controls input[type="search"] {
  flex: 1 1 240px; min-width: 180px; padding: 6px 10px; font-size: 13px;
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--bg-card); color: var(--text);
}
.kb-controls select {
  padding: 6px 8px; font-size: 13px; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--bg-card); color: var(--text);
}
.kb-min-score-label { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-secondary); }
.kb-min-score-val { font-family: var(--mono); font-size: 12px; min-width: 34px; color: var(--text); }
.kb-btn-primary {
  padding: 6px 14px; font-size: 13px; border-radius: var(--radius-sm);
  border: 1px solid var(--blue); background: var(--blue); color: #fff;
}
.kb-btn-primary:hover { filter: brightness(1.1); }
.kb-btn-secondary {
  padding: 6px 12px; font-size: 13px; border-radius: var(--radius-sm);
  border: 1px solid var(--border); background: var(--bg-card); color: var(--text-secondary);
}
.kb-btn-secondary:hover { border-color: var(--blue); color: var(--blue); }
.kb-timeline-btn { align-self: flex-start; }
.kb-status-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.kb-chip {
  font-size: 11px; font-family: var(--mono); padding: 2px 8px; border-radius: 10px;
  border: 1px solid var(--border); background: var(--bg-card); color: var(--text-secondary);
}
.kb-results { display: flex; flex-direction: column; gap: 12px; }
.kb-summary { font-size: 13px; color: var(--text-secondary); }
.kb-empty {
  padding: 32px 16px; text-align: center; color: var(--text-secondary);
  border: 1px dashed var(--border); border-radius: var(--radius-sm); font-size: 13px;
}
.kb-empty .kb-empty-title { font-size: 15px; margin-bottom: 6px; color: var(--text); }
.kb-group {
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: 12px 14px; background: var(--bg-card);
  display: flex; flex-direction: column; gap: 10px;
}
.kb-group-header { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.kb-group-title { font-weight: 600; font-size: 14px; }
.kb-latest-badge {
  font-size: 10px; padding: 1px 6px; border-radius: 8px;
  background: var(--green-bg); color: var(--green); font-weight: 600;
}
.kb-group-meta { font-size: 12px; color: var(--text-tertiary); }
.kb-hit { display: flex; flex-direction: column; gap: 4px; padding: 8px 10px; border-radius: var(--radius-sm); background: var(--bg); }
.kb-hit-head { display: flex; align-items: center; gap: 8px; }
.kb-hit-kind { font-size: 11px; border-radius: 8px; padding: 0 6px; font-weight: 600; }
.kb-hit-kind.tool { background: var(--blue-bg); color: var(--blue); }
.kb-hit-kind.prompt { background: var(--purple-bg); color: var(--purple); }
.kb-hit-title { font-size: 13px; font-weight: 600; font-family: var(--mono); }
.kb-score { margin-left: auto; display: flex; align-items: center; gap: 6px; }
.kb-score-bar { width: 80px; height: 6px; border-radius: 3px; background: var(--bg-hover); overflow: hidden; }
.kb-score-fill { display: block; height: 100%; border-radius: 3px; }
.score-high .kb-score-fill { background: var(--green); }
.score-high .kb-score-num { color: var(--green); }
.score-mid .kb-score-fill { background: var(--blue); }
.score-mid .kb-score-num { color: var(--blue); }
.score-low .kb-score-fill { background: var(--text-tertiary); }
.score-low .kb-score-num { color: var(--text-tertiary); }
.kb-score-num { font-family: var(--mono); font-size: 12px; }
.kb-hit-text { font-size: 12px; color: var(--text-secondary); line-height: 1.5; }
.kb-hit-text.full { font-family: var(--mono); white-space: pre-wrap; }
.kb-expand-btn {
  align-self: flex-start; border: none; background: none; padding: 0;
  font-size: 12px; color: var(--blue); cursor: pointer;
}
.kb-history-fold > summary { cursor: pointer; font-size: 12px; color: var(--text-secondary); }
.kb-history-fold .kb-group { margin-top: 8px; }
.kb-timeline { list-style: none; margin: 4px 0 0; padding: 0 0 0 12px; border-left: 2px solid var(--border); }
.kb-timeline li { position: relative; padding: 3px 0 3px 10px; font-size: 12px; color: var(--text-secondary); }
.kb-timeline li::before {
  content: ""; position: absolute; left: -17px; top: 9px;
  width: 8px; height: 8px; border-radius: 50%; background: var(--border);
}
.kb-timeline li.kb-timeline-current { color: var(--text); font-weight: 600; }
.kb-timeline li.kb-timeline-current::before { background: var(--green); }
```

- [ ] **Step 3b: 替换控制区 HTML（1186-1195 行 `.kb-controls` 块整体替换为）**

```html
    <div class="kb-controls">
      <input id="kb-query" type="search" data-i18n-placeholder="kb_search_placeholder" placeholder="搜索 prompt 规则或工具定义…">
      <select id="kb-kind">
        <option value="" data-i18n="kb_kind_all">全部</option>
        <option value="tool" data-i18n="kb_kind_tool">工具</option>
        <option value="prompt_section" data-i18n="kb_kind_prompt">Prompt 段落</option>
      </select>
      <label class="kb-min-score-label">
        <span data-i18n="kb_min_score">最低分数</span>
        <input id="kb-min-score" type="range" min="0" max="1" step="0.05" value="0">
        <span id="kb-min-score-val" class="kb-min-score-val">0.00</span>
      </label>
      <button id="kb-search-btn" class="kb-btn-primary" data-i18n="kb_search_btn">搜索</button>
      <button id="kb-reindex-btn" class="kb-btn-secondary" data-i18n="kb_reindex_btn">重建索引</button>
    </div>
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_kb_page.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add claude_tap/dashboard.html tests/prompt_kb/test_kb_page.py
git commit -m "feat(prompt_kb): 知识库页控制区改造与卡片样式全套 CSS"
```

---

### Task 3: 归并/过滤纯逻辑（JS + Python 复制算法测试）

**Files:**
- Modify: `claude_tap/dashboard.html`（`renderKbResults` 之前插入新函数，约 2397 行处）
- Test: `tests/prompt_kb/test_kb_render_logic.py`（新建）

**Interfaces:**
- Produces（Task 4/5 依赖的 JS 全局函数签名）:
  - `kbBestScore(group) -> number`
  - `kbFilterHits(group, minScore) -> group`（浅拷贝，hits 过滤）
  - `kbFoldGroups(groups) -> [{main, history}]`（同 client+model 归并，main=first_seen 最新，按 main 最高分排序）
  - `kbScoreClass(score) -> "score-high"|"score-mid"|"score-low"`（≥0.9 / ≥0.7 / 其余）

- [ ] **Step 1: 写失败测试（Python 复制算法，js-in-html-testing Layer 1）**

新建 `tests/prompt_kb/test_kb_render_logic.py`：

```python
"""Python replica of the dashboard KB fold/filter pipeline.

Mirrors kbBestScore / kbFilterHits / kbFoldGroups / kbScoreClass in
claude_tap/dashboard.html (js-in-html-testing Layer 1).
"""

from claude_tap.dashboard import read_dashboard_template


def best_score(group):
    return max((h["score"] for h in group.get("hits", [])), default=0.0)


def filter_hits(group, min_score):
    return {**group, "hits": [h for h in group.get("hits", []) if h["score"] >= min_score]}


def fold_groups(groups):
    by_key = {}
    for g in groups:
        by_key.setdefault((g["client"], g["model"]), []).append(g)
    folded = []
    for lst in by_key.values():
        ordered = sorted(lst, key=lambda g: g["first_seen"], reverse=True)
        folded.append({"main": ordered[0], "history": ordered[1:]})
    folded.sort(key=lambda f: best_score(f["main"]), reverse=True)
    return folded


def score_class(score):
    if score >= 0.9:
        return "score-high"
    if score >= 0.7:
        return "score-mid"
    return "score-low"


def _group(client, model, first_seen, scores):
    return {
        "client": client, "model": model, "first_seen": first_seen,
        "hits": [{"kind": "tool", "title": "T", "text": "x", "score": s} for s in scores],
    }


def test_js_functions_exist_in_template():
    html = read_dashboard_template()
    for fn in ("kbBestScore", "kbFilterHits", "kbFoldGroups", "kbScoreClass"):
        assert f"function {fn}(" in html, fn


def test_fold_merges_same_client_model_newest_first():
    groups = [
        _group("claude", "k3", "2026-08-05T10:00:00", [0.8]),
        _group("claude", "k3", "2026-08-06T16:00:00", [0.9]),
        _group("codex", "gpt5", "2026-08-04T09:00:00", [0.7]),
    ]
    folded = fold_groups(groups)
    assert len(folded) == 2
    claude = next(f for f in folded if f["main"]["client"] == "claude")
    assert claude["main"]["first_seen"] == "2026-08-06T16:00:00"
    assert [g["first_seen"] for g in claude["history"]] == ["2026-08-05T10:00:00"]
    codex = next(f for f in folded if f["main"]["client"] == "codex")
    assert codex["history"] == []


def test_fold_orders_by_main_best_score():
    groups = [
        _group("claude", "k3", "2026-08-06T16:00:00", [0.75]),
        _group("codex", "gpt5", "2026-08-04T09:00:00", [0.95]),
    ]
    folded = fold_groups(groups)
    assert folded[0]["main"]["client"] == "codex"


def test_filter_hits_drops_below_threshold():
    g = _group("claude", "k3", "2026-08-06T16:00:00", [0.82, 0.91, 0.65])
    assert [h["score"] for h in filter_hits(g, 0.7)["hits"]] == [0.82, 0.91]
    assert filter_hits(g, 0.95)["hits"] == []
    assert [h["score"] for h in filter_hits(g, 0.0)["hits"]] == [0.82, 0.91, 0.65]


def test_score_class_thresholds():
    assert score_class(0.95) == "score-high"
    assert score_class(0.9) == "score-high"
    assert score_class(0.82) == "score-mid"
    assert score_class(0.7) == "score-mid"
    assert score_class(0.69) == "score-low"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_kb_render_logic.py -v`
Expected: `test_js_functions_exist_in_template` FAIL，其余 PASS（纯 Python 复制逻辑）

- [ ] **Step 3: 实现 JS 函数**

在 `renderKbResults` 函数定义之前插入：

```javascript
function kbBestScore(group) {
  return Math.max(0, ...(group.hits || []).map(h => h.score));
}

function kbFilterHits(group, minScore) {
  return { ...group, hits: (group.hits || []).filter(h => h.score >= minScore) };
}

function kbFoldGroups(groups) {
  const byKey = new Map();
  for (const g of groups) {
    const key = `${g.client}\n${g.model}`;
    const list = byKey.get(key) || [];
    list.push(g);
    byKey.set(key, list);
  }
  const folded = [];
  for (const list of byKey.values()) {
    const sorted = [...list].sort(
      (a, b) => new Date(b.first_seen).getTime() - new Date(a.first_seen).getTime()
    );
    folded.push({ main: sorted[0], history: sorted.slice(1) });
  }
  folded.sort((a, b) => kbBestScore(b.main) - kbBestScore(a.main));
  return folded;
}

function kbScoreClass(score) {
  if (score >= 0.9) return "score-high";
  if (score >= 0.7) return "score-mid";
  return "score-low";
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/prompt_kb/test_kb_render_logic.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add claude_tap/dashboard.html tests/prompt_kb/test_kb_render_logic.py
git commit -m "feat(prompt_kb): 新增搜索结果归并/过滤/分档纯逻辑函数"
```

---

### Task 4: 渲染管线重写（卡片/分数条/展开/折叠/摘要/空状态/时间线/状态徽章）

**Files:**
- Modify: `claude_tap/dashboard.html`（`renderKbResults` 整体重写、新增渲染函数、`kbLoadTimeline` 当前项改 class、`kbLoadStatus` 改 chips、`kbSearch` 409/501 分支、底部事件绑定区加滑块监听）
- Test: `tests/prompt_kb/test_kb_page.py`

**Interfaces:**
- Consumes: Task 1 i18n key、Task 2 DOM/CSS、Task 3 纯函数
- Produces:
  - `kbLastGroups`（模块级 `let`，缓存上次搜索结果）
  - `renderKbResults(results)`（签名不变，`kbSearch` 调用处不用改）
  - `kbRenderFiltered()`（读 `#kb-min-score` 重渲染；滑块 `input` 事件绑定它）
  - `kbRenderGroupCard(group, isLatest, history)` / `kbRenderHit(hit)` / `kbEmptyState(title, hint)`

- [ ] **Step 1: 写失败测试**

在 `tests/prompt_kb/test_kb_page.py` 追加：

```python
def test_kb_render_pipeline_wired():
    html = read_dashboard_template()
    for fn in ("kbRenderFiltered", "kbRenderGroupCard", "kbRenderHit", "kbEmptyState"):
        assert f"function {fn}(" in html, fn
    assert "kbLastGroups" in html
    assert '$("#kb-min-score").addEventListener("input"' in html
    assert "kbFoldGroups(filtered)" in html


def test_kb_timeline_current_uses_class():
    html = read_dashboard_template()
    assert 'item.classList.add("kb-timeline-current")' in html
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/prompt_kb/test_kb_page.py::test_kb_render_pipeline_wired tests/prompt_kb/test_kb_page.py::test_kb_timeline_current_uses_class -v`
Expected: FAIL

- [ ] **Step 3a: 重写 `renderKbResults`（现有 2398-2429 行的整个函数替换为）**

```javascript
let kbLastGroups = [];

function renderKbResults(results) {
  kbLastGroups = results || [];
  kbRenderFiltered();
}

function kbRenderFiltered() {
  const container = $("#kb-results");
  container.innerHTML = "";
  const slider = $("#kb-min-score");
  const minScore = slider ? parseFloat(slider.value) : 0;
  $("#kb-min-score-val").textContent = minScore.toFixed(2);

  if (!kbLastGroups.length) {
    container.appendChild(kbEmptyState(`📭 ${t("kb_no_results_title")}`, t("kb_no_results_hint")));
    return;
  }
  const totalHits = kbLastGroups.reduce((n, g) => n + (g.hits || []).length, 0);
  const filtered = kbLastGroups.map(g => kbFilterHits(g, minScore)).filter(g => g.hits.length);
  const shownHits = filtered.reduce((n, g) => n + g.hits.length, 0);
  const hiddenHits = totalHits - shownHits;
  if (!filtered.length) {
    container.appendChild(kbEmptyState(
      `📭 ${t("kb_no_results_title")}`,
      t("kb_filtered_by_score", { n: hiddenHits })
    ));
    return;
  }
  const folded = kbFoldGroups(filtered);
  const foldedCount = folded.reduce((n, f) => n + f.history.length, 0);
  const summary = document.createElement("div");
  summary.className = "kb-summary";
  summary.textContent = t("kb_summary", { n: folded.length })
    + (foldedCount ? t("kb_summary_folded", { m: foldedCount }) : "")
    + (hiddenHits ? " · " + t("kb_filtered_by_score", { n: hiddenHits }) : "");
  container.appendChild(summary);
  for (const { main, history } of folded) {
    container.appendChild(kbRenderGroupCard(main, true, history));
  }
}

function kbEmptyState(title, hint) {
  const box = document.createElement("div");
  box.className = "kb-empty";
  const titleEl = document.createElement("div");
  titleEl.className = "kb-empty-title";
  titleEl.textContent = title;
  const hintEl = document.createElement("div");
  hintEl.textContent = hint;
  box.append(titleEl, hintEl);
  return box;
}

function kbRenderGroupCard(group, isLatest, history = []) {
  const card = document.createElement("div");
  card.className = "kb-group";
  const header = document.createElement("div");
  header.className = "kb-group-header";
  const title = document.createElement("span");
  title.className = "kb-group-title";
  title.textContent = `${group.client} / ${group.model}`;
  header.appendChild(title);
  if (isLatest) {
    const badge = document.createElement("span");
    badge.className = "kb-latest-badge";
    badge.textContent = t("kb_latest");
    header.appendChild(badge);
  }
  const meta = document.createElement("span");
  meta.className = "kb-group-meta";
  meta.textContent = `${t("kb_first_seen")} ${fmtTime(group.first_seen)} · sessions ${group.session_count}`;
  header.appendChild(meta);
  card.appendChild(header);
  for (const hit of group.hits) card.appendChild(kbRenderHit(hit));

  const timelineBtn = document.createElement("button");
  timelineBtn.className = "kb-btn-secondary kb-timeline-btn";
  timelineBtn.textContent = t("kb_timeline");
  timelineBtn.addEventListener("click", () => kbLoadTimeline(group, card));
  card.appendChild(timelineBtn);

  if (history.length) {
    const fold = document.createElement("details");
    fold.className = "kb-history-fold";
    const summaryEl = document.createElement("summary");
    summaryEl.textContent = `▸ ${t("kb_history_snapshots", { n: history.length })}`;
    fold.appendChild(summaryEl);
    for (const g of history) fold.appendChild(kbRenderGroupCard(g, false, []));
    card.appendChild(fold);
  }
  return card;
}

function kbRenderHit(hit) {
  const row = document.createElement("div");
  row.className = "kb-hit";
  const head = document.createElement("div");
  head.className = "kb-hit-head";
  const badge = document.createElement("span");
  badge.className = `kb-hit-kind ${hit.kind === "tool" ? "tool" : "prompt"}`;
  badge.textContent = hit.kind === "tool" ? t("kb_kind_tool") : t("kb_kind_prompt");
  const title = document.createElement("span");
  title.className = "kb-hit-title";
  title.textContent = hit.title;
  const score = document.createElement("span");
  score.className = `kb-score ${kbScoreClass(hit.score)}`;
  const bar = document.createElement("span");
  bar.className = "kb-score-bar";
  const fill = document.createElement("span");
  fill.className = "kb-score-fill";
  fill.style.width = `${Math.round(Math.max(0, Math.min(1, hit.score)) * 100)}%`;
  bar.appendChild(fill);
  const num = document.createElement("span");
  num.className = "kb-score-num";
  num.textContent = hit.score.toFixed(3);
  score.append(bar, num);
  head.append(badge, title, score);
  row.appendChild(head);

  const PREVIEW = 200;
  const full = hit.text || "";
  const text = document.createElement("div");
  text.className = "kb-hit-text";
  text.textContent = full.length > PREVIEW ? full.slice(0, PREVIEW) + "…" : full;
  row.appendChild(text);
  if (full.length > PREVIEW) {
    const btn = document.createElement("button");
    btn.className = "kb-expand-btn";
    btn.type = "button";
    btn.textContent = `▸ ${t("kb_expand")}`;
    btn.addEventListener("click", () => {
      const expanded = btn.dataset.expanded === "1";
      btn.dataset.expanded = expanded ? "0" : "1";
      text.textContent = expanded ? full.slice(0, PREVIEW) + "…" : full;
      text.classList.toggle("full", !expanded);
      btn.textContent = expanded ? `▸ ${t("kb_expand")}` : `▾ ${t("kb_collapse")}`;
    });
    row.appendChild(btn);
  }
  return row;
}
```

- [ ] **Step 3b: `kbSearch` 的 409/501 分支改用空状态卡片**

把：

```javascript
  if (resp.status === 501 || resp.status === 409) {
    $("#kb-results").textContent = data.hint || data.error;
    return;
  }
```

改为：

```javascript
  if (resp.status === 501 || resp.status === 409) {
    kbLastGroups = [];
    const container = $("#kb-results");
    container.innerHTML = "";
    container.appendChild(kbEmptyState("⚠️", data.hint || data.error));
    return;
  }
```

- [ ] **Step 3c: `kbLoadStatus` 改 chips**

把成功分支（`el.textContent = ...` 那行）改为：

```javascript
  const s = data.stats;
  el.innerHTML = "";
  el.classList.add("kb-status-chips");
  for (const label of [
    kbEmbedderLabel(data.embedder),
    `indexed=${s.indexed} pending=${s.pending} failed=${s.failed}`,
    `snapshots=${s.snapshots}`,
  ]) {
    const chip = document.createElement("span");
    chip.className = "kb-chip";
    chip.textContent = label;
    el.appendChild(chip);
  }
```

- [ ] **Step 3d: `kbLoadTimeline` 当前项改 class**

把 `if (v.id === group.snapshot_id) item.style.fontWeight = "bold";` 改为：

```javascript
    if (v.id === group.snapshot_id) item.classList.add("kb-timeline-current");
```

（`card.querySelector("ul.kb-timeline")`、`list.className = "kb-timeline"`、`fmtTime(v.first_seen)` 三处锚点保持不变。）

- [ ] **Step 3e: 绑定滑块事件**

在底部 `$("#kb-query").addEventListener("keydown", ...)`（约 2871 行）旁边加：

```javascript
$("#kb-min-score").addEventListener("input", () => kbRenderFiltered());
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/prompt_kb/ -v`
Expected: 全部 PASS（含既有锚点测试与新测试）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/dashboard.html tests/prompt_kb/test_kb_page.py
git commit -m "feat(prompt_kb): 重写知识库结果渲染管线(卡片/折叠/过滤/空状态)"
```

---

### Task 5: Playwright 浏览器测试 + 端到端验证

**Files:**
- Test: `tests/prompt_kb/test_kb_render_browser.py`（新建）

**Interfaces:**
- Consumes: Task 4 的 `renderKbResults` / `kbRenderFiltered` / `kbLastGroups` 与全部 CSS 类

- [ ] **Step 1: 写浏览器测试**

新建 `tests/prompt_kb/test_kb_render_browser.py`：

```python
"""Browser tests for the dashboard KB result rendering (js-in-html-testing Layer 2).

Loads the real dashboard template via file://, injects mock search results,
and drives renderKbResults / kbRenderFiltered directly. Top-level dashboard
init fetches fail on file:// but all function declarations are hoisted and
callable.
"""

import json

import pytest

from claude_tap.dashboard import read_dashboard_template

pytest.importorskip("playwright")

LONG_TEXT = "Read tool description. " * 30  # > 200 chars

GROUPS = [
    {
        "snapshot_id": 1, "client": "claude", "model": "k3-256k",
        "first_seen": "2026-08-05T10:28:53", "last_seen": "2026-08-05T10:28:53",
        "session_count": 1,
        "hits": [{"kind": "tool", "title": "Read", "text": LONG_TEXT, "score": 0.823}],
    },
    {
        "snapshot_id": 2, "client": "claude", "model": "k3-256k",
        "first_seen": "2026-08-06T16:34:47", "last_seen": "2026-08-06T16:34:47",
        "session_count": 1,
        "hits": [
            {"kind": "tool", "title": "Read", "text": LONG_TEXT, "score": 0.95},
            {"kind": "prompt_section", "title": "Rules", "text": "short rule", "score": 0.65},
        ],
    },
    {
        "snapshot_id": 3, "client": "codex", "model": "gpt-5",
        "first_seen": "2026-08-04T09:00:00", "last_seen": "2026-08-04T09:00:00",
        "session_count": 2,
        "hits": [{"kind": "tool", "title": "apply_patch", "text": "patch files", "score": 0.72}],
    },
]


def _build_html(tmp_path):
    inject = f"<script>window.__KB_TEST_GROUPS = {json.dumps(GROUPS)};</script>"
    path = tmp_path / "kb_dashboard.html"
    path.write_text(read_dashboard_template().replace("</head>", inject + "</head>", 1), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    from playwright.sync_api import sync_playwright

    html = _build_html(tmp_path_factory.mktemp("kb"))
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    pg = browser.new_page()
    pg.goto(f"file://{html}")
    pg.evaluate("renderKbResults(window.__KB_TEST_GROUPS)")
    yield pg
    browser.close()
    pw.stop()


def test_same_client_model_folded_into_latest_card(page):
    # 3 groups, 2 share client+model -> 2 top-level cards, 1 history fold
    assert page.eval_on_selector_all("#kb-results > .kb-group", "els => els.length") == 2
    assert page.eval_on_selector_all("#kb-results > .kb-group .kb-history-fold", "els => els.length") == 1
    # latest card is the 2026-08-06 snapshot with the "latest" badge
    assert page.eval_on_selector_all("#kb-results > .kb-group .kb-latest-badge", "els => els.length") == 2
    summary = page.eval_on_selector(".kb-summary", "el => el.textContent")
    assert "2" in summary  # 2 result groups after folding


def test_score_bar_badge_and_tier_classes(page):
    assert page.eval_on_selector_all(".kb-score.score-high", "els => els.length") >= 1
    assert page.eval_on_selector_all(".kb-score.score-mid", "els => els.length") >= 1
    assert page.eval_on_selector_all(".kb-score.score-low", "els => els.length") >= 1
    assert page.eval_on_selector_all(".kb-hit-kind.tool", "els => els.length") >= 1
    assert page.eval_on_selector_all(".kb-hit-kind.prompt", "els => els.length") >= 1
    width = page.eval_on_selector(".kb-score-fill", "el => el.style.width")
    assert width.endswith("%") and width != "0%"


def test_expand_full_text_toggles(page):
    before = page.eval_on_selector(".kb-hit-text", "el => el.textContent.length")
    page.click(".kb-expand-btn")
    after = page.eval_on_selector(".kb-hit-text", "el => el.textContent.length")
    assert after > before
    assert page.eval_on_selector(".kb-hit-text", "el => el.classList.contains('full')")
    page.click(".kb-expand-btn")
    assert page.eval_on_selector(".kb-hit-text", "el => el.textContent.length") == before


def test_min_score_slider_filters_without_refetch(page):
    page.evaluate("""() => {
        const slider = document.querySelector('#kb-min-score');
        slider.value = '0.8';
        kbRenderFiltered();
    }""")
    # 0.95 and 0.823 survive; 0.65 / 0.72 filtered out -> codex card gone entirely
    titles = page.eval_on_selector_all(".kb-hit-title", "els => els.map(e => e.textContent)")
    assert "apply_patch" not in titles
    assert page.eval_on_selector_all("#kb-results > .kb-group", "els => els.length") == 1
    # summary reports the min-score filter kicked in (zh default lang)
    summary = page.eval_on_selector(".kb-summary", "el => el.textContent")
    assert "过滤" in summary
    page.evaluate("""() => {
        const slider = document.querySelector('#kb-min-score');
        slider.value = '0';
        kbRenderFiltered();
    }""")
    assert page.eval_on_selector_all("#kb-results > .kb-group", "els => els.length") == 2


def test_empty_state_shows_guidance(page):
    page.evaluate("renderKbResults([])")
    assert page.eval_on_selector_all(".kb-empty", "els => els.length") == 1
    hint = page.eval_on_selector(".kb-empty", "el => el.textContent")
    assert len(hint) > 10
```

- [ ] **Step 2: 跑浏览器测试**

Run: `uv run pytest tests/prompt_kb/test_kb_render_browser.py -v`
Expected: 全部 PASS。若 `renderKbResults` 因模板初始化报错不可得，确认是 file:// 下 fetch 失败的运行时错误（函数声明已提升不受影响）；若是语法错误则修复 Task 4 代码。

- [ ] **Step 3: 跑 e2e 验证**

调用 `e2e-test` 技能跑 claude-tap 端到端测试（知识库页真实搜索 → 渲染 → 折叠展开 → 时间线）。

- [ ] **Step 4: 全量回归**

Run: `uv run pytest tests/ --ignore=tests/test_e2e.py -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/prompt_kb/test_kb_render_browser.py
git commit -m "test(prompt_kb): 知识库渲染的 Playwright 浏览器测试"
```

---

## Self-Review 记录

- Spec 覆盖：控制区滑块/次要按钮 ✅(T2)、状态徽章 ✅(T4-3c)、结果卡片/分数分档 ✅(T2 CSS + T4)、展开全文 ✅(T4)、快照折叠+摘要 ✅(T3+T4)、空状态 ✅(T4)、时间线美化 ✅(T2 CSS + T4-3d)、i18n ✅(T1)、测试双层 ✅(T3 Layer1 + T5 Layer2)、e2e ✅(T5)
- 兼容锚点：逐条核对保留（见 Global Constraints）
- 类型一致：`kbFoldGroups` 返回 `{main, history}`，T4 解构一致；`kbFilterHits` 浅拷贝保留 `snapshot_id`（时间线高亮依赖）；`kbRenderGroupCard(main, true, history)` 签名 T4/T5 一致
- 无占位符
