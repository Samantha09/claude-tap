# 知识库混合检索与重排（FTS5 hybrid + cross-encoder reranker）设计文档

日期：2026-08-13
状态：已确认需求与设计，待实现

## 背景

prompt 知识库检索已完成"聊天内容为主要依据"的改造（见
`2026-08-12-chat-content-search-design.md`），但仍是**单通道稠密向量检索**。
实测与复盘中确认的短板：

1. **无关键词通道**：搜工具名、错误码、函数名等必须字面命中的查询时，纯向量天然弱
   （中文查询"沙箱 shell"下 `getDiagnostics` 0.850 压过 `Bash` 0.826）
2. **分数不可校准**：e5-small 余弦分挤在 0.79–0.90 窄带，绝对阈值无意义，
   只能靠 `rel_delta` 相对截断变通
3. **排序质量卡在 embedding 模型上限**：向量是"分别编码再比对"，查询与文档的
   逐词交互信息在压缩中丢失

用户拍板：同时实现 **FTS5 混合检索**（关键词 + 向量 RRF 融合）与
**本地 cross-encoder reranker**（召回后重排）。

### 已确认的决策（与用户逐项确认）

| 决策点 | 结论 |
|--------|------|
| FTS5 中文分词 | 双路并行：trigram 表（子串/英文）+ jieba 预切分表（中文词级），两通道独立打分再融合 |
| reranker 模型 | BAAI/bge-reranker-base（~280MB，中英双语训练） |
| 架构方案 | 方案 B：新模块 tokenize.py / rerank.py 分离职责，store.py 管 FTS 表，search.py 只做管线编排 |
| 融合算法 | RRF（k=60），三通道平等，各取 top 20 排名 |
| 降级策略 | reranker 不可用/关闭 → 跳过重排不报错，结果带 `reranked: false` 标记 |
| rel_delta | 参数保留兼容；reranker 可用时忽略，降级时回退现有相对截断行为 |

## 目标与非目标

### 目标

1. 字面命中查询（工具名、错误码、函数名）进入候选并排到前列
2. 中文 2 字词查询（截图、沙箱、取消……）可被关键词通道检索
3. 最终分数经 cross-encoder 重排校准，`min_score` 恢复语义
4. 全链路优雅降级：缺 jieba / 缺 reranker 模型 / 缺 sentence-transformers 时检索仍可用
5. 存量数据自动迁移回填，用户零操作；jieba 后装可重建升级

### 非目标

- 不换 embedding 模型（e5-small 与现有向量空间不动，不重索引向量）
- 不做 ANN 向量索引（HNSW 等）；数据量未达阈值
- 不动 chunking 策略与快照采集
- 不删除 `rel_delta` 参数（仅语义降级）
- 不做查询改写 / HyDE（agentic 架构下属调用方职责）

## 设计

### §1 Schema 与 FTS 同步（store.py）

新增 4 张 FTS5 虚表，覆盖 chunks 与 messages 两类内容：

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_chunks_tri USING fts5(text, content='', tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_chunks_jieba USING fts5(text, content='', tokenize='unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_messages_tri USING fts5(text, content='', tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_messages_jieba USING fts5(text, content='', tokenize='unicode61');
```

- **contentless 模式**（`content=''`）：FTS 表只存倒排索引不存原文；rowid 与
  `kb_chunks.id` / `kb_messages.id` 一一对应，正文只存主表
- tri 表存原文；jieba 表存 `segment()` 预切分文本（词间空格）
- 同步点（与主表写操作同事务，FTS 异常 → 整体回滚报错，不静默）：
  - `replace_chunks()`：先按 snapshot 关联的 chunk id 删旧 FTS 行再插新行
  - `upsert_message()`：仅新插入时写 FTS（dedup 命中不动）
  - `delete_messages_for_session()`：同步删 FTS 行
  - contentless 表删除用 `INSERT INTO fts(fts, rowid, text) VALUES('delete', ?, ?)`
    特殊命令（需传入被删原文才能维护索引）
- 查询入口：`store.fts_rank(table, match_query, limit) -> list[(rowid, score)]`，
  用内置 `bm25()` 打分，归一化为正相关分（分数越高越相关）
- **迁移**：`_migrate()` 幂等分支，`kb_meta` 的 `fts_backfilled` 标记守卫；
  未标记时全量扫描主表回填 4 张表。jieba 未装时 jieba 表回填降级为原样文本；
  回填中途失败不写标记，下次打开重试（DELETE + INSERT 全量重来，无中间态）
- **重建**：`rebuild_fts()` 方法清空 4 张表全量重建（jieba 后装升级、索引损坏自救）

### §2 分词模块（tokenize.py，新文件）

```python
def segment(text: str) -> str:
    """jieba cut_for_search 切词、空格连接；jieba 缺失时降级返回原文。"""
```

- 懒加载：首次调用时 `import jieba`，ImportError → 永久降级返回原文
  （一次性 warning，不刷屏）；不在模块顶部 import，`claude-tap[rag]` 未装时 KB 不受影响
- 用 `cut_for_search` 搜索引擎模式：多切出子词，召回更全，索引略大
- 入库侧（store.py 写 FTS）与查询侧（search.py 构造 MATCH）**必须共用此函数**，
  不允许各自实现（切分不一致 = 倒排索引条目对不上）
- 无状态纯函数

依赖变更：`pyproject.toml` 的 `rag` extra 增加 `jieba>=0.42`（纯 Python，离线可装，不进主依赖）。

### §3 召回与融合管线（search.py 改造）

`search()` / `search_messages()` 改为三阶段管线，**对外签名与返回结构不变**
（`SnapshotResult` / `SessionResult` 不动，live.py / MCP / CLI / dashboard 零改动）：

```
查询
 ├─ 通道1：向量余弦（现有逻辑，取 top recall 排名）
 ├─ 通道2：FTS trigram 表 BM25（原文 MATCH，取 top recall 排名）
 └─ 通道3：FTS jieba 表 BM25（segment(查询) MATCH，取 top recall 排名）
        ↓
RRF 融合：score(d) = Σ 1/(60 + rank_i(d))，三通道平等
        ↓
跨快照去重（chunks 路径，现有逻辑上移：只对召回候选去重，session_count 累加不变）
        ↓
融合后 top recall 候选 → reranker 重排（§4，可用时）
        ↓
现有收尾：分组（快照/session）、组内 hits top 3、limit 截断
```

- 新 keyword-only 参数 `recall: int = 20`：每通道召回数与进 reranker 的候选数
- 通道容错：FTS 表为空/不存在、MATCH sanitize 后无词项 → 该通道贡献为零；
  向量通道照旧必跑；任一通道失败不拖垮整体
- MATCH sanitize：抽取 `[A-Za-z0-9_一-鿿]+` 词项以 `OR` 连接；
  jieba 通道先 `segment()` 再抽取；空词项跳过该通道
- **min_score 语义**：reranker 可用时作用于重排分（0–1 校准分）；
  降级时作用于向量余弦分（现状）。RRF 分不对外暴露，仅内部排序
- **rel_delta**：reranker 可用时忽略；降级时回退现有相对截断。参数保留，签名兼容
- `search_messages()` 同构改造，无跨快照去重步骤（维持现状）

### §4 Reranker 模块（rerank.py，新文件）

```python
class RerankerUnavailable(Exception): ...

class Reranker(Protocol):
    name: str
    def rerank(self, query: str, texts: list[str]) -> list[float]: ...

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
```

- `LocalReranker`：`sentence_transformers.CrossEncoder` 加载 bge-reranker-base；
  `rerank()` 批量 `predict([[query, text], ...])`，逐分过 sigmoid 归一化到 0–1
- 加载异常处理复刻 `LocalEmbedder` 模式：任何加载/下载异常 → `RerankerUnavailable`，
  不让裸异常漏给上层
- `name = f"reranker:{model_name}"` 写入 `kb_meta`（诊断用，不触发 reindex——
  重排不改向量空间）
- 配置（KbConfig 扩展，TOML + 环境变量双通道）：

  ```toml
  [prompt_kb]
  reranker = "on"            # "on" | "off"，默认 on
  reranker_model = "BAAI/bge-reranker-base"
  ```

  环境变量：`CLAUDE_TAP_KB_RERANKER`、`CLAUDE_TAP_KB_RERANKER_MODEL`
- 降级链（search.py 编排处）：
  1. `reranker = "off"` → 跳过重排，最终分 = 向量余弦分
  2. `RerankerUnavailable` → 同上
  3. `rerank()` 调用抛异常 → 同上兜底
  降级时结果照常返回并带 `reranked: false` 标记
- 延迟预算：top 20 候选一次批量 predict，CPU 约 100–300ms；一次搜索一次调用
- 测试用 `FakeReranker`（tests/prompt_kb/）：对标 FakeEmbedder，
  按词面重叠出确定性分数

### §5 对外 API 面变化

纯增量字段，向后兼容：

- **MCP `kb_search`**：响应顶层加 `reranked: bool`；`rel_delta` 参数保留；
  docstring 更新（三通道、reranked 字段、rel_delta 仅降级时生效）。
  `kb_status` 加 reranker 状态（模型名 / off / unavailable）
- **HTTP API（live.py `_handle_kb_search`）**：透传 `reranked` 字段；
  顺手补齐上轮遗留的 `rel_delta` 查询参数支持
- **CLI**：`kb search` 输出头部加 `reranked: yes/no`；新增子命令
  **`claude-tap kb rebuild-fts`**（幂等可重入）；`kb stats` 加 reranker 状态
- **Dashboard 知识库页**：`reranked: false` 时结果区顶部显示降级提示
  （zh "重排不可用，按融合分数排序" / en "Reranker unavailable, fused ranking"）；
  搜索框提示语更新（zh "支持关键词与语义混合搜索" / en "Hybrid keyword + semantic search"）；
  i18n key 中英双份，沿用 `kb_*` 命名
- **README**：`claude-tap[rag]` 说明补 jieba 依赖与 rebuild-fts 用法

### §6 错误处理与成本

| 场景 | 行为 |
|------|------|
| FTS 同步写失败 | 与主表同事务回滚，报错不静默 |
| jieba 未装 | jieba 通道降级为原文切分（英文可用）；rebuild-fts 可升级 |
| sentence-transformers 未装 | 向量与重排不可用，FTS 通道独立工作（现有 EmbedderUnavailable 语义） |
| reranker 模型加载失败 | 跳过重排，`reranked: false` |
| MATCH 语法错误（特殊字符查询） | sanitize 后为空 → 跳过该通道 |
| FTS 表不存在（老库异常） | 视为空通道，不报错 |

成本预估：4 张 contentless 倒排索引约为正文体积 30–50%（当前库 <1MB）；
搜索延迟 ~10ms → ~150–350ms（reranker 占大头）；索引写入多一次 jieba 切词（毫秒级）。

### §7 测试策略

| 层 | 内容 |
|---|---|
| `test_tokenize.py` | 中文切词、中英混合、jieba 缺失降级（monkeypatch import） |
| `test_store.py` / `test_store_messages.py` | FTS 随行同步（增/删/replace）；老库迁移回填幂等；rebuild_fts |
| `test_search.py` / `test_search_messages.py` | FakeEmbedder + FakeReranker：RRF 融合排序、字面命中召回（纯向量排不上的工具名被 FTS 捞回）、通道容错、reranked 标记、降级路径、rel_delta 兼容 |
| `test_rerank.py` | FakeReranker 契约、sigmoid 归一化、RerankerUnavailable 转换 |
| `test_mcp_server.py` / `test_kb_api.py` / `test_cli.py` | reranked 字段、rebuild-fts 子命令、kb_status/stats reranker 状态 |
| dashboard | i18n key 锚点、降级提示渲染逻辑单测 |
| 端到端 | MCP stdio 冒烟回归；真库 rebuild-fts + 7 组参考查询对比（验收：Q1 中文"沙箱 shell" Bash 进 top；Q5 CronDelete 字面命中仍居首；reranked: true） |

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| jieba 切词与查询侧不一致导致漏召 | 两侧强制共用 tokenize.segment()，代码评审检查点 |
| trigram 索引膨胀 | contentless 模式 + 当前库体量小；5 万 chunk 阈值内无压力 |
| reranker 下载失败（网络/TLS） | RerankerUnavailable 降级链 + 模型缓存复用现有 modelscope 路径 |
| RRF 三通道平等未必最优 | 权重参数化预留（默认平等），实测后可调 |
| 双 FTS 表同步遗漏导致索引漂移 | 同步点与主表同事务；rebuild-fts 兜底；迁移回填幂等 |
