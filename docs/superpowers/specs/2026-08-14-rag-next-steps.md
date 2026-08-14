# RAG 后续开发点（backlog）

**日期**：2026-08-14
**状态**：backlog —— 记录已评估但未排期的方向，每条附背景与切入点

本文档汇总 prompt 知识库（方向 D）与 trace 语义检索（方向 A）完成后的后续事项。
已完成的主线见各 spec（2026-08-06 prompt-kb / 2026-08-10 trace-semantic-search /
2026-08-11 kb-mcp-server / 2026-08-12 chat-content-search / 2026-08-13 hybrid-search-reranker）。

## 中等（一两天）

### 1. modelscope 一等支持（模型来源配置化）

**背景**：本机 DLP 代理下 HF 直连与 hf-mirror 的大文件下载均停滞（小文件可下，
model.safetensors 挂起），embedder/reranker 目前靠 `config.toml` 指向 modelscope
本地路径绕过（2026-08-13 spec 遗留 1）。2026-08-14 `~/.cache/modelscope` 被整体删除后
重新下载到 hub 布局，顺带暴露一个痛点：**embedder_name 含模型路径**，路径一变即触发
ReindexRequired 全量重建。

**切入点**：
- `embed.py` / `rerank.py` 增加 `source = "modelscope"` 配置项，内部走 modelscope 下载与缓存解析
- embedder 身份（embedder_name）改为「模型名 + 变体」而非路径，避免换路径强制 reindex

### 2. RRF 权重调优

**背景**：三通道（向量/trigram/jieba）RRF 融合默认平等权重，spec 已预留参数化
（2026-08-13 spec 风险表）。实测关键词独有召回有效，但权重未调过。

**切入点**：`search.py` `_rrf_fuse` 加权重参数；用真实查询集（2026-08-13 spec
验证记录的 7 组）做前后对比。

### 3. 去重共享行的 occurrences 表（方向 A 让步的正解）

**背景**：`kb_messages` 按 `(content_hash, client, role)` 跨会话去重，`session_id`
保留首现会话；删除首现会话时共享行整体消失，该文本在历史中不再可搜
（2026-08-10 spec 文档化的让步，当时拍板接受）。

**切入点**：新增 `kb_message_occurrences(message_id, session_id, seen_at)` 表；
级联删除只删 occurrence，message 行在无 occurrence 时才清除。涉及 schema 迁移 +
extract/store/删除路径，工作量中等。

## 大方向（roadmap 暂缓）

### 4. 方向 B：claude-tap 作为 RAG 数据源

定位成 agent 行为语料的采集/清洗管道，导出结构化语料给外部 RAG 或微调数据集
（Phistory 是雏形）。

### 5. 方向 C：RAG 增强代理

proxy 转发前从历史 trace 检索片段注入上下文，给 CLI 加跨会话记忆。最强但最重，
涉及修改用户请求的风险。方向 D 的 embed/store 抽象可直接复用。

## 暂不需要（阈值触发）

### 6. sqlite-vec 向量索引

当前 numpy 暴力余弦：2 万条 <200ms。约定阈值：**chunk 超 5 万再换 sqlite-vec**，
换装点在 `search.py` `_cosine_scores`（接口不变，只改函数内部）。
当前规模（2026-08-14）：334 chunks / 1937 messages，远未到阈值。

## 已解决（2026-08-14）

- ~~MCP 2.x 移植~~：双版本兼容导入（`MCPServer`/`FastMCP`），extra 放宽至 `mcp>=1.0,<3`，
  stdio 冒烟在 1.29.0 与 2.0.0 均通过（commit 913b262）
- ~~reranker 中性分样板占位~~：reranked 路径默认下限 DEFAULT_MIN_SCORE=0.505
  （commit d4fa389）
- ~~空 query 响应缺 messages 键~~（commit d4fa389）
- ~~never-RAG 用户级联删建空 KB 库~~（commit d4fa389）
