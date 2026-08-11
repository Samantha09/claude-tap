# Trace 语义检索（RAG 方向 A）设计文档

日期:2026-08-10
状态:已实现(2026-08-11)

## 背景

prompt 知识库(方向 D,已实现)目前只索引 system/developer prompt 与工具定义,
不覆盖对话正文。用户回溯历史工作时最常见的诉求是:"上次我是怎么让 Agent 解决
某个问题的"——答案散落在各会话的用户消息里,现有 dashboard 只能按日期/客户端
翻列表,无法按语义搜索。

本设计实现方向 A:**把用户消息索引进知识库**,在 KB 页新增"会话"分区,命中结果
按相关度排序并可跳转会话详情。

基础设施全部就位:`extract.py` 的按会话增量抽取、`index.py` 的懒索引循环、
`search.py` 的余弦检索、`prompt_snapshot.py` 的 provider 归一化层均可复用。

### 已确认的决策(与用户逐项确认)

| 决策点 | 结论 |
|--------|------|
| 索引内容 | 只索引用户消息(含附件文本);不索引助手回复、工具调用、工具结果 |
| 结果呈现 | KB 页内新增"会话"分区,命中卡片可跳转 `/dashboard/session/{id}` |
| 历史回填 | 全量回填所有历史 trace 会话(本地 e5 免费,后台线程渐进完成) |
| 存储架构 | 新增 `kb_messages` 表,与 `kb_chunks` 平行的双路管线(方案 1) |
| 检索实现 | 维持 numpy 暴力余弦;两路查询写成可替换的私有函数,超 5 万 chunk 再换 sqlite-vec |

方案取舍:复用 `kb_chunks` + synthetic snapshot(方案 2)会污染 timeline 语义,
独立第二 KB 库(方案 3)复杂度超出需求,均已否决。

## 目标与非目标

### 目标(预期效果)

1. **会话语义搜索**:KB 页输入自然语言(如"race condition 怎么修的"),
   "会话"分区返回命中的用户消息,按相关度排序,显示来源(client/model/时间)
2. **跳转定位**:命中卡片点击跳转对应会话详情页
3. **跨会话去重**:同一句消息(如"继续")发 N 次只存储一行、只 embedding 一次
4. **全量回填**:reindex 后所有历史会话的用户消息可被搜到
5. **优雅降级**:未装 `[rag]`、embedding 不可用时,trace 录制与 dashboard 其他
   功能完全不受影响(与方向 D 行为一致)

### 非目标(YAGNI)

- 不索引助手回复、工具调用、工具结果(后续方向,不在本期)
- 不做"该消息还出现于另外 N 个会话"(需 `kb_message_occurrences` 关联表,后续可加)
- 不做混合检索(FTS5 关键词召回)、不做重排
- 不做 LLM 生成式问答
- 不引入向量索引库(sqlite-vec 等),保留换装点即可

### 验收标准

1. 全量回填后,自然语言查询能在 KB 页"会话"分区命中历史用户消息并按相关度排序
2. 命中卡片可跳转对应会话详情,且该会话存在(删除会话后不再出现其命中)
3. 相同文本的消息跨会话重复出现时,`kb_messages` 只有一行且只 embedding 一次
   (stats 可见)
4. 未装 `[rag]` 依赖时行为与方向 D 完全一致:知识库页显示安装提示,其余功能不受影响
5. 2 万条消息 chunk 规模下搜索响应 < 200ms(合成数据测试验证)

## 架构

### 数据模型(`prompt_kb/store.py`)

新增 `kb_messages` 表,索引状态机与 `kb_chunks` 完全一致
(pending/indexed/failed + attempts ≤ 3 重试):

```sql
CREATE TABLE IF NOT EXISTS kb_messages (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,        -- trace sessions.id,跳转详情用
  record_index INTEGER NOT NULL,   -- 第几条 trace 记录
  message_index INTEGER NOT NULL,  -- 该请求内第几条 user 消息
  client TEXT NOT NULL,            -- 冗余,搜索过滤用
  model TEXT NOT NULL,
  timestamp TEXT NOT NULL,         -- 消息产生时间,展示与排序
  content_hash TEXT NOT NULL,      -- 规范化文本的 sha256
  text TEXT NOT NULL,
  last_seen TEXT NOT NULL,         -- 去重命中时更新
  embedding BLOB,
  index_state TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_kb_messages_state ON kb_messages(index_state);
CREATE INDEX idx_kb_messages_session ON kb_messages(session_id);
CREATE UNIQUE INDEX idx_kb_messages_dedup ON kb_messages(content_hash, client);
```

关键决策:

- **跨会话去重**:`(content_hash, client)` 唯一。重复消息只更新 `last_seen`,
  `session_id` 保留首次出现的会话(命中跳转目标即首现会话)
- **删除会话联动**:`DELETE /api/sessions/{id}` 时级联删除该会话的 `kb_messages`
  行。已知让步:去重共享行随首现会话删除而整体消失(该文本在历史中不再可搜),
  首版接受,文档化
- `KbStore._migrate()` 幂等建表,老库无感升级;`stats()` 增加 `messages` 计数

### 抽取与分块(`prompt_kb/messages.py`,新增)

职责单一:records → 用户消息列表。复用 `prompt_snapshot.py` 的 provider 判定
(`infer_provider`),按 anthropic / openai chat / openai responses / gemini
四种格式解析每条 record 请求体中的 `role=user` 消息:

- **保留**:text block 拼接后的文本、附件文本
- **跳过**:空文本/纯空白;图片与文件二进制;工具结果回传(anthropic 的
  `tool_result` block、openai 的 `role=tool`);harness 注入的伪 user 消息
  (`<system-reminder>`、command-message 等,判定规则参照 viewer 前端同类逻辑在
  Python 侧重写)
- **分块**:单条消息 ≤ 2000 字符(复用 `MAX_SECTION_CHARS`)为一个 chunk;超长
  按 `_split_long` 同规则切段。不合并短消息——用户消息是完整语义单元,合并会模糊
  跳转目标
- **接入点**:`extract_session()` 在 snapshot 抽取后追加 `extract_messages()`;
  `kb_sources` 仍按会话粒度记录,一次处理同时产出 snapshot 与 messages

### 索引(`prompt_kb/index.py`)

`index_pending` 扩展为处理两张表:先 `kb_chunks` 后 `kb_messages`,同一 embedder、
同一批次大小、同一重试语义。`KbStore` 镜像新增
`pending_messages / mark_message_indexed / mark_message_failed /
requeue_failed_messages / indexed_messages / reset_message_embeddings`。
`rebuild_index` 同时重置两表。

### 检索(`prompt_kb/search.py`)

- `search()` 签名与返回不变;新增 `search_messages()` 返回会话消息分区结果,
  API/CLI 层组合两路输出(向后兼容)。`search_messages()` 对
  `indexed_messages()` 做暴力余弦,按 session 分组(每组最多 3 条 hit),支持
  client 过滤与 `min_score`,与 prompt 路相同的 `_check_embedder_meta` 前置校验
- 两路查询都是独立私有函数,未来换 sqlite-vec 只改函数内部实现,接口不动

### API(`live.py`)

- `GET /api/kb/search` 响应在现有 `results` 键之外新增 `messages` 键;
  现有消费者(dashboard.html、kb CLI、测试)不受影响
- `GET /api/kb/status` 增加 `messages` 计数
- `POST /api/kb/reindex` 同时重建两表(复用现有 202 后台线程模式)
- `DELETE /api/sessions/{id}` 级联删除 `kb_messages` 对应行
- 搜索响应中的 messages hit:`{session_id, client, model, timestamp, text,
  score}`,由 API 层附带

### CLI(`prompt_kb/cli.py`)

`claude-tap kb search` 输出增加"会话消息"小节(在 prompt 结果之后);
`kb status` 显示 messages 统计。

### UI(`dashboard.html` KB 页)

- 搜索结果区在现有 prompt 卡片下方新增"会话"分区:每条命中一张卡片,
  显示 client/model 徽标、消息时间、消息摘录、相关度;点击跳转
  `/dashboard/session/{id}`
- 空状态、加载态、错误态(reindex_required / embedder 不可用)复用现有 KB 页
  刚重设计的三态模式与卡片样式
- i18n 词条中英文双份(其余语言走 translate-i18n 流程补齐)

## 错误处理与降级

- 未装 `[rag]`:知识库页安装提示,其余功能不受影响(与方向 D 一致)
- embedder 不可用:messages 与 chunks 一样停留 pending,索引循环每 10 轮重试
  embedder 创建
- 单会话抽取失败:不 `record_source`,下一轮重试(现有语义)
- embedding 批量失败:整批 messages 标 failed,attempts 封顶 3 次后可
  requeue(现有语义)
- 换 embedding 模型:kb_meta 维度校验触发 `ReindexRequired`,reindex 同时重建两表
- 隐私:用户消息原文存本地 KB 库,与 trace 库同级(本机工具,不脱敏对话内容;
  认证头脱敏在代理层已完成)

## 测试

沿用 `tests/prompt_kb/` 现有布局与 fake_embedder 基建,TDD:

- `test_messages.py`:四种 provider 格式的用户消息抽取;tool_result/role=tool/
  空消息/harness 注入消息过滤;超长消息切段;附件文本保留
- `test_store_messages.py`:建表迁移幂等;`(content_hash, client)` 去重 upsert
  与 last_seen 更新;状态机方法;stats 计数
- `test_index_messages.py`:批量 embedding、失败重试封顶、rebuild 双表重置
- `test_search_messages.py`:排序、按 session 分组、client 过滤、min_score、
  ReindexRequired
- kb API 测试:search 响应双分区、status 计数、reindex 双表、删除会话级联
- KB 页 Playwright 渲染测试:会话分区卡片渲染、空状态、跳转链接(真实 trace
  fixture,遵循 js-in-html-testing 两层策略)
- CLI 测试:search 输出含会话小节、status 含 messages 计数
- 性能测试:合成 2 万条消息验证搜索 < 200ms

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 全量回填首次 embedding 耗时长(数千会话) | 后台线程渐进索引,不阻塞;stats 可见进度 |
| 用户消息含敏感信息 | 与 trace 库同级的本地存储,文档说明;不新增外发路径 |
| 去重导致跳转只到首现会话 | 首版接受,命中卡片显示 last_seen;后续可加 occurrences 表 |
| 暴力检索随数据量变慢 | 5 万 chunk 阈值文档化;search 私有函数预留 sqlite-vec 换装点 |
