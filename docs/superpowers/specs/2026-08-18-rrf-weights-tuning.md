# RRF 权重调优验证记录

**日期**：2026-08-18
**背景**：backlog 项 2（2026-08-14 rag-next-steps）——三通道（向量/trigram/jieba）RRF 融合
默认平等权重，spec 预留参数化，需用真实查询集评估是否调权。

## 方法

- 参数化落地：`_rrf_fuse` 加 `weights` 参数（按通道标签对齐，FTS 通道缺省时不错位）；
  `KbConfig.rrf_weights`（`[prompt_kb] rrf_weights = "v,t,j"` / `CLAUDE_TAP_KB_RRF_WEIGHTS`），
  CLI / dashboard / MCP 三端透传
- 对比：真实库副本（3781 messages / 334 chunks）× 2026-08-13 验证记录的 7 组验收查询
  × 5 组权重：(1,1,1) 基线、(1,2,2) 关键词升、(2,1,1) 向量升、(1,2,1) trigram 升、(1,1,2) jieba 升；
  全管线含 bge-reranker 校准重排

## 结果

| 查询 | 权重敏感？ | 观察 |
|---|---|---|
| Q1 哪个 CLI 有沙箱 shell 工具 | 否 | 5 组权重结果完全一致（Bash 居首） |
| Q2 which CLI has a sandboxed shell tool | 否 | 完全一致（Bash top 组首命中） |
| Q3 怎么写 commit message | 是 | 关键词升把代码 dump 碎片挤进 top-3（更差）；向量升保住相关命中（略好） |
| Q4 前端页面截图验证 | 否 | 完全一致 |
| Q5 取消定时任务 cron (kind=tool) | 微 | jieba 升引入 CronCreate 0.536 噪声；其余一致，CronDelete 稳居首 |
| Q6 沙箱 sandbox 执行命令 (min_score=0.86) | 否 | 全部为空（校准分数分布下无命中），一致 |
| Q7 Playwright 浏览器截图验证 | 是 | 向量升/trigram 升引入代码碎片噪声；关键词升与基线一致 |

## 结论

**默认权重维持 (1,1,1)，不调。**

- 没有权重组合一致优于基线；调权在 Q3/Q5/Q7 引入的噪声多于收益
- 权重只影响候选池构成（recall=20），最终排序由 reranker 校准分数决定；
  当前语料规模（千级）下候选池对权重不敏感
- 参数化能力保留：日后语料规模或查询分布变化时，改配置即可重新评估，
  重跑本对比即可（harness 为一次性脚本，未入库）

## 附：embedder 身份解耦（backlog 项 1 瘦身版，25b54b5）

`canonical_model_id` 把 modelscope/HF 缓存路径归一为 `<org>/<name>`，换路径不再触发
全量 reindex。存量库首次打开会因 embedder_name 归一触发**最后一次** reindex。
modelscope SDK 一等支持（自动下载）经与用户讨论后砍掉——本机模型已就位，
收益只是下次重装时少几步手工操作，不值得引入 SDK 依赖。
