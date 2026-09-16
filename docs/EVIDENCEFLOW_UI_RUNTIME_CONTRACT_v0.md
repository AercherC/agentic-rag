# EvidenceFlow UI / Runtime Contract v0

本文件只定义未来 UI 读取的稳定数据边界，不实现 FastAPI、SSE、前端、事件总线或
后台 Worker。Phase 15A 的实现来源是 `KnowledgeBaseManifest`、`BuildJob` 和索引
lifecycle state；UI 不得自行模拟进度或推断 active identity。

## 页面边界

未来第一版只定义三个一级页面：

- `Ask`：提交问题，查看回答、来源和可展开的运行阶段状态。
- `Knowledge Base`：查看 source、active/previous index、manifest 和 BuildJob。
- `Evaluation`：查看既有冻结评测与 failure attribution；不在 UI 端重新计算 Gold/qrels。

## BuildJob 读取合同

`BuildJob` 至少包含：

```text
job_id
knowledge_base_id
source_corpus_signature
target_index_signature
target_collection_name
status
created_at / started_at / finished_at
error_code / error_message
document_count / chunk_count
```

允许的状态为：

```text
PENDING → SCANNING → PARSING → CHUNKING → EMBEDDING → INDEXING
         → VALIDATING → READY → ACTIVATED
```

失败状态 `FAILED` 与前置条件不满足的 `BLOCKED` 必须保持区分。UI 只展示真实已持久化
状态，不添加“预计完成百分比”。

## Knowledge Base 读取合同

Knowledge Base 页面读取：

- 当前 authoritative active pointer；
- 可选的 previous known-good pointer；
- staging pointer（若存在）；
- manifest 中的 source path、document_id、content_hash、pipeline identity、
  corpus/index signature 和 document/chunk count；
- BuildJob 的当前状态与失败原因。

active Collection 的生命周期切换必须由后端 lifecycle 完成。当前 v2 active 与 v1
legacy/read-only 的身份均来自仓库真实状态，UI 不得硬编码或删除 legacy Collection。

## Ask Trace 事件 taxonomy（只定义名称）

未来 RunEvent 可使用以下事件名：

```text
run.started
retrieval.dense.started
retrieval.dense.completed
retrieval.lexical.started
retrieval.lexical.completed
retrieval.pool.completed
rerank.started
rerank.completed
context.started
context.completed
generation.started
generation.completed
citation_validation.completed
run.completed
run.failed
run.blocked
```

这些名称不是 Phase 15A 的 runtime 行为，也不授权修改当前 RAG pipeline。事件 payload
未来应只包含检索、来源和状态审计信息，不包含模型 private chain-of-thought。

## Evaluation 读取合同

Evaluation 页面读取项目已有冻结产物中的：

- Candidate Generator：HitRate@K、EvidenceAnchorRecall@K、AllRequired@K；
- Reranker：Recall@5、MRR@5、MAP@5、nDCG@5；
- Failure Attribution：UNINDEXABLE、CANDIDATE_MISS、RERANKER_MISS、CONTEXT_MISS、
  GENERATION_MISS。

Gold、qrels 和评测指标的权威性属于 EvidenceFlow evaluation artifacts，不由 UI 改写。
