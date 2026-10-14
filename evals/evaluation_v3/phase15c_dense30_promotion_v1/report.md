# EvidenceFlow Phase 15C — STOP

## Result

**STOP**。D30 promotion 未通过 exact parity，Production 已回滚为 Dense Top20 → Learned Reranker → Top5。

## Failed hard gate

- Candidate Top30 mismatch: `ARR-032, ARR-033`
- Final Top5 mismatch: `ARR-033`
- IR metrics parity: `PASS`
- Active relevance projection parity: `PASS`

指标相同不能替代逐题 Candidate/Top5 identity；因此没有接受后续 latency、failure attribution
或 reranker miss diagnostic seed 作为 Phase 15C 结果。

只读复现诊断显示：同一模型下 `embed_documents` 与 `embed_query` 的向量逐元素一致；
但 active Chroma HNSW（`ef_search=100`）在不同新进程的 D30 边界查询上出现候选波动，
已观察到 ARR-032/ARR-033 的候选差异。该现象是 exact parity 的实际阻塞因素，
不能通过放宽比较规则消除；在重新申请 Promotion 前必须先建立并冻结确定性的 ANN/replay contract。

## Rollback

当前 Settings candidate K 为 `20`，final K 为 `5`；
active Collection 仍为 `ef_research_bge_m3_hybrid_c512_v2`，index identity 未改变。历史 frozen artifact
Hash 未改变。

## Retained evidence

`baseline_manifest.json`、`promotion_audit.md`、`production_replay.json`、`parity_result.json`、
`latency_metrics.json`、`candidate_metrics.json`、`final_ir_metrics.json`、`failure_attribution.json`
以及 Candidate/Reranker 明细保留用于人工复核。其中 latency 是失败 D30 replay 随流程产生的诊断数据，
不是 parity 通过后执行的独立 latency gate；`cleanup_manifest.json` 标明 STOP 后未执行最终 cleanup。
