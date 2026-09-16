# Phase 7 Learned Reranker Validation

| Architecture | K | Recall@5 | Recall@10 | MRR@5 | nDCG@5 | AllRequired@5 | Rerank latency/query(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| c0_original_dense | 20 | 0.562 | 0.738 | 0.432 | 0.463 | 0.550 | 4.515 |
| c0_original_dense | 30 | 0.537 | 0.738 | 0.422 | 0.449 | 0.525 | 7.212 |
| c0_original_dense | 50 | 0.525 | 0.713 | 0.412 | 0.440 | 0.525 | 11.706 |
| c0_original_dense | 100 | 0.525 | 0.662 | 0.408 | 0.437 | 0.525 | 22.416 |
| c1_translation_dense | 20 | 0.562 | 0.738 | 0.432 | 0.463 | 0.550 | 4.034 |
| c1_translation_dense | 30 | 0.537 | 0.738 | 0.422 | 0.449 | 0.525 | 6.426 |
| c1_translation_dense | 50 | 0.525 | 0.713 | 0.412 | 0.440 | 0.525 | 10.434 |
| c1_translation_dense | 100 | 0.525 | 0.662 | 0.408 | 0.437 | 0.525 | 20.004 |
| c2_hybrid | 20 | 0.562 | 0.738 | 0.421 | 0.455 | 0.550 | 3.902 |
| c2_hybrid | 30 | 0.575 | 0.738 | 0.431 | 0.466 | 0.575 | 6.237 |
| c2_hybrid | 50 | 0.550 | 0.738 | 0.420 | 0.452 | 0.550 | 10.120 |
| c2_hybrid | 100 | 0.525 | 0.688 | 0.408 | 0.437 | 0.525 | 19.473 |

## Selection

- Architecture: `c0_original_dense`
- Candidate budget: `20`
- Rule: nDCG@5 within 0.01 of best, then simplest architecture, lower budget, lower latency。

## Gate 与实验边界

- Gate：`PASS`。Candidate Replay 两次完全一致；3 个架构、K=20/30/50/100 均完成；Candidate Coverage 未因 rerank 改变；中文、英文、多文档切片、paired bootstrap 与延迟 artifact 均已保存。
- Reranker：`BAAI/bge-reranker-base`，CPU float32，batch size 4，max length 512；本机未使用 GPU。
- 为消除本机 Chroma ANN 重放漂移，本轮使用冻结 Chroma embedding 的 exhaustive cosine 生成候选；未修改正式索引、Gold、Chunk、Embedding 或 Runtime 默认值。

## 决策摘要

- Learned Reranker 确实缓解 Ranking 瓶颈：Original Dense K=20 的 nDCG@5 从 0.3024 提升到 0.4634（paired delta 0.1610，95% CI [0.0402, 0.2911]），MRR@5 从 0.2621 提升到 0.4321；但多文档 complete evidence 仍未解决。
- 选择 K=20：Hybrid K=30 的 nDCG@5 最高（0.4665），但只比 c0 K=20 高 0.0031，未显示稳定显著优势，而 c0 K=20 更简单、延迟更低。
- Translation：reranker 后 c1 与 c0 的浅层指标相同，降为 `REMOVE_CANDIDATE`（保留离线 artifact 与代码，不在本阶段删除）。
- BM25/Hybrid：仅保留为 `EXPERIMENT_ONLY` / `KEEP_CONDITIONAL` 候选分支；Hybrid K=100 的候选覆盖达到 1.0，但浅层排序没有同步收益。
- RRF stage 1/2：降为 `EXPERIMENT_ONLY`，只用于 baseline、去重和来源追踪；最终排序优先直接对 Candidate Union 做 Learned Rerank。

下一阶段只进入 Runtime Promotion Evaluation；若 promotion 不通过，再回到 Embedding、Query 或 Chunk Representation 研究。本阶段已按计划停止。
