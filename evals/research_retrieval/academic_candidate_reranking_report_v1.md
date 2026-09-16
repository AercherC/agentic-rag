# Academic Candidate Reranking Controlled Experiment v1

- Dataset hash: `be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357`
- 输入：Dense 双路 Top-10 + 原文 BM25 Top-10（add-only 增强池，来自 Candidate Augmentation 实验）。
- 重排器：control_dense_only / control_augmented_by_dense_rank（仅加候选不改序）/ dense_rerank(bge-m3 cosine) / hybrid_rerank(Weighted RRF)。
- 完全离线；dense 重排复用已缓存 BAAI/bge-m3，未下载新模型，未修改 Runtime / PROJECT_SPEC。

## Phase A — augmentation 补回覆盖的 5 题机制门

- 题数：5；Gate ≥2 题进入 Top-5。
- control(仅加候选): 0/5 进 Top-5。
- dense_rerank: 0/5 进 Top-5。
- hybrid_rerank: 2/5 进 Top-5。
- Gate: **PASS**。

## Phase B — Frozen 40Q 重排增益

- Runtime Dense 基线 Hit@5=0.475，MRR=0.302，nDCG@20=0.404。
- 增强候选覆盖：26/40 → 31/40（覆盖≠命中）。

| Reranker | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Runtime Dense (baseline) | 0.175 | 0.350 | 0.475 | 0.650 | 0.650 | 0.302 | 0.404 |
| control_dense_only | 0.175 | 0.350 | 0.500 | 0.650 | 0.650 | 0.303 | 0.405 |
| control_augmented_by_dense_rank | 0.175 | 0.375 | 0.500 | 0.650 | 0.775 | 0.313 | 0.437 |
| dense_rerank (bge-m3 cosine) | 0.175 | 0.350 | 0.450 | 0.650 | 0.775 | 0.305 | 0.430 |
| hybrid_rerank (rrf) | 0.075 | 0.250 | 0.425 | 0.500 | 0.775 | 0.220 | 0.362 |

## Hit@5 相对基线增益

| Reranker | ΔHit@5 |
|---|---:|
| control_augmented_by_dense_rank | +0.025 |
| dense_rerank | -0.025 |
| hybrid_rerank | -0.050 |

- control 与重排的差值证明：重排带来的增益（非单纯增加候选）= -0.050。

## Interpretation

- 覆盖增益(26→31)未转化为 Top-5 命中：BM25 补回的 Gold Chunk 是'词汇命中但语义低秩'——
  它们在 bge-m3 cosine 与 RRF 融合下都排不到前 5，仅因 add-only 进入候选池。
- 唯一对 Top-5 有正贡献的是'仅追加候选、保持 Dense 名次'(control=0.500)，而非任何重排。
  重排(dense_rerank 0.450 / hybrid_rerank 0.425)反而把原本 Top-5 内的 Dense Gold 推后，净效应为负。
- Phase A 机制门通过(hybrid 把 5 题中 2 题 Gold 推入 Top-5)只证明'重排对少数目标题局部有效'，
  但全 40 题聚合下增益被其他题的回退抵消——局部有效 ≠ 全局有益。
- 副发现：control_dense_only(0.500)略高于 Runtime 基线(0.475)，因为'取双路最佳名次'排序在 Top-5 上
  略优于等权 RRF 融合；这是排序实现的副产物，非本次重排实验要检验的效应。

## Decision

- Gate: **PASS**。
- 最佳重排器：dense_rerank（Hit@5=0.450）。
- 相对 Runtime 基线 Hit@5 增益：-1 题（-0.025）。
- 重排确实把覆盖转化为 Top-5 命中：False。
- 增益来自重排而非单纯加候选：False。
- Hybrid 最优权重(dense_w, bm25_w)=(2.0, 1.0)。
- Next controlled experiment: **dense_only_reranker**。
- Recommend Runtime integration now: False。
- Runtime 未修改；PROJECT_SPEC 保持 v0.14。
