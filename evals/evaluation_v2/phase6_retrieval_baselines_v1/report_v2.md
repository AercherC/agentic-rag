# Phase 6 Retrieval Baselines v1 — Decision Revision v2

本文件保留 `report.md` 的全部数值结果，只更正其 Architecture Ablation：原判定只读取
K=50，遗漏了计划要求的语言切片与 K=100 深层 Candidate Coverage。原始排名、指标与
paired comparison 未修改。

| Variant | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Recall@100 | MRR@5 | nDCG@5 | AllRequired@20 | Retrieval Latency(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Original Dense | 0.386 | 0.614 | 0.795 | 0.909 | 0.955 | 0.262 | 0.302 | 0.800 | 0.086 |
| B1 Frozen Translation Dense | 0.432 | 0.614 | 0.795 | 0.909 | 0.955 | 0.275 | 0.320 | 0.800 | 0.131 |
| B2 Runtime Translation Dense | 0.409 | 0.614 | 0.818 | 0.909 | 0.955 | 0.281 | 0.319 | 0.800 | 0.119 |
| B3 Hybrid | 0.409 | 0.659 | 0.818 | 0.886 | 1.000 | 0.210 | 0.270 | 0.825 | 0.143 |

## Recall@K Curve

- B0: @1=0.159, @3=0.318, @5=0.386, @10=0.614, @20=0.795, @30=0.841, @50=0.909, @100=0.955。
- B1: @1=0.159, @3=0.318, @5=0.432, @10=0.614, @20=0.795, @30=0.864, @50=0.909, @100=0.955。
- B2: @1=0.182, @3=0.318, @5=0.409, @10=0.614, @20=0.818, @30=0.864, @50=0.909, @100=0.955。
- B3: @1=0.068, @3=0.341, @5=0.409, @10=0.659, @20=0.818, @30=0.841, @50=0.886, @100=1.000。

## Slice

- Frozen Translation 只在中文浅层切片显示清晰收益：Recall@5 由 0.450 升至 0.525；英文保持 0.400。
- Runtime Translation 的 multi-document EvidenceGroupRecall@20 为 0.625，对照 B0/B1 为 0.500。
- Hybrid 的 multi-document AllRequired@20 为 0.500，对照 B0/B1/B2 为 0.250。

## Translation

40 次 Runtime Translation 全部成功，总翻译延迟 686.486 秒；保存译文后的 Retrieval 重放
候选 ID 与顺序完全一致。当前 Client 不暴露 token usage，因此该字段为 null。

## Architecture Ablation

- Original Dense：**KEEP**。
- Translation branch：**KEEP_CONDITIONAL**，仅保留中文/已证明有效的语言切片。
- BM25：**NEEDS_RERANKER_VALIDATION**。
- Hybrid Fusion：**NEEDS_RERANKER_VALIDATION**。
- RRF stage 1：**KEEP_CONDITIONAL**。
- RRF stage 2：**NEEDS_RERANKER_VALIDATION**。

## Bottleneck Diagnosis

深层 Candidate Coverage 已达到 95.5%–100%，而浅层 Recall@5 只有 38.6%–43.2%；Hybrid
还能补齐最后的深层证据，但同时降低 MRR@5 与 nDCG@5。因此当前最可能瓶颈是
**Ranking**，不是优先修改 Chunk。下一阶段建议以冻结 K=100 候选验证 Learned Reranker；
该预算只用于实验输入，不改变 Runtime Context Top-K。
