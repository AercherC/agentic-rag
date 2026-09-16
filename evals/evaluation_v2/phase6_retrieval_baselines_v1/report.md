# Phase 6 Retrieval Baselines v1

| Variant | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR@5 | nDCG@5 | AllRequired@20 | Latency(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b0_original_dense | 0.386 | 0.614 | 0.795 | 0.909 | 0.262 | 0.302 | 0.800 | 0.086 |
| b1_frozen_translation_dense | 0.432 | 0.614 | 0.795 | 0.909 | 0.275 | 0.320 | 0.800 | 0.131 |
| b2_runtime_translation_dense | 0.409 | 0.614 | 0.818 | 0.909 | 0.281 | 0.319 | 0.800 | 0.119 |
| b3_hybrid | 0.409 | 0.659 | 0.818 | 0.886 | 0.210 | 0.270 | 0.825 | 0.143 |

## Recall@K Curve

- b0_original_dense: @1=0.159, @3=0.318, @5=0.386, @10=0.614, @20=0.795, @30=0.841, @50=0.909, @100=0.955
- b1_frozen_translation_dense: @1=0.159, @3=0.318, @5=0.432, @10=0.614, @20=0.795, @30=0.864, @50=0.909, @100=0.955
- b2_runtime_translation_dense: @1=0.182, @3=0.318, @5=0.409, @10=0.614, @20=0.818, @30=0.864, @50=0.909, @100=0.955
- b3_hybrid: @1=0.068, @3=0.341, @5=0.409, @10=0.659, @20=0.818, @30=0.841, @50=0.886, @100=1.000

## Architecture Ablation

- Original Dense: **KEEP** — B0 micro Recall@50=0.909091; simplest no-call baseline.
- Translation branch: **REMOVE_CANDIDATE** — Frozen translation micro Recall@50 delta=+0.000000; Runtime delta=+0.000000.
- BM25: **REMOVE_CANDIDATE** — Hybrid vs frozen Dense micro Recall@50=-0.022727; nDCG@5=-0.049828.
- Hybrid Fusion: **EXPERIMENT_ONLY** — 由 BM25 的深层覆盖增益与浅层排序变化共同决定。
- RRF stage 1: **EXPERIMENT_ONLY** — Original/Translation 融合的独立收益见 B1-B0。
- RRF stage 2: **EXPERIMENT_ONLY** — 仅 Hybrid 使用，不能脱离 B3-B1 解释。
