# Phase 8 Runtime Promotion Readiness Check

- Runtime Promotion: **ALLOWED**
- ANN: **PASS**
- Truncation: **PASS**
- Performance: **NEEDS_OPTIMIZATION** (acceptable for local research)
- InputTruncationRate: 0.995
- GoldPairTruncationRate: 1.000
- Gold retention: `{"FULLY_RETAINED": 45, "PARTIALLY_RETAINED": 0, "DROPPED": 0, "UNKNOWN": 0}`
- CPU rerank p50 / p95: 4.747s / 4.961s
- Exact / ANN nDCG@5: 0.463 / 0.463
- Recommended candidate: `BGE-M3 Dense -> Chroma ANN Top20 -> Learned Reranker -> Top5`

本阶段未修改 Runtime 默认或正式 Chroma 数据。
