# References-Aware Metadata / Filtering Controlled Experiment v1

- Status: `formal_evaluation_complete`
- Dataset hash: `be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357`
- Reference-targeted questions: 0/40.
- Chunks: 361（body 228；references 86；supplementary 47）。
- Detection: References 10/10；Supplementary 10/10；FP/FN 0/0。
- Chunk ID、文本和 embedding 不变；只增加实验 metadata。

## Mechanism check

- Questions: 22.
- Reference slots removed: 32.
- Gold rank improved: 8.
- Gold promoted into Top-5: 1.
- No change: 13.
- Unexpected rank regression: 1.

## Formal frozen 40Q evaluation

| Metric | Baseline | Variant |
|---|---:|---:|
| Hit@1 | 0.175 | 0.175 |
| Hit@3 | 0.350 | 0.350 |
| Hit@5 | 0.475 | 0.475 |
| Hit@10 | 0.650 | 0.600 |
| Hit@20 | 0.650 | 0.625 |
| MRR | 0.302 | 0.297 |
| nDCG@20 | 0.404 | 0.399 |

## Outcomes

- Recovered / preserved / regressed: 1 / 38 / 1.
- Candidate failure recovered: 1；ranking failure recovered: 1.
- Reference Top-5 questions: 22 → 0.
- Reference Top-5 slots: 32 → 0.
- Candidate / ranking / chunk failures: 12/7/2 → 13/6/2.
- Multi-document complete hit: 0/4 → 0/4.
- Subgroup regressions: 1.
- Recovered Top-5 case: `ARR-024`（rank 6 → 5）。
- Regressed Top-5 case: `ARR-036`（rank 1 → miss）；其 Conclusion Gold 与 References 位于同一边界页，暴露了单一 `section_type` 对混合 Chunk 的近似性。
- Subgroup regression: `conclusion_limitation` Hit@5 由 0.500 降至 0.000（N=2）。
- Multi-document partial coverage：`ARR-039` 从 0/2 来源提升到 1/2，但完整覆盖仍为 0/4。

## Decision

- Gate: **PARTIAL**.
- References pollution proven causal for at least one failure: True.
- Recommend Runtime integration: False.
- Runtime 未修改；PROJECT_SPEC 保持 v0.14；未调用 Generator/LLM。

## Verification

- 专项与边界测试：14 passed。
- Full pytest：281 passed，15 warnings。
- 15 条 warning 均为 Chroma legacy embedding configuration deprecation。
- Runtime fingerprint 与当前正式 Loader/Cleaner/Chunker/Retriever/Runtime 文件一致。
- 本实验到此停止；不修改 Runtime，不升级 PROJECT_SPEC，不自动开始下一实验。
