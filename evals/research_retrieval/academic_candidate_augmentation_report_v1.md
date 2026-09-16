# Academic Candidate Augmentation Controlled Experiment v1

- Dataset hash: `be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357`
- Variant: Original Dense Top-10 + Translation Dense Top-10 + Original-query BM25 Top-10。
- Add-only；dedup=`chunk_id`；ranking=none；不报告 Variant Hit@5。
- 未调用 Generator/Context/Citation/LLM，未使用 Reranker，未运行参数 sweep。

## Phase A — fixed 12 failures

- Recovered: 5/12。
- IDs: ['ARR-007', 'ARR-020', 'ARR-030', 'ARR-039', 'ARR-040']。
- Gate ≥4/12: PASS。

## Phase B — Frozen 40Q Candidate Coverage

| Metric | Baseline | Variant |
|---|---:|---:|
| Complete Candidate Coverage | 26/40 (0.650) | 31/40 (0.775) |
| Regressed | 0 | 0 |

- Net recovery: 5；IDs: ['ARR-007', 'ARR-020', 'ARR-030', 'ARR-039', 'ARR-040']。
- Preserved / regressed / still missed: 26 / 0 / 9。
- BM25-only Gold recovery: 5；Dense+BM25 shared Gold questions: 16。

## Candidate cost

| Pool | Min | Median | Mean | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Baseline | 10 | 10.0 | 10.82 | 14 | 15 |
| Augmented | 11 | 15.0 | 16.00 | 20 | 22 |
| Unique BM25 additions | 1 | 5.0 | 5.17 | 9 | 10 |

- BM25 duplicate rate: 193/400 (0.482)。

## Recovery breakdown

| Slice | N | Baseline | Variant | Recovered |
|---|---:|---:|---:|---:|
| Method | 8 | 7 | 8 | 1 |
| Experiment setting | 5 | 2 | 3 | 1 |
| Quantitative results / dataset-metric related | 6 | 4 | 5 | 1 |
| Table evidence | 6 | 4 | 5 | 1 |
| Ablation | 4 | 2 | 2 | 0 |
| Multi-document | 4 | 0 | 2 | 2 |

| Language group | N | Baseline | Variant | Recovered |
|---|---:|---:|---:|---:|
| English → English | 18 | 11 | 12 | 1 |
| Chinese → English paper | 18 | 15 | 17 | 2 |
| Mixed / multi-document | 4 | 0 | 2 | 2 |

## Multi-document

- Complete coverage: 0/4 → 2/4。
- Partial coverage: 2/4 → 1/4。
- Miss: 2/4 → 1/4。

## Remaining misses

- Dense + BM25 still miss complete Gold sets: 9/40。
- IDs: []。

## Decision

- Generalization gate: **PASS**。
- Benefit vs complexity: Recovered 5/40 complete Gold sets with 5.17 average unique candidates added.
- Next controlled experiment: **augmented_candidate_reranking**。
- Recommend Runtime integration now: False。
- Runtime 未修改；PROJECT_SPEC 保持 v0.14。
