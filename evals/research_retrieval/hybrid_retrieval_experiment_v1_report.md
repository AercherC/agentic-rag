# Academic Research Hybrid Retrieval (Dense + BM25) Experiment v1

- Dataset hash: `be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357`
- 控制组：多语 Dense（中文原始分支 + 冻结人工英文翻译分支，内部 RRF）。
- 处理组：多语 Dense 结果再融合 BM25 lexical 分支（Equal-Rank RRF）。
- BM25 配置：k1=1.5, b=0.75；融合 rrf_k=60。

## Overall（控制组 vs 处理组）

| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 40 | 0.175 | 0.350 | 0.475 | 0.650 | 0.650 | 0.302 | 0.404 |
| treatment | 40 | 0.075 | 0.350 | 0.475 | 0.725 | 0.800 | 0.255 | 0.393 |

## Failure distribution

- control: {'ranking_failure': 7, 'success': 19, 'chunk_failure': 2, 'candidate_retrieval_failure': 12}
- treatment: {'success': 19, 'chunk_failure': 2, 'ranking_failure': 13, 'candidate_retrieval_failure': 6}

## Phase A：定向候选失败题的处理组效果

| question_id | control_rank | treatment_rank | effect |
|---|---:|---:|---|
| ARR-007 | None | 7 | RECOVERED→Top20 |
| ARR-015 | None | None | still_missed |
| ARR-020 | None | 11 | RECOVERED→Top20 |
| ARR-021 | None | None | still_missed |
| ARR-022 | None | None | still_missed |
| ARR-025 | None | None | still_missed |
| ARR-030 | None | 10 | RECOVERED→Top20 |
| ARR-031 | None | 14 | RECOVERED→Top20 |
| ARR-037 | None | None | still_missed |
| ARR-038 | None | None | still_missed |
| ARR-039 | None | 10 | RECOVERED→Top20 |
| ARR-040 | None | 9 | RECOVERED→Top20 |

## Gate

- Phase A 定向题数：12
- 救回至 Top-20：6（阈值 4）
- 救回至 Top-5：0
- 控制组 Hit@5：0.475；处理组 Hit@5：0.475
- Hit@5 劣化：0.0（容差 0.025）
- **判定：PASSED → 建议进入 Runtime 集成候选（Dense+BM25 检索时融合）**

结论：若 Gate 通过，说明在检索时融合 BM25 lexical 分支能稳定把部分 Gold 拉进候选集合，是比查询改写更可靠的 recall 杠杆；否则瓶颈仍在语料覆盖 / chunk 切分 / 嵌入表示。