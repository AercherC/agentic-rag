# Academic Research Query Rewrite Experiment v1

- Dataset hash: `be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357`
- 控制组：中文原始分支 + 冻结人工英文翻译分支（当前基线机制）。
- 处理组：中文原始分支不变，仅把翻译分支英文 Query 换成 LLM 扩展检索 Query。
- LLM 改写状态：成功 6 / 缓存命中 34 / 回退冻结 0 （有 LLM 能力）。

## Overall（控制组 vs 处理组）

| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 40 | 0.175 | 0.350 | 0.475 | 0.650 | 0.650 | 0.302 | 0.404 |
| treatment | 40 | 0.250 | 0.400 | 0.600 | 0.650 | 0.700 | 0.368 | 0.473 |

## Failure distribution

- control: {'ranking_failure': 7, 'success': 19, 'chunk_failure': 2, 'candidate_retrieval_failure': 12}
- treatment: {'success': 24, 'chunk_failure': 2, 'candidate_retrieval_failure': 10, 'ranking_failure': 4}

## Phase A：12 个定向候选失败题的处理组效果

| question_id | control_rank | treatment_rank | effect | rewrite |
|---|---:|---:|---|---|
| ARR-007 | None | None | still_missed | cache_hit |
| ARR-015 | None | None | still_missed | cache_hit |
| ARR-020 | None | None | still_missed | cache_hit |
| ARR-021 | None | None | still_missed | cache_hit |
| ARR-022 | None | None | still_missed | cache_hit |
| ARR-025 | None | None | still_missed | cache_hit |
| ARR-030 | None | 11 | RECOVERED→Top20 | cache_hit |
| ARR-031 | None | 11 | RECOVERED→Top20 | cache_hit |
| ARR-037 | None | None | still_missed | llm_rewrite |
| ARR-038 | None | None | still_missed | llm_rewrite |
| ARR-039 | None | None | still_missed | llm_rewrite |
| ARR-040 | None | None | still_missed | llm_rewrite |

## Gate

- Phase A 定向题数：12
- 救回至 Top-20：2（阈值 4）
- 救回至 Top-5：0
- 控制组 Hit@5：0.475；处理组 Hit@5：0.600
- Hit@5 劣化：-0.125（容差 0.025）
- **判定：NOT PASSED → 停止，不回写 Runtime**

结论：瓶颈在 recall（候选生成失败）。本实验检验查询改写能否把 Gold 拉进候选集合；若 Gate 不通过，说明问题更可能在于语料覆盖 / chunk 切分 / 嵌入表示，而非查询表层改写。
