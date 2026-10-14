# Phase 15C Promotion Audit

## CANONICAL_CANDIDATE_K_SOURCE

`Settings.reranker_input_k` 是正式 candidate depth 的唯一运行时来源；当前值为 `20`。
`runtime.py` 将该值同时传给 `DenseRetriever.top_k`、`LearnedReranker.max_candidates`、
`DenseRerankerRetriever.candidate_pool_k` 和 `reranker_input_k`。

## DUPLICATED_K_LOCATIONS

- `src/evidenceflow/config.py`: `Settings.reranker_input_k` 默认值 `20`，本阶段唯一需要晋升的 canonical default。
- `src/evidenceflow/retrieval/reranker.py`: 通用 `DenseRerankerRetriever` 构造参数默认值 `20`；正式 Runtime 总是显式传入 Settings，不修改该通用 fallback，避免扩大本阶段行为范围。
- `experiments/` 与历史 `evals/` 中的 `20` 是冻结实验口径，不属于当前 Production source，不修改。

## FILES_TO_MODIFY

- `src/evidenceflow/config.py`
- `src/evidenceflow/runtime.py`
- `tests/test_config.py`
- `tests/test_runtime.py`
- `PROJECT_SPEC.md`
- `README.md`
- `experiments/run_phase15c_dense30_promotion.py`

## FILES_TO_TEST

- `tests/test_config.py`
- `tests/test_runtime.py`
- `tests/test_retriever.py`
- `tests/test_reranker.py`
- `tests/test_architecture_boundaries.py`
- `python -m pytest -q`

## FILES_NOT_TO_TOUCH

- `src/evidenceflow/retrieval/reranker.py` 的排序、tie-break 和评分实现
- Embedding、Chroma、Chunk、ContextBuilder、Generator、Citation、RAGService
- active Chroma、source PDFs、Gold、qrels、mapping、projection
- Phase 13F、15A、15B、15B.1、15B.5 冻结 artifacts
- `experiments/archive/` 与历史 Phase 15B.1 记录

## REQUIRED_CHANGE

只将正式 Runtime candidate depth 从 `20` 改为 `30`；Reranker 输入同步为 30，最终输出继续为 5。
