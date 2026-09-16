# Academic Candidate Retrieval Failure Diagnosis v1

- Dataset hash: `be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357`
- Collection: `ef_research_bge_m3_c512_o64_v1`；Full Dense depth 361。
- BM25 仅作诊断：Original Query，k1=1.5，b=0.75；未接入 Runtime。
- 未调用 Generator、ContextBuilder、Citation 或 LLM；未运行参数 sweep。

## 12-question decision matrix

| QID | Type | Lang | Original Dense | Translation Dense | Best Dense | BM25 | Depth | Gold Type | Primary Pattern |
|---|---|---|---:|---:|---:|---:|---|---|---|
| ARR-007 | method | english_to_english | 25 | N/A | 25 | 3 | near_miss | figure_caption | fine_grained_method_confusion |
| ARR-015 | implementation | english_to_english | 97 | N/A | 97 | 55 | moderate_miss | body_text | experiment_setting_confusion |
| ARR-020 | experiment_setup | chinese_to_english | 19 | 16 | 16 | 6 | within_candidate_boundary | body_text | same_paper_section_confusion |
| ARR-021 | experiment_setup | english_to_english | 169 | N/A | 169 | 24 | deep_miss | body_text | same_paper_section_confusion |
| ARR-022 | experiment_setup | chinese_to_english | 11 | 13 | 11 | 51 | within_candidate_boundary | body_text | same_paper_section_confusion |
| ARR-025 | quantitative_results | english_to_english | 31 | N/A | 31 | 27 | near_miss | table | table_representation |
| ARR-030 | quantitative_results | chinese_to_english | 13 | 12 | 12 | 4 | within_candidate_boundary | table | table_representation |
| ARR-031 | ablation | english_to_english | 15 | N/A | 15 | 12 | within_candidate_boundary | supplementary | same_paper_section_confusion |
| ARR-037 | multi_document | mixed_multiple_sources | 78 | N/A | 78 | 45 | moderate_miss | multi_document | multi_document_coverage |
| ARR-038 | multi_document | mixed_multiple_sources | 48 | 45 | 45 | 98 | moderate_miss | multi_document | multi_document_coverage |
| ARR-039 | multi_document | mixed_multiple_sources | 20 | N/A | 20 | 6 | within_candidate_boundary | multi_document | multi_document_coverage |
| ARR-040 | multi_document | mixed_multiple_sources | 43 | 12 | 12 | 6 | within_candidate_boundary | multi_document | multi_document_coverage |

## Aggregate

- Candidate failures: 12；IDs: ARR-007, ARR-015, ARR-020, ARR-021, ARR-022, ARR-025, ARR-030, ARR-031, ARR-037, ARR-038, ARR-039, ARR-040。
- Dense branch boundary 11–20 / bands 21–40 / 41–100 / >100: 6 / 2 / 3 / 1。
- BM25 Top-10 / Top-20 recovered: 5 / 6；Top-20 IDs: ['ARR-007', 'ARR-020', 'ARR-030', 'ARR-031', 'ARR-039', 'ARR-040']。
- Translation improved / runtime recovered: 4 / 0。
- Table-related / Multi-document-related: 2 / 4。
- Dense + BM25 both weak: 0。
- Gold Chunk quality: {'good': 14, 'weak': 2}。
- Gold question quality: {'good': 10, 'weak': 2}。
- Primary patterns: {'fine_grained_method_confusion': 1, 'experiment_setting_confusion': 1, 'same_paper_section_confusion': 4, 'table_representation': 2, 'multi_document_coverage': 4}。
- Pattern presence（primary 或 secondary）: {'fine_grained_method_confusion': 1, 'same_paper_section_confusion': 7, 'cross_paper_method_confusion': 5, 'experiment_setting_confusion': 4, 'dataset_metric_confusion': 2, 'table_representation': 2, 'generic_topic_similarity': 1, 'multi_document_coverage': 4}。
- Multi-document coverage: {'complete_miss': 2, 'partial_coverage': 2, 'complete_coverage': 0}。

## Decision

- Candidate cutoff primary: True。
- Lexical complementarity threshold met: True。
- Cross-language unified pattern: True。
- Embedding limitation evidence sufficient: False。
- Multi-document systematic coverage problem: True。
- Candidate failures highly heterogeneous: True。
- Unique recommendation: **academic_candidate_augmentation**。
- Rationale: BM25 Top-20 recovers six fixed Dense candidate failures (five within Top-10), meeting the frozen lexical-complementarity gate with a more specific mechanism than indiscriminate Dense expansion.
- Benefit vs complexity: Prefer one bounded follow-up only when a frozen threshold is met; no multi-technique bundle is justified.
- Dataset defect / new Chunk defect evidence: False / False。
- Stop Candidate Retrieval optimization: False。
- Runtime 未修改；PROJECT_SPEC 保持 v0.14。

## Interpretation

- 当前正式策略每路只取 Dense Top-10；因此 6 个最佳 Rank 位于 11–20 的题仍会成为融合 Candidate Failure。它们不是附件定义的 21–40 Near band，但构成更直接的分支 cutoff 证据。
- Lexical complementarity 成立：BM25 Top-20 恢复 6/12，覆盖方法图、实验数据、表格、补充材料和多文档问题；不是单一题型偶然收益。
- Cross-language 形成一致的 representation signal：5 个中文失败题中 4 个经冻结译文改善 Rank；但译文完整 Gold Rank 均未进入分支 Top-10，Runtime recovery 为 0。
- Embedding limitation 证据不足：只有 1 个 Dense deep miss，且 Dense 与 BM25 同时弱为 0。
- Multi-document coverage 是系统性问题（2 complete miss、2 partial、0 complete），但只覆盖 4/12；本轮不同时引入 decomposition 或 document diversification。
- Candidate failures 高度异质；Academic Candidate Augmentation 是唯一达到预冻结量化门槛的低复杂度下一实验，不能据此直接把 BM25/Hybrid 接入 Runtime。

## Verification

- 专项与边界测试：21 passed。
- Full pytest：297 passed，15 warnings。
- Warning 均为 Chroma legacy embedding configuration deprecation。
- Runtime fingerprint 与当前正式文件一致；未修改 Runtime，未更新 PROJECT_SPEC。
- 本阶段到此停止，不自动实现 Candidate Augmentation。
