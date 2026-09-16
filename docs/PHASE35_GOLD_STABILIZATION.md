# Phase 3.5 Gold Stabilization

## A. 可执行性与结果

判定：EXECUTABLE_WITH_ADJUSTMENT。新增 2.1 Locator 与独立 gold-mapper-v3，保留
Phase 3 的 2.0 数据、mapper 和报告，避免历史重放语义漂移。

Source Gold：44/44 组唯一有效；Multi-document：4/4 题全部来源确定。
当前 C512/O64 Corpus：44/44 FULLY_MAPPED。Gate：PASS。
下一轮 Phase 4A–6：ALLOWED；本轮没有执行这些阶段。

## B. 修改前 15 组证据与来源选择

“旧 Chunk 数”是未消歧 anchor 的匹配数，不代表正确 occurrence；选择仅依据原文、
问题语义和原 Gold 设计。完整 document_id、anchor、全部 occurrence 上下文与旧 Chunk
匹配保存在 unresolved_before.json，生成时间在修改之前。

| question/group | 来源 PDF / 页 | 原状态 | 原 occurrence 数 | 旧 Chunk 数 | 选定 occurrence | 原文理由 |
|---|---|---|---:|---:|---:|---|
| ARR-003/G1 | proxyformer / 1 | NORMALIZATION_FAILURE | 1 | 0 | 1 | 摘要中的采集缺陷与有限视角 |
| ARR-006/G1 | hyper_pcn / 1 | AMBIGUOUS | 2 | 2 | 1 | 摘要阐明高阶关系建模动机 |
| ARR-010/G1 | snowflakenet / 3 | AMBIGUOUS | 2 | 2 | 1 | Figure 3 图注列出 SPD |
| ARR-012/G1 | proxyformer / 1 | AMBIGUOUS | 2 | 1 | 2 | 摘要解释 alignment refine missing proxies，排除标题 |
| ARR-013/G1 | svdformer / 1 | AMBIGUOUS | 2 | 2 | 2 | 摘要说明 dual-generator，排除标题 |
| ARR-024/G1 | hyper_pcn / 5 | AMBIGUOUS | 4 | 1 | 1 | Experiment settings 的完整数据集清单 |
| ARR-025/G1 | adapointr / 9 | AMBIGUOUS | 2 | 2 | 1 | Table 1 ShapeNet-55 表注 |
| ARR-026/G1 | pointr_geometry_aware_transformers / 7 | AMBIGUOUS | 2 | 2 | 1 | Table 3 说明 L1 Chamfer distance |
| ARR-028/G1 | proxyformer / 6 | AMBIGUOUS | 2 | 3 | 1 | Table 1 的类别表头，另一次是不同指标表 |
| ARR-033/G1 | pcn_point_completion_network / 12 | NORMALIZATION_FAILURE | 1 | 0 | 1 | Supplementary overview |
| ARR-037/G2 | adapointr / 4 | AMBIGUOUS | 2 | 1 | 1 | 首次定义 dynamic query embeddings，与 static query 对照 |
| ARR-038/G2 | seedformer / 1 | AMBIGUOUS | 3 | 1 | 2 | 摘要对 Patch Seeds 的定义，排除标题和关键词 |
| ARR-039/G1 | snowflakenet / 1 | AMBIGUOUS | 3 | 3 | 3 | 说明 attention 汇总上一 SPD splitting patterns |
| ARR-040/G1 | svdformer / 1 | AMBIGUOUS | 4 | 2 | 3 | 形状先验和几何自相似性 |
| ARR-040/G2 | hyper_pcn / 1 | AMBIGUOUS | 5 | 2 | 2 | hypergraphs 显式建模高阶关系的摘要句 |

## C. Gold 修改记录

保留全部问题文本、原 anchor、document_id、content_version、页码、group 与 family。
为 44 组统一增加左右上下文（各最多 64 个规范化字符）、原页 occurrence 序号与辅助
source span；13 组使用上表经原文审阅的 occurrence，其余使用已验证的唯一 occurrence。
offset 仅核对漂移，不是唯一标识。

gold_change_log_v3.json 逐组保存 old_locator/new_locator、原因和原文证据。
source_gold_validation_v3.json 保存原始与规范化页面、anchor、具体规则、匹配计数和备注。
corpus_alignment_trace_v3.json 保存实际匹配 Chunk 的原始与规范化文本及对齐方法。

规范化只作用于 Mapper：

- ARR-003 的具体断词是 view-换行-points；Chunk 为 view- points。合并为 viewpoints。
- ARR-033 的具体断词是 addi-换行-tional；Chunk 为 addi- tional。合并为 additional。
- 完整词必须在 Gold 中存在，拆分形式必须由原始来源页的换行证明。
- 合法连接符 state-of-the-art、geometry-aware、cross-modal 保留。
- 来源上的合法换行连接词（如 geometry-换行-aware）只消除换行，不删除连接符。

Source 验证函数不消费 Chunk 或排名。Gold 冻结后，Mapper 才读索引文本，将完整 Chunk
对齐回选定来源位置，或使用唯一的上下文证据。不是查到同名 anchor 就算命中。

## D. 全量 Corpus Mapping

| 指标 | 结果 |
|---|---:|
| Source valid / unique | 44 / 44 |
| FULLY_MAPPED | 44 |
| FRAGMENTED_ACROSS_CHUNKS | 0 |
| PARTIALLY_MAPPED | 0 |
| UNMAPPED | 0 |
| GoldMappabilityRate | 1.0 |
| FragmentedEvidenceRate | 0.0 |
| PartialMappingRate | 0.0 |
| UnmappedRate | 0.0 |

每组状态与合法 Chunk 集合见 gold_mapping_v3.json。这说明冻结 anchor 在当前 Corpus
可定位，不代表题目所需的所有事实已经完成 Claim-level 标注，也不是 Retrieval 提升结论。

新增 evaluate_evidence_coverage（evaluation_version=2.1）：

- eligible_evidence_group_recall_at_K = 命中可映射组 / 全部可映射组。
- overall_evidence_coverage_at_K = 命中组 / 全部 canonical required groups。
- 两者按证据组 micro 聚合，保存每题分子与分母；零分母为 null。
- 未映射组在 Overall 中计零。没有组的不可回答题不稀释分母。
- 原 evaluate_rankings 的 2.0 整题筛选语义保持不变。

本轮没有运行新的检索或重用检索排名来选 Gold。系统级分母通过固定测试验证：两组中
一组不可映射、另一组找回时，Eligible=1.0，Overall=0.5。

## E. 多文档

| 题号 | Source 来源组 | Corpus 来源组 | 可评估 |
|---|---:|---:|---|
| ARR-037 | 2/2 | 2/2 | 是 |
| ARR-038 | 2/2 | 2/2 | 是 |
| ARR-039 | 2/2 | 2/2 | 是 |
| ARR-040 | 2/2 | 2/2 | 是 |

## F. 修改文件与测试

| 文件 | 职责 |
|---|---|
| PROJECT_SPEC.md | 0.16；定位、规范化、双分母与 Gate 合同 |
| src/evidenceflow/domain/evaluation_v2.py | 兼容 2.0，增加 2.1 locator 字段 |
| src/evidenceflow/eval/stable_mapping.py | 独立 Source 验证与选定位置的 Corpus 映射 |
| src/evidenceflow/eval/v2.py | 新增组级 Eligible/Overall 覆盖入口，旧输出不变 |
| experiments/stabilize_gold_v3.py | 冻结来源选择、校验当前 index 版本、排他生成产物 |
| tests/test_gold_stabilization.py | 定位、负例、四类映射、分母和全量重建测试 |
| evals/evaluation_v2/phase35_gold_stabilization_v1/ | 修改前证据、新数据/映射/来源日志与最终 Gate |
| evals/evaluation_v2/cleanup_manifest_v2.json | 仅标记生成缓存，不删除 |

完整 pytest：collected 358，passed 358，failed 0，collection errors 0。
17 条 Chroma legacy embedding function config 弃用警告，与本轮前一致。
全量产物重建在测试中逐项比较 JSON 内容相等，没有写回已存在的文件。

复现（Conda agent）：

```powershell
python -B -m pytest -q -p no:cacheprovider
python -B experiments/stabilize_gold_v3.py --output-dir <新的输出目录>
```

构建阶段 summary 的测试 Gate 保持 PENDING，实际执行完成后的最终判定单独写入
phase35_execution_result.json，不覆盖构建产物。

## G–H. Gate 与停止点

Source 44/44；multi-document 4/4；全部 44 组有 Corpus 状态；测试通过；产物可复现。
受保护的原始 PDF、Chroma、历史产物、Runtime 源码以修改前指纹核对，结果保存在最终
执行产物。无文件删除、无索引写入、无运行时检索策略变更。

Gate：PASS。下一轮 Phase 4A–6：ALLOWED。本轮在 Phase 3.5 停止。
