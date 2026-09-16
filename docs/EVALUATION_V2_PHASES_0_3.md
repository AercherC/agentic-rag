# Evaluation v2：Phase 0–3 实施报告

日期：2026-09-05。首轮到 Phase 3 停止；没有进入项目清理、候选合同修复或算法实验。

## Phase 0：可行性复核

当前 375 个文件；源码位于 src/evidenceflow，35 个原有测试文件，17 个原有实验脚本。
完整 pytest 收集复现 2 个错误：inspect 测试仍引用 experiments 根目录的旧路径。
debug 下 5 个脚本仍用 parents[1]，gen_flowchart 使用依赖工作目录的输出路径。

当前实际配置：10 篇科研 PDF，C512/O64，BAAI/bge-m3，原文/翻译每路 Top-10，
最终 Top-5，Equal RRF，Hybrid=False，Context BGE 预算 4096；生成与翻译使用
Qwen/Qwen3.5-35B-A3B。Collection 为 ef_research_bge_m3_c512_o64_v1，361 个 Chunk。
翻译运行时使用 LLMTranslationProvider，研究基线使用冻结人工译文。

两条评估链仍然存在：

1. JSON 数组 → eval/dataset → EvaluationRunner → legacy metrics → report。
2. research_questions_v1 → baseline._load_and_verify/_gold_groups → 各实验检索 →
   gold_group_rank + _source_ndcg/_metrics → 各实验 writer。

研究数据为对象封装的 40 题、44 个来源组，已反复用于实验；本轮适配为
legacy_dev_diagnostic。Gold 仍采用文档 ID、页码、原文锚点。静态复核确认多语
融合仍提前截到 candidate_top_k；本轮按阶段边界保留，Phase 5 再修复。
只读索引重映射再次复现 ARR-003、ARR-033 无完整匹配。

| 阶段 | 判定 | 前置条件或调整 |
|---|---|---|
| 1 | EXECUTABLE | 修复旧路径，不删除测试 |
| 2 | EXECUTABLE_WITH_ADJUSTMENT | 新模型与 legacy 适配并存；不静默改义 |
| 3 | EXECUTABLE | 原始 PDF 与索引文本全量核验，允许明确记录未解决项 |
| 4 | BLOCKED | 依赖本轮结论和后续阶段批准 |
| 5 | BLOCKED | 独立修复候选预算，不与评估变更混在一起 |
| 6 | BLOCKED | 依赖 5 和可用于基线的 Gold |
| 7 | BLOCKED | 依赖统一基线及新 Dev 数据 |
| 8 | BLOCKED | 依赖固定 corpus/chunk |
| 9 | BLOCKED | 依赖候选池与预算曲线 |
| 10 | BLOCKED | 依赖稳定 retrieval/reranking |
| 11 | BLOCKED | 依赖 Claim/Citation 标注、生成预算及性能预算 |
| Runtime Promotion | BLOCKED | 依赖端到端和 Locked Holdout Gate |

## Phase 1：恢复复现基础

### 修改

- tests/test_inspect_retrieval.py、test_inspect_vector_store.py：使用 debug 中的新路径。
- experiments/debug/inspect_retrieval.py、inspect_vector_store.py、chunk_id_verify.py、
  audit_research_ingestion.py、gen_eval.py：项目根改为 parents[2]。
- inspect_vector_store：同步内部工具导入。
- gen_flowchart：输出使用项目根绝对路径。
- tests/test_debug_paths.py：从其他工作目录验证路径，不运行真实模型。
- evals/evaluation_v2/phase1_runtime_snapshot_v1.json：配置白名单、语料/源码/历史产物指纹。

### 验证与 Gate

完整测试：332 passed，0 failed，0 collection errors；17 条 Chroma legacy embedding
function config 弃用警告。测试未排除任何文件。Gate PASS。
本阶段无检索干预，指标 delta/CI/wins-ties-losses 不适用；未删除或归档任何文件。

## Phase 2：统一合同与指标核心

### 修改

| 文件 | 职责与原因 |
|---|---|
| PROJECT_SPEC.md | 升到 0.15，§20.1 明确覆盖旧评估命名；纠正 Holdout/Cross-Encoder 描述 |
| domain/evaluation_v2.py | 严格数据、定位、证据组、映射与 family split 合同 |
| eval/dataset.py | 显式 research-v1 适配与 v2 loader，不猜测旧数组含义 |
| eval/metrics.py | 共享历史指标与 CUSTOM bundle nDCG，保留旧字段意义 |
| eval/v2.py | 冻结排名评估、切片、排除原因与排他写入 |
| eval/comparison.py | 题族配对 Bootstrap、随机化检验、Holm 校正 |
| eval/quality.py | 对显式标注计算 Context、Claim、Citation、拒答指标 |
| run_research_dense_baseline.py | 保留历史函数签名，指标改为共享调用 |
| run_research_candidate_reranking_experiment.py | 去除指标函数内部副本，保留历史 wrapper |
| tests/test_evaluation_v2.py | 合同、指标、排除逻辑、统计与防覆盖验证 |

### 指标约定

- 标准 FirstRelevantRR 取第一个相关 Chunk，聚合字段为 MRR；碎片证据仍可能有相关
  Chunk，但不能因此算完整组覆盖。
- CompleteEvidenceRR 为 CUSTOM；所有组完整覆盖后才得分。
- CUSTOM EvidenceGroup nDCG 每 rank 最多给一个首次未覆盖组增益，重复组不重复得分；
  组顺序使用冻结数据顺序。原始 nDCG 命名仅存在于 legacy 适配报告。
- 无法映射题单列，不混入 Retriever 平均。必须同时展示全量与 eligible 分母。
- Unanswerable 不进入 Retrieval Recall；服务 ERROR 独立计数。
- ContextKnownEvidencePrecision 仅针对已知标注；Token 数必须来自具名生成计数器或
  估算器。当前 Runtime 的 Tokenizer 尚未更换。
- quality 函数消费显式判断，不自动产生 Judge 真值。本轮没有新增人工 Claim 真值，
  实际回答质量与性能字段保持 null。
- 比较按同题配对，题族整体 resample/交换系统标签；baseline=0 时相对差为 null。
  当前统计结果均为 exploratory，不把不显著解释成非劣。

### 验证与 Gate

完整测试：339 passed，0 failed，0 collection errors，17 warnings。Gate PASS。
历史 40 题逐题指标分别通过 baseline wrapper、reranking wrapper 和共享核心重算，
与原聚合报告精确一致，delta=0。没有重新运行检索，也没有写回历史报告。
结果见 phase2_contract_validation_v1.json。

## Phase 3：Gold 全量验证

### 修改与产物

- eval/mapping.py：先在原始 locator 验证锚点，再匹配同内容版本 Chunk；来源重复、
  断词差异、碎片与不可映射分开记录。只有能唯一对齐到原文的区间才能证明碎片覆盖。
- experiments/audit_gold_v2.py：SQLite mode=ro 读取现有索引，加载原始 PDF，校验
  manifest SHA256，生成新数据包装、映射与历史排名回放；没有初始化正式 Chroma Client。
- tests/test_gold_mapping_v2.py：完整/碎片/部分/歧义/规范化/缺来源及版本不一致测试。
- evals/evaluation_v2/phase3_gold_audit_v1/：4 个独立 JSON 产物。

### 全量结果

| 状态 | 来源组数 |
|---|---:|
| FULLY_MAPPED | 29 |
| ANCHOR_NORMALIZATION_FAILURE | 2 |
| AMBIGUOUS_MAPPING | 13 |
| 合计 | 44 |

10/10 来源提取成功。GoldMappabilityRate=29/44=0.65909。
没有确认跨 Chunk 碎片案例；这不代表所有未解决组都已排除 Chunk 影响。

- ARR-003：源页包含 equipment defects or limited viewpoints；Chunk 9119edb3…
  仅在诊断性的断词合并后匹配。
- ARR-033：源页包含 technical details and additional quantitative；Chunk 5a4b6df5…
  解码正文出现 addi- tional，宽松诊断命中。
- 这两题属于规范化差异，未证明 Gold 原文错误或 Chunk 边界切碎。保持 Canonical
  原文不变，未自动接受可能误合并合法连字符的宽松规则。
- 其余 13 组在指定页码内有重复锚点。v2 保守要求进一步限定锚点/位置；这是新增的
  歧义检查，不是对旧 Gold 全部无效的宣告。逐组原因见 gold_mapping_v2.json。

### 历史排名回放

数据：research-v2-from-be1e8354…；corpus 仍是原 10 篇论文；evaluation_version=2.0。
40 题中 26 题可评分、14 题因映射未解决排除。四道多文档题目前均有未解决组，
因此本次回放不能用于多文档检索结论。

可评分子集的 AnyEvidenceHit@5=0.53846，EvidenceGroupRecall@20=0.73077。
它们不是改进指标：分母和合同已变化，不能与旧 0.475/0.650 直接比较。
baseline/variant A-B、delta、CI、wins/ties/losses 本阶段不适用。

### Gate

完整测试：344 passed，0 failed，0 collection errors，17 warnings。
Phase 3 审计 Gate PASS：所有组有记录，未解决映射与检索失败分开。
新基线数据就绪状态 NEEDS_REVIEW：29/44 组可评分不足以代表完整科研题集。
Phase 4–6 未启动。

## 新发现与下一步

- FACT_CONFIRMED：两个空 Gold 为原文/解码断词规范化差异；13 个来源组存在定位歧义。
- FACT_CONFIRMED：候选截断仍存在，本轮没有改变 Runtime。
- HYPOTHESIS：源位置感知的规范化可恢复两题；需要独立测试，不能全局删除连字符。
- NOT_CURRENT_PRIORITY：Chroma 弃用警告迁移、OCR、Vector DB 更换。
- OBSOLETE：两个 inspect 测试的旧导入与 debug 项目根路径问题。

建议下一次先处理上述 15 个来源组：在新 dataset/mapping 版本中补精确锚点或位置，
验证对齐后的断词规则，恢复多文档题可评分性；再审查进入 Phase 4–6 的条件。
本轮没有删除文件、切换检索策略、更新正式索引或覆盖旧实验结果。

## 复现命令

使用已有 Conda agent 环境：

```powershell
python -B -m pytest -q -p no:cacheprovider
python -B experiments/audit_gold_v2.py --output-dir <全新的产物目录>
```

审计命令不调用 Embedding/LLM；输出目录已存在时明确拒绝，防止覆盖历史结果。
